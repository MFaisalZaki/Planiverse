"""Shared machinery for driving a Game Boy cartridge through PyBoy, for search.

Three environments play real cartridges (Puzznic, Flipull and Super Mario Land) and they
share a shape. States carry an emulator save-state so search can branch, actions are button
combinations held for a number of frames, and applying an action rewinds the machine to the
parent state first. This module owns everything that is a property of *driving PyBoy for
planning* rather than of any particular cartridge:

- the emulator lifecycle and save-state plumbing (`create_pyboy`, `save_state`,
  `load_state`), and the screen grab built on it (`screen`, `screens`);
- OAM sprite decoding (`sprites`): the buffer address stays a per-game fact, because each
  cartridge chooses its own OAM DMA source;
- the ROM revision check (`verify_rom`);
- the `"buttons,ticks"` action vocabulary with its parse/cost/apply skeleton (`GBAction`);
- the settled-state scaffolding (`GBState`);
- the environment tail every game repeats: `successors` with its self-loop filter,
  `simulate`, `step`, `get_actions`, `render`, `close` (`GBEnv`; `validate` comes from
  `Environment`).

What stays in each game module is everything measured off its cartridge: the memory map
and its decoding, the settle predicate (Puzznic watches its grid; Flipull must also watch
sprites, because a thrown block *is* a sprite), the calibration probes, and the boot route.
Those differ for real reasons, and homogenising them here would trade fidelity for
tidiness; the seams below (`__settle__`, `__next_state__`, `__advance__`) are where each
game plugs its own answers in.
"""
import io
import os
import hashlib
import warnings

from pyboy import PyBoy

from planiverse.environments.base import Environment

# ------------------------------------------------------------------------- emulation

#: Where a Game Boy's OAM DMA source conventionally sits. Each game module keeps its own
#: verified constant and passes it in; this is only the default.
OAM_BUFFER_ADDR = 0xC000
OAM_BUFFER_BYTES = 160


def create_pyboy(romfile, render):
    return PyBoy(romfile, sound_emulated=False, window="SDL2" if render else "null")


def save_state(pyboy):
    with io.BytesIO() as handle:
        pyboy.save_state(handle)
        handle.seek(0)
        return handle.getvalue()


def load_state(pyboy, state_bytes, render=False):
    with io.BytesIO(state_bytes) as handle:
        pyboy.load_state(handle)
        pyboy.tick(1, render)


def screen(pyboy, scale=4):
    """The emulator's current frame as a PIL image, magnified `scale` times.

    Nearest-neighbour, not Pillow's default bicubic. A Game Boy frame is 160x144 pixels of
    four-colour pixel art, and interpolating it invents colours the console never drew: the
    block edges Puzznic's grid is read from come out as grey mush. The one thing a
    screenshot of a console is for is showing what the console showed.

    The caller owns the result. `pyboy.screen.image` hands back a view onto the live frame
    buffer, so keeping it across a tick would quietly mutate a frame already collected;
    `resize` copies, which is what makes a list of these safe to hold.
    """
    image = pyboy.screen.image
    if image is None:
        raise RuntimeError("PyBoy could not render the screen — is Pillow installed?")
    from PIL import Image

    return image.convert("RGB").resize((160 * scale, 144 * scale), Image.NEAREST)


def screens(romfile, states, scale=4):
    """The console's own screen for each of `states`, in order.

    One throwaway emulator for the whole list rather than one per state: booting PyBoy is
    the expensive part and a save-state carries everything, so loading each in turn into
    the same machine gives identical frames for a fraction of the work. A forty-step
    history is forty `load_state` calls instead of forty cold boots.

    Deliberately not `self.pyboy`: the environment's own emulator is parked on whatever
    `step` last left it on, and rewinding it here to draw a picture would move the state
    out from under the caller.
    """
    if not states:
        return []
    dummy = create_pyboy(romfile, False)
    try:
        frames = []
        for state in states:
            load_state(dummy, state.gb_state, render=True)
            frames.append(screen(dummy, scale))
        return frames
    finally:
        dummy.stop(save=False)


def sprites(pyboy, oam_addr=OAM_BUFFER_ADDR, visible_only=False):
    """The OAM DMA buffer, as `(y, x, tile)` for every sprite.

    `visible_only` drops entries whose Y is zero, which is where a game parks the sprites
    it is not using.
    """
    buffer = pyboy.memory[oam_addr:oam_addr + OAM_BUFFER_BYTES]
    entries = [(buffer[i], buffer[i + 1], buffer[i + 2])
               for i in range(0, OAM_BUFFER_BYTES, 4)]
    return [entry for entry in entries if entry[0]] if visible_only else entries


def verify_rom(romfile, expected_md5, rom_name, stacklevel=4):
    """Warn when the dump is not the revision an environment's addresses were read from."""
    if not os.path.isfile(romfile):
        return
    with open(romfile, "rb") as handle:
        digest = hashlib.md5(handle.read()).hexdigest()
    if digest != expected_md5:
        warnings.warn(
            f"{romfile} has MD5 {digest}, not {expected_md5} ({rom_name}). The addresses "
            "this environment reads are revision-specific and may not hold.",
            UserWarning, stacklevel=stacklevel)


# ----------------------------------------------------------------------------- state

class GBState:
    """A snapshotted position: the emulator save-state plus facts read out of RAM.

    Subclasses read their game's memory in `__init__` after calling up here, and must
    decide their own equality: which facts make two positions the same position is a
    per-game judgment (Puzznic compares grid and cursor; Flipull adds what is in hand).
    Depth and history are never part of it, so a state reached two ways can compare equal
    and search can close.
    """

    def __init__(self, pyboy, depth):
        self.depth = depth
        self.literals = frozenset()
        self.gb_state = save_state(pyboy)

    def __lt__(self, other):
        return self.depth < other.depth

    def save(self, gamerom, file, scale=4):
        """Write a PNG of this state by booting a throwaway emulator to it. Needs Pillow.

        The `render=True` on the tick is load-bearing and was missing: `load_state` defaults
        to ticking *without* rendering, which is right everywhere else (search never looks
        at the screen and drawing it is wasted work). Here it means the frame buffer is
        never filled and every screenshot came out a blank white rectangle.
        """
        dummy = create_pyboy(gamerom, False)
        try:
            load_state(dummy, self.gb_state, render=True)
            screen(dummy, scale).save(file)
        finally:
            dummy.stop(save=False)


# ---------------------------------------------------------------------------- actions

class GBAction:
    """A button combination held for a number of frames, spelled `"buttons,ticks"`.

    `apply` is the shared shape: rewind the emulator to the parent state, press the
    buttons, wait for the game to stop moving, and snapshot. Subclasses supply the two
    per-game halves: `__settle__`, because what "stopped moving" means is a property of
    the cartridge, and `__next_state__`, because each game's state reads its own memory.
    """

    #: What each button costs; subclasses set their game's map.
    cost_map = {}

    def __init__(self, action):
        self.action = action
        self.actions_tick_list = self.__parse_action__(action)
        self.cost_value = self.__cost__()

    def __parse_action__(self, act):
        buttons, ticks = act.split(",")
        return [(button, int(ticks)) for button in buttons.split("+")]

    def __cost__(self):
        return sum(self.cost_map[button] for button, _ in self.actions_tick_list)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.action == other.action

    def __hash__(self):
        return hash(self.action)

    def __lt__(self, other):
        return self.action < other.action

    def __str__(self):
        return self.action.replace(",", "_for_").replace("+", "_with_")

    def __repr__(self):
        return str(self)

    def cost(self):
        return self.cost_value

    def __press__(self, pyboy, render):
        ticks = set()
        for button, hold in self.actions_tick_list:
            if button != "nop":
                pyboy.button(button, hold)
            ticks.add(hold)
        pyboy.tick(max(ticks) + 1, render)

    def __settle__(self, pyboy, render, **settle_kwargs):
        """Run the emulator until the game stops moving. The predicate is per-game."""
        return True

    def __next_state__(self, pyboy, state):
        """The successor state, read out of the emulator once settled."""
        raise NotImplementedError(f"{type(self).__name__} must implement __next_state__()")

    def apply(self, pyboy, state, render=False, **settle_kwargs):
        """Rewind the emulator to `state`, press the buttons, and snapshot once settled."""
        load_state(pyboy, state.gb_state, render)
        self.__press__(pyboy, render)
        self.__settle__(pyboy, render, **settle_kwargs)
        return self.__next_state__(pyboy, state)


# ------------------------------------------------------------------------ environment

class GBEnv(Environment):
    """The tail every Game Boy environment shares.

    Subclasses set `rom_md5`, `rom_name` and `action_class`, implement `reset` (boot
    routes differ per cartridge) and `__score__`, and are expected to leave behind the
    attributes the tail drives: `pyboy`, `romfile`, `render_window`, `actions`,
    `settle_kwargs`, `state`, `state_history`.
    """

    #: The dump the subclass's addresses were read from, and its conventional name.
    rom_md5 = None
    rom_name = None
    #: The GBAction subclass `__advance__` builds from an action string.
    action_class = None

    def __verify_rom__(self):
        """Warn when the dump is not the revision these addresses were read from."""
        verify_rom(self.romfile, self.rom_md5, self.rom_name)

    def __restart_emulator__(self):
        """Stop any previous emulator and boot a fresh one from power-on."""
        if self.pyboy is not None:
            self.pyboy.stop(save=False)
        self.pyboy = create_pyboy(self.romfile, self.render_window)

    def __score__(self, state):
        """What `step` reports as the score of a state; a per-game judgment."""
        raise NotImplementedError(f"{type(self).__name__} must implement __score__()")

    def __advance__(self, state, action):
        """Apply one action, treating won and lost stages as absorbing.

        Clearing a stage ends it and the cartridge loads the next one over the top, so
        pressing on past a goal state would silently hand back a position from a different
        stage; a lost stage has nothing left to plan for. Absorbing states expand to
        nothing: every action returns the state itself, and `successors`' self-loop filter
        drops it.
        """
        if self.is_goal(state) or self.is_terminal(state):
            return state
        if isinstance(action, str):
            action = self.action_class(action)
        return action.apply(self.pyboy, state, self.render_window, **self.settle_kwargs)

    def successors(self, state):
        """Every action applied to `state`, minus the ones that change nothing."""
        successors = []
        for actionstr in self.actions:
            action = self.action_class(actionstr)
            successor = self.__advance__(state, action)
            if successor == state:
                continue
            successors.append((action, successor))
        return successors

    def simulate(self, plan):
        state, _ = self.reset()
        trace = [state]
        for action in plan:
            trace.append(self.__advance__(trace[-1], action))
        return trace

    def step(self, action):
        """Stateful play, as opposed to expansion. Returns the new state and its score."""
        if self.state is None:
            raise ValueError("Game not initialized. Call reset() first.")
        self.state = self.__advance__(self.state, action)
        self.state_history.append(self.state)
        return self.state, self.__score__(self.state)

    def get_actions(self):
        return list(self.actions)

    def __played__(self):
        """The history of `step` calls with consecutive repeats dropped.

        Keyed on the state, not on `str(state)`. The text board is derived from a subset of
        what a state knows, so two genuinely different positions can typeset identically --
        Mario's does, since the board says nothing about where in a screen he is -- and
        de-duplicating on the text threw away frames the console drew differently. A state
        already compares on exactly the facts that make it that position, which is the
        question being asked here.
        """
        played = []
        for state in self.state_history:
            if played and played[-1] == state:
                continue
            played.append(state)
        return played

    def render(self, target=None, scale=4, **kwargs):
        """The game's own screen for every position `step` has played through.

        This used to return `str(state)` per step and stop there, which made the one family
        of environments that *has* real pixels the one family that threw them away: a
        cartridge draws the position, and the text board is a reading of RAM taken next to
        it, not a picture of it. So the frames are the return value now -- one PIL image per
        de-duplicated step, at console resolution magnified `scale` times.

        The text board is still printed alongside, one block per frame. A terminal cannot
        show a picture and a caption that survives a copy-paste is worth keeping; it is a
        caption now rather than the whole product.

        `target` writes the frames instead of returning them, in any format
        `planiverse.rendering.render_trace` spells -- `"play.gif"`, `"play.png"` for a
        contact sheet, `"play.pdf"`, or a directory for one PNG per step -- and returns
        whatever `render_trace` returns. Remaining keyword arguments go to it, so
        `env.render("play.png", actions=plan, env=env)` captions the sheet. `scale` is for
        the returned frames only: written ones go through each state's own `save`, which is
        where a game that wants a bigger default sets one (Super Mario Land does).
        """
        played = self.__played__()
        for step, state in enumerate(played):
            print(f"Step: {step}")
            print(str(state))
            print("--------------")
        if target is not None:
            return self.render_trace(played, target, **kwargs)
        return screens(self.romfile, played, scale=scale)

    def close(self):
        if self.pyboy is not None:
            self.pyboy.stop(save=False)
            self.pyboy = None
