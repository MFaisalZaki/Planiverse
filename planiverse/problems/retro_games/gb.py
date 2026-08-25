"""Shared machinery for driving a Game Boy cartridge through PyBoy, for search.

Three environments play real cartridges — Puzznic, Flipull and Super Mario Land — and they
share a shape: states carry an emulator save-state so search can branch, actions are button
combinations held for a number of frames, and applying an action rewinds the machine to the
parent state first. This module owns everything that is a property of *driving PyBoy for
planning* rather than of any particular cartridge:

- the emulator lifecycle and save-state plumbing (`create_pyboy`, `save_state`,
  `load_state`);
- OAM sprite decoding (`sprites`) — the buffer address stays a per-game fact, because each
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
tidiness — the seams below (`__settle__`, `__next_state__`, `__advance__`) are where each
game plugs its own answers in.
"""
import io
import os
import hashlib
import warnings

from pyboy import PyBoy

from planiverse.problems.retro_games.base import RetroGame

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
        """Write a PNG of this state by booting a throwaway emulator to it. Needs Pillow."""
        dummy = create_pyboy(gamerom, False)
        try:
            load_state(dummy, self.gb_state)
            image = dummy.screen.image
            if image is None:
                raise RuntimeError("PyBoy could not render the screen — is Pillow installed?")
            image.resize((160 * scale, 144 * scale)).save(file)
        finally:
            dummy.stop(save=False)


# ---------------------------------------------------------------------------- actions

class GBAction:
    """A button combination held for a number of frames, spelled `"buttons,ticks"`.

    `apply` is the shared shape: rewind the emulator to the parent state, press the
    buttons, wait for the game to stop moving, and snapshot. Subclasses supply the two
    per-game halves — `__settle__`, because what "stopped moving" means is a property of
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

class GBEnv(RetroGame):
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

    def render(self):
        """Print the de-duplicated history of `step` calls, and return it as strings."""
        rendered = []
        for state in self.state_history:
            if rendered and rendered[-1] == str(state):
                continue
            rendered.append(str(state))
        for step, text in enumerate(rendered):
            print(f"Step: {step}")
            print(text)
            print("--------------")
        return rendered

    def close(self):
        if self.pyboy is not None:
            self.pyboy.stop(save=False)
            self.pyboy = None
