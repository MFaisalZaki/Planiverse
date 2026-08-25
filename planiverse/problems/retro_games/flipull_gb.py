"""Flipull on a Game Boy, driven through PyBoy.

Flipull — Taito's *Plotting* — sits the player at the right of a wall of blocks holding one
of them. Throwing it sends it left along a row: it destroys blocks of its own type as it
goes, and swaps with the first block of a different type, which becomes the new held block.
Destroying a block drops its column. The stage is finished when few enough blocks are left.

That makes the action set unusually small for a Game Boy game — pick a row, throw — while
the *consequences* of a throw run several moves deep, which is the shape a planner wants.

The addresses come from a reverse-engineering pass over `Flipull (USA).gb`
(MD5 `4fcc13db8144687e6b28200387aed25c`) and are documented in
`docs/environments/flipull-gb-memory-map.md`. That map was derived behaviourally — recording WRAM and
HRAM every frame against known on-screen values — so its confidence varies field by field,
and the constants below carry that grading rather than pretending to it.

    env = FlipullGBEnv("Flipull (USA).gb")
    state, info = env.reset()
    print(state)
    for action, successor in env.successors(state):
        ...
"""
import io
import os
import hashlib
import warnings
from collections import Counter, namedtuple

from pyboy import PyBoy

from planiverse.problems.retro_games.base import RetroGame

#: The dump these addresses were read from. No mapper: the whole ROM is flat at $0000-$7FFF.
ROM_MD5 = "4fcc13db8144687e6b28200387aed25c"

# ------------------------------------------------------------------- the block field
# Verified: base, stride and the wall pattern at both ends. The address calculator was never
# found in code — unlike Puzznic's at `0:29CE` — so the geometry is read off the dump's
# structure: 14 evenly spaced rows with consistent borders.
FIELD_ADDR = 0xC840
ROW_STRIDE = 0x20                # 32 bytes per row...
FIELD_ROWS = 14
FIELD_COLS = 16                  # ...of which only the first 16 carry meaning
FIELD_BYTES = FIELD_ROWS * ROW_STRIDE

CELL_OUTSIDE = 0x00              # outside the field
CELL_BORDER = 0x80               # ceiling, floor, left wall
BLOCK_MIN = 0x83                 # $83-$86 are the four playable block types
BLOCK_MAX = 0x86
CELL_STAIRCASE = 0x87            # fixed structural diagonal; never clearable

CEILING_ROW = 0
FLOOR_ROW = FIELD_ROWS - 1
LEFT_WALL_COL = 0

CELL_GLYPHS = {CELL_OUTSIDE: " ", CELL_BORDER: "#", CELL_STAIRCASE: "="}

# --------------------------------------------------------------------- HRAM counters
# Flipull keeps its counters as separate decimal digits, ones first — not binary and not
# packed BCD. Searching for 25 or $19 finds nothing; the value is `05` and `02` in adjacent
# bytes. Nearly every counter is in HRAM rather than WRAM.
BLOCKS_ONES_ADDR = 0xFFC9        # verified: 05 -> 04 as the HUD went 25 -> 24
BLOCKS_TENS_ADDR = 0xFFCA        # good
INITIAL_ONES_ADDR = 0xFFC0       # moderate: the stage's *starting* total, not the live one
INITIAL_TENS_ADDR = 0xFFC1       # moderate
TIMER_SECONDS_ONES_ADDR = 0xFFCB  # verified: 09 -> 02 as the HUD went 2:59 -> 2:52
TIMER_SECONDS_TENS_ADDR = 0xFFCC  # good
TIMER_MINUTES_ADDR = 0xFFCE      # good
SUBSECOND_ADDR = 0xFFCD          # moderate: free-running tick
CLEAR_TARGET_ADDR = 0xFFCF       # moderate: the CLEAR number; never seen change
STAGE_ADDR = 0xFFC6              # unverified: read 01 in stage 1, never seen change

# ----------------------------------------------------------------------- the throw
HELD_BLOCK_ADDR = 0xFFD4         # moderate: held / in-flight block type
THROW_FLAG_ADDRS = (0xFFD2, 0xFFD3)   # moderate: both 00 -> 01 on release
INFLIGHT_X_ADDR = 0xFFDF         # moderate: falls as the block travels left, then resets
INFLIGHT_Y_ADDR = 0xFFDE         # low
PLAYER_STATE_ADDR = 0xC002       # tracked vertical input (89/8F); not a row index

#: OAM DMA source. The player is a sprite, and which one is discovered at runtime rather
#: than assumed — see `probe_player_sprite`.
OAM_BUFFER_ADDR = 0xC000
OAM_BUFFER_BYTES = 160

# --------------------------------------------------------------------------- driving
PRESS_TICKS = 8                  # fallback only; `calibrate` measures the real window
PROBE_MAX_HOLD = 60
SETTLE_MAX_TICKS = 900           # a throw crosses the field, then columns fall
SETTLE_STABLE_TICKS = 6
BOOT_MAX_TICKS = 2400
BOOT_PRESS_EVERY = 12
INTRO_MAX_TICKS = 900
INTRO_STEP_TICKS = 30

THROW_BUTTONS = ("a", "b")       # which one throws is probed, not assumed
MOVE_BUTTONS = ("up", "down")

action_cost_map = {"up": 1, "down": 1, "a": 1, "b": 1, "nop": 0}

Block = namedtuple("Block", ["row", "col", "type"])
Calibration = namedtuple("Calibration", ["press_ticks", "hold_window", "throw_button",
                                         "throw_ticks", "player_sprite", "row_pitch"],
                         defaults=(None, None, None, None, None, None))


def button_actions(calibration=None):
    """The buttons this game has: pick a row, throw."""
    ticks = (calibration.press_ticks if calibration and calibration.press_ticks
             else PRESS_TICKS)
    throw = (calibration.throw_button if calibration and calibration.throw_button
             else THROW_BUTTONS[0])
    throw_ticks = (calibration.throw_ticks if calibration and calibration.throw_ticks
                   else ticks)
    return [f"{button},{ticks}" for button in MOVE_BUTTONS] + [f"{throw},{throw_ticks}"]


action_list = button_actions()


# --------------------------------------------------------------------- pure decoding
# Split out from the emulator so they can be tested against synthetic RAM.

def cell_address(row, col):
    """The address of a field cell. Row stride is $20 even though only 16 columns count."""
    return FIELD_ADDR + ROW_STRIDE * row + col


def decode_field(raw):
    """The bytes at `$C840` as a 14x16 field, dropping the unused half of each row."""
    return tuple(tuple(raw[ROW_STRIDE * row + col] for col in range(FIELD_COLS))
                 for row in range(FIELD_ROWS))


def is_playable(value):
    """`$83`-`$86` are the four block types. `$87` is the fixed staircase, not a block."""
    return BLOCK_MIN <= value <= BLOCK_MAX


def decode_blocks(field):
    """Every playable block on the field, top-left first."""
    return tuple(Block(row, col, field[row][col] - BLOCK_MIN + 1)
                 for row in range(FIELD_ROWS) for col in range(FIELD_COLS)
                 if is_playable(field[row][col]))


def decode_staircase(field):
    """The fixed `$87` cells, which are structural and can never be cleared."""
    return tuple((row, col) for row in range(FIELD_ROWS) for col in range(FIELD_COLS)
                 if field[row][col] == CELL_STAIRCASE)


def decode_digits(tens, ones):
    """Flipull stores counters as separate decimal digits, ones first."""
    return tens * 10 + ones


def decode_timer(minutes, seconds_tens, seconds_ones):
    return minutes * 60 + seconds_tens * 10 + seconds_ones


def block_counts(blocks):
    return Counter(block.type for block in blocks)


def column_blocks(field, col):
    """A column top-down, which is the direction it collapses in."""
    return tuple(field[row][col] for row in range(FIELD_ROWS))


def bounding_box(field):
    """The rows and columns that are part of the field at all.

    Only stage 1 was ever observed and it uses columns 1-5, so the full width is inferred
    from the ceiling and floor reading `$80` across 16 columns. Derive it rather than assume.
    """
    used = [(row, col) for row in range(FIELD_ROWS) for col in range(FIELD_COLS)
            if field[row][col] != CELL_OUTSIDE]
    if not used:
        return (0, -1), (0, -1)
    rows = [row for row, _ in used]
    cols = [col for _, col in used]
    return (min(rows), max(rows)), (min(cols), max(cols))


def render_field(field, held=None, player=None, trim=True):
    """The field as ASCII: `#` border, `=` staircase, `.` empty, digits for block types.

    `held` and `player` are appended as trailing lines rather than drawn into the grid: the
    held block is not on the field, and the player's position is only known as a sprite Y
    (the memory map has no row variable for him), which no row index here can honestly
    stand in for.
    """
    (top, bottom), (left, right) = (bounding_box(field) if trim
                                    else ((0, FIELD_ROWS - 1), (0, FIELD_COLS - 1)))
    lines = []
    for row in range(top, bottom + 1):
        line = []
        for col in range(left, right + 1):
            value = field[row][col]
            if is_playable(value):
                line.append(str(value - BLOCK_MIN + 1))
            else:
                line.append(CELL_GLYPHS.get(value, "?"))
        lines.append("".join(line))
    text = "\n".join(lines)
    if held is not None:
        text += f"\nheld: {held}"
    if player is not None:
        text += f"\nplayer: {player}"
    return text


# ------------------------------------------------------------------------- emulation

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


def read_field(pyboy):
    return pyboy.memory[FIELD_ADDR:FIELD_ADDR + FIELD_BYTES]


def read_blocks_remaining(pyboy):
    return decode_digits(pyboy.memory[BLOCKS_TENS_ADDR], pyboy.memory[BLOCKS_ONES_ADDR])


def read_timer(pyboy):
    return decode_timer(pyboy.memory[TIMER_MINUTES_ADDR],
                        pyboy.memory[TIMER_SECONDS_TENS_ADDR],
                        pyboy.memory[TIMER_SECONDS_ONES_ADDR])


def throw_in_flight(pyboy):
    """Whether a thrown block is still crossing the field."""
    return any(pyboy.memory[addr] for addr in THROW_FLAG_ADDRS)


def sprites(pyboy):
    """The OAM DMA buffer, as `(y, x, tile)` for every visible sprite."""
    buffer = pyboy.memory[OAM_BUFFER_ADDR:OAM_BUFFER_ADDR + OAM_BUFFER_BYTES]
    return [(buffer[i], buffer[i + 1], buffer[i + 2])
            for i in range(0, OAM_BUFFER_BYTES, 4)]


def stage_is_loaded(pyboy):
    """True once a field with blocks on it is up and nothing is mid-throw."""
    remaining = read_blocks_remaining(pyboy)
    if remaining == 0 or throw_in_flight(pyboy):
        return False
    blocks = decode_blocks(decode_field(read_field(pyboy)))
    # The field and the HUD counter have to agree, which is the check the memory map
    # itself leans on: 25 cells in $83-$86 against BLOCK 25.
    return len(blocks) == remaining


def settle(pyboy, render=False, max_ticks=SETTLE_MAX_TICKS, stable_ticks=SETTLE_STABLE_TICKS):
    """Run the emulator until the field stops moving.

    A throw is not instantaneous: the block crosses the field destroying its own type as it
    goes, then every column it emptied falls. Snapshotting straight after the button press
    would catch the middle of that. Settled means no throw in flight and the field
    byte-identical for `stable_ticks` frames.
    """
    previous, stable = None, 0
    for _ in range(max_ticks):
        pyboy.tick(1, render)
        if throw_in_flight(pyboy):
            previous, stable = None, 0
            continue
        raw = read_field(pyboy)
        if raw == previous:
            stable += 1
            if stable >= stable_ticks:
                return True
        else:
            previous, stable = raw, 0
    return False


def _tap(pyboy, button, hold, render=False, gap=2):
    pyboy.button(button, hold)
    pyboy.tick(hold + gap, render)


def probe_player_sprite(pyboy, render=False, press_ticks=PRESS_TICKS, **settle_kwargs):
    """Which OAM entry is the player, found by moving and seeing what moved.

    The memory map has no row index for the player — `$C002` tracks vertical *input*, not
    position — so the only thing that says where he is, is his sprite. Which sprite that is
    gets discovered rather than assumed, so it survives a different stage or revision.
    """
    snapshot = save_state(pyboy)
    before = sprites(pyboy)
    deltas = {}
    for button in MOVE_BUTTONS:
        load_state(pyboy, snapshot, render)
        _tap(pyboy, button, press_ticks, render)
        settle(pyboy, render, **settle_kwargs)
        deltas[button] = [after[0] - was[0] for was, after in zip(before, sprites(pyboy))]
    load_state(pyboy, snapshot, render)

    # The player is whatever went up for "up" and down for "down". A counter that merely
    # changes is not enough — the first attempt at this named a scratch variable that had
    # been parked inside the OAM buffer.
    candidates = [index for index in range(len(before))
                  if before[index][0]
                  and deltas["up"][index] < 0 < deltas["down"][index]]
    return candidates[0] if len(candidates) == 1 else None


def player_y(pyboy, sprite):
    """The player sprite's Y, or None when we never worked out which sprite he is."""
    return None if sprite is None else sprites(pyboy)[sprite][0]


def measure_hold_window(pyboy, state_bytes, sprite, render=False, max_hold=PROBE_MAX_HOLD,
                        **settle_kwargs):
    """The closed range of holds that move the player exactly one row.

    Same bound as every other Game Boy menu: too short and the press is never sampled, too
    long and auto-repeat moves two rows, so the state handed back is not the one the action
    described. Measured off the cartridge rather than guessed.
    """
    if sprite is None:
        return None, None
    origin = None
    step = None
    low = None
    for hold in range(1, max_hold + 1):
        load_state(pyboy, state_bytes, render)
        origin = player_y(pyboy, sprite)
        _tap(pyboy, "down", hold, render)
        settle(pyboy, render, **settle_kwargs)
        moved = abs(player_y(pyboy, sprite) - origin)
        if moved == 0:
            continue
        if step is None:
            step, low = moved, hold
        elif moved >= 2 * step:
            return (low, hold - 1), step
    load_state(pyboy, state_bytes, render)
    return (None if low is None else (low, max_hold)), step


def probe_throw_button(pyboy, state_bytes, render=False, press_ticks=PRESS_TICKS,
                       **settle_kwargs):
    """Which button throws, found by pressing each and watching for a throw.

    A throw shows up in two independent places — the flags at `$FFD2`/`$FFD3` go up, and the
    field changes — so a button that only moves the player is not mistaken for one.
    """
    for button in THROW_BUTTONS:
        load_state(pyboy, state_bytes, render)
        before = read_field(pyboy)
        pyboy.button(button, press_ticks)
        launched = False
        for _ in range(press_ticks + 30):
            pyboy.tick(1, render)
            launched = launched or throw_in_flight(pyboy)
        settle(pyboy, render, **settle_kwargs)
        if launched or read_field(pyboy) != before:
            load_state(pyboy, state_bytes, render)
            return button
    load_state(pyboy, state_bytes, render)
    return None


def calibrate(pyboy, state_bytes, render=False, max_hold=PROBE_MAX_HOLD, **settle_kwargs):
    """Measure how this cartridge wants to be driven. A property of the game, so once is
    enough — every probe rewinds to `state_bytes`, so nothing it does survives."""
    load_state(pyboy, state_bytes, render)
    sprite = probe_player_sprite(pyboy, render, **settle_kwargs)
    window, row_pitch = measure_hold_window(pyboy, state_bytes, sprite, render, max_hold,
                                            **settle_kwargs)
    press_ticks = (window[0] + window[1]) // 2 if window else PRESS_TICKS
    throw = probe_throw_button(pyboy, state_bytes, render, press_ticks, **settle_kwargs)
    load_state(pyboy, state_bytes, render)
    return Calibration(press_ticks=press_ticks, hold_window=window,
                       throw_button=throw or THROW_BUTTONS[0], throw_ticks=press_ticks,
                       player_sprite=sprite, row_pitch=row_pitch)


def wait_until_interactive(pyboy, render=False, max_ticks=INTRO_MAX_TICKS,
                           step=INTRO_STEP_TICKS, press_ticks=PRESS_TICKS):
    """Advance until the stage accepts a button, and report how many frames it took.

    A stage can be entirely readable while its intro is still running — Puzznic ignores input
    for 210 frames after its field is in memory — and a state snapshotted in that window
    looks normal and answers nothing. Probes from a snapshot at increasing offsets, then
    rewinds and replays only the waiting, so the field is untouched.
    """
    snapshot = save_state(pyboy)
    for waited in range(0, max_ticks, step):
        for button in MOVE_BUTTONS:
            load_state(pyboy, snapshot, render)
            if waited:
                pyboy.tick(waited, render)
            before = sprites(pyboy)
            _tap(pyboy, button, press_ticks, render)
            pyboy.tick(12, render)
            if sprites(pyboy) != before:
                load_state(pyboy, snapshot, render)
                if waited:
                    pyboy.tick(waited, render)
                return waited
    load_state(pyboy, snapshot, render)
    return None


def boot(pyboy, render=False, max_ticks=BOOT_MAX_TICKS, press_every=BOOT_PRESS_EVERY):
    """Tap through the title screens until a stage is on the field."""
    for frame in range(0, max_ticks, press_every):
        pyboy.button("start" if (frame // press_every) % 2 == 0 else "a", 4)
        pyboy.tick(press_every, render)
        if stage_is_loaded(pyboy):
            return True
    return False


# ----------------------------------------------------------------------------- state

class FlipullGBState:
    """A settled position: the emulator save-state plus the facts read out of RAM."""

    def __init__(self, pyboy, depth, calibration=None, stage_types=None):
        self.depth = depth
        self.literals = frozenset()
        self.gb_state = save_state(pyboy)
        self.calibration = calibration
        self.__update__(pyboy, stage_types)

    def __update__(self, pyboy, stage_types):
        raw = read_field(pyboy)
        self.field = decode_field(raw)
        self.blocks = decode_blocks(self.field)
        self.staircase = decode_staircase(self.field)

        self.blocks_remaining = read_blocks_remaining(pyboy)
        self.blocks_initial = decode_digits(pyboy.memory[INITIAL_TENS_ADDR],
                                            pyboy.memory[INITIAL_ONES_ADDR])
        self.clear_target = pyboy.memory[CLEAR_TARGET_ADDR]
        self.timer_seconds = read_timer(pyboy)
        self.stage = pyboy.memory[STAGE_ADDR]

        held = pyboy.memory[HELD_BLOCK_ADDR]
        self.held_block = held - BLOCK_MIN + 1 if is_playable(held) else None
        self.player_y = player_y(pyboy, self.calibration.player_sprite) if self.calibration else None

        # The block types a stage started with, carried down the search tree rather than
        # re-derived — a type cleared out would otherwise look like one that never existed.
        self.stage_types = (frozenset(block.type for block in self.blocks)
                            if stage_types is None else stage_types)

        # Sampled here, while the emulator still holds this state: read later from live
        # memory they would describe whichever state was applied last.
        self.stage_cleared = (self.blocks_initial > 0
                              and self.blocks_remaining <= self.clear_target)
        self.out_of_time = self.timer_seconds == 0

        predicates = [f"remaining({self.blocks_remaining})",
                      f"clear-target({self.clear_target})"]
        if self.player_y is not None:
            # The sprite's Y, not a row index: the memory map has no row variable for the
            # player, so where he is, is only readable from where he is drawn.
            predicates.append(f"at(player, {self.player_y})")
        predicates += [f"at(block-{block.type}, {block.row}, {block.col})"
                       for block in self.blocks]
        if self.held_block is not None:
            predicates.append(f"holding(block-{self.held_block})")
        matched = self.stage_types - {block.type for block in self.blocks}
        predicates += [f"all-blocks-cleared(block-{kind})" for kind in sorted(matched)]
        if self.stage_cleared:
            predicates.append("goal-reached")
        if self.out_of_time:
            predicates.append("terminal-state")
        self.literals = frozenset(predicates)

    def is_consistent(self):
        """Does the field agree with the HUD counter?

        The memory map's own cross-check: 25 cells in `$83`-`$86` against `BLOCK 25`, and 24
        against 24 after a throw. A mismatch means one of the two has drifted.
        """
        return len(self.blocks) == self.blocks_remaining

    def block_counts(self):
        return block_counts(self.blocks)

    def bounding_box(self):
        return bounding_box(self.field)

    def __eq__(self, other):
        # The position is the field, what is in hand, and which row the player is on — a
        # throw from a different row does something different, so it is not the same state.
        # Depth and history are not part of it, so a state reached two ways compares equal.
        return (isinstance(other, FlipullGBState) and self.field == other.field
                and self.held_block == other.held_block and self.player_y == other.player_y)

    def __hash__(self):
        return hash((self.field, self.held_block, self.player_y))

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        return render_field(self.field, self.held_block, self.player_y)

    def __repr__(self):
        return (f"<FlipullGBState(depth={self.depth}, remaining={self.blocks_remaining}"
                f"/target {self.clear_target}, held={self.held_block})>")

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

class FlipullGBAction:
    """A button held for a number of frames, spelled `"button,ticks"`.

    The same spelling the Puzznic and Super Mario Land environments use. Flipull's whole
    input vocabulary is up, down and throw, so this is the console's interface with nothing
    layered over it.
    """

    def __init__(self, action):
        self.action = action
        self.actions_tick_list = self.__parse_action__(action)
        self.cost_value = sum(action_cost_map[button] for button, _ in self.actions_tick_list)

    def __parse_action__(self, act):
        buttons, ticks = act.split(",")
        return [(button, int(ticks)) for button in buttons.split("+")]

    def __eq__(self, other):
        return isinstance(other, FlipullGBAction) and self.action == other.action

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

    def apply(self, pyboy, state, render=False, **settle_kwargs):
        """Rewind the emulator to `state`, press the buttons, and snapshot once settled."""
        load_state(pyboy, state.gb_state, render)
        ticks = set()
        for button, hold in self.actions_tick_list:
            if button != "nop":
                pyboy.button(button, hold)
            ticks.add(hold)
        pyboy.tick(max(ticks) + 1, render)
        settle(pyboy, render, **settle_kwargs)
        return FlipullGBState(pyboy, state.depth + 1, state.calibration, state.stage_types)


# ------------------------------------------------------------------------ environment

class FlipullGBEnv(RetroGame):
    """Flipull, played on the cartridge.

    The ROM is copyrighted and is not distributed with this repo; pass the path to your own
    dump.
    """

    def __init__(self, romfile, render=False, verify_rom=True, calibrate=True,
                 settle_max_ticks=SETTLE_MAX_TICKS, settle_stable_ticks=SETTLE_STABLE_TICKS,
                 boot_max_ticks=BOOT_MAX_TICKS):
        self.romfile = romfile
        # Not `self.render`: that name belongs to the method which prints the history.
        self.render_window = render
        self.pyboy = None
        self.stage_index = None
        self.state = None
        self.state_history = []
        self.calibration = None
        self.should_calibrate = calibrate
        self.intro_ticks = None
        self.actions = action_list
        self.settle_kwargs = {"max_ticks": settle_max_ticks, "stable_ticks": settle_stable_ticks}
        self.boot_max_ticks = boot_max_ticks
        if verify_rom:
            self.__verify_rom__()

    def __verify_rom__(self):
        """Warn when the dump is not the revision these addresses were read from."""
        if not os.path.isfile(self.romfile):
            return
        with open(self.romfile, "rb") as handle:
            digest = hashlib.md5(handle.read()).hexdigest()
        if digest != ROM_MD5:
            warnings.warn(
                f"{self.romfile} has MD5 {digest}, not {ROM_MD5} (Flipull (USA)). The "
                "addresses this environment reads are revision-specific and may not hold.",
                UserWarning, stacklevel=3)

    def fix_index(self, index):
        """Select the stage — except that no way to do so has been established.

        The memory map offers `$FFC6` as a stage number but grades it unverified, having
        never seen it change, and no password or level-select route has been looked for. So
        index 0 is the stage the cartridge boots into, and anything else fails loudly rather
        than quietly starting stage 1 and calling it stage 9.
        """
        assert index == 0, (
            "Invalid index: no verified way to select a stage exists yet. $FFC6 looks like "
            "the stage number but was never watched changing, and no password route has "
            "been found — see docs/environments/flipull-gb.md.")
        self.stage_index = index

    def reset(self):
        if self.pyboy is not None:
            self.pyboy.stop(save=False)
        self.pyboy = create_pyboy(self.romfile, self.render_window)
        if not boot(self.pyboy, self.render_window, self.boot_max_ticks):
            raise RuntimeError(
                f"no stage was loaded within {self.boot_max_ticks} frames of booting "
                f"{self.romfile}. Check the ROM is Flipull (USA) (MD5 {ROM_MD5}).")
        self.intro_ticks = wait_until_interactive(self.pyboy, self.render_window)
        settle(self.pyboy, self.render_window, **self.settle_kwargs)

        if self.should_calibrate and self.calibration is None:
            # A property of the cartridge, not of the stage, so once is enough.
            self.calibration = calibrate(self.pyboy, save_state(self.pyboy),
                                         self.render_window, **self.settle_kwargs)
            self.actions = button_actions(self.calibration)

        self.state = FlipullGBState(self.pyboy, 0, self.calibration)
        self.state_history = [self.state]
        return self.state, {"stage": self.state.stage,
                            "blocks": self.state.blocks_remaining,
                            "clear_target": self.state.clear_target,
                            "intro_ticks": self.intro_ticks,
                            "calibration": self.calibration}

    def is_goal(self, state):
        """Few enough blocks left.

        Flipull finishes a stage when the count is down to the `CLEAR` number rather than to
        zero — the HUD shows `BLOCK 25` against `CLEAR 09`. `$FFCF` is that number, and the
        map grades it moderate: it read 09 and never changed, which is consistent with a
        target but was never watched being met.
        """
        return state.stage_cleared

    def is_terminal(self, state):
        """The clock ran out.

        Unlike Puzznic there is no positional dead end to detect: a stage is lost on time,
        and whether a given field can still reach its target depends on the throw mechanics,
        which the map does not describe.
        """
        return state.out_of_time

    def __advance__(self, state, action):
        """Apply one action, treating won and lost stages as absorbing."""
        if self.is_goal(state) or self.is_terminal(state):
            return state
        if isinstance(action, str):
            action = FlipullGBAction(action)
        return action.apply(self.pyboy, state, self.render_window, **self.settle_kwargs)

    def successors(self, state):
        """Every action applied to `state`, minus the ones that change nothing."""
        successors = []
        for actionstr in self.actions:
            action = FlipullGBAction(actionstr)
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
        return self.state, self.state.blocks_initial - self.state.blocks_remaining

    def validate(self, plan):
        return self.is_goal(self.simulate(plan)[-1])

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


# --------------------------------------------------------------------------- reporting

def _report(romfile, render=False):
    """Print what this cartridge wants: the measurements, and the stage it booted into."""
    env = FlipullGBEnv(romfile, render=render)
    try:
        state, info = env.reset()
        calibration = info["calibration"]
        print(f"stage {info['stage']}: {state.blocks_remaining} blocks, "
              f"clear target {state.clear_target}, {state.timer_seconds}s on the clock\n")
        print(state, "\n")
        if calibration.player_sprite is None:
            print("  player       not found — no OAM entry moved up for up and down for down")
        else:
            print(f"  player       OAM slot {calibration.player_sprite}, "
                  f"{calibration.row_pitch} pixels per row")
        if calibration.hold_window is None:
            print("  move hold    not measurable — the player never moved")
        else:
            print(f"  move hold    {calibration.hold_window}  -> press_ticks "
                  f"{calibration.press_ticks}")
        print(f"  throw        {calibration.throw_button.upper()} held "
              f"{calibration.throw_ticks} frames")
        print(f"  round intro  {info['intro_ticks']} frames of ignored input")
        print(f"\n  button actions: {button_actions(calibration)}")
    finally:
        env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure how a Flipull cartridge wants to be driven: which sprite the "
                    "player is, how long a button must be held to move exactly one row, "
                    "and which button throws.")
    parser.add_argument("rom", help="path to Flipull (USA).gb")
    parser.add_argument("--render", action="store_true", help="open an SDL2 window")
    args = parser.parse_args()
    _report(args.rom, args.render)
