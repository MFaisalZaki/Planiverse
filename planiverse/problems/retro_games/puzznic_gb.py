"""Puzznic on a Game Boy, driven through PyBoy.

The sibling module `puzznic.py` re-implements Puzznic in Python. This one plays the real
Japanese Game Boy cartridge inside an emulator and reads the game's own state out of work
RAM, so the transition function is the cartridge's code rather than a reconstruction of
its rules. States carry an emulator save-state, which is what lets search branch: applying
an action rewinds the machine to the parent first.

The addresses below come from a reverse-engineering pass over `Puzznic (J).gb`
(MD5 `9a777d82cd7a8913ba1aed2cc854fa50`) and are documented in
`docs/environments/puzznic-gb.md`. They are revision-specific: another dump will read
garbage, which is why `PuzznicGBEnv` checks the ROM's MD5 and warns when it differs.

    env = PuzznicGBEnv("Puzznic (J).gb")
    env.fix_index(0)
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

# --------------------------------------------------------------------------- the ROM

ROM_MD5 = "9a777d82cd7a8913ba1aed2cc854fa50"

# --------------------------------------------------------------------- memory map §1-5

STAGE_INDEX_ADDR = 0xD003        # which stage the loader will build
CURSOR_COL_ADDR = 0xD012
CURSOR_ROW_ADDR = 0xD013
TOTAL_BLOCKS_ADDR = 0xD018       # blocks this stage loaded with; never decremented
BLOCKS_REMAINING_ADDR = 0xD019   # decremented once per block removed

RECORDS_ADDR = 0xDD00            # 6-byte block records, one per slot
RECORD_BYTES = 6

GRID_ADDR = 0xDF00               # 12 rows x 10 columns, 2 bytes per cell
GRID_ROWS = 12
GRID_COLS = 10
ROW_STRIDE = 20
CELL_STRIDE = 2
GRID_BYTES = GRID_ROWS * ROW_STRIDE

STAGE_LOADER_ENTRY = 0x0430      # bank 0; reads STAGE_INDEX_ADDR a few instructions in

# Cell type codes. Only $00 permits movement; $02, $03 and $06 all obstruct.
CELL_EMPTY = 0x00
CELL_CLEARING = 0x01             # transient: a block is being removed or is in motion
CELL_LEDGE = 0x02
CELL_OUTSIDE = 0x03
CELL_WALL = 0x06
BLOCK_MIN = 0x08                 # $08-$0F are blocks; the type is the value minus 7
BLOCK_TYPE_OFFSET = 7

CELL_GLYPHS = {
    CELL_EMPTY: ".", CELL_CLEARING: "*", CELL_LEDGE: "=", CELL_OUTSIDE: " ", CELL_WALL: "#",
}

# --------------------------------------------------------------------------- driving

PRESS_TICKS = 6                  # frames a button is held; long enough to register, short
                                 # enough not to trip the cursor's auto-repeat
SETTLE_MAX_TICKS = 600           # give up waiting for the board after ten seconds
SETTLE_STABLE_TICKS = 4          # frames the grid must hold still to count as settled
BOOT_MAX_TICKS = 1800            # thirty seconds of title screens is more than enough
BOOT_PRESS_EVERY = 12

action_cost_map = {"a": 0, "left": 1, "right": 1, "up": 1, "down": 1, "nop": 0}

# The cursor moves in four directions; A turns left/right into a push. There is no
# `a+up`/`a+down` because Puzznic only slides blocks sideways — you cannot lift one.
action_list = [f"{buttons},{PRESS_TICKS}"
               for buttons in ("left", "right", "up", "down", "a+left", "a+right")]

position = namedtuple("Position", ["row", "col"])
Block = namedtuple("Block", ["row", "col", "type", "slot"])
Record = namedtuple("Record", ["slot", "type", "state", "row", "col"])


# --------------------------------------------------------------------- pure decoding
# Split out from the emulator so they can be tested against synthetic RAM.

def cell_address(row, col):
    """The address of a cell, per the row-offset table at ROM `$29E8`."""
    return GRID_ADDR + ROW_STRIDE * row + CELL_STRIDE * col


def decode_grid(raw):
    """The 240 bytes at `$DF00` as a 12x10 grid of cell type codes."""
    return tuple(tuple(raw[ROW_STRIDE * row + CELL_STRIDE * col] for col in range(GRID_COLS))
                 for row in range(GRID_ROWS))


def decode_blocks(raw):
    """Every block on the grid, with the record slot its second byte points at."""
    return tuple(
        Block(row, col, raw[offset] - BLOCK_TYPE_OFFSET, raw[offset + 1])
        for row in range(GRID_ROWS) for col in range(GRID_COLS)
        for offset in (ROW_STRIDE * row + CELL_STRIDE * col,)
        if raw[offset] >= BLOCK_MIN
    )


def decode_records(raw, total):
    """The live entries of the record array at `$DD00`.

    Cleared records are zeroed in place and the slots are never compacted, so walking to
    the first `$00` type byte would stop at the first hole and report almost nothing.
    Iterate all `total` slots and skip the dead ones instead.
    """
    records = []
    for slot in range(total):
        base = slot * RECORD_BYTES
        fields = raw[base:base + RECORD_BYTES]
        if len(fields) < RECORD_BYTES or fields[0] == 0x00:
            continue
        records.append(Record(slot, fields[0] - BLOCK_TYPE_OFFSET, fields[1], fields[2], fields[3]))
    return tuple(records)


def block_counts(blocks):
    """How many blocks of each type are on the grid."""
    return Counter(block.type for block in blocks)


def is_dead_end(blocks):
    """True when some type has exactly one block left.

    Matching is pairwise, so a lone block can never be removed and the stage can no longer
    be cleared. This is a property of the position, not something the cartridge flags.
    """
    return 1 in set(block_counts(blocks).values())


def bounding_box(grid):
    """The rows and columns that are actually part of the playfield.

    Every stage fills all 120 cells; a small stage is the same array with more `$03`
    around it, so the effective shape has to be derived rather than read.
    """
    occupied = [(row, col)
                for row in range(GRID_ROWS) for col in range(GRID_COLS)
                if grid[row][col] != CELL_OUTSIDE]
    if not occupied:
        return (0, -1), (0, -1)
    rows = [row for row, _ in occupied]
    cols = [col for _, col in occupied]
    return (min(rows), max(rows)), (min(cols), max(cols))


def render_grid(grid, cursor=None, trim=True):
    """The grid as ASCII: `#` wall, `=` ledge, `.` empty, `*` clearing, digits for blocks.

    The cursor is drawn as `c` on an empty cell and `¢` on top of anything else, which is
    the alphabet `puzznic.py` renders with.
    """
    (top, bottom), (left, right) = bounding_box(grid) if trim else ((0, GRID_ROWS - 1), (0, GRID_COLS - 1))
    lines = []
    for row in range(top, bottom + 1):
        line = []
        for col in range(left, right + 1):
            value = grid[row][col]
            glyph = CELL_GLYPHS.get(value, str(value - BLOCK_TYPE_OFFSET) if value >= BLOCK_MIN else "?")
            if cursor is not None and (row, col) == tuple(cursor):
                glyph = "c" if value == CELL_EMPTY else "¢"
            line.append(glyph)
        lines.append("".join(line))
    return "\n".join(lines)


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


def read_grid(pyboy):
    return pyboy.memory[GRID_ADDR:GRID_ADDR + GRID_BYTES]


def stage_is_loaded(pyboy):
    """True once a stage sits on the playfield with nothing cleared from it yet."""
    total = pyboy.memory[TOTAL_BLOCKS_ADDR]
    if total == 0 or pyboy.memory[BLOCKS_REMAINING_ADDR] != total:
        return False
    raw = read_grid(pyboy)
    if any(value == CELL_CLEARING for value in raw[::CELL_STRIDE]):
        return False
    return len(decode_blocks(raw)) == total


def settle(pyboy, render=False, max_ticks=SETTLE_MAX_TICKS, stable_ticks=SETTLE_STABLE_TICKS):
    """Run the emulator until the board stops moving, and report whether it did.

    A push is not instantaneous: the block slides, blocks above it fall, matches clear
    through a `$01` transient and the fall can cascade. Snapshotting straight after the
    button press would capture the middle of that. The board counts as settled once no
    cell is mid-clear and the grid has been byte-identical for `stable_ticks` frames.

    Two things cut the wait short. Clearing the last block ends the stage, and the
    cartridge then loads the next round over the top of it, so the moment
    `$D019` reaches zero is the last moment this stage can be observed. And if `$D018`
    changes underneath us a new stage has loaded anyway, so there is nothing left to wait
    for.
    """
    total = pyboy.memory[TOTAL_BLOCKS_ADDR]
    previous, stable = None, 0
    for _ in range(max_ticks):
        pyboy.tick(1, render)
        if pyboy.memory[TOTAL_BLOCKS_ADDR] != total:
            return False
        if total and pyboy.memory[BLOCKS_REMAINING_ADDR] == 0:
            return True
        raw = read_grid(pyboy)
        if any(value == CELL_CLEARING for value in raw[::CELL_STRIDE]):
            previous, stable = raw, 0
            continue
        if raw == previous:
            stable += 1
            if stable >= stable_ticks:
                return True
        else:
            previous, stable = raw, 0
    return False


def _force_stage(context):
    """Hook body: pin `$D003` on the way into the stage loader.

    Writing the stage index from outside on a frame boundary is not enough — a title
    screen can reset it and call the loader within the same frame, and the loader wins.
    Hooking its entry puts the write between the reset and the read.
    """
    pyboy, stage_index = context
    pyboy.memory[STAGE_INDEX_ADDR] = stage_index


def boot(pyboy, render=False, max_ticks=BOOT_MAX_TICKS, press_every=BOOT_PRESS_EVERY):
    """Tap through the boot ROM and title screens until a stage is on the playfield."""
    for frame in range(0, max_ticks, press_every):
        pyboy.button("start" if (frame // press_every) % 2 == 0 else "a", 4)
        pyboy.tick(press_every, render)
        if stage_is_loaded(pyboy):
            return True
    return False


# ----------------------------------------------------------------------------- state

class PuzznicGBState:
    """A settled position: the emulator save-state plus the facts read out of WRAM."""

    def __init__(self, pyboy, depth, stage_types=None):
        self.depth = depth
        self.literals = frozenset()
        self.gb_state = save_state(pyboy)
        self.__update__(pyboy, stage_types)

    def __update__(self, pyboy, stage_types):
        raw = read_grid(pyboy)
        self.grid = decode_grid(raw)
        self.blocks = decode_blocks(raw)
        self.cursor = position(row=pyboy.memory[CURSOR_ROW_ADDR], col=pyboy.memory[CURSOR_COL_ADDR])

        self.total_blocks = pyboy.memory[TOTAL_BLOCKS_ADDR]
        self.blocks_remaining = pyboy.memory[BLOCKS_REMAINING_ADDR]
        self.blocks_cleared = self.total_blocks - self.blocks_remaining
        self.records = decode_records(
            pyboy.memory[RECORDS_ADDR:RECORDS_ADDR + RECORD_BYTES * self.total_blocks],
            self.total_blocks,
        )

        # Block types are drawn by a PRNG at load time, so which types a stage holds is a
        # property of the playthrough. The initial set is carried down the search tree
        # rather than re-derived, or a cleared type would look like one that never existed.
        self.stage_types = (frozenset(block.type for block in self.blocks)
                            if stage_types is None else stage_types)

        # Sampled here, while the emulator still holds this state. Read later from live
        # memory they would describe whichever state was applied last.
        self.stage_cleared = self.total_blocks > 0 and self.blocks_remaining == 0 and not self.blocks
        self.dead_end = is_dead_end(self.blocks)

        predicates = [f"at(cursor, {self.cursor.row}, {self.cursor.col})",
                      f"remaining({self.blocks_remaining})"]
        predicates += [f"at(block-{block.type}, {block.row}, {block.col})" for block in self.blocks]
        matched = self.stage_types - {block.type for block in self.blocks}
        predicates += [f"all-blocks-matched(block-{block_type})" for block_type in sorted(matched)]
        if self.stage_cleared:
            predicates.append("goal-reached")
        if self.dead_end:
            predicates.append("terminal-state")
        self.literals = frozenset(predicates)

    def is_consistent(self):
        """Do the grid, the record array and `$D019` tell the same story?

        The memory map's own cross-check. A mismatch means one of the three addresses has
        drifted — the grid scan and `$D019` are the two that were verified against live
        RAM, so trust those.
        """
        return len(self.blocks) == self.blocks_remaining == len(self.records)

    def block_counts(self):
        return block_counts(self.blocks)

    def bounding_box(self):
        return bounding_box(self.grid)

    def __eq__(self, other):
        # The position is the grid and the cursor; depth, score and history are not part
        # of it, so a state reached two ways compares equal and search can close.
        return (isinstance(other, PuzznicGBState)
                and self.grid == other.grid and self.cursor == other.cursor)

    def __hash__(self):
        return hash((self.grid, self.cursor))

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        return render_grid(self.grid, self.cursor)

    def __repr__(self):
        return (f"<PuzznicGBState(depth={self.depth}, remaining={self.blocks_remaining}"
                f"/{self.total_blocks}, cursor=({self.cursor.row}, {self.cursor.col}))>")

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

class PuzznicGBAction:
    """A button combination held for a number of frames, spelled `"buttons,ticks"`."""

    def __init__(self, action):
        self.action = action
        self.actions_tick_list = self.__parse_action__(action)
        # One input is one unit of plan, whether or not A was held: A is a modifier that
        # turns a direction into a push, not a move of its own.
        self.cost_value = sum(action_cost_map[button] for button, _ in self.actions_tick_list)

    def __parse_action__(self, act):
        buttons, ticks = act.split(",")
        return [(button, int(ticks)) for button in buttons.split("+")]

    def __eq__(self, other):
        return isinstance(other, PuzznicGBAction) and self.action == other.action

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
        return PuzznicGBState(pyboy, state.depth + 1, state.stage_types)


# ------------------------------------------------------------------------ environment

class PuzznicGBEnv(RetroGame):
    """Puzznic, played on the cartridge.

    The ROM is copyrighted and is not distributed with this repo; pass the path to your
    own dump. `fix_index` selects the stage the cartridge's loader will build.
    """

    def __init__(self, romfile, render=False, verify_rom=True,
                 settle_max_ticks=SETTLE_MAX_TICKS, settle_stable_ticks=SETTLE_STABLE_TICKS,
                 boot_max_ticks=BOOT_MAX_TICKS):
        self.romfile = romfile
        # Not `self.render`: that name belongs to the method which prints the history.
        self.render_window = render
        self.pyboy = None
        self.stage_index = None
        self.state = None
        self.state_history = []
        self.actions = action_list
        self.settle_kwargs = {"max_ticks": settle_max_ticks, "stable_ticks": settle_stable_ticks}
        self.boot_max_ticks = boot_max_ticks
        if verify_rom:
            self.__verify_rom__()

    def __verify_rom__(self):
        """Warn when the dump is not the revision these addresses were read from."""
        if not os.path.isfile(self.romfile):
            return
        digest = hashlib.md5(open(self.romfile, "rb").read()).hexdigest()
        if digest != ROM_MD5:
            warnings.warn(
                f"{self.romfile} has MD5 {digest}, not {ROM_MD5} (Puzznic (J).gb). The "
                "addresses this environment reads are revision-specific and may not hold.",
                UserWarning, stacklevel=3)

    def fix_index(self, index):
        """Select the stage. The index is the raw `$D003` value the loader indexes with.

        `$D003` is one byte, so that is the whole range the loader can be pointed at; how
        many of those entries are real stages was never established, and an index past the
        end of the pointer table will build whatever follows it in ROM.
        """
        assert 0 <= index <= 0xFF, "Invalid index: the stage index is a single byte"
        self.stage_index = index

    def reset(self):
        if self.pyboy is not None:
            self.pyboy.stop(save=False)
        self.pyboy = create_pyboy(self.romfile, self.render_window)
        if self.stage_index is not None:
            # Registered before anything runs, so the very first stage load is ours too.
            self.pyboy.hook_register(0, STAGE_LOADER_ENTRY, _force_stage,
                                     (self.pyboy, self.stage_index))
        if not boot(self.pyboy, self.render_window, self.boot_max_ticks):
            raise RuntimeError(
                f"no stage was loaded within {self.boot_max_ticks} frames of booting "
                f"{self.romfile}. Check the ROM is Puzznic (J).gb (MD5 {ROM_MD5}).")
        settle(self.pyboy, self.render_window, **self.settle_kwargs)
        self.state = PuzznicGBState(self.pyboy, 0)
        self.state_history = [self.state]
        return self.state, {"stage_index": self.pyboy.memory[STAGE_INDEX_ADDR],
                            "total_blocks": self.state.total_blocks}

    def is_goal(self, state):
        return state.stage_cleared

    def is_terminal(self, state):
        return state.dead_end

    def __advance__(self, state, action):
        """Apply one action, treating won and lost stages as absorbing.

        Clearing the last block ends the stage and the cartridge loads the next round
        straight over the top of it, so pressing on past a goal state would silently hand
        back a position from a different stage. A dead end is absorbing for the same reason
        the pure-Python environment makes it so: there is nothing left to plan for.
        """
        if self.is_goal(state) or self.is_terminal(state):
            return state
        action = action if isinstance(action, PuzznicGBAction) else PuzznicGBAction(action)
        return action.apply(self.pyboy, state, self.render_window, **self.settle_kwargs)

    def successors(self, state):
        """Every action applied to `state`, minus the ones that change nothing.

        Absorbing states expand to nothing: every action returns the state itself, and the
        self-loop filter drops it.
        """
        successors = []
        for actionstr in self.actions:
            action = PuzznicGBAction(actionstr)
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
        return self.state, self.state.blocks_cleared

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
