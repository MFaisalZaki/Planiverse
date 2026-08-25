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

PRESS_TICKS = 6                  # frames a button is held. A fallback only: `calibrate`
                                 # measures the real window off the cartridge, because the
                                 # bound on it is the cursor's auto-repeat delay and that
                                 # is not something the memory map records.
PROBE_MAX_HOLD = 40              # longest hold `calibrate` will try before giving up
SETTLE_MAX_TICKS = 600           # give up waiting for the board after ten seconds
SETTLE_STABLE_TICKS = 4          # frames the grid must hold still to count as settled
BOOT_MAX_TICKS = 1800            # thirty seconds of title screens is more than enough
BOOT_PRESS_EVERY = 12

action_cost_map = {"a": 0, "left": 1, "right": 1, "up": 1, "down": 1, "nop": 0}

# The cursor moves in four directions; A turns left/right into a push. There is no
# `a+up`/`a+down` because Puzznic only slides blocks sideways — you cannot lift one.
def button_actions(calibration=None):
    """The primitive button actions, each held for however long calibration settled on.

    Cursor moves and pushes get their own hold: the cartridge repeats a held block on its
    own schedule, and one number for both is one of them being wrong.
    """
    calibration = calibration or Calibration(PRESS_TICKS, None, "modifier", "a")
    moves = [f"{buttons},{calibration.press_ticks}"
             for buttons in ("left", "right", "up", "down")]
    prefix, combined = PUSH_SCHEMES[calibration.push_scheme]
    pushes = [f"{prefix}+{buttons},{push_hold(calibration)}" if prefix and combined
              else f"{buttons},{push_hold(calibration)}"
              for buttons in ("left", "right")]
    return moves + pushes


action_list = [f"{buttons},{PRESS_TICKS}"
               for buttons in ("left", "right", "up", "down", "a+left", "a+right")]

position = namedtuple("Position", ["row", "col"])
Block = namedtuple("Block", ["row", "col", "type", "slot"])
Record = namedtuple("Record", ["slot", "type", "state", "row", "col"])

#: What `calibrate` learned from the cartridge.
#:
#: `press_ticks` is the hold a cursor move should use, and `hold_window` the closed range of
#: holds that move the cursor exactly one cell — its upper end one frame short of the
#: cursor's auto-repeat. `push_ticks` and `push_window` are the same two things for a *push*,
#: measured separately because a held block need not repeat on the same schedule as the
#: cursor; `push_ticks` of None means nothing was measured, so fall back to `press_ticks`.
#: `push_scheme` names how a block is pushed and `push_prefix` is the button to press before
#: the direction when the scheme needs one.
Calibration = namedtuple(
    "Calibration",
    ["press_ticks", "hold_window", "push_scheme", "push_prefix", "push_ticks", "push_window"],
    defaults=(None, None))


def push_hold(calibration):
    """How long to hold a push. Falls back to the cursor hold when none was measured."""
    return calibration.push_ticks or calibration.press_ticks

# How a block gets pushed. Which one a cartridge uses is not in the memory map, so
# `calibrate` finds out by trying each on a real block.
PUSH_SCHEMES = {
    # Hold A and press a direction — one input.
    "modifier": ("a", True),
    # Press A to pick the block up, then press a direction — two inputs.
    "grab": ("a", False),
    # A direction alone moves the block the cursor is sitting on.
    "direct": (None, True),
}


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


# --------------------------------------------------------------------------- probing

DIRECTIONS = {"left": (0, -1), "right": (0, 1), "up": (-1, 0), "down": (1, 0)}


def _press(pyboy, state, presses, render=False, **settle_kwargs):
    """Rewind to `state`, deliver `presses`, and read back the settled board.

    Each press is `(buttons, hold)`, where buttons is a `+`-joined combination. Several
    presses run one after another, which is what a two-input push needs.
    """
    load_state(pyboy, state.gb_state, render)
    for buttons, hold in presses:
        for button in buttons.split("+"):
            if button != "nop":
                pyboy.button(button, hold)
        pyboy.tick(hold + 1, render)
        settle(pyboy, render, **settle_kwargs)
    raw = read_grid(pyboy)
    cursor = position(row=pyboy.memory[CURSOR_ROW_ADDR], col=pyboy.memory[CURSOR_COL_ADDR])
    return decode_grid(raw), cursor


def _cells_moved(before, after, direction):
    """How far the cursor travelled along `direction`, negative if it went the other way."""
    step = DIRECTIONS[direction]
    delta = (after.row - before.row, after.col - before.col)
    if step[0]:
        return delta[0] // step[0] if delta[1] == 0 else 0
    return delta[1] // step[1] if delta[0] == 0 else 0


def _probe_direction(pyboy, state, render, settle_kwargs, max_hold):
    """A direction the cursor can actually travel in from here, or None."""
    for direction in DIRECTIONS:
        _, cursor = _press(pyboy, state, [(direction, max_hold // 2)], render, **settle_kwargs)
        if _cells_moved(state.cursor, cursor, direction) > 0:
            return direction
    return None


def measure_hold_window(pyboy, state, render=False, max_hold=PROBE_MAX_HOLD, **settle_kwargs):
    """The closed range of hold lengths that move the cursor exactly one cell.

    The lower end is where a press starts registering at all; the upper end is one frame
    short of the cursor's auto-repeat, past which a single action moves two cells and the
    state a planner gets back is not the one its action described. Neither bound is in the
    memory map, so both are measured here.

    Returns `(low, high)`, or None if no hold moves the cursor exactly one cell.
    """
    direction = _probe_direction(pyboy, state, render, settle_kwargs, max_hold)
    if direction is None:
        return None
    low = None
    for hold in range(1, max_hold + 1):
        _, cursor = _press(pyboy, state, [(direction, hold)], render, **settle_kwargs)
        moved = _cells_moved(state.cursor, cursor, direction)
        if moved == 1 and low is None:
            low = hold
        elif moved > 1 and low is not None:
            return (low, hold - 1)
    return None if low is None else (low, max_hold)


def probe_push_scheme(pyboy, state, press_ticks, render=False, **settle_kwargs):
    """Find out how this cartridge moves a block, by trying each scheme on a real one.

    Returns `(scheme, prefix)` from `PUSH_SCHEMES`, or `(None, None)` if no candidate moved
    a block — which usually means the cursor could not be walked onto one.
    """
    for block in state.blocks:
        for direction in ("left", "right"):
            neighbour = (block.row, block.col + DIRECTIONS[direction][1])
            if not (0 <= neighbour[1] < GRID_COLS):
                continue
            if state.grid[neighbour[0]][neighbour[1]] != CELL_EMPTY:
                continue                      # nowhere to push it, so it proves nothing
            walk = walk_presses(state.cursor, position(*block[:2]), press_ticks)
            for scheme, (prefix, combined) in PUSH_SCHEMES.items():
                if prefix is None:
                    push = [(direction, press_ticks)]
                elif combined:
                    push = [(f"{prefix}+{direction}", press_ticks)]
                else:
                    push = [(prefix, press_ticks), (direction, press_ticks)]
                grid, _ = _press(pyboy, state, walk + push, render, **settle_kwargs)
                if grid[block.row][block.col] != state.grid[block.row][block.col]:
                    return scheme, prefix
    return None, None


def walk_presses(start, target, press_ticks):
    """The presses that walk the cursor from `start` to `target`, rows first."""
    presses = []
    row_step = "down" if target.row > start.row else "up"
    col_step = "right" if target.col > start.col else "left"
    presses += [(row_step, press_ticks)] * abs(target.row - start.row)
    presses += [(col_step, press_ticks)] * abs(target.col - start.col)
    return presses


def _slide_distance(before, after, row, col, step, max_cells):
    """How far the block that was at `(row, col)` travelled, or None if it is not there.

    None means the probe destroyed its own evidence — the block fell, or met a same-typed
    neighbour and cleared — so the hold it was testing cannot be read off this board.
    """
    if after[row][col] == before[row][col]:
        return 0
    for cells in range(1, max_cells + 1):
        target = col + cells * step
        if not 0 <= target < GRID_COLS:
            break
        if after[row][target] == before[row][col]:
            return cells
        if after[row][target] != CELL_EMPTY:
            break
    return None


def push_probe_candidates(state, cells=2):
    """Blocks that can be slid `cells` cells without falling, matching, or hitting anything.

    A probe has to be readable afterwards. A block that drops down a hole, or lands next to
    its own colour and vanishes, tells you nothing about how far the push went — so those
    are filtered out here rather than discovered halfway through a measurement.
    """
    for block in state.blocks:
        kind = state.grid[block.row][block.col]
        for direction in ("left", "right"):
            step = DIRECTIONS[direction][1]
            path = [block.col + n * step for n in range(1, cells + 1)]
            if any(not 0 <= col < GRID_COLS for col in path):
                continue
            if any(state.grid[block.row][col] != CELL_EMPTY for col in path):
                continue
            # Something solid underneath every cell it passes over, or it falls.
            if block.row + 1 < GRID_ROWS and any(
                    state.grid[block.row + 1][col] == CELL_EMPTY for col in path):
                continue
            # And no same-coloured neighbour anywhere along it, or it clears mid-probe.
            touching = [(block.row + dr, col)
                        for col in path + [block.col + (cells + 1) * step]
                        for dr in (-1, 0, 1)
                        if 0 <= block.row + dr < GRID_ROWS and 0 <= col < GRID_COLS]
            if any(state.grid[r][c] == kind and (r, c) != (block.row, block.col)
                   for r, c in touching):
                continue
            yield block, direction


def measure_push_window(pyboy, state, press_ticks, scheme, prefix, render=False,
                        max_hold=PROBE_MAX_HOLD, **settle_kwargs):
    """The closed range of holds that push a block exactly one cell.

    Measured separately from the cursor's window because the two need not agree: a cartridge
    is free to repeat a held block on its own schedule, and holding past that point slides
    the block two cells, which can match and clear it. The state the planner is handed then
    describes something its action never asked for.

    Returns `(low, high)`, or None if no block on this board could be probed safely.
    """
    _, combined = PUSH_SCHEMES[scheme]
    for block, direction in push_probe_candidates(state):
        step = DIRECTIONS[direction][1]
        walk = walk_presses(state.cursor, position(block.row, block.col), press_ticks)
        low = None
        for hold in range(1, max_hold + 1):
            if prefix is None:
                push = [(direction, hold)]
            elif combined:
                push = [(f"{prefix}+{direction}", hold)]
            else:
                push = [(prefix, press_ticks), (direction, hold)]
            grid, _ = _press(pyboy, state, walk + push, render, **settle_kwargs)
            moved = _slide_distance(state.grid, grid, block.row, block.col, step, max_cells=4)
            if moved is None:
                break                       # unreadable; try another block
            if moved == 1 and low is None:
                low = hold
            elif moved > 1 and low is not None:
                return (low, hold - 1)
        if low is not None:
            return (low, max_hold)
    return None


def calibrate(pyboy, state, render=False, max_hold=PROBE_MAX_HOLD, **settle_kwargs):
    """Measure how this cartridge wants to be driven.

    Everything here is a property of the game, not of the stage, so one call per cartridge
    is enough. It costs on the order of a second: a few dozen presses, each rewound to
    `state` first, so nothing it does survives.
    """
    window = measure_hold_window(pyboy, state, render, max_hold, **settle_kwargs)
    if window is None:
        # The cursor never moved. Rather than silently drive the game with a made-up hold,
        # fall back to the documented default and say the window is unknown.
        return Calibration(PRESS_TICKS, None, "modifier", "a")
    low, high = window
    # Middle of the window: far enough above `low` to survive a frame of jitter in when the
    # game samples input, far enough below `high` not to trip auto-repeat. The extra frames
    # cost about 0.07 ms each, so there is nothing to win by shaving them.
    press_ticks = (low + high) // 2
    scheme, prefix = probe_push_scheme(pyboy, state, press_ticks, render, **settle_kwargs)
    if scheme is None:
        scheme, prefix = "modifier", "a"
    push_window = measure_push_window(pyboy, state, press_ticks, scheme, prefix, render,
                                      max_hold, **settle_kwargs)
    push_ticks = None if push_window is None else (push_window[0] + push_window[1]) // 2
    return Calibration(press_ticks, window, scheme, prefix, push_ticks, push_window)


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


class PuzznicGBPush:
    """Walk the cursor onto the block at `(row, col)` and push it one cell sideways.

    The primitive action set spends over 90% of its expansions moving the cursor around:
    walking to a block is not a decision, it is the overhead of making one. This collapses
    that into a single action, so the branching factor is the number of pushes actually
    available — at most two per block — rather than four directions from wherever the
    cursor happens to be.

    The walk is adaptive rather than a precomputed button sequence: it presses towards the
    target, re-reads `$D012`/`$D013`, and stops when it arrives or stops making progress.
    A cartridge whose cursor cannot cross some cell therefore yields a no-op, which
    `successors` filters, instead of a plan that quietly does the wrong thing.
    """

    def __init__(self, row, col, direction, calibration=None, input_count=None):
        if direction not in ("left", "right"):
            raise ValueError("Puzznic slides blocks sideways; direction must be left or right")
        self.row = row
        self.col = col
        self.direction = direction
        self.calibration = calibration or Calibration(PRESS_TICKS, None, "modifier", "a")
        self.input_count = input_count
        self.action = f"push({row},{col},{direction})"

    def __eq__(self, other):
        return isinstance(other, PuzznicGBPush) and self.action == other.action

    def __hash__(self):
        return hash(self.action)

    def __lt__(self, other):
        return self.action < str(getattr(other, "action", other))

    def __str__(self):
        return f"push_{self.row}_{self.col}_{self.direction}"

    def __repr__(self):
        return str(self)

    def cost(self):
        """How many button presses this is, which is what a person would have to do."""
        return self.input_count if self.input_count is not None else 1

    def push_presses(self):
        """The presses that push, once the cursor is on the block."""
        ticks = push_hold(self.calibration)
        prefix, combined = PUSH_SCHEMES[self.calibration.push_scheme]
        if prefix is None:
            return [(self.direction, ticks)]
        if combined:
            return [(f"{prefix}+{self.direction}", ticks)]
        return [(prefix, ticks), (self.direction, ticks)]

    def apply(self, pyboy, state, render=False, **settle_kwargs):
        load_state(pyboy, state.gb_state, render)
        target = position(row=self.row, col=self.col)
        presses = walk_cursor(pyboy, target, self.calibration.press_ticks, render,
                              **settle_kwargs)
        if presses is None:
            # The cursor could not be walked onto the block. Hand back the parent so this
            # shows up as a self-loop and gets dropped, rather than a half-done move.
            return PuzznicGBState(pyboy, state.depth, state.stage_types)
        for buttons, hold in self.push_presses():
            for button in buttons.split("+"):
                pyboy.button(button, hold)
            pyboy.tick(hold + 1, render)
            settle(pyboy, render, **settle_kwargs)
            presses += 1
        if self.input_count is None:
            self.input_count = presses
        return PuzznicGBState(pyboy, state.depth + 1, state.stage_types)


def walk_cursor(pyboy, target, press_ticks, render=False, max_presses=None, **settle_kwargs):
    """Steer the cursor to `target`, returning how many presses it took, or None.

    Presses one direction at a time and re-reads the cursor after each, so an action that
    overshoots (auto-repeat) or is refused (an obstacle) is noticed rather than assumed.
    Gives up as soon as a press makes no progress.
    """
    max_presses = max_presses or (GRID_ROWS + GRID_COLS) * 2
    presses = 0
    for _ in range(max_presses):
        row, col = pyboy.memory[CURSOR_ROW_ADDR], pyboy.memory[CURSOR_COL_ADDR]
        if (row, col) == (target.row, target.col):
            return presses
        if row != target.row:
            button = "down" if target.row > row else "up"
        else:
            button = "right" if target.col > col else "left"
        pyboy.button(button, press_ticks)
        pyboy.tick(press_ticks + 1, render)
        settle(pyboy, render, **settle_kwargs)
        presses += 1
        if (pyboy.memory[CURSOR_ROW_ADDR], pyboy.memory[CURSOR_COL_ADDR]) == (row, col):
            return None                      # the press changed nothing; we are stuck
    return None


def available_pushes(state, calibration=None):
    """Every push the board allows, without touching the emulator.

    A block only moves into an empty cell — the movement check at `1:506E` rejects every
    non-zero cell value, so ledges and walls obstruct exactly as each other. Filtering here
    means the emulator is never run for a push that cannot happen.
    """
    calibration = calibration or Calibration(PRESS_TICKS, None, "modifier", "a")
    # "grab" needs A and then the direction; the other two schemes are a single press.
    push_inputs = 2 if calibration.push_scheme == "grab" else 1
    pushes = []
    for block in state.blocks:
        for direction in ("left", "right"):
            col = block.col + DIRECTIONS[direction][1]
            if not 0 <= col < GRID_COLS or state.grid[block.row][col] != CELL_EMPTY:
                continue
            walk = abs(state.cursor.row - block.row) + abs(state.cursor.col - block.col)
            pushes.append(PuzznicGBPush(block.row, block.col, direction, calibration,
                                        input_count=walk + push_inputs))
    return pushes


# ------------------------------------------------------------------------ environment

class PuzznicGBEnv(RetroGame):
    """Puzznic, played on the cartridge.

    The ROM is copyrighted and is not distributed with this repo; pass the path to your
    own dump. `fix_index` selects the stage the cartridge's loader will build.
    """

    #: `"push"` expands one action per legal push; `"button"` expands one per button press.
    ACTION_MODELS = ("push", "button")

    def __init__(self, romfile, render=False, verify_rom=True, actions="push", calibrate=True,
                 settle_max_ticks=SETTLE_MAX_TICKS, settle_stable_ticks=SETTLE_STABLE_TICKS,
                 boot_max_ticks=BOOT_MAX_TICKS):
        assert actions in self.ACTION_MODELS, f"actions must be one of {self.ACTION_MODELS}"
        self.romfile = romfile
        # Not `self.render`: that name belongs to the method which prints the history.
        self.render_window = render
        self.pyboy = None
        self.stage_index = None
        self.state = None
        self.state_history = []
        self.action_model = actions
        self.should_calibrate = calibrate
        self.calibration = None
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
        if self.should_calibrate and self.calibration is None:
            # A property of the cartridge, not of the stage, so once is enough.
            self.calibration = calibrate(self.pyboy, self.state, self.render_window,
                                         **self.settle_kwargs)
            self.actions = button_actions(self.calibration)
            load_state(self.pyboy, self.state.gb_state, self.render_window)
        self.state_history = [self.state]
        return self.state, {"stage_index": self.pyboy.memory[STAGE_INDEX_ADDR],
                            "total_blocks": self.state.total_blocks,
                            "calibration": self.calibration}

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
        if isinstance(action, str):
            action = PuzznicGBAction(action)
        return action.apply(self.pyboy, state, self.render_window, **self.settle_kwargs)

    def available_actions(self, state):
        """The actions worth trying from `state`, before any of them are applied.

        In `push` mode this is the legal pushes, which is where the branching factor comes
        from; in `button` mode it is the fixed list of button presses.
        """
        if self.action_model == "push":
            return available_pushes(state, self.calibration)
        return [PuzznicGBAction(actionstr) for actionstr in self.actions]

    def successors(self, state):
        """Every action applied to `state`, minus the ones that change nothing.

        Absorbing states expand to nothing: every action returns the state itself, and the
        self-loop filter drops it.
        """
        successors = []
        for action in self.available_actions(state):
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
        """The actions on offer. In `push` mode that depends on the board, so this needs a
        state and reports the ones available from the current one."""
        if self.action_model == "push":
            if self.state is None:
                raise ValueError("Game not initialized. Call reset() first.")
            return available_pushes(self.state, self.calibration)
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


def _report(romfile, stage=None, render=False):
    """Print what this cartridge wants, for every stage asked about."""
    env = PuzznicGBEnv(romfile, render=render)
    try:
        for index in ([stage] if stage is not None else [None]):
            if index is not None:
                env.fix_index(index)
            state, info = env.reset()
            calibration = info["calibration"]
            print(f"stage {info['stage_index']}: {state.total_blocks} blocks\n")
            print(state, "\n")
            print(f"  cursor hold  {calibration.hold_window}  -> press_ticks "
                  f"{calibration.press_ticks}")
            if calibration.push_window is None:
                print("  push hold    not measurable on this stage — no block could be slid "
                      "two cells without falling or matching")
            else:
                print(f"  push hold    {calibration.push_window}  -> push_ticks "
                      f"{push_hold(calibration)}")
            print(f"  push scheme  {calibration.push_scheme}"
                  f" ({'A + direction' if calibration.push_scheme == 'modifier' else calibration.push_scheme})")
            print(f"\n  button actions: {button_actions(calibration)}")
    finally:
        env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure how a Puzznic cartridge wants to be driven: how long a button "
                    "must be held to move the cursor, or a held block, exactly one cell.")
    parser.add_argument("rom", help="path to Puzznic (J).gb")
    parser.add_argument("--stage", type=int, default=None, help="stage index to load")
    parser.add_argument("--render", action="store_true", help="open an SDL2 window")
    args = parser.parse_args()
    _report(args.rom, args.stage, args.render)
