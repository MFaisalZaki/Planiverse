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
import os
from collections import Counter, deque, namedtuple

from planiverse.problems.retro_games.gb import (
    GBAction, GBEnv, GBState, create_pyboy, load_state, save_state,
    sprites as _oam_sprites,
)

# --------------------------------------------------------------------------- the ROM

ROM_MD5 = "9a777d82cd7a8913ba1aed2cc854fa50"

# --------------------------------------------------------------------- memory map §1-5

STAGE_INDEX_ADDR = 0xD003        # which stage the loader will build
CURSOR_COL_ADDR = 0xD012
CURSOR_ROW_ADDR = 0xD013
TOTAL_BLOCKS_ADDR = 0xD018       # blocks this stage loaded with; never decremented
BLOCKS_REMAINING_ADDR = 0xD019   # decremented once per block removed

OAM_BUFFER_ADDR = 0xC000         # OAM DMA source: 40 sprites of 4 bytes
OAM_BUFFER_BYTES = 160

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

# ------------------------------------------------------------------ passwords
# Every round has a password, and the table is in the cartridge: 128 ten-byte entries at
# `$47FA`. Eight are the password itself in the game's own text encoding (`A` is `$0A`, so
# `$00`-`$09` are the digits), the ninth is the round number, and the tenth is a check byte.
# Reading it from the ROM rather than hard-coding a transcription means the passwords always
# match the cartridge in hand, and the round numbers validate the parse as it goes.

PASSWORD_TABLE_ADDR = 0x47FA
PASSWORD_LENGTH = 8
PASSWORD_STRIDE = 10
TEXT_LETTER_BASE = 0x0A          # 'A' in the game's text encoding
TEXT_PERIOD = 0x24               # '.' as the ROM stores it
TILE_PERIOD = 0x8B               # '.' as the screen shows it
PASSWORD_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ."

# The password screen. Its cursor, its eight slots and the title menu's cursor are all
# sprites, so the entered password can be read back and every keystroke checked.
MENU_ARROW_TILE = 0xAC
MENU_ENTRIES = {"1player": 88, "2players": 104, "password": 120}
SLOT_Y = 48
SLOT_X = (24, 40, 56, 72, 96, 112, 128, 144)
EMPTY_SLOT_TILE = 0x8C
ENTRY_ORIGIN = (80, 16)          # where the arrow sits for the cell holding 'A'
ENTRY_PITCH = 16
END_CELL = (3, 6)                # the bottom row is NEXT (3,0), BACK (3,3), END (3,6)
TITLE_MAX_TICKS = 900
MENU_PRESS_TICKS = 8
MENU_GAP_TICKS = 26


def decode_text(byte):
    """One byte of the game's text encoding, as a character."""
    if TEXT_LETTER_BASE <= byte < TEXT_LETTER_BASE + 26:
        return chr(ord("A") + byte - TEXT_LETTER_BASE)
    if byte < 0x0A:
        return str(byte)
    if byte in (TEXT_PERIOD, TILE_PERIOD):
        return "."
    return None


def sprites(pyboy):
    """The OAM DMA buffer at `$C000`, as `(y, x, tile)` for every visible sprite."""
    return _oam_sprites(pyboy, OAM_BUFFER_ADDR, visible_only=True)


def menu_cursor(pyboy):
    """Which title-menu entry is selected, or None when the menu is not up."""
    for y, _, tile in sprites(pyboy):
        if tile == MENU_ARROW_TILE:
            for name, row in MENU_ENTRIES.items():
                if y == row:
                    return name
    return None


def entry_cursor(pyboy):
    """Which character cell the password screen's cursor is on, as `(row, col)`."""
    for y, x, tile in sprites(pyboy):
        if tile == MENU_ARROW_TILE:
            return ((y - ENTRY_ORIGIN[0]) // ENTRY_PITCH,
                    (x - ENTRY_ORIGIN[1]) // ENTRY_PITCH)
    return None


def entered_password(pyboy):
    """What the password screen currently shows, with `-` for an empty slot."""
    filled = {x: tile for y, x, tile in sprites(pyboy) if y == SLOT_Y}
    return "".join("-" if filled.get(x, EMPTY_SLOT_TILE) == EMPTY_SLOT_TILE
                   else decode_text(filled[x]) or "?" for x in SLOT_X)


def _tap(pyboy, button, render=False, hold=MENU_PRESS_TICKS, gap=MENU_GAP_TICKS):
    pyboy.button(button, hold)
    pyboy.tick(gap, render)


def wait_for_title(pyboy, render=False, max_ticks=TITLE_MAX_TICKS):
    """Tick until the title menu is up. True if it appeared."""
    for _ in range(0, max_ticks, 30):
        if menu_cursor(pyboy) is not None:
            return True
        pyboy.tick(30, render)
    return menu_cursor(pyboy) is not None


def select_menu_entry(pyboy, entry, render=False):
    """Move the title-menu cursor onto `entry` and press START."""
    order = list(MENU_ENTRIES)
    for _ in range(len(order) * 3):
        here = menu_cursor(pyboy)
        if here is None:
            return False
        if here == entry:
            _tap(pyboy, "start", render)
            return True
        _tap(pyboy, "down", render)
    return False


def steer_entry_cursor(pyboy, target, render=False, budget=32):
    """Walk the password screen's cursor onto `target`, checking it after every press."""
    for _ in range(budget):
        here = entry_cursor(pyboy)
        if here is None:
            return False
        if here == target:
            return True
        if here[0] != target[0]:
            _tap(pyboy, "down" if target[0] > here[0] else "up", render)
        else:
            _tap(pyboy, "right" if target[1] > here[1] else "left", render)
        if entry_cursor(pyboy) == here:
            return False                  # the press did nothing; the screen is not listening
    return False


def enter_password(pyboy, password, render=False, attempts=3):
    """Type `password` on the password screen and confirm it with END.

    Every keystroke is checked against the slot it was meant to fill and retried if it did
    not land — the screen drops a press now and again, and a password that is one character
    short is silently the wrong round rather than an error.
    """
    cells = {ch: (i // 9, i % 9) for i, ch in enumerate(PASSWORD_ALPHABET)}
    for index, character in enumerate(password):
        if character not in cells:
            raise ValueError(f"{character!r} is not on the password screen")
        for _ in range(attempts):
            if not steer_entry_cursor(pyboy, cells[character], render):
                return False
            _tap(pyboy, "a", render)
            if entered_password(pyboy)[index] == character:
                break
        else:
            return False
    if entered_password(pyboy) != password:
        return False
    return steer_entry_cursor(pyboy, END_CELL, render) and (_tap(pyboy, "a", render) or True)


# --------------------------------------------------------------------------- driving

PRESS_TICKS = 6                  # frames a button is held. A fallback only: `calibrate`
                                 # measures the real window off the cartridge, because the
                                 # bound on it is the cursor's auto-repeat delay and that
                                 # is not something the memory map records.
PROBE_MAX_HOLD = 60              # longest hold `calibrate` will try before giving up.
                                 # `Puzznic (J)` repeats its cursor around frame 30.
SETTLE_MAX_TICKS = 600           # give up waiting for the board after ten seconds
SETTLE_STABLE_TICKS = 4          # frames the grid must hold still to count as settled
BOOT_MAX_TICKS = 1800            # thirty seconds of title screens is more than enough
BOOT_PRESS_EVERY = 12
INTRO_MAX_TICKS = 900            # how long to keep waiting for a round to become playable
INTRO_STEP_TICKS = 30            # granularity of that wait

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


def block_counts(blocks):
    """How many blocks of each type are on the grid."""
    return Counter(block.type for block in blocks)


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


# ------------------------------------------------------------------------- emulation
# `create_pyboy`, `save_state` and `load_state` come from the shared `gb` module.

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
    """A direction with room for the cursor to show a repeat, which needs two cells.

    Picking the first direction that moves at all is not enough: with a wall one cell away
    every hold looks identical, and the window comes back as wide as the probe, which is how
    `Puzznic (J)` first reported `(1, 40)` for a cursor that in fact repeats around frame 30.
    """
    fallback = None
    for direction in DIRECTIONS:
        _, cursor = _press(pyboy, state, [(direction, max_hold)], render, **settle_kwargs)
        moved = _cells_moved(state.cursor, cursor, direction)
        if moved >= 2:
            return direction
        if moved >= 1 and fallback is None:
            fallback = direction
    return fallback


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
            for scheme, (prefix, combined) in PUSH_SCHEMES.items():
                if prefix is None:
                    push = [(direction, press_ticks)]
                elif combined:
                    push = [(f"{prefix}+{direction}", press_ticks)]
                else:
                    push = [(prefix, press_ticks), (direction, press_ticks)]
                grid = _walk_then_press(pyboy, state, position(*block[:2]), press_ticks, push,
                                        render, **settle_kwargs)
                if grid is None:
                    break                     # the cursor cannot reach this block at all
                if grid[block.row][block.col] != state.grid[block.row][block.col]:
                    return scheme, prefix
    return None, None


def _walk_then_press(pyboy, state, target, press_ticks, presses, render=False, **settle_kwargs):
    """Rewind to `state`, walk the cursor to `target`, then deliver `presses`.

    Returns the settled grid, or None when the cursor could not be walked there.
    """
    load_state(pyboy, state.gb_state, render)
    if walk_cursor(pyboy, target, press_ticks, render, grid=state.grid, **settle_kwargs) is None:
        return None
    for buttons, hold in presses:
        for button in buttons.split("+"):
            if button != "nop":
                pyboy.button(button, hold)
        pyboy.tick(hold + 1, render)
        settle(pyboy, render, **settle_kwargs)
    return decode_grid(read_grid(pyboy))


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
        low = None
        for hold in range(1, max_hold + 1):
            if prefix is None:
                push = [(direction, hold)]
            elif combined:
                push = [(f"{prefix}+{direction}", hold)]
            else:
                push = [(prefix, press_ticks), (direction, hold)]
            grid = _walk_then_press(pyboy, state, position(block.row, block.col), press_ticks,
                                    push, render, **settle_kwargs)
            if grid is None:
                break                         # the cursor cannot reach this block
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
    measured = scheme is not None
    if not measured:
        scheme, prefix = "modifier", "a"
    push_window = measure_push_window(pyboy, state, press_ticks, scheme, prefix, render,
                                      max_hold, **settle_kwargs)
    push_ticks = None if push_window is None else (push_window[0] + push_window[1]) // 2
    return Calibration(press_ticks, window, scheme, prefix, push_ticks, push_window)


def cursor_of(pyboy):
    return position(row=pyboy.memory[CURSOR_ROW_ADDR], col=pyboy.memory[CURSOR_COL_ADDR])


def wait_until_interactive(pyboy, render=False, max_ticks=INTRO_MAX_TICKS,
                           step=INTRO_STEP_TICKS, press_ticks=PRESS_TICKS):
    """Advance past the round's intro, and report how many frames it took.

    The stage loader fills work RAM before the round has finished announcing itself, so the
    board is fully readable while every button is still ignored — about 200 frames on
    `Puzznic (J)`. A state snapshotted in that window looks perfectly normal and answers no
    action, which is the worst way for this to go wrong: search sees a stage with no legal
    moves rather than an error.

    Rather than hard-code the delay, this presses a direction from a snapshot at increasing
    offsets until the cursor answers, then rewinds and replays only the waiting — so the
    state handed back still has the cursor exactly where the loader put it. If the wait
    alone never works it tries again having pressed START first, because START is the pause
    button once a round is running and the boot sequence taps it.
    """
    start = save_state(pyboy)

    def responds_after(waited, unpause):
        for direction in ("right", "left", "down", "up"):
            load_state(pyboy, start, render)
            if unpause:
                pyboy.button("start", press_ticks)
                pyboy.tick(press_ticks + 2, render)
            if waited:
                pyboy.tick(waited, render)
            before = cursor_of(pyboy)
            pyboy.button(direction, press_ticks)
            pyboy.tick(press_ticks + 12, render)
            if cursor_of(pyboy) != before:
                return True
        return False

    for unpause in (False, True):
        for waited in range(0, max_ticks, step):
            if not responds_after(waited, unpause):
                continue
            load_state(pyboy, start, render)
            if unpause:
                pyboy.button("start", press_ticks)
                pyboy.tick(press_ticks + 2, render)
            if waited:
                pyboy.tick(waited, render)
            return waited
    load_state(pyboy, start, render)
    return None


def boot(pyboy, password=None, render=False, max_ticks=BOOT_MAX_TICKS,
         press_every=BOOT_PRESS_EVERY, title_seen=None):
    """Get from power-on to a loaded stage, and report how it did it.

    Returns `"password"`, `"1player"` or `"tapped"`, or None if no stage ever loaded.

    The cartridge's own title menu is the route worth taking: `PASSWORD` puts the game on
    the round the password belongs to with all of its own state set up the way it expects,
    where poking `$D003` merely swaps the layout under a game that still thinks it is on
    round one. `"tapped"` is the fallback for a cartridge with no such menu — the test ROM
    is one — and is what `stage_index` needs the loader hook for.
    """
    if title_seen if title_seen is not None else wait_for_title(pyboy, render):
        if password is not None:
            if not select_menu_entry(pyboy, "password", render):
                return None
            pyboy.tick(90, render)
            if not enter_password(pyboy, password, render):
                return None
            route = "password"
        else:
            if not select_menu_entry(pyboy, "1player", render):
                return None
            route = "1player"
        for _ in range(0, max_ticks, press_every):
            pyboy.tick(press_every, render)
            if stage_is_loaded(pyboy):
                return route
        return None

    for frame in range(0, max_ticks, press_every):
        pyboy.button("start" if (frame // press_every) % 2 == 0 else "a", 4)
        pyboy.tick(press_every, render)
        if stage_is_loaded(pyboy):
            return "tapped"
    return None


# ----------------------------------------------------------------------------- state

class PuzznicGBState(GBState):
    """A settled position: the emulator save-state plus the facts read out of WRAM."""

    def __init__(self, pyboy, depth, stage_types=None):
        super().__init__(pyboy, depth)
        self.__update__(pyboy, stage_types)

    def __update__(self, pyboy, stage_types):
        raw = read_grid(pyboy)
        self.grid = decode_grid(raw)
        self.blocks = decode_blocks(raw)
        self.cursor = position(row=pyboy.memory[CURSOR_ROW_ADDR], col=pyboy.memory[CURSOR_COL_ADDR])

        self.total_blocks = pyboy.memory[TOTAL_BLOCKS_ADDR]
        self.blocks_remaining = pyboy.memory[BLOCKS_REMAINING_ADDR]
        self.blocks_cleared = self.total_blocks - self.blocks_remaining
        self.records = self.decode_records(
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
        self.dead_end = self.is_dead_end(self.blocks)

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

    # --------------------------------------------------------------- pure derivation
    # Static because they read synthetic RAM as happily as live memory; they live on the
    # class because this state is their only production caller.

    @staticmethod
    def decode_records(raw, total):
        """The live entries of the record array at `$DD00`.

        Cleared records are zeroed in place and the slots are never compacted, so walking
        to the first `$00` type byte would stop at the first hole and report almost
        nothing. Iterate all `total` slots and skip the dead ones instead.
        """
        records = []
        for slot in range(total):
            base = slot * RECORD_BYTES
            fields = raw[base:base + RECORD_BYTES]
            if len(fields) < RECORD_BYTES or fields[0] == 0x00:
                continue
            records.append(Record(slot, fields[0] - BLOCK_TYPE_OFFSET, fields[1],
                                  fields[2], fields[3]))
        return tuple(records)

    @staticmethod
    def is_dead_end(blocks):
        """True when some type has exactly one block left.

        Matching is pairwise, so a lone block can never be removed and the stage can no
        longer be cleared. This is a property of the position, not something the cartridge
        flags.
        """
        return 1 in set(block_counts(blocks).values())

    @staticmethod
    def render_grid(grid, cursor=None, trim=True):
        """The grid as ASCII: `#` wall, `=` ledge, `.` empty, `*` clearing, digits for blocks.

        The cursor is drawn as `c` on an empty cell and `¢` on top of anything else, which
        is the alphabet `puzznic.py` renders with.
        """
        (top, bottom), (left, right) = (bounding_box(grid) if trim
                                        else ((0, GRID_ROWS - 1), (0, GRID_COLS - 1)))
        lines = []
        for row in range(top, bottom + 1):
            line = []
            for col in range(left, right + 1):
                value = grid[row][col]
                glyph = CELL_GLYPHS.get(value, str(value - BLOCK_TYPE_OFFSET)
                                        if value >= BLOCK_MIN else "?")
                if cursor is not None and (row, col) == tuple(cursor):
                    glyph = "c" if value == CELL_EMPTY else "¢"
                line.append(glyph)
            lines.append("".join(line))
        return "\n".join(lines)

    def __eq__(self, other):
        # The position is the grid and the cursor; depth, score and history are not part
        # of it, so a state reached two ways compares equal and search can close.
        return (isinstance(other, PuzznicGBState)
                and self.grid == other.grid and self.cursor == other.cursor)

    def __hash__(self):
        return hash((self.grid, self.cursor))

    def __str__(self):
        return self.render_grid(self.grid, self.cursor)

    def __repr__(self):
        return (f"<PuzznicGBState(depth={self.depth}, remaining={self.blocks_remaining}"
                f"/{self.total_blocks}, cursor=({self.cursor.row}, {self.cursor.col}))>")


# ---------------------------------------------------------------------------- actions

class PuzznicGBAction(GBAction):
    """A button combination held for a number of frames, spelled `"buttons,ticks"`."""

    # One input is one unit of plan, whether or not A was held: A is a modifier that
    # turns a direction into a push, not a move of its own.
    cost_map = action_cost_map

    def __settle__(self, pyboy, render, **settle_kwargs):
        return settle(pyboy, render, **settle_kwargs)

    def __next_state__(self, pyboy, state):
        return PuzznicGBState(pyboy, state.depth + 1, state.stage_types)


#: Cell values the cursor may sit on. Verified on `Puzznic (J)`: it walks onto blocks and
#: over ledges, and refuses walls; `$03` is outside the playfield entirely.
CURSOR_IMPASSABLE = (CELL_WALL, CELL_OUTSIDE)


def cursor_path(grid, start, target):
    """The shortest route the cursor can take from `start` to `target`, as directions.

    Stages are not rectangles — Round 1's bottom row is two cells narrower than the row
    above it — and the cursor cannot cross a wall, so stepping the rows and then the columns
    walks into one. This is a breadth-first search over the cells the cursor may occupy.
    Returns None when no route exists.
    """
    start, target = (start.row, start.col), (target.row, target.col)
    if start == target:
        return []
    passable = lambda r, c: (0 <= r < GRID_ROWS and 0 <= c < GRID_COLS
                             and grid[r][c] not in CURSOR_IMPASSABLE)
    if not passable(*target):
        return None
    frontier, came_from = deque([start]), {start: None}
    while frontier:
        cell = frontier.popleft()
        for direction, (row_step, col_step) in DIRECTIONS.items():
            neighbour = (cell[0] + row_step, cell[1] + col_step)
            if neighbour in came_from or not passable(*neighbour):
                continue
            came_from[neighbour] = (cell, direction)
            if neighbour == target:
                route = []
                while neighbour != start:
                    neighbour, direction = came_from[neighbour]
                    route.append(direction)
                return route[::-1]
            frontier.append(neighbour)
    return None


def walk_cursor(pyboy, target, press_ticks, render=False, grid=None, **settle_kwargs):
    """Steer the cursor to `target`, returning how many presses it took, or None.

    Follows a route found by `cursor_path`, re-reading `$D012`/`$D013` after every press so
    that a move which overshoots (auto-repeat) or is refused is noticed rather than assumed,
    and re-planning from wherever it actually ended up. Gives up when a press makes no
    progress at all.
    """
    grid = decode_grid(read_grid(pyboy)) if grid is None else grid
    presses = 0
    for _ in range((GRID_ROWS + GRID_COLS) * 2):
        here = cursor_of(pyboy)
        if (here.row, here.col) == (target.row, target.col):
            return presses
        route = cursor_path(grid, here, target)
        if not route:
            return None
        pyboy.button(route[0], press_ticks)
        pyboy.tick(press_ticks + 1, render)
        settle(pyboy, render, **settle_kwargs)
        presses += 1
        if cursor_of(pyboy) == here:
            return None                      # the press changed nothing; we are stuck
    return None


# ------------------------------------------------------------------------ environment

class PuzznicGBEnv(GBEnv):
    """Puzznic, played on the cartridge.

    The ROM is copyrighted and is not distributed with this repo; pass the path to your
    own dump. `fix_index` selects the stage the cartridge's loader will build.
    """

    rom_md5 = ROM_MD5
    rom_name = "Puzznic (J).gb"
    action_class = PuzznicGBAction

    @staticmethod
    def read_passwords(romfile):
        """The round passwords, read out of the cartridge.

        Walks the table until an entry stops being a password whose round-number byte
        follows the last one, which both finds the end and checks the parse. Returns them
        round-ordered, so index 0 is round 1.
        """
        with open(romfile, "rb") as handle:
            rom = handle.read()
        passwords, offset, expected = [], PASSWORD_TABLE_ADDR, 1
        while offset + PASSWORD_STRIDE <= len(rom):
            entry = rom[offset:offset + PASSWORD_STRIDE]
            text = [decode_text(byte) for byte in entry[:PASSWORD_LENGTH]]
            if entry[PASSWORD_LENGTH] != expected & 0xFF or any(c is None for c in text):
                break
            passwords.append("".join(text))
            offset, expected = offset + PASSWORD_STRIDE, expected + 1
        return tuple(passwords)

    def __init__(self, romfile, render=False, verify_rom=True, calibrate=True,
                 settle_max_ticks=SETTLE_MAX_TICKS, settle_stable_ticks=SETTLE_STABLE_TICKS,
                 boot_max_ticks=BOOT_MAX_TICKS):
        self.romfile = romfile
        # Read from the cartridge, so they always match the ROM in hand.
        self.passwords = self.read_passwords(romfile) if os.path.isfile(romfile) else ()
        # Not `self.render`: that name belongs to the method which prints the history.
        self.render_window = render
        self.pyboy = None
        self.stage_index = None
        self.state = None
        self.boot_route = None
        self.intro_ticks = None
        self.state_history = []
        self.should_calibrate = calibrate
        self.calibration = None
        self.actions = action_list
        self.settle_kwargs = {"max_ticks": settle_max_ticks, "stable_ticks": settle_stable_ticks}
        self.boot_max_ticks = boot_max_ticks
        if verify_rom:
            self.__verify_rom__()

    def fix_index(self, index):
        """Select the round, zero-based: `fix_index(3)` is round 4.

        `reset()` reaches it the way a player would, by typing the round's password on the
        title screen's PASSWORD entry. The cartridge carries 128 of them, so that is the
        range; `self.passwords[index]` is the one this will type.
        """
        limit = len(self.passwords) or 0x100
        assert 0 <= index < limit, \
            f"Invalid index: this cartridge has {limit} rounds, so 0..{limit - 1}"
        self.stage_index = index

    def password_for(self, index):
        """The password `fix_index(index)` would type."""
        return self.passwords[index] if index < len(self.passwords) else None

    def reset(self):
        self.__restart_emulator__()
        # Whether the cartridge offers a title menu decides how a round is selected, and it
        # has to be settled before anything is pressed: the fallback pokes the stage loader,
        # and the hook that does it must be in place before the loader first runs.
        title = wait_for_title(self.pyboy, self.render_window)
        password = (self.password_for(self.stage_index)
                    if title and self.stage_index is not None else None)
        if self.stage_index is not None and password is None:
            self.pyboy.hook_register(0, STAGE_LOADER_ENTRY, _force_stage,
                                     (self.pyboy, self.stage_index))
        self.boot_route = boot(self.pyboy, password, self.render_window, self.boot_max_ticks,
                               title_seen=title)
        if self.boot_route is None:
            raise RuntimeError(
                f"no stage was loaded within {self.boot_max_ticks} frames of booting "
                f"{self.romfile}"
                + (f" and typing the password {password!r}" if password else "")
                + f". Check the ROM is Puzznic (J).gb (MD5 {ROM_MD5}).")
        settle(self.pyboy, self.render_window, **self.settle_kwargs)
        self.intro_ticks = wait_until_interactive(self.pyboy, self.render_window)
        if self.intro_ticks is None:
            raise RuntimeError(
                f"{self.romfile} loaded a stage but never accepted a button press. The board "
                "is readable during the round intro, so this usually means the intro is "
                "longer than INTRO_MAX_TICKS, or the game is paused.")
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
                            "boot_route": self.boot_route,
                            "password": password,
                            "intro_ticks": self.intro_ticks,
                            "calibration": self.calibration}

    def is_goal(self, state):
        return state.stage_cleared

    def is_terminal(self, state):
        """A dead end: some type has exactly one block left, so the stage cannot clear.

        Absorbing for the same reason the pure-Python environment makes it so — there is
        nothing left to plan for. The win side of `__advance__`'s absorbing rule matters
        here too: clearing the last block ends the stage and the cartridge loads the next
        round straight over the top of it, so pressing on past a goal state would silently
        hand back a position from a different stage.
        """
        return state.dead_end

    def __score__(self, state):
        return state.blocks_cleared


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
