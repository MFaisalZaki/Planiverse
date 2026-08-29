"""Boxxle II on a Game Boy, driven through PyBoy.

The sibling module `gameboy_py/boxxle2.py` re-implements Sokoban in Python over the same 120
boards. This one plays the real cartridge inside an emulator and reads the game's own board
out of work RAM, so the transition function is the cartridge's code rather than a
reconstruction of its rules. States carry an emulator save-state, which is what lets search
branch: applying an action rewinds the machine to the parent first.

The addresses below come from a reverse-engineering pass over `Boxxle II (USA, Europe).gb`
(MD5 `308abd707a48ee9d69c287d818469fd6`) and are documented in
`docs/environments/boxxle2-gb-memory-map.md`. They are revision-specific: another dump will
read garbage, which is why `Boxxle2GBEnv` checks the ROM's MD5 and warns when it differs.

    env = Boxxle2GBEnv("Boxxle II (USA, Europe).gb")
    env.fix_index(0)
    state, info = env.reset()
    print(state)
    for action, successor in env.successors(state):
        ...

Two things about this cartridge shape the module.

**The board is already decoded in RAM.** `StartLevel` decompresses each level into three
360-byte planes (goals, boxes and walls, 20 bytes per row), so the environment reads the
position directly rather than inferring it from tiles. That also means `is_goal` is exact
(every box on a goal) and, unusually for a cartridge environment, so is a useful part of
`is_terminal`: the walls are known, so a box shoved into a corner can be recognised as
unrecoverable without asking the game.

**Clearing a level destroys the evidence.** A completed board triggers the cartridge's
congratulation-and-replay sequence, which redraws the plane buffers with something that is
not a Sokoban position at all; a state snapshotted 30 frames after the winning push decodes
as garbage. `settle` therefore stops the moment the boxes are home, and `__advance__` treats
a solved position as absorbing.
"""
from collections import namedtuple

from planiverse.environments.gameboy.gb import GBAction, GBEnv, GBState, load_state

# --------------------------------------------------------------------------- the ROM

ROM_MD5 = "308abd707a48ee9d69c287d818469fd6"
ROM_NAME = "Boxxle II (USA, Europe).gb"
ROM_BYTES = 32768                # ROM ONLY: $0000-$7FFF maps 1:1 onto file offsets

# ------------------------------------------------------------------- memory map §4-5
# Every address here is in `docs/environments/boxxle2-gb-memory-map.md`, with how it was
# established. The three plane buffers and the player offset are the load-bearing ones and
# were checked against the ROM's own level data for all 120 levels.

BOARD_GOAL_ADDR = 0xC922         # plane 0: goal squares
BOARD_BOX_ADDR = 0xCA8A          # plane 1: boxes
BOARD_WALL_ADDR = 0xCBF2         # plane 2: walls
PLANE_BYTES = 360                # 20 columns x 18 rows
ROW_STRIDE = 20                  # *not* the hardware tilemap's 32

PLAYER_OFFSET_LOW = 0xC10F       # 16-bit linear board offset, low byte
PLAYER_OFFSET_HIGH = 0xC110      # ... and high byte
BOARD_HEIGHT_ADDR = 0xC120
BOARD_WIDTH_ADDR = 0xC121
CELL_SIZE_ADDR = 0xC0FE          # $10 for a board that fits at 16px cells, else $08
MOVE_DIRECTION_ADDR = 0xC116     # 1 left, 2 up, 3 down, 4 right

STAGE_ADDR = 0xC162              # 0-11
LEVEL_IN_STAGE_ADDR = 0xC352     # 0-9
GAME_STATE_ADDR = 0xC34E
DEMO_LEVEL_ADDR = 0xC350         # non-zero forces the attract-mode level at $4F08
EDIT_MODE_ADDR = 0xC351

OAM_BUFFER_ADDR = 0xC000         # OAM DMA source: 40 sprites of 4 bytes

LOAD_LEVEL_HEADER = 0x0F53       # bank 0; reads STAGE_ADDR/LEVEL_IN_STAGE_ADDR a few
                                 # instructions in, which is what makes it hookable

#: `$C34E` on each screen the boot route walks through. Measured by booting the cartridge and
#: watching the byte, not read off the dispatch table: the disassembly's numbering and the
#: running game's disagree, and the running game is the one being driven.
STATE_TITLE = 0x04               # "PUSH START KEY"
STATE_MUSIC = 0x10               # MUSIC: BGM A / B / C
STATE_MENU = 0x20                # MENU: PLAY / PASSKEY / CREATE
STATE_CUTSCENE = 0x06            # the story panel before stage 1
STATE_PLAYING = 0x00
STATE_PAUSED = 0x40              # what START opens mid-level; the action set never sends it

#: Screens where pressing START moves the boot along. START during play opens the pause
#: overlay instead, so booting has to stop tapping the moment a board appears.
STATE_ADVANCED_BY_START = (STATE_TITLE, STATE_MUSIC, STATE_MENU)

# ---------------------------------------------------------------- memory map §4.2
# The level table, so the environment can say what it is about to load without booting it,
# and so the Python twin's boards can be regenerated from a cartridge rather than retyped.

LEVEL_TABLE_ADDR = 0x4E18        # 120 little-endian words
LEVELS_PER_STAGE = 10
STAGE_COUNT = 12
LEVEL_COUNT = LEVELS_PER_STAGE * STAGE_COUNT

WALL, BOX, GOAL, BOX_ON_GOAL, PLAYER, PLAYER_ON_GOAL, FLOOR = "#", "$", "o", "*", "@", "+", " "

#: The glyphs a cell can carry, in the alphabet `boxxle2.py` renders with.
BOX_GLYPHS = (BOX, BOX_ON_GOAL)
PLAYER_GLYPHS = (PLAYER, PLAYER_ON_GOAL)
GOAL_GLYPHS = (GOAL, BOX_ON_GOAL, PLAYER_ON_GOAL)

Position = namedtuple("Position", ["row", "col"])

#: What `calibrate` learned from the cartridge: the hold to use, and the closed range of holds
#: that move the player exactly one cell. The upper end is one frame short of the d-pad's
#: auto-repeat, past which a single action moves two cells and the state a planner is handed
#: back is not the one its action described.
Calibration = namedtuple("Calibration", ["press_ticks", "hold_window"])


# ------------------------------------------------------------------- reading the ROM
# `$26C3` then `$27F6`, in Python. Pure functions of the cartridge image, so they are the one
# part of this module that needs neither an emulator nor a boot.

def level_pointers(rom):
    """The 120 little-endian words at `$4E18`, in play order."""
    return tuple(rom[LEVEL_TABLE_ADDR + 2 * i] | (rom[LEVEL_TABLE_ADDR + 2 * i + 1] << 8)
                 for i in range(LEVEL_COUNT))


def expand_record(rom, address, width, height):
    """Stage 1 (`$26C3`): the sparse-byte flag/literal stream, as `(bytes, record_size)`.

    A set flag bit takes the next literal, a clear one emits `$00`. The record is a bitmap of
    `ceil(3*W*H/64)` bytes followed by one literal per set bit, and the length is implied by
    the board's dimensions rather than stored, which is why `record_size` comes back too:
    it is the only way to know where the next level begins, and agreeing with the pointer
    table's deltas for all 120 records is what verifies this decoder.
    """
    bits = 3 * width * height
    flag_bytes = -(-bits // 64)
    output_bytes = -(-bits // 8)
    flags = rom[address:address + flag_bytes]
    literals, used = address + flag_bytes, 0
    out = bytearray()
    for index in range(output_bytes):
        byte, bit = divmod(index, 8)
        if byte < len(flags) and (flags[byte] >> (7 - bit)) & 1:
            out.append(rom[literals + used])
            used += 1
        else:
            out.append(0)
    return bytes(out), 4 + flag_bytes + used


def unpack_planes(data, width, height):
    """Stage 2 (`$27F6`): one continuous MSB-first bitstream into three `height x width` planes.

    In order: goals, boxes, walls. `popcount(goals) == popcount(boxes)` for every level on the
    cartridge, which is the invariant Sokoban requires and the second check on this decoder.
    """
    planes, index = [], 0
    for _ in range(3):
        plane = []
        for _row in range(height):
            row = []
            for _col in range(width):
                row.append((data[index >> 3] >> (7 - (index & 7))) & 1)
                index += 1

            plane.append(tuple(row))
        planes.append(tuple(plane))
    return tuple(planes)


def decode_level(rom, address):
    """One level record, as `(rows, record_size)` with `rows` in the ASCII alphabet above."""
    width, height = rom[address], rom[address + 1]
    start_col, start_row = rom[address + 2] - 1, rom[address + 3] - 1   # stored 1-based
    data, size = expand_record(rom, address + 4, width, height)
    goal, box, wall = unpack_planes(data, width, height)
    rows = []
    for row in range(height):
        line = []
        for col in range(width):
            if wall[row][col]:
                glyph = WALL
            elif box[row][col]:
                glyph = BOX_ON_GOAL if goal[row][col] else BOX
            elif goal[row][col]:
                glyph = GOAL
            else:
                glyph = FLOOR
            if (row, col) == (start_row, start_col):
                glyph = PLAYER_ON_GOAL if goal[row][col] else PLAYER
            line.append(glyph)
        rows.append("".join(line))
    return tuple(rows), size


def read_levels(romfile):
    """All 120 boards, as tuples of ASCII rows, read out of a cartridge image.

    This is where `boxxle2.py`'s levels came from. Reading them back is also how a claim about
    a level can be settled without a screenshot: the cartridge is the authority, and it is
    thirty lines of decoding away.
    """
    with open(romfile, "rb") as handle:
        rom = handle.read()
    return tuple(decode_level(rom, pointer)[0] for pointer in level_pointers(rom))


def verify_level_table(rom):
    """Check every decoded record against the pointer-table delta that follows it.

    Returns the indices that disagree. An empty result means the whole table decoded
    consistently, which is the cheapest evidence there is that the format above is right.
    """
    pointers = level_pointers(rom)
    mismatched = []
    for index, pointer in enumerate(pointers[:-1]):
        _, size = decode_level(rom, pointer)
        if size != pointers[index + 1] - pointer:
            mismatched.append(index)
    return tuple(mismatched)


# ------------------------------------------------------------------- pure decoding
# Split out from the emulator so they can be tested against synthetic RAM.

def decode_board(goal, box, wall, width, height, player_offset):
    """The three plane buffers as a grid of glyphs, `height` rows of `width` glyphs."""
    rows = []
    for row in range(height):
        line = []
        for col in range(width):
            cell = row * ROW_STRIDE + col
            if wall[cell]:
                glyph = WALL
            elif box[cell]:
                glyph = BOX_ON_GOAL if goal[cell] else BOX
            elif goal[cell]:
                glyph = GOAL
            else:
                glyph = FLOOR
            if cell == player_offset and glyph in (FLOOR, GOAL):
                glyph = PLAYER_ON_GOAL if glyph == GOAL else PLAYER
            line.append(glyph)
        rows.append(tuple(line))
    return tuple(rows)


def offset_to_position(offset):
    """A 16-bit board offset as `(row, col)`. The stride is 20, not the tilemap's 32."""
    return Position(*divmod(offset, ROW_STRIDE))


def position_to_offset(position):
    return position.row * ROW_STRIDE + position.col


def boxes(grid):
    """Where the boxes are, as positions."""
    return tuple(Position(row, col)
                 for row, cells in enumerate(grid) for col, glyph in enumerate(cells)
                 if glyph in BOX_GLYPHS)


def goals(grid):
    return tuple(Position(row, col)
                 for row, cells in enumerate(grid) for col, glyph in enumerate(cells)
                 if glyph in GOAL_GLYPHS)


def boxes_home(grid):
    """How many boxes are standing on a goal."""
    return sum(1 for row in grid for glyph in row if glyph == BOX_ON_GOAL)


def is_solved(grid):
    """Every box on a goal, and there is at least one box.

    The box count guard matters: an empty plane buffer (a level that has not loaded, or one
    the clear sequence has already wiped) satisfies "no box is off a goal" vacuously, and
    without this would read as a win.
    """
    return any(glyph in BOX_GLYPHS for row in grid for glyph in row) \
        and not any(glyph == BOX for row in grid for glyph in row)


def stuck_boxes(grid):
    """Boxes wedged in a corner of walls, off a goal: the position can no longer be solved.

    This is the sound half of Sokoban deadlock detection and no more: a box with a wall on one
    of the vertical sides *and* one of the horizontal sides can never be moved again by
    anyone, so if it is not already home the level is lost. Positions that are dead for
    subtler reasons (a wall-hugging row with no goal on it, two boxes frozen against each
    other) are not claimed here, because a wrong `is_terminal` prunes a solvable branch and
    that is a much worse failure than missing a dead one.
    """
    height, width = len(grid), len(grid[0]) if grid else 0
    wall = lambda row, col: not (0 <= row < height and 0 <= col < width) or grid[row][col] == WALL
    stuck = []
    for position in boxes(grid):
        if grid[position.row][position.col] == BOX_ON_GOAL:
            continue
        vertical = wall(position.row - 1, position.col) or wall(position.row + 1, position.col)
        horizontal = wall(position.row, position.col - 1) or wall(position.row, position.col + 1)
        if vertical and horizontal:
            stuck.append(position)
    return tuple(stuck)


def render_grid(grid):
    return "\n".join("".join(row).rstrip() for row in grid)


# ------------------------------------------------------------------------- emulation
# `create_pyboy`, `save_state` and `load_state` come from the shared `gb` module.

def read_planes(pyboy):
    """The three board planes, as raw 360-byte buffers."""
    return (bytes(pyboy.memory[BOARD_GOAL_ADDR:BOARD_GOAL_ADDR + PLANE_BYTES]),
            bytes(pyboy.memory[BOARD_BOX_ADDR:BOARD_BOX_ADDR + PLANE_BYTES]),
            bytes(pyboy.memory[BOARD_WALL_ADDR:BOARD_WALL_ADDR + PLANE_BYTES]))


def player_offset(pyboy):
    return pyboy.memory[PLAYER_OFFSET_HIGH] * 256 + pyboy.memory[PLAYER_OFFSET_LOW]


def board_shape(pyboy):
    return pyboy.memory[BOARD_WIDTH_ADDR], pyboy.memory[BOARD_HEIGHT_ADDR]


def read_board(pyboy):
    """The position on screen, as a grid of glyphs."""
    width, height = board_shape(pyboy)
    goal, box, wall = read_planes(pyboy)
    return decode_board(goal, box, wall, width, height, player_offset(pyboy))


def shadow_oam(pyboy):
    """The OAM DMA buffer at `$C000`. The player's slide lives here and nowhere else."""
    return bytes(pyboy.memory[OAM_BUFFER_ADDR:OAM_BUFFER_ADDR + 160])


#: Board sizes the cartridge actually ships, from `$4E18`: 6x5 at the smallest, 16x16 at the
#: largest. `level_is_loaded` uses them to tell a decompressed board from whatever `$C120`
#: and `$C121` happen to hold on a menu screen.
MIN_BOARD, MAX_BOARD = 5, 20


def level_is_loaded(pyboy):
    """True once a playable board sits in the plane buffers.

    Four things have to agree, because `$C34E` is `$00` both during play *and* for the first
    sixty frames after power-on, and `$C120`/`$C121` keep whatever the last board left in
    them: the game must be in the playing state, the dimensions must be a board the cartridge
    could have loaded, the player must be somewhere on it, and there must be boxes to push.
    """
    if pyboy.memory[GAME_STATE_ADDR] != STATE_PLAYING:
        return False
    width, height = board_shape(pyboy)
    if not (MIN_BOARD <= width <= MAX_BOARD and MIN_BOARD <= height <= MAX_BOARD):
        return False
    offset = player_offset(pyboy)
    if not 0 < offset < PLANE_BYTES:
        return False
    _, box, _ = read_planes(pyboy)
    return any(box)


SETTLE_MAX_TICKS = 240           # four seconds; a single step animates for sixteen frames
SETTLE_STABLE_TICKS = 3          # frames the board and the sprites must both hold still
SETTLE_MIN_TICKS = 22            # ... but never call it settled before the frame the d-pad
                                 # would have repeated on. The slide runs sixteen frames and
                                 # the repeat fires on the twentieth; a board that stops
                                 # changing earlier than that has either paused mid-slide or
                                 # is about to move again, and both read as "settled".
PRESS_TICKS = 10                 # frames a d-pad press is held. A fallback only: `calibrate`
                                 # measures the real window off the cartridge.
PROBE_MAX_HOLD = 40              # longest hold `calibrate` will try. The cartridge repeats
                                 # the d-pad at frame 20, so this is twice what it needs.
BOOT_MAX_TICKS = 4000            # the story cutscene alone runs about 970 frames
BOOT_PRESS_TICKS = 4
BOOT_STEP_TICKS = 20

#: Frames the emulator is run after a rewind and before the button is pressed. See
#: `Boxxle2GBAction.__press__`; two is what the cartridge needs, and one is not enough.
LEAD_IN_TICKS = 2

DIRECTIONS = {"left": (0, -1), "up": (-1, 0), "down": (1, 0), "right": (0, 1)}

#: Every button worth giving a planner. Boxxle II has no modifier (a direction is the whole
#: move), so the action set is the d-pad and nothing else. A (undo) and START (pause) both
#: exist on the cartridge and are deliberately absent: undo would let search escape a
#: deadlock the game itself cannot escape, which is precisely the thing being planned around.
action_cost_map = {"left": 1, "right": 1, "up": 1, "down": 1, "nop": 0}

action_list = [f"{button},{PRESS_TICKS}" for button in DIRECTIONS]


def button_actions(calibration=None):
    """The primitive actions, each held for however long calibration settled on."""
    ticks = (calibration or Calibration(PRESS_TICKS, None)).press_ticks
    return [f"{button},{ticks}" for button in DIRECTIONS]


def settle(pyboy, render=False, max_ticks=SETTLE_MAX_TICKS, stable_ticks=SETTLE_STABLE_TICKS,
           min_ticks=SETTLE_MIN_TICKS):
    """Run the emulator until the game stops moving, and report whether it did.

    The plane buffers are updated the frame the button is read, so watching them alone would
    call a move finished sixteen frames before the game will accept another one, and the
    dropped presses that causes look exactly like a planner's action having no effect. What
    actually takes those sixteen frames is the player sliding one pixel per frame, and that
    slide is written to the shadow OAM at `$C000` and to nothing else in work RAM: no counter,
    no flag, no direction byte moves while it runs. So the settle predicate is the planes
    *and* the sprite buffer holding still together.

    Stability alone is still not enough, which cost a long afternoon: the slide pauses for a
    frame or two partway through, and, worse, a hold long enough to trip the d-pad's
    auto-repeat looks settled in the gap between the first move and the second. So
    `min_ticks` holds the settle open until the frame the repeat would have fired on. It costs
    nothing: the slide plus its stable frames already take about that long.

    The early exit is not an optimisation. Once every box is home the cartridge starts its
    congratulation sequence and redraws the plane buffers with something that is not a
    Sokoban position, so a solved board has to be snapshotted while it still exists.
    """
    previous, stable = None, 0
    for elapsed in range(1, max_ticks + 1):
        pyboy.tick(1, render)
        if is_solved(read_board(pyboy)):
            return True
        current = (player_offset(pyboy), read_planes(pyboy), shadow_oam(pyboy))
        if current == previous:
            stable += 1
            if stable >= stable_ticks and elapsed >= min_ticks:
                return True
        else:
            previous, stable = current, 0
    return False


def _force_level(context):
    """Hook body: pin `$C162` and `$C352` on the way into `LoadLevelHeader`.

    Writing them from outside on a frame boundary is not enough: the menu resets both and
    calls the loader within the same frame, and the loader wins. Hooking its entry puts the
    write between the reset and the read at `$0F5D`/`$0F6E`. The demo flag is cleared in the
    same breath: the attract-mode board at `$4F08` is not one of the 120 and would be loaded
    over the top of whichever one was asked for.
    """
    pyboy, index = context
    pyboy.memory[STAGE_ADDR] = index[0] // LEVELS_PER_STAGE
    pyboy.memory[LEVEL_IN_STAGE_ADDR] = index[0] % LEVELS_PER_STAGE
    pyboy.memory[DEMO_LEVEL_ADDR] = 0


def boot(pyboy, render=False, max_ticks=BOOT_MAX_TICKS):
    """Get from power-on to a loaded board. True if one appeared.

    The cartridge's front end is three screens deep (title, then a music choice, then
    PLAY/PASSKEY/CREATE), and START advances all three, so this presses START whenever
    `$C34E` says one of them is up and waits everywhere else. Waiting is most of the wall
    clock: the story cutscene between the menu and stage 1 runs about 970 frames and no
    button shortens it.

    Tapping blindly would be simpler and is wrong. START during play opens the pause overlay,
    so the one screen where a stray press does damage is the one being aimed at, and the
    check has to be on the state rather than on the clock.
    """
    elapsed = 0
    while elapsed < max_ticks:
        if level_is_loaded(pyboy):
            return True
        if pyboy.memory[GAME_STATE_ADDR] in STATE_ADVANCED_BY_START:
            pyboy.button("start", BOOT_PRESS_TICKS)
            pyboy.tick(BOOT_STEP_TICKS, render)
            elapsed += BOOT_STEP_TICKS
        else:
            pyboy.tick(BOOT_PRESS_TICKS, render)
            elapsed += BOOT_PRESS_TICKS
    return level_is_loaded(pyboy)


# --------------------------------------------------------------------------- probing

def _press(pyboy, state, button, hold, render=False, **settle_kwargs):
    """Rewind to `state`, hold `button` for `hold` frames, and read back the settled board."""
    load_state(pyboy, state.gb_state, render)
    pyboy.tick(LEAD_IN_TICKS, render)
    pyboy.button(button, hold)
    pyboy.tick(hold + 1, render)
    settle(pyboy, render, **settle_kwargs)
    return player_offset(pyboy)


def open_direction(grid, player):
    """A direction with two clear cells of walkable room ahead, or None.

    Read off the board rather than found by pressing. Two cells is the least that can tell
    "moved once" from "repeated", and a probe with only one cell of room reports every hold
    as a single move, which is how a cartridge whose d-pad repeats on frame 20 came back
    claiming a window of `(1, 40)`. Where the board offers no such direction, nothing here can
    be measured and `calibrate` says so rather than inventing a number.
    """
    height, width = len(grid), len(grid[0]) if grid else 0
    for direction, (row_step, col_step) in DIRECTIONS.items():
        ahead = [(player.row + row_step * n, player.col + col_step * n) for n in (1, 2)]
        if all(0 <= row < height and 0 <= col < width and grid[row][col] in (FLOOR, GOAL)
               for row, col in ahead):
            return direction
    return None


def _cells_moved(before, after, step):
    """How far the player travelled along `step`, or 0 if it went somewhere else entirely."""
    delta = (after.row - before.row, after.col - before.col)
    if step[0]:
        return delta[0] // step[0] if delta[1] == 0 else 0
    return delta[1] // step[1] if delta[0] == 0 else 0


def measure_hold_window(pyboy, state, render=False, max_hold=PROBE_MAX_HOLD, **settle_kwargs):
    """The closed range of hold lengths that move the player exactly one cell.

    The lower end is where a press starts registering at all; the upper end is one frame short
    of the d-pad's auto-repeat, past which one action walks two cells. In a Sokoban that
    is not a longer plan, it is a box pushed somewhere nobody asked for. Neither bound is in
    the memory map, so both are measured here.

    Both ends move by two or three frames depending on which board it is measured on, because
    the phase between a save-state and the frame the main loop next samples the pad is not
    fixed: `Boxxle II (USA, Europe)` reports anything from `(1, 18)` to `(3, 24)`. That jitter
    is why `calibrate` takes the *middle* of the window rather than anything near an end, and
    why the number to distrust is a window as wide as the probe; see `open_direction`.

    Returns `(low, high)`, or None if the board offers nowhere to measure it.
    """
    direction = open_direction(state.grid, state.player)
    if direction is None:
        return None
    step = DIRECTIONS[direction]
    low = None
    for hold in range(1, max_hold + 1):
        offset = _press(pyboy, state, direction, hold, render, **settle_kwargs)
        moved = _cells_moved(state.player, offset_to_position(offset), step)
        if moved == 1 and low is None:
            low = hold
        elif moved > 1 and low is not None:
            return (low, hold - 1)
    return None if low is None else (low, max_hold)


def calibrate(pyboy, state, render=False, max_hold=PROBE_MAX_HOLD, **settle_kwargs):
    """Measure how this cartridge wants to be driven. A property of the game, so once is enough.

    Costs on the order of a few dozen presses, each rewound to `state` first, so nothing it
    does survives.
    """
    window = measure_hold_window(pyboy, state, render, max_hold, **settle_kwargs)
    if window is None:
        # Nowhere on this board to measure it: the keeper starts boxed in. Rather than drive
        # the game with a made-up hold, fall back to the documented default and say the
        # window is unknown.
        return Calibration(PRESS_TICKS, None)
    low, high = window
    # Middle of the window: far enough above `low` to survive a frame of jitter in when the
    # game samples the pad, far enough below `high` not to trip auto-repeat.
    return Calibration((low + high) // 2, window)


# ----------------------------------------------------------------------------- state

class Boxxle2GBState(GBState):
    """A settled position: the emulator save-state plus the board read out of work RAM."""

    def __init__(self, pyboy, depth):
        super().__init__(pyboy, depth)
        self.__update__(pyboy)

    def __update__(self, pyboy):
        self.width, self.height = board_shape(pyboy)
        self.grid = read_board(pyboy)
        self.player = offset_to_position(player_offset(pyboy))
        self.boxes = boxes(self.grid)
        self.goals = goals(self.grid)
        self.boxes_home = boxes_home(self.grid)

        # Sampled here, while the emulator still holds this position. Read later from live
        # memory they would describe whichever state was applied last.
        self.solved = is_solved(self.grid)
        self.stuck = stuck_boxes(self.grid)

        predicates = [f"at(player, {self.player.row}, {self.player.col})",
                      f"boxes-home({self.boxes_home})"]
        predicates += [f"at(box, {box.row}, {box.col})" for box in self.boxes]
        predicates += [f"goal(cell, {goal.row}, {goal.col})" for goal in self.goals]
        if self.solved:
            predicates.append("goal-reached")
        if self.stuck:
            predicates.append("terminal-state")
        self.literals = frozenset(predicates)

    def is_consistent(self):
        """Whether the cartridge's own Sokoban invariant holds on this board.

        Every level ships with as many goals as boxes. A position where they disagree means
        one of the three plane addresses has drifted, or the board was read while the clear
        sequence was rewriting it.
        """
        return len(self.boxes) == len(self.goals)

    def __eq__(self, other):
        # The position is the board and the player; depth and history are not part of it, so
        # a state reached two ways compares equal and search can close.
        return (isinstance(other, Boxxle2GBState)
                and self.grid == other.grid and self.player == other.player)

    def __hash__(self):
        return hash((self.grid, self.player))

    def __str__(self):
        return render_grid(self.grid)

    def __repr__(self):
        return (f"<Boxxle2GBState(depth={self.depth}, home={self.boxes_home}"
                f"/{len(self.boxes)}, player=({self.player.row}, {self.player.col}))>")


# ---------------------------------------------------------------------------- actions

class Boxxle2GBAction(GBAction):
    """A d-pad press held for a number of frames, spelled `"button,ticks"`."""

    cost_map = action_cost_map

    def __press__(self, pyboy, render):
        """Let the machine run a couple of frames after the rewind, then press.

        `apply` rewinds to the parent state before pressing, and `load_state` ticks exactly
        one frame afterwards. On some states that one frame is not enough: the press lands
        before the main loop next samples the pad, `ReadJoypad` never sees the edge, and the
        move is silently dropped: the successor comes back equal to its parent and
        `successors` deletes the action as a no-op. It is not every state, which is the worst
        kind of bug to have: replaying a 500-move plan on the cartridge failed at move 19 and
        the same move applied on its own worked.

        Two idle frames is what it takes; the constant is here rather than in the shared
        `GBAction` because the other cartridges do not need it and paying for it everywhere
        would be a tax on them.
        """
        pyboy.tick(LEAD_IN_TICKS, render)
        super().__press__(pyboy, render)

    def __settle__(self, pyboy, render, **settle_kwargs):
        return settle(pyboy, render, **settle_kwargs)

    def __next_state__(self, pyboy, state):
        return Boxxle2GBState(pyboy, state.depth + 1)


# ------------------------------------------------------------------------ environment

class Boxxle2GBEnv(GBEnv):
    """Boxxle II, played on the cartridge.

    The ROM is copyrighted and is not distributed with this repo; pass the path to your own
    dump. `fix_index` selects which of the 120 levels the cartridge's loader will build.
    """

    rom_md5 = ROM_MD5
    rom_name = ROM_NAME
    action_class = Boxxle2GBAction

    def __init__(self, romfile, render=False, verify_rom=True, calibrate=True,
                 settle_max_ticks=SETTLE_MAX_TICKS, settle_stable_ticks=SETTLE_STABLE_TICKS,
                 boot_max_ticks=BOOT_MAX_TICKS):
        self.romfile = romfile
        # Not `self.render`: that name belongs to the method which prints the history.
        self.render_window = render
        self.pyboy = None
        self.level_index = [0]           # a one-element list so the loader hook can see edits
        self.state = None
        self.state_history = []
        self.should_calibrate = calibrate
        self.calibration = None
        self.actions = action_list
        self.settle_kwargs = {"max_ticks": settle_max_ticks, "stable_ticks": settle_stable_ticks}
        self.boot_max_ticks = boot_max_ticks
        if verify_rom:
            self.__verify_rom__()

    def levels(self):
        """The 120 boards, decoded out of this cartridge. Needs no emulator."""
        return read_levels(self.romfile)

    def fix_index(self, index):
        """Select the level, zero-based: `fix_index(10)` is the cartridge's level 2-1."""
        assert 0 <= index < LEVEL_COUNT, \
            f"Invalid index: this cartridge has {LEVEL_COUNT} levels, so 0..{LEVEL_COUNT - 1}"
        self.level_index[0] = index

    def label_for(self, index=None):
        """How the cartridge itself numbers a level, as `"stage-level"` counting from one."""
        index = self.level_index[0] if index is None else index
        return f"{index // LEVELS_PER_STAGE + 1}-{index % LEVELS_PER_STAGE + 1:02d}"

    def reset(self):
        self.__restart_emulator__()
        # The hook has to be in place before the first press: the front end reaches
        # `LoadLevelHeader` inside the same frame that it resets the stage counters.
        self.pyboy.hook_register(0, LOAD_LEVEL_HEADER, _force_level,
                                 (self.pyboy, self.level_index))
        if not boot(self.pyboy, self.render_window, self.boot_max_ticks):
            raise RuntimeError(
                f"no level was loaded within {self.boot_max_ticks} frames of booting "
                f"{self.romfile}. Check the ROM is {ROM_NAME} (MD5 {ROM_MD5}).")
        settle(self.pyboy, self.render_window, **self.settle_kwargs)
        self.state = Boxxle2GBState(self.pyboy, 0)
        if self.should_calibrate and self.calibration is None:
            # A property of the cartridge, not of the level, so once is enough.
            self.calibration = calibrate(self.pyboy, self.state, self.render_window,
                                         **self.settle_kwargs)
            self.actions = button_actions(self.calibration)
            load_state(self.pyboy, self.state.gb_state, self.render_window)
        self.state_history = [self.state]
        return self.state, {"level_index": self.level_index[0],
                            "level": self.label_for(),
                            "stage": self.pyboy.memory[STAGE_ADDR],
                            "level_in_stage": self.pyboy.memory[LEVEL_IN_STAGE_ADDR],
                            "size": (self.state.width, self.state.height),
                            "boxes": len(self.state.boxes),
                            "calibration": self.calibration}

    def is_goal(self, state):
        return state.solved

    def is_terminal(self, state):
        """A box wedged in a corner off a goal: nothing anyone presses will move it again.

        Sound but not complete, and deliberately so; see `stuck_boxes`. Absorbing, because
        there is nothing left to plan for; the win side of `__advance__`'s absorbing rule
        matters more here, since a cleared board is overwritten by the cartridge's
        congratulation sequence within a couple of seconds.
        """
        return bool(state.stuck)

    def __score__(self, state):
        return state.boxes_home


def _report(romfile, index=None, render=False):
    """Print what this cartridge wants, and the level it was asked about."""
    env = Boxxle2GBEnv(romfile, render=render)
    try:
        if index is not None:
            env.fix_index(index)
        state, info = env.reset()
        print(f"level {info['level']} (index {info['level_index']}): "
              f"{info['size'][0]}x{info['size'][1]}, {info['boxes']} boxes\n")
        print(state)
        print()
        calibration = info["calibration"]
        if calibration.hold_window is None:
            print("  hold window  not measurable — the player never moved")
        else:
            print(f"  hold window  {calibration.hold_window}  -> press_ticks "
                  f"{calibration.press_ticks}")
        print(f"\n  button actions: {button_actions(calibration)}")
    finally:
        env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure how a Boxxle II cartridge wants to be driven, and show a level "
                    "as the console has it in work RAM.")
    parser.add_argument("rom", help=f"path to {ROM_NAME}")
    parser.add_argument("--level", type=int, default=None, help="level index, 0-119")
    parser.add_argument("--render", action="store_true", help="open an SDL2 window")
    parser.add_argument("--dump", action="store_true",
                        help="decode all 120 levels out of the ROM and print them")
    args = parser.parse_args()
    if args.dump:
        for index, rows in enumerate(read_levels(args.rom)):
            print(f"--- {index:3d} level "
                  f"{index // LEVELS_PER_STAGE + 1}-{index % LEVELS_PER_STAGE + 1:02d}")
            for row in rows:
                print(f"  |{row}|")
    else:
        _report(args.rom, args.level, args.render)
