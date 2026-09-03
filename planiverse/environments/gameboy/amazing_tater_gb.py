"""Amazing Tater on a Game Boy, driven through PyBoy.

The sibling module `gameboy_py/amazing_tater.py` re-implements the game in Python over the
same 105 rooms. This one plays the real cartridge inside an emulator and reads the game's own
board out of work RAM, so the transition function is the cartridge's code rather than a
reconstruction of its rules. States carry an emulator save-state, which is what lets search
branch: applying an action rewinds the machine to the parent first.

The addresses below come from a reverse-engineering pass over `Amazing Tater (U).gb`
(MD5 `53b746bff74c50cd3ebcf41161c66cf3`) and are documented in
`docs/environments/amazing-tater-gb-memory-map.md`. They are revision-specific, which is why
`AmazingTaterGBEnv` checks the ROM's MD5 and warns when it differs.

    env = AmazingTaterGBEnv("Amazing Tater (U).gb")
    env.set_index(0)
    state, info = env.reset()
    print(state)
    for action, successor in env.successors(state):
        ...

Four things about this cartridge shape the module.

**The board is already composed in RAM.** `LoadLevel` at `$08C0` builds a 20x18 map of cell
codes at `$C2F2`, one byte per cell, and the whole game runs on it: walls, pits, blocks,
turnstiles, the taters and the exit flag are all there, and each code carries the shape
information a planner needs: a block square says which of its neighbours belong to the same
block, a turnstile pivot says which arms it has. So the environment reads the position
directly rather than inferring it from tiles or sprites, and `is_goal` is exact.

**The level is chosen by a register, not by a variable.** `LoadLevel` is entered with `HL`
holding twice the level index, computed by its caller from a stage and a level-within-stage
pair. Rather than reverse the stage arithmetic and hope it stays put, `reset` hooks the
loader's entry and writes `HL` itself, which selects any of a set's levels directly.

**A tater that reaches the flag leaves the board.** `$C2AD` counts the taters still to get
home and drops as each one arrives; the room is solved when it reaches zero. That is the goal
test, and it is the cartridge's own.

**The sprite never stops moving.** The tater bobs on the spot forever, so a settle predicate
that watches the shadow OAM the way `puzznic_gb` and `flipull_gb` do would never fire here.
This one watches the board buffer alone, which is sound because everything a move changes is
in it, and because the cartridge takes the next press about six frames after the last one,
long before the settle window closes.
"""
from collections import namedtuple

from planiverse.environments.gameboy.gb import (
    GBAction, GBEnv, GBState, load_state, save_state, sprites as _oam_sprites,
)
from planiverse.environments.gameboy_py.amazing_tater import (
    ARM_GLYPHS, ARM_OVER_PIT_GLYPHS, BLOCK_GLYPHS, EXIT, FLOOR, OUTSIDE, PIT, PIVOT,
    SETTLED_GLYPHS, TATER_GLYPHS, WALL, friendly,
)

# --------------------------------------------------------------------------- the ROM

ROM_MD5 = "53b746bff74c50cd3ebcf41161c66cf3"
ROM_NAME = "Amazing Tater (U).gb"
ROM_BYTES = 65536                # MBC1: bank 0 fixed, banks 1-3 switched into $4000-$7FFF

# ------------------------------------------------------------------- memory map §5-6
# Every address here is in `docs/environments/amazing-tater-gb-memory-map.md`, with how it
# was established.

BOARD_ADDR = 0xC2F2              # the composed board: one byte per cell
BOARD_BYTES = 360                # 20 columns x 18 rows
ROW_STRIDE = 20                  # *not* the hardware tilemap's 32
SCREEN_COLS, SCREEN_ROWS = 20, 18

BOARD_WIDTH_ADDR = 0xC2BD        # the room's width plus its border, so W + 2
BOARD_HEIGHT_ADDR = 0xC2BE       # ... and H + 2
BOARD_X_OFFSET_ADDR = 0xC2BF     # where the bordered box sits in the 20x18 buffer
BOARD_Y_OFFSET_ADDR = 0xC2C0

GAME_MODE_ADDR = 0xC131          # 0 PUZZLE, 1/2 PRACTICE, 3 BEGINNER and ACTION
OVERLAY_ADDR = 0xC2AC            # 0 while the room is being played; non-zero under a menu
TATERS_ON_BOARD_ADDR = 0xC2AD    # bit per character still to reach the flag: 1 is tater 1,
                                 # 8 is tater 4. Zero means the room is solved.
ACTIVE_TATER_ADDR = 0xC2AE       # which one SELECT has the controls on
MENU_CURSOR_ADDR = 0xD37D        # SELECT MODE highlight, 0-3; $80 anywhere else
ROM_BANK_ADDR = 0xC2D3           # shadow of the $2000 write

OAM_BUFFER_ADDR = 0xC000         # OAM DMA source: 40 sprites of 4 bytes

LOAD_LEVEL = 0x08C0              # bank 0; entered with HL = level index * 2

#: The set descriptor table, nine words: three per level set, holding the base of that set's
#: record, plane A and plane B pointer arrays. Read so the environment can state each set's
#: size from the cartridge rather than from a constant that could drift.
SET_DESCRIPTORS = 0x0C05
SET_COUNTS = (41, 96, 64)        # set A, set B, set C
LEVEL_DATA_BANK = 3

# --------------------------------------------------------------------- the level sets
# Three sets on the cartridge; two of them are rooms. Set B, behind PRACTICE MODE, is a timed
# climb through ten floors whose board buffer holds the corridors of the neighbouring floors
# as well as the room, and whose tater starts outside the room (a different game, not a
# different level), so this environment does not offer it. The index `set_index` takes runs
# over set A and then set C, which is also the order `amazing_tater.LEVELS` is in.

#: `(letter, the game mode that reaches the set, how many rooms)`, in index order.
LEVEL_SETS = (("A", 0, 41), ("C", 3, 64))
LEVEL_COUNT = sum(size for _letter, _mode, size in LEVEL_SETS)

#: How far down SELECT MODE each game mode sits. The menu reads BEGINNER, PUZZLE, PRACTICE,
#: ACTION, and BEGINNER and ACTION both land on set C; this environment takes BEGINNER,
#: which is the entry the cursor already starts on.
MODE_MENU_ROW = {0: 1, 1: 2, 3: 0}

# ------------------------------------------------------------------- memory map §4
# The cell codes. Every one was matched against the rendered screen and, for the shapes,
# against the cartridge's own 15-entry shape table at $0BF6.

CODE_FLOOR = 0x00
CODE_PIT = 0xE0
CODE_OUTSIDE = 0xFF
BLOCK_CODES = range(0x40, 0x50)          # 0x40 | mask of which neighbours share the block
SETTLED_BLOCK_CODES = range(0x50, 0x60)  # the same, for a square that is over a pit
ARM_CODES = range(0x80, 0x84)            # 0x80 | direction from the pivot: up right down left
ARM_OVER_PIT_CODES = range(0x90, 0x94)
PIVOT_CODES = range(0xA0, 0xAF)          # 0xA0 | index into the shape table at $0BF6
TATER_CODES = range(0xC0, 0xC4)          # 0xC0 | which of the four characters
EXIT_CODES = range(0xD0, 0xD1)           # every room on the cartridge has exactly one flag
WALL_CODES = range(0xF0, 0xFF)           # fifteen wall graphics; all of them are just wall

#: The 15-byte table at `$0BF6`. The high nibble of entry `i` is the arm mask of the pivot
#: whose code is `$A0 + i`, with bit 8 up, 4 right, 2 down and 1 left. This is the one piece
#: of ROM the board decoder needs, and it is checked against the cartridge in the tests.
SHAPE_TABLE_ADDR = 0x0BF6
SHAPE_TABLE = (0x80, 0x41, 0x22, 0x13, 0xCE, 0x6D, 0x34, 0x95,
               0xE6, 0x77, 0xB8, 0xD9, 0xFA, 0xAB, 0x5C)
ARM_MASKS = tuple(entry >> 4 for entry in SHAPE_TABLE)

# --------------------------------------------------------------------------- alphabet
# The glyphs a decoded board is written in are the twin's, imported rather than repeated.
# `gameboy_py.amazing_tater` is pure Python with no dependencies, so importing it here costs
# nothing, and a cell code table that existed in two places would eventually mean two
# different games. The direction is the one that makes sense: the twin names the glyphs, this
# module knows which of the cartridge's codes each one stands for.

Position = namedtuple("Position", ["row", "col"])

#: What `calibrate` learned from the cartridge: the hold to use, and the closed range of
#: holds that move a tater exactly one cell. Past the upper end the d-pad repeats and one
#: action walks two cells, which in a puzzle with irreversible pushes is not a longer plan
#: but a different one.
Calibration = namedtuple("Calibration", ["press_ticks", "hold_window"])


# ------------------------------------------------------------------- reading the ROM
# Pure functions of the cartridge image, so they need neither an emulator nor a boot.

def read_rom(romfile):
    with open(romfile, "rb") as handle:
        return handle.read()


def bank_view(rom, bank):
    """Bank 0 plus one switched bank, as the CPU sees `$0000-$7FFF`."""
    return rom[:0x4000] + rom[bank * 0x4000:(bank + 1) * 0x4000]


def set_descriptors(rom):
    """The nine words at `$0C05`, as three `(records, plane_a, plane_b)` triples.

    The array bases are what tells the loader where each set's levels live, and the gaps
    between them are what `level_counts` turns back into level counts.
    """
    words = [rom[SET_DESCRIPTORS + 2 * i] | (rom[SET_DESCRIPTORS + 2 * i + 1] << 8)
             for i in range(9)]
    return tuple(tuple(words[3 * s:3 * s + 3]) for s in range(3))


def level_counts(rom):
    """How many levels each set holds, derived from its own pointer arrays.

    Each set stores three parallel word arrays of N entries back to back, so the distance
    from the record-pointer array to the plane-A array is `2N`. Reading N out rather than
    hard-coding it is what makes `SET_COUNTS` a checkable claim instead of a comment.
    """
    return tuple((planes_a - records) // 2 for records, planes_a, _ in set_descriptors(rom))


def shape_table(rom):
    """The 15 turnstile shapes at `$0BF6`, straight out of the cartridge."""
    return tuple(rom[SHAPE_TABLE_ADDR:SHAPE_TABLE_ADDR + 15])


# ------------------------------------------------------------------- pure decoding
# Split out from the emulator so they can be tested against synthetic RAM.

def _glyph_table():
    """Every cell code the cartridge writes, as the glyph the twin's levels are written in.

    Built once from the twin's glyph strings, so the two modules cannot drift apart. The
    fifteen wall graphics all collapse to `#` and the fifteen turnstile shapes all collapse
    to `@`: neither distinction changes what a tater can do, and a pivot's arms are written
    out beside it anyway.
    """
    table = {CODE_OUTSIDE: OUTSIDE, CODE_FLOOR: FLOOR, CODE_PIT: PIT}
    for code in WALL_CODES:
        table[code] = WALL
    for code in PIVOT_CODES:
        table[code] = PIVOT
    for code in EXIT_CODES:
        table[code] = EXIT
    for glyphs, codes in ((BLOCK_GLYPHS, BLOCK_CODES), (SETTLED_GLYPHS, SETTLED_BLOCK_CODES),
                          (ARM_GLYPHS, ARM_CODES), (ARM_OVER_PIT_GLYPHS, ARM_OVER_PIT_CODES),
                          (TATER_GLYPHS, TATER_CODES)):
        for offset, glyph in enumerate(glyphs):
            table[codes.start + offset] = glyph
    return table


GLYPH_BY_CODE = _glyph_table()


def cell_glyph(code):
    """One of the cartridge's cell codes as the glyph the twin writes its levels in."""
    try:
        return GLYPH_BY_CODE[code]
    except KeyError:
        raise ValueError(f"unknown Amazing Tater cell code ${code:02X}") from None


def board_bounds(buffer):
    """The rectangle of the 20x18 buffer the room occupies, as `(top, left, bottom, right)`.

    Cropping to the cells that are not `$FF` rather than to `$C2BD`/`$C2BE` is deliberate:
    the two agree on every room the cartridge ships (the tests check that), and the crop
    keeps working on a buffer read before the dimension bytes have been written.
    """
    live = [(index // ROW_STRIDE, index % ROW_STRIDE)
            for index, code in enumerate(buffer) if code != CODE_OUTSIDE]
    if not live:
        return None
    rows = [row for row, _ in live]
    cols = [col for _, col in live]
    return min(rows), min(cols), max(rows), max(cols)


def decode_board(buffer):
    """The composed board as a tuple of glyph rows, cropped to the room.

    This is the shape `amazing_tater.Level` reads, which is what makes a board dumped here
    and a level stored there the same object.
    """
    bounds = board_bounds(buffer)
    if bounds is None:
        return ()
    top, left, bottom, right = bounds
    return tuple("".join(cell_glyph(buffer[row * ROW_STRIDE + col])
                         for col in range(left, right + 1)).rstrip()
                 for row in range(top, bottom + 1))


def find(buffer, codes):
    """Where the cells with one of `codes` are, as positions in the uncropped buffer."""
    return tuple(Position(index // ROW_STRIDE, index % ROW_STRIDE)
                 for index, code in enumerate(buffer) if code in codes)


def pivots(buffer):
    """Every turnstile, as `(pivot position, arm mask)`.

    The mask is the twin's: bit 8 up, 4 right, 2 down, 1 left, read straight out of
    `ARM_MASKS`. Turning a turnstile rotates its arms and so changes this mask, which makes
    it *state* rather than scenery -- and a state a planner has to be told about, because
    which arms are where decides what a tater can walk through.
    """
    return tuple((Position(index // ROW_STRIDE, index % ROW_STRIDE),
                  ARM_MASKS[code - PIVOT_CODES.start])
                 for index, code in enumerate(buffer) if code in PIVOT_CODES)


def taters(buffer):
    """The taters still on the board, as `{character: position}`."""
    return {code - TATER_CODES.start: Position(index // ROW_STRIDE, index % ROW_STRIDE)
            for index, code in enumerate(buffer) if code in TATER_CODES}


def is_solved(pyboy):
    """Every tater home, read off `$C2AD` rather than off the board.

    The board cannot answer this. A tater in mid-step is *taken off* the board map and drawn
    as a sprite until it arrives, so for a dozen frames after every single press the board
    shows no tater at all, which is indistinguishable from a solved room if that is all you
    look at. `$C2AD` keeps the character's bit set for the whole step, and only clears it when
    the tater reaches the flag.
    """
    return taters_on_board(pyboy) == 0


def render_board(rows):
    """The friendly view: `$` for every block square, `+` for every arm, `o` for a pivot.

    The same one `str(amazing_tater.AmazingTaterState)` prints, so a position looks the same
    whichever of the two is holding it. `state.rows` is the exact view, and that is what the
    tests compare, because collapsing the block letters is exactly the information that tells
    two blocks apart.
    """
    return "\n".join(friendly(rows))


# ------------------------------------------------------------------------- emulation
# `create_pyboy`, `save_state` and `load_state` come from the shared `gb` module.

def read_buffer(pyboy):
    """The composed board at `$C2F2`, as a 360-byte buffer."""
    return bytes(pyboy.memory[BOARD_ADDR:BOARD_ADDR + BOARD_BYTES])


def read_board(pyboy):
    return decode_board(read_buffer(pyboy))


def board_shape(pyboy):
    """The room's own width and height, without its border."""
    return pyboy.memory[BOARD_WIDTH_ADDR] - 2, pyboy.memory[BOARD_HEIGHT_ADDR] - 2


def taters_on_board(pyboy):
    """`$C2AD`: one bit per character still to reach the flag.

    A mask and not a count, which took a four-tater room to notice: three of the six rooms
    with more than one tater read 15 there, and a room with the first and fourth characters
    reads 9. The board says the same thing, and `AmazingTaterGBState.is_consistent` checks
    that the two agree.
    """
    return pyboy.memory[TATERS_ON_BOARD_ADDR]


def taters_left(pyboy):
    return bin(taters_on_board(pyboy)).count("1")


def active_tater(pyboy):
    return pyboy.memory[ACTIVE_TATER_ADDR]


def menu_cursor(pyboy):
    """Which SELECT MODE entry is highlighted, or None when that menu is not up."""
    cursor = pyboy.memory[MENU_CURSOR_ADDR]
    return cursor if cursor < len(MODE_MENU_ROW) + 1 else None


def sprites(pyboy):
    """The OAM DMA buffer as `(y, x, tile)` for every visible sprite."""
    return _oam_sprites(pyboy, OAM_BUFFER_ADDR, visible_only=True)


#: Room sizes the cartridge actually ships, off the two sets this environment offers: 7x3 at
#: the smallest, 18x16 at the largest. `level_is_loaded` uses them to tell a composed board
#: from whatever `$C2BD` and `$C2BE` happen to hold on a menu screen.
MIN_ROOM, MAX_ROOM = 3, 18


def level_is_loaded(pyboy):
    """True once a playable room sits in the board buffer.

    Five things have to agree, because none of them alone is enough: the dimension bytes keep
    whatever the last room left in them, `$C2AD` is zero both before a room loads and after
    one is solved, and the buffer holds the previous room until the next one overwrites it.
    """
    if pyboy.memory[OVERLAY_ADDR] != 0:
        return False
    width, height = board_shape(pyboy)
    if not (MIN_ROOM <= width <= MAX_ROOM and MIN_ROOM <= height <= MAX_ROOM):
        return False
    if taters_on_board(pyboy) == 0:
        return False
    return bool(find(read_buffer(pyboy), EXIT_CODES))


SETTLE_MAX_TICKS = 180           # a move's board writes span sixteen frames at the longest
SETTLE_STABLE_TICKS = 20         # measured: the longest still spell inside an unfinished
                                 # move is eight frames, when a block is dissolving
PRESS_TICKS = 5                  # frames a d-pad press is held. A fallback only: `calibrate`
                                 # measures the real window off the cartridge.
SWITCH_LOCKOUT_TICKS = 48        # measured: after SELECT the cartridge ignores the next press
                                 # for 33 frames, and nothing on the board moves to say so
PROBE_MAX_HOLD = 24              # longest hold `calibrate` will try; the pad repeats near 12
BOOT_MAX_TICKS = 6000            # the title screen alone runs about 400 frames, and BEGINNER
                                 # MODE opens with a tutor who has several screens to say
BOOT_PRESS_TICKS = 5
BOOT_STEP_TICKS = 30
INTRO_MAX_TICKS = 300            # how long `wait_until_interactive` will keep probing
INTRO_STEP_TICKS = 20
INTRO_FALLBACK_TICKS = 120       # twice the measured wait, for a room where no press does
                                 # anything at all, so the probe cannot tell

DIRECTIONS = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}

#: Handing the controls to the next tater. SELECT on the console; free, because it moves
#: nobody; otherwise the cheapest plan for a two-tater room would be measured partly in how
#: often you swapped.
SWITCH = "switch"
SWITCH_BUTTON = "select"

#: Every button worth giving a planner. A opens the pause menu with its RETRY and QUIT, and
#: is deliberately absent: a planner that can retry can escape a block it has sunk into the
#: wrong pit, which is precisely the thing being planned around.
action_cost_map = {"up": 1, "right": 1, "down": 1, "left": 1, SWITCH: 0, "nop": 0}

action_list = [f"{button},{PRESS_TICKS}" for button in list(DIRECTIONS) + [SWITCH]]


def button_actions(calibration=None):
    """The primitive actions, each held for however long calibration settled on."""
    ticks = (calibration or Calibration(PRESS_TICKS, None)).press_ticks
    return [f"{button},{ticks}" for button in list(DIRECTIONS) + [SWITCH]]


def settle(pyboy, render=False, max_ticks=SETTLE_MAX_TICKS, stable_ticks=SETTLE_STABLE_TICKS):
    """Run the emulator until the board stops changing, and report whether it did.

    The shadow OAM is deliberately not part of the predicate. Every other Game Boy
    environment here watches it, because in those games the piece in motion is a sprite and
    work RAM says nothing while it slides. Here the opposite is true twice over: the board
    buffer is written for every part of a move, and the tater's idle bob rewrites the OAM
    every eight frames for as long as the game is running, so a predicate that included it
    would never fire at all.

    Twenty frames of stillness is measured rather than guessed. Nothing observed takes longer
    than sixteen frames of board writes, and the longest still spell *inside* an unfinished
    move is eight, when a block that has been shoved onto pits is dissolving into them.

    The early exit is not an optimisation. Once the last tater is home the cartridge starts
    its own sequence over the top of the room, so a solved board has to be snapshotted while
    it still exists.
    """
    previous, stable = None, 0
    for _ in range(max_ticks):
        pyboy.tick(1, render)
        current = read_buffer(pyboy)
        if is_solved(pyboy):
            return True
        if current == previous:
            stable += 1
            if stable >= stable_ticks:
                return True
        else:
            previous, stable = current, 0
    return False


def _force_level(context):
    """Hook body: write `HL` on the way into `LoadLevel`.

    The loader is entered with `HL` holding twice the level index, worked out by its caller
    from a stage counter and a level-within-stage counter that between them do not cover a
    set evenly: set A's last room is the one left over after four stages of ten. Writing the
    index the loader is about to use sidesteps all of that, and it is the only write: the
    game mode still comes from the menu, so the cartridge stays in a state it put itself in.
    """
    pyboy, index = context
    pyboy.register_file.HL = index[0] * 2


def boot(pyboy, mode, render=False, max_ticks=BOOT_MAX_TICKS):
    """Get from power-on to a loaded room in `mode`. True if one appeared.

    The front end is the title screen, then SELECT MODE, then, depending on the mode chosen,
    either a tutor with several screens of advice or an ENTER PASSWORD grid, and START
    advances all of them. Only one screen needs the d-pad: SELECT MODE, where BEGINNER MODE
    is already under the cursor and PUZZLE MODE is one press down.

    Knowing *which* screen is up is what `$D37D` is for. It holds `$80` while the title is on
    screen and a small number everywhere a cursor exists, so the title screen is what marks
    the boundary: the first small number after it is SELECT MODE's highlight. That matters
    because PUZZLE MODE's password grid reuses the same byte as its own row cursor, and a
    d-pad press aimed at the mode menu but landing there walks the alphabet instead. That
    is exactly how this got written the wrong way round the first time, and why the boot stops
    touching the d-pad the moment the mode is chosen.
    """
    wanted, elapsed = MODE_MENU_ROW[mode], 0
    past_title, chosen = False, wanted == 0
    while elapsed < max_ticks:
        if level_is_loaded(pyboy):
            return True
        cursor = menu_cursor(pyboy)
        if not past_title:
            # `$80` shows only while the title is up, so seeing it once and then seeing a
            # cursor is what says SELECT MODE has replaced it.
            past_title = cursor is None
            pyboy.button("start", BOOT_PRESS_TICKS)
        elif chosen or cursor is None:
            # Either the mode is picked, or the menu has not drawn itself yet, and `chosen`
            # must not be set here. Setting it on the frame between the title going away and
            # the menu appearing is what made the first version of this pick whatever the
            # cursor happened to be resting on.
            pyboy.button("start", BOOT_PRESS_TICKS)
        elif cursor != wanted:
            pyboy.button("down", BOOT_PRESS_TICKS)
        else:
            chosen = True
            pyboy.button("start", BOOT_PRESS_TICKS)
        pyboy.tick(BOOT_STEP_TICKS, render)
        elapsed += BOOT_STEP_TICKS
    return level_is_loaded(pyboy)


def wait_until_interactive(pyboy, render=False, max_ticks=INTRO_MAX_TICKS,
                           step=INTRO_STEP_TICKS, press_ticks=PRESS_TICKS):
    """Advance past the room's intro, and report how many frames it took.

    The loader fills the board buffer before the room has finished announcing itself, so the
    board is fully readable, and settled, while every button is still ignored, for about
    sixty frames. A state snapshotted in that window looks normal and answers no
    action, which is the worst way for this to go wrong: search sees a room with no legal
    moves rather than an error.

    Rather than hard-code the delay, this presses each button from a snapshot at increasing
    offsets until something answers. It then rewinds and replays only the waiting, so the state
    handed back is still the one the loader built. SELECT counts as an answer, and has to:
    room A-14 opens with its first tater walled in on all four sides, and handing the
    controls to the other one is the only thing a player can do there.
    """
    start = save_state(pyboy)

    def responds_after(waited):
        for button in list(DIRECTIONS) + [SWITCH_BUTTON]:
            load_state(pyboy, start, render)
            if waited:
                pyboy.tick(waited, render)
            before = (read_buffer(pyboy), active_tater(pyboy))
            pyboy.button(button, press_ticks)
            pyboy.tick(press_ticks + 2, render)
            settle(pyboy, render)
            if (read_buffer(pyboy), active_tater(pyboy)) != before:
                return True
        return False

    waited = 0
    while waited <= max_ticks:
        if responds_after(waited):
            load_state(pyboy, start, render)
            if waited:
                pyboy.tick(waited, render)
            return waited
        waited += step
    # Nothing answered. Either the room genuinely has no move in it, or the probe is wrong;
    # either way, wait out twice the delay that was measured and hand the room back rather
    # than fail on a room a planner could still legitimately be handed.
    load_state(pyboy, start, render)
    pyboy.tick(INTRO_FALLBACK_TICKS, render)
    return None


# --------------------------------------------------------------------------- probing

def _press(pyboy, state, button, hold, render=False, **settle_kwargs):
    """Rewind to `state`, hold `button` for `hold` frames, and read back the settled board."""
    load_state(pyboy, state.gb_state, render)
    pyboy.button(button, hold)
    pyboy.tick(hold + 1, render)
    settle(pyboy, render, **settle_kwargs)
    return taters(read_buffer(pyboy))


def _cells_moved(before, after, step):
    """How far the tater travelled along `step`, or 0 if it went somewhere else entirely."""
    delta = (after.row - before.row, after.col - before.col)
    if step[0]:
        return delta[0] // step[0] if delta[1] == 0 else 0
    return delta[1] // step[1] if delta[0] == 0 else 0


def _open_direction(pyboy, state, render, settle_kwargs, max_hold):
    """A direction with two clear cells of room, so a repeat has somewhere to show itself.

    Picking the first direction that moves at all is not enough: with a wall one cell away
    every hold looks identical and the window comes back as wide as the probe.
    """
    who = state.active
    fallback = None
    for direction, step in DIRECTIONS.items():
        after = _press(pyboy, state, direction, max_hold, render, **settle_kwargs)
        if who not in after:
            continue                      # walked into the flag; that tells us nothing
        moved = _cells_moved(state.taters[who], after[who], step)
        if moved >= 2:
            return direction
        if moved >= 1 and fallback is None:
            fallback = direction
    return fallback


def measure_hold_window(pyboy, state, render=False, max_hold=PROBE_MAX_HOLD, **settle_kwargs):
    """The closed range of hold lengths that move a tater exactly one cell.

    The lower end is where a press starts registering at all; the upper end is one frame
    short of the d-pad's auto-repeat, past which one action walks two cells. In a game
    where a block shoved into the wrong pit is gone for good, that is not a longer plan but a
    different one. Neither bound is in the memory map, so both are measured here.
    `Amazing Tater (U)` reports `(1, 11)`.

    Returns `(low, high)`, or None if no hold moves a tater exactly one cell.
    """
    direction = _open_direction(pyboy, state, render, settle_kwargs, max_hold)
    if direction is None:
        return None
    who, step = state.active, DIRECTIONS[direction]
    low = None
    for hold in range(1, max_hold + 1):
        after = _press(pyboy, state, direction, hold, render, **settle_kwargs)
        moved = _cells_moved(state.taters[who], after[who], step) if who in after else 0
        if moved == 1 and low is None:
            low = hold
        elif moved > 1 and low is not None:
            return (low, hold - 1)
    return None if low is None else (low, max_hold)


def calibrate(pyboy, state, render=False, max_hold=PROBE_MAX_HOLD, **settle_kwargs):
    """Measure how this cartridge wants to be driven. A property of the game, so once is
    enough.

    Costs a couple of dozen presses, each rewound to `state` first, so nothing it does
    survives.
    """
    window = measure_hold_window(pyboy, state, render, max_hold, **settle_kwargs)
    if window is None:
        # No tater moved. Rather than drive the game with a made-up hold, fall back to the
        # documented default and say the window is unknown.
        return Calibration(PRESS_TICKS, None)
    low, high = window
    # Middle of the window: far enough above `low` to survive a frame of jitter in when the
    # game samples the pad, far enough below `high` not to trip auto-repeat.
    return Calibration((low + high) // 2, window)


# ----------------------------------------------------------------------------- state

class AmazingTaterGBState(GBState):
    """A settled position: the emulator save-state plus the board read out of work RAM."""

    def __init__(self, pyboy, depth, total=None):
        super().__init__(pyboy, depth)
        self.__update__(pyboy, total)

    def __update__(self, pyboy, total):
        buffer = read_buffer(pyboy)
        self.width, self.height = board_shape(pyboy)
        self.rows = decode_board(buffer)
        self.taters = taters(buffer)
        self.active = active_tater(pyboy)
        self.on_board = taters_on_board(pyboy)
        self.taters_left = taters_left(pyboy)
        # How many the room started with, carried down from the position it started in.
        # `$C2AD` only counts the ones still out, and a state cannot say how far along it is
        # without knowing what it is counting down from.
        self.total = self.taters_left if total is None else total
        self.taters_home = self.total - self.taters_left
        exits = find(buffer, EXIT_CODES)
        self.exit = exits[0] if exits else None
        self.block_squares = find(buffer, list(BLOCK_CODES) + list(SETTLED_BLOCK_CODES))
        self.pits = find(buffer, [CODE_PIT])
        self.turnstiles = pivots(buffer)

        # Sampled here, while the emulator still holds this position. Read later from live
        # memory they would describe whichever state was applied last.
        self.solved = is_solved(pyboy)

        predicates = [f"taters-home({self.taters_home})"]
        predicates += [f"at(tater{who + 1}, {cell.row}, {cell.col})"
                       for who, cell in sorted(self.taters.items())]
        predicates += [f"at(block, {cell.row}, {cell.col})" for cell in self.block_squares]
        # Spelled exactly as the twin spells it. Without this a position that differs only
        # in how its turnstiles are turned is invisible to a planner, which reasons over
        # these predicates alone: it closes the two as one and can empty its frontier while
        # calling that a proof there is no plan.
        predicates += [f"turnstile({cell.row}, {cell.col}, {mask})"
                       for cell, mask in self.turnstiles]
        predicates += [f"pit({cell.row}, {cell.col})" for cell in self.pits]
        if self.exit is not None:
            predicates.append(f"exit({self.exit.row}, {self.exit.col})")
        if self.taters:
            predicates.append(f"controlled(tater{self.active + 1})")
        if self.solved:
            predicates.append("goal-reached")
        self.literals = frozenset(predicates)

    def is_consistent(self):
        """Whether the cartridge's own bookkeeping agrees with its own board.

        `$C2AD` carries a bit for every tater still to get home and the board draws one code
        per tater, so the two must name the same characters. A position where they disagree
        means an address has drifted, or the board was read in the middle of a step.
        """
        return sum(1 << who for who in self.taters) == self.on_board

    def __eq__(self, other):
        # The position is the board and who holds the controls; depth and history are not
        # part of it, so a position reached two ways compares equal and search can close.
        return (isinstance(other, AmazingTaterGBState)
                and self.rows == other.rows and self.active == other.active)

    def __hash__(self):
        return hash((self.rows, self.active))

    def __str__(self):
        return render_board(self.rows)

    def __repr__(self):
        return (f"<AmazingTaterGBState(depth={self.depth}, home={self.taters_home}"
                f"/{self.total}, active={self.active})>")


# ---------------------------------------------------------------------------- actions

class AmazingTaterGBAction(GBAction):
    """A press held for a number of frames, spelled `"button,ticks"`.

    `switch` is spelled the way a planner reads it and pressed the way the console does: the
    console's button is SELECT, and nothing else here needs to know that.
    """

    cost_map = action_cost_map

    def __press__(self, pyboy, render):
        ticks = set()
        for button, hold in self.actions_tick_list:
            if button == SWITCH:
                pyboy.button(SWITCH_BUTTON, hold)
            elif button != "nop":
                pyboy.button(button, hold)
            ticks.add(hold)
        pyboy.tick(max(ticks) + 1, render)

    def __settle__(self, pyboy, render, **settle_kwargs):
        """Wait out the move, and, after SELECT, the lockout that follows it.

        Handing the controls over changes nothing on the board, so the ordinary settle
        predicate is satisfied the instant the press is made and the next action arrives
        about twenty-six frames later. The cartridge ignores anything pressed inside
        thirty-three frames of a SELECT, which meant every second switch in a row was
        silently dropped: the plan a planner was building and the game it was building it in
        had different taters under the controls, and the boards only diverged several moves
        afterwards.
        """
        if any(button == SWITCH for button, _ in self.actions_tick_list):
            pyboy.tick(SWITCH_LOCKOUT_TICKS, render)
        return settle(pyboy, render, **settle_kwargs)

    def __next_state__(self, pyboy, state):
        return AmazingTaterGBState(pyboy, state.depth + 1, state.total)


# ------------------------------------------------------------------------ environment

class AmazingTaterGBEnv(GBEnv):
    """Amazing Tater, played on the cartridge.

    The ROM is copyrighted and is not distributed with this repo; pass the path to your own
    dump. `set_index` selects which of the 105 rooms the cartridge's loader will build: 0-40
    are PUZZLE MODE's and 41-104 are BEGINNER and ACTION MODE's, which is the order
    `amazing_tater.LEVELS` is in.
    """

    rom_md5 = ROM_MD5
    rom_name = ROM_NAME
    action_class = AmazingTaterGBAction

    def __init__(self, romfile, render=False, verify_rom=True, calibrate=True,
                 settle_max_ticks=SETTLE_MAX_TICKS, settle_stable_ticks=SETTLE_STABLE_TICKS,
                 boot_max_ticks=BOOT_MAX_TICKS):
        self.romfile = romfile
        # Not `self.render`: that name belongs to the method which prints the history.
        self.render_window = render
        self.pyboy = None
        self.level_index = [0]           # a one-element list so the loader hook can see edits
        self.index = 0
        self.state = None
        self.state_history = []
        self.should_calibrate = calibrate
        self.calibration = None
        self.intro_ticks = None
        self.actions = action_list
        self.settle_kwargs = {"max_ticks": settle_max_ticks,
                              "stable_ticks": settle_stable_ticks}
        self.boot_max_ticks = boot_max_ticks
        if verify_rom:
            self.__verify_rom__()

    # ------------------------------------------------------------------ level choice

    def set_index(self, index):
        """Select the room, zero-based over set A and then set C."""
        assert 0 <= index < LEVEL_COUNT, \
            f"Invalid index: this environment offers {LEVEL_COUNT} rooms, so " \
            f"0..{LEVEL_COUNT - 1}"
        self.index = index
        self.level_index[0] = self.__within_set__(index)[1]

    @staticmethod
    def __within_set__(index):
        """`(letter, index inside that set, the game mode that reaches it)`."""
        start = 0
        for letter, mode, size in LEVEL_SETS:
            if index < start + size:
                return letter, index - start, mode
            start += size
        raise IndexError(index)

    def label_for(self, index=None):
        """How the cartridge's own menus number a room, as `"set-number"` counting from one."""
        letter, within, _mode = self.__within_set__(self.index if index is None else index)
        return f"{letter}-{within + 1:02d}"

    def levels(self):
        """Every room, decoded off this cartridge, in the shape the twin stores them.

        This is where `amazing_tater.LEVELS` came from, and re-running it is how a claim
        about a level can be settled without a screenshot. It boots the emulator once per
        room, so it is slow (a minute or two), and it is the only thing here that is.
        """
        return tuple(self.__dump__(index) for index in range(LEVEL_COUNT))

    def __dump__(self, index):
        """Boot to one room and read its board, leaving the environment's selection alone."""
        chosen = self.index
        try:
            self.set_index(index)
            self.__restart_emulator__()
            self.pyboy.hook_register(0, LOAD_LEVEL, _force_level,
                                     (self.pyboy, self.level_index))
            if not boot(self.pyboy, self.__within_set__(index)[2], self.render_window,
                        self.boot_max_ticks):
                raise RuntimeError(f"no room loaded for index {index}")
            settle(self.pyboy, self.render_window, **self.settle_kwargs)
            return read_board(self.pyboy)
        finally:
            self.set_index(chosen)

    # ------------------------------------------------------------------------- play

    def reset(self):
        self.__restart_emulator__()
        # The hook has to be in place before the first press: the front end reaches
        # `LoadLevel` on its own once the mode is chosen.
        self.pyboy.hook_register(0, LOAD_LEVEL, _force_level, (self.pyboy, self.level_index))
        mode = self.__within_set__(self.index)[2]
        if not boot(self.pyboy, mode, self.render_window, self.boot_max_ticks):
            raise RuntimeError(
                f"no room was loaded within {self.boot_max_ticks} frames of booting "
                f"{self.romfile}. Check the ROM is {ROM_NAME} (MD5 {ROM_MD5}).")
        settle(self.pyboy, self.render_window, **self.settle_kwargs)
        self.intro_ticks = wait_until_interactive(self.pyboy, self.render_window)
        self.state = AmazingTaterGBState(self.pyboy, 0)
        if self.should_calibrate and self.calibration is None:
            # A property of the cartridge, not of the room, so once is enough.
            self.calibration = calibrate(self.pyboy, self.state, self.render_window,
                                         **self.settle_kwargs)
            self.actions = button_actions(self.calibration)
            load_state(self.pyboy, self.state.gb_state, self.render_window)
        self.state_history = [self.state]
        return self.state, {"level_index": self.index,
                            "level": self.label_for(),
                            "mode": mode,
                            "size": (self.state.width, self.state.height),
                            "taters": self.state.total,
                            "block_squares": len(self.state.block_squares),
                            "pits": len(self.state.pits),
                            "intro_ticks": self.intro_ticks,
                            "calibration": self.calibration}

    def is_goal(self, state):
        return state.solved

    def is_terminal(self, state):
        """Never, and the honest answer is that this cartridge does not offer a cheap test.

        Amazing Tater has real dead ends (a block settled into the one pit that had to be
        crossed somewhere else is gone for good, and so is the room), but recognising them
        needs reachability under moving turnstiles, which is not something the board buffer
        answers. The twin's test, "no move changes anything", is not available here either:
        finding that out means expanding the state, and expanding calls back through
        `is_terminal`. A position with nothing left in it simply expands to no successors,
        which search treats as a dead end anyway; claiming more than that would risk pruning
        a branch that was still solvable, which is the worse failure of the two.
        """
        return False

    def __score__(self, state):
        """How much of the room is done: taters home."""
        return state.taters_home
