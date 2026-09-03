"""Adventures of Lolo on a Game Boy, driven through PyBoy.

The sibling module `gameboy_py/lolo.py` re-implements the puzzle in Python over the same 163
rooms. This one plays the real cartridge inside an emulator, so the transition function is
HAL's code rather than a reconstruction of it. States carry an emulator save-state, which is
what lets search branch: applying an action rewinds the machine to the parent first.

The addresses below come from a reverse-engineering pass over
`Adventures of Lolo (U) [S][!].gb` (MD5 `8f6b6ef366a787852f664d945c86eb72`, internal title
`LOLO2`) and are documented in `docs/environments/lolo-gb-memory-map.md`. They are
revision-specific, which is why `LoloGBEnv` checks the ROM's MD5 and warns when it differs.

    env = LoloGBEnv("Adventures of Lolo (U) [S][!].gb")
    env.set_index(38)                 # intermediate 1-1
    state, info = env.reset()
    print(state)
    for action, successor in env.successors(state):
        ...

Four things about this cartridge shape the module.

**The board buffer is the level, not the position.** `LoadRoom` at `$11E5` copies 64 bytes
out of bank 13 into `$C3BF` (eight rows of eight cells, one byte each) and *never touches
them again*. Collecting a heart, pushing an Emerald Framer and opening the door all leave
`$C3BF` exactly as it was loaded. So the live position cannot be read there. It is read off
the BG tilemap instead, where every cell is a 2x2 block of tiles at columns 1-16, rows 1-16,
and it is the tilemap that the game actually redraws.

**Tile numbers are per-environment.** The cartridge ships eight terrain themes and the same
cell code renders with different tiles in each, so no fixed tile table would work. `reset`
learns the handful of tile numbers it needs (heart, closed door, open door, Emerald Framer)
by reading the tilemap while the board buffer still describes it, and decodes every later
frame against that. See `learn_tiles`.

**Lolo lives on a half-cell grid.** A d-pad press held 1-14 frames moves him 8 pixels, half a
cell; 16-28 frames moves a whole one. `PRESS_TICKS` is 20 so that one action is one cell,
which is the granularity the room is designed on; `HALF_STEP_TICKS` is offered for the
positions where standing on a half-cell matters.

**Enemies move only when Lolo does.** Left alone the board is completely still, measured on
all eight enemy kinds for 540 frames. That is what makes a settle predicate work at all here,
and it is why the cartridge is a sane thing to search: the game advances on Lolo's clock.
"""
from collections import namedtuple

from planiverse.environments.gameboy.gb import (
    GBAction, GBEnv, GBState, load_state, sprites as _oam_sprites,
)

# --------------------------------------------------------------------------- the ROM

ROM_MD5 = "8f6b6ef366a787852f664d945c86eb72"
ROM_NAME = "Adventures of Lolo (U) [S][!].gb"

#: Bank 13 holds the room table, flat and uncompressed: `$4000 + N*64`, which is file offset
#: `13 * 0x4000 + N*64`. 16 KiB / 64 gives 256 slots; only the first 163 are rooms.
ROOM_TABLE_OFFSET = 13 * 0x4000
ROOM_BYTES = 64
ROOM_COUNT = 163

# ------------------------------------------------------------------ how rooms are grouped
# `$26AA` computes `floor = (room - 38) // 14` and `level = (room - 38) % 14`, which fixes
# where the tutorial ends and the graded rooms begin. The rest of the split is the shape of
# the 163 live slots and matches the published description of the European release: a
# tutorial of 19 rooms, five intermediate floors of 14, ten advanced floors of 5, and 5 Pro
# rooms: 144 distinct puzzles, with each tutorial room stored twice.

#: Rooms 0-37: 19 puzzles, each stored as a (demonstration, play-it-yourself) pair.
TUTORIAL_START, TUTORIAL_PAIRS = 0, 19
TUTORIAL_END = TUTORIAL_START + 2 * TUTORIAL_PAIRS       # 38

INTERMEDIATE_START, INTERMEDIATE_FLOORS, INTERMEDIATE_PER_FLOOR = 38, 5, 14
ADVANCED_START, ADVANCED_FLOORS, ADVANCED_PER_FLOOR = 108, 10, 5
PRO_START, PRO_COUNT = 158, 5

# ------------------------------------------------------------------- memory map §5, §7
# Every address here is in `docs/environments/lolo-gb-memory-map.md`, with how it was
# established.

ROOM_NUMBER_ADDR = 0xC3A6        # index into the bank 13 room table
FLOOR_ADDR = 0xC3A4              # (room - 38) // 14, written by $26B8
LEVEL_IN_FLOOR_ADDR = 0xC3A5     # (room - 38) % 14
HEARTS_LEFT_ADDR = 0xC3A9        # heart framers still to collect; the status bar's number
MAGIC_SHOTS_ADDR = 0xC4AD        # magic shots in hand; +2 per magic heart framer, -1 per shot
SCENE_ADDR = 0xC3BE              # which screen the game is running
BOARD_ADDR = 0xC3BF              # the room as loaded, 8x8 stride 8, never updated
OAM_BUFFER_ADDR = 0xC000         # OAM DMA source: 40 sprites of 4 bytes

#: `$C3BE` while an intermediate room is being played. The tutorial's demonstration rooms run
#: as `$14` and play themselves, which is the one boot outcome that has to be rejected rather
#: than merely waited out: a self-playing board answers every action with a different one.
SCENE_PLAYING = 0x17
SCENE_TUTORIAL_DEMO = 0x14

#: `LD A,($C3A6)` inside `LoadRoom`, three instructions past the bank switch. Hooking the
#: entry point at `$11E5` would be one instruction too early to matter and one bank switch
#: too soon to be safe; hooking here puts the write between the caller's choice of room and
#: the loader's read of it.
LOAD_ROOM_READ = 0x11EC

# ------------------------------------------------------------------------- the alphabet
# Shared with `gameboy_py/lolo.py`, so a room printed by either module reads the same. The
# names are the cartridge's own: the object list at `$2CA9` is plain ASCII and reads
# "EMERALD FRAMERS / TREES / ROCKS / DESERTS / ENEMY HOLES / RIVERS / BREAK TILE /
# FLOWER BEDS / AND JEWEL BOXES", then "BRIDGE / ONE-WAY PASS / HAMMER", and the character
# list just above it reads "SNAKEY / LEEPER / GOL / ROCKY / ALMA / SKULL / MEDUSA /
# DON MEDUSA".

ROCK, TREE, RIVER, FLOOR = "#", "T", "~", "."
BRIDGE, FRAMER, HEART, MAGIC_HEART, DOOR, DOOR_OPEN = "=", "O", "H", "h", "D", "d"
DESERT, BREAK_TILE, FLOWER_BED, MARKER = ",", "x", "*", "o"
ONE_WAY = {"v": (1, 0), "<": (0, -1), "^": (-1, 0), ">": (0, 1)}
LOLO = "@"

#: Enemy glyphs, keyed by the cell code's family. Each family is four consecutive codes, one
#: per facing. Identification is argued in the memory map: SNAKEY and MEDUSA are pinned by
#: name and by behaviour, SKULL by its sprite, and the rest by what they do when Lolo moves.
SNAKEY, LEEPER, ROCKY, ALMA, GOL, SKULL, MEDUSA, DON_MEDUSA = "S", "L", "R", "A", "G", "K", "M", "N"
ENEMY_GLYPHS = frozenset({SNAKEY, LEEPER, ROCKY, ALMA, GOL, SKULL, MEDUSA, DON_MEDUSA})

#: Cell code -> glyph. Built rather than written out because most of the map is families of
#: four consecutive codes and a table would hide that.
CELL_GLYPHS = {0x80: TREE, 0x81: ROCK, 0x88: FLOOR, 0x89: BRIDGE, 0x8A: BRIDGE,
               0x8B: "v", 0x8C: "<", 0x8D: "^", 0x8E: ">", 0x8F: FRAMER,
               0x90: HEART, 0x91: MAGIC_HEART, 0x92: DESERT, 0x93: BREAK_TILE, 0x94: BREAK_TILE,
               0x95: FLOWER_BED, 0x96: DOOR, 0x9F: BREAK_TILE, 0x00: LOLO}
CELL_GLYPHS.update({code: RIVER for code in range(0x82, 0x88)})
CELL_GLYPHS.update({code: MARKER for code in range(0x97, 0x9D)})
for _base, _glyph in ((0x04, LEEPER), (0x08, ROCKY), (0x0C, ALMA), (0x10, GOL),
                      (0x14, SKULL), (0x18, SNAKEY), (0x1C, MEDUSA), (0x20, DON_MEDUSA)):
    CELL_GLYPHS.update({_base + _facing: _glyph for _facing in range(4)})

#: Both heart framers, so a caller can ask "is this a heart" without caring which kind.
HEART_GLYPHS = frozenset({HEART, MAGIC_HEART})

#: What a step is refused by. Rivers and rocks and trees are walls to Lolo; a Framer is a
#: wall unless it can be pushed; an enemy is a wall until it is shot into an egg.
BLOCKING = frozenset({ROCK, TREE, RIVER, FRAMER}) | ENEMY_GLYPHS

Position = namedtuple("Position", ["row", "col"])

#: The four tile numbers `read_grid` decodes against, learned per room by `learn_tiles`.
TileMap = namedtuple("TileMap", ["heart", "framer", "door_closed", "door_open"])


# ------------------------------------------------------------------- reading the ROM
# Pure functions of the cartridge image, so this is the one part of the module that needs
# neither an emulator nor a boot.

def read_room(rom, index):
    """One room's 64 bytes, straight out of bank 13."""
    start = ROOM_TABLE_OFFSET + index * ROOM_BYTES
    return rom[start:start + ROOM_BYTES]


def decode_room(cells):
    """64 cell codes as eight rows of eight glyphs."""
    return tuple("".join(CELL_GLYPHS.get(cells[row * 8 + col], "?") for col in range(8))
                 for row in range(8))


def read_rooms(romfile):
    """All 163 rooms, as tuples of eight 8-character rows.

    This is where `lolo.py`'s rooms came from. Reading them back is also how a claim about a
    room can be settled without a screenshot: the cartridge is the authority, and it is four
    lines of decoding away.
    """
    with open(romfile, "rb") as handle:
        rom = handle.read()
    return tuple(decode_room(read_room(rom, index)) for index in range(ROOM_COUNT))


def room_label(index):
    """How the game itself numbers a room.

    The tutorial stores each of its 19 puzzles twice (the demonstration the game plays for
    you, then the same room to try), so its labels carry which half of the pair a slot is.
    """
    if index < TUTORIAL_END:
        pair, half = divmod(index, 2)
        return f"tutorial {pair + 1}{'a' if half == 0 else 'b'}"
    if index < ADVANCED_START:
        floor, level = divmod(index - INTERMEDIATE_START, INTERMEDIATE_PER_FLOOR)
        return f"int {floor + 1}-{level + 1}"
    if index < PRO_START:
        floor, level = divmod(index - ADVANCED_START, ADVANCED_PER_FLOOR)
        return f"adv {floor + 1}-{level + 1}"
    return f"pro {index - PRO_START + 1}"


def verify_room_table(rom):
    """The indices whose 64 bytes are not a room, checked against the cell vocabulary.

    Bank 13 has 256 slots and only 163 of them are rooms; the rest is other data. A room uses
    only codes this module has a glyph for, holds exactly one Lolo (`$00`) and exactly one
    door (`$96`), and every one of the first 163 slots passes all three. Returns the offending
    indices, so an empty result for `range(163)` is the cheapest evidence there is that the
    table has been read correctly.
    """
    bad = []
    for index in range(ROOM_COUNT):
        cells = read_room(rom, index)
        if (any(code not in CELL_GLYPHS for code in cells)
                or cells.count(0x00) != 1 or cells.count(0x96) != 1):
            bad.append(index)
    return tuple(bad)


# ------------------------------------------------------------------- pure decoding
# Split out from the emulator so they can be tested against synthetic tilemaps.

def cell_of(sprite_y, sprite_x):
    """An OAM entry's `(y, x)` as a cell, in halves.

    A sprite sits 16 pixels below and 8 to the right of where it draws, and the playfield's
    top-left cell starts at screen `(8, 8)`. Returned as floats because Lolo genuinely stands
    on half-cells: `(4.0, 3.5)` is a real, reachable position and rounding it away would make
    two different positions compare equal.
    """
    return ((sprite_y - 24) / 16.0, (sprite_x - 16) / 16.0)


def decode_grid(board, tiles, tilemap):
    """The live room, as eight rows of eight glyphs.

    `board` is the static room from `$C3BF` and supplies the terrain, which never changes.
    Everything that *does* change (hearts being taken, Framers being pushed, the door opening)
    is read from `tilemap`, an 8x8 of the top-left tile of each cell's 2x2 block, against
    the numbers `learn_tiles` measured for this room.

    A cell whose static glyph is an object but whose tile is no longer that object's tile has
    had the object removed, and renders as floor. That is the whole trick: neither source
    alone is the position, and the pair of them is.
    """
    static = decode_room(board)
    rows = []
    for row in range(8):
        line = []
        for col in range(8):
            tile, glyph = tilemap[row][col], static[row][col]
            if tiles.heart is not None and tile == tiles.heart:
                # Both heart framers draw the same tile, so which kind this one is has to come
                # from the static room. Hearts never move, so that is exact.
                line.append(glyph if glyph in HEART_GLYPHS else HEART)
            elif tiles.framer is not None and tile == tiles.framer:
                line.append(FRAMER)
            elif tiles.door_open is not None and tile == tiles.door_open:
                line.append(DOOR_OPEN)
            elif tile == tiles.door_closed:
                line.append(DOOR)
            elif glyph in HEART_GLYPHS or glyph in (FRAMER, DOOR):
                line.append(FLOOR)          # the object was taken, pushed away, or entered
            elif glyph == LOLO or glyph in ENEMY_GLYPHS:
                line.append(FLOOR)          # actors are sprites; the cell under them is floor
            else:
                line.append(glyph)

        rows.append("".join(line))
    return tuple(rows)


def door_cell(board):
    """Where the room's one door is. Every room has exactly one; see `verify_room_table`."""
    index = list(board).index(0x96)
    return Position(*divmod(index, 8))


def start_cell(board):
    """Where Lolo starts. Every room has exactly one `$00`."""
    index = list(board).index(0x00)
    return Position(*divmod(index, 8))


def render_grid(grid, lolo, enemies=()):
    """A position as ASCII, with Lolo and the enemies drawn over the terrain.

    Lolo is placed by rounding his half-cell position, so a room caught mid-slide still prints
    somewhere sensible rather than nowhere.
    """
    rows = [list(row) for row in grid]
    for row, col, glyph in enemies:
        if 0 <= round(row) < 8 and 0 <= round(col) < 8:
            rows[round(row)][round(col)] = glyph
    if lolo is not None and 0 <= round(lolo[0]) < 8 and 0 <= round(lolo[1]) < 8:
        rows[round(lolo[0])][round(lolo[1])] = LOLO
    return "\n".join("".join(row) for row in rows)


# ------------------------------------------------------------------------- emulation

def read_board(pyboy):
    """The room as `LoadRoom` left it. Static for the whole room; see the module docstring."""
    return bytes(pyboy.memory[BOARD_ADDR:BOARD_ADDR + ROOM_BYTES])


def read_tilemap(pyboy):
    """The top-left tile of each cell's 2x2 block, as an 8x8.

    The playfield is BG columns 1-16 and rows 1-16, so cell `(r, c)` starts at tile
    `(1 + 2c, 1 + 2r)`. One tile per cell is enough: no two objects in a room share a
    top-left tile.
    """
    background = pyboy.tilemap_background
    return tuple(tuple(background[1 + 2 * col, 1 + 2 * row] for col in range(8))
                 for row in range(8))


def learn_tiles(pyboy):
    """Measure this room's tile numbers for heart, Framer and door, while they still agree.

    Called once, on a freshly loaded room, when the board buffer and the tilemap describe the
    same thing: every heart cell still holds a heart, every Framer is where it was loaded, and
    the door is shut. The open-door tile cannot be read yet (no room starts with it open), so
    it is taken as the closed tile plus two, which is what every environment does and what
    `docs/environments/lolo-gb-memory-map.md` records the measurement for.

    Returns None for an object the room does not contain, which is not an error: plenty of
    rooms have no Framer, and six have no heart at all.
    """
    board, tilemap = read_board(pyboy), read_tilemap(pyboy)
    found = {}
    for row in range(8):
        for col in range(8):
            glyph = CELL_GLYPHS.get(board[row * 8 + col])
            # The two heart framers draw the same tile, so they share one entry.
            found.setdefault(HEART if glyph in HEART_GLYPHS else glyph, tilemap[row][col])
    closed = found.get(DOOR)
    return TileMap(heart=found.get(HEART), framer=found.get(FRAMER),
                   door_closed=closed, door_open=None if closed is None else closed + 2)


def read_grid(pyboy, tiles):
    return decode_grid(read_board(pyboy), tiles, read_tilemap(pyboy))


def shadow_oam(pyboy):
    """The OAM DMA buffer at `$C000`. Lolo's slide and every enemy live here and nowhere else."""
    return bytes(pyboy.memory[OAM_BUFFER_ADDR:OAM_BUFFER_ADDR + 160])


def lolo_position(pyboy):
    """Lolo's cell, in halves.

    Lolo is the first pair of sprites in the shadow OAM in every room measured, which is what
    lets this be a two-byte read rather than a search of the buffer. `is_playing` refuses a
    boot where slot 0 is empty, so the assumption is checked before it is used.
    """
    buffer = pyboy.memory[OAM_BUFFER_ADDR:OAM_BUFFER_ADDR + 2]
    return cell_of(buffer[0], buffer[1])


def other_sprites(pyboy):
    """Every visible sprite that is not Lolo, as `(row, col)` cells.

    Every actor on this cartridge is drawn as two 8x16 sprites side by side, filled into the
    shadow OAM in pairs: Lolo in slots 0 and 1, then one pair per enemy or egg. Taking every
    second entry keeps the left half of each pair and drops the right, which is what makes an
    enemy one cell rather than two: reading both halves would put a Snakey at columns 5 and
    5.5 and paint over whatever is standing at column 6.
    """
    entries = _oam_sprites(pyboy, OAM_BUFFER_ADDR, visible_only=True)[2:]
    return tuple(sorted({cell_of(y, x) for y, x, _tile in entries[::2]}))


def hearts_left(pyboy):
    """Heart framers still to collect. The number the status bar shows."""
    return pyboy.memory[HEARTS_LEFT_ADDR]


def magic_shots(pyboy):
    """Magic shots in hand.

    Zero on a fresh boot, +2 for each magic heart framer collected, -1 for each shot fired.
    Found by diffing work RAM across collecting one: it is the only byte in the game state
    that goes 0, 2, 1, 0 across a magic heart and two shots.
    """
    return pyboy.memory[MAGIC_SHOTS_ADDR]


def is_playing(pyboy):
    """Whether a graded room is up and waiting for input.

    Three things have to agree. The scene has to be the one the intermediate route runs in:
    the tutorial's demonstration rooms are `$14` and play themselves, and a board that moves
    on its own is worse than no board at all. The board buffer has to be non-empty, because it
    is zero on every menu. And Lolo has to have a sprite, because the buffer keeps the last
    room's bytes through the cutscene that precedes the first one.
    """
    return (pyboy.memory[SCENE_ADDR] == SCENE_PLAYING
            and any(read_board(pyboy))
            and pyboy.memory[OAM_BUFFER_ADDR] != 0)


# ----------------------------------------------------------------------- driving it

SETTLE_MAX_TICKS = 300           # five seconds; one cell of walking animates for sixteen
SETTLE_STABLE_TICKS = 8          # frames the sprite buffer must hold still
PRESS_TICKS = 20                 # frames a d-pad press is held: one whole cell
HALF_STEP_TICKS = 8              # ... and half of one
SHOOT_TICKS = 6                  # the magic shot is an edge, not a hold
PROBE_MAX_HOLD = 72              # longest hold `calibrate` will try: four cells' worth

#: The hold that moves Lolo one cell, and the closed range of holds that do. Measured off the
#: cartridge by `calibrate` rather than trusted from here.
Calibration = namedtuple("Calibration", ["press_ticks", "hold_window"])

DIRECTIONS = {"left": (0, -1), "up": (-1, 0), "down": (1, 0), "right": (0, 1)}

#: What each button costs. `a` is the magic shot: it turns the enemy Lolo is facing into an
#: egg, and a second shot at an egg blasts it out of the room. Shots are rationed by the
#: cartridge (the status bar's "PWR" meter), so they are worth what a move is and no more.
action_cost_map = {"left": 1, "right": 1, "up": 1, "down": 1, "a": 1, "nop": 0}

#: START opens the pause overlay and B does nothing in a room; neither is offered.
action_list = [f"{button},{PRESS_TICKS}" for button in DIRECTIONS] + [f"a,{SHOOT_TICKS}"]


def button_actions(calibration=None):
    """The primitive actions, each held for however long calibration settled on."""
    ticks = (calibration or Calibration(PRESS_TICKS, None)).press_ticks
    return [f"{button},{ticks}" for button in DIRECTIONS] + [f"a,{SHOOT_TICKS}"]


def settle(pyboy, render=False, max_ticks=SETTLE_MAX_TICKS, stable_ticks=SETTLE_STABLE_TICKS):
    """Run the emulator until the room stops moving, and report whether it did.

    Watching the shadow OAM is enough, and watching anything else would be wrong. The tilemap
    is updated the frame a heart is taken, so a move that ends on a heart would be called
    finished while Lolo is still sliding, and the dropped presses that causes look exactly
    like a planner's action having no effect. Lolo's slide, every enemy's step and the magic
    shot in flight are all sprites, so when `$C000` holds still for eight frames, nothing on
    the board is moving.
    """
    previous, stable = None, 0
    for _ in range(max_ticks):
        pyboy.tick(1, render)
        current = shadow_oam(pyboy)
        if current == previous:
            stable += 1
            if stable >= stable_ticks:
                return True
        else:
            previous, stable = current, 0
    return False


BOOT_TITLE_TICKS = 400           # the title screen's own animation, before it takes input
BOOT_MENU_TICKS = 120
BOOT_STORY_TICKS = 40            # one line of the King's introduction
BOOT_ROOM_TICKS = 24
BOOT_MAX_ROOM_PRESSES = 500

#: How many taps of A walk the King's introduction from the NEW GAME wheel to its last screen,
#: the one offering "Push A: ENTRY / Push B: INTERMEDIATE". Measured, and checked afterwards:
#: `reset` tries the neighbouring counts if this one lands somewhere else, because being one
#: screen out drops the boot into the tutorial, whose demonstration rooms play themselves.
BOOT_STORY_PRESSES = 27
BOOT_STORY_ALTERNATIVES = (27, 28, 26, 29, 25)


def boot(pyboy, story_presses=BOOT_STORY_PRESSES, render=False):
    """Get from power-on to a graded room. True if one appeared.

    The route is: wait out the title, START for the NEW GAME wheel, A to choose it, then A
    through the King's introduction to the screen that offers ENTRY or INTERMEDIATE, B to take
    INTERMEDIATE, and A through the orchestra cutscene until a room is up.

    B rather than A at that one screen is the whole point. A starts "First steps in Eden", the
    tutorial, where the first room of every pair is a demonstration the game plays for itself;
    B goes to the graded rooms, which is the only route that takes input.
    """
    pyboy.tick(BOOT_TITLE_TICKS, render)
    pyboy.button("start", 5)
    pyboy.tick(BOOT_MENU_TICKS, render)
    pyboy.button("a", 5)
    pyboy.tick(BOOT_MENU_TICKS, render)
    for _ in range(story_presses):
        pyboy.button("a", 4)
        pyboy.tick(BOOT_STORY_TICKS, render)
    pyboy.button("b", 6)
    pyboy.tick(BOOT_MENU_TICKS + 30, render)
    for _ in range(BOOT_MAX_ROOM_PRESSES):
        pyboy.button("a", 3)
        pyboy.tick(BOOT_ROOM_TICKS, render)
        if is_playing(pyboy):
            return True
    return False


def _force_room(context):
    """Hook body: pin `$C3A6` on the way into `LoadRoom`, once.

    Writing it from outside on a frame boundary is not enough: the route into a room sets the
    number and calls the loader within the same frame. Hooking `$11EC` puts the write between
    the two. It fires once and then stands down, so that clearing the room lets the cartridge
    advance to the next one of its own accord instead of being pinned into replaying this one.
    """
    pyboy, box = context
    if box[0] is not None:
        pyboy.memory[ROOM_NUMBER_ADDR] = box[0]
        box[0] = None


# --------------------------------------------------------------------------- probing

def _press(pyboy, state, button, hold, render=False, **settle_kwargs):
    """Rewind to `state`, hold `button` for `hold` frames, and read back where Lolo stopped."""
    load_state(pyboy, state.gb_state, render)
    pyboy.button(button, hold)
    pyboy.tick(hold + 1, render)
    settle(pyboy, render, **settle_kwargs)
    return lolo_position(pyboy)


def _open_direction(pyboy, state, render, settle_kwargs, max_hold):
    """A direction with two clear cells of room, so a long hold has somewhere to show itself.

    Picking the first direction that moves at all is not enough: with a wall one cell away
    every hold looks identical and the window comes back as wide as the probe.
    """
    fallback = None
    for direction, step in DIRECTIONS.items():
        moved = _cells_moved(state.lolo, _press(pyboy, state, direction, max_hold, render,
                                                **settle_kwargs), step)
        if moved >= 2:
            return direction
        if moved >= 1 and fallback is None:
            fallback = direction
    return fallback


def _cells_moved(before, after, step):
    """How far Lolo travelled along `step`, or 0 if he went somewhere else entirely."""
    delta = (after[0] - before[0], after[1] - before[1])
    if step[0]:
        return delta[0] / step[0] if abs(delta[1]) < 1e-9 else 0
    return delta[1] / step[1] if abs(delta[0]) < 1e-9 else 0


def measure_hold_window(pyboy, state, render=False, max_hold=PROBE_MAX_HOLD, **settle_kwargs):
    """The closed range of hold lengths that move Lolo exactly one cell.

    Lolo walks eight pixels (half a cell) per sixteen frames of hold, so this measures both
    ends of the middle band rather than assuming either: below it a press is a half-step, above
    it two cells, and a planner handed back a two-cell move is not being told what its action
    did. `Adventures of Lolo (U)` reports a band about sixteen frames wide (`(16, 28)` walking
    right out of int 1-1's start, `(17, 31)` on whichever direction `_open_direction` picks),
    and `PRESS_TICKS` sits inside it either way.

    Returns `(low, high)`, or None if no hold moves Lolo exactly one cell.
    """
    direction = _open_direction(pyboy, state, render, settle_kwargs, max_hold)
    if direction is None:
        return None
    step, low = DIRECTIONS[direction], None
    for hold in range(1, max_hold + 1):
        moved = _cells_moved(state.lolo, _press(pyboy, state, direction, hold, render,
                                                **settle_kwargs), step)
        if moved == 1 and low is None:
            low = hold
        elif moved > 1 and low is not None:
            return (low, hold - 1)
    return None if low is None else (low, max_hold)


def calibrate(pyboy, state, render=False, max_hold=PROBE_MAX_HOLD, **settle_kwargs):
    """Measure how this cartridge wants to be driven. A property of the game, so once is enough.

    Costs on the order of eighty presses, each rewound to `state` first, so nothing it does
    survives.
    """
    window = measure_hold_window(pyboy, state, render, max_hold, **settle_kwargs)
    if window is None:
        # Lolo never moved: a room where he starts boxed in. Rather than drive the game with
        # a made-up hold, fall back to the documented default and say the window is unknown.
        return Calibration(PRESS_TICKS, None)
    low, high = window
    return Calibration((low + high) // 2, window)


# ----------------------------------------------------------------------------- state

class LoloGBState(GBState):
    """A settled position: the emulator save-state plus the room read back off the screen."""

    def __init__(self, pyboy, depth, tiles, door, died=False):
        super().__init__(pyboy, depth)
        self.tiles = tiles
        self.door = door
        self.died = died
        self.__update__(pyboy)

    def __update__(self, pyboy):
        self.grid = read_grid(pyboy, self.tiles)
        self.lolo = lolo_position(pyboy)
        self.cell = Position(round(self.lolo[0]), round(self.lolo[1]))
        self.enemies = other_sprites(pyboy)
        self.hearts_left = hearts_left(pyboy)
        self.shots = magic_shots(pyboy)

        # Sampled here, while the emulator still holds this position. Read later from live
        # memory they would describe whichever state was applied last.
        self.door_open = self.grid[self.door.row][self.door.col] == DOOR_OPEN
        self.solved = self.hearts_left == 0 and self.cell == self.door

        predicates = [f"at(lolo, {self.cell.row}, {self.cell.col})",
                      f"hearts-left({self.hearts_left})",
                      f"shots({self.shots})"]
        # Both heart glyphs, not just `HEART`: the magic heart counts towards `hearts_left`
        # and sits on the board like any other, so leaving it out left a state whose own
        # counter said two hearts and whose predicates named one. A planner reads only the
        # predicates, so the second heart may as well not have existed.
        predicates += [f"at(heart, {row}, {col})"
                       for row, line in enumerate(self.grid)
                       for col, glyph in enumerate(line) if glyph in HEART_GLYPHS]
        predicates += [f"at(framer, {row}, {col})"
                       for row, line in enumerate(self.grid)
                       for col, glyph in enumerate(line) if glyph == FRAMER]
        predicates += [f"at(enemy, {round(row)}, {round(col)})" for row, col in self.enemies]
        if self.door_open:
            predicates.append("door-open")
        if self.solved:
            predicates.append("goal-reached")
        if self.died:
            predicates.append("terminal-state")
        self.literals = frozenset(predicates)

    def __eq__(self, other):
        # The position is the board, where Lolo is, where the enemies are, and how many hearts
        # are left. Depth and history are not part of it, so a state reached two ways compares
        # equal and search can close. `died` is part of it, because a room that has just been
        # restarted after a death looks exactly like a fresh one and must not be mistaken for
        # somewhere worth expanding.
        return (isinstance(other, LoloGBState) and self.grid == other.grid
                and self.lolo == other.lolo and self.enemies == other.enemies
                and self.hearts_left == other.hearts_left and self.shots == other.shots
                and self.died == other.died)

    def __hash__(self):
        return hash((self.grid, self.lolo, self.enemies, self.hearts_left, self.shots,
                     self.died))

    def __str__(self):
        return render_grid(self.grid, self.lolo,
                           [(row, col, "e") for row, col in self.enemies])

    def __repr__(self):
        return (f"<LoloGBState(depth={self.depth}, hearts_left={self.hearts_left}, "
                f"shots={self.shots}, lolo=({self.lolo[0]}, {self.lolo[1]}), "
                f"died={self.died})>")


# ---------------------------------------------------------------------------- actions

class LoloGBAction(GBAction):
    """A button held for a number of frames, spelled `"button,ticks"`.

    The four directions and `a`, the magic shot. `apply` inherits the shared shape from
    `GBAction`; what is added here is telling a death from a move, which cannot be read from a
    single frame and has to be a comparison against the position the action started from.
    """

    cost_map = action_cost_map

    def __settle__(self, pyboy, render, **settle_kwargs):
        return settle(pyboy, render, **settle_kwargs)

    def __next_state__(self, pyboy, state):
        return LoloGBState(pyboy, state.depth + 1, state.tiles, state.door,
                           died=self.__died__(pyboy, state))

    def __died__(self, pyboy, state):
        """Whether Lolo lost a life during this action.

        The cartridge answers by restarting the room where he stands, which is not a flag
        anywhere in work RAM but is unmistakable from either side of the transition: the
        hearts he had collected come back, or he is suddenly somewhere no single action could
        have taken him. One cell is the most any action moves him, so a jump of more than one
        is a respawn and nothing else. Both tests are needed: the first misses a death in a
        room where he had not collected anything yet, the second misses one where he died on
        the cell he started the action from.
        """
        if hearts_left(pyboy) > state.hearts_left:
            return True
        row, col = lolo_position(pyboy)
        return abs(row - state.lolo[0]) > 1.0 or abs(col - state.lolo[1]) > 1.0


# ------------------------------------------------------------------------ environment

class LoloGBEnv(GBEnv):
    """Adventures of Lolo, played on the cartridge.

    The ROM is copyrighted and is not distributed with this repo; pass the path to your own
    dump. `set_index` selects which of the 163 rooms the cartridge's loader will build: the
    same indices `gameboy_py/lolo.py` uses, so `set_index(38)` is the same room in both.
    """

    rom_md5 = ROM_MD5
    rom_name = ROM_NAME
    action_class = LoloGBAction

    def __init__(self, romfile, render=False, verify_rom=True, calibrate=False, magic_shots=0,
                 settle_max_ticks=SETTLE_MAX_TICKS, settle_stable_ticks=SETTLE_STABLE_TICKS):
        self.romfile = romfile
        # Not `self.render`: that name belongs to the method which prints the history.
        self.render_window = render
        self.pyboy = None
        self.room_index = 0
        self.state = None
        self.state_history = []
        self.tiles = None
        self.door = None
        # Off by default, unlike the other cartridges here. Calibration costs eighty boots'
        # worth of presses and this ROM's answer (`(16, 28)`, so 20) has been measured and
        # written down; turn it on to check that claim, not to discover it.
        self.should_calibrate = calibrate
        self.calibration = None
        self.actions = action_list
        # Booting straight into a room starts the magic-shot meter empty, because the meter is
        # the player's and not the room's: on a real playthrough whatever was left over from
        # the room before comes with you. Rooms that need a shot they cannot earn in-room
        # (int 1-5 is one, a Snakey in the only gap in a wall of trees and not a magic heart
        # framer anywhere) are unclearable from a cold boot for that reason and not because
        # anything is wrong. Seed the meter here to play them. `gameboy_py/lolo.py` takes the
        # same argument and means the same thing by it.
        self.magic_shots = magic_shots
        self.settle_kwargs = {"max_ticks": settle_max_ticks, "stable_ticks": settle_stable_ticks}
        if verify_rom:
            self.__verify_rom__()

    # ------------------------------------------------------------------- the catalogue

    def rooms(self):
        """The 163 rooms, decoded out of this cartridge. Needs no emulator."""
        return read_rooms(self.romfile)

    def set_index(self, index):
        """Select the room, zero-based. See `room_label` for how the game numbers them."""
        if not 0 <= index < ROOM_COUNT:
            raise IndexError(
                f"Invalid index: {index}. This cartridge has {ROOM_COUNT} rooms, so the index "
                f"must be 0-{ROOM_COUNT - 1}.")
        self.room_index = index

    def label_for(self, index=None):
        return room_label(self.room_index if index is None else index)

    # ------------------------------------------------------------------------ the game

    def reset(self):
        """Boot the cartridge to the selected room.

        Booting is retried across `BOOT_STORY_ALTERNATIVES` rather than trusted once. The one
        screen where the route presses B instead of A is reached by counting taps through the
        King's introduction, and landing one screen early or late starts the tutorial instead,
        whose demonstration rooms play themselves and would answer every action with a
        position nobody asked for. `is_playing` catches that, and the retry fixes it.
        """
        for presses in BOOT_STORY_ALTERNATIVES:
            self.__restart_emulator__()
            # The hook has to be in place before the first press: the route into a room sets
            # `$C3A6` and calls the loader inside one frame.
            self.pyboy.hook_register(0, LOAD_ROOM_READ, _force_room,
                                     (self.pyboy, [self.room_index]))
            if boot(self.pyboy, presses, self.render_window):
                break
        else:
            raise RuntimeError(
                f"no room was reached booting {self.romfile}. Check the ROM is {ROM_NAME} "
                f"(MD5 {ROM_MD5}).")
        settle(self.pyboy, self.render_window, **self.settle_kwargs)

        # Learned now, while the board buffer and the tilemap still describe the same room.
        self.tiles = learn_tiles(self.pyboy)
        self.door = door_cell(read_board(self.pyboy))
        if self.magic_shots:
            self.pyboy.memory[MAGIC_SHOTS_ADDR] = self.magic_shots
        self.state = LoloGBState(self.pyboy, 0, self.tiles, self.door)

        if self.should_calibrate and self.calibration is None:
            # A property of the cartridge, not of the room, so once is enough.
            self.calibration = calibrate(self.pyboy, self.state, self.render_window,
                                         **self.settle_kwargs)
            self.actions = button_actions(self.calibration)
            load_state(self.pyboy, self.state.gb_state, self.render_window)

        self.state_history = [self.state]
        return self.state, {"room_index": self.room_index,
                            "room": self.label_for(),
                            "floor": self.pyboy.memory[FLOOR_ADDR],
                            "level_in_floor": self.pyboy.memory[LEVEL_IN_FLOOR_ADDR],
                            "hearts": self.state.hearts_left,
                            "shots": self.state.shots,
                            "door": tuple(self.door),
                            "start": tuple(start_cell(read_board(self.pyboy))),
                            "calibration": self.calibration}

    def is_goal(self, state):
        """Every heart collected and Lolo standing on the door.

        This is the cartridge's own win condition, read off the screen rather than waited for:
        the door tile changes the frame the last heart is taken, and stepping onto it is what
        ends the room. It is checked here rather than by watching `$C3A6` advance because the
        room number only moves once the between-rooms sequence has run, and a state snapshotted
        during that sequence is not a Lolo position at all.
        """
        return state.solved

    def is_terminal(self, state):
        """Lolo lost a life, and the cartridge put the room back to the start.

        Absorbing, and it has to be: the restarted room is byte-for-byte the room `reset`
        handed out, so without the flag search would walk back into the initial state by a
        different route and never notice it had thrown a life away.
        """
        return state.died

    def __score__(self, state):
        """Hearts collected. `hearts_left` counts down, so this counts up."""
        return self.state_history[0].hearts_left - state.hearts_left


def _report(romfile, index=None, render=False, calibrate=False):
    """Print what this cartridge wants, and the room it was asked about."""
    env = LoloGBEnv(romfile, render=render, calibrate=calibrate)
    try:
        if index is not None:
            env.set_index(index)
        state, info = env.reset()
        print(f"room {info['room']} (index {info['room_index']}): {info['hearts']} hearts, "
              f"door at {info['door']}, Lolo starts at {info['start']}\n")
        print(state, "\n")
        calibration = info["calibration"]
        if calibration is None:
            print(f"  press_ticks   {PRESS_TICKS} (not measured; pass --calibrate)")
        elif calibration.hold_window is None:
            print("  hold window   not measurable — Lolo never moved")
        else:
            print(f"  hold window   {calibration.hold_window}  -> press_ticks "
                  f"{calibration.press_ticks}")
        print(f"\n  button actions: {button_actions(calibration)}")
    finally:
        env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure how an Adventures of Lolo cartridge wants to be driven, and show "
                    "a room as the console has it in work RAM.")
    parser.add_argument("rom", help=f"path to {ROM_NAME}")
    parser.add_argument("--room", type=int, default=None,
                        help=f"room index, 0-{ROOM_COUNT - 1}")
    parser.add_argument("--render", action="store_true", help="open an SDL2 window")
    parser.add_argument("--calibrate", action="store_true",
                        help="measure the d-pad hold window off the cartridge")
    parser.add_argument("--dump", action="store_true",
                        help="decode all 163 rooms out of the ROM and print them")
    args = parser.parse_args()
    if args.dump:
        for index, rows in enumerate(read_rooms(args.rom)):
            print(f"--- {index:3d} {room_label(index)}")
            for row in rows:
                print(f"  |{row}|")
    else:
        _report(args.rom, args.room, args.render, args.calibrate)
