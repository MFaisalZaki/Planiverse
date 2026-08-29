"""Tests for the Adventures of Lolo Game Boy environment.

Two tiers. The first is pure functions of bytes (the room decoder, the tilemap decoder, the
sprite arithmetic) checked against synthetic data, which is cheaper than booting a Game Boy
and covers the parts most likely to be quietly wrong. The second needs the real cartridge,
which is copyrighted and cannot ship here, so it is opt-in:

    PLANIVERSE_LOLO_ROM="/path/to/Adventures of Lolo (U) [S][!].gb" \\
        poetry run pytest tests/test_lolo_gb.py

There is no synthetic cartridge standing in for the ROM, unlike Puzznic and Flipull. This
game reads its live position off the BG tilemap rather than out of work RAM, so a fake
cartridge would have to draw a convincing screen rather than just park bytes at addresses:
much more work, for a much weaker check than `decode_grid` against a synthetic tilemap.
"""
import pytest

pytest.importorskip("pyboy", reason="pyboy is not installed")

from planiverse.environments.gameboy.lolo_gb import (  # noqa: E402
    ADVANCED_START, BLOCKING, CELL_GLYPHS, DOOR, DOOR_OPEN, ENEMY_GLYPHS, FLOOR, FRAMER,
    HEART, HEART_GLYPHS, INTERMEDIATE_START, LOLO, MAGIC_HEART, PRESS_TICKS, PRO_START,
    ROM_MD5, ROOM_BYTES, ROOM_COUNT, ROOM_TABLE_OFFSET, SHOOT_TICKS, TUTORIAL_END, TileMap,
    LoloGBAction, LoloGBEnv, action_cost_map, action_list, button_actions, cell_of,
    decode_grid, decode_room, door_cell, read_room, read_rooms, render_grid, room_label,
    start_cell, verify_room_table,
)

from conftest import (  # noqa: E402
    assert_string_literals, assert_successors_contract, lolo_rom_path,
)

needs_rom = pytest.mark.skipif(
    lolo_rom_path() is None,
    reason='set PLANIVERSE_LOLO_ROM to an "Adventures of Lolo (U) [S][!].gb" ROM',
)


# ---------------------------------------------------------------------- the alphabet

def test_every_enemy_family_is_four_consecutive_codes():
    """One per facing, which is why the map is built rather than typed out."""
    for base in (0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C, 0x20):
        glyphs = {CELL_GLYPHS[base + facing] for facing in range(4)}
        assert len(glyphs) == 1, f"${base:02X} family does not agree on a glyph: {glyphs}"
        assert glyphs <= ENEMY_GLYPHS


def test_lolo_is_the_one_code_that_appears_once_in_every_room():
    assert CELL_GLYPHS[0x00] == LOLO


def test_the_two_heart_framers_are_different_codes_with_the_same_meaning():
    """`$91` gives Lolo two magic shots and `$90` gives none, and they draw the same tile."""
    assert CELL_GLYPHS[0x90] == HEART
    assert CELL_GLYPHS[0x91] == MAGIC_HEART
    assert HEART_GLYPHS == {HEART, MAGIC_HEART}


def test_blocking_holds_the_things_a_step_is_refused_by():
    assert {"#", "T", "~", FRAMER} <= BLOCKING
    assert ENEMY_GLYPHS <= BLOCKING
    assert not ({FLOOR, DOOR, HEART, "="} & BLOCKING)


# ------------------------------------------------------------------- reading the ROM

def synthetic_room(cells):
    """64 cell codes as a bytes object, from a `{(row, col): code}` map over floor."""
    board = bytearray([0x88] * ROOM_BYTES)
    for (row, col), code in cells.items():
        board[row * 8 + col] = code
    return bytes(board)


def test_decode_room_turns_cell_codes_into_glyphs():
    board = synthetic_room({(0, 0): 0x81, (0, 1): 0x80, (0, 2): 0x87, (0, 3): 0x8F,
                            (0, 4): 0x90, (0, 5): 0x91, (0, 6): 0x96, (0, 7): 0x00,
                            (1, 0): 0x8B, (1, 1): 0x8C, (1, 2): 0x8D, (1, 3): 0x8E,
                            (1, 4): 0x18, (1, 5): 0x1C, (1, 6): 0x92, (1, 7): 0x95})
    rows = decode_room(board)
    assert rows[0] == "#T~OHhD@"
    assert rows[1] == "v<^>SM,*"
    assert rows[2] == "........"


def test_read_room_finds_the_table_where_the_loader_does():
    """Bank 13, `$4000 + N*64`, which is file offset `13*$4000 + N*64`."""
    rom = bytearray(ROOM_TABLE_OFFSET + 3 * ROOM_BYTES)
    rom[ROOM_TABLE_OFFSET + 2 * ROOM_BYTES: ROOM_TABLE_OFFSET + 3 * ROOM_BYTES] = \
        synthetic_room({(4, 4): 0x00})
    assert read_room(bytes(rom), 2)[4 * 8 + 4] == 0x00


def test_door_and_start_are_read_out_of_the_board():
    board = synthetic_room({(2, 3): 0x96, (6, 1): 0x00})
    assert door_cell(board) == (2, 3)
    assert start_cell(board) == (6, 1)


def test_verify_room_table_rejects_a_slot_that_is_not_a_room():
    """The check that established where the 163 rooms end (see the memory map §2)."""
    rom = bytearray(ROOM_TABLE_OFFSET + ROOM_COUNT * ROOM_BYTES)
    good = synthetic_room({(0, 0): 0x96, (7, 7): 0x00})
    for index in range(ROOM_COUNT):
        start = ROOM_TABLE_OFFSET + index * ROOM_BYTES
        rom[start:start + ROOM_BYTES] = good
    assert verify_room_table(bytes(rom)) == ()

    # Two doors is not a room, and neither is a code with no glyph.
    start = ROOM_TABLE_OFFSET + 7 * ROOM_BYTES
    rom[start:start + ROOM_BYTES] = synthetic_room({(0, 0): 0x96, (0, 1): 0x96, (7, 7): 0x00})
    start = ROOM_TABLE_OFFSET + 9 * ROOM_BYTES
    rom[start:start + ROOM_BYTES] = synthetic_room({(0, 0): 0x96, (7, 7): 0x00, (3, 3): 0x77})
    assert verify_room_table(bytes(rom)) == (7, 9)


# ------------------------------------------------------------------ how rooms group

def test_room_label_follows_the_cartridges_own_arithmetic():
    """`$26AA` divides by 14 after subtracting 38, which is what fixes these boundaries."""
    assert TUTORIAL_END == 38 and INTERMEDIATE_START == 38
    assert ADVANCED_START == 108 and PRO_START == 158
    assert room_label(0) == "tutorial 1a"
    assert room_label(37) == "tutorial 19b"
    assert room_label(38) == "int 1-1"
    assert room_label(51) == "int 1-14"
    assert room_label(52) == "int 2-1"
    assert room_label(108) == "adv 1-1"
    assert room_label(162) == "pro 5"


# --------------------------------------------------------------- reading the screen

def tilemap_of(rows, tiles):
    """An 8x8 of top-left tile numbers, from glyphs and a `TileMap`.

    Anything that is not one of the four objects `learn_tiles` measures draws as floor, which
    is what the cartridge does too: actors are sprites and the cell under them is ground.
    """
    lookup = {"#": 0x80, HEART: tiles.heart, MAGIC_HEART: tiles.heart, FRAMER: tiles.framer,
              DOOR: tiles.door_closed, DOOR_OPEN: tiles.door_open}
    return tuple(tuple(lookup.get(glyph, 0xA7) for glyph in row) for row in rows)


TILES = TileMap(heart=0xE2, framer=0xCE, door_closed=0x8C, door_open=0x8E)


def test_decode_grid_on_an_untouched_room_is_the_room():
    """Nothing has moved yet, so the screen and the board buffer still say the same thing,
    except for Lolo, who is a sprite and leaves floor behind him."""
    board = synthetic_room({(1, 1): 0x96, (3, 3): 0x90, (3, 5): 0x8F, (2, 2): 0x81,
                            (6, 1): 0x00})
    grid = decode_grid(board, TILES, tilemap_of(decode_room(board), TILES))
    assert grid[1][1] == DOOR and grid[3][3] == HEART and grid[3][5] == FRAMER
    assert grid[2][2] == "#"
    assert grid[6][1] == FLOOR, "Lolo is a sprite; the cell under him is ground"


def test_a_collected_heart_reads_as_floor():
    """The board buffer still says `H`; the tilemap says floor, and the tilemap is the truth."""
    board = synthetic_room({(1, 1): 0x96, (3, 3): 0x90, (6, 1): 0x00})
    tilemap = [list(row) for row in tilemap_of(decode_room(board), TILES)]
    tilemap[3][3] = 0xA7
    grid = decode_grid(board, TILES, tuple(tuple(row) for row in tilemap))
    assert grid[3][3] == FLOOR


def test_a_pushed_framer_is_read_where_it_now_stands():
    board = synthetic_room({(1, 1): 0x96, (3, 3): 0x8F, (6, 1): 0x00})
    tilemap = [list(row) for row in tilemap_of(decode_room(board), TILES)]
    tilemap[3][3], tilemap[3][4] = 0xA7, TILES.framer
    grid = decode_grid(board, TILES, tuple(tuple(row) for row in tilemap))
    assert grid[3][3] == FLOOR and grid[3][4] == FRAMER


def test_an_open_door_reads_as_open():
    board = synthetic_room({(1, 1): 0x96, (6, 1): 0x00})
    tilemap = [list(row) for row in tilemap_of(decode_room(board), TILES)]
    tilemap[1][1] = TILES.door_open
    grid = decode_grid(board, TILES, tuple(tuple(row) for row in tilemap))
    assert grid[1][1] == DOOR_OPEN


def test_which_kind_of_heart_a_surviving_one_is_comes_from_the_board():
    """Both draw the same tile, and hearts never move, so the static room is exact here."""
    board = synthetic_room({(1, 1): 0x96, (3, 3): 0x91, (3, 4): 0x90, (6, 1): 0x00})
    grid = decode_grid(board, TILES, tilemap_of(decode_room(board), TILES))
    assert grid[3][3] == MAGIC_HEART and grid[3][4] == HEART


def test_a_room_with_no_framer_decodes_anyway():
    """`learn_tiles` returns None for an object the room does not have; that is not an error."""
    tiles = TileMap(heart=0xE2, framer=None, door_closed=0x8C, door_open=0x8E)
    board = synthetic_room({(1, 1): 0x96, (6, 1): 0x00})
    grid = decode_grid(board, tiles, tilemap_of(decode_room(board), tiles))
    assert grid[1][1] == DOOR


def test_cell_of_puts_a_sprite_on_the_right_cell_and_keeps_the_halves():
    """A sprite draws 16 below and 8 right of where it sits; Lolo really stands on halves."""
    assert cell_of(24, 16) == (0.0, 0.0)
    assert cell_of(24 + 16 * 3, 16 + 16 * 5) == (3.0, 5.0)
    assert cell_of(24 + 8, 16) == (0.5, 0.0), "half-cells must survive, not round away"


def test_render_grid_draws_lolo_and_the_enemies_over_the_terrain():
    board = synthetic_room({(1, 1): 0x96, (6, 1): 0x00})
    grid = decode_grid(board, TILES, tilemap_of(decode_room(board), TILES))
    drawn = render_grid(grid, (6.0, 1.0), [(3.0, 4.0, "e")]).split("\n")
    assert drawn[6][1] == LOLO and drawn[3][4] == "e" and drawn[1][1] == DOOR


# ------------------------------------------------------------------------- actions

def test_the_action_set_is_the_d_pad_and_the_magic_shot():
    assert action_list == [f"left,{PRESS_TICKS}", f"up,{PRESS_TICKS}", f"down,{PRESS_TICKS}",
                           f"right,{PRESS_TICKS}", f"a,{SHOOT_TICKS}"]
    assert set(action_cost_map) == {"left", "right", "up", "down", "a", "nop"}


def test_an_action_parses_its_buttons_and_costs_them():
    action = LoloGBAction(f"right,{PRESS_TICKS}")
    assert action.cost() == 1
    assert str(action) == f"right_for_{PRESS_TICKS}"
    assert LoloGBAction("nop,1").cost() == 0


def test_button_actions_uses_whatever_calibration_settled_on():
    from planiverse.environments.gameboy.lolo_gb import Calibration

    assert button_actions(Calibration(24, (16, 28)))[0] == "left,24"
    assert button_actions(None) == action_list


# --------------------------------------------------------------- against the cartridge

@needs_rom
def test_the_rom_is_the_dump_these_addresses_came_from():
    import hashlib

    with open(lolo_rom_path(), "rb") as handle:
        assert hashlib.md5(handle.read()).hexdigest() == ROM_MD5


@needs_rom
def test_the_cartridge_holds_exactly_163_rooms():
    with open(lolo_rom_path(), "rb") as handle:
        rom = handle.read()
    assert verify_room_table(rom) == ()
    rooms = read_rooms(lolo_rom_path())
    assert len(rooms) == ROOM_COUNT
    assert all(len(row) == 8 for room in rooms for row in room)


@needs_rom
def test_a_wrong_rom_is_warned_about_not_refused():
    with pytest.warns(UserWarning, match="revision-specific"):
        LoloGBEnv(__file__)


@needs_rom
def test_fix_index_refuses_a_room_that_does_not_exist():
    env = LoloGBEnv(lolo_rom_path())
    with pytest.raises(IndexError):
        env.fix_index(ROOM_COUNT)


@needs_rom
def test_booting_reaches_the_room_it_was_asked_for():
    """The boot has to land in the graded rooms, not the tutorial's self-playing demos."""
    env = LoloGBEnv(lolo_rom_path())
    env.fix_index(38)
    try:
        state, info = env.reset()
        assert info["room"] == "int 1-1"
        assert info["hearts"] == 6 and info["door"] == (1, 1) and info["start"] == (6, 1)
        assert str(state).split("\n") == list(read_rooms(lolo_rom_path())[38])
        assert_string_literals(state)
    finally:
        env.close()


@needs_rom
def test_the_cartridge_agrees_with_the_rom_decoder_on_the_board_it_loaded():
    """The environment reads the room off the screen; `read_rooms` reads it out of bank 13."""
    env = LoloGBEnv(lolo_rom_path())
    env.fix_index(108)
    try:
        state, _ = env.reset()
        expected = read_rooms(lolo_rom_path())[108]
        # Enemies are sprites and print as `e` whatever kind they are, so compare everything
        # else cell by cell.
        for row in range(8):
            for col in range(8):
                if expected[row][col] in ENEMY_GLYPHS or expected[row][col] == LOLO:
                    continue
                assert str(state).split("\n")[row][col] == expected[row][col], \
                    f"cell ({row}, {col}) differs"
    finally:
        env.close()


@needs_rom
def test_successors_drop_the_actions_that_change_nothing():
    env = LoloGBEnv(lolo_rom_path())
    env.fix_index(38)
    try:
        state, _ = env.reset()
        successors = env.successors(state)
        assert_successors_contract(successors)
        # Lolo starts in the bottom-left corner: only up and right do anything, and there is
        # nothing to shoot at.
        assert sorted(str(action) for action, _ in successors) == \
            [f"right_for_{PRESS_TICKS}", f"up_for_{PRESS_TICKS}"]
    finally:
        env.close()


@needs_rom
def test_a_known_plan_clears_int_1_1_on_the_cartridge():
    env = LoloGBEnv(lolo_rom_path())
    env.fix_index(38)
    try:
        plan = ([f"right,{PRESS_TICKS}"] * 5 + [f"up,{PRESS_TICKS}"] * 2
                + [f"left,{PRESS_TICKS}"] * 2 + [f"up,{PRESS_TICKS}"] * 3
                + [f"right,{PRESS_TICKS}"] * 2 + [f"left,{PRESS_TICKS}"] * 5)
        trace = env.simulate(plan)
        assert env.is_goal(trace[-1])
        assert trace[-1].hearts_left == 0
        assert not any(state.died for state in trace)
    finally:
        env.close()


@needs_rom
def test_walking_into_a_medusas_line_is_noticed_as_a_death():
    """int 1-2's Medusa at (4, 6): entering column 6 is fatal on the action after."""
    env = LoloGBEnv(lolo_rom_path())
    env.fix_index(39)
    try:
        trace = env.simulate([f"right,{PRESS_TICKS}"] * 3 + [f"up,{PRESS_TICKS}"])
        assert not any(state.died for state in trace[:-1])
        assert trace[-1].died and env.is_terminal(trace[-1])
        assert env.successors(trace[-1]) == [], "a death is absorbing"
    finally:
        env.close()


@needs_rom
def test_a_magic_heart_framer_gives_two_shots_and_a_plain_one_gives_none():
    """The `$90`/`$91` split, on the cartridge. tutorial 1a has one of each."""
    env = LoloGBEnv(lolo_rom_path())
    env.fix_index(0)
    try:
        state, _ = env.reset()
        assert state.shots == 0
        # Lolo starts at (6, 5); the magic heart framer is three cells to his left.
        after = env.simulate([f"left,{PRESS_TICKS}"] * 3)[-1]
        assert after.hearts_left == 1 and after.shots == 2
    finally:
        env.close()


@needs_rom
def test_seeding_the_magic_shot_meter_works():
    env = LoloGBEnv(lolo_rom_path(), magic_shots=2)
    env.fix_index(38)
    try:
        state, info = env.reset()
        assert info["shots"] == 2 and state.shots == 2
    finally:
        env.close()


@needs_rom
def test_the_d_pad_hold_that_moves_one_cell_is_still_the_documented_one():
    """`PRESS_TICKS` is a measurement, not a preference. This is the measurement."""
    env = LoloGBEnv(lolo_rom_path(), calibrate=True)
    env.fix_index(38)
    try:
        _, info = env.reset()
        low, high = info["calibration"].hold_window
        assert low <= PRESS_TICKS <= high, \
            f"PRESS_TICKS={PRESS_TICKS} is outside the cartridge's one-cell band {(low, high)}"
        # Lolo covers half a cell per sixteen frames of hold, so the band is about that wide.
        assert 12 <= high - low <= 20, f"one-cell band {(low, high)} is not a half-cell wide"
    finally:
        env.close()
