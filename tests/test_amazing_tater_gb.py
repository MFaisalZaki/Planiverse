"""Tests for the Amazing Tater Game Boy environment.

Two tiers. The first needs neither PyBoy nor a cartridge: almost everything this environment
does before it touches the emulator is a pure function of bytes (the cell-code decoder, the
board cropper, the ROM's own pointer tables), and those are checked against synthetic buffers
and a synthetic ROM image, which is cheaper and covers more than driving the machine would.

The second needs the real cartridge, which is copyrighted and cannot ship here, so it is
opt-in:

    PLANIVERSE_AMAZING_TATER_ROM="/path/to/Amazing Tater (U).gb" \\
        poetry run pytest tests/test_amazing_tater_gb.py
"""
import pytest

pytest.importorskip("pyboy", reason="pyboy is not installed")

from planiverse.environments.gameboy.amazing_tater_gb import (  # noqa: E402
    ARM_CODES, ARM_MASKS, ARM_OVER_PIT_CODES, BLOCK_CODES, BOARD_BYTES, CODE_FLOOR,
    CODE_OUTSIDE, CODE_PIT, EXIT_CODES, GLYPH_BY_CODE, LEVEL_COUNT, LEVEL_SETS, MODE_MENU_ROW,
    PIVOT_CODES, ROM_MD5, ROW_STRIDE, SET_COUNTS, SETTLED_BLOCK_CODES, SHAPE_TABLE,
    SHAPE_TABLE_ADDR, SWITCH, TATER_CODES, WALL_CODES, AmazingTaterGBAction,
    AmazingTaterGBEnv, Calibration, action_cost_map, action_list, board_bounds, button_actions,
    cell_glyph, decode_board, find, is_solved, level_counts, read_rom, render_board,
    set_descriptors, shape_table, taters,
)
from planiverse.environments.gameboy_py import amazing_tater as twin  # noqa: E402

from conftest import (  # noqa: E402
    amazing_tater_rom_path, assert_string_literals, assert_successors_contract,
)

needs_rom = pytest.mark.skipif(
    amazing_tater_rom_path() is None,
    reason='set PLANIVERSE_AMAZING_TATER_ROM to an "Amazing Tater (U).gb" ROM to run '
           "cartridge tests",
)


def buffer_of(rows, top=0, left=0):
    """A 20x18 board buffer with `rows` of cell codes stamped into it."""
    buffer = bytearray([CODE_OUTSIDE] * BOARD_BYTES)
    for row, codes in enumerate(rows):
        for col, code in enumerate(codes):
            buffer[(top + row) * ROW_STRIDE + left + col] = code
    return bytes(buffer)


# ------------------------------------------------------------------------- the alphabet

def test_the_two_modules_agree_on_the_alphabet():
    """The glyphs come from the twin, so this is really a check that nothing shadowed them."""
    assert GLYPH_BY_CODE[CODE_FLOOR] == twin.FLOOR
    assert GLYPH_BY_CODE[CODE_PIT] == twin.PIT
    assert GLYPH_BY_CODE[CODE_OUTSIDE] == twin.OUTSIDE
    assert {GLYPH_BY_CODE[code] for code in WALL_CODES} == {twin.WALL}
    assert {GLYPH_BY_CODE[code] for code in PIVOT_CODES} == {twin.PIVOT}
    assert [GLYPH_BY_CODE[code] for code in BLOCK_CODES] == list(twin.BLOCK_GLYPHS)
    assert [GLYPH_BY_CODE[code] for code in SETTLED_BLOCK_CODES] == list(twin.SETTLED_GLYPHS)
    assert [GLYPH_BY_CODE[code] for code in ARM_CODES] == list(twin.ARM_GLYPHS)
    assert [GLYPH_BY_CODE[code] for code in ARM_OVER_PIT_CODES] == \
        list(twin.ARM_OVER_PIT_GLYPHS)
    assert [GLYPH_BY_CODE[code] for code in TATER_CODES] == list(twin.TATER_GLYPHS)


def test_every_glyph_but_wall_and_pivot_names_exactly_one_code():
    """Fifteen wall graphics and fifteen turnstile shapes collapse; nothing else may.

    The collapse is deliberate: which wall graphic a wall uses and which shape a pivot is
    change nothing a tater can do, and a pivot's arms are written out beside it anyway. Every
    other code has to survive the trip, or a board could not be read back.
    """
    collapsed = set(WALL_CODES) | set(PIVOT_CODES)
    distinct = [glyph for code, glyph in GLYPH_BY_CODE.items() if code not in collapsed]
    assert len(distinct) == len(set(distinct))


def test_an_unknown_cell_code_is_reported_rather_than_guessed():
    with pytest.raises(ValueError):
        cell_glyph(0x77)


# --------------------------------------------------------------------- decoding a board

def test_a_board_is_cropped_to_the_room():
    buffer = buffer_of([[0xF0, 0xF0, 0xF0],
                        [0xF0, 0x00, 0xF0],
                        [0xF0, 0xF0, 0xF0]], top=6, left=4)
    assert board_bounds(buffer) == (6, 4, 8, 6)
    assert decode_board(buffer) == ("###", "#.#", "###")


def test_an_empty_buffer_decodes_to_nothing():
    assert board_bounds(bytes([CODE_OUTSIDE]) * BOARD_BYTES) is None
    assert decode_board(bytes([CODE_OUTSIDE]) * BOARD_BYTES) == ()


def test_a_block_square_carries_which_neighbours_share_its_block():
    """`$40 | mask`, with 1 right, 2 down, 4 left, 8 up. A 2x2 reads d/g over j/m."""
    buffer = buffer_of([[0xF0, 0xF0, 0xF0, 0xF0],
                        [0xF0, 0x43, 0x46, 0xF0],
                        [0xF0, 0x49, 0x4C, 0xF0],
                        [0xF0, 0xF0, 0xF0, 0xF0]], top=5, left=5)
    assert decode_board(buffer) == ("####", "#dg#", "#jm#", "####")
    # And the twin recovers one block from it, not four.
    level = twin.Level(0, ("####", "#dg#", "#jm#", "#1E#"))
    assert len(level.start_blocks) == 1


def test_a_turnstile_pivot_carries_its_shape_and_its_arms_their_direction():
    buffer = buffer_of([[0xF0, 0xF0, 0xF0, 0xF0],
                        [0xF0, 0x80, 0x00, 0xF0],       # arm pointing up
                        [0xF0, 0xA4, 0x81, 0xF0],       # pivot with arms up and right
                        [0xF0, 0xF0, 0xF0, 0xF0]], top=5, left=5)
    assert decode_board(buffer) == ("####", "#^.#", "#@>#", "####")
    assert ARM_MASKS[0xA4 - PIVOT_CODES.start] == 0b1100      # up and right


def test_a_square_over_a_pit_decodes_to_the_settled_alphabet():
    buffer = buffer_of([[0xE0, 0x50, 0x90]], top=8, left=8)
    assert decode_board(buffer) == ("OAU",)


def test_finding_taters_and_the_flag():
    buffer = buffer_of([[0xC0, 0x00, 0xD0, 0xC3]], top=8, left=8)
    assert {who: (cell.row, cell.col) for who, cell in taters(buffer).items()} == \
        {0: (8, 8), 3: (8, 11)}
    assert [(cell.row, cell.col) for cell in find(buffer, EXIT_CODES)] == [(8, 10)]


def test_render_board_prints_the_friendly_view():
    assert render_board(("#dg#", "#^@#")) == "#$$#\n#+o#"


# ------------------------------------------------------------------------- the ROM tables

def synthetic_rom():
    """A 64 KiB image with the two tables this module reads off a cartridge, and nothing else.

    Cheaper than a real ROM and it fails for the right reason: if `set_descriptors` or
    `level_counts` stops reading what it claims to read, this notices without anyone having to
    own a cartridge.
    """
    rom = bytearray(0xFF for _ in range(65536))
    words = [(0x5000, 0x5052, 0x50A4), (0x5CE2, 0x5DA2, 0x5E62), (0x686F, 0x68EF, 0x696F)]
    for index, word in enumerate(word for triple in words for word in triple):
        rom[0x0C05 + 2 * index] = word & 0xFF
        rom[0x0C05 + 2 * index + 1] = word >> 8
    rom[SHAPE_TABLE_ADDR:SHAPE_TABLE_ADDR + 15] = bytes(SHAPE_TABLE)
    return bytes(rom)


def test_the_set_descriptors_read_back_as_three_triples():
    assert set_descriptors(synthetic_rom()) == ((0x5000, 0x5052, 0x50A4),
                                                (0x5CE2, 0x5DA2, 0x5E62),
                                                (0x686F, 0x68EF, 0x696F))


def test_level_counts_are_derived_from_the_pointer_arrays():
    """41, 96 and 64 are not constants here; they are `(plane_a - records) / 2`."""
    assert level_counts(synthetic_rom()) == SET_COUNTS == (41, 96, 64)


def test_the_shape_table_is_the_fifteen_non_empty_arm_masks():
    assert len(SHAPE_TABLE) == 15
    assert sorted(ARM_MASKS) == list(range(1, 16))
    assert shape_table(synthetic_rom()) == SHAPE_TABLE


def test_the_two_sets_this_environment_offers_add_up():
    assert LEVEL_COUNT == 105
    assert [size for _letter, _mode, size in LEVEL_SETS] == [41, 64]
    assert LEVEL_COUNT == len(twin.LEVELS)
    assert set(mode for _letter, mode, _size in LEVEL_SETS) <= set(MODE_MENU_ROW)


# -------------------------------------------------------------------------- the actions

def test_the_action_vocabulary_is_the_d_pad_and_select():
    assert sorted(name for name in action_cost_map if name != "nop") == \
        ["down", "left", "right", SWITCH, "up"]
    assert action_cost_map[SWITCH] == 0
    assert "a" not in action_cost_map          # the pause menu, deliberately absent


def test_button_actions_use_the_calibrated_hold():
    assert button_actions(Calibration(7, (1, 9))) == \
        ["up,7", "right,7", "down,7", "left,7", "switch,7"]


def test_an_action_parses_and_costs_what_it_says():
    action = AmazingTaterGBAction("left,5")
    assert action.cost() == 1 and str(action) == "left_for_5"
    assert AmazingTaterGBAction("switch,5").cost() == 0


def test_the_default_action_list_covers_every_button():
    assert len(action_list) == 5


# -------------------------------------------------------------------------- the cartridge

@pytest.fixture(scope="module")
def env():
    game = AmazingTaterGBEnv(amazing_tater_rom_path())
    yield game
    game.close()


@needs_rom
def test_the_rom_is_the_revision_these_addresses_came_from():
    import hashlib
    with open(amazing_tater_rom_path(), "rb") as handle:
        assert hashlib.md5(handle.read()).hexdigest() == ROM_MD5


@needs_rom
def test_the_tables_read_off_the_real_cartridge_match_the_constants():
    rom = read_rom(amazing_tater_rom_path())
    assert level_counts(rom) == SET_COUNTS
    assert shape_table(rom) == SHAPE_TABLE


@needs_rom
def test_booting_reaches_the_first_room(env):
    env.set_index(0)
    state, info = env.reset()
    assert info["level"] == "A-01" and info["mode"] == 0
    assert info["size"] == (15, 5)
    assert state.is_consistent()
    assert_string_literals(state)
    assert not env.is_goal(state)


@needs_rom
def test_the_board_read_off_the_cartridge_is_the_room_the_twin_ships(env):
    env.set_index(0)
    state, _ = env.reset()
    assert state.rows == tuple(row.rstrip() for row in twin.LEVELS[0])


@needs_rom
def test_calibration_finds_the_hold_that_moves_exactly_one_cell(env):
    env.set_index(0)
    _, info = env.reset()
    calibration = info["calibration"]
    assert calibration.hold_window is not None
    low, high = calibration.hold_window
    assert low <= calibration.press_ticks <= high
    assert high < 12                     # past this the d-pad repeats and one press walks two


@needs_rom
def test_the_room_is_waited_out_before_the_first_press(env):
    """The loader fills the board before the room listens; `reset` reports how long it took."""
    env.set_index(0)
    _, info = env.reset()
    assert info["intro_ticks"] is not None and info["intro_ticks"] > 0


@needs_rom
def test_successors_are_the_moves_the_cartridge_accepts(env):
    env.set_index(0)
    state, _ = env.reset()
    successors = env.successors(state)
    assert_successors_contract(successors)
    assert {str(action).split("_")[0] for action, _ in successors} == \
        {"up", "down", "left", "right"}          # one tater, so no switch


@needs_rom
def test_a_room_with_a_walled_in_tater_still_offers_the_switch(env):
    """A-14 opens with its first tater boxed in, which is why SELECT has to be an action."""
    env.set_index(13)
    state, _ = env.reset()
    assert state.total == 2
    assert SWITCH in {str(action).split("_")[0] for action, _ in env.successors(state)}


@needs_rom
def test_two_switches_in_a_row_both_land(env):
    """The cartridge ignores anything pressed within 33 frames of a SELECT, and nothing on
    the board says so, which silently turned every second switch into a no-op."""
    env.set_index(13)
    state, _ = env.reset()
    once = env.__advance__(state, f"{SWITCH},5")
    twice = env.__advance__(once, f"{SWITCH},5")
    assert once.active != state.active
    assert twice.active == state.active


@needs_rom
def test_the_dumped_rooms_are_the_ones_the_twin_ships(env):
    for index in (0, 20, 40, 41, 80, 104):
        assert env.__dump__(index) == tuple(row.rstrip() for row in twin.LEVELS[index])


@needs_rom
def test_reaching_the_flag_solves_the_room(env):
    """`is_goal` reads `$C2AD`, not the board: a tater in mid-step is off the board entirely."""
    plan = twin.solve(1)
    env.set_index(1)
    state, _ = env.reset()
    for name in plan[:-1]:
        state = env.__advance__(state, f"{name},5")
    assert not env.is_goal(state)
    state = env.__advance__(state, f"{plan[-1]},5")
    assert env.is_goal(state)
    assert state.taters_home == state.total


@needs_rom
def test_a_solved_room_is_absorbing(env):
    env.set_index(1)
    state, _ = env.reset()
    for name in twin.solve(1):
        state = env.__advance__(state, f"{name},5")
    assert env.is_goal(state)
    assert env.successors(state) == []


@needs_rom
def test_label_for_names_a_room_the_way_the_cartridge_does(env):
    assert env.label_for(0) == "A-01"
    assert env.label_for(40) == "A-41"
    assert env.label_for(41) == "C-01"
    assert env.label_for(104) == "C-64"


@needs_rom
def test_set_index_rejects_a_room_that_is_not_there(env):
    with pytest.raises(AssertionError):
        env.set_index(LEVEL_COUNT)


@needs_rom
def test_a_position_reached_two_ways_compares_equal(env):
    env.set_index(0)
    state, _ = env.reset()
    there = env.__advance__(state, "left,5")
    back = env.__advance__(there, "right,5")
    assert back == state and hash(back) == hash(state)
