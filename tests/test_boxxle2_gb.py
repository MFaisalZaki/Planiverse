"""Tests for the Boxxle II Game Boy environment.

Two tiers. The first needs neither a ROM nor an emulator: almost everything this environment
does is a pure function of bytes (the ROM's two-stage level decompressor, the board decoder
over the three plane buffers, the deadlock test), and those run against synthetic RAM and a
synthetic ROM image assembled here in the cartridge's own format. That is a different bargain
from the Puzznic and Flipull tests, which build a small homebrew cartridge to exercise the
emulator seam; here the emulator seam is thin and the decoding is where the errors would live.

The second tier boots the real cartridge, which is copyrighted and cannot ship here, so it is
opt-in:

    PLANIVERSE_BOXXLE2_ROM="/path/to/Boxxle II (USA, Europe).gb" poetry run pytest tests/test_boxxle2_gb.py
"""
import pytest

pytest.importorskip("pyboy", reason="pyboy is not installed")

from planiverse.environments.gameboy.boxxle2_gb import (  # noqa: E402
    BOX, BOX_ON_GOAL, FLOOR, GOAL, LEVEL_COUNT, LEVEL_TABLE_ADDR, LEVELS_PER_STAGE,
    PLANE_BYTES, PLAYER, PLAYER_ON_GOAL, ROM_MD5, ROW_STRIDE, STATE_ADVANCED_BY_START,
    STATE_MENU, STATE_MUSIC, STATE_PLAYING, STATE_TITLE, WALL, Boxxle2GBAction, Boxxle2GBEnv,
    Calibration, LEAD_IN_TICKS, PRESS_TICKS, Position, SETTLE_MAX_TICKS, SETTLE_MIN_TICKS,
    SETTLE_STABLE_TICKS, action_cost_map, action_list, boxes, boxes_home, button_actions,
    decode_board, decode_level, expand_record, goals, is_solved, level_pointers,
    offset_to_position, open_direction, position_to_offset, read_levels, render_grid,
    stuck_boxes, unpack_planes, verify_level_table,
)

from conftest import (  # noqa: E402
    assert_state_contract, assert_string_literals, assert_successors_contract, boxxle2_rom_path,
)

needs_rom = pytest.mark.skipif(
    boxxle2_rom_path() is None,
    reason='set PLANIVERSE_BOXXLE2_ROM to a "Boxxle II (USA, Europe).gb" ROM to run '
           "cartridge tests")


# --------------------------------------------------------------- a synthetic ROM image
# Not a runnable cartridge: nothing here is executed. It is the cartridge's *level format*,
# written out by hand, so the decoder can be tested against records whose contents are known
# rather than only against the 120 nobody wrote down.

def encode_level(width, height, start_col, start_row, goal, box, wall):
    """A level record in the cartridge's format: header, flag bitmap, literals.

    The inverse of `expand_record` + `unpack_planes`, so a round trip through the two says
    the decoder reads what the encoder wrote, and the encoder is short enough to check by
    eye against the memory map.
    """
    bits = []
    for plane in (goal, box, wall):
        for row in range(height):
            for col in range(width):
                bits.append(plane[row][col])
    stream = bytearray(-(-len(bits) // 8))
    for index, bit in enumerate(bits):
        if bit:
            stream[index >> 3] |= 0x80 >> (index & 7)

    flags = bytearray(-(-(3 * width * height) // 64))
    literals = bytearray()
    for index, byte in enumerate(stream):
        if byte:
            flags[index >> 3] |= 0x80 >> (index & 7)
            literals.append(byte)
    return bytes([width, height, start_col, start_row]) + bytes(flags) + bytes(literals)


def synthetic_rom(records):
    """A 32 KiB image with `records` behind a pointer table at `$4E18`."""
    rom = bytearray(32768)
    offset = 0x4F23                              # where the cartridge puts its first record
    for index, record in enumerate(records):
        rom[LEVEL_TABLE_ADDR + 2 * index] = offset & 0xFF
        rom[LEVEL_TABLE_ADDR + 2 * index + 1] = offset >> 8
        rom[offset:offset + len(record)] = record
        offset += len(record)
    # The table is 120 words wide whatever it holds; the unused entries point at the last
    # record, which is what makes `verify_level_table` complain about them and nothing else.
    for index in range(len(records), LEVEL_COUNT):
        rom[LEVEL_TABLE_ADDR + 2 * index] = offset & 0xFF
        rom[LEVEL_TABLE_ADDR + 2 * index + 1] = offset >> 8
    return bytes(rom)


def planes_of(rows):
    """`(goal, box, wall)` bit planes from an ASCII board."""
    height, width = len(rows), max(len(row) for row in rows)
    goal = [[0] * width for _ in range(height)]
    box = [[0] * width for _ in range(height)]
    wall = [[0] * width for _ in range(height)]
    for row, line in enumerate(rows):
        for col, glyph in enumerate(line.ljust(width)):
            goal[row][col] = int(glyph in (GOAL, BOX_ON_GOAL, PLAYER_ON_GOAL))
            box[row][col] = int(glyph in (BOX, BOX_ON_GOAL))
            wall[row][col] = int(glyph == WALL)
    return goal, box, wall


def raw_planes(rows):
    """The three 360-byte buffers the console holds, from an ASCII board."""
    goal, box, wall = planes_of(rows)
    buffers = []
    for plane in (goal, box, wall):
        raw = bytearray(PLANE_BYTES)
        for row, cells in enumerate(plane):
            for col, bit in enumerate(cells):
                raw[row * ROW_STRIDE + col] = bit
        buffers.append(bytes(raw))
    return buffers


SAMPLE = ("#####",
          "#@$o#",
          "#   #",
          "#####")


# --------------------------------------------------------------------- the level format

def test_a_record_round_trips_through_the_decoder():
    goal, box, wall = planes_of(SAMPLE)
    record = encode_level(5, 4, 2, 2, goal, box, wall)     # start is stored 1-based
    rows, size = decode_level(synthetic_rom([record]), 0x4F23)
    assert rows == SAMPLE
    assert size == len(record)


def test_the_record_size_is_what_locates_the_next_level():
    """It is implied by W and H, not stored, so a decoder that gets it wrong loses the table."""
    first = encode_level(5, 4, 2, 2, *planes_of(SAMPLE))
    second = encode_level(5, 4, 2, 2, *planes_of(SAMPLE))
    rom = synthetic_rom([first, second])
    pointers = level_pointers(rom)
    assert pointers[1] - pointers[0] == len(first)
    # Both real records check out; the padding entries after them do not, which is what
    # `verify_level_table` is for.
    mismatched = verify_level_table(rom)
    assert 0 not in mismatched and 1 not in mismatched and mismatched[0] == 2


def test_expand_record_emits_a_zero_for_every_clear_flag_bit():
    # 3*2*2 = 12 bits -> ceil(12/8) = 2 output bytes -> ceil(12/64) = 1 flag byte.
    rom = bytes([0b01000000, 0xAB]) + bytes(16)
    data, size = expand_record(rom, 0, 2, 2)
    assert data == b"\x00\xab"
    assert size == 4 + 1 + 1


def test_unpack_planes_reads_one_continuous_bitstream():
    """W bits per row, H rows, three planes, MSB first, with no padding between them."""
    # 2x2, so 4 bits per plane and 12 bits in all: goals 1000, boxes 0100, walls 0011.
    stream = bytes([0b10000100, 0b00110000])
    goal, box, wall = unpack_planes(stream, 2, 2)
    assert goal == ((1, 0), (0, 0))
    assert box == ((0, 1), (0, 0))
    assert wall == ((0, 0), (1, 1))


def test_a_keeper_on_a_goal_is_drawn_as_plus():
    rows = ("###", "#+#", "###")
    record = encode_level(3, 3, 2, 2, *planes_of(rows))
    assert decode_level(synthetic_rom([record]), 0x4F23)[0] == rows


# --------------------------------------------------------------------- the board decoder

def test_decode_board_reads_the_three_planes():
    goal, box, wall = raw_planes(SAMPLE)
    grid = decode_board(goal, box, wall, 5, 4, position_to_offset(Position(1, 1)))
    assert render_grid(grid) == "\n".join(SAMPLE)


def test_decode_board_uses_a_stride_of_twenty():
    """Not the hardware tilemap's 32. Reading it as 32 silently shears every board."""
    goal, box, wall = raw_planes(SAMPLE)
    assert wall[ROW_STRIDE * 1 + 0] == 1 and wall[ROW_STRIDE * 1 + 1] == 0


def test_offsets_and_positions_round_trip():
    for offset in (0, 1, 19, 20, 23, 359):
        assert position_to_offset(offset_to_position(offset)) == offset
    assert offset_to_position(23) == Position(1, 3)


def test_boxes_and_goals_are_read_off_the_grid():
    goal, box, wall = raw_planes(("####", "#$o#", "####"))
    grid = decode_board(goal, box, wall, 4, 3, position_to_offset(Position(0, 0)))
    assert boxes(grid) == (Position(1, 1),)
    assert goals(grid) == (Position(1, 2),)
    assert boxes_home(grid) == 0


def test_a_box_on_a_goal_counts_as_home():
    goal, box, wall = raw_planes(("####", "#*@#", "####"))
    grid = decode_board(goal, box, wall, 4, 3, position_to_offset(Position(1, 2)))
    assert boxes_home(grid) == 1 and is_solved(grid)


def test_an_empty_board_is_not_solved():
    """A wiped plane buffer satisfies "no box is off a goal" vacuously, and must not win.

    This is the state the cartridge leaves behind while it runs its level-cleared sequence,
    so without the box-count guard every finished level would be followed by a false one.
    """
    goal, box, wall = raw_planes(("####", "#  #", "####"))
    grid = decode_board(goal, box, wall, 4, 3, position_to_offset(Position(1, 1)))
    assert not is_solved(grid)


# ------------------------------------------------------------------------- dead ends

def test_a_box_in_a_corner_off_a_goal_is_stuck():
    goal, box, wall = raw_planes(("####", "#$ #", "# @#", "####"))
    grid = decode_board(goal, box, wall, 4, 4, position_to_offset(Position(2, 2)))
    assert stuck_boxes(grid) == (Position(1, 1),)


def test_a_box_in_a_corner_on_a_goal_is_not_stuck():
    goal, box, wall = raw_planes(("####", "#* #", "# @#", "####"))
    grid = decode_board(goal, box, wall, 4, 4, position_to_offset(Position(2, 2)))
    assert stuck_boxes(grid) == ()


def test_a_box_against_one_wall_is_not_stuck():
    """Sound before complete: this box can still be pushed along the wall."""
    goal, box, wall = raw_planes(("#####", "#@$ #", "#  o#", "#####"))
    grid = decode_board(goal, box, wall, 5, 4, position_to_offset(Position(1, 1)))
    assert stuck_boxes(grid) == ()


# --------------------------------------------------------------------------- actions

def test_the_action_set_is_the_d_pad_and_nothing_else():
    """No undo, no restart, no START. Undo would make every dead end escapable."""
    assert {action.split(",")[0] for action in action_list} == {"left", "up", "down", "right"}


def test_every_action_costs_one():
    for action in action_list:
        assert Boxxle2GBAction(action).cost() == 1
    assert action_cost_map["nop"] == 0


def test_button_actions_use_the_measured_hold():
    assert button_actions(Calibration(7, (1, 18))) == ["left,7", "up,7", "down,7", "right,7"]


def test_an_action_spells_itself_readably():
    assert str(Boxxle2GBAction("left,9")) == "left_for_9"


def test_the_probe_only_measures_where_there_is_room_to_measure():
    """A direction with one clear cell reports every hold as a single move, so a window as
    wide as the probe comes back, which is how this cartridge once claimed `(1, 40)` for a
    d-pad that repeats on frame 20."""
    goal, box, wall = raw_planes(("#####", "#@  #", "#####"))
    grid = decode_board(goal, box, wall, 5, 3, position_to_offset(Position(1, 1)))
    assert open_direction(grid, Position(1, 1)) == "right"

    goal, box, wall = raw_planes(("####", "#@ #", "####"))
    boxed_in = decode_board(goal, box, wall, 4, 3, position_to_offset(Position(1, 1)))
    assert open_direction(boxed_in, Position(1, 1)) is None


def test_a_box_does_not_count_as_room_for_the_probe():
    """The keeper would push it, which measures something else entirely."""
    goal, box, wall = raw_planes(("#####", "#@$ #", "#####"))
    grid = decode_board(goal, box, wall, 5, 3, position_to_offset(Position(1, 1)))
    assert open_direction(grid, Position(1, 1)) is None


def test_calibration_falls_back_rather_than_inventing_a_number():
    assert button_actions(Calibration(PRESS_TICKS, None)) == button_actions(None)


def test_a_press_waits_out_the_frame_the_rewind_costs():
    """`load_state` ticks one frame; on some states that is one short of the pad being read,
    and the press is silently dropped. Two idle frames is what the cartridge needs."""
    assert LEAD_IN_TICKS == 2


def test_settle_is_held_open_past_the_auto_repeat_frame():
    """The slide is 16 frames and the d-pad repeats on the 20th; a board that stops changing
    before then has either paused mid-slide or is about to move again."""
    assert SETTLE_MIN_TICKS > 20
    assert SETTLE_STABLE_TICKS >= 1 and SETTLE_MAX_TICKS > SETTLE_MIN_TICKS


def test_start_is_only_pressed_on_the_menu_screens():
    """START during play opens the pause overlay, which is why booting checks the state."""
    assert STATE_PLAYING not in STATE_ADVANCED_BY_START
    assert set(STATE_ADVANCED_BY_START) == {STATE_TITLE, STATE_MUSIC, STATE_MENU}


# ------------------------------------------------------- the cartridge, when a ROM is here

@pytest.fixture(scope="module")
def rom():
    return boxxle2_rom_path()


@pytest.fixture
def env(rom):
    game = Boxxle2GBEnv(rom)
    game.fix_index(0)
    yield game
    game.close()


@needs_rom
def test_the_rom_is_the_dump_these_addresses_came_from(rom):
    import hashlib

    with open(rom, "rb") as handle:
        assert hashlib.md5(handle.read()).hexdigest() == ROM_MD5


@needs_rom
def test_the_level_table_decodes_without_a_single_mismatch(rom):
    """Every record's computed size lands exactly on the next record's start address."""
    with open(rom, "rb") as handle:
        assert verify_level_table(handle.read()) == ()


@needs_rom
def test_every_level_has_as_many_goals_as_boxes(rom):
    """popcount(plane 0) == popcount(plane 1): the invariant that pins the plane order."""
    for index, rows in enumerate(read_levels(rom)):
        text = "".join(rows)
        assert text.count(BOX) + text.count(BOX_ON_GOAL) == \
            text.count(GOAL) + text.count(BOX_ON_GOAL) + text.count(PLAYER_ON_GOAL), \
            f"level {index} is not balanced"


@needs_rom
def test_reset_loads_the_level_it_was_asked_for(env):
    state, info = env.reset()
    assert info["level_index"] == 0 and info["level"] == "1-01"
    assert info["stage"] == 0 and info["level_in_stage"] == 0
    assert (state.width, state.height) == (9, 8)
    assert len(state.boxes) == 3 and state.boxes_home == 0
    assert state.is_consistent()


@needs_rom
def test_the_loader_hook_reaches_a_late_level(rom):
    game = Boxxle2GBEnv(rom, calibrate=False)
    game.fix_index(119)
    try:
        state, info = game.reset()
        assert info["level"] == "12-10"
        assert info["stage"] == 11 and info["level_in_stage"] == 9
        assert state.is_consistent()
    finally:
        game.close()


@needs_rom
def test_the_board_read_from_ram_matches_the_board_decoded_from_the_rom(rom):
    """The check that verifies the plane addresses, the stride and the hook, all at once."""
    levels = read_levels(rom)
    game = Boxxle2GBEnv(rom, calibrate=False)
    try:
        for index in (0, 25, 63, 99, 119):
            game.fix_index(index)
            state, _ = game.reset()
            assert render_grid(state.grid) == "\n".join(row.rstrip() for row in levels[index]), \
                f"level {index} differs between work RAM and the ROM"
    finally:
        game.close()


@needs_rom
def test_calibration_finds_the_hold_window(env):
    _, info = env.reset()
    calibration = info["calibration"]
    low, high = calibration.hold_window
    assert low == 1 and 15 <= high <= 19, "the d-pad's auto-repeat moved"
    assert low <= calibration.press_ticks <= high


@needs_rom
def test_state_literals_are_strings(env):
    state, _ = env.reset()
    assert_string_literals(state)
    assert_state_contract(state)
    assert f"at(player, {state.player.row}, {state.player.col})" in state.literals


@needs_rom
def test_successors_drop_the_moves_a_wall_refuses(env):
    state, _ = env.reset()
    successors = env.successors(state)
    assert_successors_contract(successors)
    # There is a wall directly above the keeper on level 1-01.
    assert {str(action).split("_for_")[0] for action, _ in successors} == {"left", "down", "right"}
    assert all(successor != state for _, successor in successors)


@needs_rom
def test_a_move_and_its_reverse_come_back_to_the_same_position(env):
    """Search can only close if identity excludes the route taken."""
    state, _ = env.reset()
    hold = env.calibration.press_ticks
    there = env.__advance__(state, f"down,{hold}")
    back = env.__advance__(there, f"up,{hold}")
    assert there != state
    assert back == state and hash(back) == hash(state)


@needs_rom
def test_a_push_moves_a_box(env):
    state, _ = env.reset()
    hold = env.calibration.press_ticks
    trace = env.simulate([f"down,{hold}"] * 3)
    assert trace[-1].player == Position(4, 3)
    assert Position(5, 3) in trace[-1].boxes, "the box the keeper walked into did not move"


@needs_rom
def test_the_cartridge_and_the_python_twin_agree_move_for_move(rom):
    """The strongest claim either environment makes, so it is checked rather than asserted."""
    import random

    from planiverse.environments.gameboy_py.boxxle2 import Boxxle2Game

    rng = random.Random(7)
    cartridge = Boxxle2GBEnv(rom)
    twin = Boxxle2Game()
    try:
        for index in (0, 33, 66, 99, 119):
            cartridge.fix_index(index)
            here, _ = cartridge.reset()
            twin.fix_index(index)
            there, _ = twin.reset()
            hold = cartridge.calibration.press_ticks
            assert str(here) == str(there), f"level {index} differs before a move is made"
            for step in range(20):
                direction = rng.choice(["left", "up", "down", "right"])
                here = cartridge.__advance__(here, f"{direction},{hold}")
                there = twin.__advance__(there, direction)
                assert str(here) == str(there), \
                    f"level {index} diverged at step {step} after {direction}"
    finally:
        cartridge.close()


@needs_rom
def test_a_solved_level_is_absorbing_and_is_read_before_the_cartridge_wipes_it(rom):
    """Clearing a level starts a sequence that overwrites the planes with a non-position."""
    from planiverse.environments.gameboy_py.boxxle2 import Boxxle2Game

    plan = _plan_for(0)
    if plan is None:
        pytest.skip("level 0 has no stored solution to replay")
    twin = Boxxle2Game()
    twin.fix_index(0)
    twin.reset()
    assert twin.validate(plan), "the stored plan no longer solves level 0 in the twin"

    game = Boxxle2GBEnv(rom)
    try:
        game.fix_index(0)
        game.reset()
        hold = game.calibration.press_ticks
        trace = game.simulate([f"{direction},{hold}" for direction in plan])
        assert game.is_goal(trace[-1]), "the cartridge did not report level 0 cleared"
        assert trace[-1].is_consistent(), "the board was read after the clear sequence ran"
        assert game.__advance__(trace[-1], f"left,{hold}") is trace[-1]
    finally:
        game.close()


def _plan_for(index):
    """The stored solution for a level, as direction names, or None."""
    import json
    import os

    path = os.path.join(os.path.dirname(__file__), "data", "boxxle2_solutions.json")
    with open(path) as handle:
        return json.load(handle).get(str(index))


def test_the_level_count_is_twelve_stages_of_ten():
    assert LEVEL_COUNT == 120 == 12 * LEVELS_PER_STAGE


def test_fix_index_rejects_a_level_that_does_not_exist(tmp_path):
    game = Boxxle2GBEnv(str(tmp_path / "absent.gb"), verify_rom=False)
    with pytest.raises(AssertionError):
        game.fix_index(LEVEL_COUNT)


def test_a_missing_rom_is_not_hashed(tmp_path):
    """Constructing an environment must not require the cartridge to be there yet."""
    Boxxle2GBEnv(str(tmp_path / "absent.gb"))     # warns about nothing, raises nothing


def test_the_wrong_dump_warns(tmp_path):
    cartridge = tmp_path / "not-boxxle2.gb"
    cartridge.write_bytes(bytes(32768))
    with pytest.warns(UserWarning, match="revision-specific"):
        Boxxle2GBEnv(str(cartridge))


def test_the_glyph_alphabet_is_the_one_the_twin_uses():
    from planiverse.environments.gameboy_py import boxxle2 as twin

    assert (WALL, BOX, GOAL, BOX_ON_GOAL, PLAYER, PLAYER_ON_GOAL, FLOOR) == \
        (twin.WALL, twin.BOX, twin.GOAL, twin.BOX_ON_GOAL, twin.PLAYER, twin.PLAYER_ON_GOAL,
         twin.FLOOR)
