"""Tests for the Flipull Game Boy environment.

Two tiers, the same arrangement as Puzznic. The first runs against a synthetic cartridge
built by `fake_flipull_rom.py`, which puts Flipull's documented facts at Flipull's
documented addresses without being Flipull. The second needs the real cartridge, which is
copyrighted and cannot ship here, so it is opt-in:

    PLANIVERSE_FLIPULL_ROM="/path/to/Flipull (USA).gb" poetry run pytest tests/test_flipull_gb.py
"""
import pytest

pytest.importorskip("pyboy", reason="pyboy is not installed")

from planiverse.problems.retro_games.flipull_gb import (  # noqa: E402
    BLOCK_MAX, BLOCK_MIN, CELL_BORDER, CELL_OUTSIDE, CELL_STAIRCASE, FIELD_ADDR, FIELD_BYTES,
    FIELD_COLS, FIELD_ROWS, MOVE_BUTTONS, ROM_MD5, ROW_STRIDE, THROW_BUTTONS, Calibration,
    FlipullGBAction, FlipullGBEnv, FlipullGBState, action_cost_map, block_counts,
    bounding_box, button_actions, cell_address, column_blocks, decode_blocks, decode_digits,
    decode_field, decode_staircase, decode_timer, is_playable, render_field,
)

from conftest import (  # noqa: E402
    assert_state_contract, assert_string_literals, assert_successors_contract,
    flipull_rom_path,
)
from fake_flipull_rom import block_count, stage_one, synthetic_rom  # noqa: E402

needs_rom = pytest.mark.skipif(
    flipull_rom_path() is None,
    reason='set PLANIVERSE_FLIPULL_ROM to a "Flipull (USA).gb" ROM to run cartridge tests',
)


@pytest.fixture(scope="module")
def fake_rom():
    return synthetic_rom()


@pytest.fixture
def env(fake_rom):
    game = FlipullGBEnv(fake_rom, verify_rom=False)
    game.fix_index(0)
    yield game
    game.close()


def raw_field(cells):
    """Pack a 14x16 field into the 448 bytes the game keeps at $C840."""
    raw = bytearray(FIELD_BYTES)
    for row in range(FIELD_ROWS):
        for col in range(FIELD_COLS):
            raw[ROW_STRIDE * row + col] = cells[row][col]
    return raw


# --------------------------------------------------------------------------- decoding
# These read synthetic RAM, so they run with no ROM and no emulator.

def test_cell_address_uses_a_32_byte_row():
    """The stride is $20 even though only 16 columns carry meaning — the upper half of
    every row was zero throughout the recording."""
    assert cell_address(0, 0) == 0xC840
    assert cell_address(1, 0) == 0xC860
    assert cell_address(8, 0) == 0xC940      # the row the map dumps
    assert cell_address(13, 0) == 0xC9E0
    assert ROW_STRIDE == 0x20 and FIELD_COLS == 16


def test_only_83_to_86_are_playable():
    """`$80` is border, `$87` is the fixed staircase, and only `$83`-`$86` count."""
    assert [is_playable(value) for value in (0x00, 0x80, 0x82, 0x83, 0x86, 0x87)] == \
           [False, False, False, True, True, False]
    assert (BLOCK_MIN, BLOCK_MAX) == (0x83, 0x86)


def test_the_maps_stage_one_holds_twenty_five_blocks():
    """The rows the memory map dumps verbatim, which it checked against `BLOCK 25`."""
    field = decode_field(raw_field(stage_one()))
    blocks = decode_blocks(field)
    assert len(blocks) == 25
    assert set(block_counts(blocks)) == {1, 2, 3, 4}, "four block types"
    assert all(1 <= block.col <= 5 for block in blocks), "stage 1 uses columns 1-5"
    assert all(8 <= block.row <= 12 for block in blocks)


def test_the_staircase_is_not_counted_as_blocks():
    """`$87` is structural: it forms the stepped diagonal and can never be cleared."""
    field = decode_field(raw_field(stage_one()))
    staircase = decode_staircase(field)
    assert staircase, "stage 1 has a staircase"
    assert not any((block.row, block.col) in staircase for block in decode_blocks(field))


def test_the_borders_are_where_the_map_says():
    field = decode_field(raw_field(stage_one()))
    assert all(field[0][col] == CELL_BORDER for col in range(FIELD_COLS)), "ceiling"
    assert all(field[13][col] == CELL_BORDER for col in range(FIELD_COLS)), "floor"
    assert all(field[row][0] == CELL_BORDER for row in range(FIELD_ROWS)), "left wall"


def test_counters_are_decimal_digits_not_bcd():
    """25 lives as `05` and `02` in adjacent bytes. Searching for 25 or $19 finds nothing."""
    assert decode_digits(0x02, 0x05) == 25
    assert decode_digits(0x00, 0x00) == 0
    assert decode_digits(0x09, 0x09) == 99


def test_the_timer_is_minutes_and_two_second_digits():
    """The map's worked example: 2:59, checked against the HUD."""
    assert decode_timer(2, 5, 9) == 179
    assert decode_timer(0, 0, 0) == 0


def test_a_destroyed_block_drops_its_column():
    """The collapse the map recorded on column 5, byte for byte."""
    cells = [[CELL_OUTSIDE] * FIELD_COLS for _ in range(FIELD_ROWS)]
    for row, value in zip(range(8, 13), (0x85, 0x84, 0x86, 0x83, 0x83)):
        cells[row][5] = value
    before = decode_field(raw_field(cells))
    assert list(column_blocks(before, 5))[8:13] == [0x85, 0x84, 0x86, 0x83, 0x83]

    for row in range(12, 8, -1):                  # the bottom goes, everything above falls
        cells[row][5] = cells[row - 1][5]
    cells[8][5] = CELL_OUTSIDE
    after = decode_field(raw_field(cells))
    assert list(column_blocks(after, 5))[8:13] == [0x00, 0x85, 0x84, 0x86, 0x83]


def test_bounding_box_is_derived_not_assumed():
    """Only stage 1 was observed and it uses columns 1-5; the full width comes from the
    ceiling and floor reading `$80` across 16 columns."""
    field = decode_field(raw_field(stage_one()))
    (top, bottom), (left, right) = bounding_box(field)
    assert (top, bottom) == (0, FIELD_ROWS - 1)
    assert (left, right) == (0, FIELD_COLS - 1)


def test_render_alphabet():
    cells = [[CELL_OUTSIDE] * FIELD_COLS for _ in range(FIELD_ROWS)]
    cells[5][0] = CELL_BORDER
    cells[5][1] = CELL_STAIRCASE
    cells[5][2] = 0x83
    cells[5][3] = 0x86
    text = render_field(decode_field(raw_field(cells)), held=2)
    assert text.splitlines()[0] == "#=14"
    assert text.splitlines()[-1] == "held: 2"


def test_render_shows_the_player_so_that_moving_is_visible():
    """The player's row is part of state equality, so it has to survive into the text.

    Without it two states that differ only by which row he stands on render identically,
    and a rendered trace of a plan silently drops every move.
    """
    field = decode_field(raw_field(stage_one()))
    assert render_field(field, held=2, player=64) != render_field(field, held=2, player=72)
    assert render_field(field, held=2, player=64).splitlines()[-1] == "player: 64"


# --------------------------------------------------------------------------- actions

def test_the_action_set_is_move_and_throw():
    """Flipull's whole input vocabulary: pick a row, throw. Nothing layered over it."""
    assert [action.split(",")[0] for action in button_actions()] == ["up", "down", "a"]
    for action in button_actions():
        button, ticks = action.split(",")
        assert button in action_cost_map and int(ticks) > 0


def test_the_throw_button_comes_from_calibration():
    calibration = Calibration(press_ticks=5, throw_button="b", throw_ticks=7)
    assert button_actions(calibration) == ["up,5", "down,5", "b,7"]
    assert set(THROW_BUTTONS) == {"a", "b"} and set(MOVE_BUTTONS) == {"up", "down"}


def test_every_action_costs_one_input():
    assert FlipullGBAction("up,8").cost() == FlipullGBAction("a,8").cost() == 1
    assert FlipullGBAction("nop,8").cost() == 0


def test_action_string_is_filename_safe():
    assert str(FlipullGBAction("a,8")) == "a_for_8"


def test_fix_index_refuses_a_stage_it_cannot_reach():
    """`$FFC6` looks like a stage number but was never watched changing, and no password
    route has been found — so anything but the boot stage fails loudly."""
    game = FlipullGBEnv("unused.gb")
    game.fix_index(0)
    with pytest.raises(AssertionError, match="no verified way to select a stage"):
        game.fix_index(1)


def test_a_foreign_rom_warns_because_the_addresses_are_revision_specific(tmp_path):
    rom = tmp_path / "not-flipull.gb"
    rom.write_bytes(b"\x00" * 32768)
    with pytest.warns(UserWarning, match=ROM_MD5):
        FlipullGBEnv(str(rom))


# ------------------------------------------------------------- against a Game Boy

def test_reset_loads_the_stage_the_map_recorded(env):
    state, info = env.reset()
    assert_string_literals(state)
    assert state.depth == 0 and state.gb_state
    assert info["blocks"] == 25, "the map's stage 1"
    assert info["clear_target"] == 9, "CLEAR 09"
    assert state.timer_seconds == 179, "TIME 2:59"
    assert state.held_block is not None


def test_the_field_and_the_counter_agree(env):
    """The memory map's own cross-check: 25 cells in `$83`-`$86` against `BLOCK 25`."""
    state, _ = env.reset()
    assert state.is_consistent()
    assert len(state.blocks) == state.blocks_remaining == 25


def test_reset_waits_for_the_stage_to_start_listening(env):
    _, info = env.reset()
    assert info["intro_ticks"] is not None, "a state that answers nothing is not a state"


def test_calibration_finds_the_player_and_the_throw(env):
    _, info = env.reset()
    calibration = info["calibration"]
    assert calibration.player_sprite is not None, "the player's sprite is the only row index"
    assert calibration.throw_button in THROW_BUTTONS
    low, high = calibration.hold_window
    assert low < calibration.press_ticks < high
    assert calibration.row_pitch, "one row of movement, in pixels"


def test_the_player_sprite_is_found_by_moving_not_assumed(env):
    """It has to move up for up and down for down. A scratch byte that merely changes is
    not enough — parking one inside the OAM buffer is exactly how this went wrong first."""
    state, info = env.reset()
    up = FlipullGBAction(f"up,{info['calibration'].press_ticks}").apply(env.pyboy, state)
    down = FlipullGBAction(f"down,{info['calibration'].press_ticks}").apply(env.pyboy, state)
    assert up.player_y < state.player_y < down.player_y


def test_moving_is_a_real_state_change(env):
    """The player's row is part of the position: a throw from another row does something
    else, so up and down must not collapse into self-loops."""
    state, _ = env.reset()
    offered = {str(action) for action, _ in env.successors(state)}
    assert len(offered) == 3, "up, down and throw all do something"


def test_successors_contract(env):
    state, _ = env.reset()
    assert_successors_contract(env.successors(state))


def test_successors_are_deterministic_and_leave_the_parent_alone(env):
    state, _ = env.reset()
    before = state.literals
    first = [child.literals for _, child in env.successors(state)]
    second = [child.literals for _, child in env.successors(state)]
    assert first == second
    assert state.literals == before


def test_applying_an_action_rewinds_to_the_parent_first(env):
    state, info = env.reset()
    ticks = info["calibration"].press_ticks
    first = FlipullGBAction(f"down,{ticks}").apply(env.pyboy, state)
    FlipullGBAction(f"down,{ticks}").apply(env.pyboy, first)
    second = FlipullGBAction(f"down,{ticks}").apply(env.pyboy, state)
    assert first == second


def test_a_throw_at_a_different_type_swaps_the_held_block(env):
    state, info = env.reset()
    throw = FlipullGBAction(f"{info['calibration'].throw_button},{info['calibration'].throw_ticks}")
    after = throw.apply(env.pyboy, state)
    assert after.blocks_remaining == state.blocks_remaining, "a swap clears nothing"
    assert after.held_block != state.held_block


def test_a_throw_at_its_own_type_clears_a_block_and_drops_the_column(env):
    """The whole mechanic, end to end — and the column collapse the map recorded."""
    state, info = env.reset()
    ticks = info["calibration"].press_ticks
    down = FlipullGBAction(f"down,{ticks}")
    throw = FlipullGBAction(f"{info['calibration'].throw_button},{ticks}")

    column_before = [state.field[row][5] for row in range(8, 13)]
    for _ in range(4):                      # to the row whose right-hand block matches
        state = down.apply(env.pyboy, state)
    after = throw.apply(env.pyboy, state)

    assert after.blocks_remaining == 24, "one block gone"
    assert after.is_consistent(), "and the counter still agrees with the field"
    assert [after.field[row][5] for row in range(8, 13)] == [CELL_OUTSIDE] + column_before[:-1]


def test_goal_is_the_clear_target_not_zero(env):
    """Flipull finishes a stage when few enough blocks are left — `BLOCK 25` against
    `CLEAR 09` — rather than when the field is empty."""
    state, _ = env.reset()
    assert state.clear_target == 9
    assert not env.is_goal(state)
    assert state.blocks_remaining > state.clear_target


def test_a_fresh_stage_is_neither_won_nor_lost(env):
    state, _ = env.reset()
    assert not env.is_goal(state) and not env.is_terminal(state)
    assert state.timer_seconds > 0


def test_simulate_and_step(env):
    state, info = env.reset()
    ticks = info["calibration"].press_ticks
    plan = [FlipullGBAction(f"down,{ticks}"), FlipullGBAction(f"down,{ticks}")]
    trace = env.simulate(plan)
    assert len(trace) == 3 and [s.depth for s in trace] == [0, 1, 2]

    env.reset()
    after, cleared = env.step(f"down,{ticks}")
    assert cleared == 0, "moving clears nothing"
    assert len(env.render()) == 2


def test_get_actions(env):
    env.reset()
    assert env.get_actions() == env.actions


# ------------------------------------------------------------------ the real thing

@pytest.fixture
def cartridge():
    game = FlipullGBEnv(flipull_rom_path())
    game.fix_index(0)
    yield game
    game.close()


@needs_rom
def test_cartridge_boots_into_a_stage(cartridge):
    state, info = cartridge.reset()
    assert_state_contract(state)
    assert state.blocks_remaining > 0
    assert state.is_consistent(), "field and HUD counter disagree — see the memory map"
    assert not cartridge.is_goal(state)


@needs_rom
def test_cartridge_stage_one_is_the_one_the_map_recorded(cartridge):
    """25 blocks in columns 1-5, `CLEAR 09`, `TIME 2:59`."""
    state, info = cartridge.reset()
    assert state.blocks_remaining == 25
    assert info["clear_target"] == 9
    assert state.timer_seconds == 179


@needs_rom
def test_cartridge_calibration(cartridge):
    _, info = cartridge.reset()
    calibration = info["calibration"]
    assert calibration.player_sprite is not None
    assert calibration.throw_button in THROW_BUTTONS
    assert calibration.hold_window is not None


@needs_rom
def test_cartridge_throwing_changes_the_field(cartridge):
    state, info = cartridge.reset()
    throw = FlipullGBAction(f"{info['calibration'].throw_button},{info['calibration'].throw_ticks}")
    after = throw.apply(cartridge.pyboy, state)
    assert after.field != state.field or after.held_block != state.held_block
    assert after.is_consistent()


@needs_rom
def test_cartridge_screenshot(cartridge, tmp_path):
    pytest.importorskip("PIL", reason="Pillow is not installed")
    state, _ = cartridge.reset()
    out = tmp_path / "frame.png"
    state.save(flipull_rom_path(), str(out))
    assert out.exists() and out.stat().st_size > 0


# The synthetic cartridge's own shape, so a change to it fails here rather than somewhere
# confusing further down.

def test_the_synthetic_stage_matches_the_map():
    assert block_count(stage_one()) == 25
