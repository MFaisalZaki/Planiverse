"""Tests for the Puzznic Game Boy environment.

Two tiers. The first runs against a synthetic cartridge built by `fake_puzznic_rom.py`,
which puts Puzznic's documented facts at Puzznic's documented addresses without being
Puzznic — enough to check that the environment boots a Game Boy, forces a stage, decodes
the grid, waits for a move to settle and notices a cleared stage. The second needs the
real cartridge, which is copyrighted and cannot ship here, so it is opt-in:

    PLANIVERSE_PUZZNIC_ROM="/path/to/Puzznic (J).gb" poetry run pytest tests/test_puzznic_gb.py
"""
import pytest

pytest.importorskip("pyboy", reason="pyboy is not installed")

from planiverse.problems.retro_games.puzznic_gb import (  # noqa: E402
    STAGE_LOADER_ENTRY, _force_stage, boot, create_pyboy, cursor_of, stage_is_loaded,
    BLOCK_MIN, CELL_CLEARING, CELL_EMPTY, CELL_LEDGE, CELL_OUTSIDE, CELL_WALL, Calibration,
    GRID_ADDR, GRID_BYTES, GRID_COLS, GRID_ROWS, PROBE_MAX_HOLD, PUSH_SCHEMES,
    PuzznicGBAction, PuzznicGBEnv, PuzznicGBState, ROM_MD5, action_cost_map,
    action_list, block_counts, bounding_box, button_actions, calibrate,
    cell_address, decode_blocks, decode_grid, decode_records, is_dead_end,
    CURSOR_IMPASSABLE, cursor_path, measure_hold_window, measure_push_window,
    probe_push_scheme, push_hold, push_probe_candidates, render_grid, walk_cursor,
    wait_until_interactive,
)

from conftest import (  # noqa: E402
    assert_state_contract, assert_string_literals, assert_successors_contract,
    puzznic_rom_path,
)
from fake_puzznic_rom import stage_layouts, synthetic_rom  # noqa: E402

needs_rom = pytest.mark.skipif(
    puzznic_rom_path() is None,
    reason='set PLANIVERSE_PUZZNIC_ROM to a "Puzznic (J).gb" ROM to run cartridge tests',
)


@pytest.fixture(scope="module")
def fake_rom():
    """The synthetic cartridge; `synthetic_rom` assembles it once per process."""
    return synthetic_rom()


@pytest.fixture
def env(fake_rom):
    """The default: button presses, which is what the console has."""
    game = PuzznicGBEnv(fake_rom, verify_rom=False)
    game.fix_index(0)
    yield game
    game.close()


@pytest.fixture
def roomy_env(fake_rom):
    """Stage 3: an 8x10 room, which is the one with space to probe a push in."""
    game = PuzznicGBEnv(fake_rom, verify_rom=False)
    game.fix_index(3)
    yield game
    game.close()


def raw_grid(cells):
    """Pack a 12x10 list of cell codes into the 240 bytes the game keeps at $DF00."""
    raw = bytearray(GRID_BYTES)
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            raw[20 * row + 2 * col] = cells[row][col]
    return raw


# ------------------------------------------------------------------------- decoding
# These read synthetic RAM, so they run with no ROM and no emulator at all.

def test_cell_address_matches_the_row_offset_table():
    assert cell_address(0, 0) == 0xDF00
    assert cell_address(0, 9) == 0xDF12
    assert cell_address(1, 0) == 0xDF14          # stride 20, not 16
    assert cell_address(11, 9) == 0xDFEE
    # The last cell ends the 240-byte region exactly.
    assert cell_address(11, 9) + 2 == GRID_ADDR + GRID_BYTES


def test_decode_grid_reads_rows_of_ten():
    cells = [[row * 10 + col for col in range(GRID_COLS)] for row in range(GRID_ROWS)]
    grid = decode_grid(raw_grid(cells))
    assert len(grid) == GRID_ROWS and all(len(row) == GRID_COLS for row in grid)
    assert grid[5][3] == 53


def test_decode_blocks_uses_the_type_offset_and_slot_byte():
    raw = raw_grid([[CELL_EMPTY] * GRID_COLS for _ in range(GRID_ROWS)])
    raw[20 * 4 + 2 * 2] = 0x0A            # a block...
    raw[20 * 4 + 2 * 2 + 1] = 7           # ...whose record lives in slot 7
    blocks = decode_blocks(raw)
    assert blocks == ((4, 2, 3, 7),)      # $0A - 7 == type 3


def test_only_08_and_up_count_as_blocks():
    """The loader's own test is `CP $08` / `RET C`, so $07 and below are terrain."""
    cells = [[CELL_EMPTY] * GRID_COLS for _ in range(GRID_ROWS)]
    for col, value in enumerate([CELL_EMPTY, CELL_CLEARING, CELL_LEDGE, CELL_OUTSIDE,
                                 0x04, 0x05, CELL_WALL, 0x07, BLOCK_MIN, 0x0F]):
        cells[0][col] = value
    blocks = decode_blocks(raw_grid(cells))
    assert [(block.col, block.type) for block in blocks] == [(8, 1), (9, 8)]


def test_decode_records_skips_holes_rather_than_stopping_at_one():
    """Records are zeroed in place, so walking to the first $00 reports almost nothing."""
    raw = bytearray(6 * 4)
    raw[12:18] = bytes([0x0A, 0x00, 8, 6, 0x62, 0xC5])      # slots 0 and 1 were cleared
    raw[18:24] = bytes([0x0B, 0x00, 8, 3, 0x62, 0xB6])
    records = decode_records(raw, total=4)
    assert [record.slot for record in records] == [2, 3]
    assert records[0].type == 3 and (records[0].row, records[0].col) == (8, 6)


def test_dead_end_is_a_type_with_exactly_one_block_left():
    pair = decode_blocks(raw_grid(_cells({(0, 0): 0x08, (0, 1): 0x08})))
    assert not is_dead_end(pair)
    lone = decode_blocks(raw_grid(_cells({(0, 0): 0x08, (0, 1): 0x09, (0, 2): 0x09})))
    assert is_dead_end(lone)
    triple = decode_blocks(raw_grid(_cells({(0, 0): 0x08, (0, 1): 0x08, (0, 2): 0x08})))
    assert not is_dead_end(triple)
    assert block_counts(triple) == {1: 3}


def _cells(overrides):
    cells = [[CELL_EMPTY] * GRID_COLS for _ in range(GRID_ROWS)]
    for (row, col), value in overrides.items():
        cells[row][col] = value
    return cells


def test_bounding_box_derives_the_shape_from_the_outside_marker():
    """There are no width/height variables: a small stage is more $03 around the edges."""
    cells = [[CELL_OUTSIDE] * GRID_COLS for _ in range(GRID_ROWS)]
    for row in range(3, 7):
        for col in range(2, 8):
            cells[row][col] = CELL_WALL
    assert bounding_box(decode_grid(raw_grid(cells))) == ((3, 6), (2, 7))


def test_render_grid_alphabet():
    cells = [[CELL_OUTSIDE] * GRID_COLS for _ in range(GRID_ROWS)]
    cells[1][1] = CELL_WALL
    cells[1][2] = CELL_EMPTY
    cells[1][3] = CELL_LEDGE
    cells[2][1] = 0x08
    cells[2][2] = CELL_CLEARING
    cells[2][3] = CELL_EMPTY
    grid = decode_grid(raw_grid(cells))
    assert render_grid(grid, cursor=None) == "#.=\n1*."
    assert render_grid(grid, cursor=(2, 3)) == "#.=\n1*c"      # cursor on empty
    assert render_grid(grid, cursor=(2, 1)) == "#.=\n¢*."      # cursor on a block


# --------------------------------------------------------------------------- actions

def test_action_list_is_the_cursor_moves_plus_two_pushes():
    assert action_list == ["left,6", "right,6", "up,6", "down,6", "a+left,6", "a+right,6"]
    for action in action_list:
        buttons, ticks = action.split(",")
        assert int(ticks) > 0
        assert all(button in action_cost_map for button in buttons.split("+"))


def test_no_vertical_push():
    """Puzznic slides blocks sideways; you cannot lift one, so there is no a+up."""
    assert not any(action.startswith("a+up") or action.startswith("a+down")
                   for action in action_list)


def test_action_parsing():
    assert PuzznicGBAction("a+right,6").actions_tick_list == [("a", 6), ("right", 6)]


def test_every_move_costs_one_whether_or_not_it_pushes():
    assert PuzznicGBAction("a+right,6").cost() == PuzznicGBAction("right,6").cost() == 1
    assert PuzznicGBAction("nop,6").cost() == 0


def test_action_string_is_filename_safe():
    assert str(PuzznicGBAction("a+right,6")) == "a_with_right_for_6"


# ---------------------------------------------------------------------------- stages

def test_fix_index_accepts_the_whole_byte_the_loader_indexes_with():
    game = PuzznicGBEnv("unused.gb")
    assert game.stage_index is None
    game.fix_index(0x2A)
    assert game.stage_index == 0x2A


@pytest.mark.parametrize("index", [-1, 0x100])
def test_fix_index_rejects_values_that_do_not_fit_in_the_byte(index):
    with pytest.raises(AssertionError, match="Invalid index"):
        PuzznicGBEnv("unused.gb").fix_index(index)


def test_a_foreign_rom_warns_because_the_addresses_are_revision_specific(tmp_path):
    rom = tmp_path / "not-puzznic.gb"
    rom.write_bytes(b"\x00" * 32768)
    with pytest.warns(UserWarning, match=ROM_MD5):
        PuzznicGBEnv(str(rom))


def test_a_missing_rom_is_left_for_pyboy_to_report(tmp_path):
    PuzznicGBEnv(str(tmp_path / "absent.gb"))          # constructing must not raise


# -------------------------------------------------------------- against a Game Boy
# From here on the environment really boots a cartridge, just not a copyrighted one.

def test_reset_boots_the_rom_and_loads_a_stage(env):
    state, info = env.reset()
    assert_string_literals(state)
    assert state.depth == 0
    assert state.gb_state, "the state must carry an emulator save-state"
    assert state.total_blocks == info["total_blocks"] == 2
    assert state.blocks_remaining == 2


def test_reset_is_repeatable(env):
    first, _ = env.reset()
    second, _ = env.reset()
    assert first == second and first.literals == second.literals


def test_the_grid_the_records_and_the_counter_agree(env):
    state, _ = env.reset()
    assert state.is_consistent()
    assert sorted((record.row, record.col) for record in state.records) == \
           sorted((block.row, block.col) for block in state.blocks)


def test_state_reads_the_cursor_out_of_ram(env):
    state, _ = env.reset()
    assert (state.cursor.row, state.cursor.col) == (5, 5)
    moved = PuzznicGBAction("left,6").apply(env.pyboy, state)
    assert (moved.cursor.row, moved.cursor.col) == (5, 4)
    assert moved.depth == state.depth + 1


def test_literals_describe_the_board(env):
    state, _ = env.reset()
    assert "at(cursor, 5, 5)" in state.literals
    assert "at(block-1, 5, 3)" in state.literals
    assert "remaining(2)" in state.literals
    assert "goal-reached" not in state.literals


def test_literals_carry_no_step_counter(env):
    """`depth` deliberately stays out of the literals: with it in, no successor could ever
    equal its parent and the self-loop filter in `successors` would be dead code."""
    state, _ = env.reset()
    assert not any("depth" in literal for literal in state.literals)


def test_fix_index_selects_the_stage_the_loader_builds(env, fake_rom):
    """The title screen rewrites $D003 and calls the loader in the same frame, so this
    only works because the environment hooks the loader rather than poking memory."""
    first, _ = env.reset()
    other = PuzznicGBEnv(fake_rom, verify_rom=False)
    other.fix_index(1)
    try:
        second, _ = other.reset()
        assert second.grid != first.grid
        assert second.total_blocks == 4
    finally:
        other.close()


def test_successors_contract(env):
    state, _ = env.reset()
    assert_successors_contract(env.successors(state))


def test_successors_exclude_moves_that_change_nothing(env):
    """The cursor starts on an empty cell, so a push has nothing to push."""
    state, _ = env.reset()
    offered = {str(action).rsplit("_for_", 1)[0] for action, _ in env.successors(state)}
    assert offered == {"left", "right", "up", "down"}


def test_successors_are_deterministic_and_leave_the_parent_alone(env):
    state, _ = env.reset()
    before = state.literals
    first = [successor.literals for _, successor in env.successors(state)]
    second = [successor.literals for _, successor in env.successors(state)]
    assert first == second
    assert state.literals == before


def test_applying_an_action_rewinds_to_the_parent_first(env):
    """Siblings expand from the same machine regardless of what the last one did."""
    state, _ = env.reset()
    first = PuzznicGBAction("left,6").apply(env.pyboy, state)
    PuzznicGBAction("down,6").apply(env.pyboy, first)
    second = PuzznicGBAction("left,6").apply(env.pyboy, state)
    assert first == second


def test_pushing_a_block_moves_it_and_takes_the_cursor_along(env):
    state, _ = env.reset()
    for _ in range(2):
        state = PuzznicGBAction("left,6").apply(env.pyboy, state)
    assert state.cursor == (5, 3) and state.grid[5][3] >= BLOCK_MIN

    pushed = PuzznicGBAction("a+right,6").apply(env.pyboy, state)
    assert pushed.grid[5][3] == CELL_EMPTY
    assert pushed.grid[5][4] >= BLOCK_MIN
    assert pushed.cursor == (5, 4)
    assert pushed.blocks_remaining == 2, "a push on its own clears nothing"


def test_a_match_clears_and_the_stage_is_won(env):
    state, _ = env.reset()
    plan = [PuzznicGBAction(action) for action in ("left,6", "left,6", "a+right,6", "a+right,6")]
    trace = env.simulate(plan)

    assert len(trace) == len(plan) + 1
    assert [state.depth for state in trace] == [0, 1, 2, 3, 4]
    assert not env.is_goal(trace[-2])
    assert env.is_goal(trace[-1])
    assert trace[-1].blocks_remaining == 0 and trace[-1].blocks == ()
    assert trace[-1].total_blocks == 2, "$D018 is never decremented"
    assert trace[-1].blocks_cleared == 2
    assert "goal-reached" in trace[-1].literals
    assert "all-blocks-matched(block-1)" in trace[-1].literals
    assert env.validate(plan)


def test_settling_hides_the_clearing_transient(env):
    """Blocks pass through cell code $01 on their way out; no settled state shows one."""
    state, _ = env.reset()
    for action in ("left,6", "left,6", "a+right,6", "a+right,6"):
        state = PuzznicGBAction(action).apply(env.pyboy, state)
        assert CELL_CLEARING not in {value for row in state.grid for value in row}


def test_simulate_starts_from_the_initial_state(env):
    state, _ = env.reset()
    assert env.simulate([])[0] == state


def test_simulate_agrees_with_successors(env):
    state, _ = env.reset()
    action, expected = env.successors(state)[0]
    assert env.simulate([action])[-1].literals == expected.literals


def test_step_and_render_track_the_history(env):
    env.reset()
    state, cleared = env.step("left,6")
    assert cleared == 0
    assert state.cursor == (5, 4)
    env.step("left,6")
    rendered = env.render()
    assert len(rendered) == 3
    assert rendered[0].splitlines()[2] == "#1.c1#"        # the cursor walks left...
    assert rendered[-1].splitlines()[2] == "#¢..1#"       # ...onto the block


def test_get_actions(env):
    env.reset()
    assert [a.rsplit(",", 1)[0] for a in env.get_actions()] == \
           [a.rsplit(",", 1)[0] for a in action_list]


def test_a_won_stage_expands_to_nothing(env):
    """Clearing the last block ends the stage and the cartridge loads the next round over
    the top of it, so a goal state has to be absorbing or search wanders into it."""
    plan = [PuzznicGBAction(action) for action in ("left,6", "left,6", "a+right,6", "a+right,6")]
    won = env.simulate(plan)[-1]
    assert env.is_goal(won)
    assert env.successors(won) == []
    assert env.simulate(plan + [PuzznicGBAction("right,6")])[-1] == won


def test_a_stage_with_a_lone_block_is_a_dead_end(fake_rom):
    """Stage 2 of the synthetic cartridge holds one block, which can never be matched."""
    game = PuzznicGBEnv(fake_rom, verify_rom=False)
    game.fix_index(2)
    try:
        state, _ = game.reset()
        assert game.is_terminal(state)
        assert not game.is_goal(state)
        assert "terminal-state" in state.literals
        assert game.successors(state) == [], "a dead end is absorbing"
    finally:
        game.close()


def test_search_solves_the_synthetic_stage(env):
    """A breadth-first search over the environment finds the two-push solution."""
    from collections import deque

    root, _ = env.reset()
    frontier, seen = deque([(root, [])]), {root.literals}
    plan = None
    while frontier and plan is None:
        state, actions = frontier.popleft()
        if len(actions) > 4:
            break
        for action, successor in env.successors(state):
            if env.is_goal(successor):
                plan = actions + [action]
                break
            if successor.literals in seen or env.is_terminal(successor):
                continue
            seen.add(successor.literals)
            frontier.append((successor, actions + [action]))
    assert plan is not None, "the stage is solvable in four moves"
    assert env.validate(plan)


# ---------------------------------------------------------------- the round intro
# The stage loader fills work RAM before the round has finished announcing itself, so a
# board can be entirely readable while every button is ignored. On `Puzznic (J)` that lasts
# 210 frames; the synthetic cartridge waits 60. A state snapshotted in that window looks
# perfectly normal and answers no action at all.

def test_reset_waits_for_the_round_to_start_listening(env):
    state, info = env.reset()
    assert info["intro_ticks"] is not None, "reset must not hand back a board that is deaf"
    assert info["intro_ticks"] >= 30, "the synthetic cartridge ignores input for 60 frames"

    moved = PuzznicGBAction(f"left,{info['calibration'].press_ticks}").apply(env.pyboy, state)
    assert moved.cursor != state.cursor, "the state reset returned has to accept input"


def test_the_board_is_readable_before_it_is_playable(fake_rom):
    """Which is exactly why the wait cannot be skipped: nothing about the RAM says 'not
    yet'. Booting without the wait leaves a stage whose cursor will not move."""
    game = PuzznicGBEnv(fake_rom, verify_rom=False, calibrate=False)
    game.fix_index(0)
    try:
        game.pyboy = create_pyboy(fake_rom, False)
        game.pyboy.hook_register(0, STAGE_LOADER_ENTRY, _force_stage, (game.pyboy, 0))
        assert boot(game.pyboy)
        assert stage_is_loaded(game.pyboy), "the loader has filled the grid..."
        deaf = PuzznicGBState(game.pyboy, 0)
        game.pyboy.button("left", 8)
        game.pyboy.tick(20, False)
        assert cursor_of(game.pyboy) == deaf.cursor, "...but nothing is listening yet"

        assert wait_until_interactive(game.pyboy) is not None
        listening = cursor_of(game.pyboy)
        game.pyboy.button("left", 8)
        game.pyboy.tick(20, False)
        assert cursor_of(game.pyboy) != listening
    finally:
        game.close()


# --------------------------------------------------------------- cursor routing
# Stages are not rectangles -- Round 1 of `Puzznic (J)` has a bottom row two cells narrower
# than the one above -- and the cursor cannot cross a wall, so stepping rows then columns
# walks into one.

def test_cursor_path_routes_around_a_wall():
    cells = [[CELL_OUTSIDE] * GRID_COLS for _ in range(GRID_ROWS)]
    for col in range(1, 6):
        cells[4][col] = CELL_EMPTY
        cells[6][col] = CELL_EMPTY
    cells[5][1] = cells[5][5] = CELL_EMPTY     # only the two ends connect the rows
    for col in range(2, 5):
        cells[5][col] = CELL_WALL              # a wall straight between start and target
    grid = decode_grid(raw_grid(cells))

    route = cursor_path(grid, _cursor(4, 3), _cursor(6, 3))
    assert route is not None, "there is a way round"
    assert "down" in route and route.count("down") == 2
    # Stepping the rows first would walk into the wall at (5, 3).
    assert route[0] in ("left", "right")


def test_cursor_path_walks_over_blocks_and_ledges():
    """Verified on `Puzznic (J)`: the cursor sits on blocks and crosses ledges."""
    cells = [[CELL_OUTSIDE] * GRID_COLS for _ in range(GRID_ROWS)]
    for col in range(1, 5):
        cells[4][col] = CELL_EMPTY
    cells[4][2] = 0x08                          # a block
    cells[4][3] = CELL_LEDGE                    # and a ledge
    grid = decode_grid(raw_grid(cells))
    assert cursor_path(grid, _cursor(4, 1), _cursor(4, 4)) == ["right"] * 3


def test_cursor_path_refuses_what_it_cannot_reach():
    cells = [[CELL_OUTSIDE] * GRID_COLS for _ in range(GRID_ROWS)]
    cells[4][1] = CELL_EMPTY
    cells[4][2] = CELL_WALL
    cells[4][3] = CELL_EMPTY
    grid = decode_grid(raw_grid(cells))
    assert cursor_path(grid, _cursor(4, 1), _cursor(4, 3)) is None
    assert cursor_path(grid, _cursor(4, 1), _cursor(4, 2)) is None, "cannot stand on a wall"
    assert cursor_path(grid, _cursor(4, 1), _cursor(4, 1)) == []


def test_walls_and_the_outside_are_what_the_cursor_cannot_enter():
    assert set(CURSOR_IMPASSABLE) == {CELL_WALL, CELL_OUTSIDE}


def _cursor(row, col):
    return type("C", (), {"row": row, "col": col})()


# ------------------------------------------------------------------- calibration
# How long a button must be held is bounded above by the cursor's auto-repeat delay, which
# is not in the memory map and differs between cartridges. It is measured, not guessed. The
# synthetic cartridge repeats after 16 frames so that there is a real bound to find.

def test_calibration_measures_the_hold_window(env):
    state, _ = env.reset()
    window = measure_hold_window(env.pyboy, state)
    assert window == (1, 16), "the synthetic cartridge repeats on the 17th frame"


def test_calibration_settles_inside_that_window(env):
    state, info = env.reset()
    calibration = info["calibration"]
    low, high = calibration.hold_window
    assert low < calibration.press_ticks < high, \
        "a hold on either edge is one frame of jitter away from missing or repeating"


def test_a_hold_past_the_window_moves_two_cells(env):
    """Which is exactly what calibration exists to avoid: the state a planner gets back
    would not be the one its action described."""
    state, info = env.reset()
    inside = PuzznicGBAction(f"left,{info['calibration'].press_ticks}").apply(env.pyboy, state)
    outside = PuzznicGBAction(f"left,{info['calibration'].hold_window[1] + 8}").apply(
        env.pyboy, state)
    assert state.cursor.col - inside.cursor.col == 1
    assert state.cursor.col - outside.cursor.col > 1


def test_calibration_finds_how_this_cartridge_pushes(env):
    state, info = env.reset()
    assert info["calibration"].push_scheme in PUSH_SCHEMES
    assert info["calibration"].push_scheme == "modifier", "A plus a direction, on this ROM"
    assert probe_push_scheme(env.pyboy, state, 8) == ("modifier", "a")


def test_calibration_is_done_once_per_cartridge(env):
    """It describes the game, not the stage, so a second reset must not pay for it again."""
    _, first = env.reset()
    calibration = env.calibration
    _, second = env.reset()
    assert env.calibration is calibration
    assert first["calibration"] == second["calibration"]


def test_button_actions_use_the_calibrated_hold(env):
    _, info = env.reset()
    ticks = info["calibration"].press_ticks
    assert env.actions == button_actions(info["calibration"])
    assert all(action.endswith(f",{ticks}") for action in env.actions)


def test_calibration_can_be_turned_off(fake_rom):
    game = PuzznicGBEnv(fake_rom, verify_rom=False, calibrate=False)
    game.fix_index(0)
    try:
        _, info = game.reset()
        assert info["calibration"] is None
        assert game.actions == action_list
    finally:
        game.close()


# The push has its own window. A held block need not repeat on the cursor's schedule, and
# holding past it slides the block two cells — which can match it away, so the state handed
# back describes something the action never asked for.

def test_a_long_hold_clears_blocks_it_was_never_asked_to(env):
    """The bug this measurement exists for: held past the repeat, one action slides the
    block twice, straight into its own colour, and the stage is two blocks lighter."""
    state, info = env.reset()
    for _ in range(2):
        state = PuzznicGBAction(f"left,{info['calibration'].press_ticks}").apply(env.pyboy, state)
    assert state.grid[5][3] >= BLOCK_MIN and state.cursor == (5, 3)

    once = PuzznicGBAction(f"a+right,{push_hold(info['calibration'])}").apply(env.pyboy, state)
    assert once.grid[5][3] == CELL_EMPTY and once.grid[5][4] >= BLOCK_MIN, "exactly one cell"
    assert once.blocks_remaining == state.blocks_remaining, "and nothing cleared"

    held = PuzznicGBAction("a+right,60").apply(env.pyboy, state)
    assert held.blocks_remaining < state.blocks_remaining, \
        "held past the repeat, the block slid on and matched away"


def test_push_window_is_measured(roomy_env):
    state, info = roomy_env.reset()
    window = measure_push_window(roomy_env.pyboy, state, info["calibration"].press_ticks,
                                 "modifier", "a")
    assert window == (1, 16), "the synthetic cartridge repeats a held block after 16 frames"


def test_calibration_reports_both_windows(roomy_env):
    _, info = roomy_env.reset()
    calibration = info["calibration"]
    assert calibration.hold_window and calibration.push_window
    low, high = calibration.push_window
    assert low < push_hold(calibration) < high


def test_a_cramped_stage_cannot_be_probed_and_says_so(env):
    """Stage 0 is two same-coloured blocks four cells apart: any slide long enough to
    measure matches them. Reporting None beats reporting a number that came from a board
    which cleared halfway through the probe."""
    _, info = env.reset()
    assert info["calibration"].push_window is None
    assert info["calibration"].push_ticks is None
    assert push_hold(info["calibration"]) == info["calibration"].press_ticks


def test_button_actions_hold_moves_and_pushes_for_their_own_windows():
    calibration = Calibration(8, (1, 16), "modifier", "a", 5, (1, 10))
    assert button_actions(calibration) == [
        "left,8", "right,8", "up,8", "down,8", "a+left,5", "a+right,5"]


def test_push_hold_falls_back_to_the_cursor_hold():
    """An unmeasurable push window must not leave the push with no hold at all."""
    assert push_hold(Calibration(7, (1, 16), "modifier", "a")) == 7


def test_probe_candidates_skip_a_block_that_would_match_itself_away():
    """A block that clears mid-probe measures nothing, so it is never chosen."""
    cells = [[CELL_OUTSIDE] * GRID_COLS for _ in range(GRID_ROWS)]
    for col in range(1, 8):
        cells[6][col] = CELL_EMPTY
        cells[7][col] = CELL_WALL          # solid floor, so nothing falls
    cells[6][1] = 0x08
    cells[6][4] = 0x08                     # same type two cells past the slide
    state = _state_from_cells(cells, cursor=(6, 1))
    assert [d for b, d in push_probe_candidates(state) if b.col == 1] == []


def test_probe_candidates_skip_a_block_that_would_fall():
    """Gravity would carry it off the row being measured."""
    cells = [[CELL_OUTSIDE] * GRID_COLS for _ in range(GRID_ROWS)]
    for col in range(1, 8):
        cells[6][col] = CELL_EMPTY
    cells[7][1] = cells[7][2] = CELL_WALL  # floor runs out after column 2...
    for col in range(3, 8):
        cells[7][col] = CELL_EMPTY         # ...and past it there is nothing to rest on
    cells[6][1] = 0x08
    state = _state_from_cells(cells, cursor=(6, 1))
    assert [d for b, d in push_probe_candidates(state) if b.col == 1] == []


def test_probe_candidates_accept_a_clear_run():
    cells = [[CELL_OUTSIDE] * GRID_COLS for _ in range(GRID_ROWS)]
    for col in range(1, 8):
        cells[6][col] = CELL_EMPTY
        cells[7][col] = CELL_WALL
    cells[6][1] = 0x08
    state = _state_from_cells(cells, cursor=(6, 1))
    assert [d for b, d in push_probe_candidates(state) if b.col == 1] == ["right"]


# ------------------------------------------------------------ reachability
# The cursor's reachability decides whether a block can be got at, so the routing tests above
# stand on their own; these cover the calibration's own probe filtering.

def _state_from_cells(cells, cursor):
    """A state built from a grid by hand, for the checks that need no emulator."""
    class Fake:
        pass

    state = Fake()
    state.grid = decode_grid(raw_grid(cells))
    state.blocks = decode_blocks(raw_grid(cells))
    state.cursor = _cursor(*cursor)
    return state


def test_push_probe_candidates_still_guard_the_calibration():
    """They are calibration machinery, not an action model: a block that falls or clears
    mid-probe measures nothing."""
    cells = [[CELL_OUTSIDE] * GRID_COLS for _ in range(GRID_ROWS)]
    for col in range(1, 8):
        cells[6][col] = CELL_EMPTY
        cells[7][col] = CELL_WALL
    cells[6][1] = 0x08
    state = _state_from_cells(cells, cursor=(6, 1))
    assert [d for b, d in push_probe_candidates(state) if b.col == 1] == ["right"]


# ------------------------------------------------------------------ the real thing

@pytest.fixture
def cartridge():
    game = PuzznicGBEnv(puzznic_rom_path())
    game.fix_index(0)
    yield game
    game.close()


@needs_rom
def test_cartridge_boots_into_a_stage(cartridge):
    state, info = cartridge.reset()
    assert_state_contract(state)
    assert state.total_blocks > 0
    assert state.blocks_remaining == state.total_blocks
    assert state.is_consistent(), "grid, records and $D019 disagree — see the memory map"
    assert not cartridge.is_goal(state)


@needs_rom
def test_cartridge_stage_fills_the_whole_grid(cartridge):
    """Every stage fills all 120 cells; the playable shape is the non-$03 region."""
    state, _ = cartridge.reset()
    assert len(state.grid) == GRID_ROWS and all(len(row) == GRID_COLS for row in state.grid)
    (top, bottom), (left, right) = state.bounding_box()
    assert top <= bottom and left <= right


@needs_rom
def test_cartridge_cursor_moves(cartridge):
    state, _ = cartridge.reset()
    moved = [successor.cursor for _, successor in cartridge.successors(state)]
    assert moved, "no action moved the cursor"
    assert any(cursor != state.cursor for cursor in moved)


@needs_rom
def test_cartridge_stages_differ(cartridge, tmp_path):
    first, _ = cartridge.reset()
    other = PuzznicGBEnv(puzznic_rom_path())
    other.fix_index(1)
    try:
        second, _ = other.reset()
        assert second.grid != first.grid
    finally:
        other.close()


@needs_rom
def test_cartridge_screenshot(cartridge, tmp_path):
    pytest.importorskip("PIL", reason="Pillow is not installed")
    state, _ = cartridge.reset()
    out = tmp_path / "frame.png"
    state.save(puzznic_rom_path(), str(out))
    assert out.exists() and out.stat().st_size > 0


# The synthetic cartridge's own shape, so a change to it fails here rather than
# somewhere confusing further down.

def test_synthetic_stages_are_what_the_tests_assume():
    zero, one, two, _ = stage_layouts()
    assert [(row, col) for row in range(GRID_ROWS) for col in range(GRID_COLS)
            if zero[row][col] >= BLOCK_MIN] == [(5, 3), (5, 6)]
    assert sum(cell >= BLOCK_MIN for row in one for cell in row) == 4
    assert sum(cell >= BLOCK_MIN for row in two for cell in row) == 1
