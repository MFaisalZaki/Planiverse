"""Tests for the pure-Python Flipull environment.

Split in two. The first half pins the *rules* down on hand-made boards, because a rule set
that is only ever exercised through the shipped stages is a rule set nobody can argue with.
The second half checks the environment contract and that every stage's goal is actually
reachable — a benchmark whose goals cannot be met is worse than no benchmark.
"""
import pytest

from planiverse.environments.gameboy_py.flipull import (
    BLOCK_TYPES, EMPTY, FlipullAction, FlipullGame, FlipullState, STAGES, WALL,
    collapse, count_blocks, parse_stage, playable_rows, throw,
)
from planiverse.planners.width import BFWSSearch, Budget

from conftest import assert_string_literals, assert_successors_contract


def grid_of(*rows):
    return parse_stage("\n".join(rows))


def text_of(grid):
    return ["".join(row) for row in grid]


@pytest.fixture
def env():
    game = FlipullGame()
    game.fix_index(0)
    return game


# ------------------------------------------------------------------------- the rules

def test_a_throw_destroys_its_own_type_and_swaps_the_first_different_one():
    grid = grid_of("#####", "#   #", "#121#", "#####")
    result = throw(grid, 2, "1")
    assert result is not None
    board, held = result
    # The rightmost `1` dies and its cell empties; the `2` behind it becomes the thrown
    # block, and the `2` itself comes back into the hand.
    assert text_of(board)[2] == "#11 #"
    assert held == "2", "the block that was swapped out is now in hand"
    assert count_blocks(board) == 2, "exactly one block was destroyed"


def test_a_throw_eats_a_whole_run_of_its_own_type():
    grid = grid_of("######", "#    #", "#2111#", "######")
    board, held = throw(grid, 2, "1")
    assert count_blocks(board) == 1, "all three 1s went"
    assert held == "2"


def test_a_throw_is_refused_when_the_first_block_it_meets_is_a_different_type():
    """The rule that makes this a puzzle. Without it every throw would be legal and the
    board would be a permutation group."""
    grid = grid_of("#####", "#   #", "#112#", "#####")
    assert throw(grid, 2, "1") is None, "the 2 is met first, so nothing happens"
    assert throw(grid, 2, "2") is not None, "holding a 2, the same row is legal"


def test_a_throw_into_a_row_with_no_blocks_is_refused():
    grid = grid_of("#####", "#   #", "#111#", "#####")
    assert throw(grid, 1, "1") is None


def test_a_throw_that_clears_the_row_leaves_nothing_in_hand_to_swap():
    """Nothing different is ever met, so the held block is unchanged."""
    grid = grid_of("#####", "#   #", "#111#", "#####")
    board, held = throw(grid, 2, "1")
    assert count_blocks(board) == 0
    assert held == "1", "there was nothing to swap with"


def test_the_throw_leaves_the_board_alone_when_it_is_refused():
    grid = grid_of("#####", "#   #", "#112#", "#####")
    before = text_of(grid)
    throw(grid, 2, "1")
    assert text_of(grid) == before, "a refused throw must not mutate the caller's grid"


def test_a_legal_throw_does_not_mutate_the_caller_either():
    grid = grid_of("#####", "#   #", "#121#", "#####")
    before = text_of(grid)
    throw(grid, 2, "1")
    assert text_of(grid) == before


def test_an_out_of_range_row_or_a_nonsense_hand_is_refused():
    grid = grid_of("#####", "#   #", "#121#", "#####")
    assert throw(grid, 99, "1") is None
    assert throw(grid, None, "1") is None
    assert throw(grid, 2, "#") is None


# ------------------------------------------------------------------------ the collapse

def test_destroying_a_block_drops_the_stack_above_it():
    grid = grid_of("#####", "#1  #", "#2  #", "#####")
    collapse(grid, 2, 1)
    assert text_of(grid) == ["#####", "#   #", "#1  #", "#####"]


def test_the_collapse_never_drags_the_border_into_the_board():
    """It used to shift whatever was above, which walked the top wall down into the play
    area one throw at a time — quietly, because the board still looked plausible."""
    grid = grid_of("#####", "# 1 #", "# 2 #", "#####")
    collapse(grid, 2, 2)
    assert text_of(grid)[0] == "#####", "the border stays put"
    assert WALL not in text_of(grid)[1][1:-1], "and does not appear inside the play area"
    assert text_of(grid) == ["#####", "#   #", "# 1 #", "#####"]


def test_the_collapse_stops_at_the_first_gap():
    """Only a contiguous run of blocks falls. A block with air under it is already resting
    somewhere else and does not teleport down."""
    grid = grid_of("#####", "#1  #", "#   #", "#2  #", "#####")
    collapse(grid, 3, 1)
    assert text_of(grid) == ["#####", "#1  #", "#   #", "#   #", "#####"]


def test_a_throw_collapses_every_column_it_emptied():
    """Two columns emptied by one throw, and both stacks come down."""
    grid = grid_of("######", "#  33#", "#2 11#", "######")
    board, held = throw(grid, 2, "1")
    assert held == "2", "the 2 at the far end was swapped into the hand"
    assert text_of(board) == ["######", "#    #", "#1 33#", "######"]


# -------------------------------------------------------------------------- the stages

def test_every_stage_parses_to_a_rectangle():
    for index, (text, _) in enumerate(STAGES):
        grid = parse_stage(text)
        widths = {len(row) for row in grid}
        assert len(widths) == 1, f"stage {index} is ragged: {widths}"
        assert len(playable_rows(grid)) >= 2, f"stage {index} has nowhere to stand"


def test_every_stage_opens_with_a_move_available():
    """Resetting into a position with nothing to do would be a cruel joke."""
    for index in range(len(STAGES)):
        game = FlipullGame()
        game.fix_index(index)
        state, _ = game.reset()
        assert not game.is_terminal(state), f"stage {index} is dead on arrival"
        assert game.successors(state), f"stage {index} has no successors"


def test_every_stage_starts_short_of_its_goal():
    for index, (_, target) in enumerate(STAGES):
        game = FlipullGame()
        game.fix_index(index)
        state, _ = game.reset()
        assert state.blocks_remaining > target, f"stage {index} is already solved"


@pytest.mark.parametrize("index", range(len(STAGES)))
def test_every_stage_can_actually_be_solved(index):
    """The bar every other environment in this library had to clear. The stages are
    generated rather than hand-drawn precisely so this can hold: hand-drawn boards kept
    turning out to have unreachable targets, which makes for a useless benchmark."""
    game = FlipullGame()
    game.fix_index(index)
    result = BFWSSearch(width=2, progress=lambda s: s.blocks_remaining).solve(
        game, Budget(max_expansions=400000, max_seconds=180))
    assert result.solved, f"stage {index}: {result.status}"
    assert game.validate(result.plan), "and the plan replays to a goal"


def test_the_stages_get_harder():
    """A ramp, not eight variations of the same board."""
    sizes = [count_blocks(parse_stage(text)) for text, _ in STAGES]
    assert sizes[-1] > sizes[0], "the last stage is bigger than the first"


# ---------------------------------------------------------------------- the environment

def test_fix_index_rejects_a_stage_that_does_not_exist(env):
    with pytest.raises(IndexError, match="Invalid index"):
        env.fix_index(len(STAGES))
    with pytest.raises(IndexError):
        env.fix_index(-1)


def test_reset_reports_the_stage_it_set_up():
    game = FlipullGame()
    game.fix_index(2)
    state, info = game.reset()
    assert info["stage"] == 2
    assert info["blocks"] == state.blocks_remaining
    assert info["clear_target"] == STAGES[2][1]


def test_the_opening_hand_always_makes_the_first_throw_legal():
    """Chosen to be the type of the block the player would meet first, so no stage opens
    with the player having to guess which row to walk to."""
    for index in range(len(STAGES)):
        game = FlipullGame()
        game.fix_index(index)
        state, _ = game.reset()
        assert state.can_throw(), f"stage {index} cannot open with a throw"


def test_states_carry_string_literals(env):
    state, _ = env.reset()
    assert_string_literals(state)
    assert any(lit.startswith("holding(") for lit in state.literals)
    assert any(lit.startswith("at(player,") for lit in state.literals)


def test_successors_obey_the_contract_and_never_loop_back(env):
    state, _ = env.reset()
    successors = env.successors(state)
    assert_successors_contract(successors)
    assert all(child != state for _, child in successors), "no action is a no-op"
    assert len({action for action, _ in successors}) == len(successors), "no duplicates"


def test_walking_into_the_ceiling_is_not_offered(env):
    state, _ = env.reset()
    top = state
    while True:
        moves = dict((str(a), s) for a, s in env.successors(top))
        if "up" not in moves:
            break
        top = moves["up"]
    assert top.row == playable_rows(top.grid)[0], "stopped at the top row, not before it"


def test_simulate_replays_a_plan_from_the_start(env):
    state, _ = env.reset()
    plan = [action for action, _ in env.successors(state)][:1]
    trace = env.simulate(plan)
    assert len(trace) == len(plan) + 1
    assert trace[0] == state


def test_simulate_and_step_agree(env):
    state, _ = env.reset()
    plan, node = [], state
    for _ in range(6):
        successors = env.successors(node)
        if not successors:
            break
        action, node = successors[-1]
        plan.append(action)

    stepped = FlipullGame()
    stepped.fix_index(0)
    stepped.reset()
    for action in plan:
        stepped.step(action)
    assert stepped.state == env.simulate(plan)[-1]


def test_replaying_the_same_plan_gives_the_same_trace(env):
    """Determinism, which is the property that lets a planner treat a state as a value."""
    state, _ = env.reset()
    plan = [action for action, _ in env.successors(state)][:1] * 3
    first = [str(s) for s in env.simulate(plan)]
    second = [str(s) for s in env.simulate(plan)]
    assert first == second


def test_step_reports_how_many_blocks_went(env):
    state, _ = env.reset()
    assert state.can_throw()
    _, cleared = env.step(FlipullAction("throw"))
    assert cleared >= 1


def test_step_before_reset_is_an_error():
    game = FlipullGame()
    game.fix_index(0)
    with pytest.raises(ValueError, match="reset"):
        game.step(FlipullAction("throw"))


def test_a_goal_state_is_absorbing(env):
    result = BFWSSearch(width=2, progress=lambda s: s.blocks_remaining).solve(
        env, Budget(max_expansions=400000, max_seconds=180))
    assert result.solved
    goal = env.simulate(result.plan)[-1]
    assert env.is_goal(goal)
    assert env.successors(goal) == [], "there is nothing left to do"
    assert not env.is_terminal(goal), "a goal is not a dead end"


def test_is_terminal_is_exact_and_agrees_with_the_successors():
    """The thing this environment has that the cartridge one does not: because the rules
    are known here, a dead end can be *computed* rather than waited for."""
    game = FlipullGame()
    game.fix_index(0)
    state, _ = game.reset()

    seen, frontier, dead_ends = {state}, [state], 0
    while frontier and len(seen) < 3000:
        node = frontier.pop()
        successors = game.successors(node)
        if game.is_terminal(node):
            dead_ends += 1
            assert successors == [], "a dead end offers nothing"
            assert not node.any_throw_connects(), "and no throw anywhere would connect"
        for _, child in successors:
            if child not in seen:
                seen.add(child)
                frontier.append(child)
    assert dead_ends > 0, "stage 0 should contain dead ends, or it is not a puzzle"


def test_a_dead_end_is_recognised_immediately():
    """A board where every remaining block is a lone survivor of its type."""
    game = FlipullGame()
    game.fix_index(0)
    grid = parse_stage("#####\n#   #\n#12 #\n#####")
    state = FlipullState(grid, 2, "3", clear_target=0)
    assert not state.any_throw_connects()
    assert game.is_terminal(state)
    assert game.successors(state) == []


def test_actions_compare_and_print_as_their_names():
    assert FlipullAction("up") == FlipullAction("up")
    assert FlipullAction("up") != FlipullAction("down")
    assert str(FlipullAction("throw")) == "throw"
    assert len({FlipullAction("up"), FlipullAction("up")}) == 1
    assert sorted(FlipullAction(n) for n in ("throw", "down", "up"))[0].name == "down"
    assert FlipullAction("up").cost() == 1


def test_an_action_that_does_not_exist_is_refused():
    with pytest.raises(ValueError, match="unknown action"):
        FlipullAction("jump")


def test_get_actions_lists_the_whole_alphabet(env):
    assert {a.name for a in env.get_actions()} == {"up", "down", "throw"}


def test_render_prints_the_history(env, capsys):
    state, _ = env.reset()
    env.step(FlipullAction("throw"))
    rendered = env.render()
    assert len(rendered) == 2
    assert "held:" in capsys.readouterr().out


def test_the_state_marks_the_row_the_player_is_on(env):
    state, _ = env.reset()
    lines = str(state).split("\n")
    assert sum(1 for line in lines if line.endswith("<")) == 1


def test_states_hash_by_position_not_by_depth(env):
    """Depth is bookkeeping. Two identical boards reached by different routes are the same
    state, or search never closes anything."""
    state, _ = env.reset()
    twin = FlipullState([list(r) for r in state.grid], state.row, state.held,
                        state.clear_target, depth=99)
    assert twin == state and hash(twin) == hash(state)
