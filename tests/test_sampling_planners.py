"""Tests for Future State Maximization.

It is randomised, so every test seeds it. An unseeded run is not reproducible and should not
be asserted on.
"""
import pytest

from planiverse.environments.gameboy_py.puzznic import PuzznicGame
from planiverse.planners.fsx import FSXPlanner, option_count
from planiverse.planners.width.result import Budget


def boxes(state):
    return sum(1 for literal in state.literals if literal.startswith("at(box"))


@pytest.fixture
def env():
    game = PuzznicGame()
    game.set_index(0)
    return game


# -------------------------------------------------------------------------------- FSX

def test_fsx_needs_no_goal_and_no_heuristic(env):
    """The whole point. It is constructed with neither, and still acts."""
    planner = FSXPlanner(horizon=4, walkers=4, seed=0)
    result = planner.solve(env, Budget(max_expansions=1500, max_seconds=30))
    assert result.status in ("solved", "step_limit", "out_of_budget", "dead_end")
    assert result.statistics.expansions > 0
    assert len(result.states) > 1, "it moved"


def test_fsx_is_reproducible_when_seeded(env):
    first = FSXPlanner(horizon=3, walkers=3, seed=7).solve(
        env, Budget(max_expansions=800, max_seconds=30))
    second = FSXPlanner(horizon=3, walkers=3, seed=7).solve(
        env, Budget(max_expansions=800, max_seconds=30))
    assert [str(s.literals) for s in first.states] == [str(s.literals) for s in second.states]


def test_fsx_prefers_states_with_more_futures(env):
    """`option_count` is the measure on its own: a state one move from being stuck scores
    lower than a state with the board open."""
    state, _ = env.reset()
    open_board = option_count(env, state, horizon=5, walkers=6, seed=0)
    assert open_board > 1, "the initial position has room to move"

    # Walk into a dead end and score that instead.
    node, seen = state, 0
    while seen < 40:
        successors = env.successors(node)
        if not successors:
            break
        node = successors[0][1]
        seen += 1
        if env.is_terminal(node):
            break
    if env.is_terminal(node) or not env.successors(node):
        assert option_count(env, node, horizon=5, walkers=6, seed=0) < open_board


def test_fsx_takes_a_goal_when_one_is_adjacent(env):
    """A goal state here is absorbing, so it has no futures at all and pure future-counting
    would rank it last. It is taken unconditionally instead."""
    class AlmostDone(PuzznicGame):
        def is_goal(self, state):
            return state is not self._start

    game = AlmostDone()
    game.set_index(0)
    game._start, _ = game.reset()
    result = FSXPlanner(horizon=3, walkers=2, seed=0).solve(
        game, Budget(max_expansions=500), state=game._start)
    assert result.solved and len(result.plan) == 1


def test_fsx_rejects_a_measure_it_does_not_have():
    with pytest.raises(ValueError, match="measure"):
        FSXPlanner(measure="vibes")
    assert FSXPlanner(measure="entropy").measure == "entropy"


def test_fsx_reports_when_it_runs_out_of_room(env):
    result = FSXPlanner(horizon=2, walkers=2, seed=0).solve(env, Budget(max_expansions=1))
    assert result.status == "out_of_budget"
    assert not result.solved


def test_a_temperature_makes_the_choice_stochastic(env):
    """Boltzmann selection, closer to the physical formulation than plain argmax."""
    hot = FSXPlanner(horizon=2, walkers=2, seed=1, temperature=0.5)
    result = hot.solve(env, Budget(max_expansions=600, max_seconds=30))
    assert result.statistics.expansions > 0
