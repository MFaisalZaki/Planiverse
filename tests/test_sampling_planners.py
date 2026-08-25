"""Tests for the sampling-based planners: FSX and MCTS.

Both are randomised, so every test seeds them. An unseeded run of either is not reproducible
and should not be asserted on.
"""
import pytest

from planiverse.environments.puzznic import PuzznicGame
from planiverse.planners.fsx import FSXPlanner, option_count
from planiverse.planners.mcts import MCTSPlanner
from planiverse.planners.width.result import Budget


def boxes(state):
    return sum(1 for literal in state.literals if literal.startswith("at(box"))


@pytest.fixture
def env():
    game = PuzznicGame()
    game.fix_index(0)
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
    game.fix_index(0)
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


# ------------------------------------------------------------------------------- MCTS

@pytest.mark.slow
def test_mcts_solves_with_a_sparse_reward(env):
    """Goal-or-nothing: it has to stumble into a goal during a rollout before it learns
    anything at all."""
    result = MCTSPlanner(iterations=3000, seed=0).solve(
        env, Budget(max_expansions=60000, max_seconds=120))
    assert result.solved, f"expected a plan, got {result.status}"
    assert env.validate(result.plan)


@pytest.mark.slow
def test_a_denser_reward_buys_a_much_shorter_plan(env):
    """The sparse signal finds *a* plan; a reward that says how many blocks are gone finds a
    far better one from the same budget."""
    sparse = MCTSPlanner(iterations=3000, seed=0).solve(
        env, Budget(max_expansions=60000, max_seconds=120))
    dense = MCTSPlanner(iterations=3000, seed=0,
                        reward=lambda s: 1 - boxes(s) / 6).solve(
        env, Budget(max_expansions=60000, max_seconds=120))
    assert sparse.solved and dense.solved
    assert env.validate(dense.plan)
    assert len(dense.plan) < len(sparse.plan), "the gradient should pay for itself"


def test_mcts_never_reselects_a_proven_dead_end(env):
    """A branch that has been shown to be over is worth no further budget, however good its
    averages once looked."""
    from planiverse.planners.mcts import _Node

    planner = MCTSPlanner(iterations=1, seed=0)
    parent = _Node(state=None)
    parent.visits = 10
    dead = _Node(state=None, parent=parent)
    dead.terminal = True
    dead.visits, dead.best = 5, 1.0
    assert planner.__ucb1__(parent, dead) == float("-inf")


def test_mcts_prefers_the_shorter_of_two_solutions():
    """The length penalty, which is the only thing distinguishing solutions under a
    goal-or-nothing reward."""
    planner = MCTSPlanner(length_penalty=0.01)
    assert planner.length_penalty == 0.01


def test_mcts_rejects_a_backup_rule_it_does_not_have():
    with pytest.raises(ValueError, match="backup"):
        MCTSPlanner(backup="median")
    assert MCTSPlanner(backup="mean").backup == "mean"


def test_mcts_is_reproducible_when_seeded(env):
    first = MCTSPlanner(iterations=120, seed=3).solve(env, Budget(max_expansions=4000))
    second = MCTSPlanner(iterations=120, seed=3).solve(env, Budget(max_expansions=4000))
    assert first.status == second.status
    assert (first.plan or []) == (second.plan or [])


def test_mcts_returns_immediately_from_a_goal(env):
    class Solved(PuzznicGame):
        def is_goal(self, state):
            return True

    game = Solved()
    game.fix_index(0)
    result = MCTSPlanner(iterations=10, seed=0).solve(game)
    assert result.solved and result.plan == []


def test_mcts_reports_running_out_rather_than_failing(env):
    result = MCTSPlanner(iterations=100000, seed=0).solve(env, Budget(max_expansions=5))
    assert result.status == "out_of_budget"
    assert not result.solved


def test_the_rollout_keeps_the_best_reward_it_saw():
    """Random rollouts in a domain with dead ends nearly always end in one. Scoring only the
    final state throws away everything the rollout learned on the way, the tree sees 0
    everywhere, and UCT has no gradient to climb.

    Tested on a hand-made three-state chain rather than on Puzznic, where whether a random
    rollout happens to clear a block is luck and a test should not depend on it.
    """
    from planiverse.environments.base import Environment
    from planiverse.planners.mcts import _Node
    from planiverse.planners.width.result import SearchStatistics

    class Step:
        def __init__(self, name, value):
            self.name, self.value = name, value
            self.literals = frozenset({f"at({name})"})

    start, middle, end = Step("start", 0.0), Step("middle", 0.5), Step("end", 0.0)

    class Chain(Environment):
        """start -> middle -> end, where end is a dead end. The best moment was the middle."""

        def fix_index(self, index): pass

        def reset(self): return start, {}

        def successors(self, state):
            return {"start": [("go", middle)], "middle": [("go", end)],
                    "end": []}[state.name]

        def is_goal(self, state): return False

        def is_terminal(self, state): return state.name == "end"

        def simulate(self, plan): return [start, middle, end][:len(plan) + 1]

    env = Chain()
    planner = MCTSPlanner(iterations=1, seed=0, rollout_depth=10,
                          reward=lambda s: s.value)
    value, tail = planner.__simulate__(env, _Node(start), SearchStatistics(),
                                       Budget().start())
    assert tail is None, "the chain has no goal"
    assert value == 0.5, "the rollout died at 0.0 but peaked at 0.5, and 0.5 is what counts"
