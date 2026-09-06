"""Tests for Rollout IW and π-IW.

Both draw rollouts at random, so every test seeds them. As with the width planners, the
pure-Python Puzznic is the ground: level 1 is where IW(1) exhausts, so it is where resetting
the novelty table at every committed action can be seen to matter.
"""
import numpy as np
import pytest

from planiverse.environments.gameboy_py.puzznic import PuzznicGame
from planiverse.planners.width import (
    Budget, DepthNoveltyTable, IWSearch, PiIW, PolicyNetwork, RolloutIW,
)


def boxes(state):
    return sum(1 for literal in state.literals if literal.startswith("at(box"))


@pytest.fixture
def env():
    game = PuzznicGame()
    game.set_index(0)
    return game


# ------------------------------------------------------------------ depth novelty

def test_depth_novelty_keeps_the_shallowest_sighting_of_each_atom():
    table = DepthNoveltyTable(width=1)
    assert table.check({"a"}, depth=5), "never seen"
    assert not table.check({"a"}, depth=7), "seen shallower already"
    assert table.check({"a"}, depth=3), "shallower than before"
    assert table.depths[("a",)] == 3


def test_a_node_already_in_the_tree_is_not_pruned_for_its_own_discovery():
    """A new node needs a strictly shallower sighting; a node being re-checked may equal
    the depth it set itself."""
    table = DepthNoveltyTable(width=1)
    table.check({"a"}, depth=2, new=True)
    assert not table.check({"a"}, depth=2, new=True), "a sibling at the same depth"
    assert table.check({"a"}, depth=2, new=False), "the node that set it, revisited"


def test_depth_novelty_at_width_two_looks_at_pairs():
    table = DepthNoveltyTable(width=2)
    table.check({"a", "b"}, depth=1)
    table.check({"c"}, depth=1)
    assert table.check({"a", "c"}, depth=3), "every atom seen shallower, but not this pair"
    assert not table.check({"a", "b"}, depth=3)


def test_depth_novelty_refuses_the_same_widths_novelty_table_does():
    with pytest.raises(ValueError, match="strict=False"):
        DepthNoveltyTable(width=3)
    with pytest.raises(ValueError, match="at least 1"):
        DepthNoveltyTable(width=0)


# --------------------------------------------------------------------- Rollout IW

def test_one_rollout_iw_lookahead_has_iw_ones_reach(env):
    """Without committing, Rollout IW(1) solves its root where IW(1) exhausts: the same
    filter over roughly the same reach, and no plan."""
    narrow = IWSearch(width=1).solve(env, Budget(max_expansions=5000))
    assert narrow.status == "exhausted"
    result = RolloutIW(width=1, progress=boxes, seed=0).solve(env, Budget(max_expansions=5000))
    assert result.status == "failed" and result.plan is None
    assert result.statistics.rollouts > 0 and result.statistics.episodes == 1
    assert result.statistics.pruned_novelty > 0


def test_resetting_the_table_at_every_step_is_what_solves_it(env):
    """The online form, with a per-decision budget, finds the plan a single width-1
    lookahead cannot. The plan is the committed prefix plus the rollout that reached the
    goal, so it replays from the initial state."""
    result = RolloutIW(width=1, expansions_per_step=200, progress=boxes, seed=0).solve(
        env, Budget(max_expansions=20_000))
    assert result.solved and result.width == 1
    assert env.validate(result.plan)
    assert len(result.states) == len(result.plan) + 1


def test_rollout_iw_is_reproducible_when_seeded(env):
    runs = [RolloutIW(width=1, expansions_per_step=200, progress=boxes, seed=3).solve(
        env, Budget(max_expansions=20_000)) for _ in range(2)]
    assert [str(a) for a in runs[0].plan] == [str(a) for a in runs[1].plan]
    assert runs[0].statistics.expansions == runs[1].statistics.expansions


def test_a_dead_end_is_worth_minus_infinity_unless_told_otherwise(env):
    """Puzznic level 1's trap: clearing a pair makes progress into a state that can never
    be finished. Atari's reading scores the step; ours scores the wall."""
    from planiverse.planners.width import RolloutNode
    careful = RolloutIW(width=1, expansions_per_step=200, progress=boxes, seed=0)
    state, _ = env.reset()
    parent = RolloutNode(state)
    child = RolloutNode(state, parent=parent, action="x", depth=1, reward=2.0)
    child.terminal = True
    assert careful.__return__(child) == float("-inf")
    classical = RolloutIW(width=1, progress=boxes, avoid_dead_ends=False, seed=0)
    assert classical.__return__(child) == 2.0


def test_running_out_of_budget_says_so(env):
    result = RolloutIW(width=1, expansions_per_step=50, progress=boxes, seed=0).solve(
        env, Budget(max_expansions=3))
    assert result.status == "out_of_budget"
    assert result.statistics.expansions <= 3


def test_committing_blind_is_reported(env):
    """Without a progress measure the returns are flat, and the committed action is a
    uniform draw. The status says so rather than pretending to steer."""
    result = RolloutIW(width=1, expansions_per_step=20, max_steps=3, seed=0).solve(
        env, Budget(max_expansions=200))
    assert result.solved or "no progress measure" in result.status


def test_bad_parameters_are_refused():
    with pytest.raises(ValueError, match="width"):
        RolloutIW(width=0)
    with pytest.raises(ValueError, match="expansions_per_step"):
        RolloutIW(expansions_per_step=0)
    with pytest.raises(ValueError, match="discount"):
        RolloutIW(discount=1.5)


def test_an_already_solved_state_needs_no_plan():
    class Solved(PuzznicGame):
        def is_goal(self, state):
            return True

    game = Solved()
    game.set_index(0)
    result = RolloutIW(seed=0).solve(game)
    assert result.solved and result.plan == [] and len(result.states) == 1


# --------------------------------------------------------------------------- π-IW

def test_pi_iw_solves_the_level_and_learns_on_the_way(env):
    planner = PiIW(expansions_per_step=200, progress=boxes, seed=0)
    before = planner.network.params["W1"].copy()
    result = planner.solve(env, Budget(max_expansions=20_000))
    assert result.solved and env.validate(result.plan)
    assert planner.network.updates > 0 and planner.losses
    assert not np.array_equal(before, planner.network.params["W1"]), "it trained"
    assert {"left", "right", "up", "down"} <= set(planner.actions), "the vocabulary it met"


def test_pi_iw_is_reproducible_when_seeded(env):
    runs = [PiIW(expansions_per_step=200, progress=boxes, seed=5).solve(
        env, Budget(max_expansions=20_000)) for _ in range(2)]
    assert [str(a) for a in runs[0].plan] == [str(a) for a in runs[1].plan]


def test_pi_iw_can_measure_novelty_over_what_it_learned(env):
    """The paper's dynamic features: the hidden layer, binarised, as the atoms."""
    planner = PiIW(expansions_per_step=200, progress=boxes, seed=0, features="learned")
    result = planner.solve(env, Budget(max_expansions=20_000))
    assert result.solved and env.validate(result.plan)
    from planiverse.planners.width import RolloutNode
    state, _ = env.reset()
    atoms = planner.__atoms__(RolloutNode(state))
    assert atoms and all(isinstance(atom, int) for atom in atoms)
    both = PiIW(progress=boxes, seed=0, features="both").__atoms__(RolloutNode(state))
    assert state.literals < both


def test_pi_iw_keeps_learning_across_episodes():
    """An episode that ends without a goal is not wasted: the next starts from what it
    taught, and `max_episodes` is unbounded by default."""
    planner = PiIW(expansions_per_step=20, progress=boxes, max_steps=2, seed=0)
    game = PuzznicGame()
    game.set_index(0)
    result = planner.solve(game, Budget(max_expansions=600))
    assert result.statistics.episodes > 1
    assert planner.network.updates >= result.statistics.episodes


def test_the_policy_network_grows_with_the_vocabulary_and_learns_a_target():
    net = PolicyNetwork(inputs=8, hidden=4, learning_rate=0.05, seed=0)
    assert net.outputs == 0
    net.grow(3)
    assert net.outputs == 3 and net.params["W2"].shape == (4, 3)
    x = np.zeros((1, 8), dtype=np.float32)
    x[0, [1, 5]] = 1.0
    target = np.array([[0.1, 0.8, 0.1]], dtype=np.float32)
    first = net.update(x, target)
    for _ in range(200):
        last = net.update(x, target)
    assert last < first
    assert net.probabilities(x[0]).argmax() == 1
    net.grow(5)
    assert net.probabilities(x[0]).shape == (5,)


def test_pi_iw_refuses_bad_settings():
    with pytest.raises(ValueError, match="features"):
        PiIW(features="pixels")
    with pytest.raises(ValueError, match="temperature"):
        PiIW(temperature=0)
