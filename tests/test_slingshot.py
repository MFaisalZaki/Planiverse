"""Tests for the slingshot physics puzzle.

The physics is pymunk's, so the tests do not re-derive collisions; they pin what the
environment adds on top: which impacts break what, that a shot is deterministic and a state
is the same state wherever it was reached from, the shots-left bookkeeping, and that every
bundled level is cleared by the plan it was accepted on.
"""
import json
import os

import pytest

pytest.importorskip("pymunk", reason="pymunk is not installed")

from planiverse.environments.slingshot.environment import (  # noqa: E402
    ACTIONS, ANGLES, LEVELS, POWERS, SlingshotAction, SlingshotEnv, SlingshotState, shoot,
)

from conftest import assert_string_literals, assert_successors_contract  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")


def level_of(*bodies, shots=2):
    return {"bodies": [list(body) for body in bodies], "shots": shots}


@pytest.fixture
def env():
    game = SlingshotEnv()
    game.set_index(0)
    return game


# ------------------------------------------------------------------------------ the rules

def test_a_target_in_the_open_breaks_when_hit_and_a_miss_leaves_it_standing():
    bodies = (("target", 60.0, 1.4),)
    assert shoot(bodies, SlingshotAction(20, 60)) == (), "a flat shot lands on it and breaks it"
    after = shoot(bodies, SlingshotAction(30, 75))
    assert len(after) == 1 and after[0][0] == "target" and after[0][1] == 60.0, \
        "a shot that sails over leaves it where it stood"


def test_stone_topples_but_never_breaks_and_wood_breaks():
    stone = (("block", "stone", 2.0, 6.0, 55.0, 3.4, 0.0),)
    after = shoot(stone, SlingshotAction(20, 60))
    assert len(after) == 1 and after[0][1] == "stone"
    assert abs(after[0][6]) > 1.0, "knocked flat, but still there"
    wood = (("block", "wood", 2.0, 6.0, 55.0, 3.4, 0.0),)
    assert shoot(wood, SlingshotAction(20, 60)) == (), "the same bird shatters wood"


def test_a_shot_is_deterministic():
    game = SlingshotEnv()
    game.set_index(0)
    state, _ = game.reset()
    once = game.__advance__(state, ACTIONS[0])
    again = game.__advance__(state, ACTIONS[0])
    assert once == again and hash(once) == hash(again)


def test_actions_parse_and_print_as_their_names():
    action = SlingshotAction.parse("shoot(40,60)")
    assert action == SlingshotAction(40, 60) and str(action) == "shoot(40,60)"
    assert action.cost() == 1
    assert len(ACTIONS) == len(ANGLES) * len(POWERS)
    with pytest.raises(ValueError):
        SlingshotAction(45, 60)


# -------------------------------------------------------------------------------- the state

def test_states_hash_by_what_stands_not_by_depth():
    bodies = (("target", 60.0, 0.9),)
    one = SlingshotState(bodies, 2, depth=0)
    two = SlingshotState(bodies, 2, depth=5)
    assert one == two and hash(one) == hash(two)
    assert one != SlingshotState(bodies, 1)


def test_literals_name_cells_targets_and_shots(env):
    state, _ = env.reset()
    assert_string_literals(state)
    assert f"shots_left({state.shots_left})" in state.literals
    assert f"targets_left({state.targets_left})" in state.literals
    assert any(literal.startswith("at(target-") for literal in state.literals)


def test_the_goal_is_every_target_down_and_the_dead_end_is_running_out(env):
    cleared = SlingshotState((("block", "stone", 2.0, 6.0, 55.0, 3.0, 0.0),), 0)
    assert env.is_goal(cleared) and not env.is_terminal(cleared)
    stuck = SlingshotState((("target", 60.0, 0.9),), 0)
    assert env.is_terminal(stuck) and not env.is_goal(stuck)
    assert env.successors(stuck) == []


def test_successors_obey_the_contract_and_spend_a_shot(env):
    state, _ = env.reset()
    children = env.successors(state)
    assert_successors_contract(children)
    assert all(child.shots_left == state.shots_left - 1 for _, child in children)
    assert all(child != state for _, child in children), "a miss is not offered"


def test_simulate_and_step_agree(env):
    plan = [ACTIONS[0], ACTIONS[1]]
    trace = env.simulate(plan)
    env.reset()
    for action in plan:
        state, _ = env.step(action)
    assert state == trace[-1] and len(trace) == 3
    assert len(env.render()) == 3


# ------------------------------------------------------------------------------- instances

def test_every_level_has_shots_targets_and_a_plan():
    with open(os.path.join(DATA, "slingshot_solutions.json")) as handle:
        solutions = {int(index): plan for index, plan in json.load(handle).items()}
    assert len(LEVELS) == 100 and sorted(solutions) == list(range(100))
    for index, level in enumerate(LEVELS):
        assert level["shots"] >= len(solutions[index]) >= 3, f"level {index}"
        assert sum(1 for body in level["bodies"] if body[0] == "target") >= 2


@pytest.mark.parametrize("index", range(0, 100, 10))
def test_the_stored_plan_still_clears_its_level(index):
    """The witness each level was accepted on, replayed. A level whose plan stops working
    means the physics or the rules changed under it."""
    with open(os.path.join(DATA, "slingshot_solutions.json")) as handle:
        plan = json.load(handle)[str(index)]
    game = SlingshotEnv()
    game.set_index(index)
    game.reset()
    assert game.validate(plan), f"level {index} is no longer cleared by {plan}"


def test_set_index_refuses_a_level_that_is_not_there():
    game = SlingshotEnv()
    for index in (-1, len(LEVELS), 999):
        with pytest.raises(IndexError, match="Invalid index"):
            game.set_index(index)


@pytest.mark.slow
def test_a_generated_level_reproduces_from_its_seed_and_needs_three_shots():
    """Level 0 is seed 9000 drawn with three targets; the draw is a breadth-first search of
    the level, so it is seconds rather than milliseconds."""
    game = SlingshotEnv()
    level = game.generate_instance(seed=9000, targets=3)
    assert level["bodies"] == [list(body) for body in LEVELS[0]["bodies"]], "level 0 is seed 9000"
    assert len(game.witness) >= 3 and game.validate(game.witness)
    again = SlingshotEnv().generate_instance(seed=9000, targets=3)
    assert again == level
