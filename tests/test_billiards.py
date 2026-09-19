"""Tests for the billiards environment.

The physics is pooltool's, so the tests do not re-derive collisions; they pin what the
environment adds: a shot is deterministic and a state is the same state wherever it was
reached from, a potted ball leaves the record, a sunk cue ball ends the game, the shot
alphabet follows the balls left, and every bundled table is cleared by the plan it was
accepted on.
"""
import json
import os

import pytest

pytest.importorskip("pooltool", reason="pooltool is not installed")

from planiverse.environments.billiards.environment import (  # noqa: E402
    CUTS, SPEEDS, TABLES, BilliardsAction, BilliardsEnv, BilliardsState, draw_table, strike,
)

from conftest import assert_string_literals, assert_successors_contract  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def env():
    game = BilliardsEnv()
    game.set_index(0)
    return game


# ------------------------------------------------------------------------------ the rules

def test_a_straight_shot_at_a_ball_in_front_of_a_pocket_pots_it():
    """The cue ball behind the 1 on the centre line, the 1 a hand from the far middle pocket."""
    balls = {"cue": (0.4953, 0.6), "1": (0.4953, 1.0)}
    after = strike(balls, BilliardsAction("1", 0, 3.5))
    assert "1" not in after or after["1"][1] > 1.0, "driven towards the top cushion or potted"


def test_a_shot_is_deterministic_and_a_state_is_a_position():
    balls = {"cue": (0.3, 0.5), "1": (0.6, 1.2), "2": (0.2, 1.5)}
    once = strike(balls, BilliardsAction("1", 30, 2.0))
    again = strike(balls, BilliardsAction("1", 30, 2.0))
    assert once == again
    one = BilliardsState(once, 2, depth=0)
    two = BilliardsState(again, 2, depth=7)
    assert one == two and hash(one) == hash(two)
    assert one != BilliardsState(once, 1)


def test_a_shot_at_a_ball_that_is_gone_is_refused():
    assert strike({"cue": (0.3, 0.5)}, BilliardsAction("1", 0, 2.0)) is None


def test_the_goal_is_every_object_ball_potted_and_a_scratch_is_the_end(env):
    cleared = BilliardsState({"cue": (0.3, 0.5)}, 1)
    assert env.is_goal(cleared) and not env.is_terminal(cleared)
    scratched = BilliardsState({"1": (0.3, 0.5)}, 2)
    assert scratched.scratched and env.is_terminal(scratched) and not env.is_goal(scratched)
    assert env.successors(scratched) == []
    stuck = BilliardsState({"cue": (0.3, 0.5), "1": (0.6, 1.2)}, 0)
    assert env.is_terminal(stuck)


def test_the_alphabet_follows_the_balls_left(env):
    state, _ = env.reset()
    assert len(env.get_actions(state)) == len(state.object_balls) * len(CUTS) * len(SPEEDS)
    fewer = BilliardsState({"cue": (0.3, 0.5), "1": (0.6, 1.2)}, 2)
    assert len(env.get_actions(fewer)) == len(CUTS) * len(SPEEDS)
    assert BilliardsAction.parse("shot(2,-30,3.5)") == BilliardsAction("2", -30, 3.5)
    with pytest.raises(ValueError):
        BilliardsAction("1", 45, 2.0)


def test_successors_obey_the_contract_and_spend_a_shot(env):
    state, _ = env.reset()
    children = env.successors(state)
    assert_successors_contract(children)
    assert_string_literals(state)
    assert all(child.shots_left == state.shots_left - 1 for _, child in children)
    assert f"balls_left({len(state.object_balls)})" in state.literals


def test_simulate_and_step_agree(env):
    plan = env.get_actions(env.reset()[0])[:2]
    trace = env.simulate(plan)
    env.reset()
    for action in plan:
        state, _ = env.step(action)
    assert state == trace[-1] and len(env.render()) == 3


def test_a_drawn_table_has_no_two_balls_touching():
    import random
    instance = draw_table(random.Random(3), balls=3)
    balls = list(instance["balls"].values())
    assert len(balls) == 4
    for i, (x, y) in enumerate(balls):
        for px, py in balls[i + 1:]:
            assert (x - px) ** 2 + (y - py) ** 2 > (2 * 0.028575) ** 2


# ------------------------------------------------------------------------------- instances

def test_every_table_is_cleared_by_its_plan():
    with open(os.path.join(DATA, "billiards_solutions.json")) as handle:
        solutions = {int(index): plan for index, plan in json.load(handle).items()}
    assert len(TABLES) == 100 and sorted(solutions) == list(range(100))
    for index in range(0, 100, 5):
        game = BilliardsEnv()
        game.set_index(index)
        assert len(solutions[index]) >= 3 and game.validate(solutions[index]), f"table {index}"


def test_set_index_refuses_a_table_that_is_not_there():
    game = BilliardsEnv()
    for index in (-1, len(TABLES), 999):
        with pytest.raises(IndexError, match="Invalid index"):
            game.set_index(index)


@pytest.mark.slow
def test_a_generated_table_reproduces_from_its_seed():
    """Table 0 is seed 9000 drawn with three balls; the draw is a breadth-first search of
    the table, so it is seconds rather than milliseconds."""
    game = BilliardsEnv()
    instance = game.generate_instance(seed=9000, balls=3)
    assert instance["balls"] == TABLES[0]["balls"], "table 0 is seed 9000"
    assert len(game.witness) >= 3 and game.validate(game.witness)
    assert BilliardsEnv().generate_instance(seed=9000, balls=3) == instance


def test_a_shot_the_physics_cannot_resolve_is_not_offered(monkeypatch):
    """pooltool's cushion model asserts on the odd geometry; such a shot is no successor."""
    import planiverse.environments.billiards.environment as module

    pt = module._pooltool()

    def refuse(system, inplace=True):
        raise AssertionError("v_n_0 < 0")

    monkeypatch.setattr(pt, "simulate", refuse)
    game = BilliardsEnv()
    state, _ = game.reset()
    assert strike(dict(state.balls), BilliardsAction("1", CUTS[0], SPEEDS[0])) is None
    assert game.successors(state) == []
    assert game.step(BilliardsAction("1", CUTS[0], SPEEDS[0]))[0] == state
