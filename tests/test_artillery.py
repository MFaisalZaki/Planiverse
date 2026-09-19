"""Tests for the artillery environment.

The flight is a few lines of arithmetic, so the tests pin what the environment promises on
top of it: gravity and wind bend a shell the way they should, a crater and a fall do what the
rules say, a shot is deterministic and a state is the same state wherever it was reached
from, and every bundled field is cleared by the plan it was accepted on.
"""
import json
import os

import pytest

from planiverse.environments.artillery.environment import (
    ACTIONS, ANGLES, FIELDS, GUN, POWERS, RADIUS, WIDTH, ArtilleryAction, ArtilleryEnv,
    ArtilleryState, fire, flight,
)

from conftest import assert_string_literals, assert_successors_contract

DATA = os.path.join(os.path.dirname(__file__), "data")


def flat(height=5):
    return tuple([height] * WIDTH)


@pytest.fixture
def env():
    game = ArtilleryEnv()
    game.set_index(0)
    return game


# ------------------------------------------------------------------------------ the flight

def test_a_shell_flies_further_with_the_wind_and_a_steeper_shot_goes_higher_not_further():
    still = flight(flat(), 0.0, ArtilleryAction(45, 30))
    tail = flight(flat(), 4.0, ArtilleryAction(45, 30))
    head = flight(flat(), -4.0, ArtilleryAction(45, 30))
    assert still is not None and tail is not None and head is not None
    assert head[0] < still[0] < tail[0]
    steep = flight(flat(), 0.0, ArtilleryAction(75, 30))
    assert steep[0] < still[0], "seventy-five degrees lands short of forty-five"


def test_a_shell_that_leaves_the_field_does_nothing():
    heights = flat(2)
    assert flight(heights, 0.0, ArtilleryAction(45, 50)) is None, "a full-power lob flies off the end"
    assert fire(heights, ((60, 3),), 0.0, ArtilleryAction(45, 50)) == (heights, ((60, 3),))


def test_a_crater_lowers_the_ground_and_a_hit_destroys_a_target():
    heights = flat(5)
    landing = flight(heights, 0.0, ArtilleryAction(45, 30))
    column = int(landing[0])
    after, targets = fire(heights, ((column, 6),), 0.0, ArtilleryAction(45, 30))
    assert after[column] < 5, "the ground under the impact is blown away"
    assert all(after[c] == 5 for c in range(WIDTH) if abs(c - column) > RADIUS + 1)
    assert targets == (), "a target at the impact is destroyed"


def test_a_target_whose_ground_goes_drops_and_a_long_fall_kills():
    """With the wind at 4, a shell at forty-five degrees and power 30 lands at column 58 on
    flat ground and its crater reaches column 61, outside the blast. A target there drops to
    the new surface and lives; one on a pillar there falls eight and dies, though the blast
    never touched it."""
    heights = list(flat(5))
    landing = flight(tuple(heights), 4.0, ArtilleryAction(45, 30))
    assert int(landing[0]) == 58
    _, dropped = fire(tuple(heights), ((61, 6),), 4.0, ArtilleryAction(45, 30))
    assert dropped == ((61, 5),), "lowered with its ground, and alive"
    heights[61] = 12
    _, killed = fire(tuple(heights), ((61, 13),), 4.0, ArtilleryAction(45, 30))
    assert killed == (), "the pillar under it went, and the fall was fatal"


def test_a_shot_is_deterministic_and_actions_print_as_names(env):
    state, _ = env.reset()
    once = env.__advance__(state, ACTIONS[3])
    again = env.__advance__(state, ACTIONS[3])
    assert once == again and hash(once) == hash(again)
    assert ArtilleryAction.parse("fire(45,40)") == ArtilleryAction(45, 40)
    assert len(ACTIONS) == len(ANGLES) * len(POWERS) and ACTIONS[0].cost() == 1
    with pytest.raises(ValueError):
        ArtilleryAction(50, 40)


# -------------------------------------------------------------------------------- the state

def test_states_hash_by_terrain_targets_and_shells_not_depth():
    one = ArtilleryState(flat(), ((40, 6),), 2, depth=0)
    two = ArtilleryState(flat(), ((40, 6),), 2, depth=7)
    assert one == two and hash(one) == hash(two)
    assert one != ArtilleryState(flat(), ((40, 6),), 1)


def test_literals_name_every_column_the_targets_and_the_shells(env):
    state, _ = env.reset()
    assert_string_literals(state)
    assert sum(1 for literal in state.literals if literal.startswith("height(")) == WIDTH
    assert f"shells_left({state.shells_left})" in state.literals
    assert any(literal.startswith("target(") for literal in state.literals)


def test_the_goal_is_no_target_and_the_dead_end_is_no_shell(env):
    assert env.is_goal(ArtilleryState(flat(), (), 0)) and not env.is_terminal(ArtilleryState(flat(), (), 0))
    stuck = ArtilleryState(flat(), ((40, 6),), 0)
    assert env.is_terminal(stuck) and env.successors(stuck) == []


def test_successors_obey_the_contract_and_the_gun_stays_put(env):
    state, _ = env.reset()
    children = env.successors(state)
    assert_successors_contract(children)
    assert all(child.shells_left == state.shells_left - 1 for _, child in children)
    assert all(child.heights[GUN] == state.heights[GUN] for _, child in children)


def test_simulate_and_step_agree_and_render_prints(env):
    plan = [ACTIONS[0], ACTIONS[5]]
    trace = env.simulate(plan)
    env.reset()
    for action in plan:
        state, _ = env.step(action)
    assert state == trace[-1] and len(trace) == 3
    assert len(env.render()) == 3
    assert "G" in str(trace[0])


# ------------------------------------------------------------------------------- instances

def test_every_field_has_a_plan_of_at_least_two_shells():
    with open(os.path.join(DATA, "artillery_solutions.json")) as handle:
        solutions = {int(index): plan for index, plan in json.load(handle).items()}
    assert len(FIELDS) == 100 and sorted(solutions) == list(range(100))
    for index, field in enumerate(FIELDS):
        assert field["shells"] >= len(solutions[index]) >= 2, f"field {index}"
        assert len(field["targets"]) >= 2 and len(field["heights"]) == WIDTH


@pytest.mark.parametrize("index", range(100))
def test_the_stored_plan_still_clears_its_field(index):
    with open(os.path.join(DATA, "artillery_solutions.json")) as handle:
        plan = json.load(handle)[str(index)]
    game = ArtilleryEnv()
    game.set_index(index)
    game.reset()
    assert game.validate(plan), f"field {index} is no longer cleared by {plan}"


def test_set_index_refuses_a_field_that_is_not_there():
    game = ArtilleryEnv()
    for index in (-1, len(FIELDS), 999):
        with pytest.raises(IndexError, match="Invalid index"):
            game.set_index(index)


def test_a_generated_field_reproduces_from_its_seed():
    game = ArtilleryEnv()
    field = game.generate_instance(seed=2000)
    assert field["heights"] == FIELDS[0]["heights"] and field["wind"] == FIELDS[0]["wind"], \
        "field 0 is seed 2000"
    assert len(game.witness) >= 2 and game.validate(game.witness)
    assert ArtilleryEnv().generate_instance(seed=2000) == field
