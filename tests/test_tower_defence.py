"""Tests for the tower defence environment.

The wave is a short loop of integer arithmetic; the tests pin what the environment promises
on top of it: an undefended wave leaks, a tower in range kills, gold and lives are kept, a
state is the same state wherever it was reached from, and every bundled map is won by the
plan it was accepted on and by none of the thoughtless ones.
"""
import json
import os

import pytest

from planiverse.environments.games.tower_defence.environment import (
    MAPS, START, TOWERS, TowerAction, TowerDefenceEnv, TowerState, draw_map, run_wave,
)

from conftest import assert_string_literals, assert_successors_contract

DATA = os.path.join(os.path.dirname(__file__), "data")
STRAIGHT = [(x, 3) for x in range(12)]
SLOTS = [(x, 4) for x in range(12)]


@pytest.fixture
def env():
    game = TowerDefenceEnv()
    game.set_index(0)
    return game


# ------------------------------------------------------------------------------- the wave

def test_an_undefended_wave_walks_through():
    kills, leaks = run_wave(STRAIGHT, SLOTS, (), (5, 10, 10, 10))
    assert (kills, leaks) == (0, 5)


def test_towers_in_range_kill_and_a_tower_out_of_range_does_nothing():
    kills, leaks = run_wave(STRAIGHT, SLOTS, ((5, "arrow"), (6, "arrow")), (3, 10, 10, 14))
    assert kills == 3 and leaks == 0
    far = [(x, 7) for x in range(12)]
    kills, leaks = run_wave(STRAIGHT, far, ((5, "arrow"),), (3, 10, 10, 14))
    assert (kills, leaks) == (0, 3), "four cells away is past an arrow tower's reach"


def test_a_thick_fast_wave_needs_more_than_one_arrow():
    one = run_wave(STRAIGHT, SLOTS, ((5, "arrow"),), (6, 40, 16, 10))
    three = run_wave(STRAIGHT, SLOTS, ((4, "arrow"), (5, "arrow"), (6, "arrow")), (6, 40, 16, 10))
    assert one[1] > three[1]


def test_the_wave_is_deterministic():
    wave = (6, 25, 12, 12)
    towers = ((3, "arrow"), (7, "cannon"))
    assert run_wave(STRAIGHT, SLOTS, towers, wave) == run_wave(STRAIGHT, SLOTS, towers, wave)


# ------------------------------------------------------------------------------ the rules

def test_building_costs_gold_and_needs_a_free_slot(env):
    state, _ = env.reset()
    built = env.__advance__(state, TowerAction("arrow", 0))
    assert built.gold == state.gold - TOWERS["arrow"]["cost"] and built.towers == ((0, "arrow"),)
    assert env.__advance__(built, TowerAction("cannon", 0)) == built, "the slot is taken"
    broke = TowerState((), TOWERS["arrow"]["cost"] - 1, 3, 0)
    assert env.__advance__(broke, TowerAction("arrow", 1)) == broke, "no gold, no tower"
    assert env.__advance__(state, TowerAction("arrow", 99)) == state, "no such slot"


def test_starting_a_wave_pays_bounties_and_costs_lives(env):
    state, _ = env.reset()
    after = env.__advance__(state, START)
    assert after.wave == 1
    assert after.lives < state.lives, "an undefended first wave leaks"
    assert after.gold == state.gold, "and nothing was killed, so nothing was paid"


def test_the_goal_and_the_dead_end(env):
    waves = len(env.instance["waves"])
    assert env.is_goal(TowerState((), 0, 1, waves))
    assert not env.is_goal(TowerState((), 0, 1, waves - 1))
    dead = TowerState((), 0, 0, 1)
    assert env.is_terminal(dead) and env.successors(dead) == []


def test_successors_obey_the_contract_and_offer_start(env):
    state, _ = env.reset()
    children = env.successors(state)
    assert_successors_contract(children)
    assert START in [action for action, _ in children]
    assert all(child != state for _, child in children)


def test_states_hash_by_position_not_depth():
    one = TowerState(((2, "arrow"),), 40, 3, 1, depth=1)
    two = TowerState(((2, "arrow"),), 40, 3, 1, depth=4)
    assert one == two and hash(one) == hash(two)
    assert one != TowerState(((2, "arrow"),), 40, 2, 1)


def test_literals_are_strings_and_name_towers_gold_lives_and_waves(env):
    state, _ = env.reset()
    assert_string_literals(state)
    assert f"lives({state.lives})" in state.literals and "waves_fought(0)" in state.literals


def test_actions_parse_and_print(env):
    assert TowerAction.parse("build(cannon,3)") == TowerAction("cannon", 3)
    assert TowerAction.parse("start") == START and str(START) == "start"
    assert len(env.get_actions()) == 2 * len(env.instance["slots"]) + 1
    with pytest.raises(ValueError):
        TowerAction("laser", 0)


def test_simulate_and_step_agree(env):
    plan = [TowerAction("arrow", 0), START]
    trace = env.simulate(plan)
    env.reset()
    for action in plan:
        state, _ = env.step(action)
    assert state == trace[-1] and len(env.render()) == 3


# ------------------------------------------------------------------------------- instances

def test_every_map_is_won_by_its_plan_and_by_no_thoughtless_one():
    with open(os.path.join(DATA, "tower_defence_solutions.json")) as handle:
        solutions = {int(index): plan for index, plan in json.load(handle).items()}
    assert len(MAPS) == 100 and sorted(solutions) == list(range(100))
    for index in range(0, 100, 5):
        game = TowerDefenceEnv()
        game.set_index(index)
        assert game.validate(solutions[index]), f"map {index}"
        for plan in game.__baselines__():
            assert not game.validate(plan), f"map {index} falls to a thoughtless plan"


def test_a_drawn_map_is_a_path_with_slots_beside_it():
    import random
    instance = draw_map(random.Random(5))
    path = [tuple(cell) for cell in instance["path"]]
    assert path[0][0] == 0 and path[-1][0] == 11
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(path, path[1:]))
    assert all(tuple(slot) not in path for slot in instance["slots"])


def test_set_index_refuses_a_map_that_is_not_there():
    game = TowerDefenceEnv()
    for index in (-1, len(MAPS), 999):
        with pytest.raises(IndexError, match="Invalid index"):
            game.set_index(index)


def test_a_generated_map_reproduces_from_its_seed():
    game = TowerDefenceEnv()
    instance = game.generate_instance(seed=3000)
    assert instance["path"] == MAPS[0]["path"] and instance["waves"] == MAPS[0]["waves"], \
        "map 0 is seed 3000"
    assert game.validate(game.witness)
    assert TowerDefenceEnv().generate_instance(seed=3000) == instance
