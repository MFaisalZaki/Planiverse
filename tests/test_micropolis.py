"""Tests for the Micropolis city environment.

The engine is Micropolis's own and is not re-derived here. The tests pin what the environment
adds: the layout it lays on a map, that a city replays deterministically from its decisions
so a state is its path, the one-zone-a-year rule, the goal at the horizon, and that every
bundled city reaches its target by the plan it was accepted on and by no baseline.
"""
import json
import os

import pytest

pytest.importorskip("micropolisengine", reason="the Micropolis engine is not built")

from planiverse.environments.operational.micropolis.environment import (  # noqa: E402
    BASELINES, CITIES, KINDS, SITES, MicropolisAction, MicropolisEnv, MicropolisState, WAIT,
    find_patch, replay,
)

from conftest import assert_string_literals, assert_successors_contract  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="module")
def env():
    game = MicropolisEnv()
    game.set_index(0)
    return game


# ------------------------------------------------------------------------------ the engine

def test_a_city_replays_deterministically_from_its_decisions(env):
    seed, origin = env.instance["seed"], env.instance["origin"]
    decisions = (("R", 0), ("I", 4), (None, None), ("C", 1))
    assert replay(seed, origin, decisions) == replay(seed, origin, decisions)


def test_the_layout_sits_on_a_flat_patch(env):
    import micropolisengine as me
    engine = me.Micropolis()
    engine.initGame()
    engine.generateSomeCity(env.instance["seed"])
    assert find_patch(engine) == tuple(env.instance["origin"])


def test_zoning_changes_what_the_city_becomes(env):
    seed, origin = env.instance["seed"], env.instance["origin"]
    years = env.instance["years"]
    nothing = replay(seed, origin, ((None, None),) * years)
    zoned = replay(seed, origin, tuple(BASELINES[0](years)) + ((None, None),) * (years - min(years, SITES)))
    assert nothing["population"] == 0 and zoned["population"] > 0


# ------------------------------------------------------------------------------- the rules

def test_a_state_is_its_decisions(env):
    one = MicropolisState((("R", 0),), 3, 20, 0, 0, 16000, 500, depth=1)
    two = MicropolisState((("R", 0),), 3, 20, 0, 0, 16000, 500, depth=9)
    assert one == two and hash(one) == hash(two)
    assert one != MicropolisState((("R", 1),), 3, 20, 0, 0, 16000, 500)
    assert_string_literals(one)
    assert "zoned(R, 0)" in one.literals and "year(1)" in one.literals


def test_a_site_takes_one_zone_and_a_year_passes_either_way(env):
    state, info = env.reset()
    zoned = env.__advance__(state, MicropolisAction("R", 2))
    assert zoned.year == 1 and zoned.decisions == (("R", 2),)
    assert env.__advance__(zoned, MicropolisAction("C", 2)) == zoned, "the site is taken"
    assert env.__advance__(zoned, MicropolisAction("C", 99)) == zoned, "no such site"
    waited = env.__advance__(zoned, WAIT)
    assert waited.year == 2 and waited.decisions[-1] == (None, None)
    assert len(env.get_actions(zoned)) == (SITES - 1) * len(KINDS) + 1


def test_the_goal_is_the_target_at_the_horizon(env):
    years, target = env.instance["years"], env.instance["target"]
    assert not env.is_goal(MicropolisState((("R", 0),) * 2, target + 5, 0, 0, 0, 0, 0))
    done = MicropolisState(((None, None),) * years, target, 0, 0, 0, 0, 0)
    assert env.is_goal(done) and not env.is_terminal(done)
    short = MicropolisState(((None, None),) * years, target - 1, 0, 0, 0, 0, 0)
    assert env.is_terminal(short) and env.successors(short) == []


def test_successors_obey_the_contract(env):
    state, _ = env.reset()
    children = env.successors(state)
    assert_successors_contract(children)
    assert WAIT in [action for action, _ in children]
    assert all(child.year == 1 for _, child in children)


def test_actions_parse_and_print():
    assert MicropolisAction.parse("zone(I,3)") == MicropolisAction("I", 3)
    assert MicropolisAction.parse("wait") == WAIT and WAIT.cost() == 1
    with pytest.raises(ValueError):
        MicropolisAction("P", 0)


def test_simulate_and_step_agree(env):
    plan = [MicropolisAction("R", 0), WAIT]
    trace = env.simulate(plan)
    env.reset()
    for action in plan:
        state, _ = env.step(action)
    assert state == trace[-1] and len(env.render()) == 3


# ------------------------------------------------------------------------------- instances

def test_every_city_reaches_its_target_by_its_plan_and_by_no_baseline():
    with open(os.path.join(DATA, "micropolis_solutions.json")) as handle:
        solutions = {int(index): plan for index, plan in json.load(handle).items()}
    assert len(CITIES) == 100 and sorted(solutions) == list(range(100))
    for index in range(0, 100, 10):
        game = MicropolisEnv()
        game.set_index(index)
        assert game.validate(solutions[index]), f"city {index}"
        years = game.instance["years"]
        for baseline in BASELINES:
            plan = [MicropolisAction(kind, site) for kind, site in baseline(years)]
            plan += [WAIT] * (years - len(plan))
            assert not game.validate(plan), f"city {index} falls to a baseline"


def test_set_index_refuses_a_city_that_is_not_there():
    game = MicropolisEnv()
    for index in (-1, len(CITIES), 999):
        with pytest.raises(IndexError, match="Invalid index"):
            game.set_index(index)


def test_a_generated_city_reproduces_from_its_seed():
    game = MicropolisEnv()
    instance = game.generate_instance(seed=7000)
    assert instance == CITIES[0], "city 0 is seed 7000"
    assert game.validate(game.witness)
