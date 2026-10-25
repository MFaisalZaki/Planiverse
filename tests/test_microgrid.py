"""Tests for the microgrid environment on python-microgrid.

The modules' dynamics (the battery's efficiency and limits, the genset's output range, the
grid's prices and outages, the cost of an hour) are pymgrid's and are not re-derived here. What
the tests pin is the reduction: loss of load is a dead end the moment it happens, the operating
cost is carried in the state and bounded at the horizon, the battery's reserve is a condition on
the final state, and the target is what the best reference policy achieves, so every bundled
instance is solved by the plan it was accepted on. The benchmark microgrids ship inside pymgrid,
so nothing is downloaded.
"""
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("pymgrid", exc_type=ImportError, reason="pymgrid is not installed")

from conftest import assert_string_literals, assert_successors_contract  # noqa: E402
from planiverse.environments.operational.microgrid.environment import (  # noqa: E402
    DAYS, MODES, PERIOD, MicrogridAction, MicrogridEnv, MicrogridState, _pymgrid,
    reference_policies,
)

DATA = os.path.join(os.path.dirname(__file__), "data")


def solutions():
    with open(os.path.join(DATA, "microgrid_solutions.json")) as handle:
        return {int(index): plan for index, plan in json.load(handle).items()}


@pytest.fixture(scope="module")
def env():
    game = MicrogridEnv()
    game.set_index(0)
    yield game
    game.close()


# ------------------------------------------------------------------------ the data

def test_the_benchmark_microgrids_ship_inside_pymgrid():
    """`from_scenario` reads the pymgrid25 files in the package; there is nothing to fetch."""
    pymgrid = _pymgrid()
    root = Path(pymgrid.__file__).parent / "data" / "scenario" / "pymgrid25"
    for number in (0, 12, 24):
        assert (root / f"microgrid_{number}" / f"microgrid_{number}.yaml").is_file()


# ---------------------------------------------------------------------- the state

def test_a_period_is_a_value_state(env):
    state, info = env.reset()
    assert info["generated"] is False and info["index"] == 0
    assert state.hour == 0 and state.cost == 0 and state.lost == 0
    assert_string_literals(state)
    first, second = (MicrogridAction.parse(text) for text in solutions()[0][:2])
    a = env.__advance__(env.__advance__(state, first), second)
    b = env.__advance__(env.__advance__(state, first), second)
    assert a == b and hash(a) == hash(b) and a.literals == b.literals
    assert a.hour == 2 * PERIOD and a.depth == 2
    same = MicrogridState(a.hour, a.charge, a.cost, a.lost, (0, 0, 0, 0, 0), depth=9)
    assert same == a, "the depth and the flows are not the identity"
    assert f"hour({a.hour})" in a.literals


def test_expanding_a_state_twice_gives_the_same_children(env):
    state, _ = env.reset()
    first = env.successors(state)
    assert_successors_contract(first)
    assert len(first) == len(env.get_actions()) <= 2 * len(MODES)
    second = env.successors(state)
    assert [(str(a), s.key) for a, s in first] == [(str(a), s.key) for a, s in second]
    for action, child in first:
        assert env.__advance__(state, action) == child
        assert MicrogridAction.parse(str(action)) == action
    with pytest.raises(ValueError):
        MicrogridAction("discharge", "on")


def test_the_battery_stays_inside_pymgrid_limits(env):
    """`charge` throughout fills the battery and `cover` empties it, both no further than
    pymgrid's capacity and minimum allow."""
    _, modules = env._microgrid()
    battery = modules["battery"]
    for mode in ("charge", "cover"):
        state, _ = env.reset()
        switch = "on" if "genset" in modules else "off"
        for _ in range(env.instance["horizon"] // PERIOD):
            state = env.__advance__(state, MicrogridAction(mode, switch))
            assert battery.min_capacity - 0.1 <= state.charge <= battery.max_capacity + 0.1


def test_loss_of_load_and_the_reserve_are_dead_ends(env):
    inst = env.instance
    capacity = env._capacity()
    short = MicrogridState(PERIOD, 0.5 * capacity, 10.0, 25.0, (0, 0, 0, 0, 0), capacity=capacity)
    assert env.is_terminal(short) and not env.is_goal(short) and env.successors(short) == []
    done = MicrogridState(inst["horizon"], inst["reserve"] * capacity, inst["target"], 0,
                          (0, 0, 0, 0, 0), capacity=capacity)
    assert env.is_goal(done) and env.successors(done) == []
    dear = MicrogridState(inst["horizon"], capacity, inst["target"] + 1, 0, (0, 0, 0, 0, 0),
                          capacity=capacity)
    assert env.is_terminal(dear) and not env.is_goal(dear)
    flat = MicrogridState(inst["horizon"], (inst["reserve"] - 0.05) * capacity, 0, 0,
                          (0, 0, 0, 0, 0), capacity=capacity)
    assert env.is_terminal(flat) and not env.is_goal(flat), "the battery under its reserve"


def test_a_microgrid_without_a_grid_loses_load_with_the_genset_off():
    """On an islanded microgrid the genset is the only source after solar and the battery,
    so a night with the battery idle and the genset off loses load, and is a dead end."""
    game = MicrogridEnv()
    game.set_instance({"microgrid": 2, "start": 24 * 100, "horizon": 24, "soc": 0.5,
                       "reserve": 0.5, "target": 1e9})
    state, info = game.reset()
    assert info["genset"] and not info["grid"]
    child = game.__advance__(state, MicrogridAction("idle", "off"))
    assert child.lost > 0 and "lost_load" in child.literals and game.is_terminal(child)
    assert game.__advance__(state, MicrogridAction("idle", "on")).lost == 0
    assert [str(a) for a in game.get_actions()].count("operate(idle, on)") == 1


# ------------------------------------------------------------------- the instances

def test_the_bundled_instances_and_their_plans(env):
    plans = solutions()
    assert len(DAYS) == 100 and sorted(plans) == list(range(100))
    for index in (0, 50, 99):
        game = MicrogridEnv()
        game.set_index(index)
        assert game.validate(plans[index]), f"instance {index}"
        assert game.simulate(plans[index])[-1].cost <= game.instance["target"]
    with pytest.raises(IndexError):
        env.set_index(100)
    with pytest.raises(ValueError):
        MicrogridEnv().set_instance({"microgrid": 25, "start": 0, "horizon": 24, "soc": 0.5,
                                     "reserve": 0.5, "target": 1})


@pytest.mark.slow
def test_every_bundled_instance_is_solved_by_its_plan_and_not_by_leaving_the_battery_idle():
    plans = solutions()
    game = MicrogridEnv()
    for index in range(len(DAYS)):
        game.set_index(index)
        assert game.validate(plans[index]), f"instance {index}"
        idle = MicrogridAction("idle", "on" if "genset" in game._microgrid()[1] else "off")
        assert not game.validate([idle] * (game.instance["horizon"] // PERIOD)), \
            f"instance {index}: the battery left idle would do"


def test_the_target_is_the_best_reference_policy():
    game = MicrogridEnv()
    instance = game.generate_instance(seed=3)
    outcomes = game.run_references()
    _, modules = game._microgrid()
    assert len(outcomes) == len(reference_policies("genset" in modules, "grid" in modules))
    assert instance["target"] == min(cost for _, cost, _ in outcomes if cost is not None)
    assert game.validate(game.witness) and game.witness_expansions == len(outcomes)


def test_the_same_seed_gives_the_same_instance_and_it_round_trips():
    one, two = MicrogridEnv(), MicrogridEnv()
    first = one.generate_instance(seed=11, horizon=24)
    assert first == two.generate_instance(seed=11, horizon=24)
    assert first["horizon"] == 24 and one.reset()[1]["generated"]
    assert [str(a) for a in one.witness] == [str(a) for a in two.witness]
    other = MicrogridEnv()
    other.set_instance(json.loads(json.dumps(first)))
    assert other.reset()[0].literals == one.reset()[0].literals
    assert other.validate(one.witness)
    one.set_index(0)
    assert one.witness is None and one.reset()[1]["generated"] is False


# ------------------------------------------------------------------------- the picture

def test_the_picture_draws_the_microgrid_and_a_status_line(env):
    from planiverse.rendering.text import missing_glyphs

    state, info = env.reset()
    text = str(state)
    picture = state.picture()
    assert isinstance(picture, str) and picture and not missing_glyphs(picture)
    assert picture == state.picture(), "the same state draws the same picture"
    for element in ("☼ solar", "battery", "⌂ load", "◂ reserve", "━"):
        assert element in picture
    assert ("genset" in picture) == info["genset"] and ("grid " in picture) == info["grid"]
    assert picture.splitlines()[-1].startswith(f"hour 0/{state.horizon} ")
    assert f"reserve {state.reserve:.0%}" in picture.splitlines()[-1]
    child = env.__advance__(state, env.get_actions()[0])
    assert child.picture() != picture and "kWh" in child.picture()
    assert len(child.picture().splitlines()) == len(picture.splitlines())
    assert str(state) == text
