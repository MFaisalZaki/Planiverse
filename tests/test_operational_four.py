"""Tests for the four operational environments added together: the epidemic on Covasim, the
traffic signals on SUMO, the airspace on BlueSky and the reservoirs on pywr.

The simulators are their own and are not re-derived here. What the tests pin is the reduction
each environment makes of a trajectory problem to a goal: a constraint that must hold
throughout is a dead end the moment it breaks, an accumulated quantity is carried in the state
and bounded at the horizon, the horizon is in the state, and the target is what the scripted
reference achieves, so the bundled instances are solvable by the plans they were accepted on.
"""
import json
import os

import pytest

from conftest import assert_string_literals, assert_successors_contract

DATA = os.path.join(os.path.dirname(__file__), "data")


def solutions(name):
    with open(os.path.join(DATA, f"{name}_solutions.json")) as handle:
        return {int(index): plan for index, plan in json.load(handle).items()}


# ------------------------------------------------------------------------------ epidemic

@pytest.fixture(scope="module")
def epidemic():
    pytest.importorskip("covasim", reason="covasim is not installed")
    from planiverse.environments.operational.epidemic.environment import EpidemicEnv

    env = EpidemicEnv()
    env.set_index(0)
    return env


def test_an_epidemic_replays_exactly_and_a_state_is_its_schedule(epidemic):
    from planiverse.environments.operational.epidemic.environment import EpidemicState, replay

    first = replay(epidemic.instance, ("open", "lockdown", "open"))
    second = replay(epidemic.instance, ("open", "lockdown", "open"))
    assert first == second and first["day"] == 21
    one = EpidemicState(("open",), 7, 10, 1, 1, 0, 12, 0, depth=1)
    two = EpidemicState(("open",), 7, 99, 9, 9, 3, 99, 0, depth=5)
    assert one == two, "the schedule is the identity; the readings follow from it"
    assert_string_literals(one)
    assert "chosen(0, open)" in one.literals and "week(1)" in one.literals


def test_the_hospital_constraint_and_the_budget_are_dead_ends(epidemic):
    from planiverse.environments.operational.epidemic.environment import EpidemicState

    inst = epidemic.instance
    over = EpidemicState(("open",) * 3, 21, 0, 0, inst["capacity"] + 1, 0, 0, 0, weeks=inst["weeks"])
    assert epidemic.is_terminal(over) and epidemic.successors(over) == []
    state, _ = epidemic.reset()
    assert all(action.level != "lockdown" for action in epidemic.get_actions(
        EpidemicState((), 0, 0, 0, 0, 0, 0, inst["budget"] - 1, weeks=inst["weeks"]))), \
        "a lockdown the budget cannot pay for is not offered"
    done = EpidemicState(("open",) * inst["weeks"], 7 * inst["weeks"], 0, 0, 0, inst["target"], 0, 0)
    assert epidemic.is_goal(done)
    late = EpidemicState(("open",) * inst["weeks"], 7 * inst["weeks"], 0, 0, 0, inst["target"] + 1, 0, 0)
    assert epidemic.is_terminal(late) and not epidemic.is_goal(late)


def test_epidemic_successors_and_the_bundled_outbreaks(epidemic):
    from planiverse.environments.operational.epidemic.environment import OUTBREAKS, EpidemicAction, EpidemicEnv

    state, info = epidemic.reset()
    children = epidemic.successors(state)
    assert_successors_contract(children)
    assert {str(a) for a, _ in children} <= {"open", "distancing", "lockdown"}
    assert EpidemicAction.parse("lockdown").cost() == 1
    plans = solutions("epidemic")
    assert len(OUTBREAKS) == 100 and sorted(plans) == list(range(100))
    for index in (0, 50):
        game = EpidemicEnv()
        game.set_index(index)
        assert game.validate(plans[index]), f"outbreak {index}"
        assert not game.validate(["open"] * game.instance["weeks"]), "leaving the town open fails"


# ------------------------------------------------------------------------------- traffic

@pytest.fixture(scope="module")
def traffic():
    pytest.importorskip("libsumo", reason="libsumo is not installed")
    from planiverse.environments.operational.traffic.environment import TrafficEnv

    env = TrafficEnv()
    env.set_index(0)
    yield env
    env.close()


def test_a_morning_replays_exactly_and_switches_take_effect(traffic):
    from planiverse.environments.operational.traffic.environment import DECISION, HOLD, TrafficAction

    state, _ = traffic.reset()
    held = traffic.__advance__(state, HOLD)
    assert held.time == DECISION and held.greens == state.greens
    switched = traffic.__advance__(state, TrafficAction("B1"))
    assert switched.greens != state.greens and switched.greens.count("B") == 1
    assert traffic.__advance__(state, TrafficAction("all")).greens == ("B",) * 9
    again = traffic.__advance__(traffic.simulate([HOLD])[0], TrafficAction("B1"))
    assert again == switched and again.arrived == switched.arrived and again.travel == switched.travel
    assert_string_literals(switched)
    assert "green(B1, B)" in switched.literals


def test_traffic_goal_and_dead_ends(traffic):
    from planiverse.environments.operational.traffic.environment import TrafficState

    inst = traffic.instance
    done = TrafficState((), inst["horizon"] - 15, inst["vehicles"], 0, 0, inst["target"], ("A",) * 9,
                        total=inst["vehicles"])
    assert traffic.is_goal(done)
    late = TrafficState((), inst["horizon"], inst["vehicles"] - 1, 1, 0, 0, ("A",) * 9, total=inst["vehicles"])
    assert traffic.is_terminal(late) and traffic.successors(late) == []
    slow = TrafficState((), 300, inst["vehicles"], 0, 0, inst["target"] + 1, ("A",) * 9, total=inst["vehicles"])
    assert traffic.is_terminal(slow) and not traffic.is_goal(slow), "everyone through but over the target"


def test_traffic_successors_and_the_bundled_mornings(traffic):
    from planiverse.environments.operational.traffic.environment import ACTIONS, MORNINGS, TrafficAction, TrafficEnv

    state, _ = traffic.reset()
    children = traffic.successors(state)
    assert_successors_contract(children)
    assert len(children) == len(ACTIONS) == 17
    assert TrafficAction.parse("switch(row1)").junctions() == ("A1", "B1", "C1")
    plans = solutions("traffic")
    assert len(MORNINGS) == 100 and sorted(plans) == list(range(100))
    for index in (0, 50):
        game = TrafficEnv()
        game.set_index(index)
        assert game.validate(plans[index]), f"morning {index}"
        game.close()


# ------------------------------------------------------------------------------ airspace

@pytest.fixture(scope="module")
def airspace():
    pytest.importorskip("bluesky", reason="bluesky is not installed")
    from planiverse.environments.operational.airspace.environment import AirspaceEnv

    env = AirspaceEnv()
    env.set_index(0)
    return env


@pytest.mark.slow
def test_a_sector_replays_exactly_and_separation_is_a_dead_end(airspace):
    from planiverse.environments.operational.airspace.environment import HOLD, AirspaceAction

    state, _ = airspace.reset()
    first = airspace.__advance__(state, HOLD)
    second = airspace.__advance__(airspace.simulate([])[0], HOLD)
    assert first == second and first.aircraft == second.aircraft
    straight = airspace.simulate([HOLD] * airspace.instance["horizon"])
    assert any(s.lost for s in straight), "flying straight loses separation on a bundled sector"
    lost = next(s for s in straight if s.lost)
    assert airspace.is_terminal(lost) and airspace.successors(lost) == []
    turned = airspace.__advance__(state, AirspaceAction.parse(f"turn({state.aircraft[0][0]}, right)"))
    assert not turned.aircraft[0][7], "a turn takes the aircraft off course"
    assert any(a.verb == "direct" for a in airspace.get_actions(turned))
    assert_string_literals(turned)


@pytest.mark.slow
def test_airspace_bundled_sectors_and_actions(airspace):
    from planiverse.environments.operational.airspace.environment import SECTORS, AirspaceAction, AirspaceEnv

    assert AirspaceAction.parse("direct(AC2)") == AirspaceAction("direct", "AC2")
    assert AirspaceAction.parse("hold").cost() == 1
    plans = solutions("airspace")
    assert len(SECTORS) == 100 and sorted(plans) == list(range(100))
    game = AirspaceEnv()
    game.set_index(0)
    assert game.validate(plans[0])
    state, _ = game.reset()
    assert_successors_contract(game.successors(state))


# ----------------------------------------------------------------------------- reservoir

@pytest.fixture(scope="module")
def reservoir():
    pytest.importorskip("pywr", reason="pywr is not installed")
    from planiverse.environments.operational.reservoir.environment import ReservoirEnv

    env = ReservoirEnv()
    env.set_index(0)
    return env


def test_a_year_is_a_value_state_and_the_river_and_city_are_dead_ends(reservoir):
    from planiverse.environments.operational.reservoir.environment import MONTHS, ReservoirAction, ReservoirState

    state, _ = reservoir.reset()
    a = reservoir.__advance__(reservoir.__advance__(state, ReservoirAction(2, "full")), ReservoirAction(4, "full"))
    b = reservoir.__advance__(reservoir.__advance__(state, ReservoirAction(4, "full")), ReservoirAction(2, "full"))
    assert a.month == b.month == 2
    if a.key == b.key:
        assert a == b, "two histories leaving the same volumes are one state"
    assert_string_literals(a)
    short = ReservoirState((), 50, 20, 0, 1.0, 0, (0, 0, 0, 0))
    assert reservoir.is_terminal(short) and reservoir.successors(short) == []
    inst = reservoir.instance
    done = ReservoirState(((None,),) * MONTHS, inst["reserves"][0], inst["reserves"][1], inst["target"], 0, 0, (0, 0, 0, 0))
    assert reservoir.is_goal(done)
    low = ReservoirState(((None,),) * MONTHS, inst["reserves"][0] - 1, inst["reserves"][1], 0, 0, 0, (0, 0, 0, 0))
    assert reservoir.is_terminal(low) and not reservoir.is_goal(low)


def test_reservoir_successors_and_the_bundled_years(reservoir):
    from planiverse.environments.operational.reservoir.environment import ACTIONS, YEARS, ReservoirAction, ReservoirEnv, reference_plans

    state, _ = reservoir.reset()
    children = reservoir.successors(state)
    assert_successors_contract(children)
    assert len(children) == len(ACTIONS) == 10
    assert ReservoirAction.parse("release(8, half)") == ReservoirAction(8, "half")
    plans = solutions("reservoir")
    assert len(YEARS) == 100 and sorted(plans) == list(range(100))
    for index in (0, 50, 99):
        game = ReservoirEnv()
        game.set_index(index)
        assert game.validate(plans[index]), f"year {index}"
        steady = [plan for plan in reference_plans()[:5]]
        assert not any(game.validate(plan) for plan in steady), f"year {index}: a steady release would do"
