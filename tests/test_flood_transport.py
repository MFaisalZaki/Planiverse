"""The flood adaptation environment: a city drawn from a seed, the model taken from MAAT.

What is checked here is the model's arithmetic (the damage curves, the speeds, the storms, the
trip table), the shape of the decision problem (one decision a period, a target set by the
reference policy, a total that only grows), and what every bundled scenario promises: that
doing nothing misses the target and the reference plan meets it.
"""
import json

import pytest

from planiverse.environments import make
from planiverse.environments.operational.flood_transport.environment import (
    DESIGN_STORM, IMPASSABLE, MEASURES, SCENARIOS, FloodAction, FloodTransportEnv,
    damage_ratio, draw_city, sample_rain, speed, storm_depth, trip_distribution,
)
from planiverse.environments.generation import GenerationError, bounded_search, rng


@pytest.fixture
def env():
    game = FloodTransportEnv()
    game.set_index(0)
    return game


# ------------------------------------------------------------------------- the model

def test_the_damage_curves_are_the_assessments():
    assert damage_ratio("C1", 2.0) == pytest.approx(0.100)
    assert damage_ratio("C3", 1.0) == pytest.approx(0.004)
    assert damage_ratio("C5", 0.5) == pytest.approx(0.015)
    assert damage_ratio("C5", 0.75) == pytest.approx(0.020), "interpolated between the points"
    assert damage_ratio("C5", 0.0) == 0.0 and damage_ratio("C5", 20.0) == pytest.approx(0.065)


def test_a_measure_raises_the_road_or_halves_the_damage():
    assert damage_ratio("C5", 1.0, raise_by=MEASURES["elevate1"]["raise"]) == 0.0
    assert damage_ratio("C5", 1.5, raise_by=1.0) == pytest.approx(damage_ratio("C5", 0.5))
    assert damage_ratio("C5", 1.0, resist=MEASURES["resist50"]["resist"]) == pytest.approx(0.0125)


def test_a_flooded_road_slows_and_then_closes():
    assert speed(0.0) == 50.0
    assert speed(0.1) == pytest.approx(40.7, abs=0.1)
    assert 0 < speed(IMPASSABLE / 1000) < speed(0.2) < speed(0.1)
    assert speed(IMPASSABLE / 1000 + 0.001) == 0.0


def test_the_depth_ramps_from_the_lightest_storm_to_the_design_storm():
    assert storm_depth(1.0, DESIGN_STORM) == 1.0
    assert storm_depth(1.0, 20) == 0.0 and storm_depth(1.0, 8) == 0.0
    assert storm_depth(0.5, 90) == pytest.approx(0.25)


def test_storms_are_drawn_from_the_return_periods():
    """Rounded to four millimetres, within the table, and a function of the seed."""
    random_, _ = rng(3)
    storms = [sample_rain(random_, year) for year in range(60)]
    assert all(storm % 4 == 0 and 32 <= storm <= 120 for storm in storms)
    again, _ = rng(3)
    assert storms == [sample_rain(again, year) for year in range(60)]


def test_the_trip_table_meets_supply_and_demand():
    supply, demand = [1000, 2000, 500], [1500, 1000, 1000]
    cost = [[1.0, 0.5, 0.2], [0.5, 1.0, 0.5], [0.2, 0.5, 1.0]]
    trips = trip_distribution(supply, demand, cost)
    assert trips.sum(axis=1).tolist() == pytest.approx(supply, abs=3), "truncated to whole trips"
    assert trips.sum(axis=0).tolist() == pytest.approx(demand, abs=3)


def test_a_drawn_city_is_connected_and_floods_in_part():
    random_, _ = rng(5)
    city = draw_city(random_, 12, flood_share=0.5)
    assert len(city["zones"]) == 12 and city["edges"]
    reached, frontier = {"z00"}, ["z00"]
    while frontier:
        here = frontier.pop()
        for a, b, *_ in city["edges"]:
            for there in ((b,) if a == here else (a,) if b == here else ()):
                if there not in reached:
                    reached.add(there)
                    frontier.append(there)
    assert len(reached) == 12, "every zone can be reached by road"
    assert 0 < sum(1 for zone in city["zones"] if zone["design_depth"] > 0) < 12


# ---------------------------------------------------------------------- the instances

@pytest.mark.parametrize("index", range(len(SCENARIOS)))
def test_every_bundled_scenario_has_a_decision_in_it(index):
    """Doing nothing misses the target, and the reference plan meets it."""
    game = FloodTransportEnv()
    game.set_index(index)
    state, info = game.reset()
    assert info["generated"] is False and info["zones"] == SCENARIOS[index][1]["zones"]
    assert info["do_nothing"] > info["target"] >= info["reference"]
    assert not game.is_goal(state) and not game.is_terminal(state)
    plan = game.reference_plan()
    assert len(plan) == state.steps and game.validate(plan)
    waiting = game.simulate([FloodAction("wait")] * state.steps)
    assert game.is_terminal(waiting[-1]) and not game.is_goal(waiting[-1])


def test_a_decision_covers_a_period_of_years(env):
    state, info = env.reset()
    assert info["period"] == 5 and state.steps == 6 and info["years"] == 30
    one = env.successors(state)[0][1]
    assert one.year == 5 and one.step == 1
    assert one.rain == DESIGN_STORM, "the worst storm of the period is kept for the record"
    env.set_instance({**env.instance, "years": 12, "period": 5})
    state, _ = env.reset()
    assert state.steps == 3
    end = env.simulate(["wait", "wait", "wait"])[-1]
    assert end.year == 12, "the last period covers what is left of the horizon"


def test_the_total_only_grows_and_the_target_ends_the_search(env):
    state, _ = env.reset()
    trace = env.simulate([FloodAction("wait")] * state.steps)
    costs = [s.cost for s in trace]
    assert costs == sorted(costs) and costs[0] == 0.0
    over = next(s for s in trace if s.terminal)
    assert env.successors(over) == []
    assert "terminal-state" in over.literals and "goal-reached" not in over.literals


def test_successors_offer_each_measure_once_per_flooding_zone(env):
    state, _ = env.reset()
    offered = [action for action, _ in env.successors(state)]
    flooded = env.__city__()["flooded"]
    assert offered[0].kind == "wait"
    assert {action.zone for action in offered[1:]} == set(flooded)
    dry = [zone["id"] for zone in env.instance["zones"] if zone["design_depth"] == 0]
    assert not any(action.zone in dry for action in offered), "a dry zone is not a candidate"
    action, child = next((a, s) for a, s in env.successors(state) if a.kind != "wait")
    assert action.zone in child.protected
    assert action.zone not in {a.zone for a, _ in env.successors(child)}
    assert child.spent > 0 and f"protected({action.zone}, elevate1)" in child.literals


def test_a_protected_zone_costs_less_in_a_storm(env):
    state, _ = env.reset()
    waited = env.successors(state)[0][1]
    protected = [s for a, s in env.successors(state) if a.kind != "wait"]
    assert all(s.damage <= waited.damage for s in protected)
    assert any(s.damage + s.delay < waited.damage + waited.delay for s in protected)


def test_states_are_identified_by_their_path(env):
    state, _ = env.reset()
    first = env.simulate(["wait", "elevate1(z03)"])[-1]
    second = env.simulate(["wait", "elevate1(z03)"])[-1]
    other = env.simulate(["elevate1(z03)", "wait"])[-1]
    assert first == second and hash(first) == hash(second)
    assert first != other, "the same measure a period earlier is a different decision"
    assert other.cost <= first.cost, "and it protected the zone through one more period"


def test_actions_parse_from_their_names():
    assert FloodAction.parse("elevate1(z03)") == FloodAction("elevate1", "z03")
    assert FloodAction.parse("wait") == FloodAction("wait") and str(FloodAction("wait")) == "wait"
    with pytest.raises(ValueError, match="unknown measure"):
        FloodAction("flee", "z01")


def test_get_actions_and_step(env):
    state, _ = env.reset()
    actions = env.get_actions()
    assert actions[0].kind == "wait" and len(actions) == 1 + len(env.__city__()["flooded"])
    state, reward = env.step("wait")
    assert state.step == 1 and reward < 0, "a period's costs come back as a negative reward"
    assert len(env.render()) == 2


def test_search_finds_a_plan_that_meets_the_target(env):
    from planiverse.benchmark.measures import flood_transport

    env.reset()
    outcome = bounded_search(env, 2000, progress=flood_transport)
    assert outcome.plan is not None and env.validate(outcome.plan)
    assert env.simulate(outcome.plan)[-1].cost <= env.reset()[1]["target"]


# ---------------------------------------------------------------------- the generator

def test_the_same_seed_draws_the_same_city():
    first, second = FloodTransportEnv(), FloodTransportEnv()
    a = first.generate_instance(seed=7, zones=10, years=30, rain="design")
    b = second.generate_instance(seed=7, zones=10, years=30, rain="design")
    assert a == b and a["seed"] == 7 and len(a["rain"]) == 30
    assert first.reset()[0].literals == second.reset()[0].literals
    assert first.witness == second.witness and first.validate(first.witness)
    assert first.reset()[1]["generated"] is True


def test_an_instance_is_plain_data_and_is_measured_on_reset():
    env = FloodTransportEnv()
    drawn = env.generate_instance(seed=2, zones=8, years=30, rain="design")
    restored = json.loads(json.dumps(drawn))
    del restored["reference"], restored["do_nothing"]
    other = FloodTransportEnv()
    other.set_instance(restored)
    state, info = other.reset()
    assert info["reference"] == pytest.approx(drawn["reference"])
    assert state.literals == env.reset()[0].literals


def test_an_instance_needs_a_storm_for_every_year():
    env = FloodTransportEnv()
    env.set_index(0)
    with pytest.raises(ValueError, match="storm for every year"):
        env.set_instance({**env.instance, "years": 99})
    with pytest.raises(ValueError, match="needs 'rain'"):
        env.set_instance({"zones": [], "edges": []})


def test_the_generator_refuses_a_horizon_too_short_to_pay_for_anything():
    """A measure pays for itself over decades, so a short horizon has no decision in it."""
    with pytest.raises(GenerationError):
        FloodTransportEnv().generate_instance(seed=0, zones=8, years=5, rain="design", attempts=5)


def test_make_builds_it_by_name():
    env = make("flood_transport", index=2)
    state, info = env.reset()
    assert info["zones"] == 12 and env.successors(state)
