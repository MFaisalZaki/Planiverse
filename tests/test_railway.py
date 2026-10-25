"""Tests for the railway environment: train dispatching on Flatland.

Flatland's movement rules are its own and are not re-derived here. What the tests pin is the
reduction the environment makes of a dispatching problem to a goal: a deadlock and the horizon
are dead ends, the train-steps are carried in the state and bounded by the target, a state is
what the simulator holds rather than the route to it, and the target is what the best scripted
rule achieves, so the bundled timetables are solvable by the plans they were accepted on.
"""
import json
import os

import pytest

pytest.importorskip("flatland", exc_type=ImportError, reason="flatland-rl is not installed")

from conftest import assert_string_literals, assert_successors_contract  # noqa: E402
from planiverse.environments.operational.railway.environment import (  # noqa: E402
    PERIOD, TIMETABLES, WAIT, RailwayAction, RailwayEnv, RailwayState, reference_policies,
)

DATA = os.path.join(os.path.dirname(__file__), "data")


def solutions():
    with open(os.path.join(DATA, "railway_solutions.json")) as handle:
        return {int(index): plan for index, plan in json.load(handle).items()}


@pytest.fixture(scope="module")
def railway():
    env = RailwayEnv()
    env.set_index(0)
    return env


def test_the_actions_parse_and_cost_one(railway):
    assert RailwayAction.parse("depart(3)") == RailwayAction(3)
    assert RailwayAction.parse("wait") == WAIT and str(WAIT) == "wait"
    assert RailwayAction(2).cost() == 1
    with pytest.raises(ValueError):
        RailwayAction.parse("hold(1)")
    trains = railway.instance["trains"]
    assert railway.get_actions() == [RailwayAction(i) for i in range(trains)] + [WAIT]


def test_the_start_has_every_train_waiting(railway):
    state, info = railway.reset()
    assert_string_literals(state)
    assert info["generated"] is False and info["target"] == railway.instance["target"]
    assert state.step == 0 and state.train_steps == 0 and state.released == ()
    assert all(status == "waiting" for status, _, _ in state.trains)
    assert not railway.is_goal(state) and not railway.is_terminal(state)
    assert set(state.readings()) >= {"waiting", "running", "stopped", "arrived", "train-steps"}
    children = railway.successors(state)
    assert_successors_contract(children)
    assert [str(a) for a, _ in children] == [f"depart({i})" for i in range(len(state.trains))] + ["wait"]


def test_the_picture_draws_the_track_and_the_trains_on_it(railway):
    from planiverse.rendering.text import missing_glyphs

    state, _ = railway.reset()
    picture = str(state)
    assert isinstance(picture, str) and picture and not missing_glyphs(picture)
    assert "◎" in picture and "┼" in picture and not any(arrow in picture for arrow in "▲▶▼◀")
    out = railway.simulate(["depart(0)", "wait"])[-1]
    assert any(arrow + "0" in str(out) for arrow in "▲▶▼◀"), "train 0 is out on the line"
    assert len(str(out).splitlines()) == len(picture.splitlines()), "the map keeps its size"


def test_expanding_twice_gives_the_same_children(railway):
    state = railway.simulate(["depart(0)", "wait", "wait"])[-1]
    first = [(str(a), s.key, s.literals) for a, s in railway.successors(state)]
    second = [(str(a), s.key, s.literals) for a, s in railway.successors(state)]
    assert first == second and first
    again = railway.simulate(["depart(0)", "wait", "wait"])[-1]
    assert again == state and again.literals == state.literals


def test_a_state_is_what_the_simulator_holds_not_the_route_to_it(railway):
    one = railway.simulate(["depart(0)", "depart(1)", "wait"])[-1]
    two = railway.simulate(["depart(1)", "depart(0)", "wait"])[-1]
    assert one == two and hash(one) == hash(two) and one.depth == two.depth
    held = railway.simulate(["depart(0)", "wait"])[-1]
    assert held.step == PERIOD and "released(0)" in held.literals
    assert any(literal.startswith("at(0, ") for literal in held.literals)


def test_restoring_a_snapshot_meets_the_states_of_a_straight_run(railway):
    """The plan replayed through snapshots and the same moves given to one Flatland run
    without restoring put every train in the same cell at every decision."""
    plan = [RailwayAction.parse(a) for a in solutions()[0]]
    trace = railway.simulate(plan)
    straight = RailwayEnv()
    straight.set_instance(railway.instance)
    straight.reset()
    flatland = getattr(straight, "__env__")
    released, k = set(), 0
    for action, state in zip(plan, trace[1:]):
        if action.train is not None:
            released.add(action.train)
            continue
        everyone, steps = len(released) == len(flatland.agents), 0
        while not flatland.dones["__all__"]:
            straight._step(released)
            steps += 1
            if straight._deadlocked() or (not everyone and steps >= PERIOD):
                break
        assert flatland._elapsed_steps == state.step
        cells = [a.current_configuration and a.current_configuration[0] for a in flatland.agents]
        assert cells == [cell for _, cell, _ in state.trains]
        k += 1
    assert k == plan.count(WAIT)


def test_a_deadlock_is_a_dead_end(railway):
    """Every train let go at once meets head-on on instance 0's single track."""
    trains = railway.instance["trains"]
    plan, last = railway.run_rule(reference_policies(trains, list(range(trains)))[0][1])
    assert last is None
    end = railway.simulate(plan)[-1]
    assert end.deadlocked and railway.is_terminal(end) and not railway.is_goal(end)
    assert railway.successors(end) == []
    assert railway.__advance__(end, WAIT) is end


def test_the_target_and_the_horizon_are_dead_ends(railway):
    inst = railway.instance
    trains = [("arrived", None, None)] * inst["trains"]
    done = RailwayState(inst["horizon"], trains, range(inst["trains"]), inst["target"])
    assert railway.is_goal(done) and not railway.is_terminal(done)
    over = RailwayState(40, trains, range(inst["trains"]), inst["target"] + 1)
    assert railway.is_terminal(over) and not railway.is_goal(over)
    late = RailwayState(inst["horizon"], [("running", (1, 1), 0)] + trains[1:], range(inst["trains"]), 10)
    assert railway.is_terminal(late) and railway.successors(late) == []


def test_the_bundled_timetables_are_solved_by_their_plans():
    plans = solutions()
    assert len(TIMETABLES) == 100 and sorted(plans) == list(range(100))
    assert all(entry["solved_at"] == len(plans[k]) for k, entry in enumerate(TIMETABLES))
    for index in (0, 50):
        env = RailwayEnv()
        env.set_index(index)
        assert env.validate(plans[index]), f"timetable {index}"
        assert env.simulate(plans[index])[-1].train_steps <= env.instance["target"]
    with pytest.raises(IndexError):
        RailwayEnv().set_index(100)


def test_the_same_seed_gives_the_same_timetable_and_it_round_trips():
    first, second = RailwayEnv(), RailwayEnv()
    instance = first.generate_instance(seed=7, trains=3, size=25)
    assert instance == second.generate_instance(seed=7, trains=3, size=25)
    assert first.witness and first.validate(first.witness)
    assert first.witness_expansions == len(reference_policies(3, [0, 1, 2]))
    assert instance["solved_at"] == len(first.witness)
    restored = RailwayEnv()
    restored.set_instance(json.loads(json.dumps(instance)))
    assert restored.reset()[0].literals == first.reset()[0].literals
    assert restored.reset()[1]["generated"] is True
    with pytest.raises(ValueError):
        restored.set_instance({"seed": 1})
