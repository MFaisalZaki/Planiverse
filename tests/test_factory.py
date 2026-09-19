"""Tests for the factory environment.

The simulator is factory-sim's own and is not re-derived here. The tests pin what the
environment adds: that a scene reproduces from the instance's data, that a state is what the
factory holds rather than the path to it, the rules on where a drill and a furnace may go and
what a hand may do, the goal at the horizon, and that every bundled patch reaches its target by
the plan it was accepted on and by none of its poorer scripted lines.
"""
import json
import os

import pytest

pytest.importorskip("fsim", reason="factory-sim is not built")

from planiverse.environments.factory.environment import (  # noqa: E402
    AMOUNTS, DECISION, DIRECTIONS, MINE_COUNT, PATCHES, WAITS, FactoryAction, FactoryEnv,
    FactoryState, blueprint_of, footprint,
)

from conftest import assert_string_literals, assert_successors_contract  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="module")
def env():
    game = FactoryEnv()
    game.set_index(0)
    return game


def actions_of(env, state, verb, item=None):
    return [a for a in env.get_actions(state) if a.verb == verb and (item is None or a.item == item)]


# ---------------------------------------------------------------------------- the scene

def test_a_patch_becomes_the_scene_the_simulator_resets_to(env):
    blueprint = blueprint_of(env.instance)
    x0, y0, x1, y1 = env.instance["ore"]
    assert len(blueprint["resources"]) == (x1 - x0 + 1) * (y1 - y0 + 1)
    assert len(blueprint["entities"]) == len(env.instance["walls"])
    assert blueprint["character"]["inventory"]["coal"] == env.instance["coal"]
    state, info = env.reset()
    assert state.inventory == {"drill": env.instance["drills"], "furnace": env.instance["furnaces"],
                               "coal": env.instance["coal"]}
    assert state.tick == 0 and state.machines == () and info["generated"] is False
    assert len(state.ore) == len(blueprint["resources"])


def test_the_simulator_replays_the_same_decisions_to_the_same_state(env):
    state, _ = env.reset()
    plan = ["move(east)", "step(south)", "wait(10)"]
    first = env.simulate(plan)[-1]
    second = env.simulate(plan)[-1]
    assert first == second and hash(first) == hash(second)
    assert first.tick == DECISION * (1 + 1 + 10)


# ---------------------------------------------------------------------------- the state

def test_a_state_is_what_the_factory_holds_not_its_path(env):
    state, _ = env.reset()
    one = env.simulate(["move(east)", "wait(1)"])[-1]
    two = env.simulate(["wait(1)", "move(east)"])[-1]
    assert one == two and one.path != two.path
    assert one != env.simulate(["move(east)", "wait(10)"])[-1], "time is part of the state"
    assert_string_literals(one)
    assert f"holds(coal, {env.instance['coal']})" in one.literals
    assert "tick(60)" in one.literals and "made(0)" in one.literals
    assert one.depth == 2 and state.depth == 0


def test_a_built_line_shows_in_the_literals_and_the_text(env):
    with open(os.path.join(DATA, "factory_solutions.json")) as handle:
        plan = json.load(handle)["0"]
    trace = env.simulate(plan)
    built = next(s for s in trace if len(s.machines) == 2)
    drill = next(m for m in built.machines if m.kind == "drill")
    furnace = next(m for m in built.machines if m.kind == "furnace")
    assert f"drill({drill.x}, {drill.y}, {drill.facing})" in built.literals
    assert f"furnace({furnace.x}, {furnace.y})" in built.literals
    assert f"status(furnace, {furnace.x}, {furnace.y}, {furnace.status})" in built.literals
    assert "drill (" in str(built) and "furnace (" in str(built)
    assert len(env.render()) >= 1


# ---------------------------------------------------------------------------- the rules

def test_a_drill_goes_on_four_ore_tiles_and_a_furnace_where_it_drops(env):
    state, _ = env.reset()
    drills = actions_of(env, state, "place", "drill")
    assert drills and not actions_of(env, state, "place", "furnace"), "no drill, nowhere to drop"
    ore = set(env.tiles)
    assert all(set(footprint(a.x, a.y)) <= ore for a in drills)
    assert {a.direction for a in drills} == set(DIRECTIONS)
    placed = env.__advance__(state, drills[0])
    assert placed is not state and placed.inventory["drill"] == state.inventory["drill"] - 1
    furnaces = actions_of(env, placed, "place", "furnace")
    drill = placed.machines[0]
    assert len(furnaces) == 1 and (furnaces[0].x, furnaces[0].y) == {
        "north": (drill.x, drill.y - 2), "south": (drill.x, drill.y + 2),
        "east": (drill.x + 2, drill.y), "west": (drill.x - 2, drill.y)}[drill.facing]
    assert not any(set(footprint(a.x, a.y)) & set(footprint(drill.x, drill.y))
                   for a in actions_of(env, placed, "place", "drill")), "a drill's tiles are taken"


def test_a_machine_burns_the_coal_it_is_given_and_a_furnace_yields_plates(env):
    with open(os.path.join(DATA, "factory_solutions.json")) as handle:
        plan = [FactoryAction.parse(a) for a in json.load(handle)["0"]]
    trace = env.simulate(plan)
    fuelled = next(s for s in trace if s.machines and all(m.fuel > 0 or m.status == "working"
                                                          for m in s.machines))
    gives = [a for a in plan if a.verb == "give"]
    assert all(a.count in AMOUNTS for a in gives)
    assert fuelled.inventory.get("coal", 0) == env.instance["coal"] - sum(a.count for a in gives)
    later = trace[-2]
    assert later.produced > 0 and any(m.plates > 0 for m in later.machines if m.kind == "furnace")
    assert trace[-1].plates == later.plates + sum(m.plates for m in later.machines if m.kind == "furnace")


def test_the_hand_mines_ore_within_reach_and_feeds_a_furnace(env):
    state, _ = env.reset()
    assert not actions_of(env, state, "mine"), "the start is further from the ore than a hand reaches"
    x0, y0, x1, y1 = env.instance["ore"]
    tx, ty = x0, y0
    # walk towards the patch's corner until a tile is within reach
    for _ in range(12):
        mines = actions_of(env, state, "mine")
        if mines:
            break
        x, y = state.position[0] / 256, state.position[1] / 256
        dx, dy = tx + 0.5 - x, ty + 0.5 - y
        if abs(dx) >= abs(dy):
            verb, direction = ("move" if abs(dx) > 4.5 else "step"), ("east" if dx > 0 else "west")
        else:
            verb, direction = ("move" if abs(dy) > 4.5 else "step"), ("south" if dy > 0 else "north")
        state = env.__advance__(state, FactoryAction(verb, direction=direction))
    assert mines, "a walk to the patch brings ore within reach"
    mined = env.__advance__(state, mines[0])
    assert mined.inventory.get("ore") == MINE_COUNT
    assert mined.tick > state.tick + DECISION, "digging takes longer than a decision"
    index = env.tiles.index((mines[0].x, mines[0].y))
    assert mined.ore[index] == state.ore[index] - MINE_COUNT


def test_waits_and_walks_are_what_they_say(env):
    state, _ = env.reset()
    for wait in WAITS:
        assert env.__advance__(state, f"wait({wait})").tick == DECISION * wait
    east = env.__advance__(state, "move(east)")
    assert 4.4 < (east.position[0] - state.position[0]) / 256 < 4.5
    assert east.position[1] == state.position[1]
    step = env.__advance__(state, "step(south)")
    assert 1.0 < (step.position[1] - state.position[1]) / 256 < 1.1
    assert env.__advance__(state, "wait(999)") is state, "not a decision on offer"


def test_the_goal_is_the_target_by_the_horizon(env):
    target, horizon = env.instance["target"], env.instance["horizon"]
    late = FactoryState((), horizon + DECISION, (0, 0), {"plate": target}, (), (), (), target)
    assert not env.is_goal(late) and env.is_terminal(late)
    short = FactoryState((), horizon, (0, 0), {"plate": target - 1}, (), (), (), target - 1)
    assert env.is_terminal(short) and env.successors(short) == []
    done = FactoryState((), horizon - DECISION, (0, 0), {"plate": target}, (), (), (), target)
    assert env.is_goal(done) and not env.is_terminal(done) and env.successors(done) == []


def test_successors_obey_the_contract(env):
    state, _ = env.reset()
    children = env.successors(state)
    assert_successors_contract(children)
    assert all(child.tick >= state.tick + DECISION for _, child in children)
    assert children == env.successors(state), "expanding twice gives the same children"
    assert env.simulate([children[0][0]])[-1] == children[0][1]


def test_actions_parse_and_print():
    for text in ("move(north)", "step(west)", "place(drill, -3, 4, south)", "place(furnace, 0, 6)",
                 "give(-3, 4, coal, 5)", "give(0, 6, ore)", "take(0, 6)", "mine(-4, 3)", "wait(40)"):
        action = FactoryAction.parse(text)
        assert str(action) == text and action.cost() == 1
        assert FactoryAction.parse(str(action)) == action and hash(action) == hash(FactoryAction.parse(text))
    with pytest.raises(ValueError):
        FactoryAction.parse("build(3)")


def test_simulate_and_step_agree(env):
    plan = ["move(east)", "wait(10)"]
    trace = env.simulate(plan)
    env.reset()
    for action in plan:
        state, gained = env.step(action)
    assert state == trace[-1] and gained == 0 and len(env.render()) == 3


# ---------------------------------------------------------------------------- instances

def test_every_patch_reaches_its_target_by_its_plan_and_by_no_poorer_line():
    with open(os.path.join(DATA, "factory_solutions.json")) as handle:
        solutions = {int(index): plan for index, plan in json.load(handle).items()}
    assert len(PATCHES) == 100 and sorted(solutions) == list(range(100))
    for index in range(0, 100, 10):
        game = FactoryEnv()
        game.set_index(index)
        assert game.validate(solutions[index]), f"patch {index}"
        plans = game.reference_plans()
        assert plans[0][0] == game.instance["target"], f"patch {index}: the target is the best line"
        poorer = [plan for plates, plan in plans if plates < game.instance["target"]]
        assert all(not game.validate(plan) for plan in poorer[:3]), f"patch {index}"


def test_set_index_refuses_a_patch_that_is_not_there():
    game = FactoryEnv()
    for index in (-1, len(PATCHES), 999):
        with pytest.raises(IndexError, match="Invalid index"):
            game.set_index(index)


def test_a_generated_patch_reproduces_from_its_seed():
    game = FactoryEnv()
    instance = game.generate_instance(seed=9000)
    assert instance == PATCHES[0], "patch 0 is seed 9000"
    assert game.validate(game.witness) and game.witness_expansions > 0
    assert 0 < len(game.lines()) and game.instance["family"] == PATCHES[0]["family"]
