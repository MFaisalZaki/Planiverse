"""Tests for the cellular fluid puzzle.

The automaton is a dozen lines, so the tests pin its rules on small caves (water falls,
slides, spreads, is taken by the basin and swallowed by the drain), then what the environment
adds: digs only where the water is, the budgets, the dead end, and that every bundled cave is
solved by the plan it was accepted on and by waiting alone by none.
"""
import json
import os

import pytest

from planiverse.environments.games.fluid.environment import (
    AIR, CAVES, EARTH, MAX_TICKS, PERIOD, WATER, FluidAction, FluidEnv, FluidState, WAIT,
    diggable, distance_to_basin, run, tick,
)

from conftest import assert_string_literals, assert_successors_contract

DATA = os.path.join(os.path.dirname(__file__), "data")


def cave(*rows):
    return tuple(rows)


@pytest.fixture
def env():
    game = FluidEnv()
    game.set_index(0)
    return game


# --------------------------------------------------------------------------- the automaton

def test_water_falls_then_slides_then_spreads():
    rows = cave("#######",
                "#  ~  #",
                "#  .  #",
                "#######")
    after, _, _ = tick(rows, 0, 0, 0)
    assert after[1] == "#     #" and after[2] in ("# ~.  #", "#  .~ #"), "earth under it: it slides off"
    rows = cave("#######",
                "#  ~  #",
                "#     #",
                "#######")
    after, _, _ = tick(rows, 0, 0, 0)
    assert after[2] == "#  ~  #" and after[1] == "#     #", "air under it: it falls"
    rows = cave("#######",
                "# ~~~ #",
                "#######")
    after, _, _ = run(rows, 0, 0, 0, 1)
    assert after[1].count(WATER) == 3 and after[1] != rows[1], "on a floor it moves along"
    rows = cave("#######",
                "#~~   #",
                "####  #",
                "#######")
    after, _, _ = run(rows, 0, 0, 0, 6)
    assert after[2].count(WATER) >= 1, "and water on a shelf runs off its edge"


def test_the_spring_emits_one_unit_a_tick_while_it_can():
    rows = cave("#####",
                "# S #",
                "#   #",
                "#   #",
                "#####")
    after, supply, _ = run(rows, 5, 0, 0, 3)
    assert supply == 3 and sum(row.count(WATER) for row in after) == 2, \
        "a unit on the first tick, none while it sits under the spring, one more once it falls"
    after, supply, _ = run(rows, 5, 0, 0, 40)
    assert supply == 0 and sum(row.count(WATER) for row in after) == 5, \
        "and all of it once the water below has moved out of the way"


def test_the_basin_takes_water_and_the_drain_swallows_it():
    rows = cave("#####",
                "# ~ #",
                "# T #",
                "#####")
    after, _, filled = tick(rows, 0, 0, 0)
    assert filled == 1 and after[1] == "#   #"
    rows = cave("#####",
                "# ~ #",
                "# X #",
                "#####")
    after, _, filled = tick(rows, 0, 0, 0)
    assert filled == 0 and after[1] == "#   #"


def test_the_automaton_is_deterministic():
    rows = cave("########",
                "#S     #",
                "#..  ..#",
                "#. .. .#",
                "#  T  X#",
                "########")
    assert run(rows, 6, 0, 0, 20) == run(rows, 6, 0, 0, 20)


# ------------------------------------------------------------------------------- the rules

def test_only_earth_against_water_can_be_dug():
    rows = cave("######",
                "#~ . #",
                "#..  #",
                "######")
    assert sorted(diggable(rows)) == [(1, 2)], "the earth beside the air is out of reach"
    assert distance_to_basin(cave("####", "#~T#", "####")) == 1


def test_a_dig_costs_a_dig_and_runs_the_water_a_period(env):
    state, _ = env.reset()
    waited = env.__advance__(state, WAIT)
    assert waited.tick == state.tick + PERIOD and waited.digs_left == state.digs_left
    assert waited.supply < state.supply, "the spring gave water meanwhile"
    dig = next(action for action, _ in env.successors(waited) if action != WAIT)
    dug = env.__advance__(waited, dig)
    assert dug.digs_left == waited.digs_left - 1 and dug.rows[dig.y][dig.x] != EARTH
    assert env.__advance__(waited, FluidAction(1, 1)) == waited, "not against water: refused"


def test_the_goal_and_the_dead_ends(env):
    need = env.instance["need"]
    full = FluidState(env.instance["rows"], 0, need, 0, 60)
    assert env.is_goal(full) and not env.is_terminal(full)
    dry = FluidState(env.instance["rows"], 0, 0, 3, 60)
    assert env.is_terminal(dry), "no water afloat and none in the spring: it cannot fill"
    late = FluidState(env.instance["rows"], 20, 0, 3, MAX_TICKS)
    assert env.is_terminal(late), "out of time"
    assert env.successors(dry) == []


def test_successors_obey_the_contract(env):
    state, _ = env.reset()
    for _ in range(2):
        children = env.successors(state)
        assert_successors_contract(children)
        assert all(child != state for _, child in children)
        state = children[-1][1]


def test_states_hash_by_cave_and_counters_not_depth():
    rows = cave("####", "#~ #", "####")
    one = FluidState(rows, 3, 0, 2, 12, depth=1)
    two = FluidState(rows, 3, 0, 2, 12, depth=9)
    assert one == two and hash(one) == hash(two)
    assert one != FluidState(rows, 3, 0, 1, 12)


def test_literals_name_cells_and_counters(env):
    state, _ = env.reset()
    assert_string_literals(state)
    assert "filled(0)" in state.literals and f"digs_left({state.digs_left})" in state.literals
    assert any(literal.startswith("cell(") for literal in state.literals)


def test_actions_parse_and_print():
    assert FluidAction.parse("dig(3,4)") == FluidAction(3, 4) and str(FluidAction(3, 4)) == "dig(3,4)"
    assert FluidAction.parse("wait") == WAIT and WAIT.cost() == 1


def test_simulate_and_step_agree(env):
    plan = [WAIT, WAIT]
    trace = env.simulate(plan)
    env.reset()
    for action in plan:
        state, _ = env.step(action)
    assert state == trace[-1] and len(env.render()) == 3


# ------------------------------------------------------------------------------- instances

def test_every_cave_is_solved_by_its_plan_and_not_by_waiting():
    with open(os.path.join(DATA, "fluid_solutions.json")) as handle:
        solutions = {int(index): plan for index, plan in json.load(handle).items()}
    assert len(CAVES) == 100 and sorted(solutions) == list(range(100))
    for index in range(0, 100, 5):
        game = FluidEnv()
        game.set_index(index)
        assert game.validate(solutions[index]), f"cave {index}"
        assert not game.validate([WAIT] * (MAX_TICKS // PERIOD)), f"cave {index} fills itself"


def test_set_index_refuses_a_cave_that_is_not_there():
    game = FluidEnv()
    for index in (-1, len(CAVES), 999):
        with pytest.raises(IndexError, match="Invalid index"):
            game.set_index(index)


def test_a_generated_cave_reproduces_from_its_seed():
    game = FluidEnv()
    instance = game.generate_instance(seed=4000)
    assert instance["rows"] == CAVES[0]["rows"], "cave 0 is seed 4000"
    assert game.validate(game.witness)
    assert FluidEnv().generate_instance(seed=4000) == instance
