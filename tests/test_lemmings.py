"""Tests for the Lemmings-like.

The crowd's rules are pinned one skill at a time on small caves: walking, the step of one
and the wall of two, the fall that kills, the digger, the basher, the builder and the
blocker. Then what the environment adds: a skill costs one of the level's, the quota and the
dead ends, and that every bundled level is saved by the plan it was accepted on and by
walking unaided by none.
"""
import json
import os

import pytest

from planiverse.environments.games.lemmings.environment import (
    DECISION, FATAL_FALL, LEVELS, MAX_TICKS, SKILLS, LemmingsAction, LemmingsEnv,
    LemmingsState, WAIT, draw_level, run_ticks,
)

from conftest import assert_string_literals, assert_successors_contract

DATA = os.path.join(os.path.dirname(__file__), "data")


def crowd(rows, lemmings, ticks, entrance=(0, 0), exit_=(-1, -1), total=0):
    """Run `lemmings` (already out) for `ticks` on `rows` with nobody else entering."""
    return run_ticks(tuple(rows), lemmings, total, 0, 0, 0, None, entrance, exit_, total, ticks)


@pytest.fixture
def env():
    game = LemmingsEnv()
    game.set_index(0)
    return game


# ------------------------------------------------------------------------------- the crowd

def test_a_walker_walks_climbs_a_step_and_turns_at_a_wall():
    rows = ["         ",
            "      .  ",
            "   .  .  ",
            "#########"]
    _, out, *_ = crowd(rows, [[1, 2, 1, "walker", 0]], 2)
    assert out[0][:2] == [3, 1], "one step up onto the single block"
    _, out, *_ = crowd(rows, [[4, 2, 1, "walker", 0]], 2)
    assert out[0][:3] == [5, 2, -1], "a wall of two turns it round"


def test_a_fall_of_more_than_the_limit_kills():
    tall = [" " * 5 for _ in range(FATAL_FALL + 4)] + ["#####"]
    _, out, _, _, lost, _ = crowd(tall, [[2, 0, 1, "walker", 0]], FATAL_FALL + 6)
    assert lost == 1 and out == []
    short = [" " * 5 for _ in range(3)] + ["#####"]
    _, out, _, _, lost, _ = crowd(short, [[2, 0, 1, "walker", 0]], 6)
    assert lost == 0 and out[0][3] == "walker"


def test_a_digger_digs_down_through_earth_and_stops_at_rock():
    rows = ["     ", "  .  ", "  .  ", "#####"]
    grid, out, *_ = crowd(rows, [[2, 0, 1, "digger", 0]], 3)
    assert grid[1][2] == " " and grid[2][2] == " " and out[0][3] == "walker"


def test_a_basher_tunnels_forward_two_high_and_stops_at_rock():
    rows = ["       ", "  ...# ", "  ...# ", "#######"]
    grid, out, *_ = crowd(rows, [[1, 2, 1, "basher", 0]], 4)
    assert grid[2][2:5] == "   " and grid[1][2:5] == "   " and grid[2][5] == "#"
    assert out[0][3] == "walker"


def test_a_builder_lays_a_stair_up_and_forward():
    rows = ["        " for _ in range(5)] + ["########"]
    grid, out, *_ = crowd(rows, [[1, 4, 1, "builder", 0]], 3)
    assert grid[4][2] == "." and grid[3][3] == "." and out[0][:2] == [4, 1]


def test_a_blocker_stands_and_turns_the_others():
    rows = ["        ", "########"]
    _, out, *_ = crowd(rows, [[4, 0, 1, "blocker", 0], [1, 0, 1, "walker", 0]], 4)
    assert out[0][:2] == [4, 0] and out[1][2] == -1


def test_the_exit_saves_and_the_crowd_enters_one_at_a_time():
    rows = ["E  X", "####"]
    _, out, released, saved, _, tick = run_ticks(tuple(rows), [], 0, 0, 0, 0, None, (0, 0), (3, 0), 3, 30)
    assert released == 3 and saved == 3 and out == []


def test_the_crowd_is_deterministic():
    rows = tuple(LEVELS[0]["rows"])
    entrance, exit_ = tuple(LEVELS[0]["entrance"]), tuple(LEVELS[0]["exit"])
    once = run_ticks(rows, [], 0, 0, 0, 0, None, entrance, exit_, 6, 60)
    again = run_ticks(rows, [], 0, 0, 0, 0, None, entrance, exit_, 6, 60)
    assert once == again


# ------------------------------------------------------------------------------- the rules

def test_a_skill_costs_one_of_the_levels_and_only_a_walker_takes_it(env):
    state = env.simulate([WAIT] * 2)[-1]
    walkers = [k for k in state.alive if state.lemmings[k][3] == "walker"]
    assert walkers, "after two decisions somebody is walking"
    skill = next(skill for skill, count in state.skills if count > 0)
    after = env.__advance__(state, LemmingsAction(skill, walkers[0]))
    assert dict(after.skills)[skill] == dict(state.skills)[skill] - 1
    assert after.tick == state.tick + DECISION
    assert env.__advance__(state, LemmingsAction(skill, 99)) == state, "no such lemming"
    none = next((skill for skill, count in state.skills if count == 0), None)
    if none is not None:
        assert env.__advance__(state, LemmingsAction(none, walkers[0])) == state, "none left"


def test_the_goal_and_the_dead_ends(env):
    quota = env.instance["quota"]
    rows = env.instance["rows"]
    assert env.is_goal(LemmingsState(rows, (), 8, quota, 0, env.instance["skills"], 100))
    hopeless = LemmingsState(rows, (), env.instance["lemmings"], quota - 2, 2, env.instance["skills"], 100)
    assert env.is_terminal(hopeless), "everyone out and too few left to reach the quota"
    late = LemmingsState(rows, ((3, 3, 1, "walker", 0),), 1, 0, 0, env.instance["skills"], MAX_TICKS)
    assert env.is_terminal(late) and env.successors(late) == []


def test_successors_obey_the_contract(env):
    state = env.simulate([WAIT] * 2)[-1]
    children = env.successors(state)
    assert_successors_contract(children)
    assert_string_literals(state)
    assert WAIT in [action for action, _ in children]


def test_states_hash_by_everything_but_depth(env):
    state, _ = env.reset()
    twin = LemmingsState(state.rows, state.lemmings, 0, 0, 0, dict(state.skills), 0, state.quota, depth=5)
    assert state == twin and hash(state) == hash(twin)


def test_actions_parse_and_print():
    assert LemmingsAction.parse("assign(basher,2)") == LemmingsAction("basher", 2)
    assert LemmingsAction.parse("wait") == WAIT and WAIT.cost() == 1
    with pytest.raises(ValueError):
        LemmingsAction("climber", 0)


def test_simulate_and_step_agree(env):
    plan = [WAIT, WAIT, WAIT]
    trace = env.simulate(plan)
    env.reset()
    for action in plan:
        state, _ = env.step(action)
    assert state == trace[-1] and len(env.render()) == 4


# ------------------------------------------------------------------------------- instances

def test_every_level_is_saved_by_its_plan_and_not_by_walking():
    with open(os.path.join(DATA, "lemmings_solutions.json")) as handle:
        solutions = {int(index): plan for index, plan in json.load(handle).items()}
    assert len(LEVELS) == 100 and sorted(solutions) == list(range(100))
    for index in range(0, 100, 5):
        game = LemmingsEnv()
        game.set_index(index)
        assert game.validate(solutions[index]), f"level {index}"
        assert not game.validate([WAIT] * (MAX_TICKS // DECISION)), f"level {index} saves itself"


def test_a_drawn_level_has_an_entrance_an_exit_and_skills():
    import random
    instance = draw_level(random.Random(4))
    rows = instance["rows"]
    ex, ey = instance["exit"]
    assert rows[instance["entrance"][1]][instance["entrance"][0]] == "E" and rows[ey][ex] == "X"
    assert set(instance["skills"]) == set(SKILLS) and instance["quota"] <= instance["lemmings"]


def test_set_index_refuses_a_level_that_is_not_there():
    game = LemmingsEnv()
    for index in (-1, len(LEVELS), 999):
        with pytest.raises(IndexError, match="Invalid index"):
            game.set_index(index)


def test_a_generated_level_reproduces_from_its_seed():
    game = LemmingsEnv()
    instance = game.generate_instance(seed=6000)
    assert instance["rows"] == LEVELS[0]["rows"], "level 0 is seed 6000"
    assert game.validate(game.witness)
    assert LemmingsEnv().generate_instance(seed=6000) == instance
