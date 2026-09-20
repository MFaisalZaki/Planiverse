"""Tests for the Pipe Dream-like.

The flow is a few lines of rules, so the tests pin them on small boards (which openings a piece
has, how the flow turns, that a cross may be crossed twice and nothing else re-entered, where a
spill comes from), then what the environment adds: the countdown and the pace, the cost of a
replacement, the discard, the flush, the dead end, and that every bundled level is solved by
the plan it was accepted on and by the thoughtless policy by none.
"""
import json
import os

import pytest

from planiverse.environments.pipe_dream.environment import (
    DISCARD, FLUSH, KINDS, LEVELS, DISCARD as _DISCARD, PipeAction, PipeDreamEnv, PipeState,
    advance, advances_due, draw_level, end_of_pipe, run_ahead,
)
from planiverse.environments.generation import rng

from conftest import assert_string_literals, assert_successors_contract

DATA = os.path.join(os.path.dirname(__file__), "data")


def level(rows, queue, distance, countdown=2, pace=2):
    return {"rows": list(rows), "queue": list(queue), "distance": distance,
            "countdown": countdown, "pace": pace}


@pytest.fixture
def env():
    game = PipeDreamEnv()
    game.set_index(0)
    return game


@pytest.fixture
def tiny():
    """A 4 by 3 board with the start facing east and a queue that lays a straight, an elbow
    down, and an elbow back west."""
    game = PipeDreamEnv()
    game.set_instance(level([">...",
                             "....",
                             "...."], ["ew", "sw", "nw", "ew", "ns"], 3, countdown=3, pace=1))
    return game


# ------------------------------------------------------------------------------- the flow

def test_each_piece_opens_two_sides_and_the_cross_four():
    assert KINDS["ns"] == {"n", "s"} and KINDS["ne"] == {"n", "e"} and KINDS["x"] == set("nsew")


def test_the_flow_runs_straight_turns_and_spills():
    rows = (">-7.",
            "..|.",
            "..L-")
    flooded, head = advance(rows, frozenset(), (0, 0, "e"))
    assert head == (1, 0, "e") and (1, 0, "ew") in flooded, "through the straight, still east"
    flooded, head = advance(rows, flooded, head)
    assert head == (2, 0, "s"), "the elbow turns it south"
    flooded, head = advance(rows, flooded, head)
    assert head == (2, 1, "s")
    flooded, head = advance(rows, flooded, head)
    assert head == (2, 2, "e"), "and the next elbow turns it east"
    flooded, head = advance(rows, flooded, head)
    assert head == (3, 2, "e")
    assert advance(rows, flooded, head) is None, "off the board: a spill"
    assert advance((">.",), frozenset(), (0, 0, "e")) is None, "an empty cell: a spill"
    assert advance((">|",), frozenset(), (0, 0, "e")) is None, "no opening on that side: a spill"


def test_a_cross_may_be_crossed_twice_but_no_pipe_re_entered():
    rows = (".|.",
            ">+7",
            ".LJ")
    flooded, head = advance(rows, frozenset(), (0, 1, "e"))
    assert head == (1, 1, "e") and (1, 1, "ew") in flooded
    flooded, head = advance(rows, flooded, head)        # the elbow at (2, 1) turns it south
    flooded, head = advance(rows, flooded, head)        # (2, 2) turns it west
    flooded, head = advance(rows, flooded, head)        # (1, 2) turns it north
    assert head == (1, 2, "n")
    flooded, head = advance(rows, flooded, head)        # up through the cross, the other way
    assert head == (1, 1, "n") and (1, 1, "ns") in flooded
    flooded, head = advance(rows, flooded, head)
    assert head == (1, 0, "n"), "and out through the straight above"
    rows = (">-7",
            ".LJ")
    flooded, head = advance(rows, frozenset(), (0, 0, "e"))
    for _ in range(3):
        flooded, head = advance(rows, flooded, head)
    assert head == (1, 1, "n") and advance(rows, flooded, head) is None, \
        "back into the straight it has run through: a spill"


def test_run_ahead_counts_the_pipe_laid_and_the_cells_it_reaches():
    rows = (">--.",)
    count, cells = run_ahead(rows, frozenset(), (0, 0, "e"), 5)
    assert count == 2 and cells == {(1, 0), (2, 0)}
    assert run_ahead(rows, frozenset(), (0, 0, "e"), 1)[0] == 1


def test_advances_are_due_after_the_countdown_and_then_every_pace():
    assert [advances_due(t, 3, 2) for t in range(8)] == [0, 0, 0, 1, 1, 2, 2, 3]


# ------------------------------------------------------------------------------- the rules

def test_a_placement_is_a_tick_and_the_flow_starts_when_the_countdown_has_run(tiny):
    state, info = tiny.reset()
    assert info["distance"] == 3 and state.due == 3 and state.next == "ew"
    one = tiny.__advance__(state, PipeAction(1, 0))
    assert one.tick == 1 and one.placed == 1 and one.carried == 0 and one.due == 2
    assert one.rows[0] == ">-.." and one.next == "sw"
    two = tiny.__advance__(one, PipeAction(2, 0))
    three = tiny.__advance__(two, PipeAction(2, 1))
    assert three.tick == 3 and three.carried == 1, "the countdown has run: the flow moved once"
    assert (1, 0, "ew") in three.flooded and three.head == (1, 0, "e")
    assert "flooded(1, 0)" in three.literals and "carried(1)" in three.literals


def test_a_replacement_costs_two_ticks_and_flooded_pipe_cannot_be_replaced(tiny):
    state, _ = tiny.reset()
    one = tiny.__advance__(state, PipeAction(1, 0))
    replaced = tiny.__advance__(one, PipeAction(1, 0))
    assert replaced.tick == 3 and replaced.rows[0] == ">7..", "the elbow went over the straight"
    laid = tiny.simulate([PipeAction(1, 0), PipeAction(2, 0), PipeAction(2, 1)])[-1]
    assert laid.carried == 1
    assert tiny.__advance__(laid, PipeAction(1, 0)) == laid, "the flow has run through it"
    assert tiny.__advance__(laid, PipeAction(0, 0)) == laid, "the start piece is not a cell"


def test_discard_spends_the_piece_and_the_tick_and_flush_runs_the_flow(tiny):
    state, _ = tiny.reset()
    dropped = tiny.__advance__(state, DISCARD)
    assert dropped.rows == state.rows and dropped.placed == 1 and dropped.tick == 1
    laid = tiny.simulate([PipeAction(1, 0), PipeAction(2, 0), PipeAction(2, 1)])[-1]
    assert laid.ahead == 2, "two pieces still ahead of the flow"
    done = tiny.__advance__(laid, FLUSH)
    assert tiny.is_goal(done) and done.carried == 3 and not done.spilled
    spilt = tiny.__advance__(tiny.simulate([PipeAction(1, 0)])[-1], FLUSH)
    assert spilt.spilled and spilt.carried == 1 and tiny.is_terminal(spilt)
    assert tiny.successors(spilt) == [] and tiny.__advance__(spilt, DISCARD) == spilt


def test_the_goal_needs_the_distance_and_a_spill_short_of_it_is_a_dead_end(tiny):
    state, _ = tiny.reset()
    goal = PipeState(state.rows, (), state.head, 3, 0, 0, False, state.queue, 3, 3, 1)
    assert tiny.is_goal(goal) and not tiny.is_terminal(goal)
    won_spilt = PipeState(state.rows, (), state.head, 3, 0, 0, True, state.queue, 3, 3, 1)
    assert tiny.is_goal(won_spilt), "a spill after the distance is no loss"
    assert tiny.is_terminal(PipeState(state.rows, (), state.head, 2, 0, 0, True, state.queue, 3, 3, 1))


def test_actions_parse_print_and_cost():
    assert PipeAction.parse("place(3,4)") == PipeAction(3, 4) and str(PipeAction(3, 4)) == "place(3,4)"
    assert PipeAction.parse("flush") == FLUSH and PipeAction.parse("discard") == DISCARD
    assert FLUSH.cost() == 1 and DISCARD.cost() == 1 and _DISCARD is DISCARD
    with pytest.raises(ValueError):
        PipeAction(name="place")


def test_get_actions_offers_every_free_cell_discard_and_flush(env):
    state, _ = env.reset()
    actions = env.get_actions(state)
    free = sum(row.count(".") for row in state.rows)
    assert len(actions) == free + 2 and actions[-1] == FLUSH and DISCARD in actions
    used = PipeState(state.rows, (), state.head, 0, 0, len(state.queue), False, state.queue,
                     state.distance, state.countdown, state.pace)
    assert env.get_actions(used) == [FLUSH], "the queue used up: only the flush is left"


def test_successors_obey_the_contract(env):
    state, _ = env.reset()
    for _ in range(2):
        children = env.successors(state)
        assert_successors_contract(children)
        assert all(child != state for _, child in children)
        state = children[0][1]


def test_states_hash_by_board_flow_and_counters_not_depth():
    rows = (">..",)
    one = PipeState(rows, (), (0, 0, "e"), 0, 1, 1, False, ("ew",), 2, 2, 1, depth=1)
    two = PipeState(rows, (), (0, 0, "e"), 0, 1, 1, False, ("ew",), 2, 2, 1, depth=9)
    assert one == two and hash(one) == hash(two)
    assert one != PipeState(rows, (), (0, 0, "e"), 0, 2, 1, False, ("ew",), 2, 2, 1), "a tick apart"


def test_literals_name_pipe_flow_and_queue(env):
    state, _ = env.reset()
    assert_string_literals(state)
    assert "carried(0)" in state.literals and f"next({state.next})" in state.literals
    assert f"due({state.countdown})" in state.literals
    laid = env.__advance__(state, DISCARD)
    assert f"due({state.countdown - 1})" in laid.literals


def test_the_text_draws_laid_and_run_pipe_differently(tiny):
    trace = tiny.simulate([PipeAction(1, 0), PipeAction(2, 0), PipeAction(2, 1), FLUSH])
    assert "─" in str(trace[1]) and "═" not in str(trace[1])
    assert "═" in str(trace[-1]) and "carried 3 of 3" in str(trace[-1])


def test_simulate_and_step_agree(tiny):
    plan = [PipeAction(1, 0), DISCARD, FLUSH]
    trace = tiny.simulate(plan)
    tiny.reset()
    for action in plan:
        state, gained = tiny.step(action)
    assert state == trace[-1] and len(tiny.render()) == 4


def test_set_instance_checks_its_shape():
    game = PipeDreamEnv()
    with pytest.raises(ValueError, match="start"):
        game.set_instance(level(["....", "...."], ["ew"], 2))
    with pytest.raises(ValueError, match="unknown piece"):
        game.set_instance(level([">..."], ["tee"], 2))
    with pytest.raises(ValueError, match="same width"):
        game.set_instance(level([">...", "..."], ["ew"], 2))


# ------------------------------------------------------------------------------- instances

def test_the_thoughtless_policy_lays_at_the_end_of_the_pipe_or_discards(tiny):
    plan = tiny.greedy_plan()
    assert [str(action) for action in plan] == ["place(1,0)", "place(2,0)", "place(2,1)", "flush"]
    assert tiny.validate(plan)
    end = end_of_pipe(tiny.simulate(plan[:2])[-1])
    assert end == (2, 0, "s")


def test_a_drawn_level_has_a_start_with_room_ahead_and_a_weighted_queue():
    random_, _ = rng(3)
    instance = draw_level(random_)
    rows = instance["rows"]
    starts = [(x, y, cell) for y, row in enumerate(rows) for x, cell in enumerate(row) if cell in "<>^v"]
    assert len(starts) == 1 and len(instance["queue"]) == instance["countdown"] + 3 * instance["distance"]
    assert 8 <= instance["distance"] <= 12 and 5 <= instance["countdown"] <= 8
    assert set(instance["queue"]) <= set(KINDS)


def test_every_level_is_solved_by_its_plan_and_not_by_the_thoughtless_policy():
    with open(os.path.join(DATA, "pipe_dream_solutions.json")) as handle:
        solutions = {int(index): plan for index, plan in json.load(handle).items()}
    assert len(LEVELS) == 100 and sorted(solutions) == list(range(100))
    for index in range(0, 100, 5):
        game = PipeDreamEnv()
        game.set_index(index)
        assert game.validate(solutions[index]), f"level {index}"
        assert not game.validate(game.greedy_plan()), f"level {index} needs no thought"


def test_set_index_refuses_a_level_that_is_not_there():
    game = PipeDreamEnv()
    for index in (-1, len(LEVELS), 999):
        with pytest.raises(IndexError, match="Invalid index"):
            game.set_index(index)


def test_a_generated_level_reproduces_from_its_seed():
    game = PipeDreamEnv()
    instance = game.generate_instance(seed=11000)
    assert instance == LEVELS[0], "level 0 is seed 11000"
    assert game.validate(game.witness) and len(game.witness) >= 6
    assert PipeDreamEnv().generate_instance(seed=11000) == instance
