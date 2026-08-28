"""Tests for the pure-Python Boxxle II environment.

Three parts. The first pins the *rules* down on hand-made boards, because a rule set only ever
exercised through the shipped levels is a rule set nobody can argue with. The second checks the
environment contract and the levels themselves. The third re-decodes the cartridge and compares
it against the levels stored here, and is skipped when no ROM is around — the levels were
generated from a ROM, so that comparison is the one thing that can catch them drifting.
"""
import pytest

from planiverse.environments.gameboy_py.boxxle2 import (
    LEVELS, LEVELS_PER_STAGE, Boxxle2Action, Boxxle2Game, Boxxle2State, Level,
    dead_squares, parse_level, push, reachable, render,
)

from conftest import assert_string_literals, assert_successors_contract, boxxle2_rom_path


def board(*rows):
    """A hand-made level, as the tuple `parse_level` returns."""
    return parse_level("\n".join(rows))


@pytest.fixture
def env():
    game = Boxxle2Game()
    game.fix_index(0)
    game.reset()
    return game


# ------------------------------------------------------------------------------ the rules

def test_a_step_onto_floor_moves_the_keeper():
    walls, _, boxes, player, _ = board("#####",
                                       "# @ #",
                                       "#####")
    assert push(walls, boxes, player, "right") == ((1, 3), boxes)


def test_a_step_into_a_wall_is_refused():
    walls, _, boxes, player, _ = board("###",
                                       "#@#",
                                       "###")
    assert all(push(walls, boxes, player, name) is None
               for name in ("left", "right", "up", "down"))


def test_a_box_with_floor_behind_it_is_pushed():
    walls, _, boxes, player, _ = board("#####",
                                       "#@$ #",
                                       "#####")
    keeper, moved = push(walls, boxes, player, "right")
    assert keeper == (1, 2)
    assert moved == frozenset({(1, 3)})


def test_a_box_against_a_wall_does_not_move():
    walls, _, boxes, player, _ = board("####",
                                       "#@$#",
                                       "####")
    assert push(walls, boxes, player, "right") is None


def test_a_chain_of_two_boxes_does_not_move():
    """The rule that makes it Sokoban rather than a shunting yard."""
    walls, _, boxes, player, _ = board("######",
                                       "#@$$ #",
                                       "######")
    assert push(walls, boxes, player, "right") is None


def test_a_box_can_be_pushed_onto_and_off_a_goal():
    walls, goals, boxes, player, _ = board("######",
                                           "#@$o #",
                                           "######")
    _, on_goal = push(walls, boxes, player, "right")
    assert on_goal == goals
    keeper, off_goal = push(walls, on_goal, (1, 2), "right")
    assert off_goal == frozenset({(1, 4)})


def test_the_keeper_cannot_pull():
    """There is no action for it, which is the whole reason a wrong push is permanent."""
    assert set(Boxxle2Game().get_actions()) == {Boxxle2Action(name)
                                                for name in ("left", "up", "down", "right")}


def test_parse_level_reads_every_glyph():
    walls, goals, boxes, player, shape = board("#####",
                                               "#@$o#",
                                               "#+*.#".replace(".", " "),
                                               "#####")
    assert shape == (4, 5)
    assert (1, 1) not in walls and (0, 0) in walls
    assert boxes == frozenset({(1, 2), (2, 2)})
    assert goals == frozenset({(1, 3), (2, 1), (2, 2)})
    assert player == (2, 1)                       # `+` is the keeper standing on a goal


def test_render_round_trips_a_level():
    text = "#####\n#@$o#\n#####"
    walls, goals, boxes, player, shape = parse_level(text)
    assert render(walls, goals, boxes, player, shape) == text


def test_reachable_stops_at_boxes_and_walls():
    walls, _, boxes, player, _ = board("#####",
                                       "#@$ #",
                                       "#####")
    assert reachable(walls, boxes, player) == {(1, 1)}


# ----------------------------------------------------------------------------- dead ends

def test_a_corner_that_is_not_a_goal_is_dead():
    walls, goals, _, _, shape = board("####",
                                      "#@ #",
                                      "#  #",
                                      "####")
    assert dead_squares(walls, goals, shape) == frozenset({(1, 1), (1, 2), (2, 1), (2, 2)})


def test_a_corner_that_is_a_goal_is_not_dead():
    walls, goals, _, _, shape = board("####",
                                      "#o@#",
                                      "#  #",
                                      "####")
    assert (1, 1) not in dead_squares(walls, goals, shape)


def test_a_box_shoved_into_a_corner_makes_the_level_terminal():
    game = Boxxle2Game()
    game.level = Level(0, "######\n#  o##\n# @$ #\n#    #\n######")
    state = Boxxle2State(game.level, game.level.start_boxes, game.level.start_player)
    assert not game.is_terminal(state)
    doomed = game.__advance__(state, "right")     # into the corner under the wall at (1, 4)
    assert doomed.stuck() and game.is_terminal(doomed)
    assert "terminal-state" in doomed.literals


def test_a_terminal_state_expands_to_nothing():
    game = Boxxle2Game()
    game.level = Level(0, "#####\n#@$ #\n#  o#\n#####")
    state = Boxxle2State(game.level, game.level.start_boxes, game.level.start_player)
    doomed = game.__advance__(state, "right")
    assert game.is_terminal(doomed)
    assert game.successors(doomed) == []


def test_dead_end_detection_never_condemns_a_solvable_position():
    """Soundness is the property that matters: a wrong dead end prunes a real solution.

    Every stored solution is replayed in `test_solutions.py`; here the weaker but broader
    check is that no *initial* position of any of the 120 levels is called terminal.
    """
    game = Boxxle2Game()
    for index in range(len(LEVELS)):
        game.fix_index(index)
        state, _ = game.reset()
        assert not game.is_terminal(state), f"level {index} is terminal before a move is made"


# --------------------------------------------------------------------------- the contract

def test_reset_reports_the_level(env):
    state, info = env.reset()
    assert info == {"level_index": 0, "level": "1-01", "size": (9, 8), "boxes": 3, "goals": 3}
    assert state.boxes_home == 0 and not state.solved


def test_state_literals_are_strings(env):
    assert_string_literals(env.state)


def test_successors_returns_pairs_and_drops_no_ops(env):
    successors = env.successors(env.state)
    assert_successors_contract(successors)
    # The keeper starts with a wall above it, so `up` cannot appear.
    assert {str(action) for action, _ in successors} == {"left", "down", "right"}
    assert all(successor != env.state for _, successor in successors)


def test_equal_states_are_reached_by_different_routes(env):
    """Depth is not part of identity, or search could never close."""
    there = env.__advance__(env.state, "down")
    back = env.__advance__(there, "up")
    assert back == env.state and hash(back) == hash(env.state)
    assert back.depth == 2


def test_step_reports_boxes_going_home():
    game = Boxxle2Game()
    game.level = Level(0, "#####\n#@$o#\n#####")
    game.state = Boxxle2State(game.level, game.level.start_boxes, game.level.start_player)
    game.state_history = [game.state]
    state, gained = game.step("right")
    assert gained == 1 and state.solved and game.is_goal(state)


def test_a_solved_level_is_absorbing():
    game = Boxxle2Game()
    game.level = Level(0, "######\n#@$o #\n######")
    solved = game.__advance__(
        Boxxle2State(game.level, game.level.start_boxes, game.level.start_player), "right")
    assert game.is_goal(solved)
    assert game.__advance__(solved, "right") is solved
    assert game.successors(solved) == []


def test_fix_index_rejects_a_level_that_does_not_exist():
    game = Boxxle2Game()
    with pytest.raises(IndexError):
        game.fix_index(len(LEVELS))


def test_validate_replays_a_plan():
    game = Boxxle2Game()
    game.fix_index(0)
    game.reset()
    assert not game.validate(["left", "left"])


# -------------------------------------------------------------------------------- levels

def test_there_are_a_hundred_and_twenty_levels():
    assert len(LEVELS) == 120


def test_every_level_has_as_many_goals_as_boxes():
    """The cartridge's own invariant. A level failing it could never be solved."""
    for index, text in enumerate(LEVELS):
        _, goals, boxes, _, _ = parse_level(text)
        assert len(goals) == len(boxes), f"level {index} has {len(boxes)} boxes, {len(goals)} goals"


def test_every_level_has_exactly_one_keeper():
    for index, text in enumerate(LEVELS):
        keepers = sum(text.count(glyph) for glyph in ("@", "+"))
        assert keepers == 1, f"level {index} has {keepers} keepers"


def test_level_labels_follow_the_cartridge():
    game = Boxxle2Game()
    game.fix_index(37)
    assert game.reset()[1]["level"] == "4-08"
    assert LEVELS_PER_STAGE == 10


def test_no_level_starts_solved():
    game = Boxxle2Game()
    for index in range(len(LEVELS)):
        game.fix_index(index)
        state, _ = game.reset()
        assert not state.solved, f"level {index} needs no moves"


# ---------------------------------------------------------- against the cartridge, if here

needs_rom = pytest.mark.skipif(
    boxxle2_rom_path() is None,
    reason='set PLANIVERSE_BOXXLE2_ROM to a "Boxxle II (USA, Europe).gb" ROM')


@needs_rom
def test_the_stored_levels_still_match_the_cartridge():
    """The levels here were decoded from the ROM, so the ROM is what they answer to."""
    pytest.importorskip("pyboy", reason="pyboy is not installed")
    from planiverse.environments.gameboy.boxxle2_gb import read_levels

    decoded = read_levels(boxxle2_rom_path())
    assert len(decoded) == len(LEVELS)
    for index, rows in enumerate(decoded):
        stored = tuple(row.rstrip() for row in LEVELS[index].split("\n"))
        assert tuple(row.rstrip() for row in rows) == stored, \
            f"level {index} no longer matches the cartridge"
