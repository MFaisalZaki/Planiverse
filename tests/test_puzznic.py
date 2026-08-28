"""Tests for the Puzznic environment."""
import pytest

from planiverse.environments.gameboy_py.puzznic import (
    Box, Cursor, EmptySpace, Level, PuzznicGame, PuzznicState, Wall,
)

from conftest import assert_state_contract, assert_successors_contract

LEVEL_0 = """######
#12c #
###  #
#    #
#2  1#
##21##
######"""


def build(levelstr):
    """A game positioned on a hand-written level."""
    game = PuzznicGame()
    game.level = Level(levelstr)
    game.state, _ = game.level.reset()
    game.state_history = [game.state]
    return game


# --------------------------------------------------------------------------- levels

def test_fifty_levels_available():
    assert len(PuzznicGame().levelsstr) == 50


@pytest.mark.parametrize("index", range(50))
def test_every_level_parses_and_resets(index):
    env = PuzznicGame()
    env.fix_index(index)
    state, info = env.reset()
    assert_state_contract(state)
    # A fresh level is neither won nor lost.
    assert not env.is_goal(state)
    assert not env.is_terminal(state)


@pytest.mark.parametrize("index,expected", [
    # Read out of `Puzznic (J)` at `$DF00` after booting each round, via `PuzznicGBEnv`:
    # the two cells that are wall on the cartridge, and the four blocks whose transcribed
    # types were transposed with their neighbour's.
    (23, {"walls": [(1, 6), (2, 6)]}),
    (34, {"types": {(5, 2): "1", (5, 3): "8", (10, 1): "8", (10, 2): "7"}}),
])
def test_the_levels_match_the_cartridge(index, expected):
    """The 50 Python levels are transcriptions of the cartridge's first 50 rounds, so a
    cell that disagrees with the ROM is a transcription slip, not a design choice."""
    env = PuzznicGame()
    env.fix_index(index)
    state, _ = env.reset()
    for pos in expected.get("walls", []):
        assert isinstance(state.grid[pos[0]][pos[1]], Wall), \
            f"{pos} is a wall on the cartridge"
    for pos, letter in expected.get("types", {}).items():
        cell = state.grid[pos[0]][pos[1]]
        assert isinstance(cell, Box) and cell.letter == letter, \
            f"{pos} is a type-{letter} block on the cartridge"


def test_level_parsing_maps_the_alphabet():
    state, _ = Level("####\n#1c#\n#  #\n####").reset()
    assert isinstance(state.grid[0][0], Wall)
    assert isinstance(state.grid[1][1], Box) and state.grid[1][1].letter == "1"
    # The cursor cell itself is empty space; the cursor is tracked separately.
    assert isinstance(state.grid[1][2], EmptySpace)
    assert state.cursor.pos == (1, 2)


def test_level_without_cursor_is_rejected():
    with pytest.raises(ValueError, match="Cursor not found"):
        Level("####\n#1 #\n####")


def test_reset_is_repeatable():
    env = PuzznicGame()
    env.fix_index(3)
    first, _ = env.reset()
    second, _ = env.reset()
    assert first == second
    assert first.literals == second.literals


# --------------------------------------------------------------------------- interface

def test_initial_state_renders_as_the_level(puzznic_env):
    state, _ = puzznic_env.reset()
    assert str(state) == LEVEL_0


def test_get_actions(puzznic_env):
    puzznic_env.reset()
    assert puzznic_env.get_actions() == ["left", "right", "up", "down", "left-hold", "right-hold"]


def test_successors_contract(puzznic_env):
    state, _ = puzznic_env.reset()
    assert_successors_contract(puzznic_env.successors(state))


def test_successors_exclude_self_loops(puzznic_env):
    """Moving into a wall changes nothing, so it must not be offered as an action."""
    state, _ = puzznic_env.reset()
    successors = puzznic_env.successors(state)
    assert all(next_state != state for _, next_state in successors)
    # The cursor starts at (1,3) with a wall above, so 'up' is a no-op and 'left' is not.
    offered = [action for action, _ in successors]
    assert "left" in offered


def test_step_advances_and_records_history(puzznic_env):
    puzznic_env.reset()
    state, score = puzznic_env.step("left")
    assert state.cursor.pos == (1, 2)
    assert isinstance(score, list)
    assert len(puzznic_env.state_history) > 1


def test_step_rejects_invalid_action(puzznic_env):
    puzznic_env.reset()
    with pytest.raises(AssertionError):
        puzznic_env.step("diagonal")


def test_simulate_returns_state_trace(puzznic_env):
    puzznic_env.reset()
    plan = ["left", "down", "right"]
    trace = puzznic_env.simulate(plan)
    assert len(trace) == len(plan) + 1
    for state in trace:
        assert_state_contract(state)


def test_simulate_is_pure(puzznic_env):
    """simulate replays from the initial state and leaves the env untouched."""
    state, _ = puzznic_env.reset()
    puzznic_env.simulate(["left", "left", "down"])
    assert puzznic_env.state == state


# --------------------------------------------------------------------------- movement

def test_cursor_moves_without_dragging():
    game = build(LEVEL_0)
    after = game._compute_successor_state_(game.state, "left")
    assert after.cursor.pos == (1, 2)
    # The box the cursor lands on has not moved.
    assert isinstance(after.grid[1][2], Box)


def test_hold_drags_a_box_into_empty_space():
    # Two '1' blocks, far apart: the level is playable and the drag forms no match.
    game = build("#######\n#1c  1#\n#######")
    on_box = game._compute_successor_state_(game.state, "left")   # cursor onto the box
    dragged = game._compute_successor_state_(on_box, "right-hold")
    assert dragged.grid[1][2].letter == "1"
    assert isinstance(dragged.grid[1][1], EmptySpace)
    assert dragged.cursor.pos == (1, 2)


def test_hold_on_empty_cell_does_nothing():
    """A hold with no block under the cursor is rejected outright: the cursor stays put."""
    game = build("#######\n#1c  1#\n#######")
    after = game._compute_successor_state_(game.state, "right-hold")
    assert after.cursor.pos == (1, 2)
    # Nothing was dragged: the box is still where it started.
    assert after.grid[1][1].letter == "1"


# --------------------------------------------------------------------------- gravity

def test_gravity_settles_a_stack():
    """A stack must settle fully: a single top-down pass left the upper box floating."""
    game = build("#####\n#1  #\n#1  #\n#   #\n#  c#\n#####")
    settled = game._apply_gravity_(game.state)
    column = [settled.grid[row][1] for row in range(1, 5)]
    assert [isinstance(cell, Box) for cell in column] == [False, False, True, True]


def test_gravity_leaves_supported_boxes_alone():
    game = build("#####\n#1 c#\n#####\n#####\n#####")
    settled = game._apply_gravity_(game.state)
    assert settled.grid[1][1].letter == "1"


# --------------------------------------------------------------------------- matching

def test_adjacent_same_letters_clear():
    game = build("######\n#11 c#\n######\n######\n######")
    matched, removed = game._check_and_remove_matches_(game.state)
    assert len(removed) == 2
    assert all(isinstance(cell, EmptySpace) for cell in (matched.grid[1][1], matched.grid[1][2]))


def test_different_letters_do_not_clear():
    game = build("######\n#12 c#\n######\n######\n######")
    _, removed = game._check_and_remove_matches_(game.state)
    assert removed == set()


def test_match_cascades_after_gravity():
    """Clearing lets boxes fall, which can form a new match, which clears in the same step."""
    game = build("#####\n#1  #\n#1  #\n#   #\n#  c#\n#####")
    after = game._compute_successor_state_(game.state, "left")
    # Both boxes fell together, matched, and cleared -- the level is now won.
    assert after.is_goal()
    assert len(after.cleared_boxes) == 2


# --------------------------------------------------------------------------- scoring

def test_moving_a_box_scores_nothing():
    """A box that merely falls must not be scored as cleared.

    Scores used to be diffed by (letter, position), so a box that moved counted as
    removed twice and awarded 20 points for clearing nothing.
    """
    game = build(LEVEL_0)
    on_box = game._compute_successor_state_(game.state, "left")
    dragged = game._compute_successor_state_(on_box, "right-hold")
    assert dragged.cleared_boxes == []
    assert sum(dragged.score) == 0


def test_clearing_two_blocks_scores_ten_each():
    game = build("######\n#11 c#\n######\n######\n######")
    assert game._compute_score_({Box("1", (1, 1)), Box("1", (1, 2))}) == [20]


def test_multiple_letters_apply_the_cascade_multiplier():
    removed = {Box("1", (1, 1)), Box("1", (1, 2)), Box("2", (2, 1)), Box("2", (2, 2))}
    # 4 blocks * 10 = 40, two distinct letters -> 40 * 2 * 1.5
    assert game_score(removed) == [120.0]


def game_score(removed):
    return PuzznicGame()._compute_score_(removed)


def test_more_than_two_of_a_letter_adds_a_bonus():
    removed = {Box("1", (1, c)) for c in range(1, 4)}       # three '1' blocks
    # 3 * 10 = 30, single letter so no multiplier, +50 bonus for the third block
    assert game_score(removed) == [80]


def test_score_records_one_entry_per_clear(puzznic_env):
    """score entries are appended per clear, not per action taken."""
    puzznic_env.reset()
    trace = puzznic_env.simulate(["left", "down", "right"])
    assert trace[-1].score == []          # none of these actions cleared anything

    game = build("#####\n#1  #\n#1  #\n#   #\n#  c#\n#####")
    cleared = game._compute_successor_state_(game.state, "left")
    assert cleared.score == [20]          # one clear of two blocks


# --------------------------------------------------------------------------- goal / terminal

def test_goal_when_no_boxes_remain():
    game = build("#####\n#  c#\n#   #\n#####\n#####")
    assert game.is_goal(game.state)
    assert game.validate([])


def test_terminal_when_a_letter_is_unmatchable():
    """One lone '1' can never be matched, so the level is lost."""
    game = build("#####\n#1 c#\n#2 2#\n#####\n#####")
    assert game.is_terminal(game.state)
    assert not game.is_goal(game.state)


def test_goal_and_terminal_states_are_absorbing():
    game = build("#####\n#  c#\n#   #\n#####\n#####")
    assert game.is_goal(game.state)
    after = game._compute_successor_state_(game.state, "left")
    assert after == game.state


def test_validate_rejects_a_plan_that_does_not_win(puzznic_env):
    puzznic_env.reset()
    assert not puzznic_env.validate(["left", "right"])


# --------------------------------------------------------------------------- literals

def test_literals_describe_cursor_and_boxes(puzznic_env):
    state, _ = puzznic_env.reset()
    assert "at(cursor, 1, 3)" in state.literals
    assert "at(box-1, 1, 1)" in state.literals
    assert "at(box-2, 1, 2)" in state.literals


def test_literals_record_cleared_boxes_and_goal():
    game = build("#####\n#1  #\n#1  #\n#   #\n#  c#\n#####")
    after = game._compute_successor_state_(game.state, "left")
    assert "goal-reached" in after.literals
    assert "all-boxes-matched(box-1)" in after.literals
    assert any(lit.startswith("cleared(box-1") for lit in after.literals)
    assert any(lit.startswith("score(") for lit in after.literals)


def test_state_equality_ignores_score_and_history():
    grid = Level("#####\n#1 c#\n#####\n#####\n#####").grid
    cursor = Cursor((1, 3))
    assert PuzznicState(grid, cursor, [10]) == PuzznicState(grid, cursor, [999])
