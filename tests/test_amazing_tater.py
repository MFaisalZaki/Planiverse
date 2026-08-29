"""Tests for the pure-Python Amazing Tater environment.

Four parts. The first pins the *rules* down on hand-made rooms, because a rule set only ever
exercised through the shipped levels is a rule set nobody can argue with, and four of these
rules exist in the form they do only because the cartridge disagreed with a simpler guess.
The second checks the 105 rooms themselves. The third checks the environment contract. The
fourth replays stored solutions, and, when a ROM is around, re-dumps the cartridge to check
the rooms have not drifted and replays a solution on the cartridge itself.
"""
import pytest

from planiverse.environments.gameboy_py.amazing_tater import (
    ACTIONS, ARM_GLYPHS, BLOCK_GLYPHS, LEVELS, LEVEL_COUNT, LEVEL_SETS, SETTLED_GLYPHS,
    SWITCH, AmazingTaterAction, AmazingTaterGame, AmazingTaterState, Level, advance, board,
    friendly, group_blocks, initial_state, label_for, parse_level, solve,
)

from conftest import (
    amazing_tater_rom_path, assert_string_literals, assert_successors_contract,
)

needs_rom = pytest.mark.skipif(
    amazing_tater_rom_path() is None,
    reason='set PLANIVERSE_AMAZING_TATER_ROM to an "Amazing Tater (U).gb" ROM',
)


def play(rows, *actions):
    """Run a hand-made room through `actions` and give back `(level, state)`.

    An action the game refuses leaves the position alone, exactly as it does in play, so a
    test that expects a refusal asserts the board is unchanged rather than that something
    raised.
    """
    level = Level(0, rows)
    state = initial_state(level)
    for action in actions:
        following = advance(level, state, action)
        if following is not None:
            state = following
    return level, state


def shown(rows, *actions):
    """The board after `actions`, in the friendly alphabet, for readable assertions."""
    level, state = play(rows, *actions)
    return friendly(board(level, state))


def refused(rows, *actions):
    """Did the position stay exactly where it started? The friendly view of both sides,
    because `rows` is written in the exact alphabet and `shown` prints the readable one."""
    return shown(rows, *actions) == friendly(tuple(row.rstrip() for row in rows))


#: A room's worth of glyph reminders, so the hand-made boards below stay readable.
#: `a` is a block square joined to nothing; `b`/`e` are the halves of a horizontal pair,
#: `c`/`i` of a vertical one; `d g` over `j m` is a 2x2.
LONE, LEFT_HALF, RIGHT_HALF = BLOCK_GLYPHS[0], BLOCK_GLYPHS[1], BLOCK_GLYPHS[4]
TOP_HALF, BOTTOM_HALF = BLOCK_GLYPHS[2], BLOCK_GLYPHS[8]


# ------------------------------------------------------------------------ walking around

def test_a_step_onto_floor_moves_the_tater():
    assert shown(("#####",
                  "#E1.#",
                  "#####"), "right") == ("#####",
                                         "#E.1#",
                                         "#####")


def test_a_step_into_a_wall_is_refused():
    rows = ("###",
            "#1#",
            "#E#",
            "###")
    assert refused(rows, "left")
    assert refused(rows, "right")
    assert refused(rows, "up")


def test_a_step_into_a_pit_is_refused():
    rows = ("#####",
            "#E1O#",
            "#####")
    assert refused(rows, "right")


def test_reaching_the_flag_takes_the_tater_off_the_board():
    level, state = play(("#####",
                         "#E1.#",
                         "#####"), "left")
    assert state.taters == ()
    assert state.home == frozenset({0})
    assert state.solved


# -------------------------------------------------------------------------------- blocks

def test_a_block_with_floor_behind_it_is_pushed():
    assert shown(("######",
                  "#E1a.#",
                  "######"), "right") == ("######",
                                          "#E.1$#",
                                          "######")


def test_a_block_against_a_wall_does_not_move():
    rows = ("#####",
            "#E1a#",
            "#####")
    assert refused(rows, "right")


def test_a_whole_shape_moves_as_one_piece():
    assert shown(("#######",
                  "#..dg.#",
                  "#E1jm.#",
                  "#######"), "right") == ("#######",
                                           "#...$$#",
                                           "#E.1$$#",
                                           "#######")


def test_two_blocks_flush_together_are_two_blocks():
    """The link masks say these two squares are separate blocks, so only one of them moves.

    Grouping block squares by what touches what would weld them into a vertical pair and
    refuse the push. The cartridge moves the lower one, in room `A-04` among others, which is
    what this encoding is for.
    """
    level, _ = play(("#####",
                     "#.a.#",
                     "#1a.#",
                     "#E..#",
                     "#####"))
    assert len(level.start_blocks) == 2
    assert shown(("#####",
                  "#.a.#",
                  "#1a.#",
                  "#E..#",
                  "#####"), "right") == ("#####",
                                         "#.$.#",
                                         "#.1$#",
                                         "#E..#",
                                         "#####")


def test_a_block_shoved_fully_onto_pits_dissolves_and_fills_them():
    level, state = play(("######",
                         "#E1aO#",
                         "######"), "right")
    assert state.blocks == frozenset()
    assert state.filled == frozenset({(1, 4)})
    assert friendly(board(level, state)) == ("######",
                                             "#E.1.#",
                                             "######")


def test_a_block_partly_over_pits_is_still_a_block():
    assert shown(("#######",
                  "#E1beO#",
                  "#######"), "right") == ("#######",
                                           "#E.1$&#",
                                           "#######")


def test_a_settled_square_cannot_be_shoved():
    """Rule 4, and the cartridge's, not a guess: room `A-36` refuses exactly this push.

    `b` and `G` are the two halves of a horizontal pair: `G` is the right half, already
    settled into a pit. The first push is aimed at the half on floor and moves the block; the
    second is aimed at the half that is now in the pit, and does not.
    """
    rows = ("#######",
            "#.....#",
            "#E1bG.#",
            "#######")
    after = shown(rows, "right")
    assert after == ("#######",
                     "#.....#",
                     "#E.1&$#",
                     "#######")
    assert shown(rows, "right", "right") == after


def test_the_rest_of_a_settled_block_can_still_be_shoved():
    """The same rule from the other side: aim the push at a square that is on floor.

    `c` over `K` is a vertical pair whose lower half has settled into a pit. Shoving the
    upper half moves both, and the pit it leaves behind is still a pit.
    """
    rows = ("#####",
            "#.1.#",
            "#.c.#",
            "#EK.#",
            "#...#",
            "#####")
    level, state = play(rows)
    assert len(level.start_blocks) == 1
    assert shown(rows, "down") == ("#####",
                                   "#...#",
                                   "#.1.#",
                                   "#E&.#",
                                   "#.$.#",
                                   "#####")


# ---------------------------------------------------------------------------- turnstiles

def test_a_turnstile_turns_the_way_it_is_pushed():
    """An arm pointing up, shoved right: the whole turnstile turns a quarter clockwise."""
    assert shown(("#####",
                  "#1^.#",
                  "#.@.#",
                  "#E..#",
                  "#####"), "right") == ("#####",
                                         "#.1.#",
                                         "#.o+#",
                                         "#E..#",
                                         "#####")


def test_a_turnstile_pushed_along_its_own_axis_does_not_turn():
    rows = ("#####",
            "#...#",
            "#1<@#",
            "#E..#",
            "#####")
    assert refused(rows, "right")


def test_a_turnstile_with_nowhere_to_swing_does_not_turn():
    rows = ("####",
            "#1^#",
            "#.@#",
            "#E.#",
            "####")
    assert refused(rows, "right")


def test_a_turnstile_blocked_on_the_swept_diagonal_does_not_turn():
    """Rule 6's other half: the corner an arm passes through has to be clear as well.

    Everything the arm *lands* on here is free; only the diagonal between where it starts and
    where it stops is occupied. The cartridge refuses this, in room `A-01`.
    """
    rows = ("######",
            "#1^a.#",
            "#.@..#",
            "#E...#",
            "######")
    assert refused(rows, "right")


def test_the_pusher_steps_into_the_square_it_vacated():
    """No arm swings in behind, so the pusher simply walks into the gap."""
    _, state = play(("#####",
                     "#1^.#",
                     "#.@.#",
                     "#E..#",
                     "#####"), "right")
    assert dict(state.taters)[0] == (1, 2)


def test_the_pusher_is_carried_round_when_shut_into_a_compartment():
    """Rule 7. Two arms make a compartment, and the turnstile takes the pusher with it."""
    _, state = play(("#####",
                     "#1^.#",
                     "#<@.#",
                     "#E..#",
                     "#####"), "right")
    assert dict(state.taters)[0] == (1, 3)      # a quarter-turn round the pivot, not a step


def test_an_arm_hanging_over_a_pit_cannot_be_pushed():
    """Rule 6's last clause, from room `C-43`."""
    rows = ("#####",
            "#1U.#",
            "#.@.#",
            "#E..#",
            "#####")
    assert refused(rows, "right")


def test_a_pivot_is_solid():
    rows = ("#####",
            "#.^.#",
            "#1@.#",
            "#E..#",
            "#####")
    assert refused(rows, "right")


# ------------------------------------------------------------------------- several taters

def test_switch_hands_the_controls_to_the_next_tater():
    rows = ("#####",
            "#1.2#",
            "#.E.#",
            "#####")
    _, state = play(rows, SWITCH)
    assert state.active == 1
    _, state = play(rows, SWITCH, "left")
    assert dict(state.taters)[1] == (1, 2)      # the second one moved, not the first


def test_switch_is_not_offered_with_one_tater():
    game = AmazingTaterGame()
    game.fix_index(0)
    state, _ = game.reset()
    assert len(state.taters) == 1
    assert SWITCH not in [action.name for action, _ in game.successors(state)]


def test_the_room_is_solved_only_when_every_tater_is_home():
    rows = ("#####",
            "#1E2#",
            "#####")
    _, state = play(rows, "right")
    assert not state.solved and state.home == frozenset({0})
    _, state = play(rows, "right", "left")
    assert state.solved and state.home == frozenset({0, 1})


def test_the_controls_pass_on_when_a_tater_goes_home():
    rows = ("#####",
            "#1E2#",
            "#####")
    _, state = play(rows, "right")
    assert state.active == 1


# ------------------------------------------------------------------------------ the rooms

def test_every_room_parses_and_has_a_tater_and_a_flag():
    for index in range(LEVEL_COUNT):
        level = Level(index, LEVELS[index])
        assert level.exit is not None, level.label
        assert level.start_taters, level.label


def test_the_sets_add_up():
    assert LEVEL_COUNT == sum(size for _letter, _mode, size in LEVEL_SETS) == 105
    assert label_for(0) == "A-01"
    assert label_for(40) == "A-41"
    assert label_for(41) == "C-01"
    assert label_for(104) == "C-64"


def test_every_room_round_trips_through_its_own_renderer():
    """Parsing a room and printing the position back gives the text it was parsed from.

    This is the cheap check that the alphabet is a bijection: block link masks, arm
    directions and turnstile shapes all have to survive the trip.
    """
    for index in range(LEVEL_COUNT):
        level = Level(index, LEVELS[index])
        printed = board(level, initial_state(level))
        assert printed == tuple(row.rstrip() for row in LEVELS[index]), label_for(index)


def test_every_arm_belongs_to_a_pivot_and_every_pivot_has_its_arms():
    for index in range(LEVEL_COUNT):
        level = Level(index, LEVELS[index])
        for pivot, mask in level.start_turnstiles:
            assert 1 <= mask <= 15, label_for(index)
            assert pivot not in level.walls


def test_no_block_square_claims_a_neighbour_that_is_not_one():
    """Every square's link mask agrees with its neighbours'. True on the cartridge for all
    201 rooms, and the reason blocks can be recovered from the glyphs at all."""
    for index in range(LEVEL_COUNT):
        level = Level(index, LEVELS[index])
        squares = {cell for block in level.start_blocks for cell in block}
        for block in level.start_blocks:
            for row, col in block:
                for offset in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                    neighbour = (row + offset[0], col + offset[1])
                    if neighbour in block:
                        assert neighbour in squares


def test_a_room_is_the_size_the_cartridge_says_it_is():
    """Every stored room is its width and height plus a one-cell border, which is what
    `LoadLevel` writes to `$C2BD` and `$C2BE`."""
    game = AmazingTaterGame()
    for index in (0, 3, 40, 41, 104):
        game.fix_index(index)
        _, info = game.reset()
        height, width = Level(index, LEVELS[index]).shape
        assert info["size"] == (width - 2, height - 2)


# ------------------------------------------------------------------------ the environment

@pytest.fixture
def game():
    game = AmazingTaterGame()
    game.fix_index(0)
    game.reset()
    return game


def test_reset_reports_the_room(game):
    state, info = game.reset()
    assert info["level"] == "A-01"
    assert info["taters"] == 1
    assert info["turnstiles"] == 3
    assert not game.is_goal(state)


def test_state_contract(game):
    state, _ = game.reset()
    assert_string_literals(state)
    assert_successors_contract(game.successors(state))


def test_successors_never_include_a_move_that_changes_nothing(game):
    state, _ = game.reset()
    for _action, successor in game.successors(state):
        assert successor != state


def test_a_position_reached_two_ways_compares_equal(game):
    state, _ = game.reset()
    there = advance(game.level, state, "left")
    back = advance(game.level, there, "right")
    assert back == state and hash(back) == hash(state)


def test_fix_index_rejects_a_room_that_is_not_there(game):
    with pytest.raises(IndexError):
        game.fix_index(LEVEL_COUNT)


def test_an_unknown_action_is_rejected():
    with pytest.raises(ValueError):
        AmazingTaterAction("jump")


def test_switch_is_free_and_a_step_is_not():
    assert AmazingTaterAction(SWITCH).cost() == 0
    assert all(AmazingTaterAction(name).cost() == 1 for name in ACTIONS if name != SWITCH)


def test_step_reports_how_many_taters_got_home():
    game = AmazingTaterGame()
    game.fix_index(0)
    game.reset()
    _, gained = game.step(AmazingTaterAction("left"))
    assert gained == 0


# --------------------------------------------------------------------------- solutions
# Shortest plans, found by `solve` and replayed here. They are also replayed on the
# cartridge below, when a ROM is available, which is what makes them evidence about the game
# rather than about this module.

SOLUTIONS = {
    0: 38, 1: 22, 2: 31, 3: 30, 4: 47, 5: 41, 6: 54, 7: 47, 8: 46, 9: 64,
    10: 96, 11: 68, 12: 53, 13: 41, 17: 38, 30: 77,
}


@pytest.mark.parametrize("index", sorted(SOLUTIONS))
def test_the_stored_solutions_solve_their_rooms(index):
    plan = solve(index)
    assert plan is not None, label_for(index)
    assert len(plan) == SOLUTIONS[index], label_for(index)
    game = AmazingTaterGame()
    game.fix_index(index)
    game.reset()
    assert game.validate(plan)


def test_a_plan_that_stops_short_does_not_validate():
    game = AmazingTaterGame()
    game.fix_index(0)
    game.reset()
    assert not game.validate(solve(0)[:-1])


def test_simulate_returns_one_state_more_than_the_plan():
    game = AmazingTaterGame()
    game.fix_index(1)
    game.reset()
    plan = solve(1)
    trace = game.simulate(plan)
    assert len(trace) == len(plan) + 1
    assert game.is_goal(trace[-1])


# ------------------------------------------------------------------------ against the ROM
# The rooms were dumped off a cartridge, so re-dumping it is the one thing that can catch
# them drifting. Skipped without a ROM.

@needs_rom
def test_the_rooms_still_match_the_cartridge():
    from planiverse.environments.gameboy.amazing_tater_gb import AmazingTaterGBEnv

    env = AmazingTaterGBEnv(amazing_tater_rom_path())
    try:
        for index in (0, 3, 13, 40, 41, 104):
            assert env.__dump__(index) == tuple(row.rstrip() for row in LEVELS[index]), \
                label_for(index)
    finally:
        env.close()


@needs_rom
def test_a_solution_found_here_also_solves_the_cartridge():
    from planiverse.environments.gameboy.amazing_tater_gb import AmazingTaterGBEnv

    plan = solve(1)
    env = AmazingTaterGBEnv(amazing_tater_rom_path(), calibrate=False)
    try:
        env.fix_index(1)
        state, _ = env.reset()
        for name in plan:
            state = env.__advance__(state, f"{name},5")
        assert env.is_goal(state)
    finally:
        env.close()
