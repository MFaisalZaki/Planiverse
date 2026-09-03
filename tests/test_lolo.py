"""Tests for the pure-Python Adventures of Lolo twin.

Three tiers, in order of what they need.

The first needs nothing: the rules are pure functions of a room and a position, so pushing,
one-way passes, magic shots and Medusa's line of sight are all tested directly.

The second needs a ROM, and is the one that keeps the 163 rooms honest. They were decoded
out of the cartridge rather than transcribed, and this re-decodes and compares, so a room
cannot drift away from the cartridge unnoticed. It also pins the two modules' shared
alphabet against each other: `lolo.py` declares its own copy so that it needs no PyBoy, and
a silent divergence between the copies would make the two environments describe different
games in the same letters.

The third needs a ROM *and* PyBoy, and is the strongest evidence there is that the stated
rules are the cartridge's rules: it replays the twin's own plans on the real hardware.

    PLANIVERSE_LOLO_ROM="/path/to/Adventures of Lolo (U) [S][!].gb" \\
        poetry run pytest tests/test_lolo.py
"""
import pytest

from planiverse.environments.gameboy_py.lolo import (
    DIRECTIONS, EXACT_ROOMS, MEDUSA_SHIELDS, PUSHABLE_ONTO, ROOMS, SHOOT,
    SHOTS_PER_MAGIC_HEART, WALKABLE, LoloAction, LoloGame, Room, blocked_by_medusa, move,
    one_way_allows, parse_room, render, room_label, shoot,
)

from conftest import assert_string_literals, assert_successors_contract, lolo_rom_path

needs_rom = pytest.mark.skipif(
    lolo_rom_path() is None,
    reason='set PLANIVERSE_LOLO_ROM to an "Adventures of Lolo (U) [S][!].gb" ROM',
)


def game(index, magic_shots=0):
    instance = LoloGame(magic_shots=magic_shots)
    instance.set_index(index)
    instance.reset()
    return instance


def board(rows, magic_shots=0):
    """A LoloGame on a hand-written room, for testing one rule at a time.

    Goes through `reset` so that the initial state gets the same treatment a real room's does,
    including the Medusa check, which can kill Lolo before he has pressed anything.
    """
    instance = LoloGame(magic_shots=magic_shots)
    instance._rooms[0] = Room(0, "|".join(rows))
    instance.set_index(0)
    state, _ = instance.reset()
    return instance, state


#: A blank walled room with the door in the top wall. Every room needs exactly one door and
#: exactly one Lolo, so the helpers below start from one that has both.
WALLED = ["####D###",
          "#......#",
          "#......#",
          "#......#",
          "#......#",
          "#......#",
          "#......#",
          "########"]


def room_with(**cells):
    """`WALLED` with cells overwritten: `room_with(**{"4,1": "@"})`."""
    rows = [list(row) for row in WALLED]
    for key, glyph in cells.items():
        row, col = (int(part) for part in key.split("_"))
        rows[row][col] = glyph
    return ["".join(row) for row in rows]


# ------------------------------------------------------------------------ the catalogue

def test_the_cartridges_163_rooms_are_all_here():
    assert len(ROOMS) == 163


def test_every_room_has_exactly_one_lolo_and_one_door():
    """The invariant that told us where the room table ends (see the memory map §2)."""
    for index, text in enumerate(ROOMS):
        assert text.count("@") == 1, f"room {index} has {text.count('@')} Lolos"
        assert text.count("D") == 1, f"room {index} has {text.count('D')} doors"


def test_every_room_is_eight_by_eight():
    for index, text in enumerate(ROOMS):
        rows = text.split("|")
        assert len(rows) == 8, f"room {index} has {len(rows)} rows"
        assert all(len(row) == 8 for row in rows), f"room {index} is not 8 wide"


def test_the_room_labels_follow_the_cartridges_own_grouping():
    """19 tutorial puzzles stored twice, then 5x14, then 10x5, then 5 Pro rooms."""
    assert room_label(0) == "tutorial 1a"
    assert room_label(1) == "tutorial 1b"
    assert room_label(37) == "tutorial 19b"
    assert room_label(38) == "int 1-1"
    assert room_label(107) == "int 5-14"
    assert room_label(108) == "adv 1-1"
    assert room_label(157) == "adv 10-5"
    assert room_label(158) == "pro 1"
    assert room_label(162) == "pro 5"


def test_the_tutorial_stores_each_puzzle_twice_but_not_identically():
    """A demonstration and the room to try: near-identical, never equal.

    If a pair ever came back equal the demonstration would be teaching a different room from
    the one that follows it, which is the failure the (a)/(b) labelling exists to describe.
    """
    for pair in range(19):
        demo, play = ROOMS[2 * pair], ROOMS[2 * pair + 1]
        assert demo != play, f"tutorial {pair + 1}a and {pair + 1}b are the same room"
        differences = sum(1 for left, right in zip(demo, play) if left != right)
        assert differences <= 32, f"tutorial pair {pair + 1} differs in {differences} cells"


def test_exact_rooms_are_the_ones_with_only_the_two_immobile_enemies():
    """26 of them. The rest hold an enemy this module leaves standing still (see the docs)."""
    assert len(EXACT_ROOMS) == 26
    for index in EXACT_ROOMS:
        assert Room(index, ROOMS[index]).exact
        assert not (set(ROOMS[index]) & set("LRAGKN"))


def test_a_room_that_moves_its_enemies_says_so():
    room = Room(45, ROOMS[45])
    assert not room.exact
    assert "K" in room.unmodelled


# ------------------------------------------------------------------------------ parsing

def test_parsing_lifts_objects_off_the_terrain():
    """`terrain[r][c]` answers "what would Lolo be standing on", never "what is on it"."""
    terrain, hearts, framers, enemies, lolo, door = parse_room(
        "|".join(room_with(**{"1_1": "H", "1_2": "h", "2_2": "O", "3_3": "M", "4_1": "@"})))
    assert lolo == (4, 1) and door == (0, 4)
    assert hearts == {(1, 1): False, (1, 2): True}
    assert framers == frozenset({(2, 2)})
    assert enemies == {(3, 3): "M"}
    assert terrain[1][1] == "." and terrain[2][2] == "." and terrain[3][3] == "."
    assert terrain[0][4] == "D", "the door stays in the terrain; it is ground Lolo stands on"


def test_a_room_without_lolo_is_refused():
    with pytest.raises(ValueError):
        parse_room("|".join(WALLED))          # a door but no Lolo


# ----------------------------------------------------------------------------- movement

@pytest.mark.parametrize("glyph", ["#", "T", "~"])
def test_rocks_trees_and_rivers_refuse_a_step(glyph):
    _, state = board(room_with(**{"4_1": "@", "4_2": glyph}))
    successor = move(state, "right")
    assert successor.lolo == (4, 1), f"{glyph} let Lolo through"


@pytest.mark.parametrize("glyph", sorted(WALKABLE - {"D"}))
def test_ordinary_ground_takes_a_step(glyph):
    _, state = board(room_with(**{"4_1": "@", "4_2": glyph}))
    assert move(state, "right").lolo == (4, 2)


def test_the_door_is_walkable_before_the_last_heart():
    """Standing on a shut door is allowed and simply does nothing."""
    _, state = board(room_with(**{"0_4": "#", "4_1": "@", "4_2": "D", "6_6": "H"}))
    successor = move(state, "right")
    assert successor.lolo == (4, 2)
    assert not successor.solved, "the room is not cleared while a heart is left"


@pytest.mark.parametrize("arrow,refused", [("v", "up"), ("^", "down"), ("<", "right"),
                                           (">", "left")])
def test_a_one_way_pass_refuses_only_its_own_reverse(arrow, refused):
    starts = {"up": (5, 3), "down": (3, 3), "right": (4, 2), "left": (4, 4)}
    for direction, start in starts.items():
        rows = room_with(**{f"{start[0]}_{start[1]}": "@", "4_3": arrow})
        _, state = board(rows)
        moved = move(state, direction).lolo != start
        assert moved == (direction != refused), \
            f"{arrow} {'refused' if not moved else 'allowed'} a step {direction}"


def test_one_way_allows_is_the_rule_on_its_own():
    assert not one_way_allows("<", DIRECTIONS["right"])
    assert one_way_allows("<", DIRECTIONS["left"])
    assert one_way_allows(".", DIRECTIONS["right"])


def test_a_refused_step_still_turns_lolo():
    """Bumping into a rock is a real move: it is how you aim the magic shot."""
    _, state = board(room_with(**{"4_1": "@", "4_2": "#"}), magic_shots=1)
    turned = move(state, "right")
    assert turned is not None and turned.lolo == (4, 1) and turned.facing == "right"


def test_a_refused_step_that_changes_nothing_at_all_is_not_a_successor():
    _, state = board(room_with(**{"4_1": "@", "4_2": "#"}))
    assert move(state, "right") is not None       # the turn is new
    assert move(move(state, "right"), "right") is None


# ----------------------------------------------------------------------------- pushing

def test_a_framer_is_pushed_one_cell():
    _, state = board(room_with(**{"4_1": "@", "4_2": "O"}))
    successor = move(state, "right")
    assert successor.lolo == (4, 2) and successor.framers == frozenset({(4, 3)})


def test_a_framer_against_a_wall_does_not_move():
    _, state = board(room_with(**{"4_1": "@", "4_2": "O", "4_3": "#"}))
    assert move(state, "right").lolo == (4, 1)


def test_two_framers_cannot_be_pushed_at_once():
    _, state = board(room_with(**{"4_1": "@", "4_2": "O", "4_3": "O"}))
    assert move(state, "right").lolo == (4, 1)


@pytest.mark.parametrize("glyph", ["H", "h", "D", "~"])
def test_a_framer_is_not_pushed_onto_an_object_or_into_a_river(glyph):
    _, state = board(room_with(**{"4_1": "@", "4_2": "O", "4_3": glyph}))
    assert move(state, "right").lolo == (4, 1), f"a Framer was pushed onto {glyph}"


def test_a_framer_may_be_pushed_against_a_one_way_arrow():
    """The arrow refuses a *walk* against it, not a *push*. Measured on the cartridge."""
    _, state = board(room_with(**{"4_1": "@", "4_2": "O", "4_3": "<"}))
    successor = move(state, "right")
    assert successor.lolo == (4, 2) and successor.framers == frozenset({(4, 3)})


def test_pushable_onto_holds_every_one_way_arrow():
    assert {"v", "<", "^", ">"} <= PUSHABLE_ONTO


# ------------------------------------------------------------------------ hearts, shots

def test_collecting_a_plain_heart_gives_no_shot():
    _, state = board(room_with(**{"4_1": "@", "4_2": "H"}))
    successor = move(state, "right")
    assert successor.hearts_left == 0 and successor.shots == 0


def test_collecting_a_magic_heart_gives_two_shots():
    """The whole point of the `$90`/`$91` split, which draws the same tile either way."""
    _, state = board(room_with(**{"4_1": "@", "4_2": "h"}))
    successor = move(state, "right")
    assert successor.shots == SHOTS_PER_MAGIC_HEART == 2


def test_the_door_opens_when_the_last_heart_is_taken():
    _, state = board(room_with(**{"4_1": "@", "4_2": "H"}))
    assert not state.door_open
    assert move(state, "right").door_open


# ------------------------------------------------------------------------- magic shots

def test_a_shot_turns_the_enemy_ahead_into_an_egg():
    instance, state = board(room_with(**{"4_1": "@", "4_3": "S"}), magic_shots=1)
    state = move(state, "right")                 # to (4, 2), facing right
    successor = shoot(state)
    assert successor.eggs == frozenset({(4, 3)})
    assert (4, 3) not in successor.alive
    assert successor.shots == 0


def test_a_second_shot_blasts_the_egg_out_of_the_room():
    _, state = board(room_with(**{"4_1": "@", "4_3": "S"}), magic_shots=2)
    state = shoot(move(state, "right"))
    successor = shoot(state)
    assert successor.eggs == frozenset() and successor.alive == frozenset()


def test_the_cell_an_egg_was_blasted_out_of_becomes_walkable():
    """The bug this test exists for: an enemy's cell staying blocked after it was cleared."""
    _, state = board(room_with(**{"4_1": "@", "4_2": "S"}), magic_shots=2)
    state = move(state, "right")                 # refused, but turns him right
    state = shoot(shoot(state))
    assert move(state, "right").lolo == (4, 2)


def test_an_egg_is_pushed_like_a_framer():
    _, state = board(room_with(**{"4_1": "@", "4_3": "S"}), magic_shots=1)
    state = shoot(move(state, "right"))
    successor = move(state, "right")
    assert successor.lolo == (4, 3) and successor.eggs == frozenset({(4, 4)})


def test_shooting_with_no_shots_does_nothing():
    _, state = board(room_with(**{"4_1": "@", "4_3": "S"}))
    assert shoot(move(state, "right")) is None


def test_shooting_at_nothing_does_nothing():
    _, state = board(room_with(**{"4_1": "@"}), magic_shots=1)
    assert shoot(move(state, "right")) is None


def test_shooting_before_the_first_move_does_nothing():
    """Lolo starts facing nowhere, and the cartridge will not fire before he has moved."""
    _, state = board(room_with(**{"4_1": "@", "4_2": "S"}), magic_shots=1)
    assert state.facing is None
    assert shoot(state) is None


def test_an_egg_is_not_pushed_into_a_river():
    """Rafts are refused rather than modelled (see the docs). This pins that choice."""
    _, state = board(room_with(**{"4_1": "@", "4_3": "S", "4_4": "~"}), magic_shots=1)
    state = shoot(move(state, "right"))
    assert move(state, "right") is None, "the egg was pushed into the river"


# ---------------------------------------------------------------------------- medusa

def test_a_medusa_kills_along_its_row_and_column():
    _, state = board(room_with(**{"2_1": "@", "4_6": "M"}))
    assert not state.dead
    assert move(move(state, "down"), "down").dead, "row 4 should be fatal"


def test_a_medusa_does_not_kill_off_its_lines():
    _, state = board(room_with(**{"2_1": "@", "4_6": "M"}))
    assert not move(state, "right").dead


@pytest.mark.parametrize("glyph", ["T", "O", "H", "h", "S"])
def test_a_tree_framer_heart_or_enemy_blocks_a_medusa(glyph):
    _, state = board(room_with(**{"4_1": "@", "4_4": glyph, "4_6": "M"}))
    assert not state.dead, f"{glyph} should have blocked the Medusa"


@pytest.mark.parametrize("glyph", ["#", "~", "=", ",", "*", "x", "<"])
def test_rocks_and_ground_do_not_block_a_medusa(glyph):
    """The most surprising measurement on this cartridge, and the one most likely to be
    mistaken for a bug in either implementation."""
    _, state = board(room_with(**{"4_1": "@", "4_4": glyph, "4_6": "M"}))
    assert state.dead, f"{glyph} should not have blocked the Medusa"


def test_medusa_shields_is_the_rule_on_its_own():
    assert {"T", "O", "H", "h", "e"} <= MEDUSA_SHIELDS
    assert not ({"#", "~", "=", ",", "*", "x"} & MEDUSA_SHIELDS)


def test_pushing_the_framer_that_was_shielding_lolo_kills_him():
    """A push can open a line, so the check belongs after the move and not inside it.

    Lolo shoves the Framer out of row 4 and steps into the cell it was shielding. Nothing is
    between him and the Medusa any more, and the move that killed him is the one he made.
    """
    _, state = board(room_with(**{"3_3": "@", "4_3": "O", "4_6": "M"}))
    assert not state.dead
    pushed = move(state, "down")
    assert pushed.lolo == (4, 3) and pushed.framers == frozenset({(5, 3)})
    assert pushed.dead


def test_a_medusa_shot_into_an_egg_stops_firing():
    _, state = board(room_with(**{"4_1": "@", "4_3": "M"}), magic_shots=1)
    state = move(state, "right")                 # to (4, 2): already in the line, and dead
    assert state.dead
    assert blocked_by_medusa(state.room, state.framers, state.eggs, state.hearts,
                             frozenset(), (4, 2)) is False, \
        "a Medusa that is no longer alive does not fire"


# ----------------------------------------------------------------------- the environment

def test_reset_reports_what_the_room_is():
    instance = game(38)
    state, info = instance.reset()
    assert info == {"room_index": 38, "room": "int 1-1", "hearts": 6, "shots": 0,
                    "door": (1, 1), "start": (6, 1), "exact": True, "unmodelled_enemies": ()}
    assert_string_literals(state)


def test_set_index_refuses_a_room_that_does_not_exist():
    instance = LoloGame()
    with pytest.raises(IndexError):
        instance.set_index(len(ROOMS))


def test_successors_are_pairs_and_never_the_state_itself():
    instance = game(38)
    successors = instance.successors(instance.state)
    assert_successors_contract(successors)
    assert all(successor != instance.state for _, successor in successors)


def test_a_goal_state_and_a_dead_state_expand_to_nothing():
    instance, state = board(room_with(**{"4_1": "@", "4_6": "M"}))
    dead = move(move(state, "right"), "right")
    assert instance.is_terminal(dead) and instance.successors(dead) == []


def test_the_stored_plan_for_int_1_1_clears_it():
    instance = game(38)
    plan = (["right"] * 5 + ["up"] * 2 + ["left"] * 2 + ["up"] * 3 + ["right"] * 2
            + ["left"] * 5)
    assert instance.validate(plan)


def test_a_state_reached_two_ways_compares_equal():
    """Search cannot close otherwise, and `successors`' self-loop filter becomes dead code."""
    instance = game(38)
    left = instance.simulate(["up", "right", "down"])[-1]
    right = instance.simulate(["right", "up", "down"])[-1]
    assert left == right and hash(left) == hash(right)
    assert left.depth == right.depth == 3


def test_facing_is_part_of_the_identity():
    """Two positions that differ only in where the next shot would go are different."""
    instance = game(38)
    facing_up = instance.simulate(["up", "down", "up"])[-1]
    facing_down = instance.simulate(["up", "up", "down"])[-1]
    assert facing_up.lolo == facing_down.lolo
    assert facing_up != facing_down


def test_step_reports_the_hearts_it_collected():
    instance = game(38)
    _, reward = instance.step("right")
    assert reward == 0
    for _ in range(2):
        _, reward = instance.step("right")
    assert reward == 1, "the third step right lands on a heart"


def test_get_actions_offers_the_four_directions_and_the_shot():
    assert sorted(str(action) for action in LoloGame().get_actions()) == \
        ["down", "left", "right", SHOOT, "up"]


def test_an_unknown_action_is_refused():
    with pytest.raises(ValueError):
        LoloAction("jump")


def test_render_draws_lolo_the_objects_and_the_enemies():
    instance = game(0)
    drawn = render(instance.state)
    assert drawn.split("\n") == list(ROOMS[0].split("|"))


# ------------------------------------------------------------- against the cartridge

@needs_rom
def test_the_rooms_still_match_the_cartridge():
    """The 163 rooms were decoded, not transcribed. This is what keeps them that way."""
    from planiverse.environments.gameboy.lolo_gb import read_rooms

    decoded = read_rooms(lolo_rom_path())
    assert len(decoded) == len(ROOMS)
    for index, rows in enumerate(decoded):
        assert "|".join(rows) == ROOMS[index], f"room {index} no longer matches the cartridge"


@needs_rom
def test_the_two_modules_spell_the_game_the_same_way():
    """`lolo.py` keeps its own copy of the alphabet so that it needs no PyBoy.

    Two copies can drift, and a drift here would be silent and total: the same eight rows of
    eight letters would describe two different games.
    """
    from planiverse.environments.gameboy import lolo_gb
    from planiverse.environments.gameboy_py import lolo as twin

    for name in ("ROCK", "TREE", "RIVER", "FLOOR", "BRIDGE", "FRAMER", "HEART", "MAGIC_HEART",
                 "DOOR", "DESERT", "BREAK_TILE", "FLOWER_BED", "MARKER", "LOLO", "ONE_WAY",
                 "ENEMY_GLYPHS", "HEART_GLYPHS", "SNAKEY", "MEDUSA"):
        assert getattr(twin, name) == getattr(lolo_gb, name), f"{name} differs between the two"
    for index in (0, 37, 38, 107, 108, 157, 158, 162):
        assert twin.room_label(index) == lolo_gb.room_label(index)
