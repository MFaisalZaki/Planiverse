"""A known solution for every level, replayed.

These are regression tests for the *levels*, not for the planners. A level is a piece of
transcribed data, and the failure mode that data has is silent: a wall in the wrong cell or
two block types swapped still parses, still renders, and still looks like a puzzle; it just
quietly becomes a different one, or an unsolvable one. Two such slips were found in the
Puzznic levels by reading the cartridge back (`test_puzznic.py`), and a stored solution per
level is what would have caught them without a ROM in hand.

Each plan here was produced by a planner and validated against the environment, so a plan
that stops solving its level means the level changed. Regenerate a plan only once you know
*why* it went stale: a corrected level is a good reason, an accidental edit is not.

`tests/data/*_solutions.json` maps a level index to its action sequence. Levels with no
entry are ones no planner has solved within the benchmark's budget; the coverage test below
pins how many those are, so the gap cannot widen unnoticed.
"""
import json
import os

import pytest

from planiverse.environments.gameboy_py.flipull import STAGES, FlipullGame
from planiverse.environments.gameboy_py.lolo import EXACT_ROOMS, LoloGame
from planiverse.environments.gameboy_py.puzznic import PuzznicGame

DATA = os.path.join(os.path.dirname(__file__), "data")


def solutions(name):
    with open(os.path.join(DATA, f"{name}_solutions.json")) as handle:
        return {int(index): plan for index, plan in json.load(handle).items()}


PUZZNIC_SOLUTIONS = solutions("puzznic")
FLIPULL_SOLUTIONS = solutions("flipull")


@pytest.mark.parametrize("index", sorted(PUZZNIC_SOLUTIONS))
def test_the_stored_puzznic_solution_still_clears_its_level(index):
    env = PuzznicGame()
    env.set_index(index)
    env.reset()
    plan = PUZZNIC_SOLUTIONS[index]
    assert env.validate(plan), \
        f"puzznic level {index} is no longer cleared by its stored {len(plan)}-action plan"


@pytest.mark.parametrize("index", sorted(FLIPULL_SOLUTIONS))
def test_the_stored_flipull_solution_still_clears_its_stage(index):
    env = FlipullGame()
    env.set_index(index)
    env.reset()
    plan = FLIPULL_SOLUTIONS[index]
    assert env.validate(plan), \
        f"flipull stage {index} is no longer cleared by its stored {len(plan)}-action plan"


def test_every_flipull_stage_has_a_solution():
    """Flipull's stages were generated against a reachability check, so every one of them
    is known to be solvable and none may lose its solution."""
    assert sorted(FLIPULL_SOLUTIONS) == list(range(len(STAGES)))


def test_puzznic_solution_coverage_does_not_shrink():
    """Not every Puzznic level has a stored solution: some are unsolved at the benchmark's
    budget. That is a known gap rather than an accepted one, so it is pinned: a level
    losing its solution must fail here rather than pass quietly."""
    unsolved = sorted(set(range(50)) - set(PUZZNIC_SOLUTIONS))
    assert unsolved == [15, 17, 28, 34, 35, 42, 46, 47, 49], \
        "the set of Puzznic levels without a stored solution changed"
    # Levels 50-127 were added from the cartridge after the benchmark ran, so none of them
    # has been solved yet. They are listed here rather than silently uncovered.
    assert not set(range(50, 128)) & set(PUZZNIC_SOLUTIONS), \
        "levels 50-127 now have solutions; record them and update this test"


# ------------------------------------------------------------------------------- Lolo

LOLO_SOLUTIONS = solutions("lolo")

#: Every stored plan was found by breadth-first search over the Python twin, with the magic
#: shot meter seeded to two (see `LoloGame(magic_shots=...)`). Pinned as the solved set rather
#: than the unsolved one: it is much the shorter list, and it is the one that must not shrink.
LOLO_SOLVED = [
    0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 38, 39, 41, 45, 54, 56,
    57, 66, 75, 81, 120, 158, 160,
]

#: The plans that also cleared their room on the original game, recorded when they were
#: replayed on it. Every plan for a room the model is exact for is here; most of the plans for
#: a room whose enemies the model freezes are not, because on the original those enemies moved
#: and killed Lolo. See `docs/environments/lolo.md`.
LOLO_CARTRIDGE_VALIDATED = [0, 1, 12, 13, 20, 38, 39, 41, 54, 57]


@pytest.mark.parametrize("index", sorted(LOLO_SOLUTIONS))
def test_the_stored_lolo_solution_still_clears_its_room(index):
    game = LoloGame(magic_shots=2)
    game.set_index(index)
    game.reset()
    plan = LOLO_SOLUTIONS[index]
    assert game.validate(plan), \
        f"lolo room {index} is no longer cleared by its stored {len(plan)}-action plan"


def test_lolo_solution_coverage_does_not_shrink():
    """Most rooms have no stored plan, and that is a known gap rather than an accepted one.

    Two things put a room out of reach. Some need a mechanic the twin does not model (a raft
    ridden across a river, or the hammer), and some are simply too wide for breadth-first
    search at the quarter-million states this was run to. Pinning the solved set means a room
    *losing* its plan fails here rather than quietly widening the gap, which is the failure
    that matters: it means the room changed.
    """
    assert sorted(LOLO_SOLUTIONS) == LOLO_SOLVED, \
        "the set of Lolo rooms with a stored solution changed"


def test_every_lolo_plan_for_an_exactly_modelled_room_was_validated_on_the_original():
    """The claim the module's docstring makes, as a test.

    A room whose only enemies are Snakey and Medusa is modelled exactly, because neither of
    them ever moves in the original game either. Every plan found for one of those rooms
    cleared it there. If that stops being true, the stated ruleset is wrong somewhere.
    """
    exactly_modelled = sorted(set(LOLO_SOLUTIONS) & set(EXACT_ROOMS))
    assert exactly_modelled == [0, 1, 38, 39, 41, 54, 57]
    assert set(exactly_modelled) <= set(LOLO_CARTRIDGE_VALIDATED), \
        "a plan for an exactly-modelled room no longer clears it on the original"


def test_most_lolo_plans_for_approximated_rooms_do_not_survive_the_original():
    """The other half of the same claim, and the reason `EXACT_ROOMS` exists.

    For a room whose enemies the model freezes, a plan found here is a plan against a strictly
    easier puzzle. Three of the twenty-five happen to work anyway; the rest walk Lolo into an
    enemy that was not standing still. This is pinned so that the size of the gap is a number
    somebody has to look at rather than a caveat in a docstring.
    """
    approximated = set(LOLO_SOLUTIONS) - set(EXACT_ROOMS)
    survived = approximated & set(LOLO_CARTRIDGE_VALIDATED)
    assert len(approximated) == 25 and sorted(survived) == [12, 13, 20]


@pytest.mark.parametrize("name,count", [
    ("puzznic", 128), ("flipull", len(STAGES)), ("lolo", 163),
])
def test_solution_indices_are_in_range(name, count):
    for index in solutions(name):
        assert 0 <= index < count, f"{name} has a solution for a level that does not exist"
