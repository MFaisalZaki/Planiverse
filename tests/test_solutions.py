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

from planiverse.environments.gameboy_py.boxxle2 import LEVELS, Boxxle2Game
from planiverse.environments.gameboy_py.flipull import STAGES, FlipullGame
from planiverse.environments.gameboy_py.lolo import EXACT_ROOMS, LoloGame
from planiverse.environments.gameboy_py.puzznic import PuzznicGame

from conftest import (
    boxxle2_rom_path, flipull_rom_path, lolo_rom_path, puzznic_rom_path,
)

DATA = os.path.join(os.path.dirname(__file__), "data")


def solutions(name):
    with open(os.path.join(DATA, f"{name}_solutions.json")) as handle:
        return {int(index): plan for index, plan in json.load(handle).items()}


PUZZNIC_SOLUTIONS = solutions("puzznic")
FLIPULL_SOLUTIONS = solutions("flipull")
BOXXLE2_SOLUTIONS = solutions("boxxle2")


@pytest.mark.parametrize("index", sorted(PUZZNIC_SOLUTIONS))
def test_the_stored_puzznic_solution_still_clears_its_level(index):
    env = PuzznicGame()
    env.fix_index(index)
    env.reset()
    plan = PUZZNIC_SOLUTIONS[index]
    assert env.validate(plan), \
        f"puzznic level {index} is no longer cleared by its stored {len(plan)}-action plan"


@pytest.mark.parametrize("index", sorted(FLIPULL_SOLUTIONS))
def test_the_stored_flipull_solution_still_clears_its_stage(index):
    env = FlipullGame()
    env.fix_index(index)
    env.reset()
    plan = FLIPULL_SOLUTIONS[index]
    assert env.validate(plan), \
        f"flipull stage {index} is no longer cleared by its stored {len(plan)}-action plan"


@pytest.mark.parametrize("index", sorted(BOXXLE2_SOLUTIONS))
def test_the_stored_boxxle2_solution_still_clears_its_level(index):
    env = Boxxle2Game()
    env.fix_index(index)
    env.reset()
    plan = BOXXLE2_SOLUTIONS[index]
    assert env.validate(plan), \
        f"boxxle2 level {index} is no longer cleared by its stored {len(plan)}-action plan"


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


#: The levels with a stored plan. Pinned as the solved set rather than the unsolved one; it is
#: the shorter list of the two, and it is the one that must not shrink.
BOXXLE2_SOLVED = [
    0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 17, 20, 22, 24, 25, 26, 27, 28, 29, 30, 46, 52, 54, 55, 57,
    58, 59, 62, 63, 67, 68, 69, 72, 76, 77, 78, 92, 97, 113, 114, 115, 116,
]


def test_boxxle2_solution_coverage_does_not_shrink():
    """Not every Boxxle II level has a stored plan, and that is a known gap rather than an
    accepted one.

    Most of the plans that exist were reconstructed from a human walkthrough's box-push
    notation; the rest came out of a solver. What is left over is the tail of the cartridge:
    levels running to fifty-nine boxes, where Sokoban stops being searchable at any budget
    this repository spends. Pinning the set means a level *losing* its plan fails here rather
    than quietly widening the gap, which is the failure that matters: it means the level
    changed.
    """
    assert sorted(BOXXLE2_SOLUTIONS) == BOXXLE2_SOLVED, \
        "the set of Boxxle II levels with a stored solution changed"


# ------------------------------------------------------------------------------- Lolo

LOLO_SOLUTIONS = solutions("lolo")

#: Every stored plan was found by breadth-first search over the Python twin, with the magic
#: shot meter seeded to two (see `LoloGame(magic_shots=...)`). Pinned as the solved set rather
#: than the unsolved one: it is much the shorter list, and it is the one that must not shrink.
LOLO_SOLVED = [
    0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 38, 39, 41, 45, 54, 56,
    57, 66, 75, 81, 120, 158, 160,
]

#: The plans that also cleared their room on the real cartridge. Every plan for a room the twin
#: models exactly is here; most of the plans for a room whose enemies the twin freezes are not,
#: because on the cartridge those enemies moved and killed Lolo. See `docs/environments/lolo.md`.
LOLO_CARTRIDGE_VALIDATED = [0, 1, 12, 13, 20, 38, 39, 41, 54, 57]


@pytest.mark.parametrize("index", sorted(LOLO_SOLUTIONS))
def test_the_stored_lolo_solution_still_clears_its_room(index):
    game = LoloGame(magic_shots=2)
    game.fix_index(index)
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


def test_every_lolo_plan_for_an_exactly_modelled_room_was_validated_on_the_cartridge():
    """The claim the twin's docstring makes, as a test.

    A room whose only enemies are Snakey and Medusa is modelled exactly, because neither of
    them ever moves on the cartridge either. Every plan found for one of those rooms cleared it
    on the real hardware. If that stops being true, the stated ruleset is wrong somewhere.
    """
    exactly_modelled = sorted(set(LOLO_SOLUTIONS) & set(EXACT_ROOMS))
    assert exactly_modelled == [0, 1, 38, 39, 41, 54, 57]
    assert set(exactly_modelled) <= set(LOLO_CARTRIDGE_VALIDATED), \
        "a plan for an exactly-modelled room no longer clears it on the cartridge"


def test_most_lolo_plans_for_approximated_rooms_do_not_survive_the_cartridge():
    """The other half of the same claim, and the reason `EXACT_ROOMS` exists.

    For a room whose enemies the twin freezes, a plan found here is a plan against a strictly
    easier puzzle. Three of the twenty-five happen to work anyway; the rest walk Lolo into an
    enemy that was not standing still. This is pinned so that the size of the gap is a number
    somebody has to look at rather than a caveat in a docstring.
    """
    approximated = set(LOLO_SOLUTIONS) - set(EXACT_ROOMS)
    survived = approximated & set(LOLO_CARTRIDGE_VALIDATED)
    assert len(approximated) == 25 and sorted(survived) == [12, 13, 20]


@pytest.mark.parametrize("name,count", [
    ("puzznic", 128), ("flipull", len(STAGES)), ("puzznic_gb", 128), ("flipull_gb", 32),
    ("boxxle2", 120), ("boxxle2_gb", 120), ("lolo", 163), ("lolo_gb", 163),
])
def test_solution_indices_are_in_range(name, count):
    for index in solutions(name):
        assert 0 <= index < count, f"{name} has a solution for a level that does not exist"


# ------------------------------------------------------- the cartridges, when a ROM is here
# These plans were replayed on the real hardware by the benchmark and recorded only once the
# cartridge itself reported the stage cleared, so they are the strongest evidence in the
# repository about what these games do. Replaying one costs an emulator boot, so they are marked `slow`.

PUZZNIC_GB_SOLUTIONS = solutions("puzznic_gb")
FLIPULL_GB_SOLUTIONS = solutions("flipull_gb")
BOXXLE2_GB_SOLUTIONS = solutions("boxxle2_gb")
LOLO_GB_SOLUTIONS = solutions("lolo_gb")


def as_button_string(action):
    """`"a_with_right_for_16"` back into the `"a+right,16"` spelling actions parse from."""
    buttons, _, ticks = action.rpartition("_for_")
    return f'{buttons.replace("_with_", "+")},{ticks}'


def test_every_flipull_gb_stage_has_a_hardware_validated_solution():
    """All 32 cartridge stages were cleared on the real ROM, so none may lose its plan."""
    assert sorted(FLIPULL_GB_SOLUTIONS) == list(range(32))


@pytest.mark.slow
@pytest.mark.skipif(
    puzznic_rom_path() is None,
    reason='set PLANIVERSE_PUZZNIC_ROM to a "Puzznic (J).gb" ROM to run cartridge tests')
def test_a_cartridge_plan_still_clears_its_round():
    from planiverse.environments.gameboy.puzznic_gb import PuzznicGBEnv

    index = min(PUZZNIC_GB_SOLUTIONS)
    env = PuzznicGBEnv(puzznic_rom_path())
    env.fix_index(index)
    trace = env.simulate([as_button_string(a) for a in PUZZNIC_GB_SOLUTIONS[index]])
    env.close()
    assert trace[-1].stage_cleared, f"puzznic_gb round {index} was not cleared on the ROM"


@pytest.mark.slow
@pytest.mark.skipif(
    flipull_rom_path() is None,
    reason='set PLANIVERSE_FLIPULL_ROM to a "Flipull (USA).gb" ROM to run cartridge tests')
def test_a_cartridge_flipull_plan_still_clears_its_stage():
    from planiverse.environments.gameboy.flipull_gb import FlipullGBEnv

    index = min(FLIPULL_GB_SOLUTIONS)
    env = FlipullGBEnv(flipull_rom_path())
    env.fix_index(index)
    trace = env.simulate([as_button_string(a) for a in FLIPULL_GB_SOLUTIONS[index]])
    env.close()
    assert env.is_goal(trace[-1]), f"flipull_gb stage {index} was not cleared on the ROM"


@pytest.mark.slow
@pytest.mark.skipif(
    boxxle2_rom_path() is None,
    reason='set PLANIVERSE_BOXXLE2_ROM to a "Boxxle II (USA, Europe).gb" ROM to run '
           "cartridge tests")
def test_a_cartridge_boxxle2_plan_still_clears_its_level():
    from planiverse.environments.gameboy.boxxle2_gb import Boxxle2GBEnv

    index = min(BOXXLE2_GB_SOLUTIONS)
    env = Boxxle2GBEnv(boxxle2_rom_path())
    env.fix_index(index)
    trace = env.simulate([as_button_string(a) for a in BOXXLE2_GB_SOLUTIONS[index]])
    env.close()
    assert trace[-1].solved, f"boxxle2_gb level {index} was not cleared on the ROM"


@pytest.mark.slow
@pytest.mark.skipif(
    lolo_rom_path() is None,
    reason='set PLANIVERSE_LOLO_ROM to an "Adventures of Lolo (U) [S][!].gb" ROM to run '
           "cartridge tests")
@pytest.mark.parametrize("index", LOLO_CARTRIDGE_VALIDATED)
def test_a_cartridge_lolo_plan_still_clears_its_room(index):
    """Every one of these was replayed on the real hardware and the cartridge itself agreed the
    room was cleared, so they are the strongest evidence in this repository about what the
    stated Lolo rules actually are."""
    from planiverse.environments.gameboy.lolo_gb import LoloGBEnv

    env = LoloGBEnv(lolo_rom_path(), magic_shots=2)
    env.fix_index(index)
    try:
        trace = env.simulate([as_button_string(a) for a in LOLO_GB_SOLUTIONS[index]])
        assert env.is_goal(trace[-1]), f"lolo_gb room {index} was not cleared on the ROM"
        assert not any(state.died for state in trace), \
            f"lolo_gb room {index} killed Lolo along the way"
    finally:
        env.close()


def test_the_lolo_files_agree_about_which_plans_the_cartridge_confirmed():
    """One file holds the twin's plans and the other the subset the ROM confirmed. A plan in
    the cartridge file with no twin plan behind it means one was regenerated without the
    other."""
    assert sorted(LOLO_GB_SOLUTIONS) == LOLO_CARTRIDGE_VALIDATED
    assert set(LOLO_GB_SOLUTIONS) <= set(LOLO_SOLUTIONS)
    for index, plan in LOLO_GB_SOLUTIONS.items():
        assert len(plan) == len(LOLO_SOLUTIONS[index]), \
            f"the two files disagree about how long room {index}'s plan is"


def test_every_boxxle2_plan_was_replayed_on_the_cartridge():
    """The Python twin and the cartridge agree move for move, so every plan that solves a
    level here was checked on the ROM as well. A plan in one file and not the other means one
    of them was regenerated without the other."""
    assert sorted(BOXXLE2_GB_SOLUTIONS) == sorted(BOXXLE2_SOLUTIONS)
