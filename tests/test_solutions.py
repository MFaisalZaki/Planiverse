"""A known solution for every level, replayed.

These are regression tests for the *levels*, not for the planners. A level is a piece of
transcribed data, and the failure mode that data has is silent: a wall in the wrong cell or
two block types swapped still parses, still renders, and still looks like a puzzle — it just
quietly becomes a different one, or an unsolvable one. Two such slips were found in the
Puzznic levels by reading the cartridge back (`test_puzznic.py`), and a stored solution per
level is what would have caught them without a ROM in hand.

Each plan here was produced by a planner and validated against the environment, so a plan
that stops solving its level means the level changed. Regenerate a plan only once you know
*why* it went stale — a corrected level is a good reason, an accidental edit is not.

`tests/data/*_solutions.json` maps a level index to its action sequence. Levels with no
entry are ones no planner has solved within the benchmark's budget; the coverage test below
pins how many those are, so the gap cannot widen unnoticed.
"""
import json
import os

import pytest

from planiverse.environments.gameboy_py.flipull import STAGES, FlipullGame
from planiverse.environments.gameboy_py.puzznic import PuzznicGame

from conftest import flipull_rom_path, puzznic_rom_path

DATA = os.path.join(os.path.dirname(__file__), "data")


def solutions(name):
    with open(os.path.join(DATA, f"{name}_solutions.json")) as handle:
        return {int(index): plan for index, plan in json.load(handle).items()}


PUZZNIC_SOLUTIONS = solutions("puzznic")
FLIPULL_SOLUTIONS = solutions("flipull")


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


def test_every_flipull_stage_has_a_solution():
    """Flipull's stages were generated against a reachability check, so every one of them
    is known to be solvable and none may lose its solution."""
    assert sorted(FLIPULL_SOLUTIONS) == list(range(len(STAGES)))


def test_puzznic_solution_coverage_does_not_shrink():
    """Not every Puzznic level has a stored solution: some are unsolved at the benchmark's
    budget. That is a known gap rather than an accepted one, so it is pinned — a level
    losing its solution must fail here rather than pass quietly."""
    unsolved = sorted(set(range(50)) - set(PUZZNIC_SOLUTIONS))
    assert unsolved == [15, 17, 28, 34, 35, 42, 46, 47, 49], \
        "the set of Puzznic levels without a stored solution changed"
    # Levels 50-127 were added from the cartridge after the benchmark ran, so none of them
    # has been solved yet. They are listed here rather than silently uncovered.
    assert not set(range(50, 128)) & set(PUZZNIC_SOLUTIONS), \
        "levels 50-127 now have solutions; record them and update this test"


@pytest.mark.parametrize("name,count", [
    ("puzznic", 128), ("flipull", len(STAGES)), ("puzznic_gb", 128), ("flipull_gb", 32),
])
def test_solution_indices_are_in_range(name, count):
    for index in solutions(name):
        assert 0 <= index < count, f"{name} has a solution for a level that does not exist"


# ------------------------------------------------------- the cartridges, when a ROM is here
# These plans were replayed on the real hardware by the benchmark and recorded only once the
# cartridge itself reported the stage cleared, so they are the strongest evidence in the
# repository about what these games do. Replaying one costs an emulator boot, hence `slow`.

PUZZNIC_GB_SOLUTIONS = solutions("puzznic_gb")
FLIPULL_GB_SOLUTIONS = solutions("flipull_gb")


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
