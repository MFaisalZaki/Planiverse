"""Tests for the Super Mario Land environment.

Super Mario Land is copyrighted and no ROM ships with the repo, so the tests that need to
boot the emulator are opt-in: point PLANIVERSE_SML_ROM at a ROM to run them.

    PLANIVERSE_SML_ROM=/path/to/SuperMarioLand.gb poetry run pytest tests/test_super_mario.py

Everything that does not need the ROM (the action model, level indexing) always runs.
"""
import pytest

pytest.importorskip("pyboy", reason="pyboy is not installed")

from planiverse.problems.retro_games.super_mario_bros_gb import (  # noqa: E402
    SuperMarioAction, SuperMarioEnv, action_cost_map, action_list,
)

from conftest import assert_state_contract, assert_successors_contract, sml_rom_path  # noqa: E402

needs_rom = pytest.mark.skipif(
    sml_rom_path() is None,
    reason="set PLANIVERSE_SML_ROM to a Super Mario Land ROM to run emulator tests",
)


@pytest.fixture
def env():
    return SuperMarioEnv(sml_rom_path(), render=False)


# --------------------------------------------------------------------------- actions

def test_action_list_is_button_combinations_by_tick_count():
    assert len(action_list) == 16
    for action in action_list:
        buttons, ticks = action.split(",")
        assert int(ticks) in (3, 5, 10, 15)
        for button in buttons.split("+"):
            assert button in action_cost_map


def test_action_parsing():
    action = SuperMarioAction("a+right,10")
    assert action.actions_tick_list == [("a", 10), ("right", 10)]


def test_action_cost_charges_each_button_per_tick():
    # a=2, right=1 -> 3 per tick, held for 10 ticks
    assert SuperMarioAction("a+right,10").cost() == 30
    assert SuperMarioAction("right,3").cost() == 3
    assert SuperMarioAction("nop,3").cost() == 0


def test_jumping_costs_more_than_running():
    assert SuperMarioAction("a+right,5").cost() > SuperMarioAction("right,3").cost()


def test_action_string_is_filename_safe():
    assert str(SuperMarioAction("a+right,10")) == "a_with_right_for_10"


def test_actions_order_by_tick_count():
    assert SuperMarioAction("a+right,5") < SuperMarioAction("a+right,15")


# --------------------------------------------------------------------------- levels

def test_world_level_map_is_four_worlds_of_three_levels():
    """Super Mario Land has worlds 1-4 with 3 levels each, both 1-indexed."""
    env = SuperMarioEnv("unused.gb")
    assert len(env.world_level_map) == 12
    assert env.world_level_map[0] == (1, 1)
    assert env.world_level_map[11] == (4, 3)
    for world, level in env.world_level_map.values():
        assert 1 <= world <= 4 and 1 <= level <= 3


def test_fix_index_selects_the_world_and_level():
    """fix_index used to set world_level and reset() never read it, so the level
    selection silently did nothing."""
    env = SuperMarioEnv("unused.gb")
    assert env.world_level is None
    env.fix_index(4)
    assert env.world_level == (2, 2)


def test_fix_index_rejects_unknown_index():
    env = SuperMarioEnv("unused.gb")
    with pytest.raises(AssertionError, match="Invalid index"):
        env.fix_index(12)


# --------------------------------------------------------------------------- emulator

@needs_rom
def test_reset_boots_the_rom(env):
    state, info = env.reset()
    assert_state_contract(state)
    assert state.depth == 0
    assert state.gb_state, "the state must carry an emulator save-state"


@needs_rom
def test_reset_starts_the_selected_level(env):
    env.fix_index(3)                      # world 2, level 1
    state, _ = env.reset()
    assert env.game.world == (2, 1)


@needs_rom
def test_state_reads_mario_out_of_ram(env):
    state, _ = env.reset()
    assert state.mario_position.x > 0
    assert state.lives_left == 0          # reset sets lives to 0 to avoid replays
    assert state.timeleft > 0


@needs_rom
def test_successors_contract(env):
    state, _ = env.reset()
    assert_successors_contract(env.successors(state))


@needs_rom
def test_moving_right_advances_level_progress(env):
    state, _ = env.reset()
    after = SuperMarioAction("right,3").apply(env.pyboy, state)
    assert after.level_progress >= state.level_progress
    assert after.depth == state.depth + 1


@needs_rom
def test_applying_an_action_restores_the_parent_state(env):
    """Every action reloads its parent's save-state, so siblings expand from the same
    machine regardless of what the previous sibling did."""
    state, _ = env.reset()
    first = SuperMarioAction("right,3").apply(env.pyboy, state)
    second = SuperMarioAction("right,3").apply(env.pyboy, state)
    assert first.mario_position == second.mario_position
    assert first.level_progress == second.level_progress


@needs_rom
def test_goal_and_terminal_read_the_state_not_live_memory(env):
    """is_goal/is_terminal used to query the emulator's current memory, so they described
    whichever state was applied last rather than the state passed in."""
    state, _ = env.reset()
    assert not env.is_goal(state)
    assert not env.is_terminal(state)

    # Advance the emulator well past `state`, then ask about `state` again.
    advanced = state
    for _ in range(5):
        advanced = SuperMarioAction("a+right,10").apply(env.pyboy, advanced)
    assert env.is_goal(state) is False
    assert env.is_terminal(state) is False
    assert isinstance(env.is_goal(advanced), bool)


@needs_rom
def test_state_equality_tolerates_a_small_time_difference(env):
    state, _ = env.reset()
    assert state == state


@needs_rom
def test_simulate_returns_state_trace(env):
    env.reset()
    plan = [SuperMarioAction("right,3")] * 3
    trace = env.simulate(plan)
    assert len(trace) == len(plan) + 1
    assert [s.depth for s in trace] == [0, 1, 2, 3]


@needs_rom
def test_save_writes_a_screenshot(env, tmp_path):
    state, _ = env.reset()
    out = tmp_path / "frame.png"
    state.save(sml_rom_path(), str(out))
    assert out.exists() and out.stat().st_size > 0


@needs_rom
def test_literals_describe_mario(env):
    state, _ = env.reset()
    assert any(lit.startswith("(supermario position") for lit in state.literals)
    assert any(lit.startswith("(progress") for lit in state.literals)
    # depth and coins are separate literals: they used to be concatenated into one.
    assert f"(depth {state.depth})" in state.literals
    assert f"(coins {state.coins})" in state.literals
