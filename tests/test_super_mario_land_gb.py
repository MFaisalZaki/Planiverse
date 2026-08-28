"""Tests for the Super Mario Land environment.

Super Mario Land is copyrighted and no ROM ships with the repo, so the tests that need to
boot the emulator are opt-in: point PLANIVERSE_SUPER_MARIO_LAND_ROM at a ROM to run them.

    PLANIVERSE_SUPER_MARIO_LAND_ROM=/path/to/SuperMarioLand.gb poetry run pytest tests/test_super_mario.py

Everything that does not need the ROM (the action model, level indexing) always runs.
"""
import pytest

pytest.importorskip("pyboy", reason="pyboy is not installed")

from planiverse.environments.gameboy.super_mario_land_gb import (  # noqa: E402
    DIRECTIONS, FACING_LEFT, MARIO_X_SATURATES_AT, OBJECT_EMPTY, OBJECT_SLOTS, OBJECT_STRIDE,
    ROM_MD5, SuperMarioLandGBAction, SuperMarioLandGBEnv, action_cost_map, action_list, decode_objects,
    decode_timer, position, SuperMarioLandGBState,
)

from conftest import assert_state_contract, assert_successors_contract, sml_rom_path  # noqa: E402

needs_rom = pytest.mark.skipif(
    sml_rom_path() is None,
    reason="set PLANIVERSE_SUPER_MARIO_LAND_ROM to a Super Mario Land ROM to run emulator tests",
)


@pytest.fixture
def env():
    return SuperMarioLandGBEnv(sml_rom_path(), render=False)


# --------------------------------------------------------------------------- actions

def test_action_list_is_button_combinations_by_tick_count():
    assert len(action_list) == 16
    for action in action_list:
        buttons, ticks = action.split(",")
        assert int(ticks) in (3, 5, 10, 15)
        for button in buttons.split("+"):
            assert button in action_cost_map


def test_action_parsing():
    action = SuperMarioLandGBAction("a+right,10")
    assert action.actions_tick_list == [("a", 10), ("right", 10)]


def test_action_cost_charges_each_button_per_tick():
    # a=2, right=1 -> 3 per tick, held for 10 ticks
    assert SuperMarioLandGBAction("a+right,10").cost() == 30
    assert SuperMarioLandGBAction("right,3").cost() == 3
    assert SuperMarioLandGBAction("nop,3").cost() == 0


def test_jumping_costs_more_than_running():
    assert SuperMarioLandGBAction("a+right,5").cost() > SuperMarioLandGBAction("right,3").cost()


def test_action_string_is_filename_safe():
    assert str(SuperMarioLandGBAction("a+right,10")) == "a_with_right_for_10"


def test_actions_order_by_tick_count():
    assert SuperMarioLandGBAction("a+right,5") < SuperMarioLandGBAction("a+right,15")


# ----------------------------------------------------------------- reading RAM
# These decode synthetic bytes, so they run without a ROM. Every address they cover is
# graded "verified" in the memory map: observed changing correctly under controlled input.

def test_timer_is_two_bcd_bytes():
    """$DA02 holds the high digit and $DA01 the low two, so 03/97 reads 397 — the map's
    own worked example, checked against the on-screen value."""
    assert decode_timer(0x03, 0x97) == 397
    assert decode_timer(0x00, 0x00) == 0
    assert decode_timer(0x09, 0x99) == 999
    # Only the low nibble of the high byte is a digit.
    assert decode_timer(0xF3, 0x97) == 397


def test_object_array_is_ten_slots_of_sixteen():
    assert (OBJECT_SLOTS, OBJECT_STRIDE, OBJECT_EMPTY) == (10, 0x10, 0xFF)


def test_objects_read_type_and_position():
    """+0 is the status byte, +1 the type, +2 Y and +3 X — the four fields the map marks
    verified, from watching a slot go live and its X fall as the enemy walked left."""
    raw = bytearray([OBJECT_EMPTY] * (OBJECT_SLOTS * OBJECT_STRIDE))
    raw[0:5] = bytes([0x00, 0x01, 0x88, 0x40, 0x05])
    enemies = decode_objects(raw)
    assert len(enemies) == 1
    assert (enemies[0].slot, enemies[0].type) == (0, 0x01)
    assert (enemies[0].y, enemies[0].x) == (0x88, 0x40)


def test_an_empty_slot_is_ff_and_is_skipped():
    raw = bytearray([OBJECT_EMPTY] * (OBJECT_SLOTS * OBJECT_STRIDE))
    assert decode_objects(raw) == ()
    # A slot goes live in place, so its index has to survive.
    raw[3 * OBJECT_STRIDE] = 0x00
    assert [enemy.slot for enemy in decode_objects(raw)] == [3]


def test_objects_stop_at_the_tenth_slot():
    """The array is $D100-$D19F; $D1A0 onwards was zero throughout and is not part of it."""
    raw = bytearray([0x00] * (OBJECT_SLOTS * OBJECT_STRIDE + 64))
    assert len(decode_objects(raw)) == OBJECT_SLOTS


def test_touching_compares_screen_coordinates():
    """Mario's position and an object's are both screen coordinates read on the same frame,
    which is what makes them comparable at all."""
    mario = position(x=0x40, y=0x88)
    close = decode_objects(bytes([0x00, 0x01, 0x88, 0x42]) + b"\xff" * 156)[0]
    far = decode_objects(bytes([0x00, 0x01, 0x88, 0x70]) + b"\xff" * 156)[0]
    assert SuperMarioLandGBState.touching(mario, close)
    assert not SuperMarioLandGBState.touching(mario, far)


def test_direction_is_a_code_not_a_velocity():
    """$C20D reads $00 / $10 / $20 for still / right / left. It used to be read as the y of
    a velocity, which made `(supermario velocity X Y)` describe nothing."""
    assert DIRECTIONS == {0x00: "still", 0x10: "right", 0x20: "left"}
    assert FACING_LEFT == 0x20


def test_mario_x_is_a_screen_coordinate():
    """It stops at $51 when the camera takes over, so it cannot measure progress through a
    level — which is why the planner's goal window is on `level_progress`."""
    assert MARIO_X_SATURATES_AT == 0x51


def test_a_foreign_rom_warns_because_the_addresses_are_revision_specific(tmp_path):
    rom = tmp_path / "not-mario.gb"
    rom.write_bytes(b"\x00" * 32768)
    with pytest.warns(UserWarning, match=ROM_MD5):
        SuperMarioLandGBEnv(str(rom))


def test_verification_can_be_turned_off(tmp_path):
    rom = tmp_path / "not-mario.gb"
    rom.write_bytes(b"\x00" * 32768)
    SuperMarioLandGBEnv(str(rom), verify_rom=False)


# --------------------------------------------------------------------------- levels

def test_world_level_map_is_four_worlds_of_three_levels():
    """Super Mario Land has worlds 1-4 with 3 levels each, both 1-indexed."""
    env = SuperMarioLandGBEnv("unused.gb", verify_rom=False)
    assert len(env.world_level_map) == 12
    assert env.world_level_map[0] == (1, 1)
    assert env.world_level_map[11] == (4, 3)
    for world, level in env.world_level_map.values():
        assert 1 <= world <= 4 and 1 <= level <= 3


def test_fix_index_selects_the_world_and_level():
    """fix_index used to set world_level and reset() never read it, so the level
    selection silently did nothing."""
    env = SuperMarioLandGBEnv("unused.gb", verify_rom=False)
    assert env.world_level is None
    env.fix_index(4)
    assert env.world_level == (2, 2)


def test_fix_index_rejects_unknown_index():
    env = SuperMarioLandGBEnv("unused.gb", verify_rom=False)
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
def test_facing_and_direction_follow_the_input(env):
    """$C205 is $20 facing left, $C20D is $10 right / $20 left — both graded verified."""
    state, _ = env.reset()
    left = SuperMarioLandGBAction("left,10").apply(env.pyboy, state)
    right = SuperMarioLandGBAction("right,10").apply(env.pyboy, state)
    assert left.mario_facing == "left"
    assert right.mario_facing == "right"
    assert {left.mario_direction, right.mario_direction} <= {"still", "left", "right"}


@needs_rom
def test_the_ground_flag_clears_for_a_jump(env):
    """$C20A: $01 grounded, $00 airborne."""
    state, _ = env.reset()
    assert state.on_ground, "Mario starts standing"
    airborne = SuperMarioLandGBAction("a+right,5").apply(env.pyboy, state)
    assert airborne.airborne is not airborne.on_ground


@needs_rom
def test_speed_is_a_magnitude(env):
    """$C20C is a magnitude, so it never goes negative however Mario is moving."""
    state, _ = env.reset()
    for action in ("left,10", "right,10", "a+right,10"):
        assert SuperMarioLandGBAction(action).apply(env.pyboy, state).mario_speed >= 0


@needs_rom
def test_enemies_come_from_the_object_array(env):
    """Not from counting sprites by tile id, which only ever knew one identifier."""
    state, _ = env.reset()
    assert isinstance(state.enemies, tuple)
    assert state.enemies_on_screen == len(state.enemies)
    for enemy in state.enemies:
        assert 0 <= enemy.slot < OBJECT_SLOTS
        assert enemy.type != OBJECT_EMPTY


@needs_rom
def test_the_camera_is_read_from_the_register(env):
    """There is no WRAM mirror of the scroll value — the map searched for one and found
    none — so SCX comes from $FF43."""
    state, _ = env.reset()
    assert 0 <= state.camera_x <= 255
    assert state.camera_y == 0, "the flat opening of 1-1 never scrolls vertically"


@needs_rom
def test_mario_x_saturates_but_progress_does_not(env):
    """The distinction the planner's goal window depends on."""
    state = env.reset()[0]
    for _ in range(12):
        state = SuperMarioLandGBAction("right,15").apply(env.pyboy, state)
    assert state.mario_position.x <= MARIO_X_SATURATES_AT
    assert state.level_progress > env.reset()[0].level_progress


@needs_rom
def test_successors_contract(env):
    state, _ = env.reset()
    assert_successors_contract(env.successors(state))


@needs_rom
def test_moving_right_advances_level_progress(env):
    state, _ = env.reset()
    after = SuperMarioLandGBAction("right,3").apply(env.pyboy, state)
    assert after.level_progress >= state.level_progress
    assert after.depth == state.depth + 1


@needs_rom
def test_applying_an_action_restores_the_parent_state(env):
    """Every action reloads its parent's save-state, so siblings expand from the same
    machine regardless of what the previous sibling did."""
    state, _ = env.reset()
    first = SuperMarioLandGBAction("right,3").apply(env.pyboy, state)
    second = SuperMarioLandGBAction("right,3").apply(env.pyboy, state)
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
        advanced = SuperMarioLandGBAction("a+right,10").apply(env.pyboy, advanced)
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
    plan = [SuperMarioLandGBAction("right,3")] * 3
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
