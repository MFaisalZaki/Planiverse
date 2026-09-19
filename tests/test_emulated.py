"""The two emulator-backed environments: a Game Boy under PyBoy, and a console under
Stable-Retro.

Neither can be tested on a commercial cartridge, so the Game Boy tests run on an original
program assembled by `counter_rom.py`, which exercises the generic path (buttons in, tiles
and watched memory out, save states in between), and the Stable-Retro tests run on
Airstriker, the one game it ships a ROM for. The uniform contract and generator tests in
`test_interface.py` and `test_generators.py` cover both as well; what is here is the
behaviour particular to an emulator: that a state is a position rather than a frame count,
that a goal can be relative to the instance's own start, that losing a life is a dead end,
and that a replay lands on the same bytes every time.
"""
import json

import pytest

from planiverse.environments import make
from planiverse.environments.generation import bounded_search

from conftest import assert_string_literals, assert_successors_contract


# ================================================================================ Game Boy

def counter_env(**overrides):
    pytest.importorskip("pyboy", reason="pyboy is not installed")
    from counter_rom import COUNTER, PRESSES, counter_rom
    from planiverse.environments.emulated.game_boy import GameBoyEnv

    options = dict(watch={"counter": COUNTER, "presses": PRESSES},
                   goal={"memory": COUNTER, "at_least": 3}, actions=("right", "left", "a"),
                   hold=4, settle=2)
    return GameBoyEnv(counter_rom(), **{**options, **overrides})


@pytest.fixture(scope="module")
def counter():
    env = counter_env()
    yield env
    env.close()


def names(plan):
    return [str(action) for action in plan]


def test_a_cartridge_is_required_and_cannot_come_from_the_repository(monkeypatch, tmp_path):
    pytest.importorskip("pyboy", reason="pyboy is not installed")
    from planiverse.environments.emulated.game_boy import ROM_VARIABLE, GameBoyEnv

    monkeypatch.delenv(ROM_VARIABLE, raising=False)
    with pytest.raises(FileNotFoundError, match=ROM_VARIABLE):
        GameBoyEnv()
    with pytest.raises(FileNotFoundError):
        GameBoyEnv(str(tmp_path / "missing.gb"))


def test_the_generic_wrapper_reads_watched_memory_and_the_background_map(counter):
    counter.set_index(0)
    state, info = counter.reset()
    assert_string_literals(state)
    assert info["wrapper"] == "PyBoyGameWrapper", "no game wrapper claims the test cartridge"
    assert info["fields"] == {"counter": 0, "presses": 0}
    assert {"field(counter, 0)", "field(presses, 0)", "tile(0, 0, 0)"} <= state.literals
    trace = counter.simulate(["right", "right"])
    assert trace[-1].fields == {"counter": 2, "presses": 2}
    assert trace[-1].tiles[0][:2] == (2, 2), "the cartridge draws both counters top-left"


def test_buttons_do_what_the_cartridge_says(counter):
    counter.set_index(0)
    state, _ = counter.reset()
    successors = counter.successors(state)
    assert_successors_contract(successors)
    assert {str(action): successor.fields for action, successor in successors} == {
        "right": {"counter": 1, "presses": 1},
        "left": {"counter": 0, "presses": 1},      # never below zero, but still a press
        "a": {"counter": 0, "presses": 1},
    }
    assert len({successor for _, successor in successors}) == 2, \
        "left and A land on the same position, and the same position is the same state"


def test_search_finds_the_shortest_plan(counter):
    counter.set_index(0)
    counter.reset()
    outcome = bounded_search(counter, 200)
    assert names(outcome.plan) == ["right", "right", "right"]
    assert counter.validate(outcome.plan)
    assert not counter.validate(["right", "right"])


def test_states_are_positions_not_frame_counts(counter):
    """Two different histories that leave the cartridge in the same position are one
    state, which is what lets search close; the save states underneath may differ."""
    counter.set_index(0)
    first = counter.simulate(["right", "a"])[-1]
    second = counter.simulate(["left", "a"])[-1]
    assert first == second and hash(first) == hash(second)
    assert first.fields == {"counter": 0, "presses": 2}


def test_a_goal_can_be_relative_to_the_instance():
    """`delta` is measured from the instance's own initial state, after its opening."""
    env = counter_env(goal={"memory": 0xC000, "delta": 2})
    env.set_instance({"start": {"ticks": 120}, "timer_div": 0, "warmup": ["right", "right"]})
    state, info = env.reset()
    assert info["fields"]["counter"] == 2 and not env.is_goal(state)
    outcome = bounded_search(env, 100)
    assert names(outcome.plan) == ["right", "right"]
    env.close()


def test_a_dead_end_stops_expansion():
    env = counter_env(terminal={"memory": 0xC001, "at_least": 4})
    trace = env.simulate(["a"] * 4)
    assert env.is_terminal(trace[-1]) and not env.is_goal(trace[-1])
    assert "terminal-state" in trace[-1].literals
    assert env.successors(trace[-1]) == []
    assert not env.is_terminal(trace[-2])
    env.close()


def test_a_survival_goal_makes_the_step_count_part_of_the_state():
    env = counter_env(goal={"survive": 3})
    state, _ = env.reset()
    assert "steps(0)" in state.literals
    outcome = bounded_search(env, 100, progress=lambda s: s.progress)
    assert len(outcome.plan) == 3 and env.validate(outcome.plan)
    env.close()


def test_step_plays_statefully_and_render_gives_the_console_frames(counter):
    counter.set_index(0)
    counter.reset()
    state, reward = counter.step("right")
    assert state.fields["counter"] == 1 and isinstance(reward, (int, float))
    frames = counter.render()
    assert len(frames) == 2 and frames[0].size == (160, 144)


def test_a_written_render_is_the_console_frames_captioned(counter, tmp_path):
    from PIL import Image

    counter.set_index(0)
    counter.reset()
    counter.step("right")
    path = counter.render(tmp_path / "play.png")
    with Image.open(path) as sheet:
        assert sheet.height < 300, "two captioned screens in a row, not the tile grid typeset"
        assert sheet.width > 2 * 160
    trace = counter.simulate(["right", "right"])
    assert counter.render_trace(trace, tmp_path / "plan.gif", actions=["right", "right"]) \
        == tmp_path / "plan.gif"


def test_the_same_seed_gives_the_same_instance_down_to_the_bytes(counter):
    first = counter.generate_instance(seed=7)
    state = counter.reset()[0]
    second = counter.generate_instance(seed=7)
    again = counter.reset()[0]
    assert first == second and first["seed"] == 7
    assert state.literals == again.literals and state.snapshot == again.snapshot, \
        "the emulator is deterministic given the cartridge, the timer seed and the inputs"
    assert counter.reset()[0].snapshot == again.snapshot, "and so is a repeated reset"
    assert json.loads(json.dumps(first)) == first


def test_a_generated_instance_is_not_born_won_or_lost():
    env = counter_env(goal={"memory": 0xC000, "at_least": 1})
    instance = env.generate_instance(seed=0, warmup=(2, 2), attempts=50)
    assert len(instance["warmup"]) == 2 and "right" not in instance["warmup"], \
        "an opening that already reaches the goal is redrawn"
    state, info = env.reset()
    assert info["generated"] and not env.is_goal(state) and not env.is_terminal(state)
    env.close()


def test_a_checked_draw_keeps_its_witness(counter):
    counter.generate_instance(seed=2, warmup=(0, 2), solvable=True, search_limit=200)
    assert counter.witness and counter.validate(counter.witness)
    assert counter.witness_expansions > 0
    counter.set_index(0)
    assert counter.witness is None


def test_the_registry_builds_it_from_the_environment_variable():
    """The session fixture points `PLANIVERSE_GB_ROM` at the synthetic cartridge."""
    pytest.importorskip("pyboy", reason="pyboy is not installed")
    env = make("game_boy", index=0)
    state, info = env.reset()
    assert info["rom"] == "counter.gb" and info["goal"] == {"survive": 20}, \
        "with nothing named, the generic profile's goal is to last a while"
    assert env.successors(state)
    env.close()


def test_the_profiles_map_indices_to_start_arguments():
    """The per-wrapper knowledge is data; it can be checked without a cartridge."""
    from planiverse.environments.emulated.game_boy import GENERIC, PROFILES

    assert PROFILES["GameWrapperSuperMarioLand"].select(4) == {"world_level": (2, 2)}
    assert PROFILES["GameWrapperFlipull"].select(0) == {"stage": 1}
    assert PROFILES["GameWrapperPuzznic"].select(127) == {"stage": 127}
    assert PROFILES["GameWrapperTetris"].select(2) == {"timer_div": 32}
    for name, profile in {**PROFILES, "generic": GENERIC}.items():
        assert profile.instances > 0 and profile.hold > 0 and profile.actions, name
        for index in range(profile.instances):
            assert isinstance(profile.select(index), dict)


def test_goal_specs_read_the_wrapper_and_the_fields():
    from planiverse.environments.emulated.game_boy import _holds

    class Wrapper:
        def stage_cleared(self):
            return True

        def game_over(self):
            return False

    fields, start = {"lives_left": 1, "level_progress": 300}, {"lives_left": 2,
                                                                 "level_progress": 100}
    assert _holds({"method": "stage_cleared"}, Wrapper(), fields, start, {}, 0)
    assert not _holds({"method": "game_over"}, Wrapper(), fields, start, {}, 0)
    assert _holds({"attribute": "level_progress", "delta": 200}, Wrapper(), fields, start, {}, 0)
    assert not _holds({"attribute": "level_progress", "delta": 201}, Wrapper(), fields, start,
                      {}, 0)
    assert _holds({"method": "game_over", "attribute": "lives_left"}, Wrapper(), fields, start,
                  {}, 0, terminal=True), "a lost life is a dead end even before game over"
    assert _holds({"survive": 5}, Wrapper(), fields, start, {}, 5)
    assert not _holds({"survive": 5}, Wrapper(), fields, start, {}, 4)
    assert not _holds(None, Wrapper(), fields, start, {}, 0)


# ============================================================================ Stable-Retro

def airstriker_env(**overrides):
    pytest.importorskip("stable_retro", reason="stable-retro is not installed")
    from planiverse.environments.emulated.stable_retro import RetroEnv

    return RetroEnv(**{"goal": {"survive": 5}, **overrides})


@pytest.fixture(scope="module")
def airstriker():
    env = airstriker_env()
    yield env
    env.close()


def test_the_integration_ships_one_save_state(airstriker):
    assert airstriker.states() == ("Level1",)
    with pytest.raises(IndexError, match="ships 1 save state"):
        airstriker.set_index(1)
    with pytest.raises(ValueError, match="not a save state"):
        airstriker.set_instance({"state": "Level2"})
    airstriker.set_index(0)
    state, info = airstriker.reset()
    assert_string_literals(state)
    assert info["state"] == "Level1" and info["variables"] == {"gameover": 9, "lives": 3,
                                                               "score": 0}


def test_actions_are_named_button_combinations(airstriker):
    assert [str(action) for action in airstriker.get_actions()] == \
        ["nop", "LEFT", "RIGHT", "B", "LEFT+B", "RIGHT+B"]
    assert airstriker.__action__("B+LEFT") == airstriker.__action__("LEFT+B"), \
        "a combination is a set of buttons, however it is spelled"
    with pytest.raises(ValueError, match="not a button combination"):
        airstriker.__action__("FIRE")
    assert len(airstriker_env(actions=None).actions) == 6, "the game's default vocabulary"


def test_states_are_told_apart_by_ram(airstriker):
    """Airstriker's variables say nothing about where the ship is, so on their own they fold
    every move into one state; the RAM hash keeps them apart."""
    airstriker.set_index(0)
    state, _ = airstriker.reset()
    successors = airstriker.successors(state)
    assert_successors_contract(successors)
    assert len(successors) == 6 and len({successor for _, successor in successors}) == 6
    assert any(literal.startswith("ram(") for literal in state.literals)
    coarse = airstriker_env(identity="variables")
    coarse.set_index(0)
    coarse_state, _ = coarse.reset()
    assert len({successor for _, successor in coarse.successors(coarse_state)}) == 1
    coarse.close()


def test_losing_a_life_is_a_dead_end():
    env = airstriker_env(goal={"survive": 1000})
    state, _ = env.reset()
    steps = 0
    while not env.is_terminal(state) and steps < 200:
        state = env.__advance__(state, "nop")
        steps += 1
    assert env.is_terminal(state) and state.variables["gameover"] < 9, "hit"
    assert state.variables["lives"] == 3, "before the lives counter has moved"
    assert env.successors(state) == []
    assert not env.is_goal(state)
    env.close()


def test_a_survival_goal_is_reached_by_staying_alive(airstriker):
    airstriker.set_index(0)
    state, _ = airstriker.reset()
    assert "steps(0)" in state.literals and state.progress == 5
    outcome = bounded_search(airstriker, 100, progress=lambda s: s.progress)
    assert len(outcome.plan) == 5 and airstriker.validate(outcome.plan)
    assert airstriker.simulate(outcome.plan)[-1].progress == 0


def test_a_variable_goal_measures_what_is_left():
    env = airstriker_env(goal={"variable": "score", "delta": 10})
    state, _ = env.reset()
    assert state.progress == 10 and "steps(0)" not in state.literals
    env.close()


def test_a_replay_lands_on_the_same_bytes_every_time():
    """Deterministic given the save state and the inputs, in one environment and across
    two, which is what makes a generated instance replayable anywhere."""
    plan = ["LEFT", "B", "RIGHT+B", "nop", "LEFT+B"] * 6
    first, second = airstriker_env(), airstriker_env()
    one, two = first.simulate(plan), second.simulate(plan)
    assert [s.literals for s in one] == [s.literals for s in two]
    assert [s.literals for s in first.simulate(plan)] == [s.literals for s in one]
    first.close()
    second.close()


def test_two_environments_share_the_one_emulator_a_process_gets():
    """Stable-Retro allows one emulator per process; whichever environment is used next
    takes it over, and a state expanded under one still expands under the other."""
    first, second = airstriker_env(), airstriker_env()
    state, _ = first.reset()
    children = first.successors(state)
    second.reset()
    assert first._env is None, "handed over"
    again = first.successors(state)
    assert [c.literals for _, c in again] == [c.literals for _, c in children]
    assert second._env is None
    first.close()
    second.close()


def test_the_same_seed_gives_the_same_instance(airstriker):
    first = airstriker.generate_instance(seed=3)
    state = airstriker.reset()[0]
    second = airstriker.generate_instance(seed=3)
    assert first == second and first["seed"] == 3 and first["state"] == "Level1"
    assert airstriker.reset()[0].literals == state.literals
    assert json.loads(json.dumps(first)) == first
    other = airstriker_env()
    other.set_instance(json.loads(json.dumps(first)))
    assert other.reset()[0].literals == state.literals
    assert other.reset()[1]["generated"] and other.index is None
    other.close()


def test_a_checked_draw_keeps_its_witness(airstriker):
    airstriker.generate_instance(seed=1, warmup=(0, 5), solvable=True, search_limit=200)
    assert airstriker.witness and airstriker.validate(airstriker.witness)
    assert len(airstriker.witness) == 5
    airstriker.set_index(0)
    assert airstriker.witness is None


def test_step_plays_statefully_and_render_gives_the_console_frames(airstriker):
    airstriker.set_index(0)
    airstriker.reset()
    state, reward = airstriker.step("RIGHT")
    assert state.steps == 1 and reward == 1, "one step closer to surviving"
    frames = airstriker.render()
    assert len(frames) == 2 and frames[0].shape[2] == 3
    assert frames[1].max() > 0, "the console's picture, not the blank buffer a load leaves"


def test_the_dead_end_wins_when_it_and_the_goal_hold_on_the_same_step(airstriker, counter):
    airstriker.set_instance({"state": "Level1", "warmup": [], "goal": {"survive": 0},
                             "terminal": {"variable": "score", "at_most": 0}})
    state, _ = airstriker.reset()
    assert airstriker.is_terminal(state) and not airstriker.is_goal(state)
    counter.set_index(0)
    counter.set_instance(dict(counter.instance, goal={"survive": 0},
                              terminal={"attribute": "counter", "at_most": 0}))
    state, _ = counter.reset()
    assert counter.is_terminal(state) and not counter.is_goal(state)


def test_airstriker_registers_a_hit_the_frame_it_lands(airstriker):
    """`gameover` falls from 9 when the ship is hit; `lives` counts it twenty actions later."""
    airstriker.set_index(0)
    assert airstriker.instance["terminal"] == {"done": True, "variable": "gameover",
                                               "drop": True}
    airstriker.set_instance(dict(airstriker.instance, goal={"survive": 100}))
    state, _ = airstriker.reset()
    assert state.variables["gameover"] == 9 and state.variables["lives"] == 3
    trace = airstriker.simulate(["nop"] * 100)
    hit = next(k for k, s in enumerate(trace) if s.variables["gameover"] < 9)
    assert trace[hit].terminal and trace[hit].variables["lives"] == 3, \
        "the dead end lands with the hit, before the lives counter has moved"
    assert trace[hit + 1] == trace[hit], "and the trace stops there"
    assert not any(airstriker.is_goal(s) for s in trace)


def test_retro_goal_specs_read_the_variables():
    pytest.importorskip("stable_retro", reason="stable-retro is not installed")
    from planiverse.environments.emulated.stable_retro import _holds

    variables, start = {"lives": 2, "score": 150}, {"lives": 3, "score": 100}
    assert _holds({"variable": "score", "delta": 50}, variables, start, False, 0)
    assert not _holds({"variable": "score", "delta": 51}, variables, start, False, 0)
    assert _holds({"variable": "score", "at_least": 150}, variables, start, False, 0)
    assert _holds({"variable": "lives", "drop": True}, variables, start, False, 0, terminal=True)
    assert _holds({"done": True}, variables, start, True, 0, terminal=True)
    assert not _holds({"done": True}, variables, start, False, 0, terminal=True)
    assert _holds({"survive": 3}, variables, start, False, 3)
    assert not _holds({}, variables, start, True, 9)
