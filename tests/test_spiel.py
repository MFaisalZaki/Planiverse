"""Tests for the OpenSpiel interface: simulators as `pyspiel` games, planners on any game.

The Planiverse side runs on the pure-Python games, which need nothing installed; the other
side on OpenSpiel's own single-player games, one deterministic and one with chance nodes.
"""
import pytest

pyspiel = pytest.importorskip("pyspiel")

import planiverse.spiel as spiel  # noqa: E402
from planiverse.environments import list_environments, make  # noqa: E402
from planiverse.planners.width import Budget  # noqa: E402


def replay(game, actions):
    state = game.new_initial_state()
    for action in actions:
        assert action in state.legal_actions()
        state.apply_action(action)
    return state


def test_every_environment_is_registered():
    names = set(pyspiel.registered_names())
    assert {f"planiverse_{spec.name}" for spec in list_environments()} <= names


def test_a_game_is_the_environment_behind_openspiels_api():
    game = pyspiel.load_game("planiverse_lolo", {"index": 0})
    env = make("lolo", index=0)
    state = game.new_initial_state()

    assert game.num_players() == 1
    assert state.current_player() == 0
    assert state.literals == env.reset()[0].literals
    assert len(state.legal_actions()) == len(env.successors(env.reset()[0]))
    # The ids index the static action list, so they mean the same in every game.
    assert [game.actions[a] for a in state.legal_actions()] == [
        action for action, _ in env.successors(env.reset()[0])]


def test_clone_shares_the_simulator_and_leaves_the_original_alone():
    game = pyspiel.load_game("planiverse_puzznic", {"index": 0})
    state = game.new_initial_state()
    child = state.clone()
    child.apply_action(child.legal_actions()[0])
    assert child.literals != state.literals
    assert state.history() == []
    assert child.get_game().env is state.get_game().env


def test_seed_draws_the_same_instance_as_make():
    game = pyspiel.load_game("planiverse_puzznic", {"seed": 7})
    assert game.new_initial_state().literals == make("puzznic", seed=7).reset()[0].literals


def test_attributes_reach_the_simulators_own_state():
    state = pyspiel.load_game("planiverse_lolo", {"index": 0}).new_initial_state()
    assert state.lolo == state.inner.lolo


@pytest.mark.parametrize("tag", ["bfws", "iw", "gbfs", "brfs"])
def test_a_plan_found_through_openspiel_reaches_the_goal(tag):
    game = pyspiel.load_game("planiverse_lolo", {"index": 0})
    actions, result = spiel.solve(game, tag)
    assert result.status == "solved"
    final = replay(game, actions)
    assert final.is_terminal() and final.returns() == [1.0]
    assert final.current_player() == pyspiel.PlayerId.TERMINAL
    assert final.legal_actions() == []


def test_the_benchmark_measure_is_the_default_progress():
    game = pyspiel.load_game("planiverse_lolo", {"index": 0})
    _, guided = spiel.solve(game, "gbfs")
    _, blind = spiel.solve(game, "brfs")
    assert guided.statistics.expansions < blind.statistics.expansions


def test_openspiels_own_games_are_plannable():
    game = pyspiel.load_game("deep_sea")
    actions, result = spiel.solve(game, "iw")
    assert result.status == "solved"
    assert replay(game, actions).returns()[0] > 0


def test_chance_is_resolved_the_same_way_every_time():
    env = spiel.SpielEnvironment(pyspiel.load_game("catch"))
    assert env.reset()[0] == env.reset()[0]
    actions, result = spiel.solve(pyspiel.load_game("catch"), "bfws",
                                  budget=Budget(max_expansions=10_000))
    assert result.status == "solved"
    assert env.validate(actions)


def test_a_game_for_more_than_one_player_is_refused():
    with pytest.raises(ValueError, match="single-player"):
        spiel.SpielEnvironment(pyspiel.load_game("tic_tac_toe"))


def test_a_state_is_observed_through_openspiels_observer():
    """The text OpenSpiel prints is the simulator state's own, and its readings are the
    observation tensor, each named in the observer's `dict`."""
    from open_spiel.python.observation import make_observation

    game = pyspiel.load_game("planiverse_tower_defence", {"index": 0})
    state = game.new_initial_state()
    assert state.observation_string(0) == str(state.inner)
    observation = make_observation(game)
    observation.set_from(state, 0)
    assert {name: float(view[0]) for name, view in observation.dict.items()} == \
        {name: float(value) for name, value in state.readings().items()}
    assert list(state.observation_tensor(0)) == list(observation.tensor)

    board = pyspiel.load_game("planiverse_lolo", {"index": 0})
    assert not board.get_type().provides_observation_tensor, "a board has no readings"
    assert board.new_initial_state().observation_string(0).startswith("#")
