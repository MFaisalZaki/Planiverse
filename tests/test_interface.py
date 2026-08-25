"""The environment contract, checked uniformly across every environment.

Per-environment behaviour lives in the other test modules; this one only asserts the
things a planner is entitled to assume no matter which environment it is handed.
"""
import pytest

from planiverse.problems.real_world_problems.base import RealWorldProblem
from planiverse.problems.retro_games.base import RetroGame

from conftest import assert_string_literals, assert_successors_contract


def puzznic():
    from planiverse.problems.retro_games.puzznic import PuzznicGame

    env = PuzznicGame()
    env.fix_index(0)
    return env


def puzznic_gb():
    pytest.importorskip("pyboy", reason="pyboy is not installed")
    from fake_puzznic_rom import synthetic_rom
    from planiverse.problems.retro_games.puzznic_gb import PuzznicGBEnv

    # Puzznic is copyrighted, so the contract is checked against the synthetic cartridge
    # in `fake_puzznic_rom.py` rather than the real one.
    env = PuzznicGBEnv(synthetic_rom(), verify_rom=False)
    env.fix_index(0)
    return env


def manufacturing():
    pytest.importorskip("numpy", reason="numpy is not installed")
    from planiverse.problems.real_world_problems.manufacturing_environment.mfenv import MfgEnv

    env = MfgEnv()
    env.fix_index(0)
    return env


def urban_planning():
    pytest.importorskip("pandas", reason="pandas is not installed")
    pytest.importorskip("networkx", reason="networkx is not installed")
    from planiverse.problems.real_world_problems.urban_planning.environment import UrbanPlanningEnv

    env = UrbanPlanningEnv(horizon=100)
    env.fix_index(0)
    return env


def network_attack():
    pytest.importorskip("nasim", reason="nasim is not installed")
    from planiverse.problems.real_world_problems.cyber_security_network_attack.network_attack import EnvNASim

    env = EnvNASim()
    env.fix_index(0)
    return env


def epidemic():
    pytest.importorskip("numba", reason="numba is not installed")
    pytest.importorskip("sympy", reason="sympy is not installed")
    from planiverse.problems.real_world_problems.epidemic_control.environment import EpiEnv

    env = EpiEnv(delay_vaccination_time=30, horizon=364)
    env.fix_index(5)          # SIR_A, the cheapest scenario
    return env


ENVIRONMENTS = {
    "puzznic": puzznic,
    "puzznic_gb": puzznic_gb,
    "manufacturing": manufacturing,
    "urban_planning": urban_planning,
    "network_attack": network_attack,
    "epidemic": pytest.param(epidemic, marks=pytest.mark.slow),
}


def environment_params():
    return [
        pytest.param(factory, id=name) if not hasattr(factory, "marks")
        else pytest.param(factory.values[0], id=name, marks=factory.marks)
        for name, factory in ENVIRONMENTS.items()
    ]


@pytest.mark.parametrize("factory", environment_params())
def test_implements_the_core_interface(factory):
    env = factory()
    for method in ("reset", "fix_index", "successors", "is_goal", "is_terminal", "simulate"):
        assert callable(getattr(env, method, None)), \
            f"{type(env).__name__} does not implement {method}()"


@pytest.mark.parametrize("factory", environment_params())
def test_is_a_recognised_environment_type(factory):
    """Simulator dispatches on these base classes."""
    env = factory()
    assert isinstance(env, (RetroGame, RealWorldProblem))


@pytest.mark.parametrize("factory", environment_params())
def test_reset_returns_a_state_and_info(factory):
    env = factory()
    result = env.reset()
    assert isinstance(result, tuple) and len(result) == 2
    state, info = result
    assert_string_literals(state)
    assert isinstance(info, dict)


@pytest.mark.parametrize("factory", environment_params())
def test_reset_is_repeatable(factory):
    """Two resets of the same instance give the same state: expansion must not depend on
    how many times the environment has been reset."""
    env = factory()
    first, _ = env.reset()
    second, _ = env.reset()
    assert first.literals == second.literals


@pytest.mark.parametrize("factory", environment_params())
def test_successors_returns_action_state_pairs(factory):
    env = factory()
    state, _ = env.reset()
    successors = env.successors(state)
    assert len(successors) > 0, "the initial state must have successors"
    assert_successors_contract(successors)


@pytest.mark.parametrize("factory", environment_params())
def test_successors_exclude_self_loops(factory):
    """An action that leaves the state unchanged must not be offered."""
    env = factory()
    state, _ = env.reset()
    for action, successor in env.successors(state):
        assert successor.literals != state.literals, \
            f"{type(env).__name__} offered {action} which changes nothing"


@pytest.mark.parametrize("factory", environment_params())
def test_successors_are_deterministic(factory):
    """Expanding the same state twice gives the same successors, or search is unsound."""
    env = factory()
    state, _ = env.reset()
    first = [s.literals for _, s in env.successors(state)]
    second = [s.literals for _, s in env.successors(state)]
    assert first == second


@pytest.mark.parametrize("factory", environment_params())
def test_successors_do_not_mutate_the_parent(factory):
    env = factory()
    state, _ = env.reset()
    before = state.literals
    env.successors(state)
    assert state.literals == before


@pytest.mark.parametrize("factory", environment_params())
def test_goal_and_terminal_return_booleans(factory):
    env = factory()
    state, _ = env.reset()
    assert isinstance(bool(env.is_goal(state)), bool)
    assert isinstance(bool(env.is_terminal(state)), bool)


@pytest.mark.parametrize("factory", environment_params())
def test_initial_state_is_not_a_goal(factory):
    env = factory()
    state, _ = env.reset()
    assert not env.is_goal(state), "a fresh instance should not start solved"


@pytest.mark.parametrize("factory", environment_params())
def test_simulate_replays_a_plan_into_a_state_trace(factory):
    env = factory()
    state, _ = env.reset()
    plan = [action for action, _ in env.successors(state)][:2]
    trace = env.simulate(plan)
    assert len(trace) == len(plan) + 1
    for produced in trace:
        assert_string_literals(produced)


@pytest.mark.parametrize("factory", environment_params())
def test_simulate_starts_from_the_initial_state(factory):
    env = factory()
    state, _ = env.reset()
    trace = env.simulate([])
    assert trace[0].literals == state.literals


@pytest.mark.parametrize("factory", environment_params())
def test_simulate_agrees_with_successors(factory):
    """Replaying an action reproduces the state successors handed out for it."""
    env = factory()
    state, _ = env.reset()
    action, expected = env.successors(state)[0]
    trace = env.simulate([action])
    assert trace[-1].literals == expected.literals
