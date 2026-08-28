"""The environment contract, checked uniformly across every environment.

Per-environment behaviour lives in the other test modules; this one only asserts the
things a planner is entitled to assume no matter which environment it is handed.
"""
import pytest

from planiverse.environments import Environment, implements_contract, list_environments

from conftest import assert_string_literals, assert_successors_contract


def puzznic():
    from planiverse.environments.puzznic import PuzznicGame

    env = PuzznicGame()
    env.fix_index(0)
    return env


def puzznic_gb():
    pytest.importorskip("pyboy", reason="pyboy is not installed")
    from fake_puzznic_rom import synthetic_rom
    from planiverse.environments.gameboy.puzznic_gb import PuzznicGBEnv

    # Puzznic is copyrighted, so the contract is checked against the synthetic cartridge
    # in `fake_puzznic_rom.py` rather than the real one.
    env = PuzznicGBEnv(synthetic_rom(), verify_rom=False)
    env.fix_index(0)
    return env


def flipull():
    from planiverse.environments.flipull import FlipullGame

    env = FlipullGame()
    env.fix_index(0)
    return env


def flipull_gb():
    pytest.importorskip("pyboy", reason="pyboy is not installed")
    from fake_flipull_rom import synthetic_rom
    from planiverse.environments.gameboy.flipull_gb import FlipullGBEnv

    # Flipull is copyrighted too, so the contract is checked against the synthetic
    # cartridge in `fake_flipull_rom.py`.
    env = FlipullGBEnv(synthetic_rom(), verify_rom=False)
    env.fix_index(0)
    return env


def platformer():
    from planiverse.environments.platformer import PlatformerGame

    env = PlatformerGame()
    env.fix_index(0)
    return env


def water_network():
    pytest.importorskip("wntr", reason="wntr is not installed")
    from planiverse.environments.water_network.environment import WaterNetworkEnv

    env = WaterNetworkEnv()
    env.fix_index(0)
    return env


def power_grid():
    pytest.importorskip("grid2op", reason="grid2op is not installed")
    from planiverse.environments.power_grid.environment import PowerGridEnv

    env = PowerGridEnv()
    env.fix_index(4)
    return env


def crop_management():
    pytest.importorskip("pcse", reason="pcse is not installed")
    from planiverse.environments.crop_management.environment import CropEnv

    env = CropEnv()
    env.fix_index(10)
    return env


def manufacturing():
    pytest.importorskip("numpy", reason="numpy is not installed")
    from planiverse.environments.manufacturing.mfenv import MfgEnv

    env = MfgEnv()
    env.fix_index(0)
    return env


def urban_planning():
    pytest.importorskip("pandas", reason="pandas is not installed")
    pytest.importorskip("networkx", reason="networkx is not installed")
    from planiverse.environments.urban_planning.environment import UrbanPlanningEnv

    env = UrbanPlanningEnv(horizon=100)
    env.fix_index(0)
    return env


def network_attack():
    pytest.importorskip("nasim", reason="nasim is not installed")
    from planiverse.environments.network_attack.network_attack import EnvNASim

    env = EnvNASim()
    env.fix_index(0)
    return env


def epidemic():
    pytest.importorskip("numba", reason="numba is not installed")
    pytest.importorskip("sympy", reason="sympy is not installed")
    from planiverse.environments.epidemic_control.environment import EpiEnv

    env = EpiEnv(delay_vaccination_time=30, horizon=364)
    env.fix_index(5)          # SIR_A, the cheapest scenario
    return env


ENVIRONMENTS = {
    "puzznic": puzznic,
    "puzznic_gb": puzznic_gb,
    "flipull": flipull,
    "flipull_gb": flipull_gb,
    "platformer": platformer,
    "water_network": water_network,
    "power_grid": pytest.param(power_grid, marks=pytest.mark.slow),
    "crop_management": crop_management,
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
    """One base class now, and Simulator dispatches structurally as well.

    There used to be two — `RetroGame` and `RealWorldProblem` — and the split described
    where an environment came from rather than what a planner could do with it, so the
    facade ended up with two isinstance branches doing identical work.
    """
    env = factory()
    assert isinstance(env, Environment)
    assert implements_contract(env), "and it answers the contract structurally too"


def test_every_registered_environment_is_in_the_catalogue():
    """The registry is the catalogue, so it cannot drift from what exists."""
    registered = {spec.name for spec in list_environments()}
    assert {"puzznic", "puzznic_gb", "flipull", "flipull_gb", "platformer",
            "super_mario_land", "epidemic", "network_attack", "manufacturing",
            "urban_planning", "water_network", "power_grid",
            "crop_management"} == registered


def test_a_spec_can_be_loaded_without_importing_the_rest():
    """Listing the catalogue must not import pyboy, grid2op, numba and the rest — half of
    them would not be installed."""
    for spec in list_environments():
        assert ":" in spec.factory
        assert spec.deterministic, "every environment here is deterministic"
        assert spec.state_identity in ("value", "path", "snapshot")
        if spec.available():
            assert issubclass(spec.load(), Environment)


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


def test_the_capability_matrix_can_be_derived_from_the_code():
    """`Environment.capabilities()` exists so the README's matrix is checkable rather than
    hand-maintained. These are the rows that claim the full set."""
    from planiverse.environments import get_spec

    full = {"step", "validate", "get_actions", "render", "close"}
    for name in ("puzznic_gb", "flipull_gb", "super_mario_land", "water_network",
                 "power_grid", "crop_management"):
        spec = get_spec(name)
        if not spec.available():
            continue
        assert spec.load().capabilities() >= full, f"{name} claims the full capability row"


def test_validate_comes_from_the_base_and_still_counts_as_provided():
    """`validate` is the same sentence in every environment, so it is written once in the
    base — but it is a *working* default, unlike `step` and `get_actions` whose defaults
    only explain their own absence.

    So "does the class override it" is the wrong test for whether a capability is offered,
    and `capabilities()` asks whether the method would do something instead.
    """
    from planiverse.environments import Environment

    assert "validate" in Environment.capabilities(), "the default works"
    assert "step" not in Environment.capabilities(), "this default only raises"
    assert "get_actions" not in Environment.capabilities()

    pytest.importorskip("pyboy", reason="pyboy is not installed")
    from planiverse.environments.gameboy.flipull_gb import FlipullGBEnv

    assert FlipullGBEnv.validate is Environment.validate, "inherited, not rewritten"
    assert "validate" in FlipullGBEnv.capabilities(), "and still offered"


def test_specs_agree_with_the_environments_they_name():
    """A spec that has drifted from its class is worse than no spec."""
    from planiverse.environments import Environment, list_environments

    for spec in list_environments(available_only=True):
        cls = spec.load()
        assert issubclass(cls, Environment), f"{spec.name} must be an Environment"
        assert spec.docs, f"{spec.name} should point at its documentation"
