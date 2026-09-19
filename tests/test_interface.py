"""The environment contract, checked uniformly across every environment.

Per-environment behaviour lives in the other test modules; this one only asserts the
things a planner is entitled to assume no matter which environment it is handed.
"""
import pytest

from planiverse.environments import Environment, implements_contract, list_environments
from planiverse.environments.base import REQUIRED_METHODS
from planiverse.environments.registry import STATE_IDENTITIES

from conftest import assert_string_literals, assert_successors_contract


def puzznic():
    from planiverse.environments.games.puzznic import PuzznicGame

    env = PuzznicGame()
    env.set_index(0)
    return env


def flipull():
    from planiverse.environments.games.flipull import FlipullGame

    env = FlipullGame()
    env.set_index(0)
    return env


def lolo():
    from planiverse.environments.games.lolo import LoloGame

    env = LoloGame()
    env.set_index(0)
    return env


def amazing_tater():
    from planiverse.environments.games.amazing_tater import AmazingTaterGame

    env = AmazingTaterGame()
    env.set_index(0)
    return env


def water_network():
    pytest.importorskip("wntr", reason="wntr is not installed")
    from planiverse.environments.water_network.environment import WaterNetworkEnv

    env = WaterNetworkEnv()
    env.set_index(0)
    return env


def power_grid():
    pytest.importorskip("grid2op", reason="grid2op is not installed")
    from planiverse.environments.power_grid.environment import PowerGridEnv

    env = PowerGridEnv()
    env.set_index(4)
    return env


def crop_management():
    pytest.importorskip("pcse", reason="pcse is not installed")
    from planiverse.environments.crop_management.environment import CropEnv

    env = CropEnv()
    env.set_index(10)
    return env



def network_attack():
    pytest.importorskip("nasim", reason="nasim is not installed")
    from planiverse.environments.network_attack.network_attack import EnvNASim

    env = EnvNASim()
    env.set_index(0)
    return env


def flood_transport():
    from planiverse.environments.flood_transport.environment import FloodTransportEnv

    env = FloodTransportEnv()
    env.set_index(0)
    return env


def game_boy():
    pytest.importorskip("pyboy", reason="pyboy is not installed")
    from counter_rom import COUNTER, counter_rom
    from planiverse.environments.emulated.game_boy import GameBoyEnv

    env = GameBoyEnv(counter_rom(), watch={"counter": COUNTER},
                     goal={"memory": COUNTER, "at_least": 3}, actions=("right", "left", "a"),
                     hold=4, settle=2)
    env.set_index(0)
    return env


def artillery():
    from planiverse.environments.artillery.environment import ArtilleryEnv

    env = ArtilleryEnv()
    env.set_index(0)
    return env


def tower_defence():
    from planiverse.environments.tower_defence.environment import TowerDefenceEnv

    env = TowerDefenceEnv()
    env.set_index(0)
    return env


def fluid():
    from planiverse.environments.fluid.environment import FluidEnv

    env = FluidEnv()
    env.set_index(0)
    return env


def billiards():
    pytest.importorskip("pooltool", reason="pooltool is not installed")
    from planiverse.environments.billiards.environment import BilliardsEnv

    env = BilliardsEnv()
    env.set_index(0)
    return env


def lemmings():
    from planiverse.environments.lemmings.environment import LemmingsEnv

    env = LemmingsEnv()
    env.set_index(0)
    return env


def micropolis():
    pytest.importorskip("micropolisengine", reason="the Micropolis engine is not built")
    from planiverse.environments.micropolis.environment import MicropolisEnv

    env = MicropolisEnv()
    env.set_index(0)
    return env


def epidemic():
    pytest.importorskip("covasim", reason="covasim is not installed")
    from planiverse.environments.epidemic.environment import EpidemicEnv

    env = EpidemicEnv()
    env.set_index(0)
    return env


def traffic():
    pytest.importorskip("libsumo", reason="libsumo is not installed")
    from planiverse.environments.traffic.environment import TrafficEnv

    env = TrafficEnv()
    env.set_index(0)
    return env


def airspace():
    pytest.importorskip("bluesky", reason="bluesky is not installed")
    from planiverse.environments.airspace.environment import AirspaceEnv

    env = AirspaceEnv()
    env.set_index(0)
    return env


def reservoir():
    pytest.importorskip("pywr", reason="pywr is not installed")
    from planiverse.environments.reservoir.environment import ReservoirEnv

    env = ReservoirEnv()
    env.set_index(0)
    return env


def factory_env():
    pytest.importorskip("fsim", reason="factory-sim is not built")
    from planiverse.environments.factory.environment import FactoryEnv

    env = FactoryEnv()
    env.set_index(0)
    return env


def slingshot():
    pytest.importorskip("pymunk", reason="pymunk is not installed")
    from planiverse.environments.slingshot.environment import SlingshotEnv

    env = SlingshotEnv()
    env.set_index(0)
    return env


def retro():
    pytest.importorskip("stable_retro", reason="stable-retro is not installed")
    from planiverse.environments.emulated.stable_retro import RetroEnv

    env = RetroEnv(goal={"survive": 5})
    env.set_index(0)
    return env


ENVIRONMENTS = {
    "puzznic": puzznic,
    "flipull": flipull,
    "lolo": lolo,
    "amazing_tater": amazing_tater,
    "water_network": water_network,
    "power_grid": pytest.param(power_grid, marks=pytest.mark.slow),
    "crop_management": crop_management,
    "network_attack": network_attack,
    "flood_transport": flood_transport,
    "slingshot": slingshot,
    "artillery": artillery,
    "tower_defence": tower_defence,
    "fluid": fluid,
    "billiards": billiards,
    "lemmings": lemmings,
    "micropolis": micropolis,
    "factory": factory_env,
    "epidemic": epidemic,
    "traffic": traffic,
    "airspace": pytest.param(airspace, marks=pytest.mark.slow),
    "reservoir": reservoir,
    "game_boy": game_boy,
    "retro": retro,
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
    for method in REQUIRED_METHODS:
        assert callable(getattr(env, method, None)), \
            f"{type(env).__name__} does not implement {method}()"
        assert type(env).provides(method), \
            f"{type(env).__name__} inherits the base's {method}(), which only raises"


@pytest.mark.parametrize("factory", environment_params())
def test_is_a_recognised_environment_type(factory):
    """One base class now, and the contract check is structural as well.

    There used to be two (`RetroGame` and `RealWorldProblem`), and the split described
    where an environment came from rather than what a planner could do with it, so the
    `Simulator` facade that dispatched on them ended up with two isinstance branches doing
    identical work. The facade followed the split into history once every caller took the
    environment directly.
    """
    env = factory()
    assert isinstance(env, Environment)
    assert implements_contract(env), "and it answers the contract structurally too"


def test_an_outside_environment_needs_no_subclassing():
    """`implements_contract` is duck typing: an environment brought from outside the
    library counts as long as it answers the eight methods, which is the point of checking
    structurally instead of by base class. A bare `Environment()` has all eight attributes
    and implements none of them, so it must not count."""

    class Outsider:
        reset = set_index = successors = is_goal = is_terminal = simulate = lambda *a: None
        set_instance = generate_instance = lambda *a, **k: None

    class Searchable:
        """The six search methods without the two that make instances."""
        reset = set_index = successors = is_goal = is_terminal = simulate = lambda *a: None

    assert not isinstance(Outsider(), Environment)
    assert implements_contract(Outsider())
    assert not implements_contract(Searchable()), "a generator is part of the contract"
    assert not implements_contract(Environment()), "stubs do not satisfy the contract"
    assert not implements_contract(object())


def test_every_registered_environment_is_in_the_catalogue():
    """The registry is the catalogue, so it cannot drift from what exists."""
    registered = {spec.name for spec in list_environments()}
    assert {"puzznic", "flipull", "lolo", "amazing_tater",
            "network_attack", "water_network", "power_grid", "crop_management",
            "flood_transport", "slingshot", "artillery", "tower_defence", "fluid", "billiards",
            "lemmings", "micropolis", "factory", "epidemic", "traffic", "airspace", "reservoir",
            "game_boy", "retro"} == registered


def test_a_spec_can_be_loaded_without_importing_the_rest():
    """Listing the catalogue must not import grid2op, WNTR, PCSE and the rest; some of them
    would not be installed."""
    for spec in list_environments():
        assert ":" in spec.factory
        assert spec.deterministic, "every environment here is deterministic"
        assert spec.state_identity in STATE_IDENTITIES
        assert spec.generates, "every bundled environment says what its generator draws"
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
    for name in ("water_network", "power_grid", "crop_management", "game_boy", "retro"):
        spec = get_spec(name)
        if not spec.available():
            continue
        assert spec.load().capabilities() >= full, f"{name} claims the full capability row"


def test_validate_comes_from_the_base_and_still_counts_as_provided():
    """`validate` is the same sentence in every environment, so it is written once in the
    base, but it is a *working* default, unlike `step` and `get_actions` whose defaults
    only explain their own absence.

    So "does the class override it" is the wrong test for whether a capability is offered,
    and `capabilities()` asks whether the method would do something instead.
    """
    from planiverse.environments import Environment

    assert "validate" in Environment.capabilities(), "the default works"
    assert "step" not in Environment.capabilities(), "this default only raises"
    assert "get_actions" not in Environment.capabilities()
    assert not Environment.provides("generate_instance"), "required, and only explained here"
    assert not Environment.provides("set_instance")

    from planiverse.environments.games.flipull import FlipullGame

    assert FlipullGame.validate is Environment.validate, "inherited, not rewritten"
    assert "validate" in FlipullGame.capabilities(), "and still offered"


def test_specs_agree_with_the_environments_they_name():
    """A spec that has drifted from its class is worse than no spec."""
    from planiverse.environments import Environment, list_environments

    for spec in list_environments(available_only=True):
        cls = spec.load()
        assert issubclass(cls, Environment), f"{spec.name} must be an Environment"
        assert spec.docs, f"{spec.name} should point at its documentation"
