"""Shared fixtures and helpers for the Planiverse test suite.

Environments differ in what they need to run: Puzznic needs nothing, NASim needs the
NetworkAttackSimulator fork, and the three simulator-backed environments need WNTR, grid2op
and PCSE. Tests for an environment whose requirements are missing skip rather than fail, so
the suite is runnable from a partial install.
"""
import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def a_cartridge_for_the_game_boy():
    """Point the Game Boy environment at the synthetic cartridge, so `make("game_boy")` and
    the uniform contract and generator tests can build it like any other environment.

    No commercial ROM is involved: `counter_rom.py` assembles an original program. A
    cartridge the user already named through the variable is left alone.
    """
    try:
        import pyboy  # noqa: F401
    except ImportError:
        yield
        return
    from counter_rom import counter_rom
    from planiverse.environments.games.emulated.game_boy import ROM_VARIABLE

    already = os.environ.get(ROM_VARIABLE)
    if not already:
        os.environ[ROM_VARIABLE] = counter_rom()
    yield
    if not already:
        os.environ.pop(ROM_VARIABLE, None)


def requires(module_name):
    """Skip the test module unless `module_name` imports."""
    return pytest.importorskip(module_name, reason=f"{module_name} is not installed")


def assert_state_contract(state):
    """Every Planiverse state exposes `literals` as a frozenset.

    Planners key their visited set on it, so it has to be hashable and set-like. The
    element type is not part of the shared contract; the native environments spell
    literals as strings, which `assert_string_literals` checks separately.
    """
    assert hasattr(state, "literals"), f"{type(state).__name__} has no literals"
    assert isinstance(state.literals, frozenset), \
        f"{type(state).__name__}.literals is {type(state.literals).__name__}, expected frozenset"
    hash(state.literals)


def assert_string_literals(state):
    """Native Planiverse environments encode their literals as strings."""
    assert_state_contract(state)
    assert all(isinstance(lit, str) for lit in state.literals), \
        f"{type(state).__name__}.literals must contain only strings"


def assert_successors_contract(successors):
    """successors() returns a list of (action, next_state) pairs."""
    assert isinstance(successors, list)
    for item in successors:
        assert isinstance(item, tuple) and len(item) == 2, \
            f"successors must yield (action, state) pairs, got {item!r}"
        _, next_state = item
        assert_state_contract(next_state)


@pytest.fixture
def puzznic_env():
    from planiverse.environments.games.puzznic import PuzznicGame

    env = PuzznicGame()
    env.set_index(0)
    return env
