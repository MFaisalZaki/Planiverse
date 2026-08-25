"""Shared fixtures and helpers for the Planiverse test suite.

Environments differ in what they need to run: Puzznic needs nothing, the epidemic
environment needs numba/sympy, NASim needs the NetworkAttackSimulator fork, and the two
Game Boy environments need a ROM the user supplies. Tests for an environment whose requirements
are missing skip rather than fail, so the suite is runnable from a partial install.
"""
import os

import pytest


def requires(module_name):
    """Skip the test module unless `module_name` imports."""
    return pytest.importorskip(module_name, reason=f"{module_name} is not installed")


def sml_rom_path():
    """Path to a Super Mario Land ROM, or None.

    The ROM is copyrighted and cannot ship with the repo, so it is opt-in via the
    PLANIVERSE_SML_ROM environment variable.
    """
    rom = os.environ.get("PLANIVERSE_SML_ROM")
    if rom and os.path.isfile(rom):
        return rom
    return None


def puzznic_rom_path():
    """Path to a Puzznic (J) Game Boy ROM, or None.

    Same deal as the Super Mario Land ROM: copyrighted, so it is opt-in via the
    PLANIVERSE_PUZZNIC_ROM environment variable. The tests that do not need it run
    against a synthetic cartridge built by `fake_puzznic_rom.py`.
    """
    rom = os.environ.get("PLANIVERSE_PUZZNIC_ROM")
    if rom and os.path.isfile(rom):
        return rom
    return None


def assert_state_contract(state):
    """Every Planiverse state exposes `literals` as a frozenset.

    Planners key their visited set on it, so it has to be hashable and set-like. Native
    environments spell literals as strings; the PDDLGym wrapper passes pddlgym Literal
    objects straight through, so the element type is not part of the shared contract.
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
    from planiverse.problems.retro_games.puzznic import PuzznicGame

    env = PuzznicGame()
    env.fix_index(0)
    return env


@pytest.fixture
def mfg_env():
    pytest.importorskip("numpy", reason="numpy is not installed")
    from planiverse.problems.real_world_problems.manufacturing_environment.mfenv import MfgEnv

    env = MfgEnv()
    env.fix_index(0)
    return env
