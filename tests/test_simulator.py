"""Tests for the Simulator facade."""
import pytest

from planiverse.environments.base import Environment, implements_contract
from planiverse.environments.gameboy_py.puzznic import PuzznicGame
from planiverse.simulator.simulator import Simulator

from conftest import assert_state_contract


# --------------------------------------------------------------------------- dispatch

def test_wraps_a_retro_game_directly(puzznic_env):
    sim = Simulator(puzznic_env)
    assert sim.simulator is puzznic_env


def test_wraps_a_real_world_problem_directly():
    pytest.importorskip("numpy", reason="numpy is not installed")
    from planiverse.environments.manufacturing.mfenv import MfgEnv

    env = MfgEnv()
    env.fix_index(0)
    assert Simulator(env).simulator is env


def test_rejects_an_unsupported_object():
    with pytest.raises(AssertionError, match="Unsupported environment type"):
        Simulator(object())


# --------------------------------------------------------------------------- delegation

def test_delegates_the_interface_to_the_environment(puzznic_env):
    sim = Simulator(puzznic_env)
    state, info = sim.reset()
    assert_state_contract(state)
    assert sim.get_actions() == puzznic_env.get_actions()
    assert sim.is_goal(state) == puzznic_env.is_goal(state)
    assert sim.is_terminal(state) == puzznic_env.is_terminal(state)
    assert len(sim.successors(state)) == len(puzznic_env.successors(state))


def test_step_and_simulate_through_the_facade(puzznic_env):
    sim = Simulator(puzznic_env)
    sim.reset()
    state, score = sim.step("left")
    assert state.cursor.pos == (1, 2)
    trace = sim.simulate(["left", "down"])
    assert len(trace) == 3


def test_validate_resets_before_checking(puzznic_env):
    """Simulator.validate resets first, so it does not depend on prior stepping."""
    sim = Simulator(puzznic_env)
    sim.reset()
    sim.step("left")
    assert sim.validate(["left", "right"]) is False


# --------------------------------------------------------------------------- base classes

def test_one_base_class_and_a_structural_check():
    """`RetroGame` and `RealWorldProblem` were merged into `Environment`.

    The old pair recorded provenance, not capability, which is why the facade's dispatch
    was two branches doing the same thing. What a planner needs to know is whether the
    object answers the contract, and `implements_contract` asks exactly that — so an
    environment brought from outside works without inheriting from anything.
    """
    assert isinstance(PuzznicGame(), Environment)
    assert implements_contract(PuzznicGame())

    class Outsider:
        reset = fix_index = successors = is_goal = is_terminal = simulate = lambda *a: None

    assert not isinstance(Outsider(), Environment)
    assert implements_contract(Outsider()), "duck typing is enough for the facade"
