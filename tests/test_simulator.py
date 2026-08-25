"""Tests for the Simulator facade and the PDDLGym wrapper."""
import pytest

from planiverse.problems.real_world_problems.base import RealWorldProblem
from planiverse.problems.retro_games.base import RetroGame
from planiverse.problems.retro_games.puzznic import PuzznicGame
from planiverse.simulator.simulator import Simulator
from planiverse.simulator.wrappers.base import SimulatorBase

from conftest import assert_state_contract


# --------------------------------------------------------------------------- dispatch

def test_wraps_a_retro_game_directly(puzznic_env):
    sim = Simulator(puzznic_env)
    assert sim.simulator is puzznic_env


def test_wraps_a_real_world_problem_directly():
    pytest.importorskip("numpy", reason="numpy is not installed")
    from planiverse.problems.real_world_problems.manufacturing_environment.mfenv import MfgEnv

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

def test_simulator_base_is_an_unimplemented_interface():
    base = SimulatorBase("name", None)
    assert base.name == "name"
    for call in (lambda: base.reset(),
                 lambda: base.step("a"),
                 lambda: base.successors(None),
                 lambda: base.is_goal(None),
                 lambda: base.is_terminal(None),
                 lambda: base.simulate([]),
                 lambda: base.validate([])):
        with pytest.raises(NotImplementedError):
            call()


def test_marker_base_classes():
    assert RetroGame("puzznic", 1989).name == "puzznic"
    assert RealWorldProblem("epidemic").state is None
    assert isinstance(PuzznicGame(), RetroGame)


# --------------------------------------------------------------------------- pddlgym
# The skip lives in the fixture, not at module scope: pddlgym ships in the `pddl` extra,
# which needs Python <=3.12, and the facade tests above run happily without it.


@pytest.fixture
def blocks_sim():
    pddlgym = pytest.importorskip(
        "pddlgym", reason="pddlgym is not installed (the `pddl` extra needs Python <=3.12)")
    return Simulator(pddlgym.make("PDDLEnvBlocks-v0"))


def test_pddlgym_env_is_wrapped(blocks_sim):
    from planiverse.simulator.wrappers.pddlgymenv import PDDLGymEnv

    assert isinstance(blocks_sim.simulator, PDDLGymEnv)
    assert blocks_sim.simulator.name == "blocks"


def test_pddlgym_reset_and_successors(blocks_sim):
    state, info = blocks_sim.reset()
    assert_state_contract(state)
    successors = blocks_sim.successors(state)
    assert len(successors) > 0
    for action, next_state in successors:
        assert_state_contract(next_state)


def test_pddlgym_reset_is_repeatable(blocks_sim):
    """PDDLGym draws a random problem on every reset unless the index is pinned, which
    made reset() non-repeatable and let a plan be validated against another problem."""
    goals = [blocks_sim.reset()[0].goal for _ in range(4)]
    assert all(goal == goals[0] for goal in goals)
    states = [blocks_sim.reset()[0].literals for _ in range(4)]
    assert all(literals == states[0] for literals in states)


def test_pddlgym_fix_index_selects_a_problem(blocks_sim):
    """The wrapper follows the same fix_index convention as every other environment."""
    wrapper = blocks_sim.simulator
    wrapper.fix_index(0)
    first, _ = blocks_sim.reset()
    wrapper.fix_index(2)
    third, _ = blocks_sim.reset()
    assert first.goal != third.goal
    # And it is still repeatable once pinned.
    assert blocks_sim.reset()[0].goal == third.goal


def test_pddlgym_fix_index_rejects_unknown_problem(blocks_sim):
    with pytest.raises(AssertionError, match="not found"):
        blocks_sim.simulator.fix_index(999)


def test_pddlgym_successors_are_sorted(blocks_sim):
    """Expansion order is deterministic: ground actions are sorted by their string."""
    state, _ = blocks_sim.reset()
    actions = [str(a) for a, _ in blocks_sim.successors(state)]
    assert actions == sorted(actions)


def test_pddlgym_initial_state_is_not_a_goal(blocks_sim):
    state, _ = blocks_sim.reset()
    assert not blocks_sim.is_goal(state)


def test_pddlgym_is_goal_reads_the_state_goal(blocks_sim):
    """is_goal used to test a goal cached at construction. PDDLGym cycles problems, so
    after a reset that goal belonged to a different problem -- sometimes over objects
    that no longer exist -- and could never be satisfied."""
    state, _ = blocks_sim.reset()
    goal_state = build_goal_tower(blocks_sim, state)
    assert all(literal in goal_state.literals for literal in state.goal.literals)
    assert blocks_sim.is_goal(goal_state)


def test_pddlgym_simulate_returns_state_trace(blocks_sim):
    state, _ = blocks_sim.reset()
    plan = [action for action, _ in blocks_sim.successors(state)][:2]
    trace = blocks_sim.simulate(plan)
    assert len(trace) == len(plan) + 1


def test_pddlgym_validate_accepts_a_real_plan(blocks_sim):
    """validate used to return is_terminal(last state), which is hard-coded False, so it
    rejected every plan including correct ones."""
    plan = find_plan(blocks_sim, max_expansions=2000)
    assert plan is not None, "BFS should solve the pinned Blocks problem"
    assert blocks_sim.validate(plan) is True
    assert blocks_sim.is_goal(blocks_sim.simulate(plan)[-1])


def test_pddlgym_validate_rejects_a_bad_plan(blocks_sim):
    state, _ = blocks_sim.reset()
    assert blocks_sim.validate([]) is False
    assert blocks_sim.validate([blocks_sim.successors(state)[0][0]]) is False


def test_pddlgym_no_terminal_states(blocks_sim):
    state, _ = blocks_sim.reset()
    assert not blocks_sim.is_terminal(state)


def build_goal_tower(sim, state):
    """Stack the pinned Blocks problem into its goal tower by hand."""
    plan = ["pickup(b:block)", "stack(b:block,a:block)",
            "pickup(c:block)", "stack(c:block,b:block)",
            "pickup(d:block)", "stack(d:block,c:block)"]
    for name in plan:
        state = next(ns for a, ns in sim.successors(state) if str(a) == name)
    return state


def find_plan(sim, max_expansions):
    """Breadth-first search for a goal, returning the action list or None."""
    state, _ = sim.reset()
    frontier, visited = [(state, [])], {state.literals}
    for _ in range(max_expansions):
        if not frontier:
            return None
        current, plan = frontier.pop(0)
        if sim.is_goal(current):
            return plan
        for action, successor in sim.successors(current):
            if successor.literals in visited:
                continue
            visited.add(successor.literals)
            frontier.append((successor, plan + [action]))
    return None


def find_plan(sim, max_expansions):
    """Breadth-first search for a goal, returning the action list or None."""
    state, _ = sim.reset()
    frontier, visited = [(state, [])], {state.literals}
    for _ in range(max_expansions):
        if not frontier:
            return None
        current, plan = frontier.pop(0)
        if sim.is_goal(current):
            return plan
        for action, successor in sim.successors(current):
            if successor.literals in visited:
                continue
            visited.add(successor.literals)
            frontier.append((successor, plan + [action]))
    return None
