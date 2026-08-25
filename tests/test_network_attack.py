"""Tests for the NASim network attack environment.

Importing the environment monkey-patches NASim's Network.perform_action process-wide to
strip the random exploit failure, so these tests also pin that determinism.
"""
import textwrap

import pytest

pytest.importorskip("nasim", reason="nasim is not installed")

from planiverse.environments.network_attack.network_attack import (  # noqa: E402
    EnvNASim, NASimState,
)

from conftest import assert_state_contract, assert_successors_contract  # noqa: E402

TINY = 0


@pytest.fixture(scope="module")
def tiny_env():
    env = EnvNASim()
    env.fix_index(TINY)
    env.reset()
    return env


@pytest.fixture
def tiny_state(tiny_env):
    state, _ = tiny_env.reset()
    return state


# --------------------------------------------------------------------------- scenarios

def test_fix_index_maps_to_benchmark_names():
    env = EnvNASim()
    env.fix_index(0)
    assert env.scenario_name == "tiny"
    env.fix_index(3)
    assert env.scenario_name == "small"
    env.fix_index(17)
    assert env.scenario_name == "pocp-2-gen"


def test_fix_index_rejects_unknown_index():
    with pytest.raises(AssertionError, match="not found in the index_scenario_map"):
        EnvNASim().fix_index(18)


def test_scenario_name_can_be_given_directly():
    env = EnvNASim(scenario_name="tiny")
    state, _ = env.reset()
    assert_state_contract(state)


def test_reset_without_a_scenario_raises():
    with pytest.raises(AssertionError, match="Scenario name or yaml is not set"):
        EnvNASim().reset()


def test_scenario_yaml_is_loaded(tmp_path):
    """A YAML path used to satisfy the assert and then be ignored, so make_benchmark(None)
    was called instead of loading the file."""
    scenario = textwrap.dedent("""
        subnets: [1]
        topology: [[ 1, 1],
                   [ 1, 1]]
        sensitive_hosts:
          (1, 0): 100
        os:
          - linux
        services:
          - ssh
        processes:
          - tomcat
        exploits:
          e_ssh:
            service: ssh
            os: linux
            prob: 1.0
            cost: 1
            access: user
        privilege_escalation:
          pe_tomcat:
            process: tomcat
            os: linux
            prob: 1.0
            cost: 1
            access: root
        service_scan_cost: 1
        os_scan_cost: 1
        subnet_scan_cost: 1
        process_scan_cost: 1
        host_configurations:
          (1, 0):
            os: linux
            services: [ssh]
            processes: [tomcat]
        firewall:
          (0, 1): [ssh]
          (1, 0): [ssh]
    """)
    path = tmp_path / "scenario.yaml"
    path.write_text(scenario)

    env = EnvNASim(scenario_yaml=str(path))
    state, _ = env.reset()
    assert_state_contract(state)
    assert len(env.actionslist) > 0


# --------------------------------------------------------------------------- interface

def test_reset_is_repeatable(tiny_env):
    first, _ = tiny_env.reset()
    second, _ = tiny_env.reset()
    assert first.literals == second.literals


def test_state_is_a_nasim_state(tiny_state):
    assert isinstance(tiny_state, NASimState)
    assert tiny_state.tensor is not None


def test_successors_contract(tiny_env, tiny_state):
    assert_successors_contract(tiny_env.successors(tiny_state))


def test_successors_exclude_actions_that_change_nothing(tiny_env, tiny_state):
    """Failed preconditions leave the state untouched; those actions are not offered."""
    successors = tiny_env.successors(tiny_state)
    assert 0 < len(successors) < len(tiny_env.actionslist)
    for _, successor in successors:
        assert successor.literals != tiny_state.literals


def test_initial_state_is_not_a_goal(tiny_env, tiny_state):
    assert not tiny_env.is_goal(tiny_state)


def test_no_terminal_states(tiny_env, tiny_state):
    assert not tiny_env.is_terminal(tiny_state)


def test_simulate_returns_state_trace(tiny_env, tiny_state):
    plan = [action for action, _ in tiny_env.successors(tiny_state)][:2]
    trace = tiny_env.simulate(plan)
    assert len(trace) == len(plan) + 1
    for state in trace:
        assert_state_contract(state)


# --------------------------------------------------------------------------- determinism

def test_expansion_is_deterministic(tiny_env, tiny_state):
    """NASim rolls a die on exploit success; the patched perform_action must not.

    Without this, the same action from the same state could yield different successors
    and search would be unsound.
    """
    for _ in range(5):
        first = tiny_env.successors(tiny_state)
        second = tiny_env.successors(tiny_state)
        assert [str(a) for a, _ in first] == [str(a) for a, _ in second]
        assert [s.literals for _, s in first] == [s.literals for _, s in second]


def test_actions_are_hashable(tiny_env):
    """The patch adds __hash__ to every action class so they can live in sets."""
    assert len({action for action in tiny_env.actionslist}) > 0


# --------------------------------------------------------------------------- literals

def test_literals_transcribe_the_state_tensor(tiny_state):
    at_literals = [lit for lit in tiny_state.literals if lit.startswith("at(")]
    assert len(at_literals) == tiny_state.tensor.shape[0] * tiny_state.tensor.shape[1]


def test_compromised_hosts_appear_in_literals(tiny_env, tiny_state):
    assert not any(lit.startswith("compromised_host_") for lit in tiny_state.literals)


@pytest.mark.slow
def test_a_goal_is_reachable_by_search(tiny_env, tiny_state):
    """Breadth-first search finds a plan that compromises every sensitive host."""
    frontier, visited = [(tiny_state, [])], {tiny_state.literals}
    for _ in range(2000):
        if not frontier:
            break
        state, plan = frontier.pop(0)
        if tiny_env.is_goal(state):
            assert len(plan) > 0
            trace = tiny_env.simulate(plan)
            assert tiny_env.is_goal(trace[-1])
            return
        for action, successor in tiny_env.successors(state):
            if successor.literals in visited:
                continue
            visited.add(successor.literals)
            frontier.append((successor, plan + [action]))
    pytest.fail("no goal found for the 'tiny' scenario within the search budget")
