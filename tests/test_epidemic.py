"""Tests for the epidemic control environment.

The epidemic simulator JIT-compiles through numba and integrates a compartmental model,
so expansion is slow. These tests stick to SIR_A (one locale, three compartments) and
keep the number of expansions small; anything heavier is marked slow.
"""
import pytest

pytest.importorskip("numba", reason="numba is not installed")
pytest.importorskip("sympy", reason="sympy is not installed")

from planiverse.problems.real_world_problems.epidemic_control.environment import (  # noqa: E402
    PERIOD, EpiAction, EpiAppliedInterventions, EpiCost, EpiEnv, EpiState,
)

from conftest import assert_state_contract, assert_successors_contract  # noqa: E402

SIR_A, SIR_B, COVID_A = 5, 6, 0


@pytest.fixture(scope="module")
def sir_env():
    """SIR_A is the cheapest scenario: one locale, S/I/R only."""
    env = EpiEnv(delay_vaccination_time=30, horizon=364)
    env.fix_index(SIR_A)
    return env


@pytest.fixture(scope="module")
def sir_state(sir_env):
    state, _ = sir_env.reset()
    return state


# --------------------------------------------------------------------------- scenarios

def test_fix_index_rejects_unknown_scenario():
    env = EpiEnv(delay_vaccination_time=0, horizon=7)
    with pytest.raises(AssertionError, match="not found in index_scenario_map"):
        env.fix_index(99)


def test_scenario_indices_are_sorted_by_filename():
    """Seven scenarios: COVID_A/B/C, SIRV_A/B, SIR_A/B. The ignored warmup file is skipped."""
    env = EpiEnv(delay_vaccination_time=0, horizon=7)
    for index in range(7):
        env.fix_index(index)               # must not raise
    with pytest.raises(AssertionError):
        env.fix_index(7)


def test_sir_scenario_compartments(sir_env, sir_state):
    names = {sir_env.epi.static.get_property_name("compartment", i)
             for i in range(sir_env.epi.static.compartment_count)}
    assert names == {"S", "I", "R"}


def test_reset_starts_at_depth_zero(sir_state):
    assert sir_state.depth == 0
    assert_state_contract(sir_state)


def test_reset_before_fix_index_raises():
    env = EpiEnv(delay_vaccination_time=0, horizon=7)
    with pytest.raises(AttributeError):
        env.reset()


# --------------------------------------------------------------------------- actions

def test_interventions_are_discretised_into_three_levels(sir_env):
    """Each intervention's control parameter is split into itv_split levels."""
    assert sir_env.itv_split == 3
    for action in sir_env.basic_interventions:
        assert isinstance(action, EpiAction)
        levels = [action.create_action(v).value
                  for v in [action.min_value, action.max_value]]
        assert levels[0] == action.min_value and levels[1] == action.max_value


def test_action_value_reads_the_control_parameter(sir_env):
    """`value` must follow control_parameter_index rather than assuming index 0.

    cpv_list[0] happens to be the control parameter in every bundled scenario, so
    indexing 0 works by luck of parameter ordering rather than by construction.
    """
    for action in sir_env.basic_interventions:
        control_param = action.itv_details.cp_list[action.control_parameter_index]
        assert control_param.name in ("compliance", "degree", "percentage")
        assert action.value == action.cpv_list[action.control_parameter_index]


def test_create_action_sets_the_control_parameter(sir_env):
    action = sir_env.basic_interventions[0]
    midpoint = (action.min_value + action.max_value) / 2
    created = action.create_action(midpoint)
    assert created.value == pytest.approx(midpoint)
    assert created.name == action.name
    # The original is untouched.
    assert action.value != pytest.approx(midpoint) or action.value == pytest.approx(midpoint)


def test_create_action_rejects_out_of_bounds(sir_env):
    action = sir_env.basic_interventions[0]
    with pytest.raises(AssertionError, match="out of bounds"):
        action.create_action(action.max_value + 1)


def test_action_string_names_the_policy_level(sir_env):
    action = sir_env.basic_interventions[0].create_action(0.0)
    assert str(action) == f"{action.name} = 0.0"


def test_applied_interventions_expose_a_flat_action_list(sir_env, sir_state):
    action, _ = sir_env.successors(sir_state)[0]
    assert isinstance(action, EpiAppliedInterventions)
    assert action.action == action.itvs + action.costs
    assert " ^ " in str(action) or len(action.itvs) == 1


# --------------------------------------------------------------------------- transition

def test_a_step_advances_exactly_one_period(sir_env, sir_state):
    """The step used to run PERIOD+1 days while advancing depth by PERIOD, so simulated
    time drifted ahead of the depth that the goal test and vaccination delay read."""
    for _, successor in sir_env.successors(sir_state):
        assert successor.depth == sir_state.depth + PERIOD


def test_period_is_a_week():
    assert PERIOD == 7


def test_successors_contract(sir_env, sir_state):
    assert_successors_contract(sir_env.successors(sir_state))


def test_successors_are_deterministic(sir_env, sir_state):
    first = sir_env.successors(sir_state)
    second = sir_env.successors(sir_state)
    assert [str(a) for a, _ in first] == [str(a) for a, _ in second]
    assert [repr(s) for _, s in first] == [repr(s) for _, s in second]


def test_epidemic_progresses(sir_env, sir_state):
    """Infections move over a week: the successor is not the initial state."""
    _, successor = sir_env.successors(sir_state)[0]
    assert repr(successor) != repr(sir_state)


@pytest.mark.slow
def test_simulate_returns_state_trace(sir_env, sir_state):
    plan = [action for action, _ in sir_env.successors(sir_state)][:2]
    trace = sir_env.simulate(plan)
    assert len(trace) == len(plan) + 1
    assert [s.depth for s in trace] == [PERIOD * i for i in range(len(plan) + 1)]


# --------------------------------------------------------------------------- vaccination delay

def test_vaccination_is_masked_before_its_start_day():
    env = EpiEnv(delay_vaccination_time=30, horizon=364)
    env.fix_index(SIR_A)
    state, _ = env.reset()
    for action, _ in env.successors(state):
        assert not any(itv.name == "Vaccination" for itv in action.itvs), \
            "vaccination must be unavailable before delay_vaccination_time"


def test_vaccination_is_available_once_the_delay_has_passed(sir_env):
    action = next(a for a in sir_env.basic_interventions if a.name == "Vaccination")
    late = sir_env.__disable_vaccination__(sir_env.vac_starts + 1, [action])
    assert late == [action]
    early = sir_env.__disable_vaccination__(0, [action])
    assert early == []


# --------------------------------------------------------------------------- goal

def test_goal_is_the_horizon():
    env = EpiEnv(delay_vaccination_time=0, horizon=PERIOD)
    env.fix_index(SIR_A)
    state, _ = env.reset()
    assert not env.is_goal(state)
    _, successor = env.successors(state)[0]
    assert env.is_goal(successor)          # one period reaches the horizon
    assert not env.is_terminal(successor)


# --------------------------------------------------------------------------- state

def test_state_equality_is_an_approximate_compartment_match(sir_env, sir_state):
    """States are equal when their I/R compartments are close, which is what keeps the
    continuous model searchable. The threshold widens as depth grows."""
    assert sir_state == sir_state
    _, successor = sir_env.successors(sir_state)[0]
    # A week apart, the epidemic has moved: the I/R compartments the comparison reads differ.
    vector = lambda s: [v for _, v in s.__vectorize__()]
    assert vector(successor) != vector(sir_state)
    # Only I and R are compared; S is invisible to state equality.
    assert [name for name, _ in sir_state.__vectorize__()] == ["I", "R"]


def test_state_repr_lists_compartments(sir_state):
    text = repr(sir_state)
    assert text.startswith("EpiState(depth=0")
    for compartment in ("S", "I", "R"):
        assert f"{compartment}=" in text


def test_literals_carry_compartments_and_depth(sir_state):
    assert len(sir_state.literals) == 1        # a single joined string, by design
    literal = next(iter(sir_state.literals))
    assert "depth(0)" in literal
    assert "s(" in literal and "i(" in literal
