"""Tests for the manufacturing environment."""
import json
import os

import pytest

pytest.importorskip("numpy", reason="numpy is not installed")

from planiverse.environments.manufacturing import mfenv
from planiverse.environments.manufacturing.mfenv import (  # noqa: E402
    ActionType, ConfigurationAction, MfgEnv, MfgState, total_produced,
)

from conftest import assert_state_contract, assert_successors_contract

# Derived from the module rather than spelled out, so moving the package cannot leave a
# stale path behind — which is exactly what the flat `planiverse.environments` layout did.
DATA_DIR = os.path.join(os.path.dirname(mfenv.__file__), "data")


# --------------------------------------------------------------------------- instances

def test_index_mapping_is_sorted_and_stable():
    """Indices must name the same instance on every machine.

    They were built from a bare os.listdir, whose order is filesystem-dependent, so a
    benchmark result reported against 'index 0' was not reproducible elsewhere.
    """
    env = MfgEnv()
    names = [os.path.basename(p) for p in env.data_index.values()]
    assert names == sorted(names)
    assert list(env.data_index.keys()) == list(range(len(names)))
    assert os.path.basename(env.data_index[0]) == "data.json"


def test_index_mapping_covers_every_data_file():
    env = MfgEnv()
    on_disk = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".json"))
    assert [os.path.basename(p) for p in env.data_index.values()] == on_disk


def test_fix_index_rejects_unknown_index():
    env = MfgEnv()
    with pytest.raises(AssertionError, match="not found in data index"):
        env.fix_index(999)


@pytest.mark.parametrize("index", range(7))
def test_every_instance_loads_and_resets(index):
    env = MfgEnv()
    env.fix_index(index)
    state, info = env.reset()
    assert_state_contract(state)
    assert env.DEMAND > 0 and env.DEMAND_TIME > 0
    assert env.NUM_CFGS == 5
    assert not env.is_goal(state)


def test_instances_are_feasible_as_loaded():
    """_setup_data refuses an instance that cannot possibly meet its demand."""
    env = MfgEnv()
    for index in env.data_index:
        env.fix_index(index)
        assert env._check_problem_feasibility()


def test_infeasible_instance_is_rejected(tmp_path):
    data = json.load(open(os.path.join(DATA_DIR, "data.json")))
    data["demand"] = 10 ** 9          # unreachable in the time available
    path = tmp_path / "infeasible.json"
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError, match="Infeasible"):
        MfgEnv()._setup_data(str(path))


# --------------------------------------------------------------------------- interface

def test_reset_is_repeatable(mfg_env):
    first, _ = mfg_env.reset()
    second, _ = mfg_env.reset()
    assert first == second


def test_successors_contract(mfg_env):
    state, _ = mfg_env.reset()
    assert_successors_contract(mfg_env.successors(state))


def test_only_buy_actions_are_offered_initially(mfg_env):
    """Nothing can produce before something is bought."""
    state, _ = mfg_env.reset()
    actions = [action for action, _ in mfg_env.successors(state)]
    assert len(actions) == mfg_env.NUM_CFGS
    assert all(a.action == ActionType.BUY_CFG.value for a in actions)
    assert {str(a) for a in actions} == {f"buy_cfg_{i}" for i in range(mfg_env.NUM_CFGS)}


def test_batch_actions_offered_once_bought(mfg_env):
    state, _ = mfg_env.reset()
    bought = state.apply_action(ConfigurationAction(0, ActionType.BUY_CFG))
    actions = [str(a) for a, _ in mfg_env.successors(bought)]
    # cfg 0 is bought, so it offers batches; the other four still offer buys.
    for size in [10, 20, 50, 100]:
        assert f"batch_production_cfg_0_size_{size}" in actions
    assert "buy_cfg_0" not in actions
    assert "buy_cfg_1" in actions


def test_simulate_returns_state_trace(mfg_env):
    mfg_env.reset()
    plan = [ConfigurationAction(0, ActionType.BUY_CFG),
            ConfigurationAction(0, ActionType.BATCH_PRODUCTION, 10)]
    trace = mfg_env.simulate(plan)
    assert len(trace) == len(plan) + 1
    for state in trace:
        assert_state_contract(state)


# --------------------------------------------------------------------------- buying

def test_buy_copies_market_values_and_does_not_advance_time(mfg_env):
    state, _ = mfg_env.reset()
    bought = state.apply_action(ConfigurationAction(0, ActionType.BUY_CFG))
    cfg = bought._state["configuration_costs"][0]
    assert cfg["bought"] is True
    assert cfg["incurred_costs"] == cfg["market_incurring_costs"]
    assert cfg["recurring_costs"] == cfg["market_recurring_costs"]
    assert cfg["production_rates"] == cfg["market_production_rates"]
    # Buying takes no time.
    assert bought._state["demand_time"] == state._state["demand_time"]


def test_buy_leaves_other_configurations_untouched(mfg_env):
    state, _ = mfg_env.reset()
    bought = state.apply_action(ConfigurationAction(0, ActionType.BUY_CFG))
    assert bought._state["configuration_costs"][1]["bought"] is False


# --------------------------------------------------------------------------- production

def test_production_needs_setup_time_before_output(mfg_env):
    """A machine contributes nothing until it is fully set up."""
    state, _ = mfg_env.reset()
    state = state.apply_action(ConfigurationAction(0, ActionType.BUY_CFG))
    setup = mfg_env.SETUP_TIMES[0]
    partway = state.batch_production(0, max(1, int(setup) - 1))
    assert total_produced(partway._state) == 0

    complete = state.batch_production(0, int(setup) + 5)
    assert total_produced(complete._state) > 0


def test_one_day_of_production_costs_one_day(mfg_env):
    state, _ = mfg_env.reset()
    state = state.apply_action(ConfigurationAction(0, ActionType.BUY_CFG))
    after = state.continue_production(0)
    assert after._state["demand_time"] == state._state["demand_time"] - 1


def test_batch_production_advances_the_clock_by_the_batch_size(mfg_env):
    state, _ = mfg_env.reset()
    state = state.apply_action(ConfigurationAction(0, ActionType.BUY_CFG))
    after = state.batch_production(0, 20)
    assert after._state["demand_time"] == state._state["demand_time"] - 20


def test_demand_aggregates_across_configurations(mfg_env):
    """Demand is met by the whole shop floor, not by one configuration.

    Remaining demand used to be computed from the produced count of whichever
    configuration ran last, so output from every other machine was invisible.
    """
    state, _ = mfg_env.reset()
    state = state.apply_action(ConfigurationAction(0, ActionType.BUY_CFG))
    state = state.apply_action(ConfigurationAction(1, ActionType.BUY_CFG))
    state = state.batch_production(0, 20)
    state = state.batch_production(1, 20)

    produced_by_each = [
        float(state._state["configuration_costs"][c]["produced_counts"]) for c in (0, 1)
    ]
    assert all(p > 0 for p in produced_by_each), "both configurations should have produced"
    assert state._state["demand"] == mfg_env.DEMAND - total_produced(state._state)
    # And that total really is the sum over configurations, not just the last one.
    assert total_produced(state._state) == int(sum(int(p) for p in produced_by_each))


def test_finish_production_all_advances_one_day_per_day(mfg_env):
    """demand_time used to be decremented once per configuration inside the loop,
    so a single day of running five machines burned five days of clock."""
    state, _ = mfg_env.reset()
    for cfg in range(mfg_env.NUM_CFGS):
        state = state.apply_action(ConfigurationAction(cfg, ActionType.BUY_CFG))
    after = state.finish_production_all()
    assert after._state["demand_time"] == state._state["demand_time"] - 1


def test_finish_production_all_runs_every_configuration(mfg_env):
    state, _ = mfg_env.reset()
    for cfg in range(mfg_env.NUM_CFGS):
        state = state.apply_action(ConfigurationAction(cfg, ActionType.BUY_CFG))
    # Run long enough for every machine to finish setting up and produce.
    for _ in range(int(max(mfg_env.SETUP_TIMES)) + 2):
        state = state.finish_production_all()
    produced = [float(state._state["configuration_costs"][c]["produced_counts"])
                for c in range(mfg_env.NUM_CFGS)]
    assert all(p > 0 for p in produced)


def test_batch_production_stops_at_the_goal(mfg_env):
    """A batch that would run past the deadline stops at it."""
    state, _ = mfg_env.reset()
    state = state.apply_action(ConfigurationAction(0, ActionType.BUY_CFG))
    after = state.batch_production(0, mfg_env.DEMAND_TIME + 500)
    assert after.is_goal()
    assert after._state["demand_time"] <= 0


# --------------------------------------------------------------------------- goal

def test_goal_is_the_clock_running_out(mfg_env):
    state, _ = mfg_env.reset()
    assert not mfg_env.is_goal(state)
    state = state.apply_action(ConfigurationAction(0, ActionType.BUY_CFG))
    state = state.batch_production(0, mfg_env.DEMAND_TIME)
    assert mfg_env.is_goal(state)


def test_no_terminal_states(mfg_env):
    state, _ = mfg_env.reset()
    assert not mfg_env.is_terminal(state)


# --------------------------------------------------------------------------- literals

def test_literals_do_not_collide_on_stripped_decimal_points(mfg_env):
    """'.' is escaped, not stripped: stripping made 1.5 and 15 the same literal."""
    state, _ = mfg_env.reset()
    bought = state.apply_action(ConfigurationAction(1, ActionType.BUY_CFG))
    rate = mfg_env.PRODN_RATES[1]
    assert rate == 1.5, "fixture assumption: cfg 1 of data.json produces 1.5 units/day"
    assert "production_rates(cfg1 1_5)" in bought.literals
    assert "production_rates(cfg1 15)" not in bought.literals


def test_literals_track_demand(mfg_env):
    state, _ = mfg_env.reset()
    assert f"demand({mfg_env.DEMAND})" in state.literals
    assert f"demand_time({mfg_env.DEMAND_TIME})" in state.literals


def test_state_equality_follows_literals(mfg_env):
    state, _ = mfg_env.reset()
    same, _ = mfg_env.reset()
    assert state == same
    different = state.apply_action(ConfigurationAction(0, ActionType.BUY_CFG))
    assert state != different
    assert not (state == "not a state")
