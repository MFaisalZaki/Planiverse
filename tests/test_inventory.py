"""The inventory environment: or-gym's multi-echelon supply chain as a planning problem.

The chain's dynamics are or-gym's and are not re-derived here. What the tests pin is the
reduction: the season's demand is the instance's, so expansion is deterministic; a state is
the stock, the pipeline and the accumulators, written into or-gym's arrays to expand it, so it
agrees with a straight run of or-gym; the retailer's lost sales over the allowance are a dead
end; and the target is what the best reference policy earns, so every bundled season is solved
by the plan it was accepted on.
"""
import json
import os

import pytest

from conftest import assert_string_literals, assert_successors_contract

pytest.importorskip("or_gym", exc_type=ImportError, reason="or-gym is not installed")

from planiverse.environments.operational.inventory.environment import (  # noqa: E402
    ACTIONS, SEASONS, InventoryAction, InventoryEnv, InventoryState, reference_plans,
)

DATA = os.path.join(os.path.dirname(__file__), "data")


def solutions():
    with open(os.path.join(DATA, "inventory_solutions.json")) as handle:
        return {int(index): plan for index, plan in json.load(handle).items()}


@pytest.fixture(scope="module")
def inventory():
    env = InventoryEnv()
    env.set_index(0)
    return env


def test_the_contract_on_the_initial_state(inventory):
    state, info = inventory.reset()
    assert info == {"season": 0, "target": inventory.instance["target"],
                    "allowed": inventory.instance["allowed"], "generated": False}
    assert_string_literals(state)
    assert state.period == 0 and state.profit == 0 and state.lost == 0
    assert not inventory.is_goal(state) and not inventory.is_terminal(state)
    children = inventory.successors(state)
    assert_successors_contract(children)
    assert [action for action, _ in children] == list(ACTIONS) == inventory.get_actions()
    assert len(ACTIONS) == 9
    assert all(child.period == 1 for _, child in children)
    assert set(state.readings()) >= {"retailer", "demand", "lost sales", "profit"}
    assert str(state).splitlines()[-1].startswith("period 0/")


def test_actions_parse_and_refuse_unknown_levels():
    assert InventoryAction.parse("stock(low, high)") == InventoryAction("low", "high")
    assert str(InventoryAction("mid", "low")) == "stock(mid, low)"
    assert InventoryAction("mid", "low").levels == ("mid", "low", "low"), \
        "the distributor and the manufacturer take the upstream level"
    with pytest.raises(ValueError):
        InventoryAction("low", "huge")


def test_expansion_is_deterministic_and_identity_is_the_value(inventory):
    state, _ = inventory.reset()
    first = inventory.successors(state)
    again = inventory.successors(state)
    assert [(a, s.key) for a, s in first] == [(a, s.key) for a, s in again]
    a, b = first[0][1], again[0][1]
    assert a == b and hash(a) == hash(b) and a.literals == b.literals
    deeper = InventoryState(a.period, a.on_hand, a.pipeline, a.profit, a.lost, depth=a.depth + 5)
    assert deeper == a, "the depth is not part of a state's identity"
    assert a < deeper


def test_a_state_written_into_or_gym_agrees_with_a_straight_run(inventory):
    """Expanding writes a state's numbers into or-gym's arrays; a plan stepped straight
    through one or-gym chain, with no expansions in between, ends the same way."""
    import numpy as np

    plan = [ACTIONS[(4 * step) % len(ACTIONS)] for step in range(inventory.instance["periods"])]
    state, _ = inventory.reset()
    for action in plan:
        inventory.successors(state)                    # disturbs the arrays between steps
        state = inventory._step(state, action)
    inventory.close()
    chain = inventory._chain()
    profit = 0.0
    for action in plan:
        levels = np.array([inventory.instance["levels"][stage][level]
                           for stage, level in enumerate(action.levels)])
        chain._STEP(chain.base_stock_action(levels))
        profit += chain.P[chain.period - 1]
    assert state.on_hand == tuple(int(v) for v in chain.I[chain.period])
    assert state.lost == int(chain.LS[:, 0].sum())
    assert state.profit == pytest.approx(profit, abs=0.05)
    inventory.close()


def test_lost_sales_over_the_allowance_are_a_dead_end(inventory):
    allowed = inventory.instance["allowed"]
    over = InventoryState(3, (0, 10, 10), ((0,),) * 3, 10.0, allowed + 1)
    assert inventory.is_terminal(over) and not inventory.is_goal(over)
    assert inventory.successors(over) == []
    assert inventory.__advance__(over, ACTIONS[0]) is over
    periods, target = inventory.instance["periods"], inventory.instance["target"]
    done = InventoryState(periods, (0, 0, 0), ((0,),) * 3, target, allowed)
    assert inventory.is_goal(done) and not inventory.is_terminal(done)
    short = InventoryState(periods, (0, 0, 0), ((0,),) * 3, target - 1, 0)
    assert inventory.is_terminal(short) and not inventory.is_goal(short), \
        "the horizon passed below the target is a dead end"


def test_the_witness_and_the_stored_plans_reach_the_goal():
    plans = solutions()
    assert len(SEASONS) == 100 and sorted(plans) == list(range(100))
    for index in (0, 50, 99):
        env = InventoryEnv()
        env.set_index(index)
        plan = [InventoryAction.parse(text) for text in plans[index]]
        trace = env.simulate(plan)
        assert env.is_goal(trace[-1]), f"season {index}"
        assert trace[-1].profit == env.instance["target"], "the target is the plan's own profit"
        assert len(plan) == env.instance["periods"]


def test_the_target_is_the_best_reference_policy():
    env = InventoryEnv()
    env.set_index(0)
    feasible = []
    for plan in reference_plans(env.instance["periods"]):
        last = env.simulate(plan)[-1]
        if last.period == env.instance["periods"] and last.lost <= env.instance["allowed"]:
            feasible.append(last.profit)
    assert max(feasible) == env.instance["target"]


def test_generate_instance_is_a_function_of_the_seed_and_round_trips():
    first, second = InventoryEnv(), InventoryEnv()
    instance = first.generate_instance(seed=7)
    assert instance == second.generate_instance(seed=7)
    assert first.witness and first.validate(first.witness)
    restored = InventoryEnv()
    restored.set_instance(json.loads(json.dumps(instance)))
    assert restored.reset()[0].literals == first.reset()[0].literals
    assert restored.reset()[1]["generated"] is True
    assert restored.validate(first.witness)
    first.set_index(0)
    assert first.witness is None, "selecting a bundled season clears the witness"


def test_set_instance_round_trips_a_bundled_season():
    env = InventoryEnv()
    env.set_index(3)
    other = InventoryEnv()
    other.set_instance(json.loads(json.dumps(SEASONS[3])))
    assert other.instance == env.instance
    assert other.reset()[0] == env.reset()[0]
    with pytest.raises(ValueError):
        other.set_instance({key: value for key, value in SEASONS[3].items() if key != "target"})
    with pytest.raises(IndexError):
        env.set_index(len(SEASONS))


@pytest.mark.slow
def test_every_bundled_season_is_solved_by_its_plan_and_regenerates_from_its_seed():
    plans = solutions()
    env = InventoryEnv()
    for index, season in enumerate(SEASONS):
        env.set_index(index)
        assert env.validate([InventoryAction.parse(text) for text in plans[index]]), index
    for index in (0, 99):
        assert InventoryEnv().generate_instance(seed=SEASONS[index]["seed"]) == SEASONS[index]


# ------------------------------------------------------------------------- the picture

def test_the_picture_draws_the_chain_and_a_status_line(inventory):
    from planiverse.rendering.text import missing_glyphs

    state, _ = inventory.reset()
    picture = str(state)
    assert isinstance(picture, str) and picture and not missing_glyphs(picture)
    for name in ("raw", "manufacturer", "distributor", "retailer", "customers", "▶"):
        assert name in picture
    assert f"{state.on_hand[0]} on hand" in picture
    assert picture.splitlines()[-1] == (
        f"period 0/{state.horizon} · profit 0.00 of {state.target:.2f} · "
        f"lost sales 0 of {state.allowed}")
    child = inventory.__advance__(state, ACTIONS[4])
    assert str(child) != picture
    assert f"demand {child.last[0]}" in str(child)
    assert len(str(child).splitlines()) == len(picture.splitlines()), \
        "the geometry is the season's, so a GIF of a trace does not jump"
    assert "chain" not in state.key
