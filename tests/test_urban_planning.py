"""Tests for the urban planning environment."""
import collections

import pytest

pytest.importorskip("pandas", reason="pandas is not installed")
pytest.importorskip("networkx", reason="networkx is not installed")

from planiverse.problems.real_world_problems.urban_planning.environment import (  # noqa: E402
    CHANGE_RATIO, ConvertCommercialAction, ConvertEmptyAction, ConvertFacilitiesAction,
    ConvertGreenSpaceAction, ConvertOfficesAction, LandUseType, RemoveResidentialAction,
    UrbanPlanningEnv, landuse_map,
)

from conftest import assert_state_contract, assert_successors_contract  # noqa: E402

KENDALL, ST_ANDREWS = 0, 1


@pytest.fixture
def kendall():
    env = UrbanPlanningEnv(horizon=100)
    env.fix_index(KENDALL)
    return env


@pytest.fixture
def st_andrews():
    env = UrbanPlanningEnv(horizon=100)
    env.fix_index(ST_ANDREWS)
    return env


def landuse_counts(state):
    return collections.Counter(
        state.urban_graph.nodes[n]["type"] for n in state.urban_graph.nodes
    )


# --------------------------------------------------------------------------- cities

def test_city_index_mapping(kendall, st_andrews):
    assert kendall.urban_name == "Kendall Square"
    assert st_andrews.urban_name == "St Andrews"


def test_fix_index_rejects_unknown_city():
    with pytest.raises(AssertionError, match="not found in city_info_index_map"):
        UrbanPlanningEnv(horizon=10).fix_index(99)


def test_kendall_loads_expected_parcels(kendall):
    state, _ = kendall.reset()
    assert kendall.graph.number_of_nodes() == 749
    counts = landuse_counts(state)
    assert counts[LandUseType.RESIDENTIAL] == 567
    assert counts[LandUseType.EMPTY] == 60
    assert counts[LandUseType.COMMERCIAL] == 0


def test_st_andrews_loads_expected_parcels(st_andrews):
    state, _ = st_andrews.reset()
    assert st_andrews.graph.number_of_nodes() == 260
    counts = landuse_counts(state)
    assert counts[LandUseType.GREEN_SPACE] == 223
    assert counts[LandUseType.EMPTY] == 0
    assert counts[LandUseType.OFFICE] == 0


def test_landuse_codes_cover_the_dataset():
    assert set(landuse_map) == {-1.0, 0.0, 1.0, 2.0, 3.0, 4.0}
    assert landuse_map[4.0] is LandUseType.GREEN_SPACE


def test_graph_has_spatial_edges(kendall):
    kendall.reset()
    assert kendall.graph.number_of_edges() > 0
    for _, _, data in kendall.graph.edges(data=True):
        assert "distance_m" in data
        break


def test_reset_is_repeatable(kendall):
    first, _ = kendall.reset()
    second, _ = kendall.reset()
    assert first == second


# --------------------------------------------------------------------------- interface

def test_successors_contract(kendall):
    state, _ = kendall.reset()
    assert_successors_contract(kendall.successors(state))


def test_successors_advance_depth(kendall):
    state, _ = kendall.reset()
    for _, successor in kendall.successors(state):
        assert successor.depth == state.depth + 1


def test_simulate_returns_state_trace(kendall):
    state, _ = kendall.reset()
    plan = [action for action, _ in kendall.successors(state)][:3]
    trace = kendall.simulate(plan)
    assert len(trace) == len(plan) + 1
    for st in trace:
        assert_state_contract(st)


def test_goal_is_the_horizon():
    env = UrbanPlanningEnv(horizon=3)
    env.fix_index(KENDALL)
    state, _ = env.reset()
    assert not env.is_goal(state)
    for _ in range(3):
        state = env.successors(state)[0][1]
    assert env.is_goal(state)
    assert not env.is_terminal(state)


# --------------------------------------------------------------------------- no-op filtering

def test_actions_that_rezone_nothing_are_not_offered(kendall):
    """Kendall Square has no commercial parcels, so ConvertCommercial can do nothing.

    A state carries its depth in its literals, so a successor never compares equal to its
    parent and the `successor_state == state` filter could never fire; no-op actions were
    all being offered as successors.
    """
    state, _ = kendall.reset()
    assert landuse_counts(state)[LandUseType.COMMERCIAL] == 0
    offered = [type(action) for action, _ in kendall.successors(state)]
    assert ConvertCommercialAction not in offered
    assert ConvertGreenSpaceAction in offered


def test_st_andrews_offers_the_actions_its_land_supports(st_andrews):
    state, _ = st_andrews.reset()
    offered = {type(action) for action, _ in st_andrews.successors(state)}
    # No empty land and no offices to convert...
    assert ConvertEmptyAction not in offered
    assert ConvertOfficesAction not in offered
    # ...but green space, commercial, facilities and residential are all present.
    assert offered == {ConvertGreenSpaceAction, ConvertCommercialAction,
                       ConvertFacilitiesAction, RemoveResidentialAction}


def test_every_offered_action_changes_the_city(kendall):
    state, _ = kendall.reset()
    for action, successor in kendall.successors(state):
        assert action.converted_nodes, f"{type(action).__name__} rezoned nothing"
        assert landuse_counts(successor) != landuse_counts(state)


# --------------------------------------------------------------------------- selection

def test_selection_takes_the_change_ratio_share(kendall):
    """5% of a class, rounded up, so a small class is still rezonable."""
    state, _ = kendall.reset()
    residential = landuse_counts(state)[LandUseType.RESIDENTIAL]
    action = RemoveResidentialAction()
    action(state)
    expected = -(-residential // int(1 / CHANGE_RATIO))     # ceil(residential * 0.05)
    assert len(action.converted_nodes) == expected


def test_small_land_classes_are_still_rezonable(kendall):
    """int(15 * 0.05) truncates to 0, so any class under 20 parcels was frozen forever."""
    state, _ = kendall.reset()
    assert landuse_counts(state)[LandUseType.FACILITIES] == 15
    action = ConvertFacilitiesAction()
    successor = action(state)
    assert len(action.converted_nodes) == 1
    assert landuse_counts(successor)[LandUseType.FACILITIES] == 14


# --------------------------------------------------------------------------- conversions

def test_convert_empty_splits_evenly_across_all_five_uses(st_andrews):
    """Every land used to be paired with every type and the last assignment won,
    so 'split evenly between r/o/g/c/f' silently made everything facilities."""
    env = UrbanPlanningEnv(horizon=10)
    env.fix_index(KENDALL)
    state, _ = env.reset()
    action = ConvertEmptyAction()
    successor = action(state)

    new_types = [new for _, _, new in action.converted_nodes]
    assert set(new_types) <= {LandUseType.RESIDENTIAL, LandUseType.OFFICE,
                              LandUseType.GREEN_SPACE, LandUseType.COMMERCIAL,
                              LandUseType.FACILITIES}
    # Round-robin: no type gets more than one parcel more than any other.
    tally = collections.Counter(new_types)
    assert max(tally.values()) - min(tally.values()) <= 1
    # And they are definitely not all facilities.
    assert tally[LandUseType.FACILITIES] < len(new_types)
    assert landuse_counts(successor)[LandUseType.EMPTY] < landuse_counts(state)[LandUseType.EMPTY]


def test_convert_green_space_splits_between_facilities_and_commercial(st_andrews):
    state, _ = st_andrews.reset()
    action = ConvertGreenSpaceAction()
    action(state)
    tally = collections.Counter(new for _, _, new in action.converted_nodes)
    assert set(tally) == {LandUseType.FACILITIES, LandUseType.COMMERCIAL}
    assert max(tally.values()) - min(tally.values()) <= 1


def test_convert_offices_to_commercial(kendall):
    state, _ = kendall.reset()
    action = ConvertOfficesAction()
    successor = action(state)
    assert {new for _, _, new in action.converted_nodes} == {LandUseType.COMMERCIAL}
    assert {old for _, old, _ in action.converted_nodes} == {LandUseType.OFFICE}
    assert landuse_counts(successor)[LandUseType.OFFICE] < landuse_counts(state)[LandUseType.OFFICE]


def test_convert_commercial_to_facilities(st_andrews):
    state, _ = st_andrews.reset()
    action = ConvertCommercialAction()
    action(state)
    assert {new for _, _, new in action.converted_nodes} == {LandUseType.FACILITIES}


def test_remove_residential_clears_to_empty(kendall):
    state, _ = kendall.reset()
    action = RemoveResidentialAction()
    successor = action(state)
    assert {new for _, _, new in action.converted_nodes} == {LandUseType.EMPTY}
    assert landuse_counts(successor)[LandUseType.EMPTY] > landuse_counts(state)[LandUseType.EMPTY]


def test_conversion_preserves_the_parcel_count(kendall):
    state, _ = kendall.reset()
    for _, successor in kendall.successors(state):
        assert successor.urban_graph.number_of_nodes() == state.urban_graph.number_of_nodes()


def test_apply_does_not_mutate_the_source_state(kendall):
    state, _ = kendall.reset()
    before = landuse_counts(state)
    RemoveResidentialAction()(state)
    assert landuse_counts(state) == before


# --------------------------------------------------------------------------- action reuse

def test_a_fresh_action_can_be_applied_directly(kendall):
    """converted_nodes used to be created in __call__, so apply()/str() on a fresh
    action raised AttributeError -- which is exactly what simulate() does."""
    state, _ = kendall.reset()
    action = RemoveResidentialAction()
    assert action.converted_nodes == []
    assert str(action)                      # must not raise
    action.apply(state)
    assert action.converted_nodes


def test_action_replay_does_not_accumulate(kendall):
    """apply() recomputes its conversions, so replaying a plan is stable."""
    state, _ = kendall.reset()
    action = RemoveResidentialAction()
    first = action.apply(state)
    count = len(action.converted_nodes)
    second = action.apply(state)
    assert len(action.converted_nodes) == count
    assert landuse_counts(first) == landuse_counts(second)


def test_action_string_reflects_the_last_application(kendall):
    state, _ = kendall.reset()
    action = ConvertOfficesAction()
    action.apply(state)
    assert str(action).startswith("action_")
    assert "_c" in str(action)              # offices became commercial


def test_simulate_matches_successors(kendall):
    """Replaying the actions successors handed out reproduces the same states."""
    state, _ = kendall.reset()
    plan, expected = [], []
    for _ in range(3):
        action, successor = kendall.successors(state)[0]
        plan.append(action)
        expected.append(successor)
        state = successor
    trace = kendall.simulate(plan)
    for produced, want in zip(trace[1:], expected):
        assert landuse_counts(produced) == landuse_counts(want)


# --------------------------------------------------------------------------- scores

def test_scores_are_within_range(kendall):
    state, _ = kendall.reset()
    assert 0.0 <= state.sustainability_score <= 1.0
    assert 0.0 <= state.diversity_score <= 1.0


def test_sustainability_counts_green_commercial_and_facilities(kendall):
    state, _ = kendall.reset()
    counts = landuse_counts(state)
    non_empty = sum(v for k, v in counts.items() if k is not LandUseType.EMPTY)
    expected = round((counts[LandUseType.GREEN_SPACE] + counts[LandUseType.COMMERCIAL]
                      + counts[LandUseType.FACILITIES]) / non_empty, 1)
    assert state.sustainability_score == expected


def test_diversity_rises_when_a_monoculture_is_broken(st_andrews):
    """St Andrews is overwhelmingly green space; rezoning some of it mixes the city up."""
    state, _ = st_andrews.reset()
    successor = ConvertGreenSpaceAction()(state)
    assert successor.diversity_score >= state.diversity_score


# --------------------------------------------------------------------------- literals

def test_literals_are_landuse_counts_and_depth(kendall):
    state, _ = kendall.reset()
    counts = landuse_counts(state)
    assert f"r_{counts[LandUseType.RESIDENTIAL]}" in state.literals
    assert f"g_{counts[LandUseType.GREEN_SPACE]}" in state.literals
    assert "depth_0" in state.literals


def test_literals_break_parcel_symmetry(kendall):
    """Literals count parcels per type rather than naming them, so which particular
    parcel was rezoned does not distinguish two states."""
    state, _ = kendall.reset()
    assert not any(lit.startswith("land_") for lit in state.literals)
    assert len(state.literals) == len(LandUseType) + 1      # one per type, plus depth
