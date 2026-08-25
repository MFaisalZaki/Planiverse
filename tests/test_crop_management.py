"""Tests for the crop management environment.

The weather ships inside PCSE and the crop parameters are cached locally by PCSE itself, so
a season runs offline and there is nothing to supply.
"""
import pytest

pytest.importorskip("pcse", reason="pcse is not installed")

from planiverse.problems.real_world_problems.crop_management.environment import (  # noqa: E402
    DECISION_COUNT, IRRIGATION_AMOUNTS, SCENARIOS, CropAction, CropEnv, CropState, Scenario,
    decision_days,
)

from conftest import (  # noqa: E402
    assert_state_contract, assert_string_literals, assert_successors_contract,
)


@pytest.fixture(scope="module")
def env():
    game = CropEnv()
    game.fix_index(10)          # 1986: the season where irrigation matters most
    yield game
    game.close()


# ----------------------------------------------------------------------- determinism

def test_the_same_schedule_yields_the_same_crop(env):
    """The crop model integrates a differential equation; if it drifted, two states with
    the same schedule would not be the same state and search could not close over them."""
    env.reset()
    schedule = (0.0, 2.0, 0.0)
    first = env.__state__(schedule)
    env._cache.clear()                       # defeat the memo, so this really re-runs
    second = env.__state__(schedule)
    assert first.biomass == second.biomass
    assert first.yield_kg == second.yield_kg


def test_a_whole_season_replays_identically(env):
    plan = env.reference_plan()
    first = env.simulate(plan)[-1]
    env._cache.clear()
    second = env.simulate(plan)[-1]
    assert first.yield_kg == second.yield_kg
    assert first == second


def test_two_environments_agree():
    one, two = CropEnv(), CropEnv()
    try:
        one.fix_index(10)
        two.fix_index(10)
        assert one.simulate(one.reference_plan())[-1].yield_kg == \
               two.simulate(two.reference_plan())[-1].yield_kg
    finally:
        one.close()
        two.close()


# ------------------------------------------------------------------------- instances

def test_the_reference_and_rainfed_yields_reproduce(env):
    """Both are measurements. If PCSE changes a default this fails rather than quietly
    shifting every scenario's difficulty."""
    scenario = env.scenario()
    rainfed = env.simulate([CropAction(0.0)] * DECISION_COUNT)[-1]
    reference = env.simulate(env.reference_plan())[-1]
    assert rainfed.yield_kg == pytest.approx(scenario.rainfed, abs=1.0)
    assert reference.yield_kg == pytest.approx(scenario.reference, abs=1.0)


def test_every_season_has_a_witness():
    """The reference schedule is what proves each instance is solvable — the target is
    measured off it, so no scenario ships whose goal nobody has reached."""
    assert all(isinstance(s, Scenario) for s in SCENARIOS)
    assert all(s.reference >= s.rainfed for s in SCENARIOS), "irrigation never hurts here"
    assert len(SCENARIOS) == 22, "1990 and 1991 have gaps in the bundled weather"


def test_the_seasons_span_wet_and_dry():
    """The spread is the point. In some years irrigation is worth thousands of kg/ha and in
    others it is worth nothing, and the states look identical when the first decision is
    taken."""
    gains = {s.year: s.reference - s.rainfed for s in SCENARIOS}
    assert gains[1980] == pytest.approx(0.0, abs=1.0), "1980 was wet enough on its own"
    assert gains[1986] > 2500, "1986 needed the water badly"
    assert min(gains.values()) < 100 and max(gains.values()) > 2500


def test_fix_index_refuses_a_season_that_is_not_there():
    game = CropEnv()
    try:
        for index in (-1, len(SCENARIOS), 99):
            with pytest.raises(IndexError, match="Invalid index"):
                game.fix_index(index)
    finally:
        game.close()


def test_reset_without_fix_index_takes_the_first_season():
    game = CropEnv()
    try:
        _, info = game.reset()
        assert info["year"] == SCENARIOS[0].year
    finally:
        game.close()


# --------------------------------------------------------------------------- contract

def test_state_contract(env):
    state, _ = env.reset()
    assert_state_contract(state)
    assert_string_literals(state)


def test_successors_contract(env):
    state, _ = env.reset()
    assert_successors_contract(env.successors(state))


def test_a_state_is_identified_by_its_schedule(env):
    state, _ = env.reset()
    one = env.__advance__(state, CropAction(2.0))
    again = env.__state__((2.0,))
    assert one == again and hash(one) == hash(again)


def test_depth_is_not_part_of_state_identity(env):
    state, _ = env.reset()
    same = CropState(state.schedule, state.biomass, state.yield_kg, state.water_used,
                     state.finished, stage=99)
    assert same == state and hash(same) == hash(state)


def test_waiting_is_free_and_water_is_the_cost():
    assert CropAction(0.0).cost() == 0
    assert CropAction(2.0).cost() == 2.0
    assert str(CropAction(0.0)) == "wait"
    assert str(CropAction(2.0)) == "irrigate_2cm"


def test_there_are_ten_decision_points_ten_days_apart():
    days = decision_days()
    assert len(days) == DECISION_COUNT == 10
    assert days[0] == 10 and days[-1] == 100
    assert all(b - a == 10 for a, b in zip(days, days[1:]))


def test_successors_are_limited_by_what_is_left_in_the_budget(env):
    """A plan cannot promise water it has already spent."""
    state, _ = env.reset()
    spent = env.__state__((4.0, 2.0))
    amounts = sorted(action.amount for action, _ in env.successors(spent))
    assert max(amounts) <= env.budget_cm - spent.water_used + 1e-9
    assert amounts == sorted(a for a in IRRIGATION_AMOUNTS if a <= 2.0 + 1e-9)


def test_the_season_ends_after_the_last_decision(env):
    state, _ = env.reset()
    node = state
    for _ in range(DECISION_COUNT):
        assert not node.finished
        node = env.__advance__(node, CropAction(0.0))
    assert node.finished and node.depth == DECISION_COUNT
    assert env.successors(node) == []


def test_no_part_grown_state_is_ever_a_goal(env):
    """The yield only exists at harvest, so the whole season has to be planned before the
    outcome is visible. That is the domain, not a modelling choice."""
    state, _ = env.reset()
    node = state
    for _ in range(DECISION_COUNT - 1):
        node = env.__advance__(node, CropAction(0.0))
        assert not env.is_goal(node)
        assert not node.finished


def test_overspending_the_budget_is_terminal(env):
    over = CropState((4.0, 4.0, 4.0), 500.0, 0.0, 12.0, False, 3)
    assert env.is_terminal(over)
    assert env.successors(over) == []


# ------------------------------------------------------------------------- the agronomy

def test_the_reference_plan_solves_the_season_and_validates(env):
    """The witness. `validate` replays from scratch, so it confirms independently."""
    plan = env.reference_plan()
    final = env.simulate(plan)[-1]
    assert final.finished
    assert final.water_used == pytest.approx(env.budget_cm)
    assert env.is_goal(final)
    assert env.validate(plan)


def test_doing_nothing_misses_the_target_in_a_dry_year(env):
    """1986 loses 2698 kg/ha to drought if the farmer waits."""
    final = env.simulate([CropAction(0.0)] * DECISION_COUNT)[-1]
    assert final.finished
    assert not env.is_goal(final)
    assert final.yield_kg < env.target_yield()


def test_doing_nothing_is_enough_in_a_wet_year():
    """1980's reference schedule gains exactly nothing, so the right plan applies no water
    at all — knowing when *not* to act is part of the problem."""
    game = CropEnv()
    try:
        game.fix_index(4)                      # 1980
        game.reset()
        final = game.simulate([CropAction(0.0)] * DECISION_COUNT)[-1]
        assert game.is_goal(final), "no irrigation should already clear the target"
        assert final.water_used == 0
    finally:
        game.close()


def test_an_actions_effect_is_not_visible_when_it_is_taken(env):
    """Why this is not a PDDL domain, part one.

    Water applied at the first decision point changes nothing measurable by the second —
    the crop has not had time to respond. The effect is a change to the trajectory of a
    growth equation, and it only becomes a yield at harvest eighty days later.
    """
    state, _ = env.reset()
    biomasses = {round(child.biomass, 6) for _, child in env.successors(state)}
    assert len(biomasses) == 1, \
        "at the next decision point the four choices are indistinguishable"


def test_the_same_action_is_worth_anything_from_nothing_to_the_crop(env):
    """Why this is not a PDDL domain, part two — and the sharpest version of it.

    One 4 cm application, the identical action, is worth between nothing and +2544 kg/ha
    on the same field in the same season depending *only* on which day it lands. Early on
    it does nothing at all: the soil is already at capacity and the crop is tiny, so the
    water drains straight through. Its value then climbs to a peak around day 80 and falls
    away again.

    No add/delete list can carry that. The effect of the action is a function of a growth
    stage which is itself the integral of every decision before it.
    """
    env.reset()
    base = env.simulate([CropAction(0.0)] * DECISION_COUNT)[-1].yield_kg
    gains = []
    for index in range(DECISION_COUNT):
        plan = [CropAction(0.0)] * DECISION_COUNT
        plan[index] = CropAction(4.0)
        gains.append(env.simulate(plan)[-1].yield_kg - base)

    assert gains[0] == pytest.approx(0.0, abs=1.0), "water on day 10 is wasted entirely"
    assert max(gains) > 2000, "and mid-season it is worth most of the crop"

    # Non-monotone in timing: the best day is neither the first nor the last.
    best = gains.index(max(gains))
    assert 0 < best < DECISION_COUNT - 1
    assert gains[-1] < max(gains), "past the peak the same water is worth less again"


def test_simulate_and_step_track_the_history(env):
    state, _ = env.reset()
    trace = env.simulate([CropAction(0.0), CropAction(2.0)])
    assert len(trace) == 3 and [s.depth for s in trace] == [0, 1, 2]

    env.reset()
    after, growth = env.step(CropAction(2.0))
    assert after.schedule == (2.0,)
    assert isinstance(growth, float)
    assert len(env.render()) == 2
