"""Tests for the power grid environment.

The case and its time series ship inside grid2op, so there is nothing to download. The
tests are slower than most in this suite — every successor is an AC power-flow solve — so
the expensive ones are marked `slow`.
"""
import pytest

pytest.importorskip("grid2op", reason="grid2op is not installed")

from planiverse.environments.power_grid.environment import (  # noqa: E402
    SECURE_RHO, SCENARIOS, PowerGridAction, PowerGridEnv, PowerGridState, Scenario,
)

from conftest import (  # noqa: E402
    assert_state_contract, assert_string_literals, assert_successors_contract,
)


@pytest.fixture(scope="module")
def env():
    game = PowerGridEnv()
    game.fix_index(4)          # chronic 1 / line 6: a four-step deadline, solved at depth 1
    yield game
    game.close()


# ----------------------------------------------------------------------- determinism

def test_replaying_a_path_lands_in_the_same_place(env):
    """The property the design rests on: the action path *is* the state, which is only
    true if replaying it is deterministic. Grid2op is, once `seed` and `set_id` pin the
    time series."""
    env.reset()
    path = (0, 0)
    first = env.__state__(path)
    env._cache.clear()                       # defeat the memo, so this really re-runs
    second = env.__state__(path)
    assert first.rhos == second.rhos
    assert first.max_rho == second.max_rho
    assert first.blackout == second.blackout


def test_two_environments_agree():
    """Determinism across separate simulator instances."""
    one, two = PowerGridEnv(), PowerGridEnv()
    try:
        one.fix_index(4)
        two.fix_index(4)
        first, first_info = one.reset()
        second, second_info = two.reset()
        assert first.rhos == second.rhos
        assert first_info == second_info
    finally:
        one.close()
        two.close()


@pytest.mark.slow
def test_expanding_a_state_twice_gives_identical_children(env):
    state, _ = env.reset()
    first = env.successors(state)
    env._cache.clear()
    second = env.successors(state)
    assert [a.action_id for a, _ in first] == [a.action_id for a, _ in second]
    for (_, a), (_, b) in zip(first, second):
        assert a == b and a.rhos == b.rhos


# ------------------------------------------------------------------------- instances

@pytest.mark.slow
def test_every_scenario_reproduces_its_recorded_measurements():
    """`rho_after_trip` and `blackout_in` are measurements. If grid2op changes a solver
    default this fails rather than quietly making the benchmark easier."""
    game = PowerGridEnv()
    try:
        for index, scenario in enumerate(SCENARIOS):
            game.fix_index(index)
            state, info = game.reset()
            assert info["chronic"] == scenario.chronic
            assert info["tripped_line"] == scenario.line
            assert state.max_rho == pytest.approx(scenario.rho_after_trip, abs=0.02)
            assert not state.is_secure(), "the trip has to leave the grid insecure"
    finally:
        game.close()


@pytest.mark.slow
def test_doing_nothing_really_does_black_the_grid_out(env):
    """What makes these instances instances.

    Most overloads on this case clear themselves as demand moves — tripping line 1 on
    chronic 0 gives 1.019 and is back under the limit two steps later with no action at
    all. Those are solved by the null plan and are deliberately not in the scenario list.
    Every scenario that *is* in it blacks out if ignored.
    """
    scenario = SCENARIOS[4]
    state, _ = env.reset()
    node = state
    for _ in range(scenario.blackout_in):
        node = env.__advance__(node, PowerGridAction(0))
    assert node.blackout, "doing nothing should end in a blackout"
    assert env.is_terminal(node)


def test_fix_index_refuses_a_scenario_that_is_not_there():
    game = PowerGridEnv()
    try:
        for index in (-1, len(SCENARIOS), 99):
            with pytest.raises(IndexError, match="Invalid index"):
                game.fix_index(index)
    finally:
        game.close()


def test_the_scenarios_span_severities_and_deadlines():
    assert len(SCENARIOS) >= 6
    assert {s.blackout_in for s in SCENARIOS} >= {2, 4}, "tight and loose deadlines"
    severities = [s.rho_after_trip for s in SCENARIOS]
    assert min(severities) < 1.05 and max(severities) > 1.7
    assert all(isinstance(s, Scenario) and s.solved_at >= 1 for s in SCENARIOS)


# --------------------------------------------------------------------------- contract

def test_state_contract(env):
    state, _ = env.reset()
    assert_state_contract(state)
    assert_string_literals(state)


@pytest.mark.slow
def test_successors_contract(env):
    state, _ = env.reset()
    assert_successors_contract(env.successors(state))


def test_a_state_is_identified_by_its_action_path(env):
    state, _ = env.reset()
    one = env.__advance__(state, PowerGridAction(0))
    again = env.__state__((0,))
    assert one == again and hash(one) == hash(again)
    assert one != state


def test_depth_is_not_part_of_state_identity(env):
    """With a step counter in the identity no successor could equal its parent and the
    self-loop filter would be dead code."""
    state, _ = env.reset()
    same = PowerGridState(state.path, state.max_rho, state.rhos, state.blackout,
                          state.step + 5, state.survived)
    assert same == state and hash(same) == hash(state)


def test_the_initial_state_is_insecure_but_alive(env):
    state, _ = env.reset()
    assert not state.blackout
    assert not env.is_goal(state)
    assert not env.is_terminal(state)
    assert state.max_rho > SECURE_RHO
    assert state.overloaded_lines(), "something has to be over its rating"


def test_doing_nothing_costs_nothing_but_acting_does():
    assert PowerGridAction(0).cost() == 0
    assert PowerGridAction(7).cost() == 1
    assert str(PowerGridAction(0)) == "do_nothing"
    assert str(PowerGridAction(7)) == "topology_7"


def test_a_blackout_is_terminal_and_absorbing(env):
    state, _ = env.reset()
    blacked_out = PowerGridState((0,), float("inf"), (), True, 5, False)
    assert env.is_terminal(blacked_out)
    assert env.successors(blacked_out) == []
    assert env.__advance__(blacked_out, PowerGridAction(0)) is blacked_out


def test_a_blackout_state_has_no_numeric_literals():
    """`max_rho` is infinite for a blackout precisely so it sorts last, which means the
    bucketed loading atom cannot be computed — it is simply absent."""
    blacked_out = PowerGridState((0,), float("inf"), (), True, 5, False)
    assert "blackout" in blacked_out.literals
    assert not any(lit.startswith("max-loading") for lit in blacked_out.literals)
    assert repr(blacked_out)          # must not raise on the infinity either


def test_a_secure_state_is_a_goal_and_absorbing(env):
    secure = PowerGridState((0,), 0.5, (0.5, 0.2), False, 3, True)
    assert env.is_goal(secure)
    assert env.successors(secure) == []
    assert "secure" in secure.literals


def test_overloaded_lines_appear_in_the_literals():
    state = PowerGridState((), 1.5, (0.2, 1.5, 0.9), False, 1, True)
    assert "overloaded(line-1)" in state.literals
    assert "overloaded(line-0)" not in state.literals
    assert "secure" not in state.literals


# ------------------------------------------------------------------------- solving it

@pytest.mark.slow
def test_the_scenario_is_solvable_and_validates(env):
    """One reconfiguration clears the overload. `validate` replays from scratch, so it is
    an independent confirmation rather than a restatement."""
    state, _ = env.reset()
    for action, child in env.successors(state):
        if env.is_goal(child):
            assert env.validate([action])
            assert child.max_rho < SECURE_RHO
            return
    pytest.fail("no single-action fix found, but the scenario records one")


@pytest.mark.slow
def test_one_reconfiguration_changes_the_whole_grid(env):
    """The reason this is not a PDDL domain: a local action has a global, numerical
    effect, so it cannot be written as facts to add and delete."""
    state, _ = env.reset()
    for action, child in env.successors(state):
        if action.action_id == 0 or child.blackout:
            continue
        differing = sum(1 for before, after in zip(state.rhos, child.rhos)
                        if abs(before - after) > 1e-6)
        assert differing > 1, "one reconfiguration moves more than one line's loading"
        return
    pytest.fail("no non-trivial successor to compare")


@pytest.mark.slow
def test_simulate_and_step_track_the_history(env):
    state, _ = env.reset()
    trace = env.simulate([PowerGridAction(0)])
    assert len(trace) == 2 and trace[0] == state

    env.reset()
    after, relief = env.step(PowerGridAction(0))
    assert after.path == (0,)
    assert isinstance(relief, float)
    assert len(env.render()) == 2
