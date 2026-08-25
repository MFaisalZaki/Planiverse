"""Tests for the water distribution environment.

Everything here runs against the benchmark networks WNTR ships, so there is nothing to
download and nothing to supply — unlike the Game Boy environments, the "real thing" is
already in the package.
"""
import pytest

pytest.importorskip("wntr", reason="wntr is not installed")

from planiverse.environments.water_network.environment import (  # noqa: E402
    CONTAMINATION_GOAL, SCENARIOS, SERVICE_FLOOR, SERVICE_GOAL, Scenario, WaterNetworkAction,
    WaterNetworkEnv, WaterNetworkState, network_library, rank_sources,
)

from conftest import (  # noqa: E402
    assert_state_contract, assert_string_literals, assert_successors_contract,
)


@pytest.fixture
def env():
    game = WaterNetworkEnv()
    game.fix_index(0)          # Net1 / node 23: the smallest scenario, solved at depth 2
    yield game
    game.close()


# ----------------------------------------------------------------------- determinism
# The property the whole design rests on: the closed set is a sufficient statistic for the
# state, which is only true if the solve is deterministic.

def test_the_same_configuration_simulates_identically(env):
    """Bit-identical, not merely close. Search caches on the closed set and treats two
    states with the same pipes shut as the same state; if the solver drifted at all, that
    would silently merge states that are not the same."""
    env.reset()
    closed = frozenset({env.candidates[0], env.candidates[1]})
    first = env.__simulate__(closed)
    env._cache.clear()                       # defeat the memo, so this really re-solves
    second = env.__simulate__(closed)
    assert first == second


def test_expanding_a_state_twice_gives_identical_children(env):
    """The contract `successors` has to satisfy for search to be sound."""
    state, _ = env.reset()
    first = env.successors(state)
    env._cache.clear()
    second = env.successors(state)
    assert [str(a) for a, _ in first] == [str(a) for a, _ in second]
    for (_, a), (_, b) in zip(first, second):
        assert a == b
        assert (a.contaminated, a.service) == (b.contaminated, b.service)


def test_two_environments_agree(env):
    """Determinism across processes' worth of separation: a fresh environment on the same
    scenario reproduces the same numbers."""
    state, info = env.reset()
    other = WaterNetworkEnv()
    try:
        other.fix_index(0)
        again, other_info = other.reset()
        assert (again.contaminated, again.service) == (state.contaminated, state.service)
        assert other_info == info
    finally:
        other.close()


def test_a_plan_replays_to_the_same_place(env):
    """`simulate` re-runs from scratch, so it is an independent check on `successors`."""
    state, _ = env.reset()
    plan = [action for action, _ in env.successors(state)[:2]]
    first = env.simulate(plan)[-1]
    second = env.simulate(plan)[-1]
    assert first == second
    assert (first.contaminated, first.service) == (second.contaminated, second.service)


# ------------------------------------------------------------------------- instances

def test_every_scenario_loads_with_the_baseline_it_records():
    """The recorded contamination is a measurement, so it has to keep matching. If WNTR
    changes a solver default this fails rather than quietly shifting the benchmark."""
    env = WaterNetworkEnv()
    try:
        for index, scenario in enumerate(SCENARIOS):
            env.fix_index(index)
            state, info = env.reset()
            assert info["network"] == scenario.network
            assert info["source"] == scenario.source
            assert state.contaminated == pytest.approx(scenario.baseline, abs=0.01)
            assert state.service == pytest.approx(1.0, abs=0.01), "nothing closed yet"
    finally:
        env.close()


def test_the_scenarios_span_easy_and_hard():
    """A benchmark of nine instances that all solve in two moves is one instance."""
    depths = sorted(scenario.solved_at for scenario in SCENARIOS)
    assert min(depths) <= 2 and max(depths) >= 7
    assert len(set(depths)) >= 4, "several distinct difficulties"


def test_every_scenario_has_a_known_solution_depth():
    """No instance ships whose goal nobody has reached — a planner cannot tell an
    unreachable goal from one it has not found yet, and `Net2` was dropped for that."""
    assert all(isinstance(s, Scenario) and s.solved_at >= 1 for s in SCENARIOS)
    assert not any(s.network.startswith("Net2") for s in SCENARIOS)


def test_fix_index_refuses_a_scenario_that_is_not_there():
    env = WaterNetworkEnv()
    try:
        for index in (-1, len(SCENARIOS), 99):
            with pytest.raises(IndexError, match="Invalid index"):
                env.fix_index(index)
    finally:
        env.close()


def test_reset_without_fix_index_takes_the_first_scenario():
    env = WaterNetworkEnv()
    try:
        _, info = env.reset()
        assert info["network"] == SCENARIOS[0].network
    finally:
        env.close()


def test_rank_sources_is_how_the_scenarios_were_chosen():
    """Kept runnable so the choice of source can be re-derived rather than trusted."""
    import os

    ranked = rank_sources(os.path.join(network_library(), "Net1.inp"))
    assert ranked == sorted(ranked, reverse=True), "worst first"
    by_source = {source: contaminated for contaminated, _, source in ranked}
    for scenario in SCENARIOS:
        if scenario.network == "Net1.inp":
            assert by_source[scenario.source] == pytest.approx(scenario.baseline, abs=0.01)


# --------------------------------------------------------------------------- contract

def test_state_contract(env):
    state, _ = env.reset()
    assert_state_contract(state)
    assert_string_literals(state)


def test_successors_contract(env):
    state, _ = env.reset()
    assert_successors_contract(env.successors(state))


def test_a_state_is_identified_by_its_closed_set(env):
    """Because the solve is deterministic, the closed set determines everything else, so
    two states reached different ways compare equal and search can close over them."""
    state, _ = env.reset()
    a, b = env.candidates[0], env.candidates[1]
    one = env.simulate([WaterNetworkAction(a), WaterNetworkAction(b)])[-1]
    other = env.simulate([WaterNetworkAction(b), WaterNetworkAction(a)])[-1]
    assert one == other and hash(one) == hash(other)
    assert one.closed == {a, b}


def test_depth_is_not_part_of_state_identity(env):
    """The trap the urban planning environment fell into: with a step counter in the
    identity no successor can ever equal its parent and the self-loop filter is dead."""
    state, _ = env.reset()
    same = WaterNetworkState(state.closed, state.service, state.contaminated,
                             state.pressure_deficit, depth=99)
    assert same == state and hash(same) == hash(state)


def test_successors_do_not_repeat_a_closed_pipe(env):
    state, _ = env.reset()
    child = env.successors(state)[0][1]
    for action, _ in env.successors(child):
        assert action.pipe not in child.closed


def test_closing_a_pipe_is_recorded_in_the_literals(env):
    state, _ = env.reset()
    action, child = env.successors(state)[0]
    assert f"closed({action.pipe})" in child.literals
    assert f"closed({action.pipe})" not in state.literals


def test_actions_cost_one_each(env):
    env.reset()
    assert all(action.cost() == 1 for action in env.get_actions())


def test_action_string_is_readable():
    assert str(WaterNetworkAction("110")) == "close_pipe_110"


# ----------------------------------------------------------------- the physics itself

def test_one_closed_pipe_changes_the_whole_network(env):
    """The reason this is not a PDDL domain. A local action has a global effect, so it
    cannot be written as a list of facts to add and delete."""
    state, _ = env.reset()
    changed = [child for _, child in env.successors(state)
               if child.contaminated != state.contaminated]
    assert changed, "closing a pipe near the source has to move the contamination"


def test_closing_a_pipe_can_make_the_contamination_worse(env):
    """Not monotone, which is the other half of the argument: a planner that assumes more
    closures contain more contamination is simply wrong about this domain.

    On Net1 with the source at node 12, closing pipe 110 pushes flow down a path that
    reaches *more* customers.
    """
    game = WaterNetworkEnv()
    try:
        game.fix_index(6)                      # Net1 / node 12
        state, _ = game.reset()
        worse = [child for _, child in game.successors(state)
                 if child.contaminated > state.contaminated + 1e-6]
        assert worse, "closing some pipe should make containment worse, not better"
    finally:
        game.close()


def test_service_is_monotone_even_though_contamination_is_not(env):
    """What makes `is_terminal` sound: a network cannot deliver more water with more pipes
    shut, so a collapsed state can never recover."""
    state, _ = env.reset()
    for _, child in env.successors(state):
        assert child.service <= state.service + 1e-6


def test_goal_needs_both_containment_and_service(env):
    """Closing everything contains perfectly and is not a solution."""
    state, _ = env.reset()
    contained_but_dry = WaterNetworkState(state.closed, 0.0, 0.0, 0.0)
    served_but_dirty = WaterNetworkState(state.closed, 1.0, 1.0, 0.0)
    solved = WaterNetworkState(state.closed, 1.0, 0.0, 0.0)
    assert not env.is_goal(contained_but_dry)
    assert not env.is_goal(served_but_dirty)
    assert env.is_goal(solved)


def test_a_collapsed_network_is_terminal_and_absorbing(env):
    state, _ = env.reset()
    collapsed = WaterNetworkState(state.closed, SERVICE_FLOOR / 2, 1.0, 0.0)
    assert env.is_terminal(collapsed)
    assert env.successors(collapsed) == []


def test_a_solved_state_is_absorbing(env):
    state, _ = env.reset()
    solved = WaterNetworkState(state.closed, 1.0, 0.0, 0.0)
    assert env.is_goal(solved)
    assert env.successors(solved) == []


def test_the_fresh_incident_is_neither_solved_nor_lost(env):
    state, _ = env.reset()
    assert not env.is_goal(state) and not env.is_terminal(state)
    assert state.contaminated > CONTAMINATION_GOAL
    assert state.service >= SERVICE_GOAL


# ------------------------------------------------------------------------- solving it

def test_the_easiest_scenario_is_solvable_and_validates(env):
    """Net1 / node 23 solves in two closures. `validate` replays the plan from scratch,
    so it is an independent confirmation rather than a restatement."""
    state, info = env.reset()
    assert info["solved_at"] == 2
    for first, one in env.successors(state):
        for second, two in env.successors(one):
            if env.is_goal(two):
                assert env.validate([first, second])
                return
    pytest.fail("no two-closure solution found, but the scenario records one")


def test_simulate_and_step_track_the_history(env):
    state, _ = env.reset()
    plan = [action for action, _ in env.successors(state)[:2]]
    trace = env.simulate(plan)
    assert len(trace) == 3 and [s.depth for s in trace] == [0, 1, 2]

    env.reset()
    after, cut = env.step(plan[0])
    assert after.closed == {plan[0].pipe}
    assert cut == pytest.approx(state.contaminated - after.contaminated)
    assert len(env.render()) == 2


# ------------------------------------------------------------------------- housekeeping

def test_the_simulator_does_not_litter_the_working_directory(tmp_path, monkeypatch):
    """EpanetSimulator writes temp.inp/.bin/.rpt beside the process unless told otherwise,
    and expansion runs it hundreds of times. It must not do that in a user's repo."""
    monkeypatch.chdir(tmp_path)
    game = WaterNetworkEnv()
    try:
        game.fix_index(0)
        state, _ = game.reset()
        game.successors(state)
    finally:
        game.close()
    assert list(tmp_path.iterdir()) == []


def test_close_removes_the_scratch_directory():
    import os

    game = WaterNetworkEnv()
    game.fix_index(0)
    game.reset()
    workdir = game._workdir
    assert os.path.isdir(workdir)
    game.close()
    assert not os.path.isdir(workdir)
