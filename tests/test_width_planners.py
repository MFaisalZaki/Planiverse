"""Tests for the width-based planners.

Most run against the pure-Python Puzznic, which needs no dependencies and is small enough
that the theory is visible: IW(1) genuinely exhausts on level 1, IW(2) genuinely solves it.
"""
import pytest

from planiverse.environments.gameboy_py.puzznic import PuzznicGame
from planiverse.planners.width import (
    MAX_PRACTICAL_WIDTH, BFWSSearch, Budget, IteratedWidth, IWSearch, NoveltyTable,
    PartitionedNovelty, SIWSearch, SearchResult, SearchStatistics, path_novelty,
)


def boxes(state):
    """Blocks still on the board — a stand-in for the unachieved-goal count a simulator
    cannot provide, since `is_goal` is a black-box predicate."""
    return sum(1 for literal in state.literals if literal.startswith("at(box"))


@pytest.fixture
def env():
    game = PuzznicGame()
    game.fix_index(0)
    return game


# ---------------------------------------------------------------------------- novelty

def test_novelty_is_the_size_of_the_smallest_new_tuple():
    table = NoveltyTable(width=2)
    assert table.evaluate_and_record({"a", "b"}) == 1, "everything is new at first"
    assert table.evaluate_and_record({"a", "b"}) == 3, "nothing new: width + 1"
    assert table.evaluate_and_record({"a", "c"}) == 1, "c is a new atom"
    # Every atom seen, but this pair has not been.
    table.evaluate_and_record({"b", "c"})
    assert table.evaluate_and_record({"a", "b", "c"}) == 3
    assert table.evaluate_and_record({"b", "d"}) == 1


def test_a_state_is_never_novel_with_respect_to_itself():
    """Which is why evaluate and record are one call: recording first always scores
    `width + 1`."""
    table = NoveltyTable(width=1)
    table.record({"a"})
    assert table.evaluate({"a"}) == 2


def test_wide_novelty_is_refused_unless_asked_for_explicitly():
    """Width 3 enumerates every triple of every state's atoms, which against a simulator
    usually costs more than the successors it saves."""
    assert MAX_PRACTICAL_WIDTH == 2
    with pytest.raises(ValueError, match="strict=False"):
        NoveltyTable(width=3)
    assert NoveltyTable(width=3, strict=False).width == 3
    with pytest.raises(ValueError, match="at least 1"):
        NoveltyTable(width=0)


def test_partitioned_novelty_renews_the_budget_per_partition():
    """The reason BFWS partitions at all: plain novelty runs out once every atom has been
    seen, and then the ordering goes flat."""
    partitioned = PartitionedNovelty(width=1)
    assert partitioned.evaluate_and_record("A", {"x"}) == 1
    assert partitioned.evaluate_and_record("A", {"x"}) == 2, "not novel in A any more"
    assert partitioned.evaluate_and_record("B", {"x"}) == 1, "but brand new in B"
    assert len(partitioned.tables) == 2


def test_the_reference_path_rule_differs_from_standard_novelty_above_width_one():
    """`pyBehaviourPlanningLTL` counts atoms new *along the path*, and its own source flags
    the rule as unverified. At width 1 the two agree; above it they ask different questions
    — two new atoms versus one new pair."""
    assert path_novelty({"a", "b"}, {"a"}) == 1
    assert path_novelty({"a", "b"}, {"a", "b"}) == 0
    assert path_novelty({"a", "b", "c"}, {"a"}) == 2

    # Standard novelty calls this state novel at 2 (the pair is new); the path rule counts
    # zero new atoms and, at width 2, rejects it.
    table = NoveltyTable(width=2)
    table.evaluate_and_record({"a", "b"})
    table.evaluate_and_record({"c", "d"})
    assert table.evaluate({"a", "c"}) == 2
    assert path_novelty({"a", "c"}, {"a", "b", "c", "d"}) == 0


# ------------------------------------------------------------------------- IW and kin

def test_iw_one_exhausts_where_iw_two_solves(env):
    """IW(k) is incomplete for fixed k, and Puzznic level 1 shows it rather than merely
    asserting it: width 1 runs out of states, width 2 finds a plan."""
    narrow = IWSearch(width=1).solve(env, Budget(max_expansions=5000))
    assert narrow.status == "exhausted"
    assert not narrow.solved and narrow.plan is None

    wider = IWSearch(width=2).solve(env, Budget(max_expansions=5000))
    assert wider.solved and wider.width == 2
    assert env.validate(wider.plan), "and the plan replays to a goal"


def test_iterated_width_finds_the_width_it_needs(env):
    """And charges for the widths it tried on the way — each restarts from scratch."""
    result = IteratedWidth(max_width=2).solve(env, Budget(max_expansions=5000))
    assert result.solved and result.width == 2
    assert result.statistics.widths_tried == (1, 2)
    assert env.validate(result.plan)

    single = IWSearch(width=2).solve(env, Budget(max_expansions=5000))
    assert result.statistics.expansions > single.statistics.expansions, \
        "the failed width-1 round is not free"


def test_iterated_width_gives_up_when_its_ceiling_is_too_low(env):
    result = IteratedWidth(max_width=1).solve(env, Budget(max_expansions=5000))
    assert not result.solved
    assert result.statistics.widths_tried == (1,)


class _Step:
    def __init__(self, number):
        self.number = number
        self.literals = frozenset({f"at({number})"})


class _Chain:
    """Three states in a line, no goal. IW(1) covers it without pruning for novelty."""

    def fix_index(self, index): pass

    def reset(self): return _Step(0), {}

    def successors(self, state):
        return [] if state.number == 2 else [("go", _Step(state.number + 1))]

    def is_goal(self, state): return False

    def is_terminal(self, state): return False

    def simulate(self, plan): return [_Step(i) for i in range(len(plan) + 1)]


def test_a_huge_width_bound_costs_nothing_when_the_space_is_already_covered():
    """`max_width` is a bound, not a plan. If a width exhausts the reachable space without
    discarding anything for novelty, no larger width can reach further — so iterating on
    would re-run the identical search a thousand times."""
    result = IteratedWidth(max_width=1000, strict=False).solve(_Chain(), Budget())
    assert result.statistics.widths_tried == (1,), "it stopped after the first"
    assert result.status == "exhausted", "and says the space really was covered"


def test_exhausting_the_space_is_reported_as_such_not_as_a_plain_failure():
    """`SearchResult.__bool__` is `solved`, so `last.status if last else ...` silently
    reported "failed" for every unsolved outcome — including a proof that there is no plan."""
    assert IteratedWidth(max_width=3, strict=False).solve(_Chain(), Budget()).status \
        == "exhausted"


def test_running_out_of_budget_is_not_mistaken_for_covering_the_space():
    result = IteratedWidth(max_width=1000, strict=False).solve(
        _Chain(), Budget(max_expansions=1))
    assert result.status == "out_of_budget"


def test_a_width_strict_refuses_stops_the_iteration_rather_than_raising(env):
    """The widths already tried are a real result; an exception would throw them away."""
    result = IteratedWidth(max_width=9, strict=True).solve(env, Budget(max_expansions=5000))
    assert result.solved or result.statistics.widths_tried == (1, 2)


def test_siw_can_iterate_a_legs_width_instead_of_pinning_it(env):
    """Each SIW leg *is* an IW search, so SIW inherits IW's width sensitivity exactly: a leg
    that finds no progress within IW(k)'s pruned reach fails, and the whole search fails,
    even though a wider leg would have found some."""
    def boxes(state):
        return sum(1 for literal in state.literals if literal.startswith("at(box"))

    pinned = SIWSearch(width=2, progress=boxes).solve(env, Budget(max_expansions=5000))
    assert pinned.statistics.widths_tried == (2,), "unchanged without max_width"

    iterated = SIWSearch(width=1, max_width=1000, strict=False, progress=boxes).solve(
        env, Budget(max_expansions=5000))
    assert iterated.solved
    assert iterated.statistics.widths_tried == (1, 2), \
        "the widths the legs actually needed, which pinning hid"
    assert iterated.width == 2, "the hardest leg is what the problem cost"


def test_a_max_width_below_the_starting_width_is_refused():
    with pytest.raises(ValueError, match="max_width"):
        SIWSearch(width=3, max_width=2)


def test_a_bound_below_one_is_refused():
    with pytest.raises(ValueError, match="max_width"):
        IteratedWidth(max_width=0)


def test_novelty_stops_at_the_number_of_atoms_a_state_has():
    """A tuple longer than the state has atoms does not exist, so the work a huge width costs
    is bounded by the state, not by the width. Without that a bound of 1000 meant a thousand
    empty combination ranges per record and a thousand allocated levels per table."""
    import time

    atoms = frozenset({"a", "b", "c"})
    table = NoveltyTable(width=1000, strict=False)
    table.evaluate_and_record(atoms)
    seen_after_first = table.tuples_enumerated
    table.evaluate_and_record(atoms)
    assert table.tuples_enumerated - seen_after_first == 7, \
        "the whole power set of three atoms, and nothing above size three"
    assert len(table) == 7

    started = time.perf_counter()
    huge = NoveltyTable(width=10 ** 6, strict=False)
    huge.evaluate_and_record(atoms)
    assert time.perf_counter() - started < 1.0, "a million levels must not be allocated"
    assert huge.evaluate(atoms) == 10 ** 6 + 1, "and the answer is unchanged"


def test_a_huge_width_scores_the_same_as_one_the_size_of_the_state():
    """Widths above the atom count are the same search, so they had better agree."""
    atoms = [frozenset({"a", "b"}), frozenset({"b", "c"}), frozenset({"a", "b"})]
    small = NoveltyTable(width=2, strict=False)
    large = NoveltyTable(width=500, strict=False)
    for state in atoms:
        assert (small.evaluate_and_record(state) <= 2) == \
               (large.evaluate_and_record(state) <= 500)


def test_the_plan_and_trace_line_up(env):
    result = IWSearch(width=2).solve(env, Budget(max_expansions=5000))
    assert len(result.states) == len(result.plan) + 1, "the trace is one longer"
    assert env.is_goal(result.states[-1])
    assert list(result) == result.plan
    assert bool(result) is True
    assert result.cost == sum(a.cost() for a in result.plan) if hasattr(
        result.plan[0], "cost") else True


def test_a_budget_stops_the_search_and_says_so(env):
    """Against an expensive simulator, "found nothing" and "ran out after four nodes" are
    different answers and a planner that conflates them is not usable."""
    result = IWSearch(width=2).solve(env, Budget(max_expansions=3))
    assert result.status == "out_of_budget"
    assert result.statistics.expansions <= 3
    assert not result.solved
    assert "out_of_budget" in str(result)


def test_a_time_budget_also_stops_it(env):
    result = IWSearch(width=2).solve(env, Budget(max_seconds=0.0))
    assert result.status == "out_of_budget"
    assert result.statistics.expansions == 0


def test_an_already_solved_state_needs_no_plan(env):
    """Degenerate but real: `reset` can hand back a goal."""
    class Solved(PuzznicGame):
        def is_goal(self, state):
            return True

    game = Solved()
    game.fix_index(0)
    result = IWSearch(width=1).solve(game)
    assert result.solved and result.plan == []
    assert len(result.states) == 1


def test_novelty_pruning_actually_prunes(env):
    """If it never fired, IW would just be breadth-first search."""
    result = IWSearch(width=1).solve(env, Budget(max_expansions=5000))
    assert result.statistics.pruned_novelty > 0
    assert result.statistics.generated > result.statistics.expansions


def test_the_path_rule_is_selectable_and_validated():
    with pytest.raises(ValueError, match="novelty_rule"):
        IWSearch(novelty_rule="nonsense")
    assert IWSearch(novelty_rule="path").novelty_rule == "path"


# ------------------------------------------------------------------------------- SIW

def test_siw_without_a_progress_measure_degrades_to_iw_and_says_so(env):
    """A simulator has no goal conjunction to count, so SIW has nothing to serialise on.
    It should say that rather than quietly behaving like something else."""
    result = SIWSearch(width=1).solve(env, Budget(max_expansions=5000))
    assert not result.solved
    assert "degraded to IW" in result.status


def test_siw_walks_into_a_dead_end_unless_told_not_to(env):
    """SIW is incomplete because each leg commits irrevocably to the first improvement it
    finds, and here that trap is real: the first leg clears a pair and leaves one block of
    a colour behind, which can never be matched.

    A simulator that computes `is_terminal` lets the leg refuse — a dead end is not
    progress — and that alone turns a failure into a solved instance.
    """
    classical = SIWSearch(width=2, progress=boxes, avoid_dead_ends=False)
    assert not classical.solve(env, Budget(max_expansions=5000)).solved

    careful = SIWSearch(width=2, progress=boxes, avoid_dead_ends=True)
    result = careful.solve(env, Budget(max_expansions=5000))
    assert result.solved, "refusing dead-end progress solves it"
    assert env.validate(result.plan)


def test_siw_reports_the_widths_it_used(env):
    result = SIWSearch(width=2, progress=boxes).solve(env, Budget(max_expansions=5000))
    assert result.statistics.widths_tried == (2,)
    assert result.statistics.expansions > 0


# ------------------------------------------------------------------------------ BFWS

def test_bfws_solves_what_iw_one_cannot(env):
    """Novelty as an ordering rather than a filter: nothing is discarded, so width 1 is
    still complete where IW(1) exhausted."""
    result = BFWSSearch(width=1, progress=boxes).solve(env, Budget(max_expansions=5000))
    assert result.solved
    assert env.validate(result.plan)


def test_bfws_never_prunes_for_novelty(env):
    """The defining difference from IW. If this counter ever moves, novelty has become a
    filter again and completeness is gone."""
    result = BFWSSearch(width=1, progress=boxes).solve(env, Budget(max_expansions=5000))
    assert result.statistics.pruned_novelty == 0


def test_bfws_works_with_no_callbacks_at_all(env):
    """Degrades to breadth-first-with-a-preference-for-novel-states. Weaker, not broken —
    it has no idea which way the goal is."""
    result = BFWSSearch(width=1).solve(env, Budget(max_expansions=5000))
    assert result.status in ("solved", "exhausted", "out_of_budget")
    if result.solved:
        assert env.validate(result.plan)


def test_a_heuristic_breaks_ties_without_changing_soundness(env):
    result = BFWSSearch(width=1, progress=boxes,
                        heuristic=lambda s: len(s.literals)).solve(
        env, Budget(max_expansions=5000))
    assert result.solved
    assert env.validate(result.plan)


def test_the_progress_measure_partitions_novelty_by_default(env):
    """The classical choice, and the one that keeps exploration renewing."""
    search = BFWSSearch(width=1, progress=boxes)
    state, _ = env.reset()
    assert search.__partition_of__(state) == boxes(state)

    explicit = BFWSSearch(width=1, progress=boxes, partition=lambda s: "one")
    assert explicit.__partition_of__(state) == "one"

    bare = BFWSSearch(width=1)
    assert bare.__partition_of__(state) == 0, "a single global partition"


# ------------------------------------------------------------------------- statistics

def test_statistics_merge_for_the_iterated_searches():
    first = SearchStatistics(expansions=3, generated=9, widths_tried=(1,))
    second = SearchStatistics(expansions=4, generated=11, widths_tried=(2,))
    merged = first.merge(second)
    assert merged.expansions == 7 and merged.generated == 20
    assert merged.widths_tried == (1, 2)


def test_an_unsolved_result_is_falsey_and_explains_itself():
    result = SearchResult(status="exhausted")
    assert not result and len(result) == 0 and result.cost == 0
    assert list(result) == []
    assert "exhausted" in str(result)


# ------------------------------------------------- against the simulator environments

def test_iw_solves_the_water_network():
    """A real simulator: every expansion is a hydraulic and transport solve."""
    pytest.importorskip("wntr", reason="wntr is not installed")
    from planiverse.environments.water_network.environment import WaterNetworkEnv

    game = WaterNetworkEnv()
    try:
        game.fix_index(0)                     # solvable at depth 2
        result = IWSearch(width=1).solve(game, Budget(max_expansions=200, max_seconds=120))
        assert result.solved, f"expected a plan, got {result.status}"
        assert game.validate(result.plan)
        assert result.statistics.pruned_novelty > 0, "novelty should be doing work here"
    finally:
        game.close()


@pytest.mark.slow
def test_bfws_solves_a_growing_season():
    """Fixed depth 10 and the objective only observable at the leaves."""
    pytest.importorskip("pcse", reason="pcse is not installed")
    from planiverse.environments.crop_management.environment import CropEnv

    game = CropEnv()
    try:
        game.fix_index(10)                    # 1986: irrigation is worth 2698 kg/ha
        result = BFWSSearch(width=1, progress=lambda s: -s.biomass).solve(
            game, Budget(max_expansions=400, max_seconds=240))
        assert result.solved, f"expected a plan, got {result.status}"
        assert game.validate(result.plan)
        assert len(result.plan) == 10, "a season is ten decisions whatever else happens"
    finally:
        game.close()
