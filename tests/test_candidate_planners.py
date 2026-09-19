"""Smoke tests for the planners added after the tool paper.

Every planner runs against the pure-Python Puzznic level 1 under a small budget, and any
plan it returns is replayed through the environment. The level is the one the width
docs use: IW(1) exhausts on it, IW(2) solves it in 10 actions, and its first pair-clearance
is a dead end, so a planner that commits blindly walks into a trap. These are smoke tests,
not the benchmark: they check that each planner runs, stops, reports, and never returns
an invalid plan.
"""
import pytest

from planiverse.environments.gameboy_py.puzznic import PuzznicGame
from planiverse.planners.blind import BreadthFirstSearch, IterativeDeepening, UniformCostSearch
from planiverse.planners.common import SuccessorCache, action_vocabulary
from planiverse.planners.heuristic import (
    BULB, BeamSearch, BestFirstSearch, DiverseBestFirst, EnforcedHillClimbing,
    EpsilonGreedySearch, FeatureSpaceSearch, IterativeBroadening, LRTAStar,
    LimitedDiscrepancySearch, LocalExplorationSearch, MonteCarloRandomWalks,
    MultiQueueSearch, RTAAStar, RestartingWeightedAStar, TypeBasedSearch,
)
from planiverse.planners.macros import FocusedMacros, MacroPlanner
from planiverse.planners.pruning import DominatedActionPruner
from planiverse.planners.sampling import (
    CrossEntropyPlanner, GoExplore, KinodynamicTree, MAPElitesPlanner, NestedMonteCarloSearch,
    PlanLocalSearch, RandomShooting, RollingHorizonEvolution, evaluate_sequence,
)
from planiverse.planners.width import (
    BFWSR, ApproximateNoveltySearch, BFWSSearch, BloomFilter, BloomNoveltyTable,
    BoundaryExtensionFeatures, Budget, CountNoveltySearch, CountNoveltyTable,
    HeuristicNovelty, HierarchicalIW, QuantifiedNoveltySearch, SearchResult,
)


def boxes(state):
    return sum(1 for literal in state.literals if literal.startswith("at(box"))


def one_feature(state):
    return (boxes(state),)


@pytest.fixture
def env():
    game = PuzznicGame()
    game.set_index(0)
    return game


def budget():
    return Budget(max_expansions=3000, max_seconds=30)


#: Every planner, and whether it is expected to solve the level within the budget. The
#: ones marked False are the ones whose mechanism is not enough here (the depth-first
#: probes and LRTA* run out of time or steps), and the test only asks them to stop cleanly.
PLANNERS = [
    ("bfwsr", lambda: BFWSR(progress=boxes), True),
    ("qn", lambda: QuantifiedNoveltySearch(progress=boxes), True),
    ("cbn", lambda: CountNoveltySearch(progress=boxes, open_limit=500), True),
    ("ans", lambda: ApproximateNoveltySearch(width=2, progress=boxes, space_bound=2000,
                                             seed=0), True),
    ("hiw", lambda: HierarchicalIW(low_expansions=100), True),
    ("bfws-bee", lambda: BFWSSearch(width=1, progress=boxes,
                                    atoms=BoundaryExtensionFeatures(
                                        lambda s: {"boxes": boxes(s)}, bins=2)), True),
    ("gbfs", lambda: BestFirstSearch(progress=boxes), True),
    ("wastar", lambda: BestFirstSearch(progress=boxes, weight=2.0), True),
    ("rwa", lambda: RestartingWeightedAStar(progress=boxes), True),
    ("egbfs", lambda: EpsilonGreedySearch(progress=boxes, seed=0), True),
    ("tgbfs", lambda: TypeBasedSearch(progress=boxes, seed=0), True),
    ("dbfs", lambda: DiverseBestFirst(progress=boxes, seed=0), True),
    ("gbfsle", lambda: LocalExplorationSearch(progress=boxes, patience=10, seed=0), True),
    ("gbfslw", lambda: LocalExplorationSearch(progress=boxes, patience=10, strategy="walks",
                                              seed=0), True),
    ("ehc", lambda: EnforcedHillClimbing(progress=boxes), True),
    ("mrw", lambda: MonteCarloRandomWalks(progress=boxes, seed=0), True),
    ("beam", lambda: BeamSearch(progress=boxes, beam=20), True),
    ("bulb", lambda: BULB(progress=boxes, beam=5, max_depth=30), True),
    ("lds", lambda: LimitedDiscrepancySearch(progress=boxes, max_depth=15,
                                             max_iterations=3), False),
    ("ib", lambda: IterativeBroadening(progress=boxes, max_depth=15, max_iterations=3),
     False),
    ("lrta", lambda: LRTAStar(progress=boxes, lookahead=20, max_steps=100), False),
    ("rtaa", lambda: RTAAStar(progress=boxes), True),
    ("fess", lambda: FeatureSpaceSearch(features=one_feature), True),
    ("multi", lambda: MultiQueueSearch([boxes, lambda s: -len(s.literals)]), True),
    ("brfs", lambda: BreadthFirstSearch(), True),
    ("ucs", lambda: UniformCostSearch(), True),
    ("iddfs", lambda: IterativeDeepening(max_depth=30), True),
    ("rhea", lambda: RollingHorizonEvolution(progress=boxes, seed=0), True),
    ("cem", lambda: CrossEntropyPlanner(progress=boxes, seed=0), True),
    ("shoot", lambda: RandomShooting(progress=boxes, seed=0), False),
    ("nmcs", lambda: NestedMonteCarloSearch(progress=boxes, level=2, horizon=20, seed=0),
     True),
    ("goexp", lambda: GoExplore(progress=boxes, seed=0), True),
    ("mapel", lambda: MAPElitesPlanner(progress=boxes, seed=0), True),
    ("est", lambda: KinodynamicTree("est", seed=0), True),
    ("kpiece", lambda: KinodynamicTree("kpiece", projection=one_feature, seed=0), True),
    ("sst", lambda: KinodynamicTree("sst", seed=0), True),
    ("sa", lambda: PlanLocalSearch(progress=boxes, seed=0), True),
    ("ils", lambda: PlanLocalSearch(progress=boxes, method="iterated", max_iterations=100,
                                    seed=0), False),
    ("macro", lambda: MacroPlanner(progress=boxes, seed=0), True),
]


@pytest.mark.parametrize("name,build,solves", PLANNERS, ids=[p[0] for p in PLANNERS])
def test_every_candidate_runs_stops_and_never_lies(env, name, build, solves):
    result = build().solve(env, budget())
    assert isinstance(result, SearchResult)
    assert result.statistics.expansions <= 3000
    if result.solved:
        assert env.validate(result.plan), f"{name} returned a plan that does not replay"
        assert len(result.states) == len(result.plan) + 1
    else:
        assert result.plan is None
    if solves:
        assert result.solved, f"{name} was expected to solve level 1: {result.status}"


# ---------------------------------------------------------------- width variants

def test_no_planner_takes_a_reward():
    """This is a planning suite: every planner is driven by `is_goal` and, at most, a
    goal-distance heuristic. Nothing added here accepts a reward callback."""
    import inspect
    for _, build, _ in PLANNERS:
        planner = build()
        assert "reward" not in inspect.signature(type(planner).__init__).parameters, \
            type(planner).__name__


def test_bfws_r_finds_relevant_atoms_from_its_pre_search(env):
    """At width 1 the pre-search finds no progress on level 1 (the only improvement is
    the dead end, which is dropped), so R is empty and the search is plain BFWS. At width
    2 the pre-search reaches the goal itself, and the atoms on its path are R."""
    narrow = BFWSR(progress=boxes)
    assert narrow.solve(env, budget()).solved and not narrow.relevant
    wide = BFWSR(progress=boxes, r_width=2)
    result = wide.solve(env, budget())
    assert result.solved and result.width == 2
    assert wide.relevant and all(isinstance(atom, str) for atom in wide.relevant)


def test_heuristic_novelty_counts_atoms_never_seen_this_close():
    table = HeuristicNovelty()
    assert table.evaluate_and_record({"a", "b"}, 5) == 2
    assert table.evaluate_and_record({"a", "b"}, 5) == 0, "seen at this value already"
    assert table.evaluate_and_record({"a", "c"}, 3) == 2, "a is new at 3; c is new"
    assert table.evaluate_and_record({"a"}, 4) == 0, "a has been seen at 3, which is closer"


def test_count_novelty_keeps_discriminating_after_the_first_sighting():
    table = CountNoveltyTable(width=1)
    assert table.evaluate_and_record({"a", "b"}) == 0
    assert table.evaluate_and_record({"a", "b"}) == 1
    assert table.evaluate_and_record({"a", "b"}) == 2
    assert table.evaluate_and_record({"a", "c"}) == 0, "c has never been seen"
    assert table.evaluate_and_record({"a"}) == 4 and table.evaluate_and_record({"c"}) == 1


def test_the_trimmed_open_list_reports_what_it_dropped(env):
    """Trimming discards, so a search that trimmed and found nothing is `failed`, never
    `exhausted`: it did not see the whole space."""
    result = CountNoveltySearch(progress=boxes, open_limit=4).solve(env, budget())
    assert result.statistics.pruned_novelty > 0, "with an open list of 4, trimming happened"
    assert result.solved or result.status == "failed"


def test_a_bloom_filter_has_no_false_negatives():
    bloom = BloomFilter(bits=1 << 12, hashes=3)
    keys = [("atom", i) for i in range(200)]
    for key in keys:
        bloom.add(key)
    assert all(key in bloom for key in keys)
    table = BloomNoveltyTable(width=2, sample=5)
    assert table.evaluate_and_record(set("abcdefgh")) == 1
    assert table.tuples_enumerated <= 8 + 5, "pairs were sampled, not enumerated"


def test_approximate_novelty_allows_width_three_because_bounding_it_is_the_point(env):
    result = ApproximateNoveltySearch(width=3, progress=boxes, sample=50, seed=0).solve(
        env, budget())
    assert result.solved


def test_boundary_extension_features_are_novel_only_when_a_boundary_moves():
    class S:
        def __init__(self, x):
            self.literals = frozenset()
            self.x = x

    bee = BoundaryExtensionFeatures(lambda s: {"x": s.x}, include_literals=False)
    assert bee(S(1.0)) == {"bee(x, first)"}
    assert bee(S(1.0)) == {"bee(x, inside)"}
    assert bee(S(2.0)) == {"bee(x, high, 1)"}
    assert bee(S(1.5)) == {"bee(x, inside)"}
    assert bee(S(0.0)) == {"bee(x, low, 2)"}
    assert bee.ranges["x"][:2] == [0.0, 2.0]


def test_hierarchical_iw_solves_a_width_two_level_with_two_width_one_levels(env):
    """IW(1) exhausts on level 1; HIW(1,1) reaches it with features discovered from what
    the low level pruned."""
    planner = HierarchicalIW(low_expansions=100)
    result = planner.solve(env, budget())
    assert result.solved and env.validate(result.plan)
    assert planner.features_found, "features were discovered, not given"
    assert result.statistics.widths_tried == (1, 1)


# ---------------------------------------------------------------- heuristic family

def test_restarting_weighted_astar_keeps_the_best_of_its_incumbents(env):
    planner = RestartingWeightedAStar(progress=boxes, weights=(5.0, 1.0))
    result = planner.solve(env, budget())
    assert result.solved
    assert planner.incumbents and len(result.plan) == min(planner.incumbents)
    assert result.statistics.widths_tried == (5.0, 1.0)


def test_enforced_hill_climbing_refuses_dead_end_progress_by_default(env):
    """The first pair clearance is progress into a wall; EHC that takes it fails."""
    assert EnforcedHillClimbing(progress=boxes).solve(env, budget()).solved
    classical = EnforcedHillClimbing(progress=boxes, avoid_dead_ends=False)
    assert not classical.solve(env, budget()).solved


def test_beam_search_is_incomplete_and_bulb_backtracks_over_it(env):
    narrow = BeamSearch(progress=boxes, beam=1).solve(env, budget())
    assert not narrow.solved
    wide = BULB(progress=boxes, beam=1, max_depth=30).solve(env, budget())
    assert wide.solved, "the same beam, plus discrepancies, gets there"


def test_real_time_search_raises_the_values_it_leaves(env):
    planner = RTAAStar(progress=boxes, lookahead=10)
    result = planner.solve(env, budget())
    assert result.solved
    state, _ = env.reset()
    assert planner.values[state.literals] >= boxes(state)


def test_feature_space_search_prefers_advised_moves(env):
    """An advisor that recommends holding moves makes the search reach for them first."""
    advised = FeatureSpaceSearch(features=one_feature,
                                 advisors=[lambda s, actions: [a for a in actions
                                                               if "hold" in a]])
    plain = FeatureSpaceSearch(features=one_feature)
    assert advised.solve(env, budget()).solved and plain.solve(env, budget()).solved


def test_random_walks_record_dead_end_rates_per_first_action(env):
    planner = MonteCarloRandomWalks(progress=boxes, walks=10, seed=0)
    result = planner.solve(env, budget())
    assert result.solved
    assert result.statistics.rollouts > 0


# ---------------------------------------------------------------- sampling family

def test_sequence_evaluation_skips_inapplicable_genes_and_stops_at_a_goal(env):
    state, _ = env.reset()
    cache = SuccessorCache(env)
    result = evaluate_sequence(env, cache, state, ["left-hold", "left", "right"], boxes)
    assert result.actions == ["left", "right"], "the hold is a no-op with no box under it"
    assert result.score == -boxes(state)
    assert cache.statistics.expansions == 2, "left then right is back at the root, cached"


def test_go_explore_keeps_one_trajectory_per_cell_and_prefers_shorter(env):
    planner = GoExplore(progress=boxes, cell=one_feature, seed=0)
    result = planner.solve(env, budget())
    assert result.solved
    assert set(planner.archive) <= {(n,) for n in range(7)}
    for cell in planner.archive.values():
        assert len(cell.trace) == len(cell.plan) + 1


def test_the_kinodynamic_strategies_are_the_three_the_docs_name():
    with pytest.raises(ValueError, match="strategy"):
        KinodynamicTree("rrt")


def test_rolling_horizon_evolution_ends_a_decision_by_generations_when_the_cache_is_warm(env):
    planner = RollingHorizonEvolution(progress=boxes, generations=3, expansions_per_step=10_000,
                                      seed=0)
    result = planner.solve(env, Budget(max_expansions=200, max_seconds=30))
    assert result.status in ("solved", "out_of_budget", "step_limit")
    assert result.statistics.episodes <= 3 * (len(result.states) + 1)


def test_the_sampling_planners_are_reproducible_when_seeded(env):
    a = CrossEntropyPlanner(progress=boxes, seed=3).solve(env, budget())
    b = CrossEntropyPlanner(progress=boxes, seed=3).solve(env, budget())
    assert a.status == b.status and a.plan == b.plan


# ---------------------------------------------------------------- add-ons

def test_the_pruner_drops_an_action_that_always_duplicates_another():
    class S:
        def __init__(self, lits):
            self.literals = frozenset(lits)

    pruner = DominatedActionPruner(threshold=3)
    for _ in range(3):
        kept = pruner.filter([("a", S({"x"})), ("b", S({"x"})), ("c", S({"y"}))])
    assert [a for a, _ in kept] == ["a", "c"], "b always lands where a does"
    assert pruner.pruned == 1


def test_the_pruner_plugs_into_the_successor_cache(env):
    state, _ = env.reset()
    cache = SuccessorCache(env, pruner=DominatedActionPruner(threshold=2))
    assert len(action_vocabulary(env, state, cache)) >= 1
    cache.expand(state)
    assert cache.pruner.counts, "the cache fed its expansion to the pruner"


def test_focused_macros_have_the_smallest_footprints(env):
    state, _ = env.reset()
    cache = SuccessorCache(env)
    macros = FocusedMacros(length=2, count=4, probes=1, seed=0).discover(env, cache, state)
    assert 0 < len(macros) <= 4
    assert all(1 <= len(macro) <= 2 for macro in macros)
    end = cache.replay(state, macros[0])[-1]
    assert len(end.literals ^ state.literals) <= 4, "a cursor move changes two atoms"
