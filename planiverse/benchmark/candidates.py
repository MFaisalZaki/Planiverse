"""The planners added after the tool paper, as benchmark configurations.

None of these is part of the paper's protocol, so none runs unless `generate` and `report`
are given `--candidates`. The tags are what `solve` takes and what the result directories
are named. Every configuration takes `progress` from `measures.py` the way the paper's
planners do; the planners that need a projection (`GoExplore`'s cell, `MAPElitesPlanner`'s
descriptor, `KinodynamicTree`'s projection) run on their default, the exact state, and
`FeatureSpaceSearch` gets the progress measure as its one feature through the adapter
below. Boundary-extension features, multi-queue alternation and the macro-action and
pruning add-ons need per-environment callbacks the benchmark does not carry, so they are
not registered.

Parameters are the classes' defaults except where the paper's planners fix the same
thing: 500,000 expansions bound the approximate-novelty policy, and the per-decision
budgets of the online planners match Rollout IW's 1,000 expansions.
"""
from planiverse.planners.blind import BreadthFirstSearch, IterativeDeepening, UniformCostSearch
from planiverse.planners.heuristic import (
    BULB, BeamSearch, BestFirstSearch, DiverseBestFirst, EnforcedHillClimbing,
    EpsilonGreedySearch, FeatureSpaceSearch, IterativeBroadening, LRTAStar,
    LimitedDiscrepancySearch, LocalExplorationSearch, MonteCarloRandomWalks, RTAAStar,
    RestartingWeightedAStar, TypeBasedSearch,
)
from planiverse.planners.macros import MacroPlanner
from planiverse.planners.sampling import (
    CrossEntropyPlanner, FractalMonteCarlo, GoExplore, KinodynamicTree, MAPElitesPlanner,
    NestedMonteCarloSearch, PlanLocalSearch, RandomShooting, RollingHorizonEvolution,
    RolloutPlanner,
)
from planiverse.planners.width import (
    BFWSR, ApproximateNoveltySearch, CountNoveltySearch, HierarchicalIW, PrioritizedIW,
    QuantifiedNoveltySearch, TwoBFS,
)


class FESSOnProgress(FeatureSpaceSearch):
    """FESS whose one feature is the environment's progress measure."""

    def __init__(self, progress):
        super().__init__(features=lambda state: (progress(state),))


CANDIDATES = {
    # width-based variants
    "2bfs": (TwoBFS, {"width": 1}),
    "piw": (PrioritizedIW, {"width": 1}),
    "bfwsr": (BFWSR, {"width": 1}),
    "qn": (QuantifiedNoveltySearch, {}),
    "cbn": (CountNoveltySearch, {"width": 1}),
    "ans": (ApproximateNoveltySearch, {"width": 2, "space_bound": 500_000}),
    "hiw": (HierarchicalIW, {"low_expansions": 1000}),
    # heuristic search
    "gbfs": (BestFirstSearch, {}),
    "rwa": (RestartingWeightedAStar, {}),
    "egbfs": (EpsilonGreedySearch, {"epsilon": 0.2}),
    "tgbfs": (TypeBasedSearch, {}),
    "dbfs": (DiverseBestFirst, {}),
    "gbfsle": (LocalExplorationSearch, {}),
    "ehc": (EnforcedHillClimbing, {}),
    "mrw": (MonteCarloRandomWalks, {}),
    "beam": (BeamSearch, {"beam": 100}),
    "bulb": (BULB, {"beam": 20, "max_depth": 200}),
    "lds": (LimitedDiscrepancySearch, {"max_depth": 60}),
    "ib": (IterativeBroadening, {"max_depth": 60}),
    "lrta": (LRTAStar, {"lookahead": 50, "max_steps": 2000}),
    "rtaa": (RTAAStar, {"lookahead": 50, "max_steps": 2000}),
    "fess": (FESSOnProgress, {}),
    # sampling
    "rhea": (RollingHorizonEvolution, {"expansions_per_step": 1000}),
    "cem": (CrossEntropyPlanner, {}),
    "shoot": (RandomShooting, {}),
    "roll": (RolloutPlanner, {}),
    "nmcs": (NestedMonteCarloSearch, {}),
    "fmc": (FractalMonteCarlo, {}),
    "goexp": (GoExplore, {}),
    "mapel": (MAPElitesPlanner, {}),
    "est": (KinodynamicTree, {"strategy": "est"}),
    "sst": (KinodynamicTree, {"strategy": "sst"}),
    "sa": (PlanLocalSearch, {}),
    "macro": (MacroPlanner, {}),
    # blind
    "brfs": (BreadthFirstSearch, {}),
    "ucs": (UniformCostSearch, {}),
    "iddfs": (IterativeDeepening, {"max_depth": 200}),
}
