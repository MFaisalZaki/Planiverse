"""The surveyed planners, as benchmark configurations.

`generate` and `report` run every planner the library has: the four reference configurations
in `PLANNERS` and these, under `--reference` the four alone. The tags are what `solve` takes
and what the result directories are named. Every configuration takes `progress` from
`measures.py` the way the reference planners do, and the planners that want more than one
number from an environment get it from that same measure through the adapters below: FESS
searches the feature space it spans, KPIECE projects onto it, boundary-extension features
extend its range, and multi-queue alternation pairs it with FSX's goal-free option count.
The planners that take a projection and run without one (`GoExplore`'s cell,
`MAPElitesPlanner`'s descriptor, `KinodynamicTree`'s EST and SST) run on their default, the
exact state. The two add-ons, `DominatedActionPruner` and `FocusedMacros`, are not planners:
the second is what `MacroPlanner` runs on, and the first is a `SuccessorCache` option no
configuration here sets.

No configuration takes a reward: every planner here is driven by `is_goal` and the
progress heuristic, and the surveyed planners defined by an accumulated reward (2BFS,
prioritised IW, Fractal Monte Carlo, the rollout algorithm) were left out of the library.

Parameters are the classes' defaults, except that the protocol's 500,000 expansions bound
the approximate-novelty policy and the online planners get 1,000 expansions per decision.
"""
from planiverse.planners.blind import BreadthFirstSearch, IterativeDeepening, UniformCostSearch
from planiverse.planners.fsx import option_count
from planiverse.planners.heuristic import (
    BULB, BeamSearch, BestFirstSearch, DiverseBestFirst, EnforcedHillClimbing,
    EpsilonGreedySearch, FeatureSpaceSearch, IterativeBroadening, LRTAStar,
    LimitedDiscrepancySearch, LocalExplorationSearch, MonteCarloRandomWalks, MultiQueueSearch,
    RTAAStar, RestartingWeightedAStar, TypeBasedSearch,
)
from planiverse.planners.macros import MacroPlanner
from planiverse.planners.sampling import (
    CrossEntropyPlanner, GoExplore, KinodynamicTree, MAPElitesPlanner, NestedMonteCarloSearch,
    PlanLocalSearch, RandomShooting, RollingHorizonEvolution,
)
from planiverse.planners.width import (
    BFWS, ApproximateNoveltySearch, BFNoS, BoundaryExtensionFeatures, DualBFWS, HierarchicalIW,
    QuantifiedNoveltySearch,
)


class FESSOnProgress(FeatureSpaceSearch):
    """FESS whose one feature is the environment's progress measure."""

    def __init__(self, progress):
        super().__init__(features=lambda state: (progress(state),))


class KPIECEOnProgress(KinodynamicTree):
    """KPIECE whose projection is the environment's progress measure.

    KPIECE's grid is over a low-dimensional projection; on the exact state (the default
    EST and SST run on) every cell holds one state and the grid does nothing.
    """

    def __init__(self, progress, seed=None):
        super().__init__("kpiece", projection=lambda state: (progress(state),), seed=seed)


class BFWSOverBoundaryExtensions(BFWS):
    """BFWS whose atoms are boundary extensions of the progress measure, over the literals.

    `BoundaryExtensionFeatures` wants the environment's continuous variables, which the
    contract does not expose; the progress measure is the one number the benchmark has for
    every environment, so its range is what the search extends.
    """

    def __init__(self, progress, width=1, bins=4):
        super().__init__(width=width, progress=progress,
                         atoms=BoundaryExtensionFeatures(
                             lambda state: {"progress": progress(state)}, bins=bins))


class ProgressAndOptionsQueues(MultiQueueSearch):
    """Multi-queue alternation between the progress measure and FSX's option count.

    The option count needs the environment, so the queues are set when `solve` is given
    one. Four walkers of four steps per state keep it to sixteen simulator steps a node.
    """

    def __init__(self, progress, horizon=4, walkers=4, boost=0):
        super().__init__([progress], boost=boost)
        self.progress = progress
        self.horizon = horizon
        self.walkers = walkers

    def solve(self, env, budget=None, state=None):
        self.heuristics = [
            self.progress,
            lambda s: -option_count(env, s, horizon=self.horizon, walkers=self.walkers),
        ]
        return super().solve(env, budget, state)


CANDIDATES = {
    # width-based variants
    "dual": (DualBFWS, {"max_width": 1000}),
    "bfwsr": (BFWS, {"width": 1, "relevant": "iw"}),
    "qn": (QuantifiedNoveltySearch, {}),
    "bfnos": (BFNoS, {"width": 1}),
    "ans": (ApproximateNoveltySearch, {"width": 2, "space_bound": 500_000}),
    "hiw": (HierarchicalIW, {"low_expansions": 1000}),
    "bee": (BFWSOverBoundaryExtensions, {"width": 1, "bins": 4}),
    # heuristic search
    "gbfs": (BestFirstSearch, {}),
    "astar": (BestFirstSearch, {"weight": 1.0}),
    "wastar": (BestFirstSearch, {"weight": 2.0}),
    "rwa": (RestartingWeightedAStar, {}),
    "egbfs": (EpsilonGreedySearch, {"epsilon": 0.2}),
    "tgbfs": (TypeBasedSearch, {}),
    "dbfs": (DiverseBestFirst, {}),
    "gbfsle": (LocalExplorationSearch, {}),
    "gbfslw": (LocalExplorationSearch, {"strategy": "walks"}),
    "ehc": (EnforcedHillClimbing, {}),
    "mrw": (MonteCarloRandomWalks, {}),
    "beam": (BeamSearch, {"beam": 100}),
    "bulb": (BULB, {"beam": 20, "max_depth": 200}),
    "lds": (LimitedDiscrepancySearch, {"max_depth": 60}),
    "ib": (IterativeBroadening, {"max_depth": 60}),
    "lrta": (LRTAStar, {"lookahead": 50, "max_steps": 2000}),
    "rtaa": (RTAAStar, {"lookahead": 50, "max_steps": 2000}),
    "fess": (FESSOnProgress, {}),
    "multi": (ProgressAndOptionsQueues, {"horizon": 4, "walkers": 4}),
    # sampling
    "rhea": (RollingHorizonEvolution, {"expansions_per_step": 1000}),
    "cem": (CrossEntropyPlanner, {}),
    "shoot": (RandomShooting, {}),
    "nmcs": (NestedMonteCarloSearch, {}),
    "goexp": (GoExplore, {}),
    "mapel": (MAPElitesPlanner, {}),
    "est": (KinodynamicTree, {"strategy": "est"}),
    "kpiece": (KPIECEOnProgress, {}),
    "sst": (KinodynamicTree, {"strategy": "sst"}),
    "sa": (PlanLocalSearch, {}),
    "ils": (PlanLocalSearch, {"method": "iterated"}),
    "macro": (MacroPlanner, {}),
    # blind
    "brfs": (BreadthFirstSearch, {}),
    "ucs": (UniformCostSearch, {}),
    "iddfs": (IterativeDeepening, {"max_depth": 200}),
}
