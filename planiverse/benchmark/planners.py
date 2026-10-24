"""Every planner in the library, as benchmark configurations.

`REFERENCE` holds the two reference configurations, BFWS and IW; `PLANNERS` holds every
configuration, the reference ones first, and is what `generate` and `report` run and
tabulate (`--reference` keeps either to the two). The tags are what `solve` takes and what
the result directories are named. Anything not named in a configuration is the class's own
default.

Every configuration takes `progress` from `measures.py`, and the planners that want more
than one number from an environment get it from that same measure through the adapters
below: FESS searches the feature space it spans and KPIECE projects onto it. `GoExplore`
takes a cell projection and runs without one, on its default, the exact state. The two
add-ons, `DominatedActionPruner` and `FocusedMacros`, are not planners: the second is what
`MacroPlanner` runs on, and the first is a `SuccessorCache` option no configuration here
sets.

No configuration takes a reward: every planner here is driven by `is_goal` and the progress
heuristic, and nothing learns before it plans. Parameters are the classes' defaults, except
that the protocol's 500,000 expansions bound the approximate-novelty policy.
"""
from planiverse.planners.heuristic import (
    BULB, BestFirstSearch, DiverseBestFirst, EpsilonGreedySearch, FeatureSpaceSearch,
    LimitedDiscrepancySearch, LocalExplorationSearch, RestartingWeightedAStar, TypeBasedSearch,
)
from planiverse.planners.macros import MacroPlanner
from planiverse.planners.sampling import GoExplore, KinodynamicTree
from planiverse.planners.width import (
    BFWS, IW, ApproximateNoveltySearch, BFNoS, QuantifiedNoveltySearch,
)


class FESSOnProgress(FeatureSpaceSearch):
    """FESS whose one feature is the environment's progress measure."""

    def __init__(self, progress):
        super().__init__(features=lambda state: (progress(state),))


class KPIECEOnProgress(KinodynamicTree):
    """KPIECE whose projection is the environment's progress measure.

    KPIECE's grid is over a low-dimensional projection; on the exact state, the class's
    default, every cell holds one state and the grid does nothing.
    """

    def __init__(self, progress, seed=None):
        super().__init__(projection=lambda state: (progress(state),), seed=seed)


#: The reference configurations: the two width planners BFWS and IW.
REFERENCE = {
    "bfws": (BFWS, {"width": 1}),
    "iw": (IW, {"max_width": 1000, "strict": False}),
}

#: Every configuration, by tag, the reference ones first.
PLANNERS = {
    **REFERENCE,
    # width-based
    "bfwsr": (BFWS, {"width": 1, "relevant": "iw"}),
    "qn": (QuantifiedNoveltySearch, {}),
    "bfnos": (BFNoS, {"width": 1}),
    "ans": (ApproximateNoveltySearch, {"width": 2, "space_bound": 500_000}),
    # heuristic search
    "gbfs": (BestFirstSearch, {}),
    "astar": (BestFirstSearch, {"weight": 1.0}),
    "wastar": (BestFirstSearch, {"weight": 2.0}),
    "rwa": (RestartingWeightedAStar, {}),
    "egbfs": (EpsilonGreedySearch, {"epsilon": 0.2}),
    "tgbfs": (TypeBasedSearch, {}),
    "dbfs": (DiverseBestFirst, {}),
    "gbfslw": (LocalExplorationSearch, {}),
    "bulb": (BULB, {"beam": 20, "max_depth": 200}),
    "lds": (LimitedDiscrepancySearch, {"max_depth": 60}),
    "fess": (FESSOnProgress, {}),
    # sampling
    "goexp": (GoExplore, {}),
    "kpiece": (KPIECEOnProgress, {}),
    "macro": (MacroPlanner, {}),
}
