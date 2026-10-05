"""Every planner in the library, as benchmark configurations.

`PLANNERS` holds one configuration per planner, and one per documented variant of a planner,
under the tag `solve` takes and the result directory is named by. `generate` writes jobs for
all of them and `report` tabulates all of them. Anything not named in a configuration is the
class's own default.

Every configuration takes `progress` from `measures.py`, and the planners that want more
than one number from an environment get it from that same measure through the adapters
below: FESS searches the feature space it spans, boundary-extension features extend its
range, and multi-queue alternation pairs it with quantified novelty over it.

Every planner here is deterministic: the same instance under the same limits gives the same
result, so each runs once per instance. No configuration takes a reward, and nothing learns
before it plans. Parameters are the classes' defaults, except that the online planners get
1,000 expansions per decision.
"""
from planiverse.planners.blind import BreadthFirstSearch, IterativeDeepening, UniformCostSearch
from planiverse.planners.heuristic import (
    BULB, BeamSearch, BestFirstSearch, EnforcedHillClimbing, FeatureSpaceSearch,
    IterativeBroadening, LRTAStar, LimitedDiscrepancySearch, MultiQueueSearch, RTAAStar,
    RestartingWeightedAStar,
)
from planiverse.planners.width import (
    BFWS, IW, SIW, BFNoS, BoundaryExtensionFeatures, DualBFWS, HierarchicalIW,
    QuantifiedNoveltySearch,
)
from planiverse.planners.width.quantified import HeuristicNovelty


class FESSOnProgress(FeatureSpaceSearch):
    """FESS whose one feature is the environment's progress measure."""

    def __init__(self, progress):
        super().__init__(features=lambda state: (progress(state),))


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


class ProgressAndNoveltyQueues(MultiQueueSearch):
    """Multi-queue alternation between the progress measure and quantified novelty.

    The second queue orders states by how many of their atoms no earlier state with as good
    a progress value contained (`HeuristicNovelty`, the measure `QuantifiedNoveltySearch`
    searches on), most novel first.
    """

    def __init__(self, progress, boost=0):
        novelty = HeuristicNovelty()
        super().__init__(
            [progress, lambda s: -novelty.evaluate_and_record(s.literals, progress(s))],
            boost=boost)


#: Every configuration, by tag.
PLANNERS = {
    # width-based
    "bfws": (BFWS, {"width": 1}),
    "iw": (IW, {"max_width": 1000, "strict": False}),
    "siw": (SIW, {"width": 1, "max_width": 1000, "strict": False}),
    "dual": (DualBFWS, {"max_width": 1000}),
    "bfwsr": (BFWS, {"width": 1, "relevant": "iw"}),
    "qn": (QuantifiedNoveltySearch, {}),
    "bfnos": (BFNoS, {"width": 1}),
    "hiw": (HierarchicalIW, {"low_expansions": 1000}),
    "bee": (BFWSOverBoundaryExtensions, {"width": 1, "bins": 4}),
    # heuristic search
    "gbfs": (BestFirstSearch, {}),
    "astar": (BestFirstSearch, {"weight": 1.0}),
    "wastar": (BestFirstSearch, {"weight": 2.0}),
    "rwa": (RestartingWeightedAStar, {}),
    "ehc": (EnforcedHillClimbing, {}),
    "beam": (BeamSearch, {"beam": 100}),
    "bulb": (BULB, {"beam": 20, "max_depth": 200}),
    "lds": (LimitedDiscrepancySearch, {"max_depth": 60}),
    "ib": (IterativeBroadening, {"max_depth": 60}),
    "lrta": (LRTAStar, {"lookahead": 50, "max_steps": 2000}),
    "rtaa": (RTAAStar, {"lookahead": 50, "max_steps": 2000}),
    "fess": (FESSOnProgress, {}),
    "multi": (ProgressAndNoveltyQueues, {}),
    # blind
    "brfs": (BreadthFirstSearch, {}),
    "ucs": (UniformCostSearch, {}),
    "iddfs": (IterativeDeepening, {"max_depth": 200}),
}
