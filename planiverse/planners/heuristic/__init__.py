"""Heuristic search with black-box heuristics, and the exploration repairs for when the
heuristic misleads.

The library had one best-first search. This package has the rest of what satisficing
planning learned about searching on a weak heuristic, which `progress` usually is, and
none of it needs a model:

| Planner | What it does when the heuristic is wrong |
|---|---|
| `BestFirstSearch` | nothing: greedy, or weighted A* with `weight` |
| `RestartingWeightedAStar` | anytime: solves fast with a big weight, then restarts smaller |
| `EpsilonGreedySearch` | expands a random open node with probability ε |
| `TypeBasedSearch` | alternates with a queue that picks a random `(h, g)` type |
| `DiverseBestFirst` | samples an open node by its heuristic, then searches locally |
| `LocalExplorationSearch` | on a plateau, local greedy search or random walks |
| `EnforcedHillClimbing` | commits to the first strictly better state it finds |
| `MonteCarloRandomWalks` | jumps to the best endpoint of a batch of random walks |
| `BeamSearch`, `BULB` | keeps the best `beam` per depth; BULB backtracks over discrepancies |
| `LimitedDiscrepancySearch`, `IterativeBroadening` | depth-first with bounded deviations |
| `LRTAStar`, `RTAAStar` | commit an action per bounded lookahead, raising stored values |
| `FeatureSpaceSearch` | searches a feature space, cycling over its cells |
| `MultiQueueSearch` | round-robin over one queue per heuristic |

Every heuristic here is a `progress(state)` callback, lower is better.
"""
from planiverse.planners.heuristic.beam import (
    BULB, BeamSearch, IterativeBroadening, LimitedDiscrepancySearch,
)
from planiverse.planners.heuristic.bestfirst import (
    BestFirstSearch, DiverseBestFirst, EpsilonGreedySearch, LocalExplorationSearch,
    RestartingWeightedAStar, TypeBasedSearch,
)
from planiverse.planners.heuristic.ehc import EnforcedHillClimbing
from planiverse.planners.heuristic.fess import FeatureSpaceSearch
from planiverse.planners.heuristic.multiqueue import MultiQueueSearch
from planiverse.planners.heuristic.randomwalk import MonteCarloRandomWalks
from planiverse.planners.heuristic.realtime import LRTAStar, RTAAStar

__all__ = [
    "BULB", "BeamSearch", "BestFirstSearch", "DiverseBestFirst", "EnforcedHillClimbing",
    "EpsilonGreedySearch", "FeatureSpaceSearch", "IterativeBroadening", "LRTAStar",
    "LimitedDiscrepancySearch", "LocalExplorationSearch", "MonteCarloRandomWalks",
    "MultiQueueSearch", "RTAAStar", "RestartingWeightedAStar", "TypeBasedSearch",
]
