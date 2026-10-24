"""Heuristic search with black-box heuristics, and the exploration repairs for when the
heuristic misleads.

What satisficing planning learned about searching on a weak heuristic, which `progress`
usually is, and none of it needs a model:

| Planner | What it does when the heuristic is wrong |
|---|---|
| `BestFirstSearch` | nothing: greedy, or weighted A* with `weight` |
| `RestartingWeightedAStar` | anytime: solves fast with a big weight, then restarts smaller |
| `EpsilonGreedySearch` | expands a random open node with probability ε |
| `TypeBasedSearch` | alternates with a queue that picks a random `(h, g)` type |
| `DiverseBestFirst` | samples an open node by its heuristic, then searches locally |
| `LocalExplorationSearch` | on a plateau, random walks from the stalled node |
| `BULB` | beam search that backtracks over discrepancies |
| `LimitedDiscrepancySearch` | depth-first with bounded deviations |
| `FeatureSpaceSearch` | searches a feature space, cycling over its cells |

Every heuristic here is a `progress(state)` callback, lower is better.
"""
from planiverse.planners.heuristic.beam import BULB, LimitedDiscrepancySearch
from planiverse.planners.heuristic.bestfirst import (
    BestFirstSearch, DiverseBestFirst, EpsilonGreedySearch, LocalExplorationSearch,
    RestartingWeightedAStar, TypeBasedSearch,
)
from planiverse.planners.heuristic.fess import FeatureSpaceSearch

__all__ = [
    "BULB", "BestFirstSearch", "DiverseBestFirst", "EpsilonGreedySearch", "FeatureSpaceSearch",
    "LimitedDiscrepancySearch", "LocalExplorationSearch", "RestartingWeightedAStar",
    "TypeBasedSearch",
]
