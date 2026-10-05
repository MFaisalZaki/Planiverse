"""Heuristic search with black-box heuristics, and the exploration repairs for when the
heuristic misleads.

What satisficing planning learned about searching on a weak heuristic, which `progress`
usually is, and none of it needs a model:

| Planner | What it does when the heuristic is wrong |
|---|---|
| `BestFirstSearch` | nothing: greedy, or weighted A* with `weight` |
| `RestartingWeightedAStar` | anytime: solves fast with a big weight, then restarts smaller |
| `EnforcedHillClimbing` | commits to the first strictly better state it finds |
| `BeamSearch`, `BULB` | keeps the best `beam` per depth; BULB backtracks over discrepancies |
| `LimitedDiscrepancySearch`, `IterativeBroadening` | depth-first with bounded deviations |
| `LRTAStar`, `RTAAStar` | commit an action per bounded lookahead, raising stored values |
| `FeatureSpaceSearch` | searches a feature space, cycling over its cells |
| `MultiQueueSearch` | round-robin over one queue per heuristic |

Every heuristic here is a `progress(state)` callback, lower is better, and every search is
deterministic: the same instance and budget give the same result.
"""
from planiverse.planners.heuristic.beam import (
    BULB, BeamSearch, IterativeBroadening, LimitedDiscrepancySearch,
)
from planiverse.planners.heuristic.bestfirst import BestFirstSearch, RestartingWeightedAStar
from planiverse.planners.heuristic.ehc import EnforcedHillClimbing
from planiverse.planners.heuristic.fess import FeatureSpaceSearch
from planiverse.planners.heuristic.multiqueue import MultiQueueSearch
from planiverse.planners.heuristic.realtime import LRTAStar, RTAAStar

__all__ = [
    "BULB", "BeamSearch", "BestFirstSearch", "EnforcedHillClimbing", "FeatureSpaceSearch",
    "IterativeBroadening", "LRTAStar", "LimitedDiscrepancySearch", "MultiQueueSearch",
    "RTAAStar", "RestartingWeightedAStar",
]
