"""Width-based planners for simulators.

Four searches, all against the `successors()` / `literals` contract and nothing else:

| Planner | Novelty is used as | Complete? |
|---|---|---|
| `IWSearch` | a filter — fails the test, gets discarded | no, for fixed width |
| `IteratedWidth` | the same, at width 1, then 2, … | up to `max_width` |
| `SIWSearch` | the same, in short legs that each make progress | no |
| `BFWSSearch` | a sort key — nothing is discarded | yes |

What changes when the task is a simulator rather than a PDDL model:

* **No goal decomposition.** `is_goal` is a black-box predicate, so the unachieved-goal count
  that SIW and BFWS classically lean on does not exist. Both take a `progress` callback and
  say what they degrade to without one.
* **Expansions are expensive.** Every search takes a `Budget` and returns
  `SearchStatistics`, because "found nothing" and "ran out of budget after four nodes" are
  different answers.
* **Dead ends are real.** `is_terminal` states are dropped rather than expanded.
* **The atoms are whatever `literals` says.** How coarsely an environment spells its state is
  what fixes its width — see `novelty`.
"""
from planiverse.planners.width.bfws import BFWSSearch
from planiverse.planners.width.iw import IteratedWidth, IWSearch, SIWSearch
from planiverse.planners.width.novelty import (
    MAX_PRACTICAL_WIDTH, NoveltyTable, PartitionedNovelty, path_novelty,
)
from planiverse.planners.width.result import Budget, SearchResult, SearchStatistics

__all__ = [
    "BFWSSearch", "Budget", "IWSearch", "IteratedWidth", "MAX_PRACTICAL_WIDTH",
    "NoveltyTable", "PartitionedNovelty", "SIWSearch", "SearchResult", "SearchStatistics",
    "path_novelty",
]
