"""Width-based planners for simulators.

Width-based searches, all against the `successors()` / `literals` contract and nothing else:

| Planner | Novelty is used as | Complete? |
|---|---|---|
| `IWSearch` | a filter: fails the test, gets discarded | no, for fixed width |
| `IteratedWidth` | the same, at width 1, then 2, … | up to `max_width` |
| `SIWSearch` | the same, in short legs that each make progress | no |
| `BFWSSearch` | a sort key: nothing is discarded | yes (unless `prune`) |
| `IteratedBFWS` | a filter in cheap rounds, then a sort key in the last | yes |
| `BFWSR` | a sort key within `(progress, relevant atoms reached)` partitions | yes |
| `QuantifiedNoveltySearch` | a heuristic: how many atoms are new at this heuristic value | yes |
| `CountNoveltySearch` | a sort key on how often the rarest tuple was seen; open list trimmed | no |
| `ApproximateNoveltySearch` | a sort key from Bloom filters over sampled tuples | no |
| `HierarchicalIW` | a filter at two levels of abstraction | no |

What changes when the task is a simulator rather than a PDDL model:

* **No goal decomposition.** `is_goal` is a black-box predicate, so the unachieved-goal count
  that SIW and BFWS classically lean on does not exist. Both take a `progress` callback and
  say what they degrade to without one.
* **Expansions are expensive.** Every search takes a `Budget` and returns
  `SearchStatistics`, because "found nothing" and "ran out of budget after four nodes" are
  different answers.
* **Dead ends are real.** `is_terminal` states are dropped rather than expanded.
* **The atoms are whatever `literals` says.** How coarsely an environment spells its state is
  what fixes its width (see `novelty`).
"""
from planiverse.planners.width.approximate import (
    ApproximateNoveltySearch, BloomFilter, BloomNoveltyTable,
)
from planiverse.planners.width.bee import BoundaryExtensionFeatures
from planiverse.planners.width.bfws import BFWSSearch, IteratedBFWS
from planiverse.planners.width.count import CountNoveltySearch, CountNoveltyTable
from planiverse.planners.width.hierarchical import HierarchicalIW
from planiverse.planners.width.iw import IteratedWidth, IWSearch, SIWSearch
from planiverse.planners.width.novelty import (
    MAX_PRACTICAL_WIDTH, NoveltyTable, PartitionedNovelty, path_novelty,
)
from planiverse.planners.width.quantified import HeuristicNovelty, QuantifiedNoveltySearch
from planiverse.planners.width.relevant import BFWSR
from planiverse.planners.width.result import Budget, SearchResult, SearchStatistics

__all__ = [
    "ApproximateNoveltySearch", "BFWSR", "BFWSSearch", "BloomFilter", "BloomNoveltyTable",
    "BoundaryExtensionFeatures", "Budget", "CountNoveltySearch", "CountNoveltyTable",
    "HeuristicNovelty", "HierarchicalIW", "IWSearch", "IteratedBFWS", "IteratedWidth",
    "MAX_PRACTICAL_WIDTH", "NoveltyTable", "PartitionedNovelty", "QuantifiedNoveltySearch",
    "SIWSearch", "SearchResult", "SearchStatistics", "path_novelty",
]
