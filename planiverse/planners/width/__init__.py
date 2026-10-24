"""Width-based planners for simulators.

What the package exports is what the literature names, and nothing else: the planners one
would run as a baseline, each under the name its paper gives it. The pieces they are built
from (IW at one fixed width, the novelty tables, the Bloom filters) live in their modules and
are not exported.

| Planner | Reference | Novelty is used as | Complete? |
|---|---|---|---|
| `IW` | Lipovetzky & Geffner 2012, 2015 | a filter, at width 1, then 2, … (`width=k` for one) | up to `max_width` |
| `BFWS` | Lipovetzky & Geffner 2017; Francès et al. 2017 for `relevant="iw"` | a sort key: nothing is discarded | yes, unless `prune=True` |
| `QuantifiedNoveltySearch` | Katz, Lipovetzky, Moshkovich & Tuisov 2017 | a heuristic: how many atoms are new at this heuristic value | yes |
| `BFNoS` | Rosa & Lipovetzky 2024 | a sort key on how often the rarest tuple was seen; open list trimmed | no |
| `ApproximateNoveltySearch` | Singh, Lipovetzky, Ramírez & Segovia-Aguas 2021 | a sort key from Bloom filters over sampled tuples | no |

What changes when the task is a simulator rather than a PDDL model:

* **No goal decomposition.** `is_goal` is a black-box predicate, so the unachieved-goal count
  that BFWS classically leans on does not exist. It takes a `progress` callback instead and
  says what it degrades to without one.
* **Expansions are expensive.** Every search takes a `Budget` and returns
  `SearchStatistics`, because "found nothing" and "ran out of budget after four nodes" are
  different answers.
* **Dead ends are real.** `is_terminal` states are dropped rather than expanded.
* **The atoms are whatever `literals` says.** How coarsely an environment spells its state is
  what fixes its width (see `novelty`).
"""
from planiverse.planners.width.approximate import ApproximateNoveltySearch
from planiverse.planners.width.bfws import BFWS
from planiverse.planners.width.count import BFNoS
from planiverse.planners.width.iw import IW
from planiverse.planners.width.quantified import QuantifiedNoveltySearch
from planiverse.planners.width.result import Budget, SearchResult, SearchStatistics

__all__ = [
    "ApproximateNoveltySearch", "BFNoS", "BFWS", "Budget", "IW", "QuantifiedNoveltySearch",
    "SearchResult", "SearchStatistics",
]
