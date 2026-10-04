"""Width-based planners for simulators.

What the package exports is what the literature names, and nothing else: the planners one
would run as a baseline, each under the name its paper gives it. The pieces they are built
from (IW at one fixed width, the novelty tables, the Bloom filters) live in their modules and
are not exported.

| Planner | Reference | Novelty is used as | Complete? |
|---|---|---|---|
| `IW` | Lipovetzky & Geffner 2012, 2015 | a filter, at width 1, then 2, … (`width=k` for one) | up to `max_width` |
| `SIW` | Lipovetzky & Geffner 2012 | a filter, in short legs that each make progress | no |
| `BFWS` | Lipovetzky & Geffner 2017; Francès et al. 2017 for `relevant="iw"` | a sort key: nothing is discarded | yes, unless `prune=True` |
| `DualBFWS` | Lipovetzky & Geffner 2017 | a filter in cheap rounds, then a sort key in the last | yes |
| `QuantifiedNoveltySearch` | Katz, Lipovetzky, Moshkovich & Tuisov 2017 | a heuristic: how many atoms are new at this heuristic value | yes |
| `BFNoS` | Rosa & Lipovetzky 2024 | a sort key on how often the rarest tuple was seen; open list trimmed | no |
| `ApproximateNoveltySearch` | Singh, Lipovetzky, Ramírez & Segovia-Aguas 2021 | a sort key from Bloom filters over sampled tuples | no |
| `HierarchicalIW` | Junyent, Gómez & Jonsson 2021 | a filter at two levels of abstraction | no |

`BoundaryExtensionFeatures` (Teichteil-Königsbuch, Ramírez & Lipovetzky 2020) is not a
planner but the `atoms=` hook that makes `IW`, `BFWS` and `DualBFWS` run over continuous
state variables, so it is exported with them.

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
from planiverse.planners.width.approximate import ApproximateNoveltySearch
from planiverse.planners.width.bee import BoundaryExtensionFeatures
from planiverse.planners.width.bfws import BFWS, DualBFWS
from planiverse.planners.width.count import BFNoS
from planiverse.planners.width.hierarchical import HierarchicalIW
from planiverse.planners.width.iw import IW, SIW
from planiverse.planners.width.quantified import QuantifiedNoveltySearch
from planiverse.planners.width.result import Budget, SearchResult, SearchStatistics

__all__ = [
    "ApproximateNoveltySearch", "BFNoS", "BFWS", "BoundaryExtensionFeatures", "Budget",
    "DualBFWS", "HierarchicalIW", "IW", "QuantifiedNoveltySearch", "SIW", "SearchResult",
    "SearchStatistics",
]
