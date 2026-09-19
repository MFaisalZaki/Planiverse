# The planners added after the tool paper

The training-free planners from the [survey](candidates.md), implemented against the same
`successors` / `literals` / `is_goal` / `is_terminal` contract as everything else. None
touches the environment interface, none learns before it plans, none is a Monte Carlo tree
search, and **none takes a reward**: this is a planning suite, so every planner here is
driven by the black-box goal test and, at most, the `progress` heuristic (how far a state
is from the goal, lower is better). Four surveyed planners are defined by an accumulated
reward and were left out for that reason: 2BFS, prioritised IW, Fractal Monte Carlo and
the rollout algorithm; see [Left out](#left-out). Where a paper's exact rule could not be read from the machine this was written
on, the rule used is stated in the class's docstring and repeated here under
[What is inferred](#what-is-inferred), the way `fsx.py` does it.

Every planner has `solve(env, budget=None, state=None) -> SearchResult`, takes a `Budget`,
and reports `SearchStatistics`. Every heuristic is the `progress(state)` callback the width
planners take, lower is better. Nothing below needs anything else from an environment,
except where a column says so.

```python
from planiverse.planners.width import BFWSR, CountNoveltySearch, Budget
from planiverse.planners.heuristic import EnforcedHillClimbing, BULB
from planiverse.planners.sampling import GoExplore, RollingHorizonEvolution

env.set_index(0)
budget = Budget(max_expansions=5000, max_seconds=60)
for planner in (BFWSR(progress=boxes), GoExplore(progress=boxes, seed=0)):
    result = planner.solve(env, budget)
    print(planner.__class__.__name__, result.status, len(result), result.statistics)
```

## Width-based variants (`planiverse/planners/width/`)

| Class | File | Reference | Beyond `progress` it needs |
|---|---|---|---|
| `BFWSR` | `relevant.py` | Lipovetzky & Geffner 2017; Francès et al. 2017 | nothing |
| `QuantifiedNoveltySearch` | `quantified.py` | Katz, Lipovetzky, Moshkovich & Tuisov 2017 | nothing |
| `CountNoveltySearch` | `count.py` | Rosa & Lipovetzky 2024 | nothing |
| `ApproximateNoveltySearch` | `approximate.py` | Singh, Lipovetzky, Ramírez & Segovia-Aguas 2021 | nothing |
| `BoundaryExtensionFeatures` | `bee.py` | Teichteil-Königsbuch, Ramírez & Lipovetzky 2020 | `variables(state) -> {name: float}` |
| `HierarchicalIW` | `hierarchical.py` | Junyent, Gómez & Jonsson 2021 | nothing; `features` optional |

`IWSearch`, `BFWSSearch` and `IteratedBFWS` gained one optional argument, `atoms(state)`,
so that novelty can be measured over something other than `literals`; that is the hook
`BoundaryExtensionFeatures` plugs into, and the only change to the existing planners.

**BFWS(R)** runs IW(`r_width`) first, on `r_share` of the
expansion budget; `R` is the union of the literals on the paths to every state that strictly
improved `progress`, and if that pre-search reaches the goal its plan is returned. The main
search is BFWS with novelty measured within `(progress, #r)` partitions and ties broken on
`progress` then `#r`. **Quantified novelty** counts the atoms of a state that no earlier state
with a heuristic value at most as good contained, and searches greedily on `(-QN, h)`,
`(h, -QN)` or the binary version. **Count-based novelty** scores a state by the number of
recorded states that contained its rarest tuple and drops the worse half of the open list
whenever it exceeds `open_limit`; a search that trimmed and found nothing is `failed`, never
`exhausted`. **Approximate novelty** keeps sampled tuples in a bank of Bloom filters, one per
`progress` partition until the bank is spent and shared at random afterwards, and allows
widths above 2 without `strict=False` because bounding that cost is the point.
**Hierarchical IW** runs a high-level IW(1) whose expansion is a low-level IW(1) capped at
`low_expansions`; every low-level state whose projection onto the high-level features differs
from its parent's, or that improved `progress`, is a high-level successor.

## Heuristic search (`planiverse/planners/heuristic/`) and blind baselines (`planiverse/planners/blind.py`)

| Class | File | Reference | Beyond `progress` it needs |
|---|---|---|---|
| `BestFirstSearch` | `bestfirst.py` | greedy best-first; Pohl 1970 for `weight` | nothing; `progress` optional |
| `RestartingWeightedAStar` | `bestfirst.py` | Richter, Thayer & Ruml 2010 | nothing |
| `EpsilonGreedySearch` | `bestfirst.py` | Valenzano, Schaeffer, Sturtevant & Xie 2014 | nothing |
| `TypeBasedSearch` | `bestfirst.py` | Xie, Müller, Holte & Imai 2014 | nothing |
| `DiverseBestFirst` | `bestfirst.py` | Imai & Kishimoto 2011 | nothing |
| `LocalExplorationSearch` | `bestfirst.py` | Xie, Müller & Holte 2014, 2015 | nothing |
| `EnforcedHillClimbing` | `ehc.py` | Hoffmann & Nebel 2001 | nothing |
| `MonteCarloRandomWalks` | `randomwalk.py` | Nakhost & Müller 2009, 2013 | nothing |
| `BeamSearch`, `BULB` | `beam.py` | Furcy & Koenig 2005 | nothing |
| `LimitedDiscrepancySearch`, `IterativeBroadening` | `beam.py` | Harvey & Ginsberg 1995; Ginsberg & Harvey 1992 | nothing; `progress` optional |
| `LRTAStar`, `RTAAStar` | `realtime.py` | Korf 1990; Koenig & Likhachev 2006 | nothing |
| `FeatureSpaceSearch` | `fess.py` | Shoham & Schaeffer 2020 | `features(state) -> tuple`; `advisors` optional |
| `MultiQueueSearch` | `multiqueue.py` | Helmert 2006; Röger & Helmert 2010 | a list of heuristics |
| `BreadthFirstSearch`, `UniformCostSearch`, `IterativeDeepening` | `blind.py` | | nothing |

The best-first family shares one engine, a heap plus a list of the same entries with a
`removed` flag, so that "the best open node" and "a uniformly random open node" are both
one pop. Weighted A\* reads `action.cost()` when actions have one, else 1. Restarting weighted
A\* shares a `SuccessorCache` between its rounds, so a restart re-reads what the last round
generated instead of asking the simulator again, and reports the weights it ran in
`widths_tried`. Enforced hill-climbing refuses dead-end progress by default, the refusal
that turns SIW's failure on Puzznic level 1 into a solution, and takes `novelty_width` to
bound its legs the way SIW's are. Time-bounded A\* is not here: it walks the agent backwards
along its search tree, which assumes reversible actions.

The three depth-first probes (limited discrepancy, iterative broadening, BULB) and iterative
deepening re-expand states on every iteration. `SuccessorCache` makes those re-expansions
free in simulator calls, so the counted expansions are distinct states; their *time* is
still exponential in `max_depth`, which is why their defaults are small and their smoke
tests run with a time limit.

## Sampling, population and swarm planners (`planiverse/planners/sampling/`)

| Class | File | Reference | Beyond `progress` it needs |
|---|---|---|---|
| `RollingHorizonEvolution` | `rhea.py` | Perez et al. 2013; Gaina et al. 2017, 2021 | nothing |
| `CrossEntropyPlanner`, `RandomShooting` | `cem.py` | Rubinstein 1999; Pinneri et al. 2020 | nothing |
| `NestedMonteCarloSearch` | `nested.py` | Cazenave 2009 | nothing |
| `GoExplore` | `goexplore.py` | Ecoffet et al. 2019, 2021 | nothing; `cell` optional |
| `MAPElitesPlanner` | `qd.py` | Mouret & Clune 2015; Lehman & Stanley 2011 | nothing; `descriptor` optional |
| `KinodynamicTree` | `kinodynamic.py` | Hsu et al. 1997; Şucan & Kavraki 2008; Li et al. 2016 | nothing; `projection` optional |
| `PlanLocalSearch` | `localsearch.py` | Kirkpatrick et al. 1983; Lourenço et al. 2003 | nothing |

All of them replay action sequences through `SuccessorCache`
([`planiverse/planners/common.py`](../../planiverse/planners/common.py)), which memoises
`successors` on `literals` and counts an expansion once, so a prefix the population evaluates
a hundred times costs the simulator once. A sequence is scored by where it ends
(`sequences.py`), with the goal-distance heuristic and nothing else: the negated `progress`
of the last state, `+inf` at a goal, `-inf` in a dead end. In their home literature these
planners maximise a game score; here there is no score, only how far from the goal a
sequence leaves the simulator. A gene that is not applicable where it lands is skipped, the way a game's forward
model treats an invalid input. A goal reached during any evaluation ends the search with that
sequence as the plan. The receding-horizon planners commit the first gene the best sequence
actually applied, and fall back to a random live child only when it applied none.

Actions are drawn from `get_actions()` when the environment provides it, else from the
actions applicable in the current state (`action_vocabulary`).

Two consequences of the cache are worth knowing. Because a warm cache makes a generation
free in expansions, `RollingHorizonEvolution` ends a decision after `generations` generations
as well as after `expansions_per_step`, or it would never end. And the archive planners
(`GoExplore`, `MAPElitesPlanner`, `KinodynamicTree`) key their cells on `literals` by
default, the exact state, so a coarser projection is what makes the cell structure do
anything: with `cell=lambda s: (boxes(s),)` Go-Explore's archive on Puzznic has at most
seven cells.

## Add-ons (`planiverse/planners/pruning.py`, `planiverse/planners/macros.py`)

| Class | Reference | What it plugs into |
|---|---|---|
| `DominatedActionPruner` | Jinnai & Fukunaga 2017 | `SuccessorCache(env, pruner=...)`, so the sampling planners stop drawing duplicate actions |
| `FocusedMacros`, `MacroPlanner` | Allen et al. 2021 | a greedy best-first over discovered macros and primitives |

Under the `successors` contract every child is generated at once and duplicates are caught
on `literals`, so dominated-action pruning saves a tree search nothing; it pays for the
planners that *sample* actions, which is where it is wired in. Macro discovery enumerates
sequences up to `length` from the initial state and a few random-walk probes, keeps the
`count` with the smallest non-empty footprints (the symmetric difference of `literals`),
and spends `share` of the expansion budget doing so.

## Smoke results on Puzznic level 1

One run each, seed 0 where a planner takes one, `Budget(max_expansions=3000,
max_seconds=60)`, on the level the width docs use: IW(1) exhausts on it after 32
expansions, IW(2) solves it in 10 actions, and its first pair clearance is a dead end. These
are the runs `tests/test_candidate_planners.py` repeats; they are not the benchmark.

| Planner | Status | Plan | Expansions | Note |
|---|---|---|---|---|
| `BFWSR` | solved | 12 | 122 | `R` is empty at width 1 here; `r_width=2` returns the pre-search's plan |
| `QuantifiedNoveltySearch` | solved | 12 | 90 | |
| `CountNoveltySearch(open_limit=500)` | solved | 12 | 65 | |
| `ApproximateNoveltySearch(width=2)` | solved | 12 | 90 | |
| `HierarchicalIW` | solved | 12 | 385 | features discovered, none given |
| `BFWSSearch` over `BoundaryExtensionFeatures` | solved | 12 | 90 | |
| `BestFirstSearch` (greedy, and `weight=2`) | solved | 12 | 76 | |
| `RestartingWeightedAStar` | solved | 12 | 118 | weights 5, 3, 2, 1.5, 1 |
| `EpsilonGreedySearch` | solved | 12 | 80 | |
| `TypeBasedSearch` | solved | 12 | 108 | |
| `DiverseBestFirst` | solved | 12 | 78 | |
| `LocalExplorationSearch` (greedy / walks) | solved | 12 / 18 | 76 / 89 | |
| `EnforcedHillClimbing` | solved | 12 | 76 | fails with `avoid_dead_ends=False` |
| `MonteCarloRandomWalks` | solved | 30 | 115 | |
| `BeamSearch(beam=20)` | solved | 12 | 143 | `beam=1` fails; `BULB(beam=1)` solves |
| `BULB(beam=5)` | solved | 18 | 157 | |
| `LimitedDiscrepancySearch(max_depth=30)` | solved | 18 | 183 | 24 s: the probes are exponential in depth |
| `IterativeBroadening(max_depth=30)` | out of budget | | 30 | the same, worse |
| `LRTAStar(lookahead=20)` | failed | | 20 | one step per lookahead; 500 steps were not enough |
| `RTAAStar(lookahead=20)` | solved | 28 | 80 | |
| `FeatureSpaceSearch` on the block count | solved | 12 | 167 | |
| `MultiQueueSearch` | solved | 12 | 76 | |
| `BreadthFirstSearch`, `IterativeDeepening` | solved | 10 | 120 | the optimal length |
| `UniformCostSearch` | solved | 10 | 148 | |
| `RollingHorizonEvolution` | solved | 150 | 111 | |
| `CrossEntropyPlanner` | solved | 142 | 123 | |
| `RandomShooting` | step limit | | 55 | |
| `NestedMonteCarloSearch(level=2, horizon=20)` | solved | 14 | 177 | |
| `GoExplore` (exact cells / block-count cells) | solved | 20 / 28 | 274 / 119 | |
| `MAPElitesPlanner` (exact descriptor) | solved | 12 | 151 | fails with the block-count descriptor: one cell |
| `KinodynamicTree` (`est` / `kpiece` / `sst`) | solved | 24 / 22 / 24 | 273 / 75 / 273 | KPIECE on the block-count projection |
| `PlanLocalSearch` (annealing / iterated) | solved / failed | 14 | 138 / 47 | |
| `MacroPlanner` | solved | 12 | 76 | |

## What is inferred

The papers below could not be read in full from the machine this was written on (the
network proxy blocks arXiv, IJCAI, AAAI and the publishers' sites alike). Each class's
docstring says so; what follows is the list.

- **Approximate novelty** (`approximate.py`): the Bloom filters, the tuple sampling and the
  filter bank are the paper's; the *expansion-skipping policy* is a stand-in: a node whose
  novelty exceeds `width` is expanded with probability `1 - generated / space_bound`.
- **Count-based novelty** (`count.py`): the count-novelty of a state is the count of its
  rarest tuple; the trimmed open list drops the worse half above `open_limit`.
- **Hierarchical IW** (`hierarchical.py`): the feature-discovery rule. An atom joins the
  high-level set the first time the low level prunes a state that made it true.
- **Diverse best-first** (`bestfirst.py`): heuristic value `h` is sampled with weight
  `temperature ** (h - h_min)`; the paper's exact distribution over `g` as well is not here.
- **Arvand** (`randomwalk.py`): the dead-end avoidance bias is `exp(-bias * dead_end_rate)`
  over the first action of a walk; the paper's exact MDA formula is not.
- **FESS** (`fess.py`): cells keyed by the feature vector, round-robin over cells,
  least-weight move first, advisors marking moves. The Sokoban-specific machinery is not.
- **KPIECE** (`kinodynamic.py`): the exterior-cell preference has no counterpart in a
  symbolic projection and is left out of the importance.
- **Focused macros** (`macros.py`): exhaustive enumeration at small lengths in place of the
  paper's best-first search over sequences, with the same footprint criterion.

## Left out

Four of the survey's candidates are defined by an accumulated reward rather than by a goal
and a distance to it, and a planning suite has no reward to give them:

- **2BFS** (Lipovetzky, Ramírez & Geffner 2015): one of its two queues is ordered by the
  reward accumulated along the path.
- **Prioritised IW** (Shleyfman, Tuisov & Domshlak 2016): its novelty test keeps a state
  that reaches a known atom with more accumulated reward.
- **Fractal Monte Carlo** (Hernández Cerezo & Duran Ballester 2018): its walkers clone by a
  virtual reward, the product of accumulated reward and distance.
- **The rollout algorithm** (Bertsekas, Tsitsiklis & Wu 1997): one step of policy
  iteration on a base policy's return.

Each could be run with `progress(root) - progress(node)` standing in for the reward, and
was, in an earlier revision of this branch; the mechanism is the reward's, so they went. The
library's Rollout IW, π-IW and MCTS planners went for the same reason: a discounted return
chose their actions, π-IW learned a policy as it planned, and MCTS wanted a reward.

## Running them in the benchmark

They are registered in [`planiverse/benchmark/candidates.py`](../../planiverse/benchmark/candidates.py)
under short tags, and `planiverse-bench generate --candidates` and `report --candidates`
include them; without the flag the benchmark is the paper's protocol and nothing else.
`solve` accepts any tag either way. The seeded ones run under the same five seeds as FSX. Boundary-extension features, multi-queue alternation and the two add-ons need
per-environment callbacks the benchmark does not carry and are not registered.

## Files

| Path | What |
|---|---|
| [`common.py`](../../planiverse/planners/common.py) | `SuccessorCache`, `action_vocabulary`, `remaining`, `finish` |
| [`width/`](../../planiverse/planners/width/) | the six width-based additions |
| [`heuristic/`](../../planiverse/planners/heuristic/) | the best-first family, EHC, random walks, beam, real-time, FESS, alternation |
| [`sampling/`](../../planiverse/planners/sampling/) | RHEA, CEM, NMCS, Go-Explore, MAP-Elites, kinodynamic trees, local search |
| [`blind.py`](../../planiverse/planners/blind.py) | the three blind baselines |
| [`pruning.py`](../../planiverse/planners/pruning.py), [`macros.py`](../../planiverse/planners/macros.py) | the add-ons |
| [`benchmark/candidates.py`](../../planiverse/benchmark/candidates.py) | the opt-in benchmark registry |
| [`tests/test_candidate_planners.py`](../../tests/test_candidate_planners.py) | the smoke tests |
