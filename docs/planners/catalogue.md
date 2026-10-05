# The planners

Every planner in the library, against the same `successors` / `literals` / `is_goal` /
`is_terminal` contract. None learns before it plans, none is a Monte Carlo tree search, and
**none takes a reward**: each is driven by the black-box goal test and, at most, the
`progress` heuristic (how far a state is from the goal, lower is better). Every planner has
`solve(env, budget=None, state=None) -> SearchResult`, takes a `Budget`, and reports
`SearchStatistics`. Nothing below needs anything else from an environment, except where a
column says so.

```python
from planiverse.planners.width import BFWS, BFNoS, Budget
from planiverse.planners.heuristic import EnforcedHillClimbing, BULB
from planiverse.planners.sampling import GoExplore, RollingHorizonEvolution

env.set_index(0)
budget = Budget(max_expansions=5000, max_seconds=60)
for planner in (BFWS(progress=boxes, relevant="iw"), GoExplore(progress=boxes, seed=0)):
    result = planner.solve(env, budget)
    print(planner.__class__.__name__, result.status, len(result), result.statistics)
```

The last column of each table is the tag the benchmark runs the planner under; see
[Running them in the benchmark](#running-them-in-the-benchmark).

## Width-based (`planiverse/planners/width/`)

[width-based.md](width-based.md) describes novelty, `IW`, `SIW`, `BFWS` and `DualBFWS` and
what changes when the task is a simulator.

| Class | File | Reference | Beyond `progress` it needs | Tag |
|---|---|---|---|---|
| `IW` | `iw.py` | Lipovetzky & Geffner 2012, 2015 | nothing | `iw` |
| `SIW` | `iw.py` | Lipovetzky & Geffner 2012 | nothing | `siw` |
| `BFWS` | `bfws.py` | Lipovetzky & Geffner 2017 | nothing | `bfws` |
| `BFWS(relevant="iw")`, BFWS(R) | `bfws.py` | Lipovetzky & Geffner 2017; Francès et al. 2017 | nothing | `bfwsr` |
| `DualBFWS` | `bfws.py` | Lipovetzky & Geffner 2017 | nothing | `dual` |
| `QuantifiedNoveltySearch` | `quantified.py` | Katz, Lipovetzky, Moshkovich & Tuisov 2017 | nothing | `qn` |
| `BFNoS` | `count.py` | Rosa & Lipovetzky 2024 | nothing | `bfnos` |
| `ApproximateNoveltySearch` | `approximate.py` | Singh, Lipovetzky, Ramírez & Segovia-Aguas 2021 | nothing | `ans` |
| `BoundaryExtensionFeatures` | `bee.py` | Teichteil-Königsbuch, Ramírez & Lipovetzky 2020 | `variables(state) -> {name: float}` | `bee` |
| `HierarchicalIW` | `hierarchical.py` | Junyent, Gómez & Jonsson 2021 | nothing; `features` optional | `hiw` |

`IW`, `BFWS` and `DualBFWS` take an optional `atoms(state)`, so that novelty can be measured
over something other than `literals`; that is the hook `BoundaryExtensionFeatures` plugs
into. The package exports only the literature's names: `IW` (iterated, or one width with
`width=k`), `SIW`, `BFWS`, `DualBFWS` and the variants above; the fixed-width IW(k) search
and the novelty tables are internals.

**BFWS(R)** is `BFWS(relevant="iw")`. It runs IW(`r_width`) first, on `r_share` of the
expansion budget; `R` is the union of the literals on the paths to every state that strictly
improved `progress`, and if that pre-search reaches the goal its plan is returned. The main
search is BFWS with novelty measured within `(progress, #r)` partitions and ties broken on
`progress` then `#r`. **Quantified novelty** counts the atoms of a state that no earlier state
with a heuristic value at most as good contained, and searches greedily on `(-QN, h)`,
`(h, -QN)` or the binary version. **BFNoS**, the count-based novelty planner, scores a state
by the number of recorded states that contained its rarest tuple and drops the worse half of
the open list whenever it exceeds `open_limit`; a search that trimmed and found nothing is
`failed`, never `exhausted`. **Approximate novelty** keeps sampled tuples in a bank of Bloom
filters, one per `progress` partition until the bank is spent and shared at random
afterwards, and allows widths above 2 without `strict=False` because bounding that cost is
the point. **Boundary extension features** define the atoms of continuous variables online,
as the range of values seen so far grows: a state is novel when it pushes a boundary.
**Hierarchical IW** runs a high-level IW(1) whose expansion is a low-level IW(1) capped at
`low_expansions`; every low-level state whose projection onto the high-level features
differs from its parent's, or that improved `progress`, is a high-level successor.

## Future State Maximization (`planiverse/planners/fsx.py`)

| Class | Reference | What it needs | Tag |
|---|---|---|---|
| `FSXPlanner` | Wissner-Gross & Freer 2013; Plakolb & Strelkovskii 2023 | `successors` only: no goal, no heuristic | `fsx` |

FSX picks the action that leaves the most futures open and nothing else; `option_count`
exposes its measure on its own, a goal-free signal of how close a state is to being stuck.
See [sampling-based.md](sampling-based.md).

## Heuristic search (`planiverse/planners/heuristic/`) and blind baselines (`planiverse/planners/blind.py`)

| Class | File | Reference | Beyond `progress` it needs | Tag |
|---|---|---|---|---|
| `BestFirstSearch` | `bestfirst.py` | greedy best-first; Pohl 1970 for `weight` | nothing; `progress` optional | `gbfs`; `astar` (`weight=1`), `wastar` (`weight=2`) |
| `RestartingWeightedAStar` | `bestfirst.py` | Richter, Thayer & Ruml 2010 | nothing | `rwa` |
| `EpsilonGreedySearch` | `bestfirst.py` | Valenzano, Schaeffer, Sturtevant & Xie 2014 | nothing | `egbfs` |
| `TypeBasedSearch` | `bestfirst.py` | Xie, Müller, Holte & Imai 2014 | nothing | `tgbfs` |
| `DiverseBestFirst` | `bestfirst.py` | Imai & Kishimoto 2011 | nothing | `dbfs` |
| `LocalExplorationSearch` | `bestfirst.py` | Xie, Müller & Holte 2014, 2015 | nothing | `gbfsle` (local search), `gbfslw` (random walks) |
| `EnforcedHillClimbing` | `ehc.py` | Hoffmann & Nebel 2001 | nothing | `ehc` |
| `MonteCarloRandomWalks` | `randomwalk.py` | Nakhost & Müller 2009, 2013 | nothing | `mrw` |
| `BeamSearch`, `BULB` | `beam.py` | Furcy & Koenig 2005 | nothing | `beam`, `bulb` |
| `LimitedDiscrepancySearch`, `IterativeBroadening` | `beam.py` | Harvey & Ginsberg 1995; Ginsberg & Harvey 1992 | nothing; `progress` optional | `lds`, `ib` |
| `LRTAStar`, `RTAAStar` | `realtime.py` | Korf 1990; Koenig & Likhachev 2006 | nothing | `lrta`, `rtaa` |
| `FeatureSpaceSearch` | `fess.py` | Shoham & Schaeffer 2020 | `features(state) -> tuple`; `advisors` optional | `fess` |
| `MultiQueueSearch` | `multiqueue.py` | Helmert 2006; Röger & Helmert 2010 | a list of heuristics | `multi` |
| `BreadthFirstSearch`, `UniformCostSearch`, `IterativeDeepening` | `blind.py` | | nothing | `brfs`, `ucs`, `iddfs` |

The best-first family shares one engine, a heap plus a list of the same entries with a
`removed` flag, so that "the best open node" and "a uniformly random open node" are both
one pop. Weighted A\* reads `action.cost()` when actions have one, else 1. Restarting weighted
A\* shares a `SuccessorCache` between its rounds, so a restart re-reads what the last round
generated instead of asking the simulator again, and reports the weights it ran in
`widths_tried`. Enforced hill-climbing refuses dead-end progress by default, the same refusal
SIW makes, and takes `novelty_width` to bound its legs the way SIW's are. Time-bounded A\*
(Björnsson, Bulitko & Sturtevant 2009) is not here: it walks the agent backwards along its
search tree, which assumes reversible actions.

The three depth-first probes (limited discrepancy, iterative broadening, BULB) and iterative
deepening re-expand states on every iteration. `SuccessorCache` makes those re-expansions
free in simulator calls, so the counted expansions are distinct states; their *time* is
still exponential in `max_depth`, which is why their defaults are small.

[`tree_search.py`](../../planiverse/planners/tree_search.py) holds `TreeSearchPlanner`, a
small best-first search over a `Heuristic` and a `CostFunction`, the worked example the
README's "Writing a planner" section walks through. As a search it is A\* on `progress`,
which the benchmark runs as `astar`.

## Sampling, population and swarm planners (`planiverse/planners/sampling/`)

| Class | File | Reference | Beyond `progress` it needs | Tag |
|---|---|---|---|---|
| `RollingHorizonEvolution` | `rhea.py` | Perez et al. 2013; Gaina et al. 2017, 2021 | nothing | `rhea` |
| `CrossEntropyPlanner`, `RandomShooting` | `cem.py` | Rubinstein 1999; Pinneri et al. 2020 | nothing | `cem`, `shoot` |
| `NestedMonteCarloSearch` | `nested.py` | Cazenave 2009 | nothing | `nmcs` |
| `GoExplore` | `goexplore.py` | Ecoffet et al. 2019, 2021 | nothing; `cell` optional | `goexp` |
| `MAPElitesPlanner` | `qd.py` | Mouret & Clune 2015; Lehman & Stanley 2011 | nothing; `descriptor` optional | `mapel` |
| `KinodynamicTree` | `kinodynamic.py` | Hsu et al. 1997; Şucan & Kavraki 2008; Li et al. 2016 | nothing; `projection` optional | `est`, `kpiece`, `sst` |
| `PlanLocalSearch` | `localsearch.py` | Kirkpatrick et al. 1983; Lourenço et al. 2003 | nothing | `sa` (annealing), `ils` (iterated) |

All of them replay action sequences through `SuccessorCache`
([`planiverse/planners/common.py`](../../planiverse/planners/common.py)), which memoises
`successors` on `literals` and counts an expansion once, so a prefix the population evaluates
a hundred times costs the simulator once. A sequence is scored by where it ends
(`sequences.py`), with the goal-distance heuristic and nothing else: the negated `progress`
of the last state, `+inf` at a goal, `-inf` in a dead end. In their home literature these
planners maximise a game score; here there is no score, only how far from the goal a
sequence leaves the simulator. A gene that is not applicable where it lands is skipped, the
way a game's forward model treats an invalid input. A goal reached during any evaluation ends
the search with that sequence as the plan. The receding-horizon planners commit the first
gene the best sequence actually applied, and fall back to a random live child only when it
applied none.

Actions are drawn from `get_actions()` when the environment provides it, else from the
actions applicable in the current state (`action_vocabulary`).

Two consequences of the cache are worth knowing. Because a warm cache makes a generation
free in expansions, `RollingHorizonEvolution` ends a decision after `generations` generations
as well as after `expansions_per_step`, or it would never end. And the archive planners
(`GoExplore`, `MAPElitesPlanner`, `KinodynamicTree`) key their cells on `literals` by
default, the exact state, so a coarser projection is what makes the cell structure do
anything: with `cell=lambda s: (boxes(s),)` Go-Explore's archive on a Puzznic level has one
cell per block count.

## Add-ons (`planiverse/planners/pruning.py`, `planiverse/planners/macros.py`)

| Class | Reference | What it plugs into | Tag |
|---|---|---|---|
| `DominatedActionPruner` | Jinnai & Fukunaga 2017 | `SuccessorCache(env, pruner=...)`, so the sampling planners stop drawing duplicate actions | none |
| `FocusedMacros`, `MacroPlanner` | Allen et al. 2021 | a greedy best-first over discovered macros and primitives | `macro` |

Under the `successors` contract every child is generated at once and duplicates are caught
on `literals`, so dominated-action pruning saves a tree search nothing; it pays for the
planners that *sample* actions, which is where it is wired in. Macro discovery enumerates
sequences up to `length` from the initial state and a few random-walk probes, keeps the
`count` with the smallest non-empty footprints (the symmetric difference of `literals`),
and spends `share` of the expansion budget doing so.

## Where the implementation makes a choice

Each of these is stated in the class's docstring as well.

- **Approximate novelty** (`approximate.py`): the Bloom filters, the tuple sampling and the
  filter bank are the paper's; the expansion-skipping policy is this library's: a node whose
  novelty exceeds `width` is expanded with probability `1 - generated / space_bound`.
- **Count-based novelty** (`count.py`): the count-novelty of a state is the count of its
  rarest tuple; the trimmed open list drops the worse half above `open_limit`.
- **Hierarchical IW** (`hierarchical.py`): the feature-discovery rule. An atom joins the
  high-level set the first time the low level prunes a state that made it true.
- **Diverse best-first** (`bestfirst.py`): heuristic value `h` is sampled with weight
  `temperature ** (h - h_min)`, over `h` alone.
- **Arvand** (`randomwalk.py`): the dead-end avoidance bias is `exp(-bias * dead_end_rate)`
  over the first action of a walk.
- **FESS** (`fess.py`): cells keyed by the feature vector, round-robin over cells,
  least-weight move first, advisors marking moves. The Sokoban-specific machinery is not here.
- **KPIECE** (`kinodynamic.py`): the exterior-cell preference has no counterpart in a
  symbolic projection and is left out of the importance.
- **Focused macros** (`macros.py`): exhaustive enumeration at small lengths in place of a
  best-first search over sequences, with the same footprint criterion.
- **FSX** (`fsx.py`): two readings of "the space of futures" are offered, a count of distinct
  reachable states and an entropy over them (`measure="count"` or `"entropy"`), and neither
  is claimed as the paper's.

## What is not here, and why

This is a planning library: a planner is pointed at an instance and given a goal test and,
at most, a distance to the goal. That rules out three kinds of method.

- **Methods defined by an accumulated reward**: 2BFS (Lipovetzky, Ramírez & Geffner 2015),
  prioritised IW (Shleyfman, Tuisov & Domshlak 2016), Fractal Monte Carlo (Hernández Cerezo &
  Duran Ballester 2018), the rollout algorithm (Bertsekas, Tsitsiklis & Wu 1997), and the
  reward-driven width planners (Rollout IW). Each could be run with `progress(root) -
  progress(node)` standing in for the reward, but the mechanism is the reward's.
- **Monte Carlo tree search and its descendants**: UCT, THTS, open-loop tree search, and
  AlphaZero-style search. MCTS wants a reward to back up.
- **Anything with a trained component**: π-IW, learned features for IW, world-model
  planners (PETS, PlaNet, Dreamer, TD-MPC), model-learning pipelines (LOCM, SAM), LLM-guided
  search, and NRPA, which adapts a playout policy as it plans. Regression, plan-space, SAT and
  symbolic planners are out for a different reason: they need an action model, which a
  simulator does not give.

## Running them in the benchmark

Every planner above is registered in
[`planiverse/benchmark/planners.py`](../../planiverse/benchmark/planners.py) under the tags
in the tables, with a tag per documented variant, and `planiverse-bench generate` and
`report` include all of them; `--reference` keeps either to `bfws`, `iw`, `siw` and `fsx`.
`solve` accepts any tag. The seeded planners run under five seeds. Boundary-extension
features and multi-queue alternation want more than one number from an environment, which
the benchmark has only as the progress measure, so they run over it: `bee` is BFWS whose
atoms are the measure's boundary extensions over the literals, and `multi` alternates the
measure with FSX's option count; `fess` and `kpiece` likewise take the measure as their one
feature and their projection. See [docs/benchmark.md](../benchmark.md).

## Files

| Path | What |
|---|---|
| [`common.py`](../../planiverse/planners/common.py) | `SuccessorCache`, `action_vocabulary`, `remaining`, `finish` |
| [`width/`](../../planiverse/planners/width/) | the width-based planners |
| [`fsx.py`](../../planiverse/planners/fsx.py) | `FSXPlanner`, `option_count` |
| [`heuristic/`](../../planiverse/planners/heuristic/) | the best-first family, EHC, random walks, beam, real-time, FESS, alternation |
| [`sampling/`](../../planiverse/planners/sampling/) | RHEA, CEM, NMCS, Go-Explore, MAP-Elites, kinodynamic trees, local search |
| [`blind.py`](../../planiverse/planners/blind.py) | the three blind baselines |
| [`pruning.py`](../../planiverse/planners/pruning.py), [`macros.py`](../../planiverse/planners/macros.py) | the add-ons |
| [`tree_search.py`](../../planiverse/planners/tree_search.py) | `TreeSearchPlanner`, `Heuristic`, `CostFunction` |
| [`benchmark/planners.py`](../../planiverse/benchmark/planners.py) | the benchmark registry |
| [`tests/test_width_planners.py`](../../tests/test_width_planners.py), [`tests/test_sampling_planners.py`](../../tests/test_sampling_planners.py), [`tests/test_planners.py`](../../tests/test_planners.py) | the tests |
