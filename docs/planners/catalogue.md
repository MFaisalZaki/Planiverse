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
from planiverse.planners.heuristic import BULB, LocalExplorationSearch
from planiverse.planners.sampling import GoExplore, KinodynamicTree

env.set_index(0)
budget = Budget(max_expansions=5000, max_seconds=60)
for planner in (BFWS(progress=boxes, relevant="iw"), GoExplore(progress=boxes, seed=0)):
    result = planner.solve(env, budget)
    print(planner.__class__.__name__, result.status, len(result), result.statistics)
```

The last column of each table is the tag the benchmark runs the planner under; see
[Running them in the benchmark](#running-them-in-the-benchmark).

## Width-based (`planiverse/planners/width/`)

[width-based.md](width-based.md) describes novelty, `IW` and `BFWS` and what changes when
the task is a simulator.

| Class | File | Reference | Beyond `progress` it needs | Tag |
|---|---|---|---|---|
| `IW` | `iw.py` | Lipovetzky & Geffner 2012, 2015 | nothing | `iw` |
| `BFWS` | `bfws.py` | Lipovetzky & Geffner 2017 | nothing | `bfws` |
| `BFWS(relevant="iw")`, BFWS(R) | `bfws.py` | Lipovetzky & Geffner 2017; Francès et al. 2017 | nothing | `bfwsr` |
| `QuantifiedNoveltySearch` | `quantified.py` | Katz, Lipovetzky, Moshkovich & Tuisov 2017 | nothing | `qn` |
| `BFNoS` | `count.py` | Rosa & Lipovetzky 2024 | nothing | `bfnos` |
| `ApproximateNoveltySearch` | `approximate.py` | Singh, Lipovetzky, Ramírez & Segovia-Aguas 2021 | nothing | `ans` |

`IW` and `BFWS` take an optional `atoms(state)`, so that novelty can be measured over
something other than `literals`. The package exports only the literature's names: `IW`
(iterated, or one width with `width=k`), `BFWS` and the variants above; the fixed-width
IW(k) search and the novelty tables are internals.

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
the point.

## Heuristic search (`planiverse/planners/heuristic/`)

| Class | File | Reference | Beyond `progress` it needs | Tag |
|---|---|---|---|---|
| `BestFirstSearch` | `bestfirst.py` | greedy best-first; Pohl 1970 for `weight` | nothing; `progress` optional | `gbfs`; `astar` (`weight=1`), `wastar` (`weight=2`) |
| `RestartingWeightedAStar` | `bestfirst.py` | Richter, Thayer & Ruml 2010 | nothing | `rwa` |
| `EpsilonGreedySearch` | `bestfirst.py` | Valenzano, Schaeffer, Sturtevant & Xie 2014 | nothing | `egbfs` |
| `TypeBasedSearch` | `bestfirst.py` | Xie, Müller, Holte & Imai 2014 | nothing | `tgbfs` |
| `DiverseBestFirst` | `bestfirst.py` | Imai & Kishimoto 2011 | nothing | `dbfs` |
| `LocalExplorationSearch` | `bestfirst.py` | Xie, Müller & Holte 2014, 2015 | nothing | `gbfslw` |
| `BULB` | `beam.py` | Furcy & Koenig 2005 | nothing | `bulb` |
| `LimitedDiscrepancySearch` | `beam.py` | Harvey & Ginsberg 1995 | nothing; `progress` optional | `lds` |
| `FeatureSpaceSearch` | `fess.py` | Shoham & Schaeffer 2020 | `features(state) -> tuple`; `advisors` optional | `fess` |

The best-first family shares one engine, a heap plus a list of the same entries with a
`removed` flag, so that "the best open node" and "a uniformly random open node" are both
one pop. Weighted A\* reads `action.cost()` when actions have one, else 1. Restarting weighted
A\* shares a `SuccessorCache` between its rounds, so a restart re-reads what the last round
generated instead of asking the simulator again, and reports the weights it ran in
`widths_tried`. Local exploration is the paper's random-walk variant, GBFS-LRW: when the best
heuristic value has not improved for `patience` expansions, `walks` random walks of up to
`walk_length` actions run from the node just expanded, and their endpoints join the open
list. Time-bounded A\* (Björnsson, Bulitko & Sturtevant 2009) is not here: it walks the agent
backwards along its search tree, which assumes reversible actions.

The two depth-first probes (limited discrepancy and BULB) re-expand states on every
iteration. `SuccessorCache` makes those re-expansions free in simulator calls, so the counted
expansions are distinct states; their *time* is still exponential in `max_depth`, which is why
their defaults are small.

## Sampling (`planiverse/planners/sampling/`)

[sampling-based.md](sampling-based.md) describes both planners and how they differ from the
searches above.

| Class | File | Reference | Beyond `progress` it needs | Tag |
|---|---|---|---|---|
| `GoExplore` | `goexplore.py` | Ecoffet et al. 2019, 2021 | nothing; `cell` optional | `goexp` |
| `KinodynamicTree` | `kinodynamic.py` | Şucan & Kavraki 2008 | nothing; `projection` optional | `kpiece` |

Both explore from a cell with random actions through `SuccessorCache`
([`planiverse/planners/common.py`](../../planiverse/planners/common.py)), which memoises
`successors` on the state and counts an expansion once, so a state explored from a hundred
times costs the simulator once. Both stop at the first goal they reach, with the trajectory
to it as the plan. They key their cells on `literals` by default, the exact state, so a
coarser projection is what makes the cell structure do anything: with
`cell=lambda s: (boxes(s),)` Go-Explore's archive on a Puzznic level has one cell per block
count.

## Add-ons (`planiverse/planners/pruning.py`, `planiverse/planners/macros.py`)

| Class | Reference | What it plugs into | Tag |
|---|---|---|---|
| `DominatedActionPruner` | Jinnai & Fukunaga 2017 | `SuccessorCache(env, pruner=...)`, so a planner that draws a random child stops drawing duplicate actions | none |
| `FocusedMacros`, `MacroPlanner` | Allen et al. 2021 | a greedy best-first over discovered macros and primitives | `macro` |

Under the `successors` contract every child is generated at once and duplicates are caught
on `literals`, so dominated-action pruning saves a search that expands every child nothing;
it pays for a planner that draws one child at random, which is where it is wired in. Macro
discovery enumerates sequences up to `length` from the initial state and a few random-walk
probes, keeps the `count` with the smallest non-empty footprints (the symmetric difference
of `literals`), and spends `share` of the expansion budget doing so.

## Where the implementation makes a choice

Each of these is stated in the class's docstring as well.

- **Approximate novelty** (`approximate.py`): the Bloom filters, the tuple sampling and the
  filter bank are the paper's; the expansion-skipping policy is this library's: a node whose
  novelty exceeds `width` is expanded with probability `1 - generated / space_bound`.
- **Count-based novelty** (`count.py`): the count-novelty of a state is the count of its
  rarest tuple; the trimmed open list drops the worse half above `open_limit`.
- **Diverse best-first** (`bestfirst.py`): heuristic value `h` is sampled with weight
  `temperature ** (h - h_min)`, over `h` alone.
- **FESS** (`fess.py`): cells keyed by the feature vector, round-robin over cells,
  least-weight move first, advisors marking moves. The Sokoban-specific machinery is not here.
- **KPIECE** (`kinodynamic.py`): the exterior-cell preference has no counterpart in a
  symbolic projection and is left out of the importance.
- **Focused macros** (`macros.py`): exhaustive enumeration at small lengths in place of a
  best-first search over sequences, with the same footprint criterion.

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
`report` include all of them; `--reference` keeps either to `bfws` and `iw`. `solve` accepts
any tag. The seeded planners run under five seeds. FESS and KPIECE want a feature vector and
a projection, which the benchmark has only as the progress measure, so `fess` and `kpiece`
take the measure as their one feature and their projection. See
[docs/benchmark.md](../benchmark.md).

## Files

| Path | What |
|---|---|
| [`common.py`](../../planiverse/planners/common.py) | `SuccessorCache`, `finish` |
| [`width/`](../../planiverse/planners/width/) | the width-based planners |
| [`heuristic/`](../../planiverse/planners/heuristic/) | the best-first family, BULB, limited discrepancy search, FESS |
| [`sampling/`](../../planiverse/planners/sampling/) | Go-Explore, KPIECE |
| [`pruning.py`](../../planiverse/planners/pruning.py), [`macros.py`](../../planiverse/planners/macros.py) | the add-ons |
| [`benchmark/planners.py`](../../planiverse/benchmark/planners.py) | the benchmark registry |
| [`tests/test_width_planners.py`](../../tests/test_width_planners.py), [`tests/test_planners.py`](../../tests/test_planners.py) | the tests |
