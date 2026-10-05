# The planners

Every planner in the library, against the same `successors` / `literals` / `is_goal` /
`is_terminal` contract. Every one is deterministic: the same instance under the same budget
gives the same plan. None learns before it plans, none is a Monte Carlo tree search, and
**none takes a reward**: each is driven by the black-box goal test and, at most, the
`progress` heuristic (how far a state is from the goal, lower is better). Every planner has
`solve(env, budget=None, state=None) -> SearchResult`, takes a `Budget`, and reports
`SearchStatistics`. Nothing below needs anything else from an environment, except where a
column says so.

```python
from planiverse.planners.width import BFWS, BFNoS, Budget
from planiverse.planners.heuristic import EnforcedHillClimbing, BULB

env.set_index(0)
budget = Budget(max_expansions=5000, max_seconds=60)
for planner in (BFWS(progress=boxes, relevant="iw"), EnforcedHillClimbing(progress=boxes)):
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
`failed`, never `exhausted`. **Boundary extension features** define the atoms of continuous
variables online, as the range of values seen so far grows: a state is novel when it pushes
a boundary. **Hierarchical IW** runs a high-level IW(1) whose expansion is a low-level IW(1)
capped at `low_expansions`; every low-level state whose projection onto the high-level
features differs from its parent's, or that improved `progress`, is a high-level successor.

## Heuristic search (`planiverse/planners/heuristic/`) and blind baselines (`planiverse/planners/blind.py`)

| Class | File | Reference | Beyond `progress` it needs | Tag |
|---|---|---|---|---|
| `BestFirstSearch` | `bestfirst.py` | greedy best-first; Pohl 1970 for `weight` | nothing; `progress` optional | `gbfs`; `astar` (`weight=1`), `wastar` (`weight=2`) |
| `RestartingWeightedAStar` | `bestfirst.py` | Richter, Thayer & Ruml 2010 | nothing | `rwa` |
| `EnforcedHillClimbing` | `ehc.py` | Hoffmann & Nebel 2001 | nothing | `ehc` |
| `BeamSearch`, `BULB` | `beam.py` | Furcy & Koenig 2005 | nothing | `beam`, `bulb` |
| `LimitedDiscrepancySearch`, `IterativeBroadening` | `beam.py` | Harvey & Ginsberg 1995; Ginsberg & Harvey 1992 | nothing; `progress` optional | `lds`, `ib` |
| `LRTAStar`, `RTAAStar` | `realtime.py` | Korf 1990; Koenig & Likhachev 2006 | nothing | `lrta`, `rtaa` |
| `FeatureSpaceSearch` | `fess.py` | Shoham & Schaeffer 2020 | `features(state) -> tuple`; `advisors` optional | `fess` |
| `MultiQueueSearch` | `multiqueue.py` | Helmert 2006; Röger & Helmert 2010 | a list of heuristics | `multi` |
| `BreadthFirstSearch`, `UniformCostSearch`, `IterativeDeepening` | `blind.py` | | nothing | `brfs`, `ucs`, `iddfs` |

Greedy best-first and weighted A\* share one engine, a heap whose ties break by insertion
order. Weighted A\* reads `action.cost()` when actions have one, else 1. Restarting weighted
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

Multi-queue alternation keeps one open list per heuristic and expands them round-robin; a
queue that just improved its best value gets `boost` extra turns. It is the framework in
which a goal-directed `progress` measure and a novelty measure search together.

[`tree_search.py`](../../planiverse/planners/tree_search.py) holds `TreeSearchPlanner`, a
small best-first search over a `Heuristic` and a `CostFunction`, the worked example the
README's "Writing a planner" section walks through. As a search it is A\* on `progress`,
which the benchmark runs as `astar`.

## Add-on (`planiverse/planners/pruning.py`)

| Class | Reference | What it plugs into |
|---|---|---|
| `DominatedActionPruner` | Jinnai & Fukunaga 2017 | `SuccessorCache(env, pruner=...)`, for a planner that draws actions from a vocabulary |

Under the `successors` contract every child is generated at once and duplicates are caught
on `literals`, so dominated-action pruning saves a tree search nothing; it pays for a planner
that draws actions from `action_vocabulary` rather than expanding states, which is where it
is wired in. It is not a planner and has no benchmark tag.

## Where the implementation makes a choice

Each of these is stated in the class's docstring as well.

- **Count-based novelty** (`count.py`): the count-novelty of a state is the count of its
  rarest tuple; the trimmed open list drops the worse half above `open_limit`.
- **Hierarchical IW** (`hierarchical.py`): the feature-discovery rule. An atom joins the
  high-level set the first time the low level prunes a state that made it true.
- **FESS** (`fess.py`): cells keyed by the feature vector, round-robin over cells,
  least-weight move first, advisors marking moves. The Sokoban-specific machinery is not here.

## What is not here, and why

This is a planning library of deterministic planners: a planner is pointed at an instance
and given a goal test and, at most, a distance to the goal, and two runs give the same
answer. That rules out four kinds of method.

- **Randomised search**: ε-greedy, type-based and diverse best-first search, local
  exploration by random walks, Monte Carlo random walks (Arvand), approximate novelty over
  sampled tuples, and the sampling and population planners (rolling-horizon evolution, the
  cross-entropy method, random shooting, nested Monte Carlo search, Go-Explore, MAP-Elites,
  the kinodynamic trees EST, KPIECE and SST, local search over action sequences, macro
  discovery by random probes) and Future State Maximization, whose walkers are random. Each
  draws from a seed, and two runs under different seeds give different answers.
- **Methods defined by an accumulated reward**: 2BFS (Lipovetzky, Ramírez & Geffner 2015),
  prioritised IW (Shleyfman, Tuisov & Domshlak 2016), Fractal Monte Carlo (Hernández Cerezo &
  Duran Ballester 2018), the rollout algorithm (Bertsekas, Tsitsiklis & Wu 1997) and Rollout
  IW. The mechanism is the reward's.
- **Monte Carlo tree search and its descendants**: UCT, THTS, open-loop tree search, and
  AlphaZero-style search. MCTS wants a reward to back up, and samples.
- **Anything with a trained component**: π-IW, learned features for IW, world-model
  planners (PETS, PlaNet, Dreamer, TD-MPC), model-learning pipelines (LOCM, SAM), LLM-guided
  search, and NRPA, which adapts a playout policy as it plans. Regression, plan-space, SAT and
  symbolic planners are out for a different reason: they need an action model, which a
  simulator does not give.

## Running them in the benchmark

Every planner above is registered in
[`planiverse/benchmark/planners.py`](../../planiverse/benchmark/planners.py) under the tags
in the tables, with a tag per documented variant, and `planiverse-bench generate` and
`report` include all of them, once per instance. `solve` accepts any tag. Boundary-extension
features and multi-queue alternation want more than one number from an environment, which
the benchmark has only as the progress measure, so they run over it: `bee` is BFWS whose
atoms are the measure's boundary extensions over the literals, and `multi` alternates the
measure with quantified novelty over it; `fess` likewise takes the measure as its one
feature. See [docs/benchmark.md](../benchmark.md).

## Files

| Path | What |
|---|---|
| [`common.py`](../../planiverse/planners/common.py) | `SuccessorCache`, `action_vocabulary`, `remaining`, `finish` |
| [`width/`](../../planiverse/planners/width/) | the width-based planners |
| [`heuristic/`](../../planiverse/planners/heuristic/) | the best-first family, EHC, beam, real-time, FESS, alternation |
| [`blind.py`](../../planiverse/planners/blind.py) | the three blind baselines |
| [`pruning.py`](../../planiverse/planners/pruning.py) | the add-on |
| [`tree_search.py`](../../planiverse/planners/tree_search.py) | `TreeSearchPlanner`, `Heuristic`, `CostFunction` |
| [`benchmark/planners.py`](../../planiverse/benchmark/planners.py) | the benchmark registry |
| [`tests/test_width_planners.py`](../../tests/test_width_planners.py), [`tests/test_planners.py`](../../tests/test_planners.py) | the tests |
