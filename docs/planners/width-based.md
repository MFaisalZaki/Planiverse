# Width-based planners

This module implements five searches built on novelty, against the `successors()` / `literals`
contract and nothing else.

The *novelty* of a state (i.e., the size of the smallest tuple of its atoms that has not appeared
in any state seen before) is what all five order or filter on. A state with a brand-new atom has
novelty 1, whereas one whose atoms have all been seen individually but which combines two of them
in a new way has novelty 2.

| Planner | Novelty is used as | Complete? |
|---|---|---|
| `IWSearch` | a filter: fail the test, get discarded | no, for fixed width |
| `IteratedWidth` | the same, at width 1, then 2, … | up to `max_width` |
| `SIWSearch` | the same, in short legs that each make progress | no |
| `BFWSSearch` | a sort key: nothing is discarded | yes, unless `prune=True` |
| `IteratedBFWS` | a filter in cheap early rounds, a sort key in the last | yes |

```python
from planiverse.planners.width import IWSearch, BFWSSearch, Budget

env.set_index(0)
result = IWSearch(width=2).solve(env, Budget(max_expansions=5000, max_seconds=60))
if result:
    env.validate(result.plan)
print(result.statistics)   # 120 expansions, 443 generated, 0 pruned by novelty, ...
```

## Planning against a simulator

**There is no goal decomposition.** `is_goal` is a black-box predicate, so the unachieved-goal
count that SIW and BFWS classically lean on does not exist, and there is nothing to be one closer
to. Both take a `progress(state)` callback instead, where lower is better:

```python
BFWSSearch(width=1, progress=lambda s: s.contaminated).solve(env)
SIWSearch(width=1, progress=lambda s: s.blocks_remaining).solve(env)
```

Without one, BFWS becomes breadth-first search ordered by novelty alone and SIW becomes a single
IW call, which both planners report rather than hiding.

**Expansions are expensive.** A power-grid expansion takes 8 to 19 seconds, and a water-network
one is a full hydraulic and transport solve. So every search takes a `Budget` and returns
`SearchStatistics`, because "found nothing" and "ran out of budget after four nodes" are different
answers, and a planner treating them as the same is not usable here.

**Dead ends are real.** A PDDL benchmark usually has none, whereas several environments here do,
and `is_terminal` states are dropped rather than expanded; see [SIW](#siw-and-dead-ends) for what
that is worth.

**The atoms are whatever `literals` says.** A planner cannot see inside the transition, so each
environment's decision about how coarsely to spell its state is what fixes its width. The water
network buckets its contamination into twentieths, and had it kept the raw float every state would
carry a new atom, every state would have novelty 1, and novelty would prune nothing. Environments
here bucket deliberately for that reason.

## IW(k) is incomplete

The incompleteness can be watched on Puzznic level 1, which needs no dependencies:

| Planner | Result | Plan | Expansions |
|---|---|---|---|
| IW(1) | exhausted | | 32 |
| IW(2) | solved | 10 | 120 |
| IteratedWidth(2) | solved at width 2 | 10 | 152 |
| BFWS(1) | solved | 12 | 90 |
| SIW(2) | solved | 12 | 76 |

IW(1) runs out of states rather than time: everything reachable with a new atom has been seen, and
the plan needs a state that is only novel as a pair. That is the incompleteness, and it is what
`IteratedWidth` exists for.

Note the shape of the trade. BFWS expands fewest but returns a longer plan, since it is not
optimal, and IteratedWidth pays for the failed width-1 round on top of the width-2 one, because
each width restarts from scratch. Against a simulator that re-expansion is not free.

## SIW and dead ends

SIW chains short IW searches, each stopping at the first state that improves `progress`. It is
incomplete because each leg commits irrevocably to that first improvement, and greedy progress can
be a trap.

That is not hypothetical here. On Puzznic level 1 the first leg clears a pair and lands on a board
with one block of a colour left, which can never be matched. The position is a dead end and the
whole search is over, so classical SIW fails the level that IW(2) solves. A simulator that
computes `is_terminal` gives the leg a cheap way to refuse, since a dead end is not progress, and
that alone turns the failure into a solved instance:

```python
SIWSearch(width=2, progress=boxes, avoid_dead_ends=False).solve(env)   # fails
SIWSearch(width=2, progress=boxes, avoid_dead_ends=True).solve(env)    # solved, 12 actions
```

`avoid_dead_ends` defaults to `True`. Set it `False` for the classical behaviour.

## Iterated BFWS

Plain BFWS is already complete, so `IteratedBFWS` should not be confused with `IteratedWidth`'s
cure for incompleteness: it is a budget strategy, after the Dual-BFWS shape in Lipovetzky and
Geffner's 2017 paper.

Its rounds run k-BFWS, `BFWSSearch(prune=True)`, which keeps BFWS's `<novelty, progress,
heuristic>` ordering but discards states whose novelty exceeds the width, the way IW does. Each
round gets IW's bounded frontier, so it is cheap, while the ordering inside it heads for the goal
instead of sweeping breadth-first. On Puzznic level 1 the pruned width-1 round exhausts at exactly
IW(1)'s 32 expansions, since it is the same filter over the same reach, and the width-2 round
solves it.

The rounds escalate width only while the filter is discarding something, and if every allowed
width fails, the last of the budget goes to one unpruned round, plain BFWS(1), which is complete:

```python
IteratedBFWS(max_width=2, progress=boxes).solve(env, Budget(max_expansions=500))
```

`IteratedBFWS` reports `exhausted` only when it proved there is no plan: a pruned round that
emptied its frontier without discarding anything saw the whole reachable space, and the unpruned
round discards nothing by construction. Hitting `max_width` with the filter still biting proves
nothing and reports `failed`, and `IteratedWidth` draws the same line, because the benchmark reads
`exhausted` as unsolvability (`catalogue.is_complete`).

## Two notes on novelty

**Width 2 costs O(n²) per state, and width 3 costs O(n³).** With a hundred atoms that is 5,000
pairs and 160,000 triples per state, against a simulator whose successor is already expensive. So
`NoveltyTable` refuses widths above 2 unless `strict=False` is passed, and reports
`tuples_enumerated` so the cost is visible.

**Two definitions of novelty are available**, and the standard one is the default. The
[pyBehaviourPlanningLTL](https://github.com/MFaisalZaki/pyBehaviourPlanningLTL) planners this
module was modelled on count how many atoms are new relative to the *path* taken to the state, and
keep the state when that count is at least k, which is not the standard measure:

- At width 1 the two agree: does this state have an atom nobody has seen.
- At width 2 they part company. The path rule asks for two new atoms; standard novelty asks for
  one new pair, and a state can satisfy either without the other.
- The path rule is path-based rather than search-based, so the same state can be novel down one
  branch and not another, which makes results depend on visit order.

The path rule is available as `IWSearch(novelty_rule="path")` and `novelty.path_novelty`, for
comparability.

## Files

| Path | What |
|---|---|
| [`novelty.py`](../../planiverse/planners/width/novelty.py) | `NoveltyTable`, `PartitionedNovelty`, `path_novelty` |
| [`iw.py`](../../planiverse/planners/width/iw.py) | `IWSearch`, `IteratedWidth`, `SIWSearch` |
| [`bfws.py`](../../planiverse/planners/width/bfws.py) | `BFWSSearch`, `IteratedBFWS` |
| [`result.py`](../../planiverse/planners/width/result.py) | `SearchResult`, `SearchStatistics`, `Budget` |
| [`tests/test_width_planners.py`](../../tests/test_width_planners.py) | Tests |
