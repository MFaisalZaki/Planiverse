# Width-based planners

Five searches built on **novelty**, against the `successors()` / `literals` contract and
nothing else.

The novelty of a state is the size of the smallest tuple of its atoms that has not appeared
in any state seen before. A state with a brand-new atom has novelty 1; one whose atoms have
all been seen individually but which combines two of them in a new way has novelty 2.

| Planner | Novelty is used as | Complete? |
|---|---|---|
| `IWSearch` | a **filter** — fail the test, get discarded | no, for fixed width |
| `IteratedWidth` | the same, at width 1, then 2, … | up to `max_width` |
| `SIWSearch` | the same, in short legs that each make progress | no |
| `BFWSSearch` | a **sort key** — nothing is discarded | yes (unless `prune=True`) |
| `IteratedBFWS` | a filter in cheap early rounds, a sort key in the last | yes |

```python
from planiverse.planners.width import IWSearch, BFWSSearch, Budget

env.fix_index(0)
result = IWSearch(width=2).solve(env, Budget(max_expansions=5000, max_seconds=60))
if result:
    env.validate(result.plan)
print(result.statistics)   # 120 expansions, 443 generated, 0 pruned by novelty, ...
```

## What changes when the task is a simulator

**There is no goal decomposition.** This is the big one. `is_goal` is a black-box predicate,
so the unachieved-goal count that SIW and BFWS classically lean on simply does not exist —
there is nothing to be one-closer to. Both take a `progress(state)` callback instead (lower
is better), and both say what they degrade to without one rather than pretending:

```python
BFWSSearch(width=1, progress=lambda s: s.contaminated).solve(env)
SIWSearch(width=1, progress=lambda s: s.blocks_remaining).solve(env)
```

**Expansions are expensive.** A power-grid expansion is 8–19 seconds; a water-network one is
a full hydraulic and transport solve. Every search takes a `Budget` and returns
`SearchStatistics`, because "found nothing" and "ran out of budget after four nodes" are
different answers and a planner that conflates them is not usable.

**Dead ends are real.** A PDDL benchmark usually has none. Six environments here do, and
`is_terminal` states are dropped rather than expanded — see [SIW](#siw-and-the-dead-end-trap)
for what that is worth.

**The atoms are whatever `literals` says.** A planner cannot see inside the transition, so
each environment's decision about how coarsely to spell its state is what fixes its width.
The water network buckets its contamination into twentieths; had it kept the raw float,
every state would carry a new atom, every state would have novelty 1, and novelty would
prune nothing. Environments here bucket deliberately for this reason.

## IW(k) is incomplete, and you can watch it happen

Puzznic level 1, which needs no dependencies:

| Planner | Result | Plan | Expansions |
|---|---|---|---|
| IW(1) | **exhausted** | — | 32 |
| IW(2) | solved | 10 | 120 |
| IteratedWidth(2) | solved at width 2 | 10 | 152 |
| BFWS(1) | solved | 12 | 90 |
| SIW(2) | solved | 12 | 76 |

IW(1) does not run out of *time* — it runs out of *states*. Everything reachable with a new
atom has been seen, and the plan needs a state that is only novel as a pair. That is the
incompleteness, and it is why `IteratedWidth` exists.

Note the shape of the trade: BFWS expands fewest but returns a longer plan (it is not
optimal), and IteratedWidth pays for the failed width-1 round on top of the width-2 one,
because each width restarts from scratch. Against a simulator that re-expansion is not free.

## SIW and the dead-end trap

SIW chains short IW searches, each stopping at the first state that improves `progress`. It
is incomplete because each leg **commits irrevocably** to that first improvement, and greedy
progress can be a trap.

That is not hypothetical here. On Puzznic level 1 the first leg clears a pair and lands on a
board with one block of a colour left — which can never be matched, so the position is a dead
end and the whole search is over. Classical SIW fails the level that IW(2) solves.

A simulator that computes `is_terminal` gives the leg a cheap way to refuse — a dead end is
not progress — and that alone turns the failure into a solved instance:

```python
SIWSearch(width=2, progress=boxes, avoid_dead_ends=False).solve(env)   # fails
SIWSearch(width=2, progress=boxes, avoid_dead_ends=True).solve(env)    # solved, 12 actions
```

`avoid_dead_ends` defaults to `True`. Set it `False` for the classical behaviour.

## Iterated BFWS: polynomial first, complete last

Plain BFWS is already complete, so `IteratedBFWS` is not `IteratedWidth`'s cure for
incompleteness — it is a **budget** strategy, after the Dual-BFWS shape in Lipovetzky and
Geffner's 2017 paper.

Its rounds run **k-BFWS**: `BFWSSearch(prune=True)`, which keeps BFWS's
`<novelty, progress, heuristic>` ordering but discards states whose novelty exceeds the
width, the way IW does. That gives each round IW's bounded frontier — cheap — while the
ordering inside it heads for the goal instead of sweeping breadth-first. On Puzznic level 1
the pruned width-1 round exhausts at exactly IW(1)'s 32 expansions (same filter, same
reach), and the width-2 round solves it.

The rounds escalate width only while the filter is actually discarding something, and if
every allowed width fails, the last of the budget goes to **one unpruned round** — plain
BFWS(1), which is complete:

```python
IteratedBFWS(max_width=2, progress=boxes).solve(env, Budget(max_expansions=500))
```

One word is treated carefully. `IteratedBFWS` reports `exhausted` only when it *proved*
there is no plan: a pruned round that emptied its frontier without discarding anything saw
the whole reachable space, and the unpruned round discards nothing by construction. Hitting
`max_width` with the filter still biting proves nothing and reports `failed` —
`IteratedWidth` draws the same line — because the benchmark reads `exhausted` as
unsolvability (`catalogue.is_complete`).

## Two notes on novelty itself

**Width 2 costs O(n²) per state, width 3 costs O(n³).** With a hundred atoms that is 5,000
pairs and 160,000 triples, per state, against a simulator whose successor is already
expensive. `NoveltyTable` refuses widths above 2 unless you pass `strict=False`, and reports
`tuples_enumerated` so the cost is visible.

**The reference implementation measures novelty differently.** The
[pyBehaviourPlanningLTL](https://github.com/MFaisalZaki/pyBehaviourPlanningLTL) planners this
module was modelled on count how many atoms are new *relative to the path taken to the
state*, and keep it when that count is at least k — with a source comment flagging the rule
as unverified. That is not the standard measure:

- At width 1 the two agree: "does this state have an atom nobody has seen".
- At width 2 they part company. The path rule asks for **two new atoms**; standard novelty
  asks for **one new pair**, and a state can satisfy either without the other.
- The path rule is also path-based rather than search-based, so the same state can be novel
  down one branch and not another, which makes results depend on visit order.

The standard definition is the default here. The other is available as
`IWSearch(novelty_rule="path")` and `novelty.path_novelty`, for comparability.

## Files

| Path | What |
|---|---|
| [`novelty.py`](../../planiverse/planners/width/novelty.py) | `NoveltyTable`, `PartitionedNovelty`, `path_novelty` |
| [`iw.py`](../../planiverse/planners/width/iw.py) | `IWSearch`, `IteratedWidth`, `SIWSearch` |
| [`bfws.py`](../../planiverse/planners/width/bfws.py) | `BFWSSearch`, `IteratedBFWS` |
| [`result.py`](../../planiverse/planners/width/result.py) | `SearchResult`, `SearchStatistics`, `Budget` |
| [`tests/test_width_planners.py`](../../tests/test_width_planners.py) | Tests, including the IW(1)/IW(2) and SIW dead-end results above |
