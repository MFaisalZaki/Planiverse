# Width-based planners

This module implements the searches built on novelty, against the `successors()` / `literals`
contract and nothing else, and exports them under the names the literature uses: `IW` and
`BFWS` are described here; the other novelty planners (quantified, count-based and approximate
novelty) are in the [catalogue](catalogue.md). IW at one fixed width is `IW(width=k)`; the
novelty tables are internals of `novelty.py`.

The *novelty* of a state (i.e., the size of the smallest tuple of its atoms that has not appeared
in any state seen before) is what IW filters on and BFWS orders on. A state with a brand-new
atom has novelty 1, whereas one whose atoms have all been seen individually but which combines
two of them in a new way has novelty 2.

| Planner | Novelty is used as | Complete? |
|---|---|---|
| `IW(width=k)` | a filter: fail the test, get discarded | no, for fixed width |
| `IW` | the same, at width 1, then 2, … | up to `max_width` |
| `BFWS` | a sort key: nothing is discarded; `relevant="iw"` adds BFWS(R)'s `#r` | yes, unless `prune=True` |
| `BFWS(prune=True)` | a sort key inside IW's filter (k-BFWS) | no |

```python
from planiverse.planners.width import IW, BFWS, Budget

env.set_index(0)
result = IW(width=2).solve(env, Budget(max_expansions=5000, max_seconds=60))
if result:
    env.validate(result.plan)
print(result.statistics)   # expansions, generated, pruned by novelty, ...
```

## Planning against a simulator

**There is no goal decomposition.** `is_goal` is a black-box predicate, so the unachieved-goal
count that BFWS classically leans on does not exist, and there is nothing to be one closer to.
It takes a `progress(state)` callback instead, where lower is better:

```python
BFWS(width=1, progress=lambda s: s.contaminated).solve(env)
BFWS(width=1, progress=lambda s: s.blocks_remaining, relevant="iw").solve(env)   # BFWS(R)
```

Without one, BFWS becomes breadth-first search ordered by novelty alone. IW takes no callback.

**Expansions are expensive.** A water-network expansion is a full hydraulic and transport solve.
So every search takes a `Budget` and returns `SearchStatistics`, because "found nothing" and "ran
out of budget after four nodes" are different answers.

**Dead ends are real.** A PDDL benchmark usually has none, whereas several environments here do,
and `is_terminal` states are dropped rather than expanded. A goal is tested before the terminal
test, because every absorbing goal state in this library is terminal as well as a goal.

**The atoms are whatever `literals` says.** A planner cannot see inside the transition, so each
environment's decision about how coarsely to spell its state is what fixes its width. The water
network buckets its contamination into twentieths, and had it kept the raw float every state would
carry a new atom, every state would have novelty 1, and novelty would prune nothing. Environments
here bucket deliberately for that reason.

## IW(k) is incomplete

IW(k) discards every state of novelty above k, so when the plan needs a state that is only
novel as a pair, IW(1) runs out of states rather than time: everything reachable with a new
atom has been seen. That is the incompleteness, and it is what iterated `IW` exists for: it
runs width 1, then 2, and so on up to `max_width`. Each width restarts from scratch, so a
failed round's expansions are paid again in the next.

## k-BFWS

`BFWS(prune=True)` keeps BFWS's `<novelty, progress, heuristic>` ordering but discards states
whose novelty exceeds the width, the way IW does. It has IW's bounded frontier and IW's
incompleteness, and the ordering inside it heads for the goal instead of sweeping
breadth-first. On Puzznic level 1, which IW(1) exhausts, k-BFWS at width 1 exhausts too, after
the same number of expansions: the filter, not the ordering, is what bounds what it can see.

```python
BFWS(width=1, prune=True, progress=boxes).solve(env, Budget(max_expansions=500))
```

Leave `prune` at its default, `False`, for the complete search.

## Two notes on novelty

**Width 2 costs O(n²) per state, and width 3 costs O(n³).** With a hundred atoms that is 5,000
pairs and 160,000 triples per state, against a simulator whose successor is already expensive. So
`NoveltyTable` refuses widths above 2 unless `strict=False` is passed, and reports
`tuples_enumerated` so the cost is visible.

**Two definitions of novelty are available**, and the standard one is the default. The
[pyBehaviourPlanningLTL](https://github.com/MFaisalZaki/pyBehaviourPlanningLTL) planners count
how many atoms are new relative to the *path* taken to the state, and keep the state when that
count is at least k, which is not the standard measure:

- At width 1 the two agree: does this state have an atom nobody has seen.
- At width 2 they part company. The path rule asks for two new atoms; standard novelty asks for
  one new pair, and a state can satisfy either without the other.
- The path rule is path-based rather than search-based, so the same state can be novel down one
  branch and not another, which makes results depend on visit order.

The path rule is available as `IW(novelty_rule="path")` and `novelty.path_novelty`, for
comparability.

## Files

| Path | What |
|---|---|
| [`novelty.py`](../../planiverse/planners/width/novelty.py) | `NoveltyTable`, `PartitionedNovelty`, `path_novelty` |
| [`iw.py`](../../planiverse/planners/width/iw.py) | `IW`, and the fixed-width `IWSearch` it is built from |
| [`bfws.py`](../../planiverse/planners/width/bfws.py) | `BFWS` (with `relevant="iw"` for BFWS(R), `prune=True` for k-BFWS) |
| [`result.py`](../../planiverse/planners/width/result.py) | `SearchResult`, `SearchStatistics`, `Budget` |
| [`tests/test_width_planners.py`](../../tests/test_width_planners.py) | Tests |
