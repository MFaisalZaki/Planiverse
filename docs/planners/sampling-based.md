# Sampling-based planners

Two planners that explore with random actions from states they have already reached, rather
than expand states in an order a heuristic fixes. Both keep **cells**, a projection of the
state, pick a cell, restore a state in it and walk forward from there; both stop at the first
goal a walk reaches, with the trajectory to it as the plan. Neither trains anything and neither
takes a reward.

| Planner | Class | Picks the next cell by | Uses `progress` | Tag |
|---|---|---|---|---|
| Go-Explore | `GoExplore` | how rarely it has been chosen and seen | to score a cell's trajectory | `goexp` |
| KPIECE | `KinodynamicTree` | an importance that falls with use and with population | no | `kpiece` |

```python
from planiverse.planners.sampling import GoExplore, KinodynamicTree
from planiverse.planners.width import Budget

budget = Budget(max_expansions=5000, max_seconds=60)
GoExplore(progress=boxes, cell=lambda s: (boxes(s),), seed=0).solve(env, budget)
KinodynamicTree(projection=lambda s: (boxes(s),), seed=0).solve(env, budget)
```

Both are randomised, so both take a `seed`, and the benchmark runs them under five.

## Go-Explore

- **Class:** `GoExplore`, in
  [`planiverse/planners/sampling/goexplore.py`](../../planiverse/planners/sampling/goexplore.py)
- **References:** Ecoffet, Huizinga, Lehman, Stanley & Clune, *Go-Explore: a New Approach for
  Hard-Exploration Problems*, arXiv 1901.10995 (2019), and *First Return, Then Explore*,
  Nature 590 (2021).

This is the exploration phase. The archive holds one entry per cell, the best trajectory that
reached it. Each round picks a cell with a weight that favours cells rarely chosen and rarely
seen, **returns** to it by restoring its state, which costs nothing in a deterministic,
restorable simulator, and **explores** from it for `explore_steps` random actions, repeating
the previous action with probability `repeat`. Every new cell reached joins the archive, and a
cell's trajectory is replaced by one whose end state scores better, or as well with a shorter
plan; a state's score is its negated `progress`, and without `progress` every cell scores the
same and shorter trajectories win.

The selection weight is the paper's: a sum over *times chosen*, *times chosen since a new cell
was found from it* and *times seen* of `w · (1 / (v + 0.001)) ** 0.5 + 0.00001`, with weights
`0.1`, `0.1` and `0.3`. `repeat` is the paper's 0.95; `explore_steps` is 20 rather than the
paper's 100, because an expansion costs more here than an Atari frame. The paper's second
phase, which trains a policy on the trajectories found, is training and is not here.

## KPIECE

- **Class:** `KinodynamicTree`, in
  [`planiverse/planners/sampling/kinodynamic.py`](../../planiverse/planners/sampling/kinodynamic.py)
- **Reference:** Şucan & Kavraki, *Kinodynamic Motion Planning by Interior-Exterior Cell
  Exploration*, WAFR 2008.

KPIECE comes from motion planning, where a system's dynamics are available only as a forward
propagator: no inverse, and no distance one can steer along. It grows a tree over a grid of
cells on a low-dimensional `projection`. Each iteration picks the cell of highest importance,
`log(iteration + 1) / ((1 + times selected) · states in the cell)`, ties broken at random, a
state in it at random, and propagates between 1 and `max_duration` random actions from that
state. Every state the propagation reaches that has not been seen before joins the tree, since
under `successors` it was generated anyway; a dead end ends the propagation.

KPIECE also prefers cells on the exterior of the covered region. A symbolic projection has no
neighbouring cells to count, so that term is left out of the importance.

## Cells

Both planners key their cells on `literals` by default, the exact state, which makes every cell
hold one state and the cell structure do nothing. A coarser projection is what makes it work: the
paper's own choice for Atari was a downsampled frame, and on a Puzznic level
`cell=lambda s: (boxes(s),)` gives Go-Explore one cell per block count. The benchmark's `kpiece`
projects onto the environment's progress measure; its `goexp` runs on the exact state.

Both expand through `SuccessorCache`
([`planiverse/planners/common.py`](../../planiverse/planners/common.py)), so a state explored
from a hundred times costs the simulator once.

## Files

| Path | What |
|---|---|
| [`goexplore.py`](../../planiverse/planners/sampling/goexplore.py) | `GoExplore` |
| [`kinodynamic.py`](../../planiverse/planners/sampling/kinodynamic.py) | `KinodynamicTree` |
| [`tests/test_planners.py`](../../tests/test_planners.py) | Tests |
