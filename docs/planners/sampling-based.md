# Future State Maximization

A planner that samples futures instead of enumerating them, and that is told nothing at all:
no goal, no heuristic, no reward.

## Future State Maximization (FSX)

Every other planner here is told what it wants. FSX is not: it picks the action leaving the
largest space of reachable futures, and nothing else, with no goal, no heuristic and no reward.
That it reaches goals at all is a side effect of the fact that being dead, stuck or cornered are
all states with very few futures.

- **Class:** `FSXPlanner`, in [`planiverse/planners/fsx.py`](../../planiverse/planners/fsx.py)
- **References:** Wissner-Gross & Freer, *Causal Entropic Forces*, Phys. Rev. Lett. 110, 168702
  (2013); Plakolb & Strelkovskii, *Applicability of the Future State Maximization Paradigm to
  Agent-Based Modeling*, Systems 11(2), 105 (2023).

```python
from planiverse.planners.fsx import FSXPlanner, option_count

result = FSXPlanner(horizon=8, walkers=12, seed=0).solve(env, Budget(max_seconds=60))
```

### How it works

FSX samples bounded-horizon paths from each candidate action using *walkers* (i.e., virtual
entities probing for a maximisation of possible states), and prefers the action whose walkers
reach the most future states.

We implement two scoring functions, because "maximisation of possible states" can be read either
as a count of distinct reachable states or as an entropy over the distribution of futures, and the
two differ when some futures are much likelier than others. `measure="count"` and
`measure="entropy"` select between them, and we claim neither as the paper's.

Two parameters come from the physics: the causal horizon τ (`horizon`, how far a walker looks) and
the causal path temperature (`temperature`, which turns argmax selection into a Boltzmann sample).

### What it needs, and what it costs

FSX asks the environment for exactly one thing, `successors`. It needs no goal decomposition, no
distance-to-goal and no admissible heuristic. A walker stops at a terminal state or one with no
successors, so a dead end costs an action its score without the scoring having to know it is one.
Each decision costs up to `walkers * horizon` successor calls per candidate action.

It is a policy rather than a search: it commits to one action at a time and never backtracks.

`option_count(env, state)` exposes the measure on its own: a goal-free signal saying how close a state is to being stuck.

## Choosing between them

| | needs a goal | needs a heuristic | backtracks |
|---|---|---|---|
| [IW / BFWS](width-based.md) | yes | no; BFWS takes an optional `progress` | yes |
| [the sampling planners](catalogue.md#sampling-population-and-swarm-planners-planiverseplannerssampling) | yes | no; `progress` optional | some |
| FSX | no | no | no |

## Files

| Path | What |
|---|---|
| [`fsx.py`](../../planiverse/planners/fsx.py) | `FSXPlanner`, `option_count` |
| [`tests/test_sampling_planners.py`](../../tests/test_sampling_planners.py) | Tests |
