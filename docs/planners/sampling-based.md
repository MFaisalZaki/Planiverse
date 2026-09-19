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
distance-to-goal and no admissible heuristic, which are the three things a black-box simulator is
worst at providing, so when there is no idea how to write a heuristic this still runs. It avoids
dead ends structurally rather than by being told to: a state one move from losing has almost no
futures, so it scores badly long before it is reached. FSX would steer away from those states even
if it were never told they were terminal.

It is a policy rather than a search: it commits to one action at a time and never backtracks. On Puzznic level 1 it does solve, but in 128 actions where IW(2) takes 10. That is not
a bug to be tuned away but what a goal-free agent looks like. Its strength is the opposite case,
staying alive and mobile where the danger is getting stuck rather than reaching a specific narrow
target, since a goal down a corridor is by construction in a place with few futures.

`option_count(env, state)` exposes the measure on its own: a goal-free signal saying how close a state is to being stuck.

## Choosing between them

| | needs a goal | needs a heuristic | backtracks | good at |
|---|---|---|---|---|
| [IW / BFWS](width-based.md) | yes | helps a lot | yes | finding short plans |
| [the sampling planners](more-planners.md) | yes | helps a lot | some | long horizons under a receding horizon |
| FSX | no | no | no | staying alive; scoring how stuck a state is |

## Files

| Path | What |
|---|---|
| [`fsx.py`](../../planiverse/planners/fsx.py) | `FSXPlanner`, `option_count` |
| [`tests/test_sampling_planners.py`](../../tests/test_sampling_planners.py) | Tests |
