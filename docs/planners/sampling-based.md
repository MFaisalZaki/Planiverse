# Sampling-based planners: FSX and MCTS

These two planners sample futures instead of enumerating them. They sit at opposite ends of how
much they need to be told: MCTS wants a reward, whereas FSX wants nothing at all.

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

## Monte Carlo Tree Search (UCT)

- **Class:** `MCTSPlanner`, in [`planiverse/planners/mcts.py`](../../planiverse/planners/mcts.py)
- **References:** Kocsis & Szepesvári, *Bandit Based Monte-Carlo Planning* (ECML 2006) for UCT and
  UCB1; Browne et al., *A Survey of Monte Carlo Tree Search Methods* (IEEE TCIAIG 4(1), 2012) for
  the select / expand / simulate / backpropagate loop.

```python
from planiverse.planners.mcts import MCTSPlanner

result = MCTSPlanner(iterations=3000, seed=0,
                     reward=lambda s: 1 - blocks_left(s) / 6).solve(env)
```

MCTS was built for adversarial games with a natural terminal score. Planiverse gives it neither,
so four adaptations follow, each of which is measurable on Puzznic level 1.

**There is no score, only `is_goal`.** The default reward is 1 for a goal, 0 otherwise, less a
small length penalty. A `reward` callback supplies something denser:

| Reward | Result | Plan |
|---|---|---|
| sparse (goal or nothing) | solved | 32 actions |
| dense (blocks cleared) | solved | 16 actions |

**The transition is deterministic**, so the averaging that makes UCT work in stochastic games does
nothing. This keeps the best value seen through a node as well as the mean:

| Backup | Result |
|---|---|
| `"max"` (default) | solved, 16 actions |
| `"mean"` (classical) | out of budget |

**Dead ends are real and absorbing** (i.e., no action leads out of them). Terminal states are
marked in the tree and never selected again, which stops UCT re-exploring a branch it has already
proved is over.

**The rollout keeps the best reward it saw anywhere along the way**, rather than the reward where
it stopped. This one is easy to miss and worth more than the rest put together. Random rollouts in
a domain with dead ends nearly always end in one, so scoring only the final state throws away
everything the rollout learned: the tree sees 0 for every branch and UCT has no gradient to climb.
Without it, MCTS does not solve Puzznic level 1 at all.

The plan returned is the best goal-reaching path found rather than the tree's principal variation.
In a single-agent problem with no opponent, a solution seen once during a rollout is a solution,
and there is no reason to discard it because the averages have not caught up.

## Choosing between them

| | needs a goal | needs a heuristic | backtracks | good at |
|---|---|---|---|---|
| [IW / BFWS](width-based.md) | yes | helps a lot | yes | finding short plans |
| MCTS | yes | helps a lot | yes | long horizons, no model of the goal |
| FSX | no | no | no | staying alive; scoring how stuck a state is |

## Files

| Path | What |
|---|---|
| [`fsx.py`](../../planiverse/planners/fsx.py) | `FSXPlanner`, `option_count` |
| [`mcts.py`](../../planiverse/planners/mcts.py) | `MCTSPlanner` |
| [`tests/test_sampling_planners.py`](../../tests/test_sampling_planners.py) | Tests |
