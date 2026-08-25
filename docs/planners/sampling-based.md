# Sampling-based planners: FSX and MCTS

Two planners that sample futures instead of enumerating them. They sit at opposite ends of
how much they need to be told: MCTS wants a reward, FSX wants nothing at all.

## Future State Maximization (FSX)

Every other planner here is told what it wants. FSX is not. It picks the action leaving the
largest space of reachable futures, and nothing else — **no goal, no heuristic, no reward**.
That it reaches goals at all is a side effect of the fact that being dead, stuck or cornered
are all states with very few futures.

- **Class:** `FSXPlanner` — [`planiverse/planners/fsx.py`](../../planiverse/planners/fsx.py)
- **Paradigm:** Wissner-Gross & Freer, *Causal Entropic Forces*, Phys. Rev. Lett. 110,
  168702 (2013); Plakolb & Strelkovskii, *Applicability of the Future State Maximization
  Paradigm to Agent-Based Modeling*, Systems 11(2), 105 (2023).

```python
from planiverse.planners.fsx import FSXPlanner, option_count

result = FSXPlanner(horizon=8, walkers=12, seed=0).solve(env, Budget(max_seconds=60))
```

### What is implemented, and what is inferred

**The paper's full text was not reachable from the machine this was written on** — the
network proxy blocks MDPI, the IIASA repository, arXiv and alexwg.org alike. So this is built
from the abstract's description of the mechanism plus the causal-entropic-forces formulation
underneath it, and the split is stated rather than hidden:

- **The walker scheme is faithful to the abstract**, which describes agents that "explore
  their future state space using **walkers** as virtual entities probing for a maximization
  of possible states". Sample bounded-horizon paths from each candidate action; prefer the
  action whose walkers reach the most future states.
- **The scoring function is this module's choice.** "Maximization of possible states" reads
  as a *count* of distinct reachable states; the theory underneath it is an *entropy* over
  the distribution of futures. They differ when some futures are much likelier than others.
  Both are implemented — `measure="count"` and `measure="entropy"` — and neither is claimed
  to be the paper's, because its equations could not be read.

Two parameters come straight from the physics: the causal horizon τ (`horizon`, how far a
walker looks) and the causal path temperature (`temperature`, which turns argmax selection
into a Boltzmann sample).

### Why it suits a simulator

It asks the environment for exactly one thing: `successors`. No goal decomposition, no
distance-to-goal, no admissible heuristic — the three things a black-box simulator is worst
at providing. **When you have no idea how to write a heuristic, this still runs.**

And it avoids dead ends *structurally* rather than by being told to. A state one move from
losing has almost no futures, so it scores badly long before it is reached. Every other
planner here needs `is_terminal` to be computed and correct; FSX would steer away from those
states even if it were never told they were terminal.

### What it is not

**It is a policy, not a search.** It commits to one action at a time and never backtracks. On
Puzznic level 1 it does solve — but in **128 actions** where IW(2) takes 10. That is not a
bug to be tuned away; it is what a goal-free agent looks like. Its strength is the opposite
case: staying alive and mobile where the danger is getting stuck, not reaching a specific
narrow target. A goal down a corridor is, by construction, in a place with few futures.

`option_count(env, state)` exposes the measure on its own. That is arguably the more useful
export: a **goal-free difficulty signal** that says how close a state is to being stuck,
which makes it a heuristic for the other planners in exactly the case where heuristics are
hardest to write.

## Monte Carlo Tree Search (UCT)

- **Class:** `MCTSPlanner` — [`planiverse/planners/mcts.py`](../../planiverse/planners/mcts.py)
- **References:** Kocsis & Szepesvári, *Bandit Based Monte-Carlo Planning* (ECML 2006) for
  UCT and UCB1; Browne et al., *A Survey of Monte Carlo Tree Search Methods* (IEEE TCIAIG
  4(1), 2012) for the select / expand / simulate / backpropagate loop.

```python
from planiverse.planners.mcts import MCTSPlanner

result = MCTSPlanner(iterations=3000, seed=0,
                     reward=lambda s: 1 - blocks_left(s) / 6).solve(env)
```

MCTS was built for adversarial games with a natural terminal score. Planiverse gives it
neither, and three adaptations follow — each of which is measurable on Puzznic level 1:

**There is no score, only `is_goal`.** The default reward is blunt: 1 for a goal, 0
otherwise, less a small length penalty. A `reward` callback supplies something denser. It is
worth supplying:

| Reward | Result | Plan |
|---|---|---|
| sparse (goal or nothing) | solved | 32 actions |
| dense (blocks cleared) | solved | **16 actions** |

**The transition is deterministic**, so the averaging that makes UCT work in stochastic games
is doing nothing. This keeps the *best* value seen through a node as well as the mean:

| Backup | Result |
|---|---|
| `"max"` (default) | solved, 16 actions |
| `"mean"` (classical) | **out of budget** |

**Dead ends are real and absorbing.** Terminal states are marked in the tree and never
selected again, which stops UCT re-exploring a branch it has already proved is over.

One further adaptation, easy to miss and worth more than the rest put together: **the rollout
keeps the best reward it saw anywhere along the way**, not the reward where it stopped. Random
rollouts in a domain with dead ends nearly always end in one, so scoring only the final state
throws away everything the rollout learned, the tree sees 0 for every branch, and UCT has no
gradient to climb. Without this, MCTS does not solve Puzznic level 1 at all.

The plan returned is the best goal-reaching path **found**, not the tree's principal
variation. In a single-agent problem with no opponent, a solution seen once during a rollout
is a solution, and there is no reason to discard it because the averages have not caught up.

## Choosing between them

| | needs a goal | needs a heuristic | backtracks | good at |
|---|---|---|---|---|
| [IW / BFWS](width-based.md) | yes | helps a lot | yes | finding short plans |
| MCTS | yes | helps a lot | yes | long horizons, no model of the goal |
| FSX | **no** | **no** | no | staying alive; scoring how stuck a state is |

## Files

| Path | What |
|---|---|
| [`fsx.py`](../../planiverse/planners/fsx.py) | `FSXPlanner`, `option_count` |
| [`mcts.py`](../../planiverse/planners/mcts.py) | `MCTSPlanner` |
| [`tests/test_sampling_planners.py`](../../tests/test_sampling_planners.py) | Tests, including the reward and backup comparisons above |
