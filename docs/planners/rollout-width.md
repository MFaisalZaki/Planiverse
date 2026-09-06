# Rollout IW and π-IW

Two more width-based planners, both online, both built from rollouts rather than a
breadth-first frontier, and one of them learning as it goes.

| Planner | Where | Reference |
|---|---|---|
| `RolloutIW` | [`planiverse/planners/width/rollout.py`](../../planiverse/planners/width/rollout.py) | Bandres, Bonet & Geffner, *Planning with Pixels in (Almost) Real Time*, AAAI 2018 |
| `PiIW` | [`planiverse/planners/width/policy.py`](../../planiverse/planners/width/policy.py) | Junyent, Jonsson & Gómez, *Deep Policies for Width-Based Planning in Pixel Domains*, ICAPS 2019 |

```python
from planiverse.planners.width import RolloutIW, PiIW, Budget

env.set_index(0)
result = RolloutIW(width=1, expansions_per_step=1000, progress=boxes, seed=0).solve(
    env, Budget(max_expansions=50_000, max_seconds=60))
result = PiIW(expansions_per_step=100, progress=boxes, seed=0).solve(
    env, Budget(max_expansions=50_000, max_seconds=60))
```

## Rollout IW

IW(k) is breadth-first search with a novelty filter, and breadth-first is the trouble when
expansions are expensive and decisions are wanted now: the frontier has to be finished level by
level before anything deep is seen. Rollout IW keeps the filter and drops the order. It grows a
tree by **rollouts** from the root. Each one walks down through the tree, expands a node when it
steps off the known part, and stops as soon as it reaches a node that is not novel, is terminal,
or has nothing left beneath it. A node is *solved* when there is nothing left to learn under it,
and rollouts never enter solved nodes, so when the root is solved the lookahead is complete.

Two definitions differ from IW's:

- **Novelty is measured against depth.** `DepthNoveltyTable` remembers the shallowest depth each
  atom tuple has been seen at, not merely whether it has been seen. A node at depth *d* is novel
  when one of its tuples has never been seen that shallow. A node already in the tree is tested
  with `>=` rather than `>`, because it may be the node that put that depth in the table, and it
  must not be pruned for its own discovery. Rollouts do not come in depth order, so "seen at all"
  would let the first rollout, at whatever depth it happened to reach an atom, pre-empt every
  shallower discovery after it.
- **The lookahead is per decision.** After `expansions_per_step` expansions the action whose
  subtree backed up the best discounted return is committed to, the subtree beneath it is kept,
  and the novelty table is started afresh. Resetting the table at every step is what renews
  exploration, the way SIW's legs do, and it is why the online form solves what a single
  width-1 lookahead cannot. On Puzznic level 1, where IW(1) exhausts, one Rollout IW(1)
  lookahead solves its root after 45 expansions without a plan, and the online planner finds
  one.

`expansions_per_step=None` runs one lookahead from the initial state and never commits: an
IW-shaped search that sees depth from the first rollout, and fails when the root is solved.

### Against a simulator

- **There is no score.** Atari's rewards drive the action choice. Here the reward of a transition
  is the drop in the same `progress(state)` measure SIW and BFWS take, and a goal is returned as
  the plan the moment any rollout finds one. Without `progress` the returns are flat, the
  committed action is a uniform draw, and the result's status says so.
- **A dead end is worth minus infinity.** `is_terminal` means no goal is reachable from there, so
  whatever progress the step into it made is progress into a wall. `avoid_dead_ends=False`
  restores Atari's reading, where losing merely ends the scoring; on Puzznic level 1 that reading
  steers the planner into the trap described in the [SIW notes](width-based.md#siw-and-dead-ends).
- **An expansion yields every child.** The contract is `successors()`, so stepping off the known
  tree generates all of a node's children at once, which is what an expansion costs anyway. The
  rollout then picks one; a sibling reached by a later rollout has its novelty assessed as a new
  node, since that is the first time anyone looks.
- **Episodes.** An episode ends at a goal, a dead end or `max_steps` committed actions. Plain
  Rollout IW learns nothing between episodes, so `max_episodes=1` by default and a second
  episode would be a fresh draw. `statistics.rollouts` and `statistics.episodes` count both.

## π-IW

Rollout IW picks the child a rollout follows uniformly at random. π-IW keeps everything else
and replaces that one choice with a sample from a policy π(a|s), and it trains the policy on the
planner itself: after every committed action, the returns the lookahead backed up to the root's
children become a target distribution, softmax(R/τ), and the network is pushed toward it by
cross-entropy. Nothing about the domain is told to it. The policy is a compressed memory of
where the lookaheads found return before, and it steers new rollouts there first, which is
what lets a small per-decision budget go a long way.

The paper's other contribution is that the network's last hidden layer, binarised, works as the
atoms novelty is measured over, so pixel domains need no hand-made features. `features="learned"`
does that here, `"both"` unions it with the environment's literals, and `"literals"` (the default)
uses the atoms every other planner in this library uses.

### What is kept and what is changed

- **The function approximator is a one-hidden-layer network in NumPy.** States here are sets
  of atoms, so the input is a hashed bag of them: each atom is mapped by a stable hash to one of
  `inputs` binary inputs (2048 by default), then 64 hidden units, then a softmax over the actions.
  That keeps the library free of a deep-learning dependency and is enough for the mechanism the
  paper describes; it is not a claim to match its Atari numbers.
- **Actions are indexed as they appear.** Environments build actions per state and have no fixed
  vocabulary, so the output layer grows a zero-weight column when a new action is seen, which
  starts it at the same probability as its peers. A `floor` keeps every candidate's rollout weight
  above zero, so a policy sure of itself cannot stop exploration.
- **Returns come from `progress`.** They are scaled to `[0, 1]` among the root's children before
  the softmax, so one temperature serves every environment; the paper uses the game's raw
  returns.
- **Training is online and continues across episodes.** One Adam step per committed action on a
  batch of 32 from a replay of the last 10,000 targets. An episode that ends without a goal is
  not wasted, the next starts with what it taught, so `max_episodes=None`: π-IW runs until it
  finds a plan or the budget ends. `planner.network` is the trained policy and `planner.losses`
  the training curve.

On the four pure-Python Puzznic levels above, with 200 expansions per step, π-IW with learned
features solved three of the four in fewer expansions than Rollout IW with the environment's
literals, which is the paper's claim in miniature and nothing more: one seed, four levels.

## Files

| Path | What |
|---|---|
| [`novelty.py`](../../planiverse/planners/width/novelty.py) | `DepthNoveltyTable` |
| [`rollout.py`](../../planiverse/planners/width/rollout.py) | `RolloutIW`, `RolloutNode` |
| [`policy.py`](../../planiverse/planners/width/policy.py) | `PiIW`, `PolicyNetwork` |
| [`tests/test_rollout_planners.py`](../../tests/test_rollout_planners.py) | Tests |
