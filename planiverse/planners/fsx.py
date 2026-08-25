"""Future State Maximization: acting to keep your options open.

Every other planner here is told what it wants. FSX is not. It picks the action that leaves
the largest space of reachable futures, and nothing else — no goal, no heuristic, no reward.
That it reaches goals at all is a side effect of the fact that being dead, stuck or cornered
are all states with very few futures.

The paradigm comes from Wissner-Gross and Freer's *causal entropic forces* (Phys. Rev. Lett.
110, 168702, 2013), which defines a force along the gradient of **causal path entropy** — the
entropy over the paths a system could still take within a time horizon τ. Plakolb and
Strelkovskii's *Applicability of the Future State Maximization Paradigm to Agent-Based
Modeling* (Systems 11(2), 105, 2023) is the agent-based reading of it, in which agents
"explore their future state space using **walkers** as virtual entities probing for a
maximization of possible states".

## What this implementation is, and what is inferred

The paper's full text was not reachable from this machine — the network proxy blocks MDPI,
the IIASA repository, arXiv and alexwg.org alike — so this is built from the abstract's
description of walkers plus the causal-entropic-forces formulation underneath it. Two things
follow, and both are stated rather than hidden:

* **The walker scheme is faithful**: sample paths of bounded horizon from each candidate
  action, and prefer the action whose paths reach the most future states. That is the
  abstract's own description.
* **The scoring function is a choice this module makes.** "Maximization of possible states"
  reads as a count of distinct reachable states; the theory underneath it is an entropy over
  the distribution of futures. They differ when some futures are much likelier than others.
  Both are implemented — `measure="count"` and `measure="entropy"` — and neither is claimed
  to be *the* paper's, because its equations could not be read.

## Why it suits a simulator

It asks the environment for exactly one thing: `successors`. No goal decomposition, no
distance-to-goal, no admissible heuristic — the three things a black-box simulator is worst
at providing. When you have no idea how to write a heuristic, this still runs.

And it avoids dead ends structurally rather than by being told to. A state one move from
losing has almost no futures, so it scores badly long before it is reached. The other
planners here need `is_terminal` to be computed and correct; FSX would steer away from those
states even if it were never told they were terminal.
"""
import math
import random
from collections import Counter

from planiverse.planners.width.result import Budget, SearchResult, SearchStatistics

#: How far a walker looks ahead. The causal horizon τ. Short horizons make FSX myopic;
#: long ones make each decision cost `walkers * horizon` successor calls, which against an
#: expensive simulator is the whole budget.
DEFAULT_HORIZON = 8

#: How many walkers probe each candidate action.
DEFAULT_WALKERS = 12


class FSXPlanner:
    """Choose the action that leaves the most futures open, then do it again.

    ```python
    from planiverse.planners.fsx import FSXPlanner

    env.fix_index(0)
    result = FSXPlanner(horizon=8, walkers=12, seed=0).solve(env, Budget(max_seconds=60))
    ```

    This is a **policy**, not a search: it commits to one action at a time and never
    backtracks. It can wander, and on a problem where the goal is a narrow corridor it will
    wander away from it — a corridor is, by construction, a place with few futures. Its
    strength is the opposite case: staying alive and mobile in an environment full of ways
    to get stuck.
    """

    def __init__(self, horizon=DEFAULT_HORIZON, walkers=DEFAULT_WALKERS, measure="count",
                 max_steps=200, seed=None, temperature=0.0):
        """
        - `measure` — `"count"` scores an action by how many distinct states its walkers
          reached; `"entropy"` by the Shannon entropy of where they ended up. Count rewards
          breadth; entropy also rewards *evenness*, penalising an action whose futures nearly
          all collapse to the same place.
        - `temperature` — the causal path temperature. `0.0` takes the best action; above
          zero, actions are sampled in proportion to `exp(score / temperature)`, which is
          closer to the physical formulation and useful when scores tie often.
        - `seed` — walkers are random, so this is what makes a run reproducible.
        """
        if measure not in ("count", "entropy"):
            raise ValueError(f"measure must be 'count' or 'entropy', got {measure!r}")
        self.horizon = horizon
        self.walkers = walkers
        self.measure = measure
        self.max_steps = max_steps
        self.temperature = temperature
        self.random = random.Random(seed)

    # ------------------------------------------------------------------ the walkers

    def __walk__(self, env, state, statistics, budget):
        """One walker: a random path of up to `horizon` steps, reporting where it went.

        Returns the states visited. A walk that hits a dead end stops there, which is how a
        dead end costs an action its score without anything having to know it is a dead end.
        """
        visited = []
        node = state
        for _ in range(self.horizon):
            if budget.exhausted(statistics.expansions):
                break
            if env.is_terminal(node) or env.is_goal(node):
                break
            successors = env.successors(node)
            statistics.expansions += 1
            statistics.generated += len(successors)
            if not successors:
                break
            _, node = self.random.choice(successors)
            visited.append(node)
        return visited

    def __score__(self, env, state, statistics, budget):
        """How much future this state leaves open."""
        endpoints = Counter()
        for _ in range(self.walkers):
            if budget.exhausted(statistics.expansions):
                break
            visited = self.__walk__(env, state, statistics, budget)
            # The walk's whole trajectory counts, not just where it stopped: a path that
            # passed through many distinct states explored more future than one that
            # shuffled between two, even if both end in the same place.
            for reached in visited:
                endpoints[reached.literals] += 1
            if not visited:
                # Nowhere to go at all. Distinguished from "went somewhere and came back",
                # which does have a future.
                endpoints[("__stuck__", id(state))] += 1

        if not endpoints:
            return 0.0
        if self.measure == "count":
            return float(len(endpoints))
        total = sum(endpoints.values())
        return -sum((n / total) * math.log(n / total) for n in endpoints.values())

    def __choose__(self, scored):
        """Best action, or a Boltzmann sample of them when a temperature is set."""
        if self.temperature <= 0:
            best = max(score for _, _, score in scored)
            return self.random.choice([item for item in scored if item[2] == best])
        weights = [math.exp(score / self.temperature) for _, _, score in scored]
        return self.random.choices(scored, weights=weights, k=1)[0]

    # ------------------------------------------------------------------ the policy

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics()

        if state is None:
            state, _ = env.reset()
        plan, trace = [], [state]

        for _ in range(self.max_steps):
            if env.is_goal(state):
                return self.__result__("solved", plan, trace, statistics, budget)
            if env.is_terminal(state):
                return self.__result__("dead_end", None, trace, statistics, budget)
            if budget.exhausted(statistics.expansions):
                return self.__result__("out_of_budget", None, trace, statistics, budget)

            successors = env.successors(state)
            statistics.expansions += 1
            statistics.generated += len(successors)
            if not successors:
                return self.__result__("dead_end", None, trace, statistics, budget)

            scored = []
            for action, successor in successors:
                # A goal is worth taking whatever its futures look like — and a goal state is
                # usually absorbing here, so it would otherwise score zero and be the last
                # thing FSX picks.
                if env.is_goal(successor):
                    scored = [(action, successor, float("inf"))]
                    break
                scored.append((action, successor,
                               self.__score__(env, successor, statistics, budget)))

            action, state, _ = self.__choose__(scored)
            plan.append(action)
            trace.append(state)

        status = "solved" if env.is_goal(state) else "step_limit"
        return self.__result__(status, plan if status == "solved" else None, trace,
                               statistics, budget)

    def __result__(self, status, plan, trace, statistics, budget):
        statistics.elapsed = budget.elapsed()
        return SearchResult(plan=plan, states=trace, status=status,
                            width=None, statistics=statistics)


def option_count(env, state, horizon=DEFAULT_HORIZON, walkers=DEFAULT_WALKERS, seed=0,
                 budget=None):
    """How many distinct futures a state leaves open — FSX's measure, on its own.

    Useful without the planner. It is a **goal-free difficulty signal**: a state with few
    futures is one move from being stuck, whatever the environment thinks a goal is. That
    makes it a heuristic for the other planners in exactly the case where heuristics are
    hardest to write, and a way to spot dead ends in environments whose `is_terminal` is
    conservative.
    """
    planner = FSXPlanner(horizon=horizon, walkers=walkers, seed=seed)
    return planner.__score__(env, state, SearchStatistics(), (budget or Budget()).start())
