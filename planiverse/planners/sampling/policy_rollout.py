"""The rollout algorithm: one-step lookahead valued by a base policy.

Bertsekas, Tsitsiklis and Wu, *Rollout Algorithms for Combinatorial Optimization*, Journal
of Heuristics 1997; Bertsekas, *Rollout, Policy Iteration, and Distributed Reinforcement
Learning*, 2020. Take any policy, however crude, and improve it by one step of policy
iteration done on the fly: at each state, value every child by running the base policy
from it to the horizon, and take the best child. The rollout policy is provably no worse
than the base policy (sequential improvement), and in practice much better.

The base policy here defaults to greedy on `progress` with random tie-breaking, which
against a simulator is the crudest useful policy there is; pass `policy(state, children)
-> (action, successor)` for another. With a deterministic base policy there is no sampling
at all. `fortified=True` keeps the best complete trajectory found so far and follows it
whenever the lookahead cannot beat it, which is the fortified rollout of the 1997 paper and
what keeps the guarantee under a noisy heuristic.
"""
import random

from planiverse.planners.common import SuccessorCache, finish
from planiverse.planners.sampling.sequences import state_value
from planiverse.planners.width.result import Budget, SearchStatistics


class RolloutPlanner:
    """Bertsekas's rollout on a base policy.

    ```python
    from planiverse.planners.sampling import RolloutPlanner

    RolloutPlanner(progress=boxes, horizon=15, fortified=True, seed=0).solve(env, budget)
    ```
    """

    def __init__(self, progress=None, horizon=15, policy=None, fortified=True,
                 max_steps=200, seed=None):
        self.progress = progress
        self.horizon = horizon
        self.policy = policy
        self.fortified = fortified
        self.max_steps = max_steps
        self.random = random.Random(seed)

    def __base__(self, env, state, children):
        if self.policy is not None:
            return self.policy(state, children)
        live = [(a, s) for a, s in children if not env.is_terminal(s) or env.is_goal(s)]
        pool = live or children
        best = max(state_value(env, self.progress, s) for _, s in pool)
        return self.random.choice([(a, s) for a, s in pool
                                   if state_value(env, self.progress, s) == best])

    def __rollout__(self, env, cache, state):
        """Follow the base policy for `horizon` steps. Returns (value, actions, states)."""
        actions, states, node = [], [], state
        for _ in range(self.horizon):
            if cache.exhausted() or env.is_goal(node) or env.is_terminal(node):
                break
            children = cache.expand(node)
            if not children:
                break
            action, node = self.__base__(env, node, children)
            actions.append(action)
            states.append(node)
        return state_value(env, self.progress, node), actions, states

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics()
        cache = SuccessorCache(env, statistics, budget)
        if state is None:
            state, _ = env.reset()
        plan, trace = [], [state]
        fort = None                 # (value, actions, states) of the best trajectory ahead
        for _ in range(self.max_steps):
            if env.is_goal(state):
                return finish("solved", plan, trace, statistics, budget)
            if env.is_terminal(state):
                return finish("dead_end", None, trace, statistics, budget)
            if cache.exhausted():
                return finish("out_of_budget", None, [], statistics, budget)
            children = cache.expand(state)
            if not children:
                return finish("dead_end", None, trace, statistics, budget)
            best = None
            for action, successor in children:
                value, actions, states = self.__rollout__(env, cache, successor)
                statistics.rollouts += 1
                candidate = (value, [action] + actions, [successor] + states)
                if value == float("inf"):
                    return finish("solved", plan + candidate[1], trace + candidate[2],
                                  statistics, budget)
                if best is None or value > best[0]:
                    best = candidate
            if self.fortified and fort is not None and fort[1] and fort[0] >= best[0]:
                best = fort
            action, successor = best[1][0], best[2][0]
            fort = (best[0], best[1][1:], best[2][1:]) if self.fortified else None
            plan.append(action)
            trace.append(successor)
            state = successor
        return finish("solved" if env.is_goal(state) else "step_limit",
                      plan if env.is_goal(state) else None, trace, statistics, budget)
