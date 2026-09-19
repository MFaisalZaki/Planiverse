"""Nested Monte-Carlo Search.

Cazenave, *Nested Monte-Carlo Search*, IJCAI 2009. A level-0 search is a random playout.
A level-`k` search, at each step, runs a level-`(k-1)` search from every child, follows
the child whose search scored best, and keeps the best sequence found at its level: if no
child beats the sequence already in hand, the next move is the one that sequence
prescribes. No tree is stored, no bandit is consulted and nothing is learned between
runs; memorising the best sequence per level is what makes it far stronger than nested
random playouts alone. Its descendant NRPA adapts a playout policy by gradient steps
inside the run and is deliberately not here.

A playout is scored by where it ends, `+inf` for a goal, and a playout that reaches a goal
ends the search with its sequence as the plan. A nested search that ends without one is
started again, since its playouts are random, until the budget or `restarts` runs out.
"""
import random

from planiverse.planners.common import SuccessorCache, finish
from planiverse.planners.sampling.sequences import state_value
from planiverse.planners.width.result import Budget, SearchStatistics


class NestedMonteCarloSearch:
    """NMCS at `level`, playouts capped at `horizon` steps.

    ```python
    from planiverse.planners.sampling import NestedMonteCarloSearch

    NestedMonteCarloSearch(progress=boxes, level=2, horizon=30, seed=0).solve(env, budget)
    ```
    """

    def __init__(self, progress=None, level=2, horizon=30, restarts=100, seed=None):
        if level < 1:
            raise ValueError("level must be at least 1")
        self.progress = progress
        self.level = level
        self.horizon = horizon
        self.restarts = restarts
        self.random = random.Random(seed)

    def __playout__(self, env, cache, state, depth):
        actions, states, node = [], [], state
        for _ in range(self.horizon - depth):
            if cache.exhausted() or env.is_goal(node) or env.is_terminal(node):
                break
            children = cache.expand(node)
            if not children:
                break
            action, node = self.random.choice(children)
            actions.append(action)
            states.append(node)
        self.statistics.rollouts += 1
        return state_value(env, self.progress, node), actions, states

    def __nested__(self, env, cache, state, level, depth):
        """Returns (score, actions, states) of the best sequence found from `state`."""
        if level == 0:
            return self.__playout__(env, cache, state, depth)
        best = (float("-inf"), [], [])
        actions, states, node = [], [], state
        while depth + len(actions) < self.horizon:
            if cache.exhausted() or env.is_goal(node) or env.is_terminal(node):
                break
            children = cache.expand(node)
            if not children:
                break
            for action, successor in children:
                score, tail, tail_states = self.__nested__(env, cache, successor, level - 1,
                                                           depth + len(actions) + 1)
                candidate = (score, actions + [action] + tail, states + [successor] + tail_states)
                if candidate[0] > best[0] or (candidate[0] == best[0] and not best[1]):
                    best = candidate
                if score == float("inf") or cache.exhausted():
                    break
            if best[0] == float("inf") or cache.exhausted():
                break
            if len(best[1]) <= len(actions):
                break
            action = best[1][len(actions)]
            node = best[2][len(states)]
            actions.append(action)
            states.append(node)
        if best[1]:
            return best
        return state_value(env, self.progress, node), actions, states

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        self.statistics = SearchStatistics()
        cache = SuccessorCache(env, self.statistics, budget)
        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return finish("solved", [], [state], self.statistics, budget)
        for _ in range(self.restarts):
            score, actions, states = self.__nested__(env, cache, state, self.level, 0)
            self.statistics.episodes += 1
            if score == float("inf"):
                return finish("solved", actions, [state] + states, self.statistics, budget)
            if cache.exhausted():
                return finish("out_of_budget", None, [], self.statistics, budget)
        return finish("failed", None, [], self.statistics, budget)
