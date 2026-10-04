"""Blind search: the baselines every other number should be read against.

* `BreadthFirstSearch`: with duplicate detection on `literals`. The IJCAI 2015 Atari
  paper's baseline, and complete.
* `UniformCostSearch`: Dijkstra on `action.cost()` when actions report one, else on plan
  length, where it is breadth-first with a heap.
* `IterativeDeepening`: depth-first to depth 1, 2, 3, …, with duplicates checked on the
  current path only. Its memory is linear in the depth, and it re-expands the shallow tree
  at every iteration, which against a simulator is the whole cost; `SuccessorCache` turns
  those re-expansions into lookups, at which point it is breadth-first search with a
  different memory profile. Kept for the comparison.
"""
from collections import deque
from heapq import heappop, heappush
from itertools import count

from planiverse.planners.common import SuccessorCache, finish
from planiverse.planners.width.result import Budget, SearchStatistics


def _cost(action):
    cost = getattr(action, "cost", None)
    return float(cost()) if callable(cost) else 1.0


class BreadthFirstSearch:
    """Breadth-first search with duplicate detection."""

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics()
        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return finish("solved", [], [state], statistics, budget)
        frontier = deque([(state, [], [state])])
        closed = {state.literals}
        while frontier:
            if budget.exhausted(statistics.expansions):
                return finish("out_of_budget", None, [], statistics, budget)
            node, plan, trace = frontier.popleft()
            statistics.expansions += 1
            for action, successor in env.successors(node):
                statistics.generated += 1
                if successor.literals in closed:
                    statistics.pruned_duplicate += 1
                    continue
                closed.add(successor.literals)
                successor_plan, successor_trace = plan + [action], trace + [successor]
                if env.is_goal(successor):
                    return finish("solved", successor_plan, successor_trace, statistics,
                                  budget)
                if env.is_terminal(successor):
                    statistics.pruned_terminal += 1
                    continue
                frontier.append((successor, successor_plan, successor_trace))
        return finish("exhausted", None, [], statistics, budget)


class UniformCostSearch:
    """Dijkstra's algorithm on action costs."""

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics()
        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return finish("solved", [], [state], statistics, budget)
        tiebreak = count()
        heap = [(0.0, next(tiebreak), state, [], [state])]
        best = {state.literals: 0.0}
        closed = set()
        while heap:
            if budget.exhausted(statistics.expansions):
                return finish("out_of_budget", None, [], statistics, budget)
            g, _, node, plan, trace = heappop(heap)
            if node.literals in closed:
                statistics.pruned_duplicate += 1
                continue
            if env.is_goal(node):
                return finish("solved", plan, trace, statistics, budget)
            closed.add(node.literals)
            statistics.expansions += 1
            for action, successor in env.successors(node):
                statistics.generated += 1
                g2 = g + _cost(action)
                if best.get(successor.literals, float("inf")) <= g2:
                    statistics.pruned_duplicate += 1
                    continue
                if env.is_terminal(successor) and not env.is_goal(successor):
                    statistics.pruned_terminal += 1
                    continue
                best[successor.literals] = g2
                heappush(heap, (g2, next(tiebreak), successor, plan + [action],
                                trace + [successor]))
        return finish("exhausted", None, [], statistics, budget)


class IterativeDeepening:
    """Depth-first iterative deepening, duplicates checked on the current path."""

    def __init__(self, max_depth=100):
        self.max_depth = max_depth

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics()
        cache = SuccessorCache(env, statistics, budget)
        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return finish("solved", [], [state], statistics, budget)
        for limit in range(1, self.max_depth + 1):
            statistics.widths_tried = tuple(statistics.widths_tried) + (limit,)
            self.cut = False
            found = self.__dfs__(env, cache, state, [], [state], {state.literals}, limit,
                                 statistics)
            if found is not None:
                return finish("solved", found[0], found[1], statistics, budget)
            if cache.exhausted():
                return finish("out_of_budget", None, [], statistics, budget)
            if not self.cut:
                return finish("exhausted", None, [], statistics, budget)
        return finish("failed", None, [], statistics, budget)

    def __dfs__(self, env, cache, node, plan, trace, path, limit, statistics):
        if limit == 0:
            self.cut = True
            return None
        if cache.exhausted():
            return None
        for action, successor in cache.expand(node):
            if successor.literals in path:
                continue
            if env.is_goal(successor):
                return plan + [action], trace + [successor]
            if env.is_terminal(successor):
                statistics.pruned_terminal += 1
                continue
            path.add(successor.literals)
            found = self.__dfs__(env, cache, successor, plan + [action], trace + [successor],
                                 path, limit - 1, statistics)
            path.discard(successor.literals)
            if found is not None:
                return found
        return None
