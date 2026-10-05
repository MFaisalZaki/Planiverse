"""Best-first search on a black-box heuristic, greedy, weighted or restarting.

The heuristic is `progress(state)`, lower is better, the same callback the width planners
take. It is a search guide and nothing more: not admissible, often flat, and sometimes
wrong, and the satisficing-planning literature is largely about what to do then.

* `BestFirstSearch`: greedy best-first on `h`, or weighted A* on `g + weight * h` when a
  `weight` is given (Pohl 1970). Complete either way; A*'s optimality needs `weight=1`
  and an admissible `h`, which `progress` is not.
* `RestartingWeightedAStar` (Richter, Thayer & Ruml, *The Joy of Forgetting: Faster Anytime
  Search via Restarting*, ICAPS 2010): anytime. Solve with a large weight, keep the plan,
  restart with a smaller one, and keep the best plan found when the budget ends. Expansions
  are cached, so a restart re-reads what the previous search generated instead of asking
  the simulator again.

Both are deterministic: the open list breaks ties by insertion order.
"""
from heapq import heappop, heappush
from itertools import count

from planiverse.planners.common import SuccessorCache, finish
from planiverse.planners.width.result import Budget, SearchStatistics


def action_cost(action):
    cost = getattr(action, "cost", None)
    return float(cost()) if callable(cost) else 1.0


class _Entry:
    __slots__ = ("state", "plan", "trace", "g", "h", "removed")

    def __init__(self, state, plan, trace, g, h):
        self.state, self.plan, self.trace, self.g, self.h = state, plan, trace, g, h
        self.removed = False


class BestFirstSearch:
    """Greedy best-first on `progress`, or weighted A* with a `weight`.

    ```python
    from planiverse.planners.heuristic import BestFirstSearch

    BestFirstSearch(progress=boxes).solve(env, budget)               # greedy
    BestFirstSearch(progress=boxes, weight=2.0).solve(env, budget)   # WA*, f = g + 2h
    ```
    """

    def __init__(self, progress=None, weight=None):
        self.progress = progress
        self.weight = weight

    # ------------------------------------------------------------- the open list
    # A heap of entries with a `removed` flag, so a subclass that keeps a second view of the
    # open list can ignore what the other view already took.

    def __h__(self, state):
        return self.progress(state) if self.progress else 0.0

    def __key__(self, entry):
        if self.weight is None:
            return (entry.h, entry.g)
        return (entry.g + self.weight * entry.h, entry.h)

    def __reset__(self):
        self.heap, self.tiebreak = [], count()

    def __push__(self, entry):
        heappush(self.heap, (self.__key__(entry), next(self.tiebreak), entry))

    def __pop_best__(self):
        while self.heap:
            _, _, entry = heappop(self.heap)
            if not entry.removed:
                entry.removed = True
                return entry
        return None

    def __select__(self, statistics):
        """Which open entry to expand next."""
        return self.__pop_best__()

    def __open__(self):
        return bool(self.heap)

    # ------------------------------------------------------------- the search

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics()
        self.cache = SuccessorCache(env, statistics, budget)
        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return finish("solved", [], [state], statistics, budget)
        self.__reset__()
        self.best_g = {state.literals: 0.0}
        self.__push__(_Entry(state, [], [state], 0.0, self.__h__(state)))
        self.__started__(state, statistics)
        while self.__open__():
            if budget.exhausted(statistics.expansions):
                return self.__finished__("out_of_budget", statistics, budget)
            entry = self.__select__(statistics)
            if entry is None:
                break
            if self.best_g.get(entry.state.literals, float("inf")) < entry.g:
                statistics.pruned_duplicate += 1
                continue
            found = self.__expand__(env, entry, statistics)
            if found is not None:
                return finish("solved", found.plan, found.trace, statistics, budget)
            self.__expanded__(entry, statistics)
        return self.__finished__("exhausted", statistics, budget)

    def __expand__(self, env, entry, statistics, push=None):
        """Generate `entry`'s children into the open list; return a goal child if any."""
        push = push or self.__push__
        for action, successor in self.cache.expand(entry.state):
            g = entry.g + action_cost(action)
            if self.best_g.get(successor.literals, float("inf")) <= g:
                statistics.pruned_duplicate += 1
                continue
            child = _Entry(successor, entry.plan + [action], entry.trace + [successor], g,
                           self.__h__(successor))
            if env.is_goal(successor):
                return child
            if env.is_terminal(successor):
                statistics.pruned_terminal += 1
                continue
            self.best_g[successor.literals] = g
            push(child)
        return None

    # ------------------------------------------------------------- hooks

    def __started__(self, state, statistics):
        pass

    def __expanded__(self, entry, statistics):
        pass

    def __finished__(self, status, statistics, budget):
        return finish(status, None, [], statistics, budget)


class RestartingWeightedAStar(BestFirstSearch):
    """Anytime weighted A* that restarts with a smaller weight after each solution.

    ```python
    RestartingWeightedAStar(progress=boxes, weights=(5, 3, 2, 1.5, 1)).solve(env, budget)
    ```

    Returns the best plan found; `statistics.widths_tried` records the weights that ran,
    and `planner.incumbents` every plan length found in order.
    """

    def __init__(self, progress=None, weights=(5.0, 3.0, 2.0, 1.5, 1.0)):
        super().__init__(progress, weights[0])
        self.weights = tuple(weights)
        self.incumbents = []

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics()
        cache = SuccessorCache(env, statistics, budget)
        best = None
        for weight in self.weights:
            search = BestFirstSearch(self.progress, weight)
            search.cache = cache
            result = self.__run__(search, env, budget, state, statistics)
            statistics.widths_tried = tuple(statistics.widths_tried) + (weight,)
            if result.solved:
                self.incumbents.append(len(result.plan))
                if best is None or result.cost < best.cost:
                    best = result
            if result.status == "out_of_budget":
                break
        if best is not None:
            return finish("solved", best.plan, best.states, statistics, budget)
        return finish("out_of_budget" if budget.exhausted(statistics.expansions) else
                      "exhausted", None, [], statistics, budget)

    @staticmethod
    def __run__(search, env, budget, state, statistics):
        """One weighted A* sharing the cache, so a restart re-reads rather than re-asks."""
        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return finish("solved", [], [state], statistics, budget)
        search.__reset__()
        search.best_g = {state.literals: 0.0}
        search.__push__(_Entry(state, [], [state], 0.0, search.__h__(state)))
        while search.__open__():
            if budget.exhausted(statistics.expansions):
                return finish("out_of_budget", None, [], statistics, budget)
            entry = search.__pop_best__()
            if entry is None:
                break
            if search.best_g.get(entry.state.literals, float("inf")) < entry.g:
                continue
            found = search.__expand__(env, entry, statistics)
            if found is not None:
                return finish("solved", found.plan, found.trace, statistics, budget)
        return finish("exhausted", None, [], statistics, budget)
