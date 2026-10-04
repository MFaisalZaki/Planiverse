"""Real-time heuristic search: a bounded lookahead, one commitment, repeat.

* **LRTA\\*** (Korf, *Real-Time Heuristic Search*, AIJ 1990): from the current state, look
  ahead a bounded number of expansions, move one step toward the frontier state with the
  best `g + H`, and raise the current state's stored value `H` to that best `f`. The table
  `H` starts as `progress` and only ever rises, which is what keeps the agent from cycling
  in a heuristic depression: every visit makes the depression shallower.
* **RTAA\\*** (Koenig and Likhachev, *Real-Time Adaptive A\\**, AAMAS 2006): the same
  lookahead as an A\\* with an expansion limit, after which **every expanded state** gets
  `H(s) = f(best open) - g(s)`, a cheaper and larger update than LRTA\\*'s, and the agent
  moves along the path to the best open state.

Both are online: the value updates happen in the run they serve, with no phase before it,
and both commit to actions as they go rather than returning a plan at the end. The plan returned
is the path actually walked, so it can revisit a state; a state restore here is free, and
`max_steps` bounds the walk.

Time-bounded A\\* (Björnsson, Bulitko and Sturtevant, IJCAI 2009) is not here: it moves
the agent backwards along its search tree when the target changes, which assumes actions
can be undone, and a simulator's cannot.
"""
from heapq import heappop, heappush
from itertools import count

from planiverse.planners.common import SuccessorCache, finish
from planiverse.planners.heuristic.bestfirst import action_cost
from planiverse.planners.width.result import Budget, SearchStatistics


class LRTAStar:
    """Korf's LRTA* with a bounded lookahead.

    ```python
    from planiverse.planners.heuristic import LRTAStar

    LRTAStar(progress=boxes, lookahead=20).solve(env, budget)
    ```
    """

    def __init__(self, progress, lookahead=20, max_steps=500):
        if progress is None:
            raise ValueError("real-time search needs a progress measure as its initial H")
        self.progress = progress
        self.lookahead = lookahead
        self.max_steps = max_steps
        self.values = {}

    def __H__(self, state):
        key = state.literals
        if key not in self.values:
            self.values[key] = float(self.progress(state))
        return self.values[key]

    def __lookahead__(self, env, cache, state, statistics):
        """A* from `state` for `lookahead` expansions.

        Returns `(goal_entry, best_open, closed)`, where entries are
        `(f, g, state, plan, trace)`.
        """
        tiebreak = count()
        heap = [(self.__H__(state), next(tiebreak), 0.0, state, [], [])]   # trace: after the root
        best_g = {state.literals: 0.0}
        closed = {}
        expanded = 0
        while heap and expanded < self.lookahead and not cache.exhausted():
            f, _, g, node, plan, trace = heappop(heap)
            if best_g.get(node.literals, float("inf")) < g or node.literals in closed:
                continue
            closed[node.literals] = g
            expanded += 1
            for action, successor in cache.expand(node):
                g2 = g + action_cost(action)
                if best_g.get(successor.literals, float("inf")) <= g2:
                    continue
                entry = (g2 + self.__H__(successor), next(tiebreak), g2, successor,
                         plan + [action], trace + [successor])
                if env.is_goal(successor):
                    return entry, None, closed
                if env.is_terminal(successor):
                    statistics.pruned_terminal += 1
                    self.values[successor.literals] = float("inf")
                    continue
                best_g[successor.literals] = g2
                heappush(heap, entry)
        best = None
        while heap:
            f, _, g, node, plan, trace = heappop(heap)
            if best_g.get(node.literals, float("inf")) < g or node.literals in closed:
                continue
            best = (f, None, g, node, plan, trace)
            break
        return None, best, closed

    def __update__(self, state, best, closed):
        """LRTA*: raise the current state's value to the best frontier f."""
        self.values[state.literals] = max(self.__H__(state), best[0])

    def __move__(self, best):
        """LRTA* takes one step toward the target."""
        return best[4][:1], best[5][:1]

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics()
        cache = SuccessorCache(env, statistics, budget)
        if state is None:
            state, _ = env.reset()
        plan, trace = [], [state]
        for _ in range(self.max_steps):
            if env.is_goal(state):
                return finish("solved", plan, trace, statistics, budget)
            if cache.exhausted():
                return finish("out_of_budget", None, [], statistics, budget)
            goal, best, closed = self.__lookahead__(env, cache, state, statistics)
            if goal is not None:
                return finish("solved", plan + goal[4], trace + goal[5], statistics, budget)
            if best is None:
                return finish("failed", None, [], statistics, budget)
            self.__update__(state, best, closed)
            actions, states = self.__move__(best)
            plan += actions
            trace += states
            state = trace[-1]
        return finish("failed" if not env.is_goal(state) else "solved",
                      plan if env.is_goal(state) else None, trace, statistics, budget)


class RTAAStar(LRTAStar):
    """Koenig and Likhachev's RTAA*: update every expanded state, walk to the best."""

    def __init__(self, progress, lookahead=20, max_steps=500, move=None):
        super().__init__(progress, lookahead, max_steps)
        self.move = move            # steps to take along the path; None walks it all

    def __update__(self, state, best, closed):
        for key, g in closed.items():
            self.values[key] = max(self.values.get(key, 0.0), best[0] - g)

    def __move__(self, best):
        if self.move is None:
            return best[4], best[5]
        return best[4][:self.move], best[5][:self.move]
