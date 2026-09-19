"""Multi-heuristic alternation: one open list per measure, expanded round-robin.

Helmert, *The Fast Downward Planning System*, JAIR 2006, and Röger and Helmert, *The More,
the Merrier: Combining Heuristic Estimators for Satisficing Planning*, ICAPS 2010, which
found that alternating between queues beats combining the heuristics into one number.
Every state goes on every queue; a state expanded from one is dropped from the others.

This is the framework in which a goal-directed `progress` measure, novelty and FSX's
goal-free `option_count` can search together. Each heuristic is a callable, lower is
better; `boost` gives a queue that just improved its best value that many extra turns,
Fast Downward's preferred-operator boost applied to whole queues.
"""
from heapq import heappop, heappush
from itertools import count

from planiverse.planners.common import finish
from planiverse.planners.width.result import Budget, SearchStatistics


class MultiQueueSearch:
    """Round-robin greedy best-first over several heuristics.

    ```python
    from planiverse.planners.heuristic import MultiQueueSearch
    from planiverse.planners.fsx import option_count

    MultiQueueSearch([boxes, lambda s: -option_count(env, s, horizon=4, walkers=4)]).solve(env)
    ```
    """

    def __init__(self, heuristics, boost=0):
        if not heuristics:
            raise ValueError("MultiQueueSearch needs at least one heuristic")
        self.heuristics = list(heuristics)
        self.boost = boost

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics()
        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return finish("solved", [], [state], statistics, budget)
        tiebreak = count()
        queues = [[] for _ in self.heuristics]
        best = [float("inf")] * len(queues)
        credit = [0] * len(queues)
        closed = set()

        def push(node, plan, trace):
            for index, (queue, h) in enumerate(zip(queues, self.heuristics)):
                value = h(node)
                if value < best[index]:
                    best[index] = value
                    credit[index] += self.boost
                heappush(queue, (value, next(tiebreak), node, plan, trace))

        push(state, [], [state])
        turn = 0
        while any(queues):
            if budget.exhausted(statistics.expansions):
                return finish("out_of_budget", None, [], statistics, budget)
            index = next(i for i in range(len(queues)) if credit[i] > 0 and queues[i]) \
                if any(credit[i] > 0 and queues[i] for i in range(len(queues))) \
                else turn % len(queues)
            turn += 1
            if credit[index] > 0:
                credit[index] -= 1
            if not queues[index]:
                continue
            _, _, node, plan, trace = heappop(queues[index])
            if node.literals in closed:
                statistics.pruned_duplicate += 1
                continue
            closed.add(node.literals)
            statistics.expansions += 1
            for action, successor in env.successors(node):
                statistics.generated += 1
                if successor.literals in closed:
                    statistics.pruned_duplicate += 1
                    continue
                successor_plan, successor_trace = plan + [action], trace + [successor]
                if env.is_goal(successor):
                    return finish("solved", successor_plan, successor_trace, statistics,
                                  budget)
                if env.is_terminal(successor):
                    statistics.pruned_terminal += 1
                    continue
                push(successor, successor_plan, successor_trace)
        return finish("exhausted", None, [], statistics, budget)
