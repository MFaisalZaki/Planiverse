"""Novelty relative to a heuristic, and the number of novel atoms as a heuristic.

Katz, Lipovetzky, Moshkovich and Tuisov, *Adapting Novelty to Classical Planning as
Heuristic Search*, ICAPS 2017. Plain novelty asks whether an atom has been seen at all;
this asks whether it has been seen **in a state at least as good**: an atom of `s` is novel
when no state generated so far with heuristic value `<= h(s)` contained it. The count of
such atoms is the state's *quantified novelty*, `QN(s)`, and it doubles as a heuristic:
a state with many atoms never seen this close to the goal is worth expanding.

The search is a greedy best-first search over `(-QN, h)` by default: most novel first, the
heuristic breaking ties. `order="h"` puts the heuristic first and lets novelty break ties,
and `order="binary"` keeps only whether the state is novel at all, which is the paper's
non-quantified variant. Nothing is discarded, so all three are complete.

Here `h` is the `progress` measure, as everywhere in the library; without one every state
has the same heuristic value and the test collapses to ordinary novelty.
"""
from heapq import heappop, heappush
from itertools import count

from planiverse.planners.common import finish
from planiverse.planners.width.result import Budget, SearchStatistics

ORDERS = ("qn", "h", "binary")


class HeuristicNovelty:
    """Per atom, the best heuristic value of any state seen containing it."""

    def __init__(self):
        self.best = {}
        self.evaluations = 0

    def evaluate_and_record(self, literals, h):
        """How many atoms of the state are novel at heuristic value `h`; then record."""
        self.evaluations += 1
        novel = 0
        for atom in literals:
            seen = self.best.get(atom)
            if seen is None or h < seen:
                novel += 1
                self.best[atom] = h
        return novel


class QuantifiedNoveltySearch:
    """Greedy best-first search on quantified novelty and the heuristic.

    ```python
    from planiverse.planners.width import QuantifiedNoveltySearch

    result = QuantifiedNoveltySearch(progress=boxes).solve(env, Budget(max_expansions=5000))
    ```
    """

    def __init__(self, progress=None, order="qn"):
        if order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}, got {order!r}")
        self.progress = progress
        self.order = order

    def __key__(self, qn, h):
        if self.order == "qn":
            return (-qn, h)
        if self.order == "h":
            return (h, -qn)
        return (0 if qn else 1, h)

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics(widths_tried=(1,))
        table = HeuristicNovelty()
        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return finish("solved", [], [state], statistics, budget, 1)

        h = self.progress(state) if self.progress else 0
        tiebreak = count()
        heap = [(self.__key__(table.evaluate_and_record(state.literals, h), h),
                 next(tiebreak), state, [], [state])]
        closed = set()
        while heap:
            if budget.exhausted(statistics.expansions):
                return self.__done__("out_of_budget", statistics, budget, table)
            _, _, node, plan, trace = heappop(heap)
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
                                  budget, 1)
                if env.is_terminal(successor):
                    statistics.pruned_terminal += 1
                    continue
                h = self.progress(successor) if self.progress else 0
                qn = table.evaluate_and_record(successor.literals, h)
                heappush(heap, (self.__key__(qn, h), next(tiebreak), successor,
                                successor_plan, successor_trace))
        return self.__done__("exhausted", statistics, budget, table)

    def __done__(self, status, statistics, budget, table):
        statistics.novelty_evaluations = table.evaluations
        if self.progress is None:
            status = f"{status} (no progress measure; novelty measured at one heuristic value)"
        return finish(status, None, [], statistics, budget)
