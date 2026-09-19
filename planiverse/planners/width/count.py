"""Count-based novelty and the trimmed open list.

Rosa and Lipovetzky, *Count-Based Novelty Exploration in Classical Planning*, ECAI 2024.
Classical novelty records whether a tuple of atoms has been seen; it is a binary signal, and
once every tuple has been seen it says nothing. Count-based novelty records **how often**
each tuple has appeared in the search tree, and scores a state by its rarest tuple, so the
ordering keeps discriminating after the first sighting: a state whose atoms have been seen
a hundred times is worse than one whose atoms have been seen twice, and exploration does
not switch off the way partitioning only postpones. The paper's second contribution is a
**trimmed open list** that keeps the frontier at a constant size by discarding the nodes
with the worst novelty when it overflows, which is what bounds the memory.

The paper's full text could not be read from this machine, so the definitions here are
the abstract's, made precise:

* the count-novelty of a state is `min` over its atom tuples of size at most `width` of the
  number of states already recorded that contained the tuple (`0` is classical novelty);
* the search is best-first on `(count-novelty, progress)`, novelty measured within the
  `progress` partition as BFWS does;
* when the open list exceeds `open_limit` entries, the worst half by that key is dropped.

Trimming discards, so the search is incomplete, which is the trade the paper makes.
"""
from heapq import heapify, heappop, heappush, nsmallest
from itertools import combinations, count

from planiverse.planners.common import finish
from planiverse.planners.width.novelty import MAX_PRACTICAL_WIDTH
from planiverse.planners.width.result import Budget, SearchStatistics


class CountNoveltyTable:
    """How many recorded states contained each atom tuple up to `width`."""

    def __init__(self, width=1, strict=True):
        if width < 1:
            raise ValueError(f"width must be at least 1, got {width}")
        if strict and width > MAX_PRACTICAL_WIDTH:
            raise ValueError(f"width {width} enumerates every {width}-tuple of every state's "
                             "atoms; pass strict=False if you mean it.")
        self.width = width
        self.counts = {}
        self.evaluations = 0
        self.tuples_enumerated = 0

    def evaluate(self, literals):
        """The count of the rarest tuple; 0 when some tuple has never been seen."""
        self.evaluations += 1
        atoms = sorted(literals)
        rarest = None
        for size in range(1, min(self.width, len(atoms)) + 1):
            for combo in combinations(atoms, size):
                self.tuples_enumerated += 1
                seen = self.counts.get(combo, 0)
                if seen == 0:
                    return 0
                rarest = seen if rarest is None else min(rarest, seen)
        return rarest if rarest is not None else 0

    def record(self, literals):
        atoms = sorted(literals)
        for size in range(1, min(self.width, len(atoms)) + 1):
            for combo in combinations(atoms, size):
                self.counts[combo] = self.counts.get(combo, 0) + 1

    def evaluate_and_record(self, literals):
        value = self.evaluate(literals)
        self.record(literals)
        return value


class BFNoS:
    """Best-First Novelty Search on count-based novelty, with a trimmed open list: the
    paper's planner, under its name.

    ```python
    from planiverse.planners.width import BFNoS

    result = BFNoS(progress=boxes, open_limit=2000).solve(env, budget)
    ```

    `open_limit=None` never trims, which keeps the search complete and the memory unbounded.
    """

    def __init__(self, width=1, progress=None, open_limit=10_000, strict=True):
        if open_limit is not None and open_limit < 2:
            raise ValueError(f"open_limit must be at least 2 or None, got {open_limit}")
        self.width = width
        self.progress = progress
        self.open_limit = open_limit
        self.strict = strict

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics(widths_tried=(self.width,))
        tables = {}
        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return finish("solved", [], [state], statistics, budget, self.width)

        def table_for(node):
            key = self.progress(node) if self.progress else 0
            if key not in tables:
                tables[key] = CountNoveltyTable(self.width, strict=self.strict)
            return tables[key]

        tiebreak = count()
        heap = [((table_for(state).evaluate_and_record(state.literals),
                  self.progress(state) if self.progress else 0), next(tiebreak),
                 state, [], [state])]
        closed = set()
        trimmed = 0
        while heap:
            if budget.exhausted(statistics.expansions):
                return self.__done__("out_of_budget", statistics, budget, tables, trimmed)
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
                                  budget, self.width)
                if env.is_terminal(successor):
                    statistics.pruned_terminal += 1
                    continue
                value = table_for(successor).evaluate_and_record(successor.literals)
                heappush(heap, ((value, self.progress(successor) if self.progress else 0),
                                next(tiebreak), successor, successor_plan, successor_trace))
            if self.open_limit is not None and len(heap) > self.open_limit:
                keep = nsmallest(self.open_limit // 2, heap)
                trimmed += len(heap) - len(keep)
                heap = keep
                heapify(heap)
        return self.__done__("exhausted" if not trimmed else "failed", statistics, budget,
                             tables, trimmed)

    def __done__(self, status, statistics, budget, tables, trimmed):
        statistics.novelty_evaluations = sum(t.evaluations for t in tables.values())
        statistics.tuples_enumerated = sum(t.tuples_enumerated for t in tables.values())
        statistics.pruned_novelty = trimmed
        return finish(status, None, [], statistics, budget)
