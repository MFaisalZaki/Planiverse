"""Best-First Width Search.

IW uses novelty as a **filter**: states that fail the test are thrown away, which bounds the
frontier beautifully and makes IW(k) incomplete. BFWS uses novelty as a **sort key**: nothing
is discarded, so completeness is kept, and the search still goes where the novelty is.

The other half of the idea is that novelty is measured *within a partition* of the state
space rather than globally. Plain novelty runs out — once every atom has been seen somewhere,
no state is ever novel again and the ordering goes flat. Partitioning by something that
measures progress gives each partition its own novelty budget, so arriving somewhere new
renews exploration instead of ending it.

Classically the partition is the number of unachieved goals, giving the evaluation function
`f5 = <w_{#g}, #g>`. **A simulator has no goal conjunction to count** — `is_goal` is a black
box — so `#g` has to come from somewhere else. `BFWS` takes a `progress` callback for it, and
degrades to `<w, h>` without one, which is a weaker search rather than a broken one.
"""
from heapq import heappop, heappush
from itertools import count

from planiverse.planners.width.novelty import PartitionedNovelty
from planiverse.planners.width.result import Budget, SearchResult, SearchStatistics


class BFWSSearch:
    """Best-first search ordered by `<novelty, progress, heuristic>`.

    ```python
    from planiverse.planners.width import BFWSSearch

    result = BFWSSearch(
        width=1,
        progress=lambda s: s.blocks_remaining,      # stands in for #unachieved-goals
    ).solve(env, budget=Budget(max_expansions=500))
    ```

    Both callbacks are optional and both are worth supplying. With neither, the key is
    novelty alone and ties break FIFO, which is a breadth-first search that prefers novel
    states — respectable, but it has no idea which way the goal is.
    """

    def __init__(self, width=1, progress=None, heuristic=None, partition=None, strict=True):
        """
        - `progress(state)` — lower is better; stands in for the unachieved-goal count.
        - `heuristic(state)` — lower is better; breaks ties among equally-progressed states.
        - `partition(state)` — what novelty is measured within. Defaults to `progress`, which
          is the classical choice; pass something else to partition on more than progress.
        """
        self.width = width
        self.progress = progress
        self.heuristic = heuristic
        self.partition = partition
        self.strict = strict

    def __partition_of__(self, state):
        if self.partition is not None:
            return self.partition(state)
        if self.progress is not None:
            return self.progress(state)
        return 0            # one partition: novelty measured globally

    def __key__(self, state, novelty):
        key = [novelty]
        if self.progress is not None:
            key.append(self.progress(state))
        if self.heuristic is not None:
            key.append(self.heuristic(state))
        return tuple(key)

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics(widths_tried=(self.width,))
        novelty = PartitionedNovelty(self.width, strict=self.strict)
        tiebreak = count()          # FIFO among equal keys, and keeps states out of compares

        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return self.__result__("solved", [], [state], statistics, budget, novelty)

        heap = []
        opened = novelty.evaluate_and_record(self.__partition_of__(state), state.literals)
        heappush(heap, (self.__key__(state, opened), next(tiebreak), state, [], [state]))
        closed = set()

        while heap:
            if budget.exhausted(statistics.expansions):
                return self.__result__("out_of_budget", None, [], statistics, budget, novelty)

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

                successor_plan = plan + [action]
                successor_trace = trace + [successor]

                # Before the terminal test, because every absorbing goal state in this
                # library is terminal as well as a goal.
                if env.is_goal(successor):
                    return self.__result__("solved", successor_plan, successor_trace,
                                           statistics, budget, novelty)
                if env.is_terminal(successor):
                    statistics.pruned_terminal += 1
                    continue

                # Novelty orders; it never discards. That is the whole difference from IW,
                # and it is why BFWS stays complete at any width.
                score = novelty.evaluate_and_record(self.__partition_of__(successor),
                                                    successor.literals)
                heappush(heap, (self.__key__(successor, score), next(tiebreak),
                                successor, successor_plan, successor_trace))

        return self.__result__("exhausted", None, [], statistics, budget, novelty)

    def __result__(self, status, plan, trace, statistics, budget, novelty):
        statistics.elapsed = budget.elapsed()
        statistics.novelty_evaluations = novelty.evaluations
        statistics.tuples_enumerated = novelty.tuples_enumerated
        return SearchResult(plan=plan, states=trace, status=status,
                            width=self.width if status == "solved" else None,
                            statistics=statistics)
