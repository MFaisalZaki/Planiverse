"""BFWS(R): novelty partitioned by progress and by a set of relevant atoms.

Lipovetzky and Geffner, *Best-First Width Search: Exploration and Exploitation in Classical
Planning*, AAAI 2017, where the evaluation function `f5 = <w_{(#g, #r)}, #g>` made BFWS the
strongest satisficing planner of its year; and Francès, Ramírez, Lipovetzky and Geffner,
*Purely Declarative Action Descriptions are Overrated: Classical Planning with Simulators*,
IJCAI 2017, which showed how to get the set `R` **without a model**: run IW(1) (or IW(2))
from the initial state, and take the atoms on the paths it finds to states that achieve
something, so the planner needs nothing beyond `successors` and `literals`.

`#r(n)` counts how many atoms of `R` have been true somewhere on the path to `n`, and the
novelty of `n` is measured among the states with the same `(progress, #r)`: reaching one
more relevant atom renews the novelty budget, the way a drop in `progress` does. Ties break
on `progress`, then on `#r`.

With a simulator there is no goal conjunction whose atoms the pre-search could aim at, so
"achieves something" is read as **strictly improves `progress`**: `R` is the union of the
literals on IW's paths to every state that beat the best progress seen when it was
generated. If the pre-search finds a goal, that plan is returned directly, as the paper's
planner does. Without `progress` there is nothing to aim at, `R` is empty and the search is
plain BFWS, which the status says.
"""
from heapq import heappop, heappush
from itertools import count

from planiverse.planners.common import finish, remaining
from planiverse.planners.width.novelty import NoveltyTable, PartitionedNovelty
from planiverse.planners.width.result import Budget, SearchStatistics


class BFWSR:
    """BFWS with `R` from an IW pre-search.

    ```python
    from planiverse.planners.width import BFWSR

    result = BFWSR(width=1, progress=boxes).solve(env, Budget(max_expansions=5000))
    print(sorted(result.statistics.widths_tried), len(planner.relevant))
    ```

    - `r_width`: the IW width of the pre-search (the paper uses 1, then 2 when 1 finds
      nothing).
    - `r_share`: the fraction of the expansion budget the pre-search may spend.
    - `prune`: discard states whose novelty exceeds `width` (the k-BFWS variant) instead of
      sorting them last.
    """

    def __init__(self, width=1, progress=None, r_width=1, r_share=0.25, prune=False,
                 strict=True):
        if not 0.0 < r_share < 1.0:
            raise ValueError(f"r_share must be in (0, 1), got {r_share}")
        self.width = width
        self.progress = progress
        self.r_width = r_width
        self.r_share = r_share
        self.prune = prune
        self.strict = strict
        self.relevant = frozenset()

    # ------------------------------------------------------------------ the pre-search

    def __find_relevant__(self, env, state, budget, statistics):
        """IW(r_width) from `state`; returns `(plan, trace)` if it reaches a goal, else None.

        Fills `self.relevant` with the atoms on the paths to each state that improved the
        best progress seen so far.
        """
        table = NoveltyTable(self.r_width, strict=self.strict)
        table.evaluate_and_record(state.literals)
        best = self.progress(state)
        queue = [(state, [], [state], frozenset(state.literals))]
        closed = {state.literals}
        relevant = set()
        head = 0
        while head < len(queue):
            if budget.exhausted(statistics.expansions):
                break
            node, plan, trace, path_atoms = queue[head]
            head += 1
            statistics.expansions += 1
            for action, successor in env.successors(node):
                statistics.generated += 1
                if successor.literals in closed:
                    statistics.pruned_duplicate += 1
                    continue
                if table.evaluate_and_record(successor.literals) > self.r_width:
                    statistics.pruned_novelty += 1
                    continue
                closed.add(successor.literals)
                atoms = path_atoms | frozenset(successor.literals)
                if env.is_goal(successor):
                    self.relevant = frozenset(relevant | atoms)
                    return plan + [action], trace + [successor]
                if env.is_terminal(successor):
                    statistics.pruned_terminal += 1
                    continue
                value = self.progress(successor)
                if value < best:
                    best = value
                    relevant |= atoms
                queue.append((successor, plan + [action], trace + [successor], atoms))
        statistics.novelty_evaluations += table.evaluations
        statistics.tuples_enumerated += table.tuples_enumerated
        self.relevant = frozenset(relevant)
        return None

    # ------------------------------------------------------------------ the search

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics(widths_tried=(self.r_width, self.width))
        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return finish("solved", [], [state], statistics, budget, self.width)

        if self.progress is not None:
            share = (None if budget.max_expansions is None
                     else max(1, int(budget.max_expansions * self.r_share)))
            found = self.__find_relevant__(env, state,
                                           Budget(share, budget.max_seconds).start(), statistics)
            if found is not None:
                plan, trace = found
                return finish("solved", plan, trace, statistics, budget, self.r_width)

        novelty = PartitionedNovelty(self.width, strict=self.strict)
        tiebreak = count()
        reached = frozenset(state.literals) & self.relevant
        opened = novelty.evaluate_and_record(self.__partition__(state, reached), state.literals)
        heap = [(self.__key__(state, reached, opened), next(tiebreak), state, [], [state],
                 reached)]
        closed = set()

        while heap:
            if budget.exhausted(statistics.expansions):
                return self.__done__("out_of_budget", statistics, budget, novelty)
            _, _, node, plan, trace, reached = heappop(heap)
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
                now = reached | (frozenset(successor.literals) & self.relevant)
                score = novelty.evaluate_and_record(self.__partition__(successor, now),
                                                    successor.literals)
                if self.prune and score > self.width:
                    statistics.pruned_novelty += 1
                    continue
                heappush(heap, (self.__key__(successor, now, score), next(tiebreak),
                                successor, successor_plan, successor_trace, now))
        return self.__done__("exhausted", statistics, budget, novelty)

    def __partition__(self, state, reached):
        return ((self.progress(state) if self.progress else 0), len(reached))

    def __key__(self, state, reached, novelty):
        return (novelty, self.progress(state) if self.progress else 0, -len(reached))

    def __done__(self, status, statistics, budget, novelty):
        statistics.novelty_evaluations += novelty.evaluations
        statistics.tuples_enumerated += novelty.tuples_enumerated
        if self.progress is None:
            status = f"{status} (no progress measure; R is empty and BFWS(R) is BFWS)"
        return finish(status, None, [], statistics, budget)
