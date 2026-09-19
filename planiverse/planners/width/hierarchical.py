"""Hierarchical IW: IW(1) at two levels of abstraction.

Junyent, Gómez and Jonsson, *Hierarchical Width-Based Planning and Learning*, ICAPS 2021.
IW(k) is exponential in `k`, and width 2 is already the practical ceiling against a
simulator. The paper's answer is two IW(1) searches stacked: a **high-level** IW over a
small set of abstract features, each of whose expansions is a **low-level** IW over the
full state that runs until it finds states where an abstract feature has changed, which
become the high-level successors. Two levels of width 1 reach what one level of width 2
reaches, at a fraction of the cost, because the pairs are never enumerated: the high level
tracks one atom and the low level the other. The paper also learns a policy for the low
level; that half is not here, and this is the planning half on its own.

The high-level features are "incrementally discovered from low-level pruning decisions".
The paper's text could not be read from this machine, so the discovery rule used is
stated rather than hidden: **an atom joins the high-level feature set the first time the
low-level search discards a state that made it true**, that is, a transition whose child
was pruned for novelty while containing an atom its parent did not. The low level threw
away a change in that atom, so the high level takes it over. Pass `features` (a callable
`state -> frozenset` of abstract atoms) to fix the feature set instead, which is the
paper's setting when the abstraction is known.
"""
from collections import deque

from planiverse.planners.common import finish, remaining
from planiverse.planners.width.novelty import NoveltyTable
from planiverse.planners.width.result import Budget, SearchStatistics


class HierarchicalIW:
    """HIW(high_width, low_width), with discovered or given high-level features.

    ```python
    from planiverse.planners.width import HierarchicalIW

    result = HierarchicalIW(low_expansions=200).solve(env, Budget(max_expansions=5000))
    print(sorted(planner.features_found))
    ```

    - `low_expansions`: the expansion cap of each low-level IW call.
    - `progress`: optional; a low-level state that improves it is also offered to the high
      level, which is how the two levels cooperate when the abstraction is still empty.
    """

    def __init__(self, high_width=1, low_width=1, low_expansions=200, features=None,
                 progress=None, strict=True):
        if low_expansions < 1:
            raise ValueError("low_expansions must be at least 1")
        self.high_width = high_width
        self.low_width = low_width
        self.low_expansions = low_expansions
        self.features = features
        self.progress = progress
        self.strict = strict
        self.features_found = set()

    def __project__(self, state):
        if self.features is not None:
            return frozenset(self.features(state))
        return frozenset(state.literals) & frozenset(self.features_found)

    def __low_level__(self, env, start, budget, statistics):
        """IW(low_width) from `start`, capped at `low_expansions`.

        Returns `(goal, kept)`: a `(plan, trace)` to a goal if one was reached, and every
        low-level state the search kept, as `(state, plan, trace, improved)`. The high level
        decides which of them are its successors once the run is over, so that features the
        run discovered apply to the states it produced.
        """
        table = NoveltyTable(self.low_width, strict=self.strict)
        table.evaluate_and_record(start.literals)
        frontier = deque([(start, [], [start])])
        closed = {start.literals}
        candidates = []
        best = self.progress(start) if self.progress else None
        spent = 0
        while frontier:
            if spent >= self.low_expansions or budget.exhausted(statistics.expansions):
                break
            node, plan, trace = frontier.popleft()
            spent += 1
            statistics.expansions += 1
            for action, successor in env.successors(node):
                statistics.generated += 1
                if successor.literals in closed:
                    statistics.pruned_duplicate += 1
                    continue
                if table.evaluate_and_record(successor.literals) > self.low_width:
                    statistics.pruned_novelty += 1
                    if self.features is None:
                        gained = frozenset(successor.literals) - frozenset(node.literals)
                        self.features_found |= gained
                    continue
                closed.add(successor.literals)
                successor_plan, successor_trace = plan + [action], trace + [successor]
                if env.is_goal(successor):
                    return (successor_plan, successor_trace), candidates
                if env.is_terminal(successor):
                    statistics.pruned_terminal += 1
                    continue
                improved = self.progress is not None and self.progress(successor) < best
                if improved:
                    best = self.progress(successor)
                candidates.append((successor, successor_plan, successor_trace, improved))
                frontier.append((successor, successor_plan, successor_trace))
        statistics.novelty_evaluations += table.evaluations
        statistics.tuples_enumerated += table.tuples_enumerated
        return None, candidates

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics(widths_tried=(self.high_width, self.low_width))
        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return finish("solved", [], [state], statistics, budget, self.high_width)

        high = NoveltyTable(self.high_width, strict=self.strict)
        high.evaluate_and_record(self.__project__(state))
        frontier = deque([(state, [], [state])])
        closed = {state.literals}
        while frontier:
            if budget.exhausted(statistics.expansions):
                return finish("out_of_budget", None, [], statistics, budget)
            node, plan, trace = frontier.popleft()
            goal, candidates = self.__low_level__(env, node, budget, statistics)
            if goal is not None:
                low_plan, low_trace = goal
                return finish("solved", plan + low_plan, trace + low_trace[1:], statistics,
                              budget, self.high_width)
            # Projections are taken after the low-level run, so features it discovered
            # apply to the candidates it produced.
            base = self.__project__(node)
            for successor, low_plan, low_trace, improved in candidates:
                if successor.literals in closed:
                    continue
                if not improved and self.__project__(successor) == base:
                    continue
                if high.evaluate_and_record(self.__project__(successor)) > self.high_width:
                    statistics.pruned_novelty += 1
                    continue
                closed.add(successor.literals)
                frontier.append((successor, plan + low_plan, trace + low_trace[1:]))
        statistics.novelty_evaluations += high.evaluations
        return finish("failed", None, [], statistics, budget)
