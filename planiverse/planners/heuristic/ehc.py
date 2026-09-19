"""Enforced Hill-Climbing: breadth-first to the first strictly better state, then commit.

Hoffmann and Nebel, *The FF Planning System: Fast Plan Generation Through Heuristic
Search*, JAIR 2001. FF's search: from the current state, breadth-first search until a
state with a strictly lower heuristic value turns up, move there, and repeat until the goal.
It never backtracks over a commitment, which makes it incomplete and fast, and it is
`SIW` with the heuristic doing the job novelty does there: SIW's legs stop at the
first state that improves `progress` and are bounded by novelty; EHC's stop at the same
place and are bounded by nothing but the budget.

Two options from the simulator setting: `novelty_width` prunes each leg's breadth-first
search by novelty, which bounds it the way SIW's legs are bounded, and `avoid_dead_ends`
refuses to commit to a state `is_terminal` says is over, the same refusal that turns SIW's
failures on Puzznic into solutions.
"""
from collections import deque

from planiverse.planners.common import finish
from planiverse.planners.width.novelty import NoveltyTable
from planiverse.planners.width.result import Budget, SearchStatistics


class EnforcedHillClimbing:
    """FF's enforced hill-climbing on `progress`.

    ```python
    from planiverse.planners.heuristic import EnforcedHillClimbing

    EnforcedHillClimbing(progress=boxes).solve(env, budget)
    ```
    """

    def __init__(self, progress, novelty_width=None, avoid_dead_ends=True, max_rounds=1000,
                 strict=True):
        if progress is None:
            raise ValueError("EnforcedHillClimbing needs a progress measure to climb")
        self.progress = progress
        self.novelty_width = novelty_width
        self.avoid_dead_ends = avoid_dead_ends
        self.max_rounds = max_rounds
        self.strict = strict

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics()
        if state is None:
            state, _ = env.reset()
        plan, trace = [], [state]
        for _ in range(self.max_rounds):
            if env.is_goal(state):
                return finish("solved", plan, trace, statistics, budget, self.novelty_width)
            leg = self.__climb__(env, state, budget, statistics)
            if leg is None:
                status = "out_of_budget" if budget.exhausted(statistics.expansions) else "failed"
                return finish(status, None, [], statistics, budget)
            state, leg_plan, leg_trace = leg
            plan += leg_plan
            trace += leg_trace
        return finish("failed", None, [], statistics, budget)

    def __climb__(self, env, start, budget, statistics):
        """Breadth-first from `start` to the first state with lower progress."""
        baseline = self.progress(start)
        table = (NoveltyTable(self.novelty_width, strict=self.strict)
                 if self.novelty_width else None)
        if table is not None:
            table.evaluate_and_record(start.literals)
        frontier = deque([(start, [], [])])
        closed = {start.literals}
        while frontier:
            if budget.exhausted(statistics.expansions):
                return None
            node, plan, trace = frontier.popleft()
            statistics.expansions += 1
            for action, successor in env.successors(node):
                statistics.generated += 1
                if successor.literals in closed:
                    statistics.pruned_duplicate += 1
                    continue
                if table is not None and \
                        table.evaluate_and_record(successor.literals) > self.novelty_width:
                    statistics.pruned_novelty += 1
                    continue
                closed.add(successor.literals)
                successor_plan, successor_trace = plan + [action], trace + [successor]
                if env.is_goal(successor):
                    return successor, successor_plan, successor_trace
                if env.is_terminal(successor):
                    statistics.pruned_terminal += 1
                    if self.avoid_dead_ends:
                        continue
                if self.progress(successor) < baseline:
                    return successor, successor_plan, successor_trace
                if not env.is_terminal(successor):
                    frontier.append((successor, successor_plan, successor_trace))
        return None
