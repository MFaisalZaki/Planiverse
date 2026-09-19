"""Prioritised IW: IW(1) whose novelty test knows about reward.

Shleyfman, Tuisov and Domshlak, *Blind Search for Atari-Like Online Planning Revisited*,
IJCAI 2016. Two changes to IW(1), both aimed at the fact that IW's blindness to reward is
what makes it miss things a few steps past the first novel state:

* **Priority.** The queue is still breadth-first by depth, but among states at the same
  depth the one with the highest accumulated reward is expanded first.
* **Reward-aware novelty, with reopening.** IW records, per atom, that it has been seen.
  p-IW records, per atom, the **best accumulated reward** any state containing it was
  reached with. A state is kept if it has an atom never seen before, *or* if it reaches
  some atom with a strictly higher reward than the best recorded for that atom, in which
  case the record is raised. Duplicate detection works the same way: a state already seen
  is reopened when reached again with a higher reward.

The paper's second contribution, treating online action selection as a race between the
root's actions, is separate and not part of this class.

The accumulated reward of a node is `progress(root) - progress(node)`, as in the rest of
the library. Without `progress` every reward is zero and this is IW(k), which the status
says. `width` above 1 generalises the per-atom record to per-tuple, the way `NoveltyTable`
does; the paper only ran width 1.
"""
from heapq import heappop, heappush
from itertools import combinations, count

from planiverse.planners.common import finish
from planiverse.planners.width.novelty import MAX_PRACTICAL_WIDTH
from planiverse.planners.width.result import Budget, SearchStatistics


class RewardNoveltyTable:
    """Per tuple, the best accumulated reward a state containing it was reached with."""

    def __init__(self, width=1, strict=True):
        if width < 1:
            raise ValueError(f"width must be at least 1, got {width}")
        if strict and width > MAX_PRACTICAL_WIDTH:
            raise ValueError(f"width {width} enumerates every {width}-tuple of every state's "
                             "atoms; pass strict=False if you mean it.")
        self.width = width
        self.best = {}
        self.evaluations = 0
        self.tuples_enumerated = 0

    def evaluate_and_record(self, literals, reward):
        """The smallest tuple size that is new or reached with more reward, else width+1."""
        self.evaluations += 1
        atoms = sorted(literals)
        novelty = self.width + 1
        for size in range(1, min(self.width, len(atoms)) + 1):
            for combo in combinations(atoms, size):
                self.tuples_enumerated += 1
                seen = self.best.get(combo)
                if seen is None or reward > seen:
                    self.best[combo] = reward
                    novelty = min(novelty, size)
        return novelty


class PrioritizedIW:
    """p-IW(k): breadth-first by depth, best reward first within a depth, reward-aware
    novelty with reopening.

    ```python
    from planiverse.planners.width import PrioritizedIW

    result = PrioritizedIW(progress=boxes).solve(env, Budget(max_expansions=5000))
    ```
    """

    def __init__(self, width=1, progress=None, strict=True):
        self.width = width
        self.progress = progress
        self.strict = strict

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics(widths_tried=(self.width,))
        table = RewardNoveltyTable(self.width, strict=self.strict)
        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return finish("solved", [], [state], statistics, budget, self.width)

        base = self.progress(state) if self.progress else 0.0
        tiebreak = count()
        table.evaluate_and_record(state.literals, 0.0)
        best_reward = {state.literals: 0.0}
        queue = [((0, 0.0), next(tiebreak), state, [], [state])]

        while queue:
            if budget.exhausted(statistics.expansions):
                return self.__done__("out_of_budget", statistics, budget, table)
            (depth, _), _, node, plan, trace = heappop(queue)
            statistics.expansions += 1

            for action, successor in env.successors(node):
                statistics.generated += 1
                reward = (base - self.progress(successor)) if self.progress else 0.0
                seen = best_reward.get(successor.literals)
                if seen is not None and reward <= seen:
                    statistics.pruned_duplicate += 1
                    continue
                if table.evaluate_and_record(successor.literals, reward) > self.width:
                    statistics.pruned_novelty += 1
                    continue
                best_reward[successor.literals] = reward
                successor_plan, successor_trace = plan + [action], trace + [successor]
                if env.is_goal(successor):
                    return finish("solved", successor_plan, successor_trace, statistics,
                                  budget, self.width)
                if env.is_terminal(successor):
                    statistics.pruned_terminal += 1
                    continue
                heappush(queue, ((depth + 1, -reward), next(tiebreak), successor,
                                 successor_plan, successor_trace))
        return self.__done__("exhausted", statistics, budget, table)

    def __done__(self, status, statistics, budget, table):
        statistics.novelty_evaluations = table.evaluations
        statistics.tuples_enumerated = table.tuples_enumerated
        if self.progress is None:
            status = f"{status} (no progress measure; p-IW degraded to IW)"
        return finish(status, None, [], statistics, budget)
