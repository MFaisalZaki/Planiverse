"""2BFS: best-first search over two queues, one for novelty and one for reward.

Lipovetzky, Ramírez and Geffner, *Classical Planning with Simulators: Results on the Atari
Video Games*, IJCAI 2015. IW(1) was the paper's headline, but its best planner on Atari was
this one: a best-first search that keeps **two open lists**, one ordered by novelty with
accumulated reward breaking ties and the other by accumulated reward with novelty breaking
ties, and expands from them alternately. One queue explores, the other exploits, and neither
discards anything, so unlike IW the search is complete.

Against these simulators the reward is the drop in `progress`, as everywhere else in this
library: the accumulated reward of a node is `progress(root) - progress(node)`, higher being
better. Without a `progress` measure both queues order by novelty alone and the search is a
novelty-first breadth-first search, which the result's status says.
"""
from heapq import heappop, heappush
from itertools import count

from planiverse.planners.common import finish
from planiverse.planners.width.novelty import NoveltyTable
from planiverse.planners.width.result import Budget, SearchStatistics


class TwoBFS:
    """Alternate between a `<novelty, -reward>` queue and a `<-reward, novelty>` queue.

    ```python
    from planiverse.planners.width import TwoBFS

    result = TwoBFS(width=1, progress=boxes).solve(env, Budget(max_expansions=5000))
    ```
    """

    def __init__(self, width=1, progress=None, strict=True):
        self.width = width
        self.progress = progress
        self.strict = strict

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics(widths_tried=(self.width,))
        table = NoveltyTable(self.width, strict=self.strict)
        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return finish("solved", [], [state], statistics, budget, self.width)

        base = self.progress(state) if self.progress else 0.0
        tiebreak = count()
        novelty = table.evaluate_and_record(state.literals)
        entry = (state, [], [state])
        explore = [((novelty, 0.0), next(tiebreak), entry)]
        exploit = [((0.0, novelty), next(tiebreak), entry)]
        closed = set()
        turn = 0

        while explore or exploit:
            if budget.exhausted(statistics.expansions):
                return self.__done__("out_of_budget", statistics, budget, table)
            queue = explore if (turn % 2 == 0 and explore) or not exploit else exploit
            turn += 1
            _, _, (node, plan, trace) = heappop(queue)
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
                reward = (base - self.progress(successor)) if self.progress else 0.0
                novelty = table.evaluate_and_record(successor.literals)
                entry = (successor, successor_plan, successor_trace)
                heappush(explore, ((novelty, -reward), next(tiebreak), entry))
                heappush(exploit, ((-reward, novelty), next(tiebreak), entry))
        return self.__done__("exhausted", statistics, budget, table)

    def __done__(self, status, statistics, budget, table):
        statistics.novelty_evaluations = table.evaluations
        statistics.tuples_enumerated = table.tuples_enumerated
        if self.progress is None:
            status = f"{status} (no progress measure; both queues ordered by novelty)"
        return finish(status, None, [], statistics, budget)
