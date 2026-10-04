"""Feature Space Search: search organised by a handful of features.

Shoham and Schaeffer, *The FESS Algorithm: A Feature Based Approach to Single-Agent
Search*, IEEE Conference on Games 2020. The search space is projected onto a small
feature space, each feature a numeric property of a state (the number of blocks cleared,
how connected the board is), and the search is organised by the cells of that space: it
**cycles over the cells** giving each an equal share of expansions, and within a cell
expands the move of least accumulated weight. Moves an *advisor* recommends weigh 1, every
other move weighs `penalty`, so the cheap paths are the advised ones and the others are
still reached eventually. FESS was the first program to solve all 90 XSokoban levels.

The paper's full text could not be read from this machine. What is implemented is the
mechanism as its abstract and the description of its solver state it: cells keyed by the
feature vector, round-robin over cells, least-weight move first, advisors marking moves.
The Sokoban-specific features and advisors are the caller's to supply:

- `features(state) -> tuple` of numbers: the projection.
- `advisors`: callables `(state, actions) -> iterable of actions` recommending some of
  the applicable actions. Optional; with none, every move weighs the same and the search is
  a round-robin breadth-first over feature cells, which is already a diversity mechanism.

A node is expanded when it is created, since knowing its moves means knowing its children
under the `successors` contract; the counted expansions are those creations.
"""
from heapq import heappop, heappush
from itertools import count

from planiverse.planners.common import finish
from planiverse.planners.width.result import Budget, SearchStatistics


class FeatureSpaceSearch:
    """FESS over `features`, with optional `advisors`.

    ```python
    from planiverse.planners.heuristic import FeatureSpaceSearch

    FeatureSpaceSearch(features=lambda s: (boxes(s),)).solve(env, budget)
    ```
    """

    def __init__(self, features, advisors=(), penalty=10.0):
        if features is None:
            raise ValueError("FeatureSpaceSearch needs a features projection")
        self.features = features
        self.advisors = tuple(advisors)
        self.penalty = penalty

    def __weights__(self, state, children):
        actions = [action for action, _ in children]
        advised = set()
        for advisor in self.advisors:
            advised.update(advisor(state, actions))
        return [1.0 if action in advised else self.penalty for action in actions]

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics()
        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return finish("solved", [], [state], statistics, budget)

        cells = {}                  # feature vector -> heap of (weight, tiebreak, move)
        order = []                  # cells in creation order, cycled over
        closed = set()
        tiebreak = count()

        def open_node(node, plan, trace, weight):
            """Expand `node` and file its moves under its cell."""
            statistics.expansions += 1
            children = env.successors(node)
            statistics.generated += len(children)
            key = self.features(node)
            if key not in cells:
                cells[key] = []
                order.append(key)
            heap = cells[key]
            for (action, successor), move_weight in zip(children,
                                                        self.__weights__(node, children)):
                heappush(heap, (weight + move_weight, next(tiebreak),
                                (action, successor, plan, trace)))

        closed.add(state.literals)
        open_node(state, [], [state], 0.0)
        cursor = 0
        while any(cells.values()):
            if budget.exhausted(statistics.expansions):
                return finish("out_of_budget", None, [], statistics, budget)
            key = order[cursor % len(order)]
            cursor += 1
            heap = cells[key]
            if not heap:
                continue
            weight, _, (action, successor, plan, trace) = heappop(heap)
            if successor.literals in closed:
                statistics.pruned_duplicate += 1
                continue
            closed.add(successor.literals)
            successor_plan, successor_trace = plan + [action], trace + [successor]
            if env.is_goal(successor):
                return finish("solved", successor_plan, successor_trace, statistics, budget)
            if env.is_terminal(successor):
                statistics.pruned_terminal += 1
                continue
            open_node(successor, successor_plan, successor_trace, weight)
        return finish("exhausted", None, [], statistics, budget)
