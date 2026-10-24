"""KPIECE, a sampling-based tree planner from robotics, for a black-box forward model.

Robotics faced this problem first: a system whose dynamics are only available as a
forward propagator, no inverse, no distance one can steer along. Its planners grow a tree
by picking a node and propagating random controls from it, and differ in how they pick.

**KPIECE** (Şucan and Kavraki, *Kinodynamic Motion Planning by Interior-Exterior Cell
Exploration*, WAFR 2008) keeps a grid of cells over a low-dimensional projection; a cell is
picked by an importance that rises with how long it has gone unselected and falls with how
many motions it already holds, and a node in it at random. KPIECE also prefers cells on the
exterior of the covered region; a symbolic projection has no neighbouring cells to count, so
that term is not here.

The projection is `projection(state)`, `literals` by default, so a cell is an exact state
and only a coarser projection (a `progress` vector, a bucketing) makes the cell structure
do anything. A propagation is `random.randint(1, max_duration)` random actions, and every
state along it joins the tree, since under `successors` they were generated anyway.
"""
import math
import random

from planiverse.planners.common import SuccessorCache, finish
from planiverse.planners.width.result import Budget, SearchStatistics

class KinodynamicTree:
    """A tree grown by random propagation from the cells KPIECE selects.

    ```python
    from planiverse.planners.sampling import KinodynamicTree

    KinodynamicTree(projection=lambda s: (boxes(s),), seed=0).solve(env)
    ```
    """

    def __init__(self, projection=None, max_duration=5, max_iterations=100_000, seed=None):
        self.projection = projection or (lambda state: state.literals)
        self.max_duration = max_duration
        self.max_iterations = max_iterations
        self.random = random.Random(seed)

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics()
        cache = SuccessorCache(env, statistics, budget)
        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return finish("solved", [], [state], statistics, budget)
        cells = {}                  # projection -> list of (state, plan, trace)
        selections = {}             # projection -> times selected
        seen = {state.literals}

        def add(node, plan, trace):
            cells.setdefault(self.projection(node), []).append((node, plan, trace))

        add(state, [], [state])
        for iteration in range(1, self.max_iterations + 1):
            if cache.exhausted():
                return finish("out_of_budget", None, [], statistics, budget)
            key = self.__select__(cells, selections, iteration)
            selections[key] = selections.get(key, 0) + 1
            node, plan, trace = self.random.choice(cells[key])
            for _ in range(self.random.randint(1, self.max_duration)):
                if cache.exhausted():
                    break
                children = cache.expand(node)
                if not children:
                    break
                action, node = self.random.choice(children)
                plan, trace = plan + [action], trace + [node]
                if env.is_goal(node):
                    return finish("solved", plan, trace, statistics, budget)
                if env.is_terminal(node):
                    statistics.pruned_terminal += 1
                    break
                if node.literals in seen:
                    statistics.pruned_duplicate += 1
                    continue
                seen.add(node.literals)
                add(node, plan, trace)
            statistics.rollouts += 1
        return finish("failed", None, [], statistics, budget)

    def __select__(self, cells, selections, iteration):
        """The cell of highest importance, ties broken at random."""
        keys = list(cells)

        def importance(key):
            return (math.log(iteration + 1) / ((1 + selections.get(key, 0))
                                               * len(cells[key])))
        best = max(importance(key) for key in keys)
        return self.random.choice([key for key in keys if importance(key) == best])
