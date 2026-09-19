"""Go-Explore, the exploration phase.

Ecoffet, Huizinga, Lehman, Stanley and Clune, *Go-Explore: a New Approach for
Hard-Exploration Problems*, arXiv 1901.10995 (2019), and *First Return, Then Explore*,
Nature 590 (2021). Keep an **archive of cells**, a cell being a coarse projection of a
state, each holding the best trajectory that reached it. Repeatedly: pick a cell, with
weight favouring cells rarely chosen and rarely seen; **return** to it by restoring the
simulator, with no exploration on the way; **explore** from it with random actions for a
few steps; and add every new cell reached, or replace a cell's trajectory with a better or
shorter one. In a deterministic, restorable simulator the trajectory to a goal cell is
the plan, and that is the setting here: every state can be expanded again at any time,
so returning to a cell costs nothing.

The paper's second phase trains a policy by imitation on those trajectories so they
survive stochasticity; it is training, and it is not here.

The cell selection weight is the paper's: a sum over the attributes *times chosen*, *times
chosen since a new cell was found from it* and *times seen* of
`w_a · (1 / (v + ε1)) ** p_a + ε2`, with the paper's `ε1 = 0.001`, `ε2 = 0.00001` and
`p_a = 0.5`, and weights `0.1`, `0.1` and `0.3` in that order. Exploration repeats the
previous action with probability `repeat` (the paper's 0.95) and runs for `explore_steps`
actions (the paper's 100; expansions are dearer here, so 20). A cell's score is the
negated `progress` of its state, higher being better; without `progress` every cell scores
the same and shorter trajectories win.

`cell(state)` defaults to `literals`, the exact state, which makes the archive the set of
distinct states reached; a coarser projection is the paper's own choice for Atari and
the right one for a large environment.
"""
import random

from planiverse.planners.common import SuccessorCache, finish
from planiverse.planners.width.result import Budget, SearchStatistics

WEIGHTS = {"chosen": 0.1, "since_new": 0.1, "seen": 0.3}
EPSILON_1, EPSILON_2, POWER = 0.001, 0.00001, 0.5


class Cell:
    __slots__ = ("state", "plan", "trace", "score", "chosen", "since_new", "seen")

    def __init__(self, state, plan, trace, score):
        self.state, self.plan, self.trace, self.score = state, plan, trace, score
        self.chosen = self.since_new = 0
        self.seen = 1

    def weight(self):
        return sum(w * (1.0 / (getattr(self, name) + EPSILON_1)) ** POWER + EPSILON_2
                   for name, w in WEIGHTS.items())


class GoExplore:
    """Go-Explore's exploration phase over an archive of cells.

    ```python
    from planiverse.planners.sampling import GoExplore

    GoExplore(progress=boxes, explore_steps=20, seed=0).solve(env, budget)
    print(len(planner.archive))
    ```
    """

    def __init__(self, progress=None, cell=None, explore_steps=20, repeat=0.95,
                 max_iterations=100_000, seed=None):
        self.progress = progress
        self.cell = cell or (lambda state: state.literals)
        self.explore_steps = explore_steps
        self.repeat = repeat
        self.max_iterations = max_iterations
        self.random = random.Random(seed)
        self.archive = {}

    def __score__(self, state):
        return -float(self.progress(state)) if self.progress else 0.0

    def __visit__(self, state, plan, trace):
        """Record `state` in the archive. Returns True when the archive improved."""
        key = self.cell(state)
        score = self.__score__(state)
        cell = self.archive.get(key)
        if cell is None:
            self.archive[key] = Cell(state, plan, trace, score)
            return True
        cell.seen += 1
        if score > cell.score or (score == cell.score and len(plan) < len(cell.plan)):
            cell.state, cell.plan, cell.trace, cell.score = state, plan, trace, score
            return True
        return False

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics()
        cache = SuccessorCache(env, statistics, budget)
        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return finish("solved", [], [state], statistics, budget)
        self.archive = {}
        self.__visit__(state, [], [state])
        for _ in range(self.max_iterations):
            if cache.exhausted():
                return finish("out_of_budget", None, [], statistics, budget)
            cells = list(self.archive.values())
            cell = self.random.choices(cells, weights=[c.weight() for c in cells])[0]
            cell.chosen += 1
            cell.since_new += 1
            found_new = False
            node, plan, trace = cell.state, list(cell.plan), list(cell.trace)
            previous = None
            for _ in range(self.explore_steps):
                if cache.exhausted():
                    break
                children = cache.expand(node)
                if not children:
                    break
                choice = None
                if previous is not None and self.random.random() < self.repeat:
                    choice = next(((a, s) for a, s in children if a == previous), None)
                if choice is None:
                    choice = self.random.choice(children)
                previous, node = choice
                plan, trace = plan + [previous], trace + [node]
                if env.is_goal(node):
                    return finish("solved", plan, trace, statistics, budget)
                if self.__visit__(node, plan, trace):
                    found_new = True
                if env.is_terminal(node):
                    break
            statistics.episodes += 1
            if found_new:
                cell.since_new = 0
        return finish("failed", None, [], statistics, budget)
