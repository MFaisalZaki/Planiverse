"""Focused macro-actions: short action sequences with small net effects, found by search.

Allen, Katz, Klinger, Konidaris, Riemer and Tesauro, *Efficient Black-Box Planning Using
Macro-Actions with Focused Effects*, IJCAI 2021. The goal-count heuristic assumes the
goal's atoms can be achieved one at a time; primitive actions usually cannot (a Rubik's
cube turn moves twenty stickers), and the heuristic misleads. A **focused macro** is an
action sequence whose net effect on the state is small, which is exactly what makes the
heuristic accurate again, and such macros can be found by searching the simulator for
sequences with the smallest effect footprints. Planning then runs over the macros.

Here the footprint of a sequence is the symmetric difference between the `literals` of
the state it starts from and the state it ends in, and the discovery is a breadth-first
enumeration of sequences up to `length` from the initial state and from `probes` states
reached by random walks, keeping the `count` sequences with the smallest non-empty
footprints, one per distinct footprint. The paper searches the sequence space best-first
on the footprint size instead; the enumeration is the same criterion applied exhaustively
at small lengths. `MacroPlanner` is greedy best-first on `progress` whose successors are
the macros' applications, primitives included by default so completeness is not lost.
"""
import random
from heapq import heappop, heappush
from itertools import count

from planiverse.planners.common import SuccessorCache, finish
from planiverse.planners.width.result import Budget, SearchStatistics


class FocusedMacros:
    """Discover `count` macros of up to `length` actions with the smallest footprints."""

    def __init__(self, length=3, count=8, probes=5, walk=10, share=0.2, seed=None):
        self.length = length
        self.count = count
        self.probes = probes
        self.walk = walk
        self.share = share
        self.random = random.Random(seed)
        self.macros = []

    def discover(self, env, cache, root):
        starts = [root]
        for _ in range(self.probes):
            node = root
            for _ in range(self.walk):
                children = cache.expand(node)
                if not children or cache.exhausted():
                    break
                _, node = self.random.choice(children)
                if env.is_terminal(node) or env.is_goal(node):
                    break
            starts.append(node)
        footprints = {}
        for start in starts:
            frontier = [(start, [])]
            for _ in range(self.length):
                layer = []
                for node, actions in frontier:
                    if cache.exhausted():
                        break
                    for action, successor in cache.expand(node):
                        sequence = actions + [action]
                        effect = frozenset(successor.literals ^ start.literals)
                        if effect and (effect not in footprints
                                       or len(footprints[effect]) > len(sequence)):
                            footprints[effect] = sequence
                        if not env.is_terminal(successor):
                            layer.append((successor, sequence))
                frontier = layer
        ranked = sorted(footprints.items(), key=lambda item: (len(item[0]), len(item[1])))
        self.macros = [tuple(sequence) for _, sequence in ranked[:self.count]]
        return self.macros


class MacroPlanner:
    """Greedy best-first on `progress` over focused macros (and primitives).

    ```python
    from planiverse.planners.macros import MacroPlanner

    MacroPlanner(progress=boxes, length=3, count=8, seed=0).solve(env, budget)
    print(planner.macros.macros)
    ```
    """

    def __init__(self, progress=None, length=3, count=8, include_primitives=True, seed=None):
        self.progress = progress
        self.macros = FocusedMacros(length, count, seed=seed)
        self.include_primitives = include_primitives

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics()
        cache = SuccessorCache(env, statistics, budget)
        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return finish("solved", [], [state], statistics, budget)
        share = (None if budget.max_expansions is None
                 else max(1, int(budget.max_expansions * self.macros.share)))
        cache.budget = Budget(share, budget.max_seconds).start()
        self.macros.discover(env, cache, state)
        cache.budget = budget
        moves = list(self.macros.macros)
        tiebreak = count()
        heap = [((self.progress(state) if self.progress else 0), next(tiebreak), state, [],
                 [state])]
        closed = set()
        while heap:
            if cache.exhausted():
                return finish("out_of_budget", None, [], statistics, budget)
            _, _, node, plan, trace = heappop(heap)
            if node.literals in closed:
                continue
            closed.add(node.literals)
            options = [(a,) for a, _ in cache.expand(node)] if self.include_primitives else []
            for sequence in options + moves:
                if cache.exhausted():
                    break
                steps = cache.replay(node, sequence)
                if len(steps) <= 1:
                    continue
                end = steps[-1]
                if end.literals in closed:
                    continue
                actions = list(sequence[:len(steps) - 1])
                if env.is_goal(end):
                    return finish("solved", plan + actions, trace + steps[1:], statistics,
                                  budget)
                if env.is_terminal(end):
                    statistics.pruned_terminal += 1
                    continue
                heappush(heap, ((self.progress(end) if self.progress else 0), next(tiebreak),
                                end, plan + actions, trace + steps[1:]))
        return finish("exhausted", None, [], statistics, budget)
