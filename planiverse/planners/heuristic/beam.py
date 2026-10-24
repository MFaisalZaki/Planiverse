"""Two depth-first searches that bound how far they stray from the heuristic.

Beam search keeps only the `beam` best states of each depth (by `progress`), which bounds
memory by `beam` times the depth and makes it incomplete: a solution whose ancestors were
not among the best at their depth is never found. The two searches here bound the
deviations instead and widen the bound when it fails.

* **BULB** (Furcy and Koenig, *Limited Discrepancy Beam Search*, IJCAI 2005): beam search
  that backtracks over *discrepancies*, where a discrepancy is expanding a slice of the
  sorted successors other than the best one. BULB probes with 0 discrepancies allowed,
  then 1, then 2, so it is complete given enough depth and time, and it backtracks
  non-chronologically, unlike beam-stack search, which is what made it solve more
  instances. Here each probe is a depth-first search over slices of size `beam`, with
  `max_depth` bounding it.
* **Limited discrepancy search** (Harvey and Ginsberg, IJCAI 1995): depth-first, children
  ordered by the heuristic, at most `d` deviations from the best child on any path, for
  `d = 0, 1, 2, …`. The beam is one state wide.

Both re-expand states across iterations; `SuccessorCache` makes a re-expansion free in
simulator terms, so the counted expansions are distinct states.
"""
from planiverse.planners.common import SuccessorCache, finish
from planiverse.planners.width.result import Budget, SearchStatistics


class _DepthFirst:
    """What BULB and LDS share: a depth-first probe on sorted children, with a cache, a
    path-based duplicate check and a depth bound."""

    def __init__(self, progress=None, max_depth=100, max_iterations=50):
        self.progress = progress
        self.max_depth = max_depth
        self.max_iterations = max_iterations

    def __children__(self, env, node, statistics):
        """Successors sorted by the heuristic, dead ends dropped; a goal comes first."""
        items = []
        for action, successor in self.cache.expand(node):
            if env.is_goal(successor):
                return [(action, successor)], True
            if env.is_terminal(successor):
                statistics.pruned_terminal += 1
                continue
            items.append(((self.progress(successor) if self.progress else 0.0), action,
                          successor))
        items.sort(key=lambda item: item[0])
        return [(a, s) for _, a, s in items], False

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics()
        self.cache = SuccessorCache(env, statistics, budget)
        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return finish("solved", [], [state], statistics, budget)
        for iteration in range(self.max_iterations):
            statistics.widths_tried = tuple(statistics.widths_tried) + (iteration,)
            self.exhausted_iteration = True
            found = self.__probe__(env, state, iteration, statistics)
            if found is not None:
                plan, trace = found
                return finish("solved", plan, trace, statistics, budget)
            if self.cache.exhausted():
                return finish("out_of_budget", None, [], statistics, budget)
            if self.exhausted_iteration:
                return finish("exhausted", None, [], statistics, budget)
        return finish("failed", None, [], statistics, budget)


class LimitedDiscrepancySearch(_DepthFirst):
    """Depth-first with at most `d` deviations from the heuristic's best child."""

    def __probe__(self, env, state, discrepancies, statistics):
        path = {state.literals}

        def probe(node, plan, trace, left, depth):
            if depth >= self.max_depth or self.cache.exhausted():
                self.exhausted_iteration = False
                return None
            children, goal = self.__children__(env, node, statistics)
            if goal:
                action, successor = children[0]
                return plan + [action], trace + [successor]
            for index, (action, successor) in enumerate(children):
                if index > left:
                    self.exhausted_iteration = False
                    break
                if successor.literals in path:
                    continue
                path.add(successor.literals)
                found = probe(successor, plan + [action], trace + [successor],
                              left - index, depth + 1)
                path.discard(successor.literals)
                if found is not None:
                    return found
            return None

        return probe(state, [], [state], discrepancies, 0)


class BULB(_DepthFirst):
    """Beam search using limited discrepancy backtracking.

    A layer's successors are sorted and cut into slices of `beam` states. With no
    discrepancies left only the first slice is followed; with `d` left, each later slice
    is followed with `d - 1`, and the first slice with `d`.
    """

    def __init__(self, progress=None, beam=10, max_depth=100, max_iterations=50):
        if beam < 1:
            raise ValueError(f"beam must be at least 1, got {beam}")
        super().__init__(progress, max_depth, max_iterations)
        self.beam = beam

    def __probe__(self, env, state, discrepancies, statistics):
        seen = {state.literals}

        def slices(layer):
            items, goal = [], None
            for node, plan, trace in layer:
                children, is_goal = self.__children__(env, node, statistics)
                if is_goal:
                    action, successor = children[0]
                    return None, (plan + [action], trace + [successor])
                for action, successor in children:
                    if successor.literals in seen:
                        continue
                    items.append(((self.progress(successor) if self.progress else 0.0),
                                  successor, plan + [action], trace + [successor]))
            items.sort(key=lambda item: item[0])
            layers = [[(s, p, t) for _, s, p, t in items[i:i + self.beam]]
                      for i in range(0, len(items), self.beam)]
            return layers, goal

        def probe(layer, left, depth):
            if depth >= self.max_depth or self.cache.exhausted():
                self.exhausted_iteration = False
                return None
            layers, goal = slices(layer)
            if goal is not None:
                return goal
            if not layers:
                return None
            order = list(range(1, len(layers))) if left > 0 else []
            if left == 0 and len(layers) > 1:
                self.exhausted_iteration = False
            for index in order + [0]:
                allowance = left if index == 0 else left - 1
                for node, _, _ in layers[index]:
                    seen.add(node.literals)
                found = probe(layers[index], allowance, depth + 1)
                for node, _, _ in layers[index]:
                    seen.discard(node.literals)
                if found is not None:
                    return found
            return None

        return probe([(state, [], [state])], discrepancies, 0)
