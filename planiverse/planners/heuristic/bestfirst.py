"""Best-first search on a black-box heuristic, and four ways of not trusting it.

The heuristic is `progress(state)`, lower is better, the same callback the width planners
take. It is a search guide and nothing more: not admissible, often flat, and sometimes
wrong, and the satisficing-planning literature is largely about what to do then.

* `BestFirstSearch`: greedy best-first on `h`, or weighted A* on `g + weight * h` when a
  `weight` is given (Pohl 1970). Complete either way; A*'s optimality needs `weight=1`
  and an admissible `h`, which `progress` is not.
* `RestartingWeightedAStar` (Richter, Thayer & Ruml, *The Joy of Forgetting: Faster Anytime
  Search via Restarting*, ICAPS 2010): anytime. Solve with a large weight, keep the plan,
  restart with a smaller one, and keep the best plan found when the budget ends. Expansions
  are cached, so a restart re-reads what the previous search generated instead of asking
  the simulator again.
* `EpsilonGreedySearch` (Valenzano, Schaeffer, Sturtevant & Xie, *A Comparison of
  Knowledge-Based GBFS Enhancements and Knowledge-Free Exploration*, ICAPS 2014): with
  probability ε expand a uniformly random open node instead of the best.
* `TypeBasedSearch` (Xie, Müller, Holte & Imai, *Type-Based Exploration with Multiple
  Search Queues for Satisficing Planning*, AAAI 2014): a second queue buckets open nodes by
  type `(h, g)`, picks a bucket uniformly and a node in it uniformly, and the two queues
  alternate.
* `DiverseBestFirst` (Imai & Kishimoto, *A Novel Technique for Avoiding Plateaus of Greedy
  Best-First Search in Satisficing Planning*, AAAI 2011): pick the node to continue from
  by sampling its heuristic value, better values more likely, then run a bounded local
  greedy search from it. The sampling rule here weights a heuristic value `h` by
  `temperature ** (h - h_min)`, which is the paper's shape; its exact distribution over
  `g` as well as `h` could not be read from this machine.
* `LocalExplorationSearch` (Xie, Müller & Holte, *Adding Local Exploration to Greedy
  Best-First Search in Satisficing Planning*, AAAI 2014): when the global best has not
  improved for `patience` expansions, run a local greedy search (`"greedy"`) or a batch of
  random walks (`"walks"`) from the best node and add what they reach.
"""
import random
from heapq import heappop, heappush
from itertools import count

from planiverse.planners.common import SuccessorCache, finish
from planiverse.planners.width.result import Budget, SearchStatistics


def action_cost(action):
    cost = getattr(action, "cost", None)
    return float(cost()) if callable(cost) else 1.0


class _Entry:
    __slots__ = ("state", "plan", "trace", "g", "h", "removed")

    def __init__(self, state, plan, trace, g, h):
        self.state, self.plan, self.trace, self.g, self.h = state, plan, trace, g, h
        self.removed = False


class BestFirstSearch:
    """Greedy best-first on `progress`, or weighted A* with a `weight`.

    ```python
    from planiverse.planners.heuristic import BestFirstSearch

    BestFirstSearch(progress=boxes).solve(env, budget)               # greedy
    BestFirstSearch(progress=boxes, weight=2.0).solve(env, budget)   # WA*, f = g + 2h
    ```
    """

    def __init__(self, progress=None, weight=None):
        self.progress = progress
        self.weight = weight
        # Deterministic on its own; the exploring subclasses take a `seed` and reseed this.
        self.random = random.Random(0)

    # ------------------------------------------------------------- the open list
    # A heap plus a list of the same entries: the heap gives the best, the list gives a
    # uniformly random one, and a `removed` flag lets each ignore what the other took.

    def __h__(self, state):
        return self.progress(state) if self.progress else 0.0

    def __key__(self, entry):
        if self.weight is None:
            return (entry.h, entry.g)
        return (entry.g + self.weight * entry.h, entry.h)

    def __reset__(self):
        self.heap, self.pool, self.tiebreak = [], [], count()

    def __push__(self, entry):
        heappush(self.heap, (self.__key__(entry), next(self.tiebreak), entry))
        self.pool.append(entry)

    def __pop_best__(self):
        while self.heap:
            _, _, entry = heappop(self.heap)
            if not entry.removed:
                entry.removed = True
                return entry
        return None

    def __pop_random__(self):
        while self.pool:
            index = self.random.randrange(len(self.pool))
            entry = self.pool[index]
            self.pool[index] = self.pool[-1]
            self.pool.pop()
            if not entry.removed:
                entry.removed = True
                return entry
        return None

    def __select__(self, statistics):
        """Which open entry to expand next. The subclasses differ here."""
        return self.__pop_best__()

    def __open__(self):
        return bool(self.heap)

    # ------------------------------------------------------------- the search

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics()
        self.cache = SuccessorCache(env, statistics, budget)
        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return finish("solved", [], [state], statistics, budget)
        self.__reset__()
        self.best_g = {state.literals: 0.0}
        self.__push__(_Entry(state, [], [state], 0.0, self.__h__(state)))
        self.__started__(state, statistics)
        while self.__open__():
            if budget.exhausted(statistics.expansions):
                return self.__finished__("out_of_budget", statistics, budget)
            entry = self.__select__(statistics)
            if entry is None:
                break
            if self.best_g.get(entry.state.literals, float("inf")) < entry.g:
                statistics.pruned_duplicate += 1
                continue
            if env.is_goal(entry.state):        # a goal a local exploration pushed
                return finish("solved", entry.plan, entry.trace, statistics, budget)
            found = self.__expand__(env, entry, statistics)
            if found is not None:
                return finish("solved", found.plan, found.trace, statistics, budget)
            self.__expanded__(entry, statistics)
        return self.__finished__("exhausted", statistics, budget)

    def __expand__(self, env, entry, statistics, push=None):
        """Generate `entry`'s children into the open list; return a goal child if any."""
        push = push or self.__push__
        for action, successor in self.cache.expand(entry.state):
            g = entry.g + action_cost(action)
            if self.best_g.get(successor.literals, float("inf")) <= g:
                statistics.pruned_duplicate += 1
                continue
            child = _Entry(successor, entry.plan + [action], entry.trace + [successor], g,
                           self.__h__(successor))
            if env.is_goal(successor):
                return child
            if env.is_terminal(successor):
                statistics.pruned_terminal += 1
                continue
            self.best_g[successor.literals] = g
            push(child)
        return None

    # ------------------------------------------------------------- hooks

    def __started__(self, state, statistics):
        pass

    def __expanded__(self, entry, statistics):
        pass

    def __finished__(self, status, statistics, budget):
        return finish(status, None, [], statistics, budget)


class RestartingWeightedAStar(BestFirstSearch):
    """Anytime weighted A* that restarts with a smaller weight after each solution.

    ```python
    RestartingWeightedAStar(progress=boxes, weights=(5, 3, 2, 1.5, 1)).solve(env, budget)
    ```

    Returns the best plan found; `statistics.widths_tried` records the weights that ran,
    and `planner.incumbents` every plan length found in order.
    """

    def __init__(self, progress=None, weights=(5.0, 3.0, 2.0, 1.5, 1.0)):
        super().__init__(progress, weights[0])
        self.weights = tuple(weights)
        self.incumbents = []

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics()
        cache = SuccessorCache(env, statistics, budget)
        best = None
        for weight in self.weights:
            search = BestFirstSearch(self.progress, weight)
            search.cache = cache
            result = self.__run__(search, env, budget, state, statistics)
            statistics.widths_tried = tuple(statistics.widths_tried) + (weight,)
            if result.solved:
                self.incumbents.append(len(result.plan))
                if best is None or result.cost < best.cost:
                    best = result
            if result.status == "out_of_budget":
                break
        if best is not None:
            return finish("solved", best.plan, best.states, statistics, budget)
        return finish("out_of_budget" if budget.exhausted(statistics.expansions) else
                      "exhausted", None, [], statistics, budget)

    @staticmethod
    def __run__(search, env, budget, state, statistics):
        """One weighted A* sharing the cache, so a restart re-reads rather than re-asks."""
        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return finish("solved", [], [state], statistics, budget)
        search.__reset__()
        search.best_g = {state.literals: 0.0}
        search.__push__(_Entry(state, [], [state], 0.0, search.__h__(state)))
        while search.__open__():
            if budget.exhausted(statistics.expansions):
                return finish("out_of_budget", None, [], statistics, budget)
            entry = search.__pop_best__()
            if entry is None:
                break
            if search.best_g.get(entry.state.literals, float("inf")) < entry.g:
                continue
            found = search.__expand__(env, entry, statistics)
            if found is not None:
                return finish("solved", found.plan, found.trace, statistics, budget)
        return finish("exhausted", None, [], statistics, budget)


class EpsilonGreedySearch(BestFirstSearch):
    """Greedy best-first that expands a random open node with probability `epsilon`."""

    def __init__(self, progress=None, epsilon=0.2, weight=None, seed=None):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        super().__init__(progress, weight)
        self.random = random.Random(seed)
        self.epsilon = epsilon

    def __select__(self, statistics):
        if self.random.random() < self.epsilon:
            return self.__pop_random__()
        return self.__pop_best__()


class TypeBasedSearch(BestFirstSearch):
    """Greedy best-first alternating with a type-based queue over `(h, g)` buckets."""

    def __init__(self, progress=None, weight=None, seed=None):
        super().__init__(progress, weight)
        self.random = random.Random(seed)

    def __reset__(self):
        super().__reset__()
        self.types = {}
        self.turn = 0

    def __push__(self, entry):
        super().__push__(entry)
        self.types.setdefault((entry.h, entry.g), []).append(entry)

    def __pop_typed__(self):
        while self.types:
            key = self.random.choice(list(self.types))
            bucket = self.types[key]
            while bucket:
                index = self.random.randrange(len(bucket))
                entry = bucket[index]
                bucket[index] = bucket[-1]
                bucket.pop()
                if not entry.removed:
                    if not bucket:
                        del self.types[key]
                    entry.removed = True
                    return entry
            del self.types[key]
        return None

    def __select__(self, statistics):
        self.turn += 1
        if self.turn % 2 == 0:
            entry = self.__pop_typed__()
            if entry is not None:
                return entry
        return self.__pop_best__()


class DiverseBestFirst(BestFirstSearch):
    """Sample a node by its heuristic value, then search greedily from it for a while."""

    def __init__(self, progress=None, temperature=0.5, local_expansions=20, seed=None):
        if not 0.0 < temperature <= 1.0:
            raise ValueError(f"temperature must be in (0, 1], got {temperature}")
        super().__init__(progress, None)
        self.random = random.Random(seed)
        self.temperature = temperature
        self.local_expansions = local_expansions

    def __reset__(self):
        super().__reset__()
        self.by_h = {}

    def __push__(self, entry):
        super().__push__(entry)
        self.by_h.setdefault(entry.h, []).append(entry)

    def __pop_sampled__(self):
        while self.by_h:
            values = sorted(self.by_h)
            weights = [self.temperature ** (h - values[0]) for h in values]
            h = self.random.choices(values, weights=weights)[0]
            bucket = self.by_h[h]
            while bucket:
                entry = bucket.pop(self.random.randrange(len(bucket)))
                if not entry.removed:
                    if not bucket:
                        del self.by_h[h]
                    entry.removed = True
                    return entry
            del self.by_h[h]
        return None

    def solve(self, env, budget=None, state=None):
        # A sampled node starts a local greedy search whose expansions go on the global
        # open list; the sampling repeats when the local search runs out of its allowance.
        self.local_left = 0
        return super().solve(env, budget, state)

    def __select__(self, statistics):
        if self.local_left > 0:
            self.local_left -= 1
            return self.__pop_best__()
        self.local_left = self.local_expansions
        return self.__pop_sampled__()


class LocalExplorationSearch(BestFirstSearch):
    """Greedy best-first with local exploration when the best heuristic value stalls."""

    def __init__(self, progress=None, patience=50, strategy="greedy", local_expansions=50,
                 walks=10, walk_length=10, seed=None):
        if strategy not in ("greedy", "walks"):
            raise ValueError(f"strategy must be 'greedy' or 'walks', got {strategy!r}")
        super().__init__(progress, None)
        self.random = random.Random(seed)
        self.patience = patience
        self.strategy = strategy
        self.local_expansions = local_expansions
        self.walks = walks
        self.walk_length = walk_length

    def __started__(self, state, statistics):
        self.best_h = self.__h__(state)
        self.stalled = 0

    def solve(self, env, budget=None, state=None):
        self.env = env
        return super().solve(env, budget, state)

    def __expanded__(self, entry, statistics):
        if entry.h < self.best_h:
            self.best_h, self.stalled = entry.h, 0
        else:
            self.stalled += 1
        if self.stalled >= self.patience:
            self.stalled = 0
            self.__explore__(entry, statistics)

    def __explore__(self, start, statistics):
        """Local search from `start`; whatever it reaches joins the global open list."""
        env = self.env
        if self.strategy == "greedy":
            local = []
            heappush(local, ((start.h, 0), next(self.tiebreak), start))
            for _ in range(self.local_expansions):
                if not local or self.cache.exhausted():
                    return
                _, _, entry = heappop(local)
                found = self.__expand__(
                    env, entry, statistics,
                    push=lambda child: (self.__push__(child),
                                        heappush(local, ((child.h, child.g),
                                                         next(self.tiebreak), child))))
                if found is not None:
                    self.__push__(found)
                    return
                if entry.h < self.best_h:
                    self.best_h = entry.h
                    return
            return
        for _ in range(self.walks):
            entry = start
            for _ in range(self.walk_length):
                if self.cache.exhausted():
                    return
                children = [(a, s) for a, s in self.cache.expand(entry.state)
                            if not env.is_terminal(s) or env.is_goal(s)]
                if not children:
                    break
                action, successor = self.random.choice(children)
                g = entry.g + action_cost(action)
                entry = _Entry(successor, entry.plan + [action], entry.trace + [successor],
                               g, self.__h__(successor))
                if env.is_goal(successor):
                    self.__push__(entry)
                    return
            if self.best_g.get(entry.state.literals, float("inf")) > entry.g:
                self.best_g[entry.state.literals] = entry.g
                self.__push__(entry)
