"""Local search over one action sequence: simulated annealing, or iterated local search.

The plan is the object being optimised: a sequence of `length` actions, scored by where it
leads (see `sequences.py`). A neighbour changes one action, inserts one, deletes one or
swaps two. Simulated annealing (Kirkpatrick, Gelatt and Vecchi, 1983) accepts a worse
neighbour with probability `exp(delta / temperature)`, the temperature cooling by
`cooling` each step; iterated local search (Lourenço, Martin and Stützle, 2003) descends
greedily, then kicks the incumbent with `kick` random changes and descends again, keeping
the better of the two. Neither is a first-line planner here; both improve a plan another
planner returned, and `initial` accepts one.
"""
import math
import random

from planiverse.planners.common import SuccessorCache, action_vocabulary, finish
from planiverse.planners.sampling.sequences import evaluate_sequence
from planiverse.planners.width.result import Budget, SearchStatistics

METHODS = ("annealing", "iterated")


class PlanLocalSearch:
    """Optimise a sequence by annealing or iterated local search.

    ```python
    from planiverse.planners.sampling import PlanLocalSearch

    PlanLocalSearch(progress=boxes, length=20, method="annealing", seed=0).solve(env)
    ```
    """

    def __init__(self, progress=None, length=20, method="annealing", temperature=1.0,
                 cooling=0.995, kick=3, initial=None, best_along=True,
                 max_iterations=100_000, seed=None):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        self.progress = progress
        self.length = length
        self.method = method
        self.temperature = temperature
        self.cooling = cooling
        self.kick = kick
        self.initial = initial
        self.best_along = best_along
        self.max_iterations = max_iterations
        self.random = random.Random(seed)

    def __neighbour__(self, genome, vocabulary):
        genome = list(genome)
        roll = self.random.random()
        if roll < 0.4 and genome:
            genome[self.random.randrange(len(genome))] = self.random.choice(vocabulary)
        elif roll < 0.6 and len(genome) < self.length:
            genome.insert(self.random.randrange(len(genome) + 1), self.random.choice(vocabulary))
        elif roll < 0.8 and len(genome) > 1:
            del genome[self.random.randrange(len(genome))]
        elif len(genome) > 1:
            i, j = self.random.sample(range(len(genome)), 2)
            genome[i], genome[j] = genome[j], genome[i]
        return genome

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics()
        cache = SuccessorCache(env, statistics, budget)
        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return finish("solved", [], [state], statistics, budget)
        vocabulary = action_vocabulary(env, state, cache)

        def score(genome):
            statistics.rollouts += 1
            return evaluate_sequence(env, cache, state, genome, self.progress, self.best_along)

        current = list(self.initial) if self.initial else \
            [self.random.choice(vocabulary) for _ in range(self.length)]
        value = score(current)
        if value.goal:
            return finish("solved", value.actions, value.trace, statistics, budget)
        best = (value.score, current)
        temperature = self.temperature
        for _ in range(self.max_iterations):
            if cache.exhausted():
                return finish("out_of_budget", None, [], statistics, budget)
            if self.method == "annealing":
                candidate = self.__neighbour__(current, vocabulary)
                result = score(candidate)
                if result.goal:
                    return finish("solved", result.actions, result.trace, statistics, budget)
                delta = result.score - value.score
                if delta >= 0 or (math.isfinite(delta) and temperature > 0
                                  and self.random.random() < math.exp(delta / temperature)):
                    current, value = candidate, result
                temperature *= self.cooling
            else:
                improved = True
                while improved and not cache.exhausted():
                    improved = False
                    for _ in range(len(vocabulary) * 2):
                        candidate = self.__neighbour__(current, vocabulary)
                        result = score(candidate)
                        if result.goal:
                            return finish("solved", result.actions, result.trace,
                                          statistics, budget)
                        if result.score > value.score:
                            current, value, improved = candidate, result, True
                            break
                if value.score >= best[0]:
                    best = (value.score, current)
                kicked = current
                for _ in range(self.kick):
                    kicked = self.__neighbour__(kicked, vocabulary)
                current, value = kicked, score(kicked)
                if value.goal:
                    return finish("solved", value.actions, value.trace, statistics, budget)
            if value.score > best[0]:
                best = (value.score, current)
        return finish("failed", None, [], statistics, budget)
