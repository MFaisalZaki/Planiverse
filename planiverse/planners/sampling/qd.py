"""MAP-Elites over action sequences, with novelty-weighted parent selection.

Mouret and Clune, *Illuminating Search Spaces by Mapping Elites*, arXiv 1504.04909
(2015); Lehman and Stanley, *Abandoning Objectives: Evolution Through the Search for
Novelty Alone*, Evolutionary Computation 2011. A genome is an action sequence; its
*descriptor* is a projection of the state it ends in; the archive keeps, per descriptor
cell, the best sequence that landed there. Each iteration picks an elite, mutates it,
replays it and files the result. Selection pressure is on the descriptor, not the score,
so the archive fills the reachable behaviour space, and the score only decides who holds
each cell. `selection="novelty"` weights parents by how rarely their cell has been chosen,
which is the novelty-search reading of the same archive; `"uniform"` is MAP-Elites as
published.

Go-Explore is the same archive with the simulator restored to the elite rather than the
sequence replayed, which is why it is the better fit when restoring is free; this one
exists for the comparison, and because its population can be evaluated in parallel.
"""
import random

from planiverse.planners.common import SuccessorCache, action_vocabulary, finish
from planiverse.planners.sampling.sequences import evaluate_sequence
from planiverse.planners.width.result import Budget, SearchStatistics


class MAPElitesPlanner:
    """MAP-Elites over sequences of up to `length` actions.

    ```python
    from planiverse.planners.sampling import MAPElitesPlanner

    MAPElitesPlanner(progress=boxes, descriptor=lambda s: (boxes(s),), seed=0).solve(env)
    ```
    """

    def __init__(self, progress=None, descriptor=None, length=20, initial=20,
                 selection="novelty", max_iterations=100_000, seed=None):
        if selection not in ("novelty", "uniform"):
            raise ValueError("selection must be 'novelty' or 'uniform'")
        self.progress = progress
        self.descriptor = descriptor or (lambda state: state.literals)
        self.length = length
        self.initial = initial
        self.selection = selection
        self.max_iterations = max_iterations
        self.random = random.Random(seed)
        self.archive = {}

    def __mutate__(self, genome, vocabulary):
        genome = list(genome)
        roll = self.random.random()
        if roll < 0.5 and genome:
            genome[self.random.randrange(len(genome))] = self.random.choice(vocabulary)
        elif roll < 0.8 and len(genome) < self.length:
            genome.insert(self.random.randrange(len(genome) + 1), self.random.choice(vocabulary))
        elif genome:
            del genome[self.random.randrange(len(genome))]
        return genome or [self.random.choice(vocabulary)]

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics()
        cache = SuccessorCache(env, statistics, budget)
        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return finish("solved", [], [state], statistics, budget)
        vocabulary = action_vocabulary(env, state, cache)
        self.archive = {}
        chosen = {}

        def file(genome):
            result = evaluate_sequence(env, cache, state, genome, self.progress)
            statistics.rollouts += 1
            if result.goal:
                return result
            key = self.descriptor(result.trace[-1])
            held = self.archive.get(key)
            if held is None or result.score > held[0] or \
                    (result.score == held[0] and len(result.actions) < len(held[1])):
                self.archive[key] = (result.score, result.actions)
            return None

        for _ in range(self.initial):
            found = file([self.random.choice(vocabulary)
                          for _ in range(self.random.randint(1, self.length))])
            if found is not None:
                return finish("solved", found.actions, found.trace, statistics, budget)
        for _ in range(self.max_iterations):
            if cache.exhausted():
                return finish("out_of_budget", None, [], statistics, budget)
            keys = list(self.archive)
            if self.selection == "novelty":
                weights = [1.0 / (1 + chosen.get(key, 0)) for key in keys]
                key = self.random.choices(keys, weights=weights)[0]
            else:
                key = self.random.choice(keys)
            chosen[key] = chosen.get(key, 0) + 1
            found = file(self.__mutate__(self.archive[key][1], vocabulary))
            if found is not None:
                return finish("solved", found.actions, found.trace, statistics, budget)
        return finish("failed", None, [], statistics, budget)
