"""Rolling Horizon Evolutionary Algorithm.

Perez, Samothrakis, Lucas and Rohlfshagen, *Rolling Horizon Evolution versus Tree Search
for Navigation in Single-Player Real-Time Games*, GECCO 2013; Gaina, Lucas and
Perez-Liebana, *Rolling Horizon Evolution Enhancements in General Video Game Playing*, CIG
2017; Gaina, Devlin, Lucas and Perez-Liebana, *Rolling Horizon Evolutionary Algorithms for
General Video Game Playing*, IEEE Transactions on Games 2021. An individual is a sequence
of `length` actions; its fitness is the value of the state the forward model reaches at
its end; a population of `population` of them is evolved for as long as the per-decision
budget allows; the first action of the fittest is executed, and the population is
**shifted** one step (its first genes dropped, a random gene appended) rather than thrown
away, which is the enhancement that made the biggest difference in GVGAI. The vanilla
configuration there is population 10, length 10, uniform crossover, one-gene mutation and
one elite.

Actions come from `get_actions()` when the environment has one, else from the applicable
actions of the current state, a vocabulary that widens with every expansion seen. A gene
that is not applicable where it lands ends the evaluation early.
"""
import random

from planiverse.planners.common import SuccessorCache, action_vocabulary, finish
from planiverse.planners.sampling.sequences import Evaluation, commit, evaluate_sequence
from planiverse.planners.width.result import Budget, SearchStatistics


class RollingHorizonEvolution:
    """RHEA with a shift buffer.

    ```python
    from planiverse.planners.sampling import RollingHorizonEvolution

    RollingHorizonEvolution(progress=boxes, population=10, length=10,
                            expansions_per_step=200, seed=0).solve(env, budget)
    ```
    """

    def __init__(self, progress=None, population=10, length=10, expansions_per_step=200,
                 generations=20, mutation=None, elites=1, shift=True, best_along=False,
                 max_steps=200, seed=None):
        """A decision ends after `expansions_per_step` new expansions or `generations`
        generations, whichever comes first: the cache makes a generation that re-treads
        known states free in expansions, and without the second cap it would never end."""
        if population < 2 or length < 1:
            raise ValueError("population must be at least 2 and length at least 1")
        self.progress = progress
        self.population = population
        self.length = length
        self.expansions_per_step = expansions_per_step
        self.generations = generations
        self.mutation = mutation if mutation is not None else 1.0 / length
        self.elites = elites
        self.shift = shift
        self.best_along = best_along
        self.max_steps = max_steps
        self.random = random.Random(seed)

    def __random__(self, vocabulary):
        return [self.random.choice(vocabulary) for _ in range(self.length)]

    def __offspring__(self, a, b, vocabulary):
        child = [self.random.choice(pair) for pair in zip(a, b)]
        return [self.random.choice(vocabulary) if self.random.random() < self.mutation
                else gene for gene in child]

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics()
        cache = SuccessorCache(env, statistics, budget)
        if state is None:
            state, _ = env.reset()
        plan, trace = [], [state]
        population = None
        for _ in range(self.max_steps):
            if env.is_goal(state):
                return finish("solved", plan, trace, statistics, budget)
            if env.is_terminal(state) or not cache.expand(state):
                return finish("dead_end", None, trace, statistics, budget)
            if cache.exhausted():
                return finish("out_of_budget", None, [], statistics, budget)
            vocabulary = action_vocabulary(env, state, cache)
            if population is None:
                population = [self.__random__(vocabulary) for _ in range(self.population)]
            scored = self.__evolve__(env, cache, state, population, vocabulary, statistics)
            if scored is None:
                return finish("out_of_budget", None, [], statistics, budget)
            if isinstance(scored, Evaluation):     # a goal was reached mid-evaluation
                return finish("solved", plan + scored.actions, trace + scored.trace[1:],
                              statistics, budget)
            population, applied = scored
            action, successor = commit(env, cache, state, applied, self.random)
            plan.append(action)
            trace.append(successor)
            state = successor
            if self.shift:
                population = [ind[1:] + [self.random.choice(vocabulary)] for ind in population]
            else:
                population = None
        return finish("solved" if env.is_goal(state) else "step_limit",
                      plan if env.is_goal(state) else None, trace, statistics, budget)

    def __evolve__(self, env, cache, state, population, vocabulary, statistics):
        """Generations until the per-step budget is spent. Returns `(population, applied)`,
        the population sorted best first with the actions its best actually applied, the
        `Evaluation` that reached a goal if one did, or None when the overall budget ran
        out before anything was scored."""
        spent_at = statistics.expansions
        fitness = {}
        generation = 0

        def score(individual):
            key = tuple(individual)
            if key not in fitness:
                result = evaluate_sequence(env, cache, state, individual, self.progress,
                                           self.best_along)
                fitness[key] = result
            return fitness[key]

        while True:
            ranked = []
            for individual in population:
                result = score(individual)
                if result.goal:
                    return result
                ranked.append((result.score, individual))
            ranked.sort(key=lambda item: -item[0])
            population = [ind for _, ind in ranked]
            applied = score(population[0]).actions
            statistics.episodes += 1
            generation += 1
            if cache.exhausted():
                return (population, applied) if fitness else None
            if statistics.expansions - spent_at >= self.expansions_per_step \
                    or generation >= self.generations:
                return population, applied
            parents = population
            offspring = population[:self.elites]
            while len(offspring) < self.population:
                a = min(self.random.sample(parents, 2), key=lambda ind: -score(ind).score)
                b = min(self.random.sample(parents, 2), key=lambda ind: -score(ind).score)
                offspring.append(self.__offspring__(a, b, vocabulary))
            population = offspring
