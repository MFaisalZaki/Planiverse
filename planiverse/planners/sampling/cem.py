"""The cross-entropy method over action sequences, and random shooting.

Rubinstein, *The Cross-Entropy Method for Combinatorial and Continuous Optimization*,
Methodology and Computing in Applied Probability 1999; as a planner, Pinneri et al.,
*Sample-Efficient Cross-Entropy Method for Real-Time Planning*, CoRL 2020 (iCEM). The
distribution is one categorical per position of a `horizon`-long sequence. Each iteration
draws `samples` sequences, scores them by where they end, keeps the `elite` best and refits
each position's categorical to their frequencies, smoothed by `alpha` toward the previous
one. The first action of the best elite is executed, and iCEM's two cheap improvements
carry over: the distribution is **shifted** one step for the next decision, and a fraction
of the elites is kept and re-scored rather than discarded.

`RandomShooting` is the one-iteration case with no refit: draw, score, take the best. It is
the standard model-predictive-control baseline and, on discrete action sets, a surprisingly
strong one.
"""
import random

from planiverse.planners.common import SuccessorCache, action_vocabulary, finish
from planiverse.planners.sampling.sequences import commit, evaluate_sequence
from planiverse.planners.width.result import Budget, SearchStatistics


class CrossEntropyPlanner:
    """CEM in receding horizon.

    ```python
    from planiverse.planners.sampling import CrossEntropyPlanner

    CrossEntropyPlanner(progress=boxes, horizon=10, samples=32, elite=8, iterations=3,
                        seed=0).solve(env, budget)
    ```
    """

    def __init__(self, progress=None, horizon=10, samples=32, elite=8, iterations=3,
                 alpha=0.5, keep_elites=0.3, best_along=False, max_steps=200, seed=None):
        if not 1 <= elite <= samples:
            raise ValueError("elite must be between 1 and samples")
        self.progress = progress
        self.horizon = horizon
        self.samples = samples
        self.elite = elite
        self.iterations = iterations
        self.alpha = alpha
        self.keep_elites = keep_elites
        self.best_along = best_along
        self.max_steps = max_steps
        self.random = random.Random(seed)

    def __draw__(self, distribution, vocabulary):
        return [self.random.choices(vocabulary, weights=[table.get(a, 1e-9) for a in vocabulary])[0]
                for table in distribution]

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics()
        cache = SuccessorCache(env, statistics, budget)
        if state is None:
            state, _ = env.reset()
        plan, trace = [], [state]
        distribution, carried = None, []
        for _ in range(self.max_steps):
            if env.is_goal(state):
                return finish("solved", plan, trace, statistics, budget)
            if env.is_terminal(state) or not cache.expand(state):
                return finish("dead_end", None, trace, statistics, budget)
            if cache.exhausted():
                return finish("out_of_budget", None, [], statistics, budget)
            vocabulary = action_vocabulary(env, state, cache)
            uniform = {a: 1.0 / len(vocabulary) for a in vocabulary}
            if distribution is None:
                distribution = [dict(uniform) for _ in range(self.horizon)]
            best = None
            for _ in range(self.iterations):
                candidates = carried + [self.__draw__(distribution, vocabulary)
                                        for _ in range(self.samples - len(carried))]
                scored = []
                for sequence in candidates:
                    result = evaluate_sequence(env, cache, state, sequence, self.progress,
                                               self.best_along)
                    if result.goal:
                        return finish("solved", plan + result.actions,
                                      trace + result.trace[1:], statistics, budget)
                    scored.append((result.score, sequence, result.actions))
                    if cache.exhausted():
                        break
                scored.sort(key=lambda item: -item[0])
                elites = [sequence for _, sequence, _ in scored[:self.elite]]
                if elites and (best is None or scored[0][0] >= best[0]):
                    best = scored[0]
                statistics.episodes += 1
                distribution = self.__refit__(distribution, elites, vocabulary, uniform)
                carried = elites[:int(self.keep_elites * self.elite)]
                if cache.exhausted():
                    break
            if best is None:
                return finish("out_of_budget", None, [], statistics, budget)
            action, successor = commit(env, cache, state, best[2], self.random)
            plan.append(action)
            trace.append(successor)
            state = successor
            distribution = distribution[1:] + [dict(uniform)]
            carried = [sequence[1:] + [self.random.choice(vocabulary)] for sequence in carried]
        return finish("solved" if env.is_goal(state) else "step_limit",
                      plan if env.is_goal(state) else None, trace, statistics, budget)

    def __refit__(self, distribution, elites, vocabulary, uniform):
        if not elites or self.alpha <= 0:
            return distribution
        refitted = []
        for position, table in enumerate(distribution):
            counts = {a: 0.0 for a in vocabulary}
            n = 0
            for sequence in elites:
                if position < len(sequence) and sequence[position] in counts:
                    counts[sequence[position]] += 1.0
                    n += 1
            if n == 0:
                refitted.append(table)
                continue
            refitted.append({a: self.alpha * counts[a] / n
                             + (1 - self.alpha) * table.get(a, uniform[a]) for a in vocabulary})
        return refitted


class RandomShooting(CrossEntropyPlanner):
    """Draw `samples` random sequences, execute the first action of the best."""

    def __init__(self, progress=None, horizon=10, samples=32, best_along=False,
                 max_steps=200, seed=None):
        super().__init__(progress, horizon, samples, elite=1, iterations=1, alpha=0.0,
                         keep_elites=0.0, best_along=best_along, max_steps=max_steps,
                         seed=seed)
