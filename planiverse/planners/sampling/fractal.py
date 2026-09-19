"""Fractal Monte Carlo: a swarm of walkers that clone toward reward and diversity.

Hernández Cerezo and Duran Ballester, *Fractal AI: A Fragile Theory of Intelligence*,
arXiv 1803.05049 (2018), and *Solving Atari Games Using Fractals and Entropy*, arXiv
1807.01081 (2018). A swarm of `walkers` starts at the root, each having taken one root
action. Every iteration each walker takes a random step; then each walker `i` is paired
with a random walker `j` and computes a **virtual reward** `VR_i = R_i^α · D_i^β`, where
`R_i` is its accumulated reward and `D_i` its distance in state space from `j`, both
normalised across the swarm; and `i` **clones** onto `j` (copies its state and history)
with probability `(VR_j - VR_i) / VR_i`, clipped to `[0, 1]`, a dead walker cloning with
certainty. Reward pulls the swarm toward good states and distance keeps it spread out, so
the swarm covers reachable rewarding futures without a tree, and the root action with the
most walkers is executed. The authors report under a thousand samples per action on Atari
where MCTS uses millions.

The papers' normalisation could not be read from this machine, so the one used is stated:
each quantity is standardised across the swarm, then mapped by `1 + log(1 + x)` for
`x >= 0` and `exp(x)` for `x < 0`, which is the "relativisation" of the authors' public
code as remembered here, and it keeps every virtual reward positive. The distance between
two states is the size of the symmetric difference of their `literals`. The swarm shares
`FSXPlanner`'s shape (walkers, a horizon, an action committed per decision), and it is what
FSX becomes when a reward is added and the walkers are allowed to copy each other.
"""
import math
import random
import statistics as stats

from planiverse.planners.common import SuccessorCache, finish
from planiverse.planners.width.result import Budget, SearchStatistics


def relativise(values):
    """Standardise, then squash to positive numbers."""
    if len(values) < 2 or max(values) == min(values):
        return [1.0] * len(values)
    mean = stats.fmean(values)
    sd = stats.pstdev(values) or 1.0
    out = []
    for value in values:
        z = (value - mean) / sd
        out.append(1.0 + math.log1p(z) if z >= 0 else math.exp(z))
    return out


class _Walker:
    __slots__ = ("state", "actions", "states", "reward", "dead")

    def __init__(self, state, actions, states, reward, dead):
        self.state, self.actions, self.states = state, actions, states
        self.reward, self.dead = reward, dead

    def copy(self):
        return _Walker(self.state, list(self.actions), list(self.states), self.reward,
                       self.dead)


class FractalMonteCarlo:
    """FMC: `walkers` walkers, `iterations` perturb-and-clone rounds per decision.

    ```python
    from planiverse.planners.sampling import FractalMonteCarlo

    FractalMonteCarlo(progress=boxes, walkers=32, iterations=30, seed=0).solve(env, budget)
    ```
    """

    def __init__(self, progress=None, walkers=32, iterations=30, alpha=1.0, beta=1.0,
                 max_steps=200, seed=None):
        if walkers < 2:
            raise ValueError("walkers must be at least 2")
        self.progress = progress
        self.walkers = walkers
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.max_steps = max_steps
        self.random = random.Random(seed)

    def __reward__(self, base, state):
        return (base - self.progress(state)) if self.progress else 0.0

    def __decide__(self, env, cache, state, statistics):
        """Evolve the swarm from `state`. Returns an action, a goal `(actions, states)`,
        or None when nothing can be done."""
        children = cache.expand(state)
        if not children:
            return None
        base = self.progress(state) if self.progress else 0.0
        swarm = []
        for _ in range(self.walkers):
            action, successor = self.random.choice(children)
            if env.is_goal(successor):
                return ([action], [successor])
            swarm.append(_Walker(successor, [action], [successor],
                                 self.__reward__(base, successor), env.is_terminal(successor)))
        for _ in range(self.iterations):
            if cache.exhausted():
                break
            for walker in swarm:                     # perturb
                if walker.dead:
                    continue
                options = cache.expand(walker.state)
                if not options:
                    walker.dead = True
                    continue
                action, successor = self.random.choice(options)
                walker.state = successor
                walker.actions.append(action)
                walker.states.append(successor)
                walker.reward = self.__reward__(base, successor)
                if env.is_goal(successor):
                    return (walker.actions, walker.states)
                walker.dead = env.is_terminal(successor)
            partners = [self.random.randrange(len(swarm)) for _ in swarm]
            distances = [len(w.state.literals ^ swarm[j].state.literals)
                         for w, j in zip(swarm, partners)]
            rewards = relativise([w.reward for w in swarm])
            spread = relativise([float(d) for d in distances])
            virtual = [(r ** self.alpha) * (d ** self.beta) for r, d in zip(rewards, spread)]
            clones = []
            for index, (walker, j) in enumerate(zip(swarm, partners)):     # clone
                if walker.dead:
                    probability = 1.0 if not swarm[j].dead else 0.0
                else:
                    probability = (virtual[j] - virtual[index]) / virtual[index]
                if self.random.random() < min(1.0, max(0.0, probability)):
                    clones.append((index, j))
            for index, j in clones:
                swarm[index] = swarm[j].copy()
            statistics.rollouts += 1
        counts = {}
        for walker in swarm:
            if not walker.dead:
                counts[walker.actions[0]] = counts.get(walker.actions[0], 0) + 1
        if not counts:
            return self.random.choice(children)[0]
        best = max(counts.values())
        return self.random.choice([a for a, n in counts.items() if n == best])

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics()
        cache = SuccessorCache(env, statistics, budget)
        if state is None:
            state, _ = env.reset()
        plan, trace = [], [state]
        for _ in range(self.max_steps):
            if env.is_goal(state):
                return finish("solved", plan, trace, statistics, budget)
            if env.is_terminal(state):
                return finish("dead_end", None, trace, statistics, budget)
            if cache.exhausted():
                return finish("out_of_budget", None, [], statistics, budget)
            decision = self.__decide__(env, cache, state, statistics)
            if decision is None:
                return finish("dead_end", None, trace, statistics, budget)
            if isinstance(decision, tuple):
                actions, states = decision
                return finish("solved", plan + actions, trace + states, statistics, budget)
            successor = cache.apply(state, decision)
            plan.append(decision)
            trace.append(successor)
            state = successor
            statistics.episodes += 1
        return finish("solved" if env.is_goal(state) else "step_limit",
                      plan if env.is_goal(state) else None, trace, statistics, budget)
