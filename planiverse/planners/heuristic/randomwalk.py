"""Monte-Carlo random walks: Arvand's local search for deterministic planning.

Nakhost and Müller, *Monte-Carlo Exploration for Deterministic Planning*, IJCAI 2009, and
*Towards a Second Generation Random Walk Planner: An Experimental Exploration*, IJCAI 2013.
From the current state, run a batch of bounded random walks and evaluate only their
endpoints; jump to the best endpoint if it improves on the current state, and restart from
the initial state after too many batches without improvement. Random walks sample the
neighbourhood far more cheaply than a breadth-first search does and reach much deeper,
which is why Arvand scaled to instances the heuristic planners of its day could not, at
the price of long plans.

What is kept from Arvand, and what is a choice here:

* **Walk length adapts**: it starts at `walk_length` and is multiplied by `extend` each
  time a batch fails to improve, the paper's way of widening the neighbourhood when the
  local one is exhausted; a jump resets it.
* **Restarts** after `patience` failed batches, from the initial state.
* **Dead-end avoidance (MDA)**: Arvand biases action choice away from actions whose walks
  tend to end in dead ends, from statistics it gathers as it runs. Here the first action
  of each walk is drawn with weight `exp(-bias * dead_end_rate(action))`, the rate being
  how often walks starting with that action hit `is_terminal`. The paper's exact
  formulation could not be read from this machine; this is its shape. `bias=0` disables
  it. Its helpful-action bias (MHA) needs a relaxed plan and has no black-box counterpart.

A walk step is a full expansion under the `successors` contract; `SuccessorCache` keeps
states walked through twice from costing twice.
"""
import math
import random

from planiverse.planners.common import SuccessorCache, finish
from planiverse.planners.width.result import Budget, SearchStatistics


class MonteCarloRandomWalks:
    """Arvand-style random-walk planning on `progress`.

    ```python
    from planiverse.planners.heuristic import MonteCarloRandomWalks

    MonteCarloRandomWalks(progress=boxes, walks=20, walk_length=10, seed=0).solve(env, budget)
    ```
    """

    def __init__(self, progress, walks=20, walk_length=10, extend=1.5, max_walk_length=200,
                 patience=5, max_restarts=20, bias=2.0, seed=None):
        if progress is None:
            raise ValueError("MonteCarloRandomWalks needs a progress measure")
        self.progress = progress
        self.walks = walks
        self.walk_length = walk_length
        self.extend = extend
        self.max_walk_length = max_walk_length
        self.patience = patience
        self.max_restarts = max_restarts
        self.bias = bias
        self.random = random.Random(seed)

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics()
        cache = SuccessorCache(env, statistics, budget)
        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return finish("solved", [], [state], statistics, budget)
        start = state
        dead = {}               # action -> [walks starting with it, of which dead ends]

        for _ in range(self.max_restarts + 1):
            node, plan, trace = start, [], [start]
            length, failures = self.walk_length, 0
            while failures < self.patience:
                if cache.exhausted():
                    return finish("out_of_budget", None, [], statistics, budget)
                best = None
                for _ in range(self.walks):
                    walk = self.__walk__(env, cache, node, length, dead)
                    if walk is None:
                        continue
                    end, actions, states, goal = walk
                    statistics.rollouts += 1
                    if goal:
                        return finish("solved", plan + actions, trace + states, statistics,
                                      budget)
                    value = self.progress(end)
                    if best is None or value < best[0]:
                        best = (value, end, actions, states)
                if best is not None and best[0] < self.progress(node):
                    _, node, actions, states = best
                    plan, trace = plan + actions, trace + states
                    length, failures = self.walk_length, 0
                else:
                    failures += 1
                    length = min(self.max_walk_length, int(length * self.extend) + 1)
            statistics.episodes += 1
        return finish("failed", None, [], statistics, budget)

    def __walk__(self, env, cache, node, length, dead):
        """One random walk. Returns `(end, actions, states, reached_goal)` or None."""
        actions, states, first = [], [], None
        for step in range(length):
            if cache.exhausted():
                break
            children = cache.expand(node)
            if not children:
                break
            if step == 0 and self.bias > 0:
                weights = [math.exp(-self.bias * self.__rate__(dead, a)) for a, _ in children]
                action, node = self.random.choices(children, weights=weights)[0]
                first = action
            else:
                action, node = self.random.choice(children)
            actions.append(action)
            states.append(node)
            if env.is_goal(node):
                self.__record__(dead, first, False)
                return node, actions, states, True
            if env.is_terminal(node):
                self.__record__(dead, first, True)
                return None
        if not actions:
            return None
        self.__record__(dead, first, False)
        return node, actions, states, False

    @staticmethod
    def __rate__(dead, action):
        total, ended = dead.get(action, (0, 0))
        return ended / total if total else 0.0

    @staticmethod
    def __record__(dead, action, ended):
        if action is None:
            return
        total, count = dead.get(action, (0, 0))
        dead[action] = (total + 1, count + (1 if ended else 0))
