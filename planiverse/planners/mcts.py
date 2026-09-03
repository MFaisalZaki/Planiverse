"""Monte Carlo Tree Search (UCT) for deterministic, goal-directed simulators.

UCT, from Kocsis and Szepesvári's *Bandit Based Monte-Carlo Planning* (ECML 2006), treats the
choice of child at each node as a multi-armed bandit and picks by the UCB1 rule, so the tree
grows deepest where the returns look best while still sampling everything occasionally. Browne
et al.'s *A Survey of Monte Carlo Tree Search Methods* (IEEE TCIAIG 4(1), 2012) is the
standard reference for the four-phase loop this follows: **select, expand, simulate,
backpropagate**.

MCTS was built for adversarial games with a natural terminal reward. Planiverse gives it
neither, and three adaptations follow:

**There is no score, only `is_goal`.** A rollout that fails tells you nothing about how close
it got, so the default reward is blunt: 1 for reaching a goal, 0 otherwise, less a
small length penalty so shorter solutions win ties. A `reward` callback lets a caller
supply something denser when the environment offers one (blocks cleared, contamination
removed). Without one, MCTS needs a rollout to *stumble* into a goal before it learns
anything at all, which on a sparse problem it may never do.

**The transition is deterministic.** The averaging that makes UCT work in stochastic games is
doing nothing here, so this keeps the *best* value seen through a node as well as the mean,
and can select on either. `backup="max"` is usually right for planning and is the default;
`backup="mean"` is the classical rule.

**Dead ends are real and absorbing.** A rollout that walks into one has wasted its budget, so
terminal states are marked in the tree and never selected again, which stops UCT from
re-exploring a branch it has already proved is over.

The plan returned is the best goal-reaching path *found*, not the tree's principal variation:
in a single-agent problem with no opponent, a solution seen once during a rollout is a
solution, and there is no reason to throw it away because the averages have not caught up.
"""
import math
import random

from planiverse.planners.width.result import Budget, SearchResult, SearchStatistics

#: The UCB1 exploration constant. sqrt(2) is the textbook value for rewards in [0, 1].
DEFAULT_EXPLORATION = math.sqrt(2)


class _Node:
    """One state in the tree, with the bandit statistics for choosing among its children."""

    __slots__ = ("state", "parent", "action", "children", "untried", "visits", "total",
                 "best", "terminal", "depth")

    def __init__(self, state, parent=None, action=None, depth=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = depth
        self.children = []
        self.untried = None          # filled on first expansion
        self.visits = 0
        self.total = 0.0
        self.best = 0.0
        self.terminal = False

    @property
    def mean(self):
        return self.total / self.visits if self.visits else 0.0

    def fully_expanded(self):
        return self.untried is not None and not self.untried


class MCTSPlanner:
    """UCT over a simulator, returning the best goal-reaching path it found.

    ```python
    from planiverse.planners.mcts import MCTSPlanner

    env.set_index(0)
    result = MCTSPlanner(
        iterations=2000,
        reward=lambda s: 1.0 - len(s.blocks) / 6,   # denser than goal/no-goal
        seed=0,
    ).solve(env, Budget(max_seconds=60))
    ```
    """

    def __init__(self, iterations=1000, exploration=DEFAULT_EXPLORATION, rollout_depth=30,
                 reward=None, backup="max", seed=None, length_penalty=0.001):
        """
        - `reward(state) -> float in [0, 1]`: how good a non-goal state is. Without it the
          signal is goal-or-nothing, which is honest but sparse.
        - `backup`: `"max"` propagates the best value seen through a node, `"mean"` the
          classical average. Determinism makes max the better choice for planning.
        - `length_penalty`: subtracted per action, so among solutions the shorter wins.
        """
        if backup not in ("max", "mean"):
            raise ValueError(f"backup must be 'max' or 'mean', got {backup!r}")
        self.iterations = iterations
        self.exploration = exploration
        self.rollout_depth = rollout_depth
        self.reward = reward
        self.backup = backup
        self.length_penalty = length_penalty
        self.random = random.Random(seed)

    # ------------------------------------------------------------------ four phases

    def __select__(self, node):
        """Descend by UCB1 until a node that is not fully expanded."""
        while node.fully_expanded() and node.children and not node.terminal:
            node = max(node.children, key=lambda child: self.__ucb1__(node, child))
        return node

    def __ucb1__(self, parent, child):
        if child.terminal:
            # Proved over. Never worth selecting again, however good it once looked.
            return float("-inf")
        if not child.visits:
            return float("inf")
        value = child.best if self.backup == "max" else child.mean
        return value + self.exploration * math.sqrt(math.log(parent.visits) / child.visits)

    def __expand__(self, env, node, statistics):
        """Add one unexplored child, or report the node has none."""
        if node.untried is None:
            successors = env.successors(node.state)
            statistics.expansions += 1
            statistics.generated += len(successors)
            node.untried = list(successors)
            if not node.untried:
                node.terminal = True
        if not node.untried:
            return None
        action, state = node.untried.pop(self.random.randrange(len(node.untried)))
        child = _Node(state, parent=node, action=action, depth=node.depth + 1)
        if env.is_terminal(state) and not env.is_goal(state):
            child.terminal = True
            statistics.pruned_terminal += 1
        node.children.append(child)
        return child

    def __simulate__(self, env, node, statistics, budget):
        """Random rollout from `node`. Returns `(value, plan_to_goal or None)`."""
        if env.is_goal(node.state):
            return 1.0 - self.length_penalty * node.depth, []

        state = node.state
        path = []
        # The best reward seen *anywhere along* the rollout, not just where it stopped.
        # Random rollouts in a domain with dead ends nearly always end in one, and scoring
        # only the final state throws away everything the rollout learned on the way: the
        # tree then sees 0 for every branch and UCT has no gradient to climb. Keeping the
        # high-water mark is what makes a dense `reward` callback actually reach the tree.
        best = self.reward(state) if self.reward else 0.0

        for _ in range(self.rollout_depth):
            if budget.exhausted(statistics.expansions):
                break
            if env.is_terminal(state):
                break
            successors = env.successors(state)
            statistics.expansions += 1
            statistics.generated += len(successors)
            if not successors:
                break
            action, state = self.random.choice(successors)
            path.append(action)
            if env.is_goal(state):
                return 1.0 - self.length_penalty * (node.depth + len(path)), path
            if self.reward:
                best = max(best, self.reward(state))

        return best, None

    def __backup__(self, node, value):
        while node is not None:
            node.visits += 1
            node.total += value
            node.best = max(node.best, value)
            node = node.parent

    # ---------------------------------------------------------------------- driver

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics()

        if state is None:
            state, _ = env.reset()
        root = _Node(state)
        if env.is_goal(state):
            return self.__result__("solved", [], [state], statistics, budget)

        best_plan = None
        for _ in range(self.iterations):
            if budget.exhausted(statistics.expansions):
                break

            leaf = self.__select__(root)
            if not leaf.terminal:
                child = self.__expand__(env, leaf, statistics)
                if child is not None:
                    leaf = child

            value, tail = self.__simulate__(env, leaf, statistics, budget)
            if tail is not None:
                plan = self.__path_to__(leaf) + tail
                if best_plan is None or len(plan) < len(best_plan):
                    best_plan = plan
            self.__backup__(leaf, value)

        if best_plan is None:
            status = "out_of_budget" if budget.exhausted(statistics.expansions) else "failed"
            return self.__result__(status, None, [], statistics, budget)

        trace = env.simulate(best_plan)
        return self.__result__("solved", best_plan, trace, statistics, budget)

    def __path_to__(self, node):
        actions = []
        while node.parent is not None:
            actions.append(node.action)
            node = node.parent
        return actions[::-1]

    def __result__(self, status, plan, trace, statistics, budget):
        statistics.elapsed = budget.elapsed()
        return SearchResult(plan=plan, states=trace, status=status, width=None,
                            statistics=statistics)
