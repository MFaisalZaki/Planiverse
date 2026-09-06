"""Rollout IW: width-based lookahead built from rollouts instead of breadth-first search.

Bandres, Bonet and Geffner, *Planning with Pixels in (Almost) Real Time* (AAAI 2018).

IW(k) is breadth-first search with a novelty filter, and breadth-first is the problem when
the simulator is expensive and the decisions are wanted now: the frontier has to be finished
level by level before anything deep is seen. Rollout IW keeps the filter and drops the
breadth-first order. It grows a tree by **rollouts** from the root, each one walking down
through the tree, expanding a node when it steps off the known part, and stopping as soon as
it reaches a node that is not novel, is terminal, or has nothing left under it. That gives
the same polynomial bound IW(k) has (a node is kept only while some tuple of its atoms is
being seen shallower than before) with an anytime search that sees depth from the first
rollout.

Two definitions differ from IW's, and both are what make the rollouts work:

* **Novelty is measured against depth.** The table remembers the shallowest depth each atom
  tuple has been seen at, not merely whether it has been seen (`DepthNoveltyTable`). Rollouts
  do not come in depth order, so "seen at all" would let the first rollout, at whatever
  depth it happened to reach an atom, pre-empt every shallower discovery after it.
* **A node is "solved" when there is nothing left to learn beneath it**: it failed the
  novelty test, it is terminal, it is at the depth cap, or every child is solved. Rollouts
  never enter solved nodes, and when the root is solved the lookahead is complete.

The paper runs this **online**: a fixed budget of simulator calls per decision, then the
action whose subtree backed up the best return is executed, the subtree beneath it is kept
and the novelty table is started afresh. Resetting the table each step is what renews
exploration the way SIW's legs do, and it is why the online form solves things a single
width-1 lookahead cannot. `RolloutIW` does the same, with `expansions_per_step` as the
budget; set it to `None` for one lookahead from the initial state that runs until it finds
a goal, solves the root or runs out of budget.

What changes against these simulators rather than Atari:

* **There is no score.** Atari's rewards drive the action choice. Here the return of a
  transition is the improvement in a `progress` measure (lower is better, the same callback
  SIW and BFWS take), and a goal is simply returned as the plan the moment any rollout finds
  one. Without `progress` the returns are flat, the online action choice is uniform, and the
  planner says so rather than pretending to steer.
* **An expansion yields every child.** The contract is `successors()`, so stepping off the
  known tree generates all of a node's children at once, and the rollout then picks one.
  Rollouts through those siblings later assess their novelty as **new** nodes, since it is
  the first time their novelty is looked at.
* **Dead ends are real.** Terminal children are marked solved at generation, a rollout never
  walks into one, and the executed action avoids one whenever there is an alternative.
"""
import random
from collections import deque

from planiverse.planners.width.novelty import DepthNoveltyTable
from planiverse.planners.width.result import Budget, SearchResult, SearchStatistics


class RolloutNode:
    """One state in the lookahead tree."""

    __slots__ = ("state", "parent", "action", "depth", "children", "expanded", "solved",
                 "checked", "terminal", "reward", "value", "cache")

    def __init__(self, state, parent=None, action=None, depth=0, reward=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = depth
        self.children = []
        self.expanded = False
        self.solved = False
        self.checked = False        # has any rollout assessed this node's novelty yet
        self.terminal = False
        self.reward = reward        # for the transition from the parent
        self.value = 0.0            # best discounted return backed up from below
        self.cache = None           # whatever the action-selection policy wants to keep

    @property
    def literals(self):
        return self.state.literals

    def path(self):
        """The actions from the root to here."""
        actions = []
        node = self
        while node.parent is not None:
            actions.append(node.action)
            node = node.parent
        return actions[::-1]

    def trace(self):
        states = []
        node = self
        while node is not None:
            states.append(node.state)
            node = node.parent
        return states[::-1]


class RolloutIW:
    """Rollout IW(k), online or as one lookahead.

    ```python
    from planiverse.planners.width import RolloutIW, Budget

    env.set_index(0)
    result = RolloutIW(
        width=1,
        expansions_per_step=500,           # the paper's per-decision budget, in expansions
        progress=lambda s: s.blocks_left,  # stands in for the reward
        seed=0,
    ).solve(env, Budget(max_expansions=50_000, max_seconds=60))
    ```

    The result's plan is the sequence of actions executed plus the path a rollout found to
    the goal, so it replays from the initial state like any other planner's.
    """

    def __init__(self, width=1, expansions_per_step=None, progress=None, discount=0.99,
                 max_depth=None, max_steps=200, max_episodes=1, reuse_tree=True,
                 avoid_dead_ends=True, strict=True, policy=None, seed=None):
        """
        - `expansions_per_step`: the lookahead budget before one action is committed to.
          `None` runs a single lookahead from the start state and never commits.
        - `progress(state)`: lower is better. A transition's reward is the drop in it.
        - `discount`: applied per step when returns are backed up (the paper's γ).
        - `max_depth`: rollouts stop at this depth below the current root; `None` leaves it
          to novelty, which bounds them on its own.
        - `max_steps`: actions committed to per episode before giving up.
        - `max_episodes`: how many times to start over from the initial state when an
          episode ends without a goal; `None` keeps going until the budget runs out. Plain
          Rollout IW learns nothing between episodes, so a second one is a fresh draw; π-IW
          trains on each, which is why it sets this to `None`.
        - `reuse_tree`: keep the subtree beneath the executed action, as the paper does.
        - `avoid_dead_ends`: never commit to a terminal child while there is another.
        - `policy(state, actions) -> weights`: how a rollout picks among the unsolved
          children of a node. Uniform when `None`; π-IW supplies a learned one.
        - `seed`: for the rollouts' random choices.
        """
        if width < 1:
            raise ValueError(f"width must be at least 1, got {width}")
        if expansions_per_step is not None and expansions_per_step < 1:
            raise ValueError("expansions_per_step must be at least 1, or None")
        if not 0.0 < discount <= 1.0:
            raise ValueError(f"discount must be in (0, 1], got {discount}")
        self.width = width
        self.expansions_per_step = expansions_per_step
        self.progress = progress
        self.discount = discount
        self.max_depth = max_depth
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.reuse_tree = reuse_tree
        self.avoid_dead_ends = avoid_dead_ends
        self.strict = strict
        self.policy = policy
        self.seed = seed
        self.random = random.Random(seed)

    # ------------------------------------------------------------------- the driver

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics(widths_tried=(self.width,))
        DepthNoveltyTable(self.width, strict=self.strict)     # refuse a bad width up front

        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return self.__result__("solved", [], [state], statistics, budget)

        episodes = 0
        while True:
            outcome = self.__episode__(env, state, statistics, budget)
            episodes += 1
            statistics.episodes = episodes
            if outcome is not None:
                status, plan, trace = outcome
                return self.__result__(status, plan, trace, statistics, budget)
            self.__episode_ended__(episodes)
            if self.max_episodes is not None and episodes >= self.max_episodes:
                return self.__result__("failed", None, [], statistics, budget)
            if budget.exhausted(statistics.expansions):
                return self.__result__("out_of_budget", None, [], statistics, budget)

    def __episode__(self, env, state, statistics, budget):
        """Plan, commit, repeat, until a goal, a dead end, the step cap or the budget.

        Returns `("solved", plan, trace)`, `("out_of_budget", None, [])`, or `None` when the
        episode ended without a plan and another may follow. The plan is the actions
        committed to so far plus the path a rollout found from the current root.
        """
        root = RolloutNode(state)
        plan, trace = [], [state]
        for _ in range(self.max_steps):
            table = DepthNoveltyTable(self.width, strict=self.strict)
            outcome = self.__lookahead__(env, root, table, statistics, budget)
            statistics.novelty_evaluations += table.evaluations
            statistics.tuples_enumerated += table.tuples_enumerated
            if outcome is not None:
                status, node = outcome
                if status == "solved":
                    return status, plan + node.path(), trace + node.trace()[1:]
                return status, None, []
            if self.expansions_per_step is None:
                # One lookahead, no commitment. The root is solved without a goal beneath
                # it, which is this lookahead's word for "nothing more to see".
                return None
            child = self.__choose__(root)
            if child is None:
                return None
            self.__committed__(root, child)
            plan.append(child.action)
            trace.append(child.state)
            if env.is_goal(child.state):
                return "solved", plan, trace
            if child.terminal:
                return None                 # the episode is over
            root = self.__reroot__(child)
        return None

    def __lookahead__(self, env, root, table, statistics, budget):
        """Rollouts from `root` until it is solved, a goal is found or the step budget goes."""
        # Depths are relative to the root of this lookahead, and the table is new, so
        # everything cached beneath the root is up for assessment again.
        self.__reopen__(root)
        spent = 0
        while not root.solved:
            if budget.exhausted(statistics.expansions):
                return "out_of_budget", None
            if self.expansions_per_step is not None and spent >= self.expansions_per_step:
                return None
            before = statistics.expansions
            goal = self.__rollout__(env, root, table, statistics, budget)
            spent += statistics.expansions - before
            statistics.rollouts += 1
            if goal is not None:
                return "solved", goal
        return None

    def __rollout__(self, env, root, table, statistics, budget):
        """One walk down from the root. Returns a goal node if it found one."""
        node = root
        while True:
            if not node.expanded:
                if budget.exhausted(statistics.expansions):
                    return None
                goal = self.__expand__(env, node, statistics)
                if goal is not None:
                    return goal
            candidates = [child for child in node.children if not child.solved]
            if not candidates:
                self.__solve__(node)
                break
            child = self.__pick__(node, candidates)
            novel = table.check(self.__atoms__(child), child.depth, new=not child.checked)
            child.checked = True
            if not novel:
                statistics.pruned_novelty += 1
                self.__solve__(child)
                break
            if self.max_depth is not None and child.depth >= self.max_depth:
                self.__solve__(child)
                break
            node = child
        self.__backup__(node)
        return None

    def __expand__(self, env, node, statistics):
        """Generate every child at once, which is what `successors()` costs anyway."""
        successors = env.successors(node.state)
        statistics.expansions += 1
        statistics.generated += len(successors)
        node.expanded = True
        base = self.progress(node.state) if self.progress else 0.0
        for action, state in successors:
            reward = (base - self.progress(state)) if self.progress else 0.0
            child = RolloutNode(state, parent=node, action=action, depth=node.depth + 1,
                                reward=reward)
            node.children.append(child)
            # Goal before terminal: every absorbing goal state here is terminal as well.
            if env.is_goal(state):
                return child
            if env.is_terminal(state):
                statistics.pruned_terminal += 1
                child.terminal = True
                child.solved = True
        return None

    def __pick__(self, node, candidates):
        """Which unsolved child a rollout follows. Uniform unless a policy says otherwise."""
        if self.policy is None or len(candidates) == 1:
            return self.random.choice(candidates)
        weights = list(self.policy(node.state, [child.action for child in candidates]))
        if len(weights) != len(candidates) or min(weights) < 0 or not sum(weights) > 0:
            return self.random.choice(candidates)
        return self.random.choices(candidates, weights=weights)[0]

    def __atoms__(self, node):
        """What novelty is measured over. π-IW overrides this with learned features."""
        return node.literals

    @staticmethod
    def __solve__(node):
        """Mark `node` solved and propagate upward while every sibling is too."""
        node.solved = True
        node = node.parent
        while node is not None and node.expanded and all(c.solved for c in node.children):
            node.solved = True
            node = node.parent

    def __return__(self, child):
        """The discounted return of taking the step to `child`, as backed up so far.

        A dead end is worth minus infinity under `avoid_dead_ends`: `is_terminal` means no
        goal is reachable from there, so whatever progress the step itself made is progress
        into a wall, and a node whose every child is a dead end inherits the same value.
        Without the flag a dead end is worth its own step, which is Atari's reading, where
        losing merely ends the scoring.
        """
        if self.avoid_dead_ends and child.terminal:
            return float("-inf")
        return child.reward + self.discount * child.value

    def __backup__(self, node):
        """Max-backup of discounted returns from `node` to the root."""
        while node is not None:
            if node.children:
                node.value = max(self.__return__(child) for child in node.children)
            node = node.parent

    def __choose__(self, root):
        """The child to commit to: best backed-up return, ties broken at random."""
        if not root.children:
            return None
        best = max(self.__return__(child) for child in root.children)
        return self.random.choice(
            [c for c in root.children if self.__return__(c) == best])

    def __reroot__(self, child):
        """Make `child` the root, keeping its subtree when `reuse_tree` says so."""
        if not self.reuse_tree:
            return RolloutNode(child.state)
        child.parent = None
        child.action = None
        child.reward = 0.0
        queue = deque([(child, 0)])
        while queue:
            node, depth = queue.popleft()
            node.depth = depth
            node.cache = None
            queue.extend((grandchild, depth + 1) for grandchild in node.children)
        return child

    @staticmethod
    def __reopen__(root):
        """A fresh table means fresh chances: only terminal nodes stay solved."""
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if not node.terminal:
                node.solved = False
            queue.extend(node.children)

    # ------------------------------------------------------------------------ hooks

    def __committed__(self, root, child):
        """Called with the root and the child chosen from it. π-IW learns from it."""

    def __episode_ended__(self, episodes):
        """Called when an episode ends without a plan."""

    def __result__(self, status, plan, trace, statistics, budget):
        statistics.elapsed = budget.elapsed()
        if status != "solved" and self.progress is None and self.expansions_per_step:
            status = f"{status} (no progress measure; actions were committed to blind)"
        return SearchResult(plan=plan, states=trace, status=status,
                            width=self.width if status == "solved" else None,
                            statistics=statistics)
