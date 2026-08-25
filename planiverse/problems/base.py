"""The contract every Planiverse environment answers to.

The README promises "one small, uniform interface" over simulators; this class is that
interface, stated once. `Simulator` dispatches on it and `tests/test_interface.py` checks
it uniformly, so an environment that misses a method fails at test time rather than on the
first call a planner happens to make.

The core contract is the four questions from the README, spelled as six methods:

- What is the initial state? — `fix_index`, `reset`
- What can happen next? — `successors`
- Am I done, and did I win? — `is_goal`, `is_terminal`
- What does this plan actually do? — `simulate`

States handed out by `reset`, `successors` and `simulate` must expose `literals`: a
frozenset of ground facts as strings, which is what planners key closed lists on.

`step` and `get_actions` are extensions, not part of the core: stateful play and a static
action list only make sense for some environments (the games have both; the real-world
problems build their actions per state inside `successors`). The defaults here raise with
an explanation instead of an `AttributeError`, so a caller learns what the environment
does not offer rather than what Python could not find.
"""


class Environment:
    """What a planner may assume about any Planiverse environment."""

    def fix_index(self, index):
        """Select which instance (level, scenario, stage) `reset` will build."""
        raise NotImplementedError(f"{type(self).__name__} must implement fix_index()")

    def reset(self):
        """Build the selected instance and return `(initial_state, info)`."""
        raise NotImplementedError(f"{type(self).__name__} must implement reset()")

    def successors(self, state):
        """Every applicable action paired with the state it leads to, as
        `[(action, successor_state)]`, excluding actions that change nothing."""
        raise NotImplementedError(f"{type(self).__name__} must implement successors()")

    def is_goal(self, state):
        raise NotImplementedError(f"{type(self).__name__} must implement is_goal()")

    def is_terminal(self, state):
        raise NotImplementedError(f"{type(self).__name__} must implement is_terminal()")

    def simulate(self, plan):
        """Replay `plan` from the initial state and return the full state trace,
        which is one longer than the plan."""
        raise NotImplementedError(f"{type(self).__name__} must implement simulate()")

    def validate(self, plan):
        """A plan is valid when replaying it from the initial state reaches a goal."""
        return self.is_goal(self.simulate(plan)[-1])

    # ------------------------------------------------------------------- extensions

    def step(self, action):
        """Stateful play, as opposed to expansion: advance the environment's own state."""
        raise NotImplementedError(
            f"{type(self).__name__} does not offer stateful play; expand with successors()")

    def get_actions(self):
        """The static action vocabulary, for environments that have one."""
        raise NotImplementedError(
            f"{type(self).__name__} has no static action list; its actions are built per "
            "state by successors()")
