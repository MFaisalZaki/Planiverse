"""The one thing every Planiverse environment is.

There used to be two base classes — `RealWorldProblem` and `RetroGame` — and the split cost
something without buying anything. It described where an environment *came from*, not what a
planner could *do* with it, so nothing could dispatch on it usefully: the `Simulator` facade
ended up asking `isinstance(env, RetroGame) or isinstance(env, RealWorldProblem)`, two
branches doing identical work. Meanwhile the distinctions that actually matter to a planner —
is the transition deterministic, how do you get back to a state to expand it again, how
expensive is one successor — were written nowhere at all.

So there is one base class now, and the taxonomy moved into data. `Environment` is the
contract; `EnvironmentSpec` in `registry.py` carries everything a caller might want to select
on, including the tags that used to be package directories.
"""
#: Methods a planner may always call.
REQUIRED_METHODS = ("reset", "fix_index", "successors", "is_goal", "is_terminal", "simulate")

#: Methods a planner should check for. `capabilities()` reports which are present.
OPTIONAL_METHODS = ("step", "validate", "get_actions", "render", "close")


def _stub(method):
    """Mark a default that only explains its own absence.

    The distinction matters to `capabilities()`. `validate` has a *working* default derived
    from `simulate` and `is_goal`, so an environment that does not override it still offers
    it. `step` and `get_actions` have defaults that only raise, so an environment that does
    not override them does not offer them at all. Overriding is therefore the wrong test,
    and so is `hasattr`.
    """
    method.__planiverse_stub__ = True
    return method


class Environment:
    """A simulator a planner can search.

    The contract is six methods. States must expose `literals` as a `frozenset`, and must be
    hashable and comparable, because that is what a planner keys its visited set on.

    Two properties every implementation owes the caller, neither of which the type system can
    enforce and both of which the registry records:

    * **Determinism.** Expanding the same state twice must give the same children. Every
      environment here is deterministic; the ones wrapping a stochastic simulator pin its
      seed and say so.
    * **A state identity that is not the path taken to it.** `depth` and history must stay
      out of `__eq__`, or no successor can ever equal its parent, the self-loop filter in
      `successors` becomes dead code, and search never closes.
    """

    #: Set by subclasses that want a name in the registry other than the class name.
    name = None

    def __init__(self, problem_name=None):
        self.name = problem_name or self.name or type(self).__name__
        self.state = None

    # ------------------------------------------------------------------- the contract
    # Not abstract methods. A missing one raises where it is *called*, naming the class and
    # the method, which is a better error than a class that cannot be constructed at all —
    # an environment half-written is still worth poking at in a REPL.

    @_stub
    def fix_index(self, index):
        """Select which instance (level, scenario, stage) `reset` will build."""
        raise NotImplementedError(f"{type(self).__name__} must implement fix_index()")

    @_stub
    def reset(self):
        """Build the selected instance and return `(initial_state, info)`."""
        raise NotImplementedError(f"{type(self).__name__} must implement reset()")

    @_stub
    def successors(self, state):
        """Every applicable action paired with the state it leads to, as
        `[(action, successor_state)]`, excluding actions that change nothing."""
        raise NotImplementedError(f"{type(self).__name__} must implement successors()")

    @_stub
    def is_goal(self, state):
        """Has this state solved the problem?"""
        raise NotImplementedError(f"{type(self).__name__} must implement is_goal()")

    @_stub
    def is_terminal(self, state):
        """Is this a dead end — no goal reachable from here?"""
        raise NotImplementedError(f"{type(self).__name__} must implement is_terminal()")

    @_stub
    def simulate(self, plan):
        """Replay `plan` from the initial state and return the full state trace, which is
        one longer than the plan."""
        raise NotImplementedError(f"{type(self).__name__} must implement simulate()")

    def validate(self, plan):
        """A plan is valid when replaying it from the initial state reaches a goal.

        Defined once here rather than in every environment: it is the same sentence in all
        of them, and it is derived from `simulate` and `is_goal` which are contract.
        """
        return self.is_goal(self.simulate(plan)[-1])

    # ------------------------------------------------------------------- extensions
    # Stateful play and a static action list only make sense for some environments. The
    # defaults say what the environment does not offer, rather than letting Python report
    # what it could not find.

    @_stub
    def step(self, action):
        """Stateful play, as opposed to expansion: advance the environment's own state."""
        raise NotImplementedError(
            f"{type(self).__name__} does not offer stateful play; expand with successors()")

    @_stub
    def get_actions(self):
        """The static action vocabulary, for environments that have one."""
        raise NotImplementedError(
            f"{type(self).__name__} has no static action list; its actions are built per "
            "state by successors()")

    @classmethod
    def capabilities(cls):
        """Which optional methods this environment actually *provides*.

        What the README's capability matrix is generated from, so the matrix cannot drift
        away from the code. An inherited default that only raises does not count as
        providing anything, which is the whole reason this cannot just be `hasattr`.
        """
        return frozenset(name for name in OPTIONAL_METHODS if cls.provides(name))

    @classmethod
    def provides(cls, name):
        """Would calling `name` on this environment do something?"""
        method = getattr(cls, name, None)
        if not callable(method):
            return False
        return not getattr(method, "__planiverse_stub__", False)


def implements_contract(candidate):
    """Does `candidate` satisfy the contract, whether or not it inherits from `Environment`?

    Kept because anything a user brings from outside is a legitimate environment without
    being a subclass. Structural, not nominal — which is the point of dropping the
    two-base-class taxonomy in the first place.

    A base-class default that only raises does not count: a bare `Environment()` has all six
    attributes and implements none of them.
    """
    for name in REQUIRED_METHODS:
        method = getattr(candidate, name, None)
        if not callable(method):
            return False
        if getattr(getattr(method, "__func__", method), "__planiverse_stub__", False):
            return False
    return True
