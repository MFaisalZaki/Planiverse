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
from abc import ABC, abstractmethod

#: Methods a planner may always call.
REQUIRED_METHODS = ("reset", "fix_index", "successors", "is_goal", "is_terminal", "simulate")

#: Methods a planner should check for. `capabilities()` reports which are present.
OPTIONAL_METHODS = ("step", "validate", "get_actions", "render", "close")


class Environment(ABC):
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

    @abstractmethod
    def reset(self):
        """Build the initial state. Returns `(state, info)`."""

    @abstractmethod
    def fix_index(self, index):
        """Select which instance to load. Call before `reset`."""

    @abstractmethod
    def successors(self, state):
        """Expand. Returns a list of `(action, next_state)`, self-loops filtered out."""

    @abstractmethod
    def is_goal(self, state):
        """Has this state solved the problem?"""

    @abstractmethod
    def is_terminal(self, state):
        """Is this a dead end — no goal reachable from here?"""

    @abstractmethod
    def simulate(self, plan):
        """Replay a plan from the initial state. Returns the trace of states."""

    @classmethod
    def capabilities(cls):
        """Which optional methods this environment actually provides.

        What the README's capability matrix is generated from, so the matrix cannot drift
        away from the code.
        """
        return frozenset(name for name in OPTIONAL_METHODS
                         if callable(getattr(cls, name, None)))


def implements_contract(candidate):
    """Does `candidate` satisfy the contract, whether or not it inherits from `Environment`?

    Kept because the PDDLGym wrapper and anything a user brings from outside are legitimate
    environments without being subclasses. Structural, not nominal — which is the point of
    dropping the two-base-class taxonomy in the first place.
    """
    return all(callable(getattr(candidate, name, None)) for name in REQUIRED_METHODS)
