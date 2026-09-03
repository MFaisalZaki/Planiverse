"""Restoring an electricity grid to a secure state after a line trips.

A transmission line goes out. The power it was carrying does not stop: it redistributes
over every remaining line according to Kirchhoff's laws, and some of those lines now carry
more than they are rated for. Left alone, an overloaded line trips too, which redistributes
the flow again, which is how a regional blackout happens. The operator's move is to change
the *topology*: at a substation, the equipment is split across two busbars, and reassigning
which side each line, generator and load sits on reroutes the power without switching
anything off.

That is a discrete action set (a few hundred distinct reconfigurations) sitting on top of
a transition function that is a nonlinear solve. The flow on every line is the solution of
the AC power-flow equations, found by Newton-Raphson at each step. There is no way to write
"the effect of moving this line to busbar 2 is that line 17 now carries 1.08 of its rating";
the only way to know is to solve the network.

Two properties make this hard for a declarative model, both measured on the shipped case:

* **The effect of a local action is global and numerical.** One reassignment at substation 5
  changes the loading of every line in the grid.
* **Doing nothing is not safe.** The demand time series keeps moving, so an overload that is
  survivable now may cascade three steps later. On chronic 1 the worst loading after tripping
  line 6 climbs from 1.08 to 1.98 while the operator does nothing.

    env = PowerGridEnv()
    env.set_index(0)
    state, info = env.reset()
    for action, successor in env.successors(state):
        print(action, successor.max_rho)

Built on Grid2Op, the framework RTE (the French transmission operator) uses for the L2RPN
competitions: https://github.com/Grid2Op/grid2op
"""
from collections import namedtuple

from planiverse.environments.base import Environment

#: The bundled case. `test=True` uses the time series shipped inside grid2op, so nothing is
#: downloaded and every run sees the same data.
GRID_NAME = "l2rpn_case14_sandbox"

#: A line is at `rho = 1.0` when it carries exactly its thermal rating. Above that it is
#: overloaded and the grid is insecure even though nothing has failed yet.
SECURE_RHO = 1.0

#: How many steps a plan may run before the episode is abandoned. Long enough to fix an
#: overload and see it stay fixed; short enough that search terminates.
HORIZON = 10

Scenario = namedtuple("Scenario", ["chronic", "line", "rho_after_trip",
                                   "blackout_in", "solved_at"])

#: Which line to trip, on which time series.
#:
#: Chosen by N-1 analysis: every line of every chronic was tripped in turn, and these are the
#: ones that leave the grid **standing but doomed**: it survives the trip and then blacks out
#: within `blackout_in` steps if the operator does nothing. That filter matters more than it
#: sounds. Most overloads on this case *clear themselves* as demand moves: tripping line 1 on
#: chronic 0 gives a loading of 1.019 that is back under the limit two steps later on its own.
#: An instance the null plan solves is not an instance, so those are not here.
#:
#: `rho_after_trip` and `blackout_in` are measurements, recorded so that a scenario which
#: stops reproducing fails loudly instead of quietly becoming easy. `solved_at` is the
#: shallowest depth a solution was actually found at.
SCENARIOS = (
    Scenario(0, 11, 1.013, 2, 1),
    Scenario(0, 8, 1.188, 2, 1),
    Scenario(0, 9, 1.738, 2, 1),
    Scenario(1, 16, 1.044, 4, 1),
    Scenario(1, 6, 1.078, 4, 1),
    Scenario(1, 15, 1.224, 2, 1),
    Scenario(1, 3, 1.266, 4, 1),
    Scenario(1, 17, 1.914, 2, 1),
    Scenario(2, 9, 1.664, 2, 1),
)


class PowerGridAction:
    """One substation reconfiguration, or doing nothing.

    Actions are identified by their index in grid2op's `IdToAct` converter, which enumerates
    the discrete topology space in a fixed order. The index is what a plan carries, so a
    plan is a list of integers and replays identically.
    """

    def __init__(self, action_id, label=None):
        self.action_id = action_id
        self.label = label or (f"topology_{action_id}" if action_id else "do_nothing")

    def cost(self):
        return 0 if self.action_id == 0 else 1

    def __eq__(self, other):
        return isinstance(other, PowerGridAction) and self.action_id == other.action_id

    def __hash__(self):
        return hash(self.action_id)

    def __lt__(self, other):
        return self.action_id < other.action_id

    def __str__(self):
        return self.label

    def __repr__(self):
        return str(self)


class PowerGridState:
    """A grid position: which actions have been taken, and what the solve says about it.

    The action path is the state. Grid2op is deterministic once the time series is pinned
    with `set_id` and the environment is seeded (verified by running the same sequence
    twice and comparing every line's loading), so replaying a path always lands in the same
    place, and two paths that agree are the same position.
    """

    def __init__(self, path, max_rho, rhos, blackout, step, survived):
        self.path = tuple(path)
        self.max_rho = max_rho
        self.rhos = tuple(rhos)
        self.blackout = blackout
        self.step = step
        self.survived = survived
        self.depth = len(self.path)

        literals = [f"acted({index},{action_id})"
                    for index, action_id in enumerate(self.path) if action_id]
        literals.append(f"step({step})")
        if blackout:
            # A blacked-out grid has no line loadings to report (`max_rho` is infinite
            # precisely so it sorts last), so the numeric atoms are simply absent.
            literals.append("blackout")
        else:
            # Bucketed: a planner keyed on the raw loading would find every state novel.
            literals.append(f"max-loading({int(max_rho * 10)})")
            literals += [f"overloaded(line-{line})"
                         for line, rho in enumerate(self.rhos) if rho > SECURE_RHO]
        if max_rho < SECURE_RHO and not blackout:
            literals.append("secure")
        self.literals = frozenset(literals)

    def is_secure(self):
        return not self.blackout and self.max_rho < SECURE_RHO

    def overloaded_lines(self):
        return tuple(line for line, rho in enumerate(self.rhos) if rho > SECURE_RHO)

    def __eq__(self, other):
        return isinstance(other, PowerGridState) and self.path == other.path

    def __hash__(self):
        return hash(self.path)

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        if self.blackout:
            return f"step {self.step}: BLACKOUT after {self.depth} actions"
        overloaded = self.overloaded_lines()
        status = "secure" if self.is_secure() else f"overloaded: {list(overloaded)}"
        return (f"step {self.step}, {self.depth} actions\n"
                f"worst line loading: {self.max_rho:.3f}\n"
                f"{status}")

    def __repr__(self):
        loading = "blackout" if self.blackout else f"{self.max_rho:.3f}"
        return f"<PowerGridState(depth={self.depth}, max_rho={loading})>"


class PowerGridEnv(Environment):
    """Fix an insecure grid with substation topology, before it cascades.

    Needs `grid2op`. The case and its time series ship inside the package, so there is
    nothing to download.
    """

    def __init__(self, horizon=HORIZON, grid_name=GRID_NAME, restrict_to_overloads=True):
        super().__init__("power_grid")
        self.horizon = horizon
        self.grid_name = grid_name
        # Every topology action is legal, but expanding all 209 of them costs half a minute
        # a node. Restricting to the substations that touch an overloaded line is what an
        # operator would consider, and is what makes search tractable (see `successors`).
        self.restrict_to_overloads = restrict_to_overloads

        self.scenario_index = None
        self.state = None
        self.state_history = []

        self._env = None
        self._converter = None
        self._cache = {}
        self._cursor = None          # (path, live env) so expanding a parent replays once

    # ------------------------------------------------------------------ instances

    def set_index(self, index):
        """Choose which line trips, on which time series."""
        if not 0 <= index < len(SCENARIOS):
            raise IndexError(
                f"Invalid index: {index}. There are {len(SCENARIOS)} scenarios, so the "
                f"index must be 0-{len(SCENARIOS) - 1}.")
        self.scenario_index = index
        self._cache = {}
        self.__release_cursor__()

    # ------------------------------------------------------------------ the grid

    def __make__(self):
        """A fresh grid2op environment with the stochastic parts pinned.

        `seed` and `set_id` are what make this a planning problem rather than a gamble: the
        time series is fixed, so a plan replays to the same place every time.
        """
        import grid2op
        from grid2op.Parameters import Parameters

        parameters = Parameters()
        # Leave overloads *on*: a line that stays above its rating is supposed to trip, and
        # the cascade it starts is the thing the operator is planning against.
        parameters.NO_OVERFLOW_DISCONNECTION = False
        env = grid2op.make(self.grid_name, test=True, param=parameters)
        env.seed(0)
        return env

    def __converter__(self):
        """Grid2op's enumeration of the discrete topology space, in a fixed order."""
        from grid2op.Converter import IdToAct

        if self._converter is None:
            if self._env is None:
                self._env = self.__make__()
            converter = IdToAct(self._env.action_space)
            converter.init_converter(set_line_status=False, change_line_status=False,
                                     change_bus_vect=False, set_topo_vect=True,
                                     redispatch=False)
            self._converter = converter
        return self._converter

    def __release_cursor__(self):
        if self._cursor is not None:
            self._cursor[1].close()
            self._cursor = None

    def __env_at__(self, path):
        """A live environment positioned at `path`, replayed from the start.

        Replay rather than snapshot-per-state: an environment copy is megabytes and a search
        holds thousands of states. Replaying is cheap because the result of every path is
        memoised, so a path is only ever walked once, and expanding a parent replays it once
        and then copies for each child.
        """
        scenario = SCENARIOS[self.scenario_index]
        env = self.__make__()
        env.set_id(scenario.chronic)
        env.reset()
        # The contingency: the line that trips is what creates the problem.
        env.step(env.action_space({"set_line_status": [(scenario.line, -1)]}))
        converter = self.__converter__()
        for action_id in path:
            env.step(converter.convert_act(action_id))
        return env

    def __measure__(self, env, path):
        observation = env.current_obs
        blackout = observation is None or bool(env.done)
        if blackout:
            return PowerGridState(path, float("inf"), (), True, len(path) + 1, False)
        rhos = [float(value) for value in observation.rho]
        max_rho = max(rhos) if rhos else 0.0
        step = int(observation.current_step) if hasattr(observation, "current_step") else len(path) + 1
        return PowerGridState(path, max_rho, rhos, False, step, True)

    def __state__(self, path):
        """The state a path leads to. Memoised: deterministic, so it cannot differ."""
        key = tuple(path)
        if key in self._cache:
            return self._cache[key]
        env = self.__env_at__(key)
        try:
            state = self.__measure__(env, key)
        finally:
            env.close()
        self._cache[key] = state
        return state

    # ------------------------------------------------------------------ interface

    def reset(self):
        if self.scenario_index is None:
            self.set_index(0)
        self.__release_cursor__()
        self._cache = {}
        self.state = self.__state__(())
        self.state_history = [self.state]
        scenario = SCENARIOS[self.scenario_index]
        return self.state, {"grid": self.grid_name,
                            "chronic": scenario.chronic,
                            "tripped_line": scenario.line,
                            "max_rho": self.state.max_rho,
                            "overloaded": list(self.state.overloaded_lines()),
                            "actions": self.__converter__().n}

    def is_goal(self, state):
        """Every line back within its rating, and the grid still standing."""
        return state.is_secure()

    def is_terminal(self, state):
        """Blacked out, or out of time.

        A blackout is absorbing in the simulator too (grid2op ends the episode), so there is
        genuinely nothing further to plan from.
        """
        return state.blackout or state.step >= self.horizon

    def __relevant_actions__(self, state):
        """Reconfigurations at substations touching an overloaded line, plus doing nothing.

        Every topology action is legal from every state, and offering all 209 costs about
        thirty seconds a node. An operator looks at the substations either end of the line
        that is overloading, and so does this; `restrict_to_overloads=False` turns it off and
        offers the lot.
        """
        import numpy as np

        converter = self.__converter__()
        if not self.restrict_to_overloads:
            return list(range(converter.n))

        env = self._env if self._env is not None else self.__make__()
        if self._env is None:
            self._env = env
        substations = set()
        for line in state.overloaded_lines():
            substations.add(int(env.line_or_to_subid[line]))
            substations.add(int(env.line_ex_to_subid[line]))
        if not substations:
            return [0]

        relevant = [0]                      # do-nothing is always available
        for action_id in range(1, converter.n):
            _, touched = converter.convert_act(action_id).get_topological_impact()
            if substations & set(int(s) for s in np.where(touched)[0]):
                relevant.append(action_id)
        return relevant

    def successors(self, state):
        """Every relevant reconfiguration, minus the ones that change nothing."""
        if self.is_goal(state) or self.is_terminal(state):
            return []

        converter = self.__converter__()
        # Replay to the parent once, then branch from a copy per child. Replaying per child
        # would repeat the whole prefix every time.
        parent = self.__env_at__(state.path)
        successors = []
        try:
            for action_id in self.__relevant_actions__(state):
                path = state.path + (action_id,)
                if path in self._cache:
                    successor = self._cache[path]
                else:
                    child = parent.copy()
                    try:
                        child.step(converter.convert_act(action_id))
                        successor = self.__measure__(child, path)
                    finally:
                        child.close()
                    self._cache[path] = successor
                if successor == state:
                    continue
                successors.append((PowerGridAction(action_id), successor))
        finally:
            parent.close()
        return successors

    def __advance__(self, state, action):
        if self.is_goal(state) or self.is_terminal(state):
            return state
        action_id = action.action_id if isinstance(action, PowerGridAction) else int(action)
        return self.__state__(state.path + (action_id,))

    def simulate(self, plan):
        state, _ = self.reset()
        trace = [state]
        for action in plan:
            trace.append(self.__advance__(trace[-1], action))
        return trace

    def step(self, action):
        """Stateful play. The reward is how much the worst line loading came down."""
        if self.state is None:
            raise ValueError("Game not initialized. Call reset() first.")
        before = self.state.max_rho
        self.state = self.__advance__(self.state, action)
        self.state_history.append(self.state)
        relief = before - self.state.max_rho
        return self.state, relief if relief == relief and abs(relief) != float("inf") else 0.0

    def get_actions(self):
        """Every topology action the case has, whatever state it is in."""
        return [PowerGridAction(action_id) for action_id in range(self.__converter__().n)]

    def render(self):
        rendered = [str(state) for state in self.state_history]
        for step, text in enumerate(rendered):
            print(f"Step: {step}")
            print(text)
            print("--------------")
        return rendered

    def close(self):
        self.__release_cursor__()
        if self._env is not None:
            self._env.close()
            self._env = None
        self._converter = None
