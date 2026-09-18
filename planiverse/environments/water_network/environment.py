"""Contamination containment in a drinking-water distribution network.

A contaminant enters the network at one junction. It spreads by being *carried*: every
node downstream of the source drinks a share of it, and which nodes those are depends on
where the water is flowing, which depends on the pressures, which are the solution of a
nonlinear system over the whole network. The operator's move is to close pipes. Closing a
pipe reroutes the flow, so it changes the answer everywhere at once.

That is the reason this environment exists. The effect of `close pipe 123` is not a set of
facts to add and delete: it is "re-solve the hydraulics and the transport, and see". Two
consequences show up immediately on the shipped networks and neither is expressible as a
PDDL action:

* **Effects are global.** On `Net3`, closing one pipe changes the pressure at 94 of the 97
  nodes.
* **Effects are not monotone.** On `Net1`, closing pipe `110` makes the contamination
  *worse*: it pushes flow down a path that reaches more customers. A delete-list cannot
  say that, and neither can a planner that assumes closing more pipes helps more.

The trade-off is the problem: every closed pipe contains a little more contamination and
costs a little more service, and which pipes do the work is not readable off the topology.
On `Net3` the source has four pipes on it, of which two carry essentially all the
contamination and two do nothing at all.

    env = WaterNetworkEnv()
    env.set_index(0)
    state, info = env.reset()
    for action, successor in env.successors(state):
        print(action, successor.contaminated, successor.service)

`generate_instance(seed)` draws a scenario of its own: one of the shipped networks and a
junction of it that, left alone, contaminates at least a stated share of what the network
delivers, which is the same test the bundled scenarios were chosen by, and then searched, so
that like every bundled scenario it carries the depth it was actually solved at.

Built on WNTR, the US EPA's Python interface to EPANET:
https://github.com/USEPA/WNTR
"""
import os
import shutil
import tempfile
from collections import namedtuple
from copy import deepcopy

from planiverse.environments.base import Environment
from planiverse.environments.generation import bounded_search, draw_until, rng

#: Hydraulics are run pressure-driven rather than demand-driven. It matters here: under the
#: demand-driven model a node takes its full demand no matter how little pressure is left,
#: so closing pipes would look free. Pressure-driven delivery is what makes losing service a
#: real cost of containment.
REQUIRED_PRESSURE = 20.0
MINIMUM_PRESSURE = 0.0

#: Long enough for the contaminant to reach the far end of the network and for the tanks to
#: go through a fill/draw cycle; short enough that one solve stays well under a second.
DURATION_HOURS = 12

#: Containment is judged on the share of all delivered water that came from the source, and
#: service on the share of expected demand actually delivered.
CONTAMINATION_GOAL = 0.02
SERVICE_GOAL = 0.80
SERVICE_FLOOR = 0.50

#: How far from the source a pipe may be and still be worth closing, in hops. The full pipe
#: list is a branching factor of 117 on Net3, nearly all of it nowhere near the incident.
#: `None` means "every pipe" and is honest but slow (see `WaterNetworkEnv.__init__`).
DEFAULT_RADIUS = 2


def network_library():
    """Where WNTR keeps the benchmark networks it ships."""
    import wntr

    return os.path.join(os.path.dirname(wntr.__file__), "library", "networks")


#: The scenarios `set_index` chooses between: a shipped network and the junction the
#: contaminant enters at.
#:
#: The sources are not arbitrary: they were measured with `rank_sources` below, which runs
#: every junction of a network as the source and reports how much of the delivered water
#: ends up contaminated. A source that poisons a few percent of the network is not a
#: planning problem; these each reach 40-80% of everything delivered. Two networks appear
#: twice, at different severities, so the set spans easy and hard instances of the same
#: topology.
#: `solved_at` is the shallowest depth a solution was actually found at, not an estimate.
#: It is recorded because an instance whose goal nobody has reached is a poor benchmark
#: entry, since a planner cannot tell "no solution" from "not yet". Every scenario below has been
#: solved; `Net2` was dropped for exactly this reason (see the module tests).
Scenario = namedtuple("Scenario", ["network", "source", "baseline", "solved_at"])

#: The networks `generate_instance` draws from. `Net2` is left out: no source on it has been
#: solved (see the module tests), so a scenario drawn on it could not be checked.
GENERATOR_NETWORKS = ("Net1.inp", "Net3.inp")

#: How much of the delivered water a generated scenario's source must contaminate when
#: nothing is closed. Below this there is not enough to contain.
MIN_BASELINE = 0.1

#: Expansions the generator's search may spend proving a drawn scenario solvable. Every
#: expansion is one hydraulic solve per candidate pipe, so this is small: it covers every
#: two-closure plan on the shipped networks and the shallower three-closure ones.
SEARCH_LIMIT = 150

SCENARIOS = (
    Scenario("Net1.inp", "23", 0.136, 2),
    Scenario("Net3.inp", "123", 0.619, 2),
    Scenario("Net3.inp", "199", 0.414, 2),
    Scenario("Net3.inp", "121", 0.563, 2),
    Scenario("Net1.inp", "21", 0.300, 3),
    Scenario("Net3.inp", "119", 0.562, 3),
    Scenario("Net1.inp", "12", 0.408, 4),
    Scenario("Net1.inp", "22", 0.287, 4),
    Scenario("Net1.inp", "11", 0.801, 7),
)


def rank_sources(network_file, duration_hours=DURATION_HOURS):
    """Every junction as a contamination source, worst first.

    How the scenarios above were chosen, kept so the choice can be re-derived rather than
    taken on trust. Returns `(contaminated, service, junction)` triples.
    """
    import wntr

    model = wntr.network.WaterNetworkModel(network_file)
    ranked = []
    env = WaterNetworkEnv(duration_hours=duration_hours)
    env.network_file = network_file
    try:
        for junction in model.junction_name_list:
            env.source, env._wn, env._cache = junction, None, {}
            try:
                service, contaminated, _ = env.__simulate__(frozenset())
            except Exception:
                continue          # a junction the solver cannot trace from is not a scenario
            ranked.append((contaminated, service, junction))
    finally:
        env.close()
    return sorted(ranked, reverse=True)


class WaterNetworkAction:
    """Close one pipe. The whole action set, and the whole operator interface."""

    def __init__(self, pipe):
        self.pipe = pipe

    def cost(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, WaterNetworkAction) and self.pipe == other.pipe

    def __hash__(self):
        return hash(self.pipe)

    def __lt__(self, other):
        return self.pipe < other.pipe

    def __str__(self):
        return f"close_pipe_{self.pipe}"

    def __repr__(self):
        return str(self)


class WaterNetworkState:
    """A set of closed pipes, and what the network does as a result.

    The closed set is the whole state. That is not a simplification: the hydraulic and
    transport solves are deterministic functions of the network and its controls, verified
    by running the same configuration twice and comparing pressures (identical to the last
    bit). So two states with the same pipes closed *are* the same state, and search can
    close over them without simulating anything twice.
    """

    def __init__(self, closed, service, contaminated, pressure_deficit, depth=0):
        self.closed = frozenset(closed)
        self.service = service
        self.contaminated = contaminated
        self.pressure_deficit = pressure_deficit
        self.depth = depth

        # Bucketed, because a planner keyed on the raw floats would treat every state as
        # novel. The closed set is what identifies the state; these are for width-based
        # methods that measure novelty over atoms.
        literals = [f"closed({pipe})" for pipe in sorted(self.closed)]
        literals.append(f"contaminated({int(contaminated * 20)})")
        literals.append(f"service({int(service * 20)})")
        self.literals = frozenset(literals)

    def __eq__(self, other):
        return isinstance(other, WaterNetworkState) and self.closed == other.closed

    def __hash__(self):
        return hash(self.closed)

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        closed = ", ".join(sorted(self.closed)) or "nothing"
        return (f"closed: {closed}\n"
                f"contaminated delivered: {self.contaminated:.1%}\n"
                f"service: {self.service:.1%}")

    def __repr__(self):
        return (f"<WaterNetworkState(closed={len(self.closed)}, "
                f"contaminated={self.contaminated:.3f}, service={self.service:.3f})>")


class WaterNetworkEnv(Environment):
    """Close pipes to contain a contaminant without cutting off the customers.

    Needs `wntr`, the EPA's EPANET wrapper, which ships the benchmark networks used here.
    """

    def __init__(self, radius=DEFAULT_RADIUS, duration_hours=DURATION_HOURS,
                 contamination_goal=CONTAMINATION_GOAL, service_goal=SERVICE_GOAL,
                 service_floor=SERVICE_FLOOR):
        super().__init__("water_distribution")
        self.radius = radius
        self.duration_hours = duration_hours
        self.contamination_goal = contamination_goal
        self.service_goal = service_goal
        self.service_floor = service_floor

        self.scenario_index = None
        #: The scenario `reset` builds, as `{"network": ..., "source": ...}`: a bundled one
        #: after `set_index`, or whatever `set_instance` was given.
        self.instance = None
        self.network_file = None
        self.source = None
        self.candidates = ()
        self.baseline = None
        #: The plan `generate_instance` accepted the current scenario on, when it checked
        #: one, and what the search spent finding it.
        self.witness = None
        self.witness_expansions = None
        self.state = None
        self.state_history = []

        self._wn = None
        self._cache = {}
        # EpanetSimulator writes temp.inp/.bin/.rpt beside the process's working directory
        # unless told otherwise, and expansion runs it hundreds of times. Everything goes in
        # here instead, and `close()` removes it.
        self._workdir = tempfile.mkdtemp(prefix="planiverse-wntr-")

    # ------------------------------------------------------------------ instances

    def set_index(self, index):
        """Choose the scenario: which network, and where the contaminant enters."""
        if not 0 <= index < len(SCENARIOS):
            raise IndexError(
                f"Invalid index: {index}. There are {len(SCENARIOS)} scenarios, so the "
                f"index must be 0-{len(SCENARIOS) - 1}.")
        scenario = SCENARIOS[index]
        self.set_instance({"network": scenario.network, "source": scenario.source,
                           "baseline": scenario.baseline, "solved_at": scenario.solved_at})
        self.scenario_index = index

    def set_instance(self, instance):
        """Select a scenario: `{"network": file, "source": junction}`.

        `network` is the name of one of the networks WNTR ships, or a path to an EPANET
        `.inp` file of your own; `source` is the name of a junction in it.
        """
        network, source = instance["network"], str(instance["source"])
        path = network if os.path.isfile(network) else os.path.join(network_library(), network)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no network at {network}, nor in WNTR's library")
        self.instance = dict(instance)
        self.scenario_index = None
        self.network_file = path
        self.source = source
        self.witness = self.witness_expansions = None
        self._wn = None
        self._cache = {}

    def generate_instance(self, seed=None, network=None, min_baseline=MIN_BASELINE,
                          solvable=True, search_limit=SEARCH_LIMIT, min_plan_length=1,
                          attempts=40):
        """Draw a fresh scenario, select it, and return it as a dict.

        `network` names the network to draw on (default: one of `GENERATOR_NETWORKS` at
        random; a path to your own `.inp` works too). The source is a junction of it drawn at
        random and kept only if, with nothing closed, at least `min_baseline` of the delivered
        water comes from it: the test the bundled scenarios were chosen by, which is what
        makes a draw a containment problem rather than a non-event. With `solvable` the draw
        is then searched breadth-first for up to `search_limit` expansions and kept only if a
        plan of at least `min_plan_length` closures was found, which is left in `witness`
        and recorded in the instance as `solved_at`, the way every bundled scenario carries
        the depth it was solved at. Each candidate costs one hydraulic solve to measure and
        one per candidate pipe per expansion to check, so `attempts` bounds the junctions
        tried.
        """
        import wntr

        random_, _ = rng(seed)
        network = network or random_.choice(GENERATOR_NETWORKS)
        self.set_instance({"network": network, "source": ""})
        junctions = list(wntr.network.WaterNetworkModel(self.network_file).junction_name_list)
        random_.shuffle(junctions)
        measured = {}

        def draw(attempt):
            return junctions[attempt] if attempt < len(junctions) else None

        def accept(junction):
            self.set_instance({"network": network, "source": junction})
            try:
                _, contaminated, _ = self.__simulate__(frozenset())
            except Exception:
                return False              # a junction the solver cannot trace from
            if contaminated < min_baseline:
                return False
            found = {"baseline": contaminated}
            if solvable:
                outcome = bounded_search(self, search_limit)
                if outcome.plan is None or len(outcome.plan) < min_plan_length:
                    return False
                found.update(solved_at=len(outcome.plan), plan=outcome.plan,
                             expansions=outcome.expansions)
            measured.update(found)
            return True

        source = draw_until(draw, accept, min(attempts, len(junctions)),
                            f"contamination source on {os.path.basename(network)}")
        instance = {"network": network, "source": source, "baseline": measured["baseline"]}
        if solvable:
            instance["solved_at"] = measured["solved_at"]
        self.set_instance(instance)
        if solvable:
            self.witness, self.witness_expansions = measured["plan"], measured["expansions"]
        return instance

    # ------------------------------------------------------------------ the network

    def __model__(self):
        """The network as loaded, with the options this scenario is defined by."""
        import wntr

        if self._wn is None:
            if self.network_file is None:
                raise ValueError("Call set_index() or set_instance() before reset().")
            wn = wntr.network.WaterNetworkModel(self.network_file)
            wn.options.time.duration = 3600 * self.duration_hours
            wn.options.hydraulic.demand_model = "PDD"
            wn.options.hydraulic.required_pressure = REQUIRED_PRESSURE
            wn.options.hydraulic.minimum_pressure = MINIMUM_PRESSURE
            # A trace analysis reports, at every node and every timestep, the percentage of
            # the water there that came from `trace_node`. That is exactly "who is drinking
            # the contaminant", and it is transport on top of the hydraulic solution.
            wn.options.quality.parameter = "TRACE"
            wn.options.quality.trace_node = self.source
            if self.source not in wn.junction_name_list:
                raise ValueError(
                    f"{self.source} is not a junction of {os.path.basename(self.network_file)}")
            self._wn = wn
        return self._wn

    def __candidates__(self):
        """Pipes close enough to the incident to be worth closing.

        Every pipe in the network is a legal thing to close, but on Net3 that is a branching
        factor of 117 and nearly all of it is nowhere near the contamination. The default
        keeps pipes within `radius` hops of the source; `radius=None` keeps all of them,
        which is the honest setting and a slow one.
        """
        wn = self.__model__()
        if self.radius is None:
            return tuple(sorted(wn.pipe_name_list))

        reached = {self.source}
        for _ in range(self.radius):
            frontier = set()
            for pipe in wn.pipe_name_list:
                link = wn.get_link(pipe)
                ends = {link.start_node_name, link.end_node_name}
                if ends & reached:
                    frontier |= ends
            reached |= frontier
        return tuple(sorted(
            pipe for pipe in wn.pipe_name_list
            if {wn.get_link(pipe).start_node_name,
                wn.get_link(pipe).end_node_name} & reached))

    # ------------------------------------------------------------------ simulation

    def __simulate__(self, closed):
        """Run the network with `closed` shut, and measure containment and service.

        Memoised on the closed set, which is sound precisely because the solve is
        deterministic: the same configuration cannot produce two different answers.
        """
        import wntr

        key = frozenset(closed)
        if key in self._cache:
            return self._cache[key]

        wn = self.__model__()
        model = deepcopy(wn)
        for pipe in key:
            model.get_link(pipe).initial_status = wntr.network.LinkStatus.Closed

        simulator = wntr.sim.EpanetSimulator(model)
        results = simulator.run_sim(file_prefix=os.path.join(self._workdir, "run"))

        junctions = wn.junction_name_list
        delivered = results.node["demand"][junctions].clip(lower=0)
        trace = results.node["quality"][junctions] / 100.0
        # Both sides are clipped at zero because a junction may have a *negative* demand,
        # which is an injection into the network rather than a customer drawing from it.
        # Net2 has one, and counted as written it drags the network's total expected demand
        # to -0.02 and makes the service ratio meaningless (it came out at -1791%).
        expected = wntr.metrics.expected_demand(wn)[junctions].clip(lower=0)

        total = float(expected.sum().sum())
        service = float(delivered.sum().sum()) / total if total else 1.0
        contaminated = float((delivered * trace).sum().sum()) / total if total else 0.0

        pressure = results.node["pressure"][junctions]
        deficit = float((REQUIRED_PRESSURE - pressure).clip(lower=0).mean().mean())

        # Service can drift a hair above 1.0 when a tank contributes more than the expected
        # demand over the window; clamping keeps the goal test meaning what it says.
        measured = (min(service, 1.0), max(contaminated, 0.0), deficit)
        self._cache[key] = measured
        return measured

    def __state__(self, closed, depth):
        service, contaminated, deficit = self.__simulate__(closed)
        return WaterNetworkState(closed, service, contaminated, deficit, depth)

    # ------------------------------------------------------------------ interface

    def reset(self):
        if self.instance is None:
            self.set_index(0)
        self.candidates = self.__candidates__()
        self.state = self.__state__(frozenset(), 0)
        self.baseline = self.state
        self.state_history = [self.state]
        return self.state, {"network": self.instance["network"],
                            "source": self.instance["source"],
                            "scenario": self.scenario_index,
                            "generated": self.scenario_index is None,
                            "solved_at": self.instance.get("solved_at"),
                            "pipes": len(self.__model__().pipe_name_list),
                            "candidates": len(self.candidates),
                            "contaminated": self.state.contaminated,
                            "service": self.state.service}

    def is_goal(self, state):
        """Contained, and the customers still have water.

        Both halves are needed. Closing every pipe at the source contains the contamination
        perfectly and is not a solution: on Net1 it costs 14% of all service.
        """
        return (state.contaminated <= self.contamination_goal
                and state.service >= self.service_goal)

    def is_terminal(self, state):
        """Service has collapsed, so no further closure can lead anywhere good.

        Sound because the only action is to close another pipe, and a network cannot deliver
        more water with more pipes shut: service is monotone in the closed set even though
        *contamination* is not.
        """
        return state.service < self.service_floor and not self.is_goal(state)

    def successors(self, state):
        """Close each candidate pipe that is not already closed."""
        if self.is_goal(state) or self.is_terminal(state):
            return []
        successors = []
        for pipe in self.candidates:
            if pipe in state.closed:
                continue
            action = WaterNetworkAction(pipe)
            successor = self.__state__(state.closed | {pipe}, state.depth + 1)
            if successor == state:
                continue
            successors.append((action, successor))
        return successors

    def __advance__(self, state, action):
        if self.is_goal(state) or self.is_terminal(state):
            return state
        pipe = action.pipe if isinstance(action, WaterNetworkAction) else action
        return self.__state__(state.closed | {pipe}, state.depth + 1)

    def simulate(self, plan):
        state, _ = self.reset()
        trace = [state]
        for action in plan:
            trace.append(self.__advance__(trace[-1], action))
        return trace

    def step(self, action):
        """Stateful play. The reward is the contamination cut since the incident began."""
        if self.state is None:
            raise ValueError("Game not initialized. Call reset() first.")
        before = self.state.contaminated
        self.state = self.__advance__(self.state, action)
        self.state_history.append(self.state)
        return self.state, before - self.state.contaminated

    def get_actions(self):
        return [WaterNetworkAction(pipe) for pipe in self.candidates]

    def render(self):
        """Print the history of `step` calls, and return it as strings."""
        rendered = [str(state) for state in self.state_history]
        for step, text in enumerate(rendered):
            print(f"Step: {step}")
            print(text)
            print("--------------")
        return rendered

    def close(self):
        shutil.rmtree(self._workdir, ignore_errors=True)
