"""Traffic signals on SUMO: a three-by-three grid of signalised junctions, a seeded morning of
trips across it, and a switch every fifteen seconds; get everyone through before the horizon
with the total travel time under the target.

SUMO (https://eclipse.dev/sumo/, Eclipse Public License 2.0 with GPL-2.0-or-later as a
secondary licence; Lopez et al. 2018, https://doi.org/10.1109/ITSC.2018.8569938) is the
Eclipse Foundation's microscopic traffic simulator. It is a dependency installed from PyPI
(`eclipse-sumo` for the binaries, `libsumo` for the in-process binding); nothing of it is
included here. The network is drawn by its `netgenerate` on first use and the trips by this
module from the instance's seed.

The decisions are which signals to switch: one junction, a row, a column, all nine, or none,
and a switch is three seconds of amber then the other green. Everything else is SUMO's:
car-following, lane changing, queues spilling back through the grid, and the time each trip
takes, which is only known by running it.

## Trajectory constraints as a goal

Total travel time is accumulated in the state as vehicle-seconds on the road, and the goal is
every vehicle arrived before the horizon with that total under the target; the horizon passing
with vehicles still on the road is a dead end, and so is everyone arrived over the target,
since nothing can then improve. The target is the best total any fixed cycle achieves, so
the planner is asked to do at least as well as a timer by switching where the queues are.

## Determinism

SUMO replays exactly from its seed and the same commands, and a saved state reloads with the
route file re-read, so a state is its decisions and every expansion replays the morning from
the first vehicle. A full morning of two hundred vehicles takes a tenth of a second.
"""
import os
import random
import subprocess
import tempfile

from planiverse.environments.base import Environment
from planiverse.environments.generation import draw_until, rng

#: The grid: three by three, 150 m blocks, 120 m stubs to the edge of the map.
GRID, BLOCK, STUB = 3, 150, 120
#: A decision covers this many seconds, of which a switch spends these on amber.
DECISION, AMBER = 15, 3
#: The junctions with signals worth switching, by SUMO's names for a grid.
JUNCTIONS = tuple(f"{column}{row}" for column in "ABC" for row in "012")
ROWS = tuple(tuple(f"{column}{row}" for column in "ABC") for row in "012")
COLUMNS = tuple(tuple(f"{column}{row}" for row in "012") for column in "ABC")
#: What the generator draws when not told.
VEHICLES = (120, 160, 200, 240)
SPACINGS = (1.0, 1.5, 2.0)
HORIZONS = (600, 750, 900)
#: How often a fixed cycle switches all signals, in decisions, for the reference timers.
CYCLES = (2, 3, 4, 6)

_net = {}


def _libsumo():
    import libsumo
    return libsumo


def network():
    """The grid's network file, drawn by SUMO's netgenerate once per process."""
    if "path" not in _net:
        import sumo
        home = os.path.dirname(sumo.__file__)
        folder = tempfile.mkdtemp(prefix="planiverse-traffic-")
        path = os.path.join(folder, "grid.net.xml")
        subprocess.run([os.path.join(home, "bin", "netgenerate"), "--grid", f"--grid.number={GRID}",
                        f"--grid.length={BLOCK}", f"--grid.attach-length={STUB}",
                        "--default.lanenumber=1", "--default-junction-type", "traffic_light",
                        "--tls.default-type", "static", "--output-file", path],
                       check=True, capture_output=True)
        _net["path"], _net["folder"] = path, folder
    return _net["path"]


def trips(instance):
    """The instance's trips as a route file: seeded pairs of an entry and an exit stub on
    different sides, one departure every `spacing` seconds."""
    random_ = random.Random(instance["seed"])
    sides = ("left", "right", "top", "bottom")
    entries = [f"{side}{k}{column}{row}" for side in sides for k in range(GRID)
               for column, row in [_stub_junction(side, k)]]
    exits = [f"{column}{row}{side}{k}" for side in sides for k in range(GRID)
             for column, row in [_stub_junction(side, k)]]
    path = os.path.join(_net["folder"], f"trips-{instance['seed']}-{instance['vehicles']}.rou.xml")
    if not os.path.exists(path):
        lines = ["<routes>"]
        for k in range(instance["vehicles"]):
            start = random_.choice(entries)
            end = random_.choice([e for e in exits if e[2:] != start[:-2]])
            lines.append(f'  <trip id="v{k}" depart="{k * instance["spacing"]:.1f}" from="{start}" to="{end}"/>')
        lines.append("</routes>")
        with open(path, "w") as handle:
            handle.write("\n".join(lines) + "\n")
    return path


def _stub_junction(side, k):
    """The grid junction a stub attaches to: SUMO names columns A.. and rows 0.."""
    if side == "left":
        return "A", str(k)
    if side == "right":
        return "ABC"[GRID - 1], str(k)
    if side == "bottom":
        return "ABC"[k], "0"
    return "ABC"[k], str(GRID - 1)


class TrafficAction:
    """`switch(j)` for a junction such as `B1`, `switch(row1)`, `switch(col2)`, `switch(all)`,
    or `hold`."""

    def __init__(self, target=None):
        self.target = target
        self.name = "hold" if target is None else f"switch({target})"

    @classmethod
    def parse(cls, text):
        text = str(text).strip()
        if text == "hold":
            return cls()
        return cls(text[len("switch("):-1])

    def junctions(self):
        if self.target is None:
            return ()
        if self.target == "all":
            return JUNCTIONS
        if self.target.startswith("row"):
            return ROWS[int(self.target[3:])]
        if self.target.startswith("col"):
            return COLUMNS[int(self.target[3:])]
        return (self.target,)

    def cost(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, TrafficAction) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    __repr__ = __str__


HOLD = TrafficAction()
ACTIONS = tuple([TrafficAction(j) for j in JUNCTIONS] + [TrafficAction(f"row{k}") for k in range(GRID)]
                + [TrafficAction(f"col{k}") for k in range(GRID)] + [TrafficAction("all"), HOLD])


class TrafficState:
    """The decisions so far and what the grid reads after them: the time, the vehicles arrived,
    on the road and still to depart, the vehicle-seconds spent, and which green each junction
    shows. Identity is the path, since SUMO replays exactly."""

    def __init__(self, decisions, time, arrived, running, pending, travel, greens, target=0,
                 horizon=0, total=0, depth=0):
        self.decisions = tuple(decisions)
        self.time, self.arrived, self.running, self.pending = time, arrived, running, pending
        self.travel, self.greens = travel, tuple(greens)
        self.target, self.horizon, self.total, self.depth = target, horizon, total, depth
        literals = [f"green({j}, {g})" for j, g in zip(JUNCTIONS, self.greens)]
        literals += [f"decision({len(self.decisions)})", f"arrived({arrived})", f"running({running})",
                     f"pending({pending})", f"travel({travel // 500 * 500})"]
        self.literals = frozenset(literals)

    @property
    def left(self):
        return self.running + self.pending

    def __eq__(self, other):
        return isinstance(other, TrafficState) and self.decisions == other.decisions

    def __hash__(self):
        return hash(self.decisions)

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        greens = "".join(self.greens)
        return (f"t={self.time:.0f}s: {self.arrived} of {self.total} arrived, {self.running} on the road, "
                f"{self.pending} to come; {self.travel:.0f} vehicle-seconds of {self.target}; greens {greens}")

    def __repr__(self):
        return f"<TrafficState(t={self.time:.0f}, arrived={self.arrived}, travel={self.travel:.0f})>"


class TrafficEnv(Environment):
    """Get everyone through before the horizon with the total travel time under the target."""

    def __init__(self):
        super().__init__("traffic")
        self.instance = None
        self.index = None
        self.state = None
        self.state_history = []
        self.witness = None
        self.witness_expansions = None
        self.__loaded__ = None
        self.__greens__ = {}
        self.__phases__ = {}
        self.__reading__ = None

    # ------------------------------------------------------------------ instances

    def set_index(self, index):
        if not 0 <= index < len(MORNINGS):
            raise IndexError(f"Invalid index: {index}. There are {len(MORNINGS)} mornings, so the "
                             f"index must be 0-{len(MORNINGS) - 1}.")
        self.set_instance(MORNINGS[index])
        self.index = index

    def set_instance(self, instance):
        """Select a morning: `seed`, `vehicles`, `spacing` (seconds between departures),
        `horizon` (seconds) and `target` (vehicle-seconds)."""
        for key in ("seed", "vehicles", "spacing", "horizon", "target"):
            if key not in instance:
                raise ValueError(f"a morning needs `{key}`")
        self.instance = {"seed": int(instance["seed"]), "vehicles": int(instance["vehicles"]),
                         "spacing": float(instance["spacing"]), "horizon": int(instance["horizon"]),
                         "target": int(instance["target"])}
        self.index = None
        self.witness = self.witness_expansions = None
        self.close()

    def generate_instance(self, seed=None, vehicles=None, spacing=None, horizon=None, attempts=40):
        """Draw a morning, select it, and return it as the dict `set_instance` takes.

        The trips are seeded; the vehicles, spacing and horizon come from `VEHICLES`,
        `SPACINGS` and `HORIZONS` unless given. Fixed cycles of every `CYCLES` decisions are
        then run, and holding the signals as they start; the target is the least travel any of
        them clears the grid with before the horizon, the draw is thrown back when none does,
        and that cycle is the witness.
        """
        random_, _ = rng(seed)
        found = {}

        def draw(attempt):
            return {"seed": random_.randrange(1, 2 ** 30), "vehicles": vehicles or random_.choice(VEHICLES),
                    "spacing": spacing or random_.choice(SPACINGS),
                    "horizon": horizon or random_.choice(HORIZONS), "target": 0}

        def accept(instance):
            self.set_instance(instance)
            self.instance["target"] = 10 ** 9
            outcomes = []
            decisions = instance["horizon"] // DECISION
            plans = [[TrafficAction("all") if (k + 1) % cycle == 0 else HOLD for k in range(decisions)]
                     for cycle in CYCLES] + [[HOLD] * decisions]
            for plan in plans:
                trace = self.simulate(plan)
                done = next((s for s in trace if s.left == 0), None)
                outcomes.append((done.travel, plan[:trace.index(done)]) if done else None)
            finished = [o for o in outcomes if o is not None]
            if not finished:
                return False
            best = min(finished, key=lambda o: o[0])
            if outcomes[-1] is not None and outcomes[-1][0] <= best[0]:
                return False                    # holding the signals is already best
            instance["target"] = int(best[0])
            found["plan"], found["measured"] = best[1], len(plans)
            return True

        instance = draw_until(draw, accept, attempts, "traffic morning")
        self.set_instance(instance)
        self.witness, self.witness_expansions = found["plan"], found["measured"]
        return instance

    # ------------------------------------------------------------------ the simulator

    def _start(self):
        ls = _libsumo()
        self.close()
        ls.start(["sumo", "-n", network(), "-r", trips(self.instance), "--seed", "1",
                  "--no-step-log", "true", "--no-warnings", "true", "--time-to-teleport", "-1"])
        self.__phases__ = {}
        for j in JUNCTIONS:
            phases = ls.trafficlight.getAllProgramLogics(j)[0].phases
            self.__phases__[j] = (phases[0].state, phases[1].state, phases[2].state, phases[3].state)
            ls.trafficlight.setRedYellowGreenState(j, phases[0].state)
        for j in ls.trafficlight.getIDList():
            if j not in self.__phases__:
                ls.trafficlight.setRedYellowGreenState(j, "G")
        self.__greens__ = {j: 0 for j in JUNCTIONS}
        self.__reading__ = {"arrived": 0, "travel": 0}
        self.__loaded__ = ()

    def _run(self, seconds):
        ls = _libsumo()
        for _ in range(seconds):
            ls.simulationStep()
            self.__reading__["arrived"] += ls.simulation.getArrivedNumber()
            self.__reading__["travel"] += ls.vehicle.getIDCount()

    def _apply(self, action):
        ls = _libsumo()
        switching = action.junctions()
        for j in switching:
            self.__greens__[j] = 1 - self.__greens__[j]
            ls.trafficlight.setRedYellowGreenState(j, self.__phases__[j][1 + 2 * (1 - self.__greens__[j])])
        self._run(AMBER if switching else 0)
        for j in switching:
            ls.trafficlight.setRedYellowGreenState(j, self.__phases__[j][2 * self.__greens__[j]])
        self._run(DECISION - (AMBER if switching else 0))

    def _load(self, decisions):
        if self.__loaded__ != decisions:
            if self.__loaded__ is None or decisions[:len(self.__loaded__)] != self.__loaded__:
                self._start()
            for action in decisions[len(self.__loaded__):]:
                self._apply(action)
            self.__loaded__ = decisions

    def _read(self, decisions, depth):
        ls = _libsumo()
        running = ls.vehicle.getIDCount()
        pending = max(0, self.instance["vehicles"] - self.__reading__["arrived"] - running)
        return TrafficState(decisions, ls.simulation.getTime(), self.__reading__["arrived"], running,
                            pending, self.__reading__["travel"], [("A", "B")[self.__greens__[j]] for j in JUNCTIONS],
                            target=self.instance["target"], horizon=self.instance["horizon"],
                            total=self.instance["vehicles"], depth=depth)

    def close(self):
        if self.__loaded__ is not None:
            try:
                _libsumo().close()
            except Exception:
                pass
        self.__loaded__ = None

    # ------------------------------------------------------------------- contract

    def reset(self):
        if self.instance is None:
            self.set_index(0)
        self._start()
        self.state = self._read((), 0)
        self.state_history = [self.state]
        return self.state, {"morning": self.index, "vehicles": self.instance["vehicles"],
                            "horizon": self.instance["horizon"], "target": self.instance["target"],
                            "generated": self.index is None}

    def is_goal(self, state):
        return state.left == 0 and state.time <= self.instance["horizon"] and state.travel <= self.instance["target"]

    def is_terminal(self, state):
        return not self.is_goal(state) and (state.time >= self.instance["horizon"] or state.left == 0)

    def get_actions(self, state=None):
        return list(ACTIONS)

    def successors(self, state):
        if self.is_goal(state) or self.is_terminal(state):
            return []
        children = []
        for action in ACTIONS:
            self._load(state.decisions)
            self._apply(action)
            self.__loaded__ = state.decisions + (action,)
            children.append((action, self._read(self.__loaded__, state.depth + 1)))
        return children

    def __advance__(self, state, action):
        if self.is_goal(state) or self.is_terminal(state):
            return state
        if not isinstance(action, TrafficAction):
            action = TrafficAction.parse(action)
        if action not in ACTIONS:
            return state
        self._load(state.decisions)
        self._apply(action)
        self.__loaded__ = state.decisions + (action,)
        return self._read(self.__loaded__, state.depth + 1)

    def simulate(self, plan):
        state, _ = self.reset()
        trace = [state]
        for action in plan:
            trace.append(self.__advance__(trace[-1], action))
        return trace

    def step(self, action):
        if self.state is None:
            raise ValueError("call reset() first")
        before = self.state.arrived
        self.state = self.__advance__(self.state, action)
        self.state_history.append(self.state)
        return self.state, self.state.arrived - before

    def render(self):
        lines = [f"step {k}: {state}" for k, state in enumerate(self.state_history)]
        print("\n".join(lines))
        return lines


#: The bundled mornings: `generate_instance(seed)` for the seed beside each, embedded as the
#: plain data `set_instance` takes, with the plan each was accepted on in
#: `tests/data/traffic_solutions.json`.
MORNINGS = (
    # seed 9000; 200 vehicles every 2.0 s, horizon 600 s, 35-step plan
    {"seed": 369326630, "vehicles": 200, "spacing": 2.0, "horizon": 600, "target": 15397},
    # seed 9001; 200 vehicles every 1.5 s, horizon 600 s, 30-step plan
    {"seed": 39830315, "vehicles": 200, "spacing": 1.5, "horizon": 600, "target": 18089},
    # seed 9002; 200 vehicles every 2.0 s, horizon 600 s, 35-step plan
    {"seed": 295778760, "vehicles": 200, "spacing": 2.0, "horizon": 600, "target": 15660},
    # seed 9003; 120 vehicles every 1.0 s, horizon 900 s, 17-step plan
    {"seed": 610606394, "vehicles": 120, "spacing": 1.0, "horizon": 900, "target": 11385},
    # seed 9004; 240 vehicles every 1.5 s, horizon 750 s, 35-step plan
    {"seed": 1063345461, "vehicles": 240, "spacing": 1.5, "horizon": 750, "target": 22114},
    # seed 9005; 240 vehicles every 1.0 s, horizon 750 s, 29-step plan
    {"seed": 376437504, "vehicles": 240, "spacing": 1.0, "horizon": 750, "target": 22940},
    # seed 9006; 120 vehicles every 1.5 s, horizon 750 s, 22-step plan
    {"seed": 1012944643, "vehicles": 120, "spacing": 1.5, "horizon": 750, "target": 9367},
    # seed 9007; 120 vehicles every 1.0 s, horizon 900 s, 19-step plan
    {"seed": 359003885, "vehicles": 120, "spacing": 1.0, "horizon": 900, "target": 9930},
    # seed 9008; 240 vehicles every 1.5 s, horizon 750 s, 33-step plan
    {"seed": 648715749, "vehicles": 240, "spacing": 1.5, "horizon": 750, "target": 21317},
    # seed 9009; 160 vehicles every 1.5 s, horizon 900 s, 27-step plan
    {"seed": 311388747, "vehicles": 160, "spacing": 1.5, "horizon": 900, "target": 13830},
    # seed 9010; 160 vehicles every 2.0 s, horizon 900 s, 30-step plan
    {"seed": 121293936, "vehicles": 160, "spacing": 2.0, "horizon": 900, "target": 12862},
    # seed 9011; 200 vehicles every 1.5 s, horizon 750 s, 31-step plan
    {"seed": 573281517, "vehicles": 200, "spacing": 1.5, "horizon": 750, "target": 19722},
    # seed 9012; 200 vehicles every 1.5 s, horizon 750 s, 29-step plan
    {"seed": 397153799, "vehicles": 200, "spacing": 1.5, "horizon": 750, "target": 15296},
    # seed 9013; 200 vehicles every 2.0 s, horizon 750 s, 33-step plan
    {"seed": 1038183361, "vehicles": 200, "spacing": 2.0, "horizon": 750, "target": 15367},
    # seed 9014; 240 vehicles every 2.0 s, horizon 750 s, 41-step plan
    {"seed": 344065333, "vehicles": 240, "spacing": 2.0, "horizon": 750, "target": 19857},
    # seed 9015; 120 vehicles every 1.5 s, horizon 600 s, 21-step plan
    {"seed": 691710551, "vehicles": 120, "spacing": 1.5, "horizon": 600, "target": 9446},
    # seed 9016; 160 vehicles every 2.0 s, horizon 750 s, 31-step plan
    {"seed": 468718473, "vehicles": 160, "spacing": 2.0, "horizon": 750, "target": 12503},
    # seed 9017; 160 vehicles every 1.5 s, horizon 900 s, 25-step plan
    {"seed": 996360513, "vehicles": 160, "spacing": 1.5, "horizon": 900, "target": 12397},
    # seed 9018; 240 vehicles every 2.0 s, horizon 900 s, 41-step plan
    {"seed": 284830276, "vehicles": 240, "spacing": 2.0, "horizon": 900, "target": 20183},
    # seed 9019; 240 vehicles every 1.0 s, horizon 600 s, 29-step plan
    {"seed": 968392874, "vehicles": 240, "spacing": 1.0, "horizon": 600, "target": 23798},
    # seed 9020; 200 vehicles every 1.0 s, horizon 900 s, 24-step plan
    {"seed": 975312175, "vehicles": 200, "spacing": 1.0, "horizon": 900, "target": 18894},
    # seed 9021; 160 vehicles every 1.5 s, horizon 750 s, 25-step plan
    {"seed": 85217924, "vehicles": 160, "spacing": 1.5, "horizon": 750, "target": 12835},
    # seed 9022; 200 vehicles every 1.5 s, horizon 600 s, 29-step plan
    {"seed": 139603094, "vehicles": 200, "spacing": 1.5, "horizon": 600, "target": 15901},
    # seed 9023; 120 vehicles every 2.0 s, horizon 900 s, 25-step plan
    {"seed": 1036363007, "vehicles": 120, "spacing": 2.0, "horizon": 900, "target": 9300},
    # seed 9024; 160 vehicles every 1.0 s, horizon 900 s, 23-step plan
    {"seed": 20831278, "vehicles": 160, "spacing": 1.0, "horizon": 900, "target": 16255},
    # seed 9025; 120 vehicles every 1.0 s, horizon 600 s, 17-step plan
    {"seed": 219434381, "vehicles": 120, "spacing": 1.0, "horizon": 600, "target": 9739},
    # seed 9026; 200 vehicles every 1.5 s, horizon 750 s, 29-step plan
    {"seed": 942561203, "vehicles": 200, "spacing": 1.5, "horizon": 750, "target": 16612},
    # seed 9027; 240 vehicles every 1.0 s, horizon 900 s, 29-step plan
    {"seed": 338707991, "vehicles": 240, "spacing": 1.0, "horizon": 900, "target": 24377},
    # seed 9028; 200 vehicles every 1.0 s, horizon 900 s, 23-step plan
    {"seed": 422354421, "vehicles": 200, "spacing": 1.0, "horizon": 900, "target": 17118},
    # seed 9029; 240 vehicles every 1.5 s, horizon 750 s, 33-step plan
    {"seed": 457078268, "vehicles": 240, "spacing": 1.5, "horizon": 750, "target": 19937},
    # seed 9030; 240 vehicles every 1.5 s, horizon 750 s, 35-step plan
    {"seed": 321373061, "vehicles": 240, "spacing": 1.5, "horizon": 750, "target": 20328},
    # seed 9031; 200 vehicles every 2.0 s, horizon 750 s, 33-step plan
    {"seed": 1059605557, "vehicles": 200, "spacing": 2.0, "horizon": 750, "target": 15515},
    # seed 9032; 240 vehicles every 1.0 s, horizon 900 s, 27-step plan
    {"seed": 992213298, "vehicles": 240, "spacing": 1.0, "horizon": 900, "target": 20832},
    # seed 9033; 200 vehicles every 1.0 s, horizon 750 s, 25-step plan
    {"seed": 1053202513, "vehicles": 200, "spacing": 1.0, "horizon": 750, "target": 18489},
    # seed 9034; 120 vehicles every 2.0 s, horizon 750 s, 27-step plan
    {"seed": 149446519, "vehicles": 120, "spacing": 2.0, "horizon": 750, "target": 8526},
    # seed 9035; 160 vehicles every 1.5 s, horizon 750 s, 24-step plan
    {"seed": 271017444, "vehicles": 160, "spacing": 1.5, "horizon": 750, "target": 14082},
    # seed 9036; 200 vehicles every 2.0 s, horizon 600 s, 35-step plan
    {"seed": 1049550267, "vehicles": 200, "spacing": 2.0, "horizon": 600, "target": 16206},
    # seed 9037; 200 vehicles every 1.5 s, horizon 600 s, 29-step plan
    {"seed": 961625233, "vehicles": 200, "spacing": 1.5, "horizon": 600, "target": 17678},
    # seed 9038; 160 vehicles every 1.0 s, horizon 900 s, 21-step plan
    {"seed": 25569470, "vehicles": 160, "spacing": 1.0, "horizon": 900, "target": 14566},
    # seed 9039; 160 vehicles every 1.5 s, horizon 750 s, 23-step plan
    {"seed": 943283422, "vehicles": 160, "spacing": 1.5, "horizon": 750, "target": 13965},
    # seed 9040; 120 vehicles every 2.0 s, horizon 750 s, 24-step plan
    {"seed": 398697691, "vehicles": 120, "spacing": 2.0, "horizon": 750, "target": 9086},
    # seed 9041; 160 vehicles every 1.5 s, horizon 900 s, 26-step plan
    {"seed": 815210155, "vehicles": 160, "spacing": 1.5, "horizon": 900, "target": 14132},
    # seed 9042; 200 vehicles every 1.5 s, horizon 900 s, 29-step plan
    {"seed": 271295058, "vehicles": 200, "spacing": 1.5, "horizon": 900, "target": 17114},
    # seed 9043; 200 vehicles every 2.0 s, horizon 750 s, 37-step plan
    {"seed": 273441536, "vehicles": 200, "spacing": 2.0, "horizon": 750, "target": 16664},
    # seed 9044; 240 vehicles every 2.0 s, horizon 900 s, 41-step plan
    {"seed": 696776955, "vehicles": 240, "spacing": 2.0, "horizon": 900, "target": 18528},
    # seed 9045; 240 vehicles every 1.5 s, horizon 750 s, 33-step plan
    {"seed": 803278543, "vehicles": 240, "spacing": 1.5, "horizon": 750, "target": 21058},
    # seed 9046; 160 vehicles every 1.5 s, horizon 750 s, 25-step plan
    {"seed": 578882619, "vehicles": 160, "spacing": 1.5, "horizon": 750, "target": 12924},
    # seed 9047; 200 vehicles every 1.0 s, horizon 900 s, 25-step plan
    {"seed": 513292069, "vehicles": 200, "spacing": 1.0, "horizon": 900, "target": 19436},
    # seed 9048; 200 vehicles every 1.0 s, horizon 600 s, 23-step plan
    {"seed": 295537948, "vehicles": 200, "spacing": 1.0, "horizon": 600, "target": 18524},
    # seed 9049; 200 vehicles every 2.0 s, horizon 900 s, 35-step plan
    {"seed": 684449591, "vehicles": 200, "spacing": 2.0, "horizon": 900, "target": 15199},
    # seed 9050; 200 vehicles every 2.0 s, horizon 900 s, 33-step plan
    {"seed": 528476589, "vehicles": 200, "spacing": 2.0, "horizon": 900, "target": 15959},
    # seed 9051; 120 vehicles every 1.5 s, horizon 750 s, 19-step plan
    {"seed": 196542150, "vehicles": 120, "spacing": 1.5, "horizon": 750, "target": 10147},
    # seed 9052; 120 vehicles every 1.5 s, horizon 900 s, 21-step plan
    {"seed": 17052900, "vehicles": 120, "spacing": 1.5, "horizon": 900, "target": 9337},
    # seed 9053; 240 vehicles every 1.0 s, horizon 750 s, 27-step plan
    {"seed": 389593505, "vehicles": 240, "spacing": 1.0, "horizon": 750, "target": 22062},
    # seed 9054; 240 vehicles every 1.5 s, horizon 600 s, 31-step plan
    {"seed": 767538779, "vehicles": 240, "spacing": 1.5, "horizon": 600, "target": 19290},
    # seed 9055; 160 vehicles every 1.0 s, horizon 600 s, 23-step plan
    {"seed": 944213926, "vehicles": 160, "spacing": 1.0, "horizon": 600, "target": 15259},
    # seed 9056; 120 vehicles every 1.5 s, horizon 600 s, 20-step plan
    {"seed": 240960256, "vehicles": 120, "spacing": 1.5, "horizon": 600, "target": 9374},
    # seed 9057; 120 vehicles every 2.0 s, horizon 900 s, 25-step plan
    {"seed": 1015081419, "vehicles": 120, "spacing": 2.0, "horizon": 900, "target": 9472},
    # seed 9058; 240 vehicles every 2.0 s, horizon 900 s, 41-step plan
    {"seed": 599193888, "vehicles": 240, "spacing": 2.0, "horizon": 900, "target": 18722},
    # seed 9059; 160 vehicles every 1.0 s, horizon 750 s, 21-step plan
    {"seed": 682590923, "vehicles": 160, "spacing": 1.0, "horizon": 750, "target": 14722},
    # seed 9060; 240 vehicles every 1.0 s, horizon 600 s, 27-step plan
    {"seed": 512873994, "vehicles": 240, "spacing": 1.0, "horizon": 600, "target": 24232},
    # seed 9061; 120 vehicles every 1.5 s, horizon 600 s, 21-step plan
    {"seed": 584200868, "vehicles": 120, "spacing": 1.5, "horizon": 600, "target": 9984},
    # seed 9062; 200 vehicles every 1.5 s, horizon 750 s, 31-step plan
    {"seed": 652146434, "vehicles": 200, "spacing": 1.5, "horizon": 750, "target": 16596},
    # seed 9063; 120 vehicles every 1.0 s, horizon 750 s, 18-step plan
    {"seed": 400279027, "vehicles": 120, "spacing": 1.0, "horizon": 750, "target": 11191},
    # seed 9064; 160 vehicles every 1.0 s, horizon 900 s, 19-step plan
    {"seed": 482156899, "vehicles": 160, "spacing": 1.0, "horizon": 900, "target": 14496},
    # seed 9065; 240 vehicles every 1.5 s, horizon 750 s, 33-step plan
    {"seed": 625277176, "vehicles": 240, "spacing": 1.5, "horizon": 750, "target": 18880},
    # seed 9066; 160 vehicles every 1.0 s, horizon 900 s, 23-step plan
    {"seed": 408229125, "vehicles": 160, "spacing": 1.0, "horizon": 900, "target": 15049},
    # seed 9067; 120 vehicles every 1.0 s, horizon 600 s, 19-step plan
    {"seed": 222515115, "vehicles": 120, "spacing": 1.0, "horizon": 600, "target": 10101},
    # seed 9068; 240 vehicles every 1.0 s, horizon 600 s, 28-step plan
    {"seed": 51399698, "vehicles": 240, "spacing": 1.0, "horizon": 600, "target": 22616},
    # seed 9069; 160 vehicles every 2.0 s, horizon 600 s, 31-step plan
    {"seed": 98048014, "vehicles": 160, "spacing": 2.0, "horizon": 600, "target": 12649},
    # seed 9070; 120 vehicles every 1.5 s, horizon 600 s, 20-step plan
    {"seed": 426812545, "vehicles": 120, "spacing": 1.5, "horizon": 600, "target": 10072},
    # seed 9071; 160 vehicles every 1.5 s, horizon 750 s, 25-step plan
    {"seed": 1051066146, "vehicles": 160, "spacing": 1.5, "horizon": 750, "target": 13307},
    # seed 9072; 120 vehicles every 2.0 s, horizon 600 s, 25-step plan
    {"seed": 172985337, "vehicles": 120, "spacing": 2.0, "horizon": 600, "target": 9714},
    # seed 9073; 200 vehicles every 1.0 s, horizon 750 s, 26-step plan
    {"seed": 829910765, "vehicles": 200, "spacing": 1.0, "horizon": 750, "target": 19755},
    # seed 9074; 160 vehicles every 1.0 s, horizon 750 s, 19-step plan
    {"seed": 872933550, "vehicles": 160, "spacing": 1.0, "horizon": 750, "target": 13881},
    # seed 9075; 120 vehicles every 2.0 s, horizon 900 s, 26-step plan
    {"seed": 541486175, "vehicles": 120, "spacing": 2.0, "horizon": 900, "target": 9378},
    # seed 9076; 200 vehicles every 1.0 s, horizon 600 s, 23-step plan
    {"seed": 813098918, "vehicles": 200, "spacing": 1.0, "horizon": 600, "target": 17170},
    # seed 9077; 160 vehicles every 1.0 s, horizon 600 s, 23-step plan
    {"seed": 116243095, "vehicles": 160, "spacing": 1.0, "horizon": 600, "target": 15869},
    # seed 9078; 160 vehicles every 1.5 s, horizon 600 s, 24-step plan
    {"seed": 326699894, "vehicles": 160, "spacing": 1.5, "horizon": 600, "target": 12638},
    # seed 9079; 200 vehicles every 1.5 s, horizon 750 s, 29-step plan
    {"seed": 146580488, "vehicles": 200, "spacing": 1.5, "horizon": 750, "target": 18842},
    # seed 9080; 200 vehicles every 1.0 s, horizon 900 s, 27-step plan
    {"seed": 508828721, "vehicles": 200, "spacing": 1.0, "horizon": 900, "target": 21937},
    # seed 9081; 200 vehicles every 1.0 s, horizon 600 s, 25-step plan
    {"seed": 526468421, "vehicles": 200, "spacing": 1.0, "horizon": 600, "target": 19136},
    # seed 9082; 160 vehicles every 1.0 s, horizon 600 s, 22-step plan
    {"seed": 540643337, "vehicles": 160, "spacing": 1.0, "horizon": 600, "target": 13198},
    # seed 9083; 240 vehicles every 1.5 s, horizon 900 s, 35-step plan
    {"seed": 399316842, "vehicles": 240, "spacing": 1.5, "horizon": 900, "target": 20693},
    # seed 9084; 160 vehicles every 1.0 s, horizon 600 s, 20-step plan
    {"seed": 1057066203, "vehicles": 160, "spacing": 1.0, "horizon": 600, "target": 13555},
    # seed 9085; 240 vehicles every 1.5 s, horizon 900 s, 36-step plan
    {"seed": 485457192, "vehicles": 240, "spacing": 1.5, "horizon": 900, "target": 22453},
    # seed 9086; 120 vehicles every 2.0 s, horizon 900 s, 28-step plan
    {"seed": 166489957, "vehicles": 120, "spacing": 2.0, "horizon": 900, "target": 9107},
    # seed 9087; 200 vehicles every 2.0 s, horizon 600 s, 36-step plan
    {"seed": 630841644, "vehicles": 200, "spacing": 2.0, "horizon": 600, "target": 15855},
    # seed 9088; 120 vehicles every 2.0 s, horizon 900 s, 25-step plan
    {"seed": 651011267, "vehicles": 120, "spacing": 2.0, "horizon": 900, "target": 9212},
    # seed 9089; 240 vehicles every 1.0 s, horizon 600 s, 29-step plan
    {"seed": 854362708, "vehicles": 240, "spacing": 1.0, "horizon": 600, "target": 24788},
    # seed 9090; 120 vehicles every 1.5 s, horizon 600 s, 21-step plan
    {"seed": 312148943, "vehicles": 120, "spacing": 1.5, "horizon": 600, "target": 9524},
    # seed 9091; 200 vehicles every 2.0 s, horizon 750 s, 33-step plan
    {"seed": 767218348, "vehicles": 200, "spacing": 2.0, "horizon": 750, "target": 15829},
    # seed 9092; 240 vehicles every 1.0 s, horizon 900 s, 26-step plan
    {"seed": 88075309, "vehicles": 240, "spacing": 1.0, "horizon": 900, "target": 23505},
    # seed 9093; 120 vehicles every 2.0 s, horizon 750 s, 23-step plan
    {"seed": 583838555, "vehicles": 120, "spacing": 2.0, "horizon": 750, "target": 8948},
    # seed 9094; 160 vehicles every 2.0 s, horizon 600 s, 29-step plan
    {"seed": 707630003, "vehicles": 160, "spacing": 2.0, "horizon": 600, "target": 12117},
    # seed 9095; 240 vehicles every 1.0 s, horizon 900 s, 28-step plan
    {"seed": 411955586, "vehicles": 240, "spacing": 1.0, "horizon": 900, "target": 24462},
    # seed 9096; 200 vehicles every 1.5 s, horizon 600 s, 28-step plan
    {"seed": 68672395, "vehicles": 200, "spacing": 1.5, "horizon": 600, "target": 17989},
    # seed 9097; 160 vehicles every 1.5 s, horizon 750 s, 25-step plan
    {"seed": 332188830, "vehicles": 160, "spacing": 1.5, "horizon": 750, "target": 12848},
    # seed 9098; 200 vehicles every 1.0 s, horizon 600 s, 22-step plan
    {"seed": 187907080, "vehicles": 200, "spacing": 1.0, "horizon": 600, "target": 17966},
    # seed 9099; 240 vehicles every 2.0 s, horizon 900 s, 40-step plan
    {"seed": 26071750, "vehicles": 240, "spacing": 2.0, "horizon": 900, "target": 19474},
)
