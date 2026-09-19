"""Air traffic on BlueSky: a few aircraft crossing a sector for exits on the far side, a
controller's instruction a minute, and everyone out before the horizon with no loss of
separation on the way.

BlueSky (https://github.com/TUDelft-CNS-ATM/bluesky, MIT; Hoekstra and Ellerbroek 2016, "BlueSky
ATC simulator project: an open data and open source approach", ICRAT) is TU Delft's open air
traffic simulator, flying aircraft on the OpenAP performance model (LGPL-3.0) over its own
navigation data (GPL-3.0). All three are dependencies installed from PyPI; nothing of them is
included here, and `scripts/install_bluesky.sh` says how to install them round a broken
requirement of the simulator's.

The decisions are a controller's: a thirty-degree turn left or right for one aircraft, a
direct-to that puts it back on course for its exit, or nothing, and a minute of flying follows
each. The turn rates, speeds and the drift of a heading-held aircraft are BlueSky's, so where
everyone is a minute later is only known by flying it.

## Trajectory constraints as a goal

The separation standard (five nautical miles or a thousand feet, always) is `is_terminal`: a
loss of separation is a dead end, so any plan returned keeps everyone apart at every second.
The sum of the aircraft's exit times is accumulated in the state, and the goal is every
aircraft at its exit before the horizon with that sum under the target, which is what a
scripted right-turn controller achieves with a tenth in hand, so the planner is asked to
resolve the crossing without dawdling.

## Determinism

BlueSky flies the same instructions to the same positions, and its state is a set of module
globals rather than an object, so a state is its instructions and every expansion replays the
sector from the aircraft's entry. A minute of four aircraft takes about fifty milliseconds.
"""
import math

from planiverse.environments.base import Environment
from planiverse.environments.generation import draw_until, rng

#: Seconds of flying per decision, the turn an instruction makes, and the separation standard.
DECISION, TURN = 60, 30
SEPARATION_NM, SEPARATION_FT = 5.0, 1000.0
#: An aircraft is out when within this of its exit.
EXIT_NM = 2.0
#: The sector: its centre, the ring the aircraft enter on, and the horizon in decisions.
CENTRE = (52.0, 4.5)
RING_NM = (16.0, 20.0)
SPEEDS = (230, 250, 270)
LEVEL = 300
COUNTS = (3, 4)
HORIZONS = (12, 15)
#: How far ahead the scripted controller looks for a conflict, and how close counts as one.
LOOKAHEAD, CONFLICT_NM = 180, 6.0
SLACK = 0.1

_bluesky = {}


def _bs():
    import bluesky
    if "ready" not in _bluesky:
        bluesky.init(mode="sim")
        _bluesky["ready"] = True
    return bluesky


def offset(lat, lon):
    """Nautical miles east and north of the sector's centre (flat, fine at this size)."""
    return ((lon - CENTRE[1]) * 60 * math.cos(math.radians(CENTRE[0])), (lat - CENTRE[0]) * 60)


def place(east, north):
    """The latitude and longitude of a point given in nautical miles from the centre."""
    return (CENTRE[0] + north / 60, CENTRE[1] + east / (60 * math.cos(math.radians(CENTRE[0]))))


def distance_nm(a, b):
    (x1, y1), (x2, y2) = offset(*a), offset(*b)
    return math.hypot(x1 - x2, y1 - y2)


class AirspaceAction:
    """`turn(id, left|right)`, `direct(id)` or `hold`."""

    def __init__(self, verb, aircraft=None, way=None):
        self.verb, self.aircraft, self.way = verb, aircraft, way
        if verb == "hold":
            self.name = "hold"
        elif verb == "turn":
            self.name = f"turn({aircraft}, {way})"
        elif verb == "direct":
            self.name = f"direct({aircraft})"
        else:
            raise ValueError(f"unknown verb: {verb!r}")

    @classmethod
    def parse(cls, text):
        text = str(text).strip()
        if text == "hold":
            return cls("hold")
        verb, inside = text[:-1].split("(")
        parts = [p.strip() for p in inside.split(",")]
        return cls(verb, parts[0], parts[1] if len(parts) > 1 else None)

    def cost(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, AirspaceAction) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    __repr__ = __str__


HOLD = AirspaceAction("hold")


class AirspaceState:
    """The instructions so far and the sector after them: each aircraft's position, level,
    heading, distance to its exit and whether it is out, the closest any pair came, the sum of
    exit times, and whether separation was lost. Identity is the path."""

    def __init__(self, instructions, time, aircraft, closest, exit_sum, lost, target=0,
                 horizon=0, depth=0):
        self.instructions = tuple(instructions)
        self.time = time
        self.aircraft = tuple(aircraft)      # (id, lat, lon, feet, heading, to_go, out, on_course)
        self.closest, self.exit_sum, self.lost = closest, exit_sum, lost
        self.target, self.horizon, self.depth = target, horizon, depth
        literals = [f"decision({len(self.instructions)})", f"closest({int(closest)})"]
        for name, _, _, feet, heading, to_go, out, on_course in self.aircraft:
            if out:
                literals.append(f"out({name})")
                continue
            literals += [f"flying({name})", f"heading({name}, {int(heading) // 30 * 30})",
                         f"to_go({name}, {int(to_go) // 4 * 4})", f"level({name}, {int(feet) // 1000})"]
            if on_course:
                literals.append(f"on_course({name})")
        if lost:
            literals.append("separation_lost")
        self.literals = frozenset(literals)

    @property
    def flying(self):
        return [a for a in self.aircraft if not a[6]]

    def __eq__(self, other):
        return isinstance(other, AirspaceState) and self.instructions == other.instructions

    def __hash__(self):
        return hash(self.instructions)

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        parts = [f"t={self.time:.0f}s, closest {self.closest:.1f} nm" + (", separation lost" if self.lost else "")]
        for name, lat, lon, feet, heading, to_go, out, on_course in self.aircraft:
            parts.append(f"{name} out" if out else
                         f"{name} hdg {heading:.0f} FL{feet / 100:.0f} {to_go:.1f} nm to go" + (" on course" if on_course else ""))
        return "; ".join(parts)

    def __repr__(self):
        return f"<AirspaceState(t={self.time:.0f}, flying={len(self.flying)}, closest={self.closest:.1f})>"


class AirspaceEnv(Environment):
    """Everyone out before the horizon, never closer than the separation standard."""

    def __init__(self):
        super().__init__("airspace")
        self.instance = None
        self.index = None
        self.state = None
        self.state_history = []
        self.witness = None
        self.witness_expansions = None
        self.__loaded__ = None
        self.__flight__ = None

    # ------------------------------------------------------------------ instances

    def set_index(self, index):
        if not 0 <= index < len(SECTORS):
            raise IndexError(f"Invalid index: {index}. There are {len(SECTORS)} sectors, so the "
                             f"index must be 0-{len(SECTORS) - 1}.")
        self.set_instance(SECTORS[index])
        self.index = index

    def set_instance(self, instance):
        """Select a sector: `aircraft` (id, lat, lon, heading, level in feet, speed in knots,
        exit lat, exit lon), `horizon` (decisions) and `target` (sum of exit times, seconds)."""
        for key in ("aircraft", "horizon", "target"):
            if key not in instance:
                raise ValueError(f"a sector needs `{key}`")
        self.instance = {"aircraft": [[str(a[0])] + [float(v) for v in a[1:]] for a in instance["aircraft"]],
                         "horizon": int(instance["horizon"]), "target": int(instance["target"])}
        self.index = None
        self.witness = self.witness_expansions = None
        self.__loaded__ = None

    def generate_instance(self, seed=None, count=None, horizon=None, attempts=60):
        """Draw a sector, select it, and return it as the dict `set_instance` takes.

        `count` aircraft (three or four unless given) enter on a ring round the centre at
        seeded bearings, speeds and radii, each bound for a point across the sector, so their
        tracks cross near the middle. A draw is kept when flying straight loses separation
        (otherwise there is nothing to resolve) and a scripted controller, turning the lower
        aircraft of a predicted conflict right and putting it back on course afterwards, gets
        everyone out; the target is its sum of exit times with `SLACK` in hand, and its
        instructions are the witness.
        """
        random_, _ = rng(seed)
        found = {}

        def draw(attempt):
            n = count or random_.choice(COUNTS)
            bearings = sorted(random_.uniform(0, 360) for _ in range(n))
            if any((bearings[(k + 1) % n] - bearings[k]) % 360 < 25 for k in range(n)):
                return None
            aircraft = []
            for k, bearing in enumerate(bearings):
                radius = random_.uniform(*RING_NM)
                east, north = radius * math.sin(math.radians(bearing)), radius * math.cos(math.radians(bearing))
                lat, lon = place(east, north)
                across = bearing + 180 + random_.uniform(-20, 20)
                exit_east, exit_north = RING_NM[1] * math.sin(math.radians(across)), RING_NM[1] * math.cos(math.radians(across))
                exit_lat, exit_lon = place(exit_east, exit_north)
                heading = math.degrees(math.atan2(exit_east - east, exit_north - north)) % 360
                aircraft.append([f"AC{k + 1}", round(lat, 4), round(lon, 4), round(heading, 1), LEVEL * 100,
                                 random_.choice(SPEEDS), round(exit_lat, 4), round(exit_lon, 4)])
            return {"aircraft": aircraft, "horizon": horizon or random_.choice(HORIZONS), "target": 0}

        def accept(instance):
            self.set_instance(instance)
            self.instance["target"] = 10 ** 9
            straight = self.simulate([HOLD] * instance["horizon"])
            if not any(s.lost for s in straight):
                return False
            plans = [self.scripted(way) for way in ("right", "left")]
            outcomes = [(trace[-1].exit_sum, plan) for plan, trace in plans if self.is_goal(trace[-1])]
            if not outcomes:
                return False
            best = min(outcomes, key=lambda o: o[0])
            instance["target"] = int(best[0] * (1 + SLACK))
            found["plan"], found["measured"] = best[1], len(plans) + 1
            return True

        instance = draw_until(draw, accept, attempts, "airspace sector")
        self.set_instance(instance)
        self.witness, self.witness_expansions = found["plan"], found["measured"]
        return instance

    def scripted(self, way):
        """A controller that turns the lower-numbered aircraft of the first predicted conflict
        `way`, and puts an aircraft back on course once nothing is predicted: its plan and trace."""
        state, _ = self.reset()
        plan, trace = [], [state]
        for _ in range(self.instance["horizon"]):
            if self.is_goal(state) or self.is_terminal(state):
                break
            conflict = self._predicted_conflict(state)
            if conflict is not None:
                action = AirspaceAction("turn", conflict, way)
            else:
                off = [a for a in state.flying if not a[7]]
                action = AirspaceAction("direct", off[0][0]) if off else HOLD
            state = self.__advance__(state, action)
            plan.append(action)
            trace.append(state)
        return plan, trace

    def _predicted_conflict(self, state):
        """The lower-numbered aircraft of the pair that flies closest within the lookahead, if
        that is under `CONFLICT_NM`; straight flight at the current heading and ground speed."""
        tracks = []
        for name, lat, lon, feet, heading, to_go, out, on_course in state.flying:
            east, north = offset(lat, lon)
            speed = self.__speeds__.get(name, 250) / 3600
            tracks.append((name, east, north, speed * math.sin(math.radians(heading)), speed * math.cos(math.radians(heading))))
        worst = None
        for i, a in enumerate(tracks):
            for b in tracks[i + 1:]:
                closest = min(math.hypot((a[1] + a[3] * t) - (b[1] + b[3] * t), (a[2] + a[4] * t) - (b[2] + b[4] * t))
                              for t in range(0, LOOKAHEAD + 1, 10))
                if closest < CONFLICT_NM and (worst is None or closest < worst[0]):
                    worst = (closest, a[0])
        return None if worst is None else worst[1]

    # ------------------------------------------------------------------ the simulator

    def _start(self):
        bs = _bs()
        bs.stack.stack("RESET")
        bs.sim.step()
        bs.stack.stack("DT 1.0")
        bs.stack.stack("ASAS OFF")
        self.__speeds__ = {}
        self.__flight__ = {"time": 0, "closest": 99.0, "exit_sum": 0, "lost": False, "out": {}, "on_course": {}}
        for name, lat, lon, heading, feet, speed, exit_lat, exit_lon in self.instance["aircraft"]:
            bs.stack.stack(f"CRE {name} A320 {lat} {lon} {heading:.1f} FL{feet / 100:.0f} {speed:.0f}")
            bs.stack.stack(f"ADDWPT {name} {exit_lat} {exit_lon}")
            bs.stack.stack(f"LNAV {name} ON")
            self.__flight__["on_course"][name] = True
            self.__speeds__[name] = speed
        bs.sim.step()
        self.__loaded__ = ()

    def _positions(self):
        bs = _bs()
        traf = bs.traf
        return {traf.id[i]: (float(traf.lat[i]), float(traf.lon[i]), float(traf.alt[i]) / 0.3048, float(traf.hdg[i]))
                for i in range(traf.ntraf)}

    def _run(self, seconds):
        bs = _bs()
        exits = {a[0]: (a[6], a[7]) for a in self.instance["aircraft"]}
        flight = self.__flight__
        for _ in range(seconds):
            bs.sim.step()
            flight["time"] += 1
            where = self._positions()
            names = list(where)
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    apart = distance_nm(where[a][:2], where[b][:2])
                    if abs(where[a][2] - where[b][2]) < SEPARATION_FT:
                        flight["closest"] = min(flight["closest"], apart)
                        if apart < SEPARATION_NM:
                            flight["lost"] = True
            for name in names:
                if distance_nm(where[name][:2], exits[name]) <= EXIT_NM:
                    flight["out"][name] = flight["time"]
                    flight["exit_sum"] += flight["time"]
                    bs.stack.stack(f"DEL {name}")
            if flight["lost"]:
                return

    def _apply(self, action):
        bs = _bs()
        if action.verb == "turn":
            where = self._positions()
            if action.aircraft in where:
                heading = (where[action.aircraft][3] + (TURN if action.way == "right" else -TURN)) % 360
                bs.stack.stack(f"HDG {action.aircraft} {heading:.0f}")
                self.__flight__["on_course"][action.aircraft] = False
        elif action.verb == "direct":
            if action.aircraft in self._positions():
                index = bs.traf.id2idx(action.aircraft)
                waypoint = bs.traf.ap.route[index].wpname[-1]
                bs.stack.stack(f"LNAV {action.aircraft} ON")
                bs.stack.stack(f"DIRECT {action.aircraft} {waypoint}")
                self.__flight__["on_course"][action.aircraft] = True
        self._run(DECISION)

    def _load(self, instructions):
        if self.__loaded__ != instructions:
            if self.__loaded__ is None or instructions[:len(self.__loaded__)] != self.__loaded__:
                self._start()
            for action in instructions[len(self.__loaded__):]:
                self._apply(action)
            self.__loaded__ = instructions

    def _read(self, instructions, depth):
        where = self._positions()
        flight = self.__flight__
        aircraft = []
        for name, lat, lon, heading, feet, speed, exit_lat, exit_lon in self.instance["aircraft"]:
            if name in flight["out"] or name not in where:
                aircraft.append((name, exit_lat, exit_lon, feet, 0.0, 0.0, True, True))
                continue
            lat, lon, feet_now, heading_now = where[name]
            aircraft.append((name, round(lat, 4), round(lon, 4), round(feet_now), round(heading_now, 1),
                             round(distance_nm((lat, lon), (exit_lat, exit_lon)), 2), False,
                             flight["on_course"].get(name, True)))
        return AirspaceState(instructions, flight["time"], aircraft, round(flight["closest"], 2),
                             flight["exit_sum"], flight["lost"], target=self.instance["target"],
                             horizon=self.instance["horizon"], depth=depth)

    # ------------------------------------------------------------------- contract

    def reset(self):
        if self.instance is None:
            self.set_index(0)
        self._start()
        self.state = self._read((), 0)
        self.state_history = [self.state]
        return self.state, {"sector": self.index, "aircraft": len(self.instance["aircraft"]),
                            "horizon": self.instance["horizon"], "target": self.instance["target"],
                            "generated": self.index is None}

    def is_goal(self, state):
        return (not state.lost and not state.flying and len(state.instructions) <= self.instance["horizon"]
                and state.exit_sum <= self.instance["target"])

    def is_terminal(self, state):
        if state.lost:
            return True
        if not state.flying:
            return state.exit_sum > self.instance["target"]
        return len(state.instructions) >= self.instance["horizon"]

    def get_actions(self, state=None):
        state = state or self.state
        actions = []
        for name, _, _, _, _, _, out, on_course in state.aircraft:
            if out:
                continue
            actions += [AirspaceAction("turn", name, "left"), AirspaceAction("turn", name, "right")]
            if not on_course:
                actions.append(AirspaceAction("direct", name))
        return actions + [HOLD]

    def successors(self, state):
        if self.is_goal(state) or self.is_terminal(state):
            return []
        children = []
        for action in self.get_actions(state):
            self._load(state.instructions)
            self._apply(action)
            self.__loaded__ = state.instructions + (action,)
            children.append((action, self._read(self.__loaded__, state.depth + 1)))
        return children

    def __advance__(self, state, action):
        if self.is_goal(state) or self.is_terminal(state):
            return state
        if not isinstance(action, AirspaceAction):
            action = AirspaceAction.parse(action)
        if action not in self.get_actions(state):
            return state
        self._load(state.instructions)
        self._apply(action)
        self.__loaded__ = state.instructions + (action,)
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
        before = len(self.state.flying)
        self.state = self.__advance__(self.state, action)
        self.state_history.append(self.state)
        return self.state, before - len(self.state.flying)

    def render(self):
        lines = [f"step {k}: {state}" for k, state in enumerate(self.state_history)]
        print("\n".join(lines))
        return lines

    def close(self):
        self.__loaded__ = None


#: The bundled sectors: `generate_instance(seed)` for the seed beside each, embedded as the
#: plain data `set_instance` takes, with the plan each was accepted on in
#: `tests/data/airspace_solutions.json`.
SECTORS = (
    # seed 9000; 3 aircraft, 15 decisions, 8-step plan
    {"aircraft": [["AC1", 51.9964, 4.9965, 272.1, 30000, 270, 52.0201, 3.9596], ["AC2", 51.699, 4.6027, 356.0, 30000, 230, 52.3328, 4.5302], ["AC3", 51.7363, 4.3045, 26.3, 30000, 250, 52.2947, 4.7531]], "horizon": 15, "target": 1267},
    # seed 9001; 3 aircraft, 15 decisions, 7-step plan
    {"aircraft": [["AC1", 51.7304, 4.4983, 1.4, 30000, 230, 52.3331, 4.5216], ["AC2", 51.7725, 4.1709, 50.2, 30000, 230, 52.1766, 4.9592], ["AC3", 52.2889, 4.4468, 178.0, 30000, 250, 51.6668, 4.4824]], "horizon": 15, "target": 1171},
    # seed 9002; 3 aircraft, 15 decisions, 9-step plan
    {"aircraft": [["AC1", 52.3055, 4.7112, 207.6, 30000, 270, 51.7178, 4.2119], ["AC2", 51.7235, 4.5649, 352.2, 30000, 270, 52.3306, 4.4304], ["AC3", 51.9432, 4.0767, 74.4, 30000, 270, 52.1046, 5.0141]], "horizon": 15, "target": 1237},
    # seed 9003; 3 aircraft, 12 decisions, 10-step plan
    {"aircraft": [["AC1", 52.0973, 5.017, 245.0, 30000, 230, 51.8185, 4.0459], ["AC2", 52.0175, 3.9848, 90.0, 30000, 270, 52.0178, 5.0406], ["AC3", 52.3052, 4.3159, 160.3, 30000, 270, 51.6851, 4.6776]], "horizon": 12, "target": 1523},
    # seed 9004; 3 aircraft, 15 decisions, 8-step plan
    {"aircraft": [["AC1", 51.771, 4.8657, 324.1, 30000, 230, 52.2952, 4.2487], ["AC2", 52.0704, 4.0534, 114.5, 30000, 250, 51.8183, 4.9539], ["AC3", 52.1848, 4.1679, 125.3, 30000, 270, 51.8348, 4.9703]], "horizon": 15, "target": 1238},
    # seed 9005; 3 aircraft, 12 decisions, 8-step plan
    {"aircraft": [["AC1", 52.215, 4.8386, 231.1, 30000, 230, 51.8203, 4.044], ["AC2", 51.693, 4.4682, 3.8, 30000, 250, 52.3325, 4.538], ["AC3", 51.7474, 4.2632, 26.8, 30000, 250, 52.3047, 4.7197]], "horizon": 12, "target": 1294},
    # seed 9006; 3 aircraft, 15 decisions, 14-step plan
    {"aircraft": [["AC1", 52.102, 4.956, 252.3, 30000, 270, 51.9096, 3.9789], ["AC2", 51.9492, 4.9617, 287.6, 30000, 230, 52.1359, 4.0056], ["AC3", 51.734, 4.2891, 21.5, 30000, 250, 52.3178, 4.6631]], "horizon": 15, "target": 1686},
    # seed 9007; 3 aircraft, 15 decisions, 7-step plan
    {"aircraft": [["AC1", 52.2939, 4.575, 195.5, 30000, 230, 51.6897, 4.3023], ["AC2", 51.9342, 4.9205, 281.2, 30000, 270, 52.0505, 3.9648], ["AC3", 51.8471, 4.1272, 64.6, 30000, 250, 52.1058, 5.0135]], "horizon": 15, "target": 1120},
    # seed 9008; 3 aircraft, 12 decisions, 10-step plan
    {"aircraft": [["AC1", 52.2759, 4.7007, 207.6, 30000, 270, 51.7134, 4.2236], ["AC2", 52.1081, 4.9928, 252.9, 30000, 270, 51.9153, 3.9763], ["AC3", 51.9644, 5.0304, 276.1, 30000, 230, 52.0352, 3.9616]], "horizon": 12, "target": 1414},
    # seed 9009; 3 aircraft, 15 decisions, 9-step plan
    {"aircraft": [["AC1", 52.2716, 4.7379, 209.7, 30000, 270, 51.714, 4.222], ["AC2", 51.7022, 4.4728, 1.8, 30000, 230, 52.3333, 4.5057], ["AC3", 52.2497, 4.3134, 159.5, 30000, 250, 51.6812, 4.658]], "horizon": 15, "target": 1278},
    # seed 9010; 3 aircraft, 15 decisions, 12-step plan
    {"aircraft": [["AC1", 52.2213, 4.8646, 226.8, 30000, 270, 51.7778, 4.0964], ["AC2", 51.8024, 4.867, 308.8, 30000, 250, 52.1991, 4.0658], ["AC3", 51.7114, 4.5161, 8.5, 30000, 270, 52.3179, 4.6628]], "horizon": 15, "target": 1480},
    # seed 9011; 4 aircraft, 12 decisions, 10-step plan
    {"aircraft": [["AC1", 52.224, 4.8555, 220.3, 30000, 250, 51.732, 4.1781], ["AC2", 51.8995, 4.9153, 290.9, 30000, 250, 52.1163, 3.9926], ["AC3", 51.7313, 4.6412, 331.3, 30000, 270, 52.2632, 4.1678], ["AC4", 52.2543, 4.2825, 143.2, 30000, 230, 51.7626, 4.88]], "horizon": 12, "target": 1837},
    # seed 9012; 3 aircraft, 15 decisions, 8-step plan
    {"aircraft": [["AC1", 52.1133, 4.904, 251.7, 30000, 270, 51.9236, 3.973], ["AC2", 51.7599, 4.1326, 52.3, 30000, 250, 52.1604, 4.9746], ["AC3", 52.1164, 4.096, 112.9, 30000, 250, 51.8799, 5.0051]], "horizon": 15, "target": 1252},
    # seed 9013; 3 aircraft, 12 decisions, 7-step plan
    {"aircraft": [["AC1", 52.0308, 4.994, 259.6, 30000, 270, 51.916, 3.9761], ["AC2", 51.8297, 4.8347, 306.8, 30000, 230, 52.1891, 4.0541], ["AC3", 51.8721, 4.0633, 55.9, 30000, 250, 52.2223, 4.9034]], "horizon": 12, "target": 1235},
    # seed 9014; 3 aircraft, 12 decisions, 12-step plan
    {"aircraft": [["AC1", 51.6849, 4.4543, 357.3, 30000, 230, 52.3281, 4.4049], ["AC2", 51.9245, 4.0621, 81.9, 30000, 250, 52.0107, 5.0411], ["AC3", 52.278, 4.2221, 139.6, 30000, 250, 51.7817, 4.9092]], "horizon": 12, "target": 1807},
    # seed 9015; 3 aircraft, 15 decisions, 10-step plan
    {"aircraft": [["AC1", 52.1753, 4.9127, 239.4, 30000, 270, 51.8495, 4.0169], ["AC2", 51.9179, 3.978, 80.3, 30000, 250, 52.0291, 5.0394], ["AC3", 52.3068, 4.326, 160.9, 30000, 270, 51.6848, 4.6759]], "horizon": 15, "target": 1491},
    # seed 9016; 3 aircraft, 12 decisions, 10-step plan
    {"aircraft": [["AC1", 52.2477, 4.734, 217.9, 30000, 270, 51.7622, 4.1206], ["AC2", 51.8657, 4.9507, 299.9, 30000, 250, 52.1852, 4.0498], ["AC3", 51.7665, 4.7128, 339.6, 30000, 250, 52.3244, 4.3757]], "horizon": 12, "target": 1315},
    # seed 9017; 3 aircraft, 12 decisions, 7-step plan
    {"aircraft": [["AC1", 52.2457, 4.6837, 195.2, 30000, 230, 51.6695, 4.4293], ["AC2", 51.7333, 4.6084, 336.0, 30000, 230, 52.282, 4.2114], ["AC3", 52.1554, 4.0454, 111.4, 30000, 270, 51.9188, 5.0251]], "horizon": 12, "target": 1147},
    # seed 9018; 4 aircraft, 15 decisions, 10-step plan
    {"aircraft": [["AC1", 52.2202, 4.7759, 210.8, 30000, 250, 51.6982, 4.2702], ["AC2", 52.0391, 4.9661, 272.5, 30000, 230, 52.0662, 3.9694], ["AC3", 51.682, 4.3938, 20.6, 30000, 270, 52.2904, 4.7658], ["AC4", 52.199, 4.1249, 139.7, 30000, 270, 51.7175, 4.7874]], "horizon": 15, "target": 1834},
    # seed 9019; 3 aircraft, 12 decisions, 12-step plan
    {"aircraft": [["AC1", 51.7175, 4.6897, 339.8, 30000, 270, 52.3167, 4.3313], ["AC2", 52.076, 4.0159, 98.3, 30000, 230, 51.9838, 5.0408], ["AC3", 52.291, 4.3182, 165.0, 30000, 250, 51.6711, 4.5884]], "horizon": 12, "target": 1527},
    # seed 9020; 3 aircraft, 15 decisions, 13-step plan
    {"aircraft": [["AC1", 51.7544, 4.2127, 35.1, 30000, 230, 52.2747, 4.8067], ["AC2", 51.8617, 4.0449, 57.4, 30000, 230, 52.2073, 4.924], ["AC3", 52.2768, 4.2386, 141.9, 30000, 230, 51.7675, 4.888]], "horizon": 15, "target": 1828},
    # seed 9021; 3 aircraft, 15 decisions, 13-step plan
    {"aircraft": [["AC1", 52.1679, 4.9453, 247.1, 30000, 230, 51.9154, 3.9763], ["AC2", 51.7236, 4.6147, 343.0, 30000, 250, 52.3148, 4.322], ["AC3", 52.2799, 4.4794, 169.6, 30000, 250, 51.6811, 4.6578]], "horizon": 15, "target": 1758},
    # seed 9022; 3 aircraft, 15 decisions, 10-step plan
    {"aircraft": [["AC1", 52.2721, 4.7987, 220.6, 30000, 250, 51.7731, 4.1034], ["AC2", 51.854, 4.9478, 291.3, 30000, 270, 52.0865, 3.9771], ["AC3", 51.7395, 4.7502, 329.8, 30000, 270, 52.2892, 4.2308]], "horizon": 15, "target": 1489},
    # seed 9023; 3 aircraft, 12 decisions, 8-step plan
    {"aircraft": [["AC1", 51.7744, 4.7672, 328.0, 30000, 230, 52.2927, 4.2408], ["AC2", 51.7115, 4.3494, 21.5, 30000, 250, 52.3023, 4.728], ["AC3", 52.184, 4.1736, 133.2, 30000, 230, 51.7695, 4.8912]], "horizon": 12, "target": 1315},
    # seed 9024; 3 aircraft, 12 decisions, 10-step plan
    {"aircraft": [["AC1", 51.9302, 5.0073, 279.9, 30000, 270, 52.0428, 3.9631], ["AC2", 51.7279, 4.406, 5.1, 30000, 250, 52.3333, 4.4933], ["AC3", 51.8624, 4.0784, 65.1, 30000, 250, 52.1264, 5.001]], "horizon": 12, "target": 1483},
    # seed 9025; 3 aircraft, 15 decisions, 9-step plan
    {"aircraft": [["AC1", 52.2974, 4.5485, 182.7, 30000, 250, 51.6667, 4.4997], ["AC2", 51.7589, 4.1604, 34.1, 30000, 230, 52.2955, 4.7506], ["AC3", 52.2281, 4.2577, 152.2, 30000, 270, 51.6939, 4.7144]], "horizon": 15, "target": 1404},
    # seed 9026; 3 aircraft, 12 decisions, 12-step plan
    {"aircraft": [["AC1", 52.2678, 4.7348, 218.5, 30000, 250, 51.7759, 4.0992], ["AC2", 51.8781, 4.9644, 300.9, 30000, 270, 52.2062, 4.0746], ["AC3", 51.706, 4.6253, 347.1, 30000, 270, 52.3269, 4.3944]], "horizon": 12, "target": 1497},
    # seed 9027; 3 aircraft, 15 decisions, 6-step plan
    {"aircraft": [["AC1", 51.7347, 4.6207, 354.3, 30000, 230, 52.333, 4.5238], ["AC2", 51.8559, 4.1265, 61.1, 30000, 270, 52.148, 4.9852], ["AC3", 52.2459, 4.25, 155.6, 30000, 230, 51.6826, 4.6654]], "horizon": 15, "target": 1133},
    # seed 9028; 3 aircraft, 15 decisions, 7-step plan
    {"aircraft": [["AC1", 51.8469, 4.0533, 52.9, 30000, 230, 52.2341, 4.8854], ["AC2", 52.1315, 4.071, 106.2, 30000, 230, 51.9591, 5.0373], ["AC3", 52.2794, 4.4523, 179.6, 30000, 250, 51.6676, 4.4586]], "horizon": 15, "target": 1203},
    # seed 9029; 3 aircraft, 15 decisions, 10-step plan
    {"aircraft": [["AC1", 52.3312, 4.5237, 183.3, 30000, 250, 51.6675, 4.4607], ["AC2", 51.9404, 4.9399, 292.4, 30000, 250, 52.17, 4.0343], ["AC3", 51.7403, 4.6084, 347.6, 30000, 270, 52.3274, 4.3982]], "horizon": 15, "target": 1463},
    # seed 9030; 3 aircraft, 15 decisions, 9-step plan
    {"aircraft": [["AC1", 51.7248, 4.5353, 5.4, 30000, 270, 52.3241, 4.6268], ["AC2", 51.8788, 4.0602, 59.0, 30000, 230, 52.2013, 4.9315], ["AC3", 52.1088, 4.0852, 105.2, 30000, 270, 51.9497, 5.0352]], "horizon": 15, "target": 1317},
    # seed 9031; 4 aircraft, 12 decisions, 10-step plan
    {"aircraft": [["AC1", 52.0722, 4.944, 245.6, 30000, 230, 51.821, 4.0432], ["AC2", 51.7122, 4.3023, 19.1, 30000, 270, 52.3212, 4.645], ["AC3", 51.9968, 3.9595, 88.5, 30000, 230, 52.0147, 5.0409], ["AC4", 52.1815, 4.12, 124.2, 30000, 230, 51.8281, 4.9639]], "horizon": 12, "target": 1975},
    # seed 9032; 3 aircraft, 12 decisions, 9-step plan
    {"aircraft": [["AC1", 51.9328, 4.9268, 276.1, 30000, 230, 51.9963, 3.9586], ["AC2", 51.7103, 4.379, 7.6, 30000, 250, 52.3332, 4.5148], ["AC3", 51.8743, 4.1003, 55.1, 30000, 230, 52.2204, 4.9062]], "horizon": 12, "target": 1317},
    # seed 9033; 3 aircraft, 15 decisions, 9-step plan
    {"aircraft": [["AC1", 52.3275, 4.5749, 191.9, 30000, 230, 51.6793, 4.3523], ["AC2", 52.2133, 4.8276, 220.5, 30000, 250, 51.7372, 4.167], ["AC3", 51.885, 4.0013, 78.7, 30000, 270, 52.013, 5.041]], "horizon": 15, "target": 1336},
    # seed 9034; 3 aircraft, 12 decisions, 8-step plan
    {"aircraft": [["AC1", 52.3068, 4.5955, 181.6, 30000, 250, 51.6693, 4.5674], ["AC2", 51.7392, 4.6221, 338.1, 30000, 230, 52.2981, 4.2577], ["AC3", 52.0714, 4.0539, 95.2, 30000, 230, 52.0156, 5.0408]], "horizon": 12, "target": 1323},
    # seed 9035; 3 aircraft, 15 decisions, 8-step plan
    {"aircraft": [["AC1", 52.2002, 4.8672, 222.1, 30000, 230, 51.7314, 4.1793], ["AC2", 51.8729, 4.9856, 300.6, 30000, 270, 52.2053, 4.0734], ["AC3", 51.9944, 4.0622, 81.7, 30000, 250, 52.0812, 5.0251]], "horizon": 15, "target": 1292},
    # seed 9036; 3 aircraft, 12 decisions, 11-step plan
    {"aircraft": [["AC1", 52.2535, 4.8405, 222.9, 30000, 270, 51.7695, 4.109], ["AC2", 52.1204, 4.0102, 109.9, 30000, 230, 51.8963, 5.0145], ["AC3", 52.2858, 4.4788, 176.2, 30000, 270, 51.6679, 4.5456]], "horizon": 12, "target": 1431},
    # seed 9037; 3 aircraft, 15 decisions, 7-step plan
    {"aircraft": [["AC1", 52.1541, 4.898, 248.2, 30000, 250, 51.9257, 3.9722], ["AC2", 51.7439, 4.65, 338.4, 30000, 250, 52.3066, 4.2874], ["AC3", 51.8474, 4.0907, 63.2, 30000, 250, 52.1298, 4.9987]], "horizon": 15, "target": 1117},
    # seed 9038; 3 aircraft, 15 decisions, 9-step plan
    {"aircraft": [["AC1", 51.8986, 5.0009, 292.7, 30000, 270, 52.1521, 4.0182], ["AC2", 52.1026, 4.0532, 117.7, 30000, 270, 51.8131, 4.9483], ["AC3", 52.2315, 4.1534, 140.5, 30000, 250, 51.7323, 4.8225]], "horizon": 15, "target": 1282},
    # seed 9039; 3 aircraft, 15 decisions, 10-step plan
    {"aircraft": [["AC1", 51.923, 4.9984, 283.8, 30000, 270, 52.078, 3.9736], ["AC2", 51.7913, 4.816, 315.4, 30000, 250, 52.232, 4.1112], ["AC3", 51.9917, 4.034, 86.0, 30000, 230, 52.0349, 5.0385]], "horizon": 15, "target": 1513},
    # seed 9040; 3 aircraft, 12 decisions, 8-step plan
    {"aircraft": [["AC1", 52.1717, 4.9417, 238.1, 30000, 250, 51.8253, 4.0389], ["AC2", 51.9318, 4.9717, 282.8, 30000, 230, 52.0713, 3.9711], ["AC3", 51.8164, 4.0987, 52.8, 30000, 250, 52.204, 4.9282]], "horizon": 12, "target": 1289},
    # seed 9041; 3 aircraft, 15 decisions, 8-step plan
    {"aircraft": [["AC1", 51.8664, 4.9615, 288.6, 30000, 270, 52.0717, 3.9713], ["AC2", 51.7078, 4.4717, 13.7, 30000, 230, 52.3076, 4.7085], ["AC3", 52.2641, 4.4243, 162.3, 30000, 270, 51.6954, 4.7199]], "horizon": 15, "target": 1239},
    # seed 9042; 3 aircraft, 15 decisions, 10-step plan
    {"aircraft": [["AC1", 51.7233, 4.6038, 356.6, 30000, 250, 52.3322, 4.5449], ["AC2", 51.9745, 4.0088, 75.4, 30000, 270, 52.1327, 4.9967], ["AC3", 52.1765, 4.1295, 132.5, 30000, 230, 51.7574, 4.8713]], "horizon": 15, "target": 1395},
    # seed 9043; 3 aircraft, 12 decisions, 11-step plan
    {"aircraft": [["AC1", 51.698, 4.34, 14.1, 30000, 230, 52.328, 4.5961], ["AC2", 52.1479, 4.0473, 112.3, 30000, 270, 51.9029, 5.018], ["AC3", 52.2391, 4.3066, 154.7, 30000, 230, 51.6964, 4.7234]], "horizon": 12, "target": 1810},
    # seed 9044; 3 aircraft, 12 decisions, 7-step plan
    {"aircraft": [["AC1", 52.2312, 4.7564, 215.7, 30000, 270, 51.7333, 4.1752], ["AC2", 52.0207, 5.0159, 271.2, 30000, 230, 52.0343, 3.9614], ["AC3", 52.2248, 4.2499, 150.8, 30000, 250, 51.6978, 4.7284]], "horizon": 12, "target": 1241},
    # seed 9045; 3 aircraft, 12 decisions, 7-step plan
    {"aircraft": [["AC1", 52.2775, 4.7541, 218.9, 30000, 270, 51.7765, 4.0983], ["AC2", 51.7387, 4.257, 38.9, 30000, 230, 52.2267, 4.8969], ["AC3", 52.2203, 4.2338, 145.9, 30000, 230, 51.7172, 4.7866]], "horizon": 12, "target": 1160},
    # seed 9046; 3 aircraft, 15 decisions, 7-step plan
    {"aircraft": [["AC1", 51.8741, 4.9234, 285.6, 30000, 230, 52.0392, 3.9623], ["AC2", 51.695, 4.518, 347.7, 30000, 250, 52.3098, 4.3002], ["AC3", 52.2962, 4.3098, 153.4, 30000, 230, 51.7157, 4.7828]], "horizon": 15, "target": 1205},
    # seed 9047; 3 aircraft, 15 decisions, 9-step plan
    {"aircraft": [["AC1", 52.0046, 4.0123, 101.1, 30000, 270, 51.8844, 5.0078], ["AC2", 52.235, 4.2093, 148.7, 30000, 270, 51.7004, 4.7373], ["AC3", 52.3108, 4.4188, 172.3, 30000, 250, 51.6687, 4.56]], "horizon": 15, "target": 1268},
    # seed 9048; 4 aircraft, 12 decisions, 10-step plan
    {"aircraft": [["AC1", 52.2869, 4.7362, 214.1, 30000, 250, 51.7483, 4.145], ["AC2", 52.1845, 4.9411, 233.0, 30000, 230, 51.7869, 4.0837], ["AC3", 52.0181, 4.9377, 274.7, 30000, 270, 52.0675, 3.9698], ["AC4", 51.6769, 4.4124, 12.8, 30000, 250, 52.3204, 4.6496]], "horizon": 12, "target": 1852},
    # seed 9049; 3 aircraft, 15 decisions, 15-step plan
    {"aircraft": [["AC1", 51.6948, 4.4758, 0.9, 30000, 230, 52.3333, 4.4917], ["AC2", 51.8053, 4.1854, 50.1, 30000, 230, 52.1936, 4.9408], ["AC3", 52.1016, 3.9969, 117.4, 30000, 250, 51.8021, 4.9357]], "horizon": 15, "target": 2028},
    # seed 9050; 3 aircraft, 15 decisions, 7-step plan
    {"aircraft": [["AC1", 52.1374, 4.8989, 241.8, 30000, 230, 51.8472, 4.0188], ["AC2", 51.7158, 4.7336, 341.7, 30000, 270, 52.3281, 4.4043], ["AC3", 51.8589, 4.0676, 65.9, 30000, 230, 52.1173, 5.0068]], "horizon": 15, "target": 1188},
    # seed 9051; 3 aircraft, 12 decisions, 7-step plan
    {"aircraft": [["AC1", 52.259, 4.839, 220.4, 30000, 230, 51.7522, 4.1378], ["AC2", 51.714, 4.537, 2.8, 30000, 230, 52.3291, 4.5865], ["AC3", 52.1917, 4.0585, 129.8, 30000, 250, 51.7669, 4.887]], "horizon": 12, "target": 1236},
    # seed 9052; 3 aircraft, 12 decisions, 11-step plan
    {"aircraft": [["AC1", 51.8376, 4.9553, 302.6, 30000, 250, 52.1915, 4.0569], ["AC2", 51.8366, 4.0995, 64.7, 30000, 250, 52.1025, 5.0152], ["AC3", 52.078, 4.0071, 107.6, 30000, 230, 51.8832, 5.0071]], "horizon": 12, "target": 1503},
    # seed 9053; 3 aircraft, 12 decisions, 8-step plan
    {"aircraft": [["AC1", 52.2591, 4.6614, 202.7, 30000, 250, 51.6958, 4.2786], ["AC2", 51.6964, 4.6132, 340.3, 30000, 230, 52.2995, 4.2623], ["AC3", 51.77, 4.1611, 43.5, 30000, 270, 52.2369, 4.8809]], "horizon": 12, "target": 1247},
    # seed 9054; 3 aircraft, 15 decisions, 11-step plan
    {"aircraft": [["AC1", 51.8209, 4.1766, 48.8, 30000, 230, 52.2166, 4.9115], ["AC2", 51.9917, 4.0502, 78.0, 30000, 230, 52.1166, 5.0072], ["AC3", 52.3142, 4.4811, 185.5, 30000, 230, 51.6749, 4.3803]], "horizon": 15, "target": 1559},
    # seed 9055; 3 aircraft, 12 decisions, 8-step plan
    {"aircraft": [["AC1", 52.2757, 4.6306, 193.0, 30000, 270, 51.6719, 4.4044], ["AC2", 51.8844, 4.9291, 283.5, 30000, 230, 52.0279, 3.9605], ["AC3", 51.6725, 4.4229, 9.5, 30000, 270, 52.3274, 4.6018]], "horizon": 12, "target": 1233},
    # seed 9056; 3 aircraft, 15 decisions, 9-step plan
    {"aircraft": [["AC1", 51.7078, 4.5606, 355.4, 30000, 250, 52.3331, 4.4797], ["AC2", 51.7233, 4.2617, 35.3, 30000, 230, 52.247, 4.8635], ["AC3", 52.3078, 4.3191, 157.8, 30000, 270, 51.6964, 4.7234]], "horizon": 15, "target": 1298},
    # seed 9057; 3 aircraft, 15 decisions, 8-step plan
    {"aircraft": [["AC1", 51.8542, 4.8912, 292.5, 30000, 270, 52.0875, 3.9776], ["AC2", 51.7988, 4.0958, 45.5, 30000, 230, 52.2545, 4.8496], ["AC3", 52.1351, 4.0671, 123.1, 30000, 250, 51.7917, 4.9227]], "horizon": 15, "target": 1248},
    # seed 9058; 3 aircraft, 12 decisions, 7-step plan
    {"aircraft": [["AC1", 52.1214, 4.9548, 236.4, 30000, 270, 51.7735, 4.1028], ["AC2", 51.724, 4.6877, 337.6, 30000, 250, 52.3087, 4.2957], ["AC3", 51.7673, 4.2552, 28.7, 30000, 270, 52.3017, 4.7301]], "horizon": 12, "target": 1083},
    # seed 9059; 4 aircraft, 12 decisions, 12-step plan
    {"aircraft": [["AC1", 51.8564, 4.8772, 308.4, 30000, 270, 52.2308, 4.1094], ["AC2", 51.7123, 4.71, 327.3, 30000, 270, 52.2522, 4.146], ["AC3", 51.9398, 4.0406, 88.6, 30000, 270, 51.9547, 5.0364], ["AC4", 52.2591, 4.2768, 159.0, 30000, 230, 51.6779, 4.6394]], "horizon": 12, "target": 2109},
    # seed 9060; 3 aircraft, 12 decisions, 11-step plan
    {"aircraft": [["AC1", 52.2063, 4.8925, 228.6, 30000, 230, 51.7755, 4.0998], ["AC2", 51.7567, 4.2289, 24.5, 30000, 250, 52.3209, 4.6465], ["AC3", 51.9412, 4.0494, 86.0, 30000, 250, 51.9838, 5.0408]], "horizon": 12, "target": 1507},
    # seed 9061; 4 aircraft, 12 decisions, 10-step plan
    {"aircraft": [["AC1", 51.7234, 4.4242, 13.1, 30000, 230, 52.3204, 4.6491], ["AC2", 51.9624, 3.9782, 89.1, 30000, 270, 51.9722, 5.0395], ["AC3", 52.1564, 4.1371, 135.6, 30000, 230, 51.7295, 4.8163], ["AC4", 52.3204, 4.3598, 169.0, 30000, 230, 51.6691, 4.5651]], "horizon": 12, "target": 1992},
    # seed 9062; 3 aircraft, 15 decisions, 14-step plan
    {"aircraft": [["AC1", 52.2854, 4.6749, 195.5, 30000, 230, 51.6725, 4.3988], ["AC2", 51.8389, 4.8602, 309.6, 30000, 270, 52.2257, 4.1016], ["AC3", 52.2601, 4.167, 141.6, 30000, 230, 51.7394, 4.8377]], "horizon": 15, "target": 2032},
    # seed 9063; 3 aircraft, 15 decisions, 7-step plan
    {"aircraft": [["AC1", 52.2686, 4.664, 208.0, 30000, 250, 51.7246, 4.195], ["AC2", 51.8891, 3.9926, 61.3, 30000, 230, 52.2042, 4.928], ["AC3", 52.091, 4.0881, 104.5, 30000, 230, 51.9405, 5.0327]], "horizon": 15, "target": 1287},
    # seed 9064; 3 aircraft, 12 decisions, 8-step plan
    {"aircraft": [["AC1", 51.8712, 4.9387, 292.9, 30000, 230, 52.1176, 3.9934], ["AC2", 51.7338, 4.6868, 333.3, 30000, 250, 52.29, 4.2329], ["AC3", 52.2831, 4.4014, 162.9, 30000, 270, 51.6897, 4.6978]], "horizon": 12, "target": 1238},
    # seed 9065; 3 aircraft, 12 decisions, 11-step plan
    {"aircraft": [["AC1", 51.7151, 4.5588, 353.5, 30000, 230, 52.3315, 4.4439], ["AC2", 51.7337, 4.2477, 38.8, 30000, 230, 52.2283, 4.8945], ["AC3", 52.1909, 4.0937, 123.1, 30000, 270, 51.8376, 4.9728]], "horizon": 12, "target": 1529},
    # seed 9066; 3 aircraft, 12 decisions, 8-step plan
    {"aircraft": [["AC1", 52.2698, 4.7344, 208.2, 30000, 230, 51.7063, 4.2438], ["AC2", 52.0506, 4.9476, 259.6, 30000, 230, 51.9395, 3.9676], ["AC3", 51.7764, 4.1909, 42.6, 30000, 270, 52.2375, 4.8799]], "horizon": 12, "target": 1303},
    # seed 9067; 3 aircraft, 15 decisions, 14-step plan
    {"aircraft": [["AC1", 52.2043, 4.8857, 238.1, 30000, 270, 51.866, 4.0042], ["AC2", 51.8945, 4.0427, 63.2, 30000, 230, 52.1787, 4.957], ["AC3", 52.2812, 4.2766, 149.7, 30000, 250, 51.7246, 4.8051]], "horizon": 15, "target": 1799},
    # seed 9068; 3 aircraft, 15 decisions, 7-step plan
    {"aircraft": [["AC1", 52.1119, 4.9546, 258.5, 30000, 230, 51.9874, 3.959], ["AC2", 51.6814, 4.6002, 343.4, 30000, 250, 52.3087, 4.2958], ["AC3", 52.1993, 4.2027, 148.3, 30000, 230, 51.693, 4.711]], "horizon": 15, "target": 1201},
    # seed 9069; 3 aircraft, 12 decisions, 8-step plan
    {"aircraft": [["AC1", 52.2507, 4.8431, 211.5, 30000, 250, 51.6933, 4.2877], ["AC2", 52.0263, 4.9897, 266.8, 30000, 230, 51.991, 3.9588], ["AC3", 51.7266, 4.2019, 37.0, 30000, 250, 52.2552, 4.8484]], "horizon": 12, "target": 1304},
    # seed 9070; 3 aircraft, 15 decisions, 6-step plan
    {"aircraft": [["AC1", 52.0235, 4.9734, 271.4, 30000, 230, 52.0386, 3.9622], ["AC2", 51.9405, 4.034, 70.8, 30000, 270, 52.1448, 4.9877], ["AC3", 52.2134, 4.137, 140.2, 30000, 250, 51.7228, 4.8008]], "horizon": 15, "target": 1150},
    # seed 9071; 3 aircraft, 12 decisions, 8-step plan
    {"aircraft": [["AC1", 51.8678, 4.9139, 290.9, 30000, 270, 52.0877, 3.9776], ["AC2", 51.7324, 4.4303, 7.4, 30000, 230, 52.3315, 4.5573], ["AC3", 52.1437, 4.025, 112.8, 30000, 270, 51.8892, 5.0106]], "horizon": 12, "target": 1259},
    # seed 9072; 3 aircraft, 12 decisions, 8-step plan
    {"aircraft": [["AC1", 51.9672, 4.9339, 274.5, 30000, 230, 52.0142, 3.9591], ["AC2", 51.7141, 4.5655, 350.9, 30000, 270, 52.3282, 4.4055], ["AC3", 52.0879, 4.0233, 98.7, 30000, 230, 51.9916, 5.0413]], "horizon": 12, "target": 1320},
    # seed 9073; 3 aircraft, 12 decisions, 9-step plan
    {"aircraft": [["AC1", 52.0862, 4.9321, 256.4, 30000, 230, 51.9429, 3.9666], ["AC2", 51.9442, 4.9898, 275.0, 30000, 230, 51.9997, 3.9586], ["AC3", 52.0139, 4.0302, 94.9, 30000, 270, 51.9608, 5.0377]], "horizon": 12, "target": 1292},
    # seed 9074; 3 aircraft, 15 decisions, 8-step plan
    {"aircraft": [["AC1", 52.2661, 4.6293, 201.2, 30000, 270, 51.698, 4.271], ["AC2", 51.6861, 4.3416, 22.0, 30000, 250, 52.298, 4.7426], ["AC3", 52.2068, 4.1468, 129.4, 30000, 230, 51.8055, 4.9397]], "horizon": 15, "target": 1301},
    # seed 9075; 3 aircraft, 12 decisions, 8-step plan
    {"aircraft": [["AC1", 51.9199, 4.9615, 282.5, 30000, 230, 52.0556, 3.9661], ["AC2", 51.8214, 4.0638, 56.8, 30000, 250, 52.1806, 4.9551], ["AC3", 52.0147, 4.056, 96.1, 30000, 270, 51.9507, 5.0355]], "horizon": 12, "target": 1243},
    # seed 9076; 3 aircraft, 12 decisions, 10-step plan
    {"aircraft": [["AC1", 52.2124, 4.7917, 221.6, 30000, 230, 51.7553, 4.1324], ["AC2", 51.9172, 4.0035, 64.7, 30000, 270, 52.1908, 4.944], ["AC3", 52.2863, 4.3985, 176.8, 30000, 270, 51.6678, 4.4548]], "horizon": 12, "target": 1439},
    # seed 9077; 3 aircraft, 12 decisions, 11-step plan
    {"aircraft": [["AC1", 51.7645, 4.8777, 305.6, 30000, 250, 52.1459, 4.0132], ["AC2", 52.1353, 4.0201, 122.6, 30000, 270, 51.7843, 4.9128], ["AC3", 52.2752, 4.2492, 157.0, 30000, 230, 51.6814, 4.659]], "horizon": 12, "target": 1538},
    # seed 9078; 3 aircraft, 12 decisions, 8-step plan
    {"aircraft": [["AC1", 51.8655, 4.0148, 57.6, 30000, 230, 52.216, 4.9123], ["AC2", 52.0701, 4.0044, 98.1, 30000, 230, 51.9791, 5.0404], ["AC3", 52.2852, 4.3481, 171.5, 30000, 270, 51.6667, 4.4975]], "horizon": 12, "target": 1288},
    # seed 9079; 3 aircraft, 15 decisions, 8-step plan
    {"aircraft": [["AC1", 52.2137, 4.7698, 206.9, 30000, 250, 51.6829, 4.3332], ["AC2", 51.8691, 4.9236, 286.9, 30000, 270, 52.0484, 3.9643], ["AC3", 52.1586, 4.0629, 121.2, 30000, 250, 51.8239, 4.9597]], "horizon": 15, "target": 1255},
    # seed 9080; 4 aircraft, 12 decisions, 7-step plan
    {"aircraft": [["AC1", 52.2892, 4.5404, 174.3, 30000, 270, 51.6778, 4.6388], ["AC2", 51.931, 5.0161, 274.1, 30000, 270, 51.9773, 3.9598], ["AC3", 51.8056, 4.8115, 321.8, 30000, 270, 52.2795, 4.2049], ["AC4", 51.7019, 4.2619, 34.3, 30000, 270, 52.2459, 4.8656]], "horizon": 12, "target": 1536},
    # seed 9081; 3 aircraft, 12 decisions, 10-step plan
    {"aircraft": [["AC1", 51.8112, 4.914, 311.1, 30000, 250, 52.2376, 4.1203], ["AC2", 51.9814, 4.0233, 76.0, 30000, 270, 52.1311, 4.9978], ["AC3", 52.2408, 4.153, 130.2, 30000, 250, 51.822, 4.9577]], "horizon": 12, "target": 1516},
    # seed 9082; 3 aircraft, 15 decisions, 10-step plan
    {"aircraft": [["AC1", 51.7301, 4.3714, 14.4, 30000, 230, 52.3251, 4.6198], ["AC2", 52.0118, 4.016, 96.5, 30000, 270, 51.9405, 5.0327], ["AC3", 52.2125, 4.155, 128.2, 30000, 230, 51.8232, 4.959]], "horizon": 15, "target": 1508},
    # seed 9083; 3 aircraft, 12 decisions, 8-step plan
    {"aircraft": [["AC1", 52.0143, 4.9386, 263.3, 30000, 230, 51.9435, 3.9664], ["AC2", 51.7887, 4.778, 319.6, 30000, 230, 52.2498, 4.1415], ["AC3", 52.1194, 4.1117, 118.0, 30000, 250, 51.8373, 4.9725]], "horizon": 12, "target": 1284},
    # seed 9084; 3 aircraft, 12 decisions, 9-step plan
    {"aircraft": [["AC1", 52.2767, 4.5716, 182.3, 30000, 270, 51.6672, 4.5309], ["AC2", 52.1736, 4.9421, 239.6, 30000, 250, 51.8422, 4.0231], ["AC3", 51.8825, 4.9313, 283.9, 30000, 230, 52.0307, 3.9609]], "horizon": 12, "target": 1381},
    # seed 9085; 3 aircraft, 12 decisions, 12-step plan
    {"aircraft": [["AC1", 51.9327, 4.9243, 278.2, 30000, 230, 52.0183, 3.9594], ["AC2", 51.9612, 4.0504, 92.5, 30000, 230, 51.9353, 5.0311], ["AC3", 52.2297, 4.2661, 153.0, 30000, 250, 51.6929, 4.7106]], "horizon": 12, "target": 1648},
    # seed 9086; 3 aircraft, 15 decisions, 7-step plan
    {"aircraft": [["AC1", 52.2882, 4.5573, 192.7, 30000, 250, 51.6825, 4.3352], ["AC2", 52.1662, 4.862, 239.9, 30000, 230, 51.8614, 4.0076], ["AC3", 51.9022, 4.0924, 70.2, 30000, 250, 52.1062, 5.0132]], "horizon": 15, "target": 1129},
    # seed 9087; 3 aircraft, 12 decisions, 7-step plan
    {"aircraft": [["AC1", 52.247, 4.729, 214.2, 30000, 270, 51.7371, 4.1672], ["AC2", 51.6899, 4.5214, 7.7, 30000, 250, 52.3185, 4.6595], ["AC3", 52.2492, 4.23, 151.8, 30000, 270, 51.6937, 4.7134]], "horizon": 12, "target": 1204},
    # seed 9088; 3 aircraft, 12 decisions, 8-step plan
    {"aircraft": [["AC1", 52.2784, 4.7335, 212.0, 30000, 250, 51.7316, 4.179], ["AC2", 51.6899, 4.3752, 16.8, 30000, 270, 52.3142, 4.6806], ["AC3", 52.2732, 4.3169, 157.1, 30000, 270, 51.6941, 4.7152]], "horizon": 12, "target": 1268},
    # seed 9089; 3 aircraft, 15 decisions, 11-step plan
    {"aircraft": [["AC1", 51.8569, 4.1167, 52.2, 30000, 230, 52.2283, 4.8945], ["AC2", 52.1123, 4.0249, 109.0, 30000, 230, 51.9023, 5.0177], ["AC3", 52.2824, 4.4537, 181.4, 30000, 230, 51.6695, 4.4294]], "horizon": 15, "target": 1558},
    # seed 9090; 3 aircraft, 12 decisions, 6-step plan
    {"aircraft": [["AC1", 52.2604, 4.692, 209.3, 30000, 270, 51.7222, 4.2008], ["AC2", 52.1394, 4.9702, 253.4, 30000, 270, 51.9542, 3.9637], ["AC3", 51.8273, 4.8895, 315.5, 30000, 230, 52.2701, 4.1827]], "horizon": 12, "target": 1116},
    # seed 9091; 3 aircraft, 12 decisions, 9-step plan
    {"aircraft": [["AC1", 51.9842, 4.9999, 272.7, 30000, 250, 52.015, 3.9591], ["AC2", 51.9474, 4.0598, 83.5, 30000, 250, 52.0161, 5.0408], ["AC3", 52.1749, 4.0751, 131.1, 30000, 250, 51.7524, 4.8624]], "horizon": 12, "target": 1290},
    # seed 9092; 3 aircraft, 15 decisions, 8-step plan
    {"aircraft": [["AC1", 52.2368, 4.8629, 223.5, 30000, 270, 51.7587, 4.1265], ["AC2", 52.0141, 5.0075, 272.0, 30000, 270, 52.0368, 3.9619], ["AC3", 51.9888, 4.019, 84.1, 30000, 250, 52.0532, 5.0345]], "horizon": 15, "target": 1277},
    # seed 9093; 3 aircraft, 12 decisions, 7-step plan
    {"aircraft": [["AC1", 51.749, 4.8353, 310.7, 30000, 230, 52.1718, 4.036], ["AC2", 51.6792, 4.3763, 5.0, 30000, 250, 52.3328, 4.4687], ["AC3", 52.3196, 4.3844, 160.0, 30000, 270, 51.7041, 4.7492]], "horizon": 12, "target": 1189},
    # seed 9094; 3 aircraft, 15 decisions, 8-step plan
    {"aircraft": [["AC1", 52.2135, 4.8293, 232.8, 30000, 250, 51.8388, 4.0261], ["AC2", 51.6734, 4.5435, 348.3, 30000, 250, 52.3159, 4.3272], ["AC3", 52.1501, 4.1206, 114.1, 30000, 250, 51.9025, 5.0177]], "horizon": 15, "target": 1247},
    # seed 9095; 3 aircraft, 15 decisions, 7-step plan
    {"aircraft": [["AC1", 52.305, 4.6794, 195.2, 30000, 270, 51.6725, 4.3994], ["AC2", 51.7778, 4.8797, 307.1, 30000, 230, 52.1713, 4.0356], ["AC3", 51.9772, 4.0338, 82.4, 30000, 250, 52.0594, 5.0328]], "horizon": 15, "target": 1172},
    # seed 9096; 3 aircraft, 15 decisions, 8-step plan
    {"aircraft": [["AC1", 52.3118, 4.5449, 192.4, 30000, 250, 51.6854, 4.3213], ["AC2", 51.7806, 4.8669, 318.5, 30000, 270, 52.2645, 4.1705], ["AC3", 52.0449, 4.0397, 90.0, 30000, 230, 52.0447, 5.0365]], "horizon": 15, "target": 1310},
    # seed 9097; 3 aircraft, 12 decisions, 6-step plan
    {"aircraft": [["AC1", 52.1999, 4.8429, 222.7, 30000, 250, 51.7424, 4.1563], ["AC2", 51.775, 4.8171, 324.7, 30000, 270, 52.2878, 4.2267], ["AC3", 52.1046, 4.0844, 118.0, 30000, 270, 51.8199, 4.9556]], "horizon": 12, "target": 1073},
    # seed 9098; 3 aircraft, 12 decisions, 7-step plan
    {"aircraft": [["AC1", 51.9279, 5.0249, 277.9, 30000, 250, 52.0184, 3.9594], ["AC2", 51.699, 4.5885, 345.8, 30000, 250, 52.3175, 4.3349], ["AC3", 52.1268, 4.0587, 107.0, 30000, 270, 51.9435, 5.0336]], "horizon": 12, "target": 1144},
    # seed 9100; 3 aircraft, 12 decisions, 11-step plan
    {"aircraft": [["AC1", 52.1045, 5.0007, 248.5, 30000, 270, 51.8637, 4.0059], ["AC2", 51.7999, 4.8137, 319.0, 30000, 230, 52.261, 4.1632], ["AC3", 51.8126, 4.186, 44.5, 30000, 230, 52.2425, 4.8715]], "horizon": 12, "target": 1523},
)
