"""Microgrid operations on python-microgrid: a load, solar panels, a battery and, on some of
the benchmark microgrids, a diesel genset and a connection to the grid, with a battery setting
and the genset's switch to choose every three hours, and a day or two to get through with
every hour's load met, the operating cost under the target and the battery handed back at least
as charged as it was found.

python-microgrid (`pymgrid`, https://github.com/ahalev/python-microgrid, LGPL-3.0; Henri,
Levent, Halev, Alami, Cordier 2020, https://arxiv.org/abs/2011.08004) is the microgrid
simulator they describe there: a microgrid is a set of modules (load, renewable, battery,
genset, grid, and an unbalanced-energy module that absorbs what is left over), and a step
dispatches them for an hour and charges each its cost. It ships twenty-five benchmark
microgrids, `pymgrid25`, with a year of hourly load, solar, price and outage data each, inside
the package. It is a dependency installed from PyPI; nothing of it is included here.

The decisions are the operator's: what the battery does for the next three hours (sits idle,
stores the solar surplus, covers the load from its charge as well, or charges at its full rate)
and whether the genset runs. Everything else is pymgrid's: the battery's efficiency and
limits, the genset's minimum and maximum output, the grid's prices, carbon cost and outages,
solar curtailment, and the cost of each hour. The grid, where there is one, takes the balance.

## Trajectory constraints as a goal

The load must be met every hour, so a period with any loss of load is `is_terminal`. The
operating cost accumulates in the state, and the goal is the horizon reached with that cost at
or under the target and the battery at or above its reserve, which is a plan's final state.
The target is the least cost any of the scripted reference policies manages inside the
constraints, so the planner is asked to do at least as well as the best rule of thumb.

## Determinism and state

pymgrid is deterministic and its time series are fixed, so a state is what a period leaves
behind: the hours elapsed, the battery's charge, the cost and the loss of load so far. Two
histories that leave the same charge and cost are one state. The environment expands one by
setting pymgrid's modules to the state's hour and charge and stepping the microgrid.
"""
import contextlib
import io

from planiverse.environments.base import Environment, Panel
from planiverse.environments.generation import draw_until, rng

#: Hours a decision holds for, and what the battery may be told to do for them.
PERIOD = 3
MODES = ("idle", "store", "cover", "charge")
#: The pymgrid25 benchmark microgrids, the horizons an instance may run for, and the hours of
#: the day it may start at.
MICROGRIDS = 25
HORIZONS = (24, 36, 48)
STARTS = (0, 6, 12, 18)
#: pymgrid's modules forecast this many hours ahead, so an instance ends this far before the
#: end of the year's data.
FORECAST = 24
HOURS = 8760
#: Energy (kWh) below which an hour's shortfall is rounding rather than loss of load.
TOLERANCE = 1e-3


def _pymgrid():
    import numpy
    if not hasattr(numpy, "product"):
        # pymgrid 1.4.1 calls `numpy.product`, the alias of `numpy.prod` that NumPy 2 removed,
        # on every step of a load module.
        numpy.product = numpy.prod
    with contextlib.redirect_stderr(io.StringIO()):     # gym's notice that it is unmaintained
        import pymgrid
    return pymgrid


class MicrogridAction:
    """`operate(mode, genset)`: `mode` from `MODES`, `genset` `on` or `off`."""

    def __init__(self, mode, genset):
        if mode not in MODES or genset not in ("on", "off"):
            raise ValueError(f"unknown setting: battery {mode!r}, genset {genset!r}")
        self.mode, self.genset = mode, genset
        self.name = f"operate({mode}, {genset})"

    @classmethod
    def parse(cls, text):
        inside = str(text).strip()[len("operate("):-1]
        mode, genset = (p.strip() for p in inside.split(","))
        return cls(mode, genset)

    def cost(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, MicrogridAction) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    __repr__ = __str__


class MicrogridState:
    """A period's end: the hours elapsed, the battery's charge, the cost and the loss of load so
    far, and the last period's energy flows. Identity is the hour, the charge, the cost and the
    loss of load."""

    #: How a chart of a trace lays the readings out: one axis per quantity, the
    #: series that share its unit, and the reading drawn across it as a target.
    panels = (
        Panel("energy a period (kWh)", ("load", "solar", "genset", "grid"), None),
        Panel("battery state of charge", ("charge",), "reserve"),
        Panel("operating cost", ("cost",), "allowed"),
    )

    def readings(self):
        """What a chart of a trace plots, by name: the state's numbers, which its
        OpenSpiel game also gives as the named pieces of its observation."""
        load, solar, battery, genset, grid = self.flows
        return {"load": load, "solar": solar, "battery": battery, "genset": genset, "grid": grid,
                "charge": self.soc, "reserve": self.reserve, "cost": self.cost,
                "allowed": self.target, "hour": self.hour}

    def __init__(self, hour, charge, cost, lost, flows, capacity=1.0, target=0.0, reserve=0.0,
                 horizon=0, start=0, depth=0):
        self.hour = hour
        self.charge, self.cost, self.lost = round(charge, 1), round(cost, 2), round(lost, 1)
        self.flows = tuple(round(v, 1) for v in flows)   # load, solar, battery out, genset, grid in
        self.capacity, self.target, self.reserve = capacity, target, reserve
        self.horizon, self.start, self.depth = horizon, start, depth
        self.soc = round(self.charge / capacity, 4)
        self.key = (self.hour, self.charge, self.cost, self.lost)
        literals = [f"hour({hour})", f"soc({int(self.soc * 100)})"]
        if 0 < target < float("inf"):
            literals.append(f"cost({int(100 * self.cost / target)})")
        if self.lost > 0:
            literals.append("lost_load")
        self.literals = frozenset(literals)

    def __eq__(self, other):
        return isinstance(other, MicrogridState) and self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        load, solar, battery, genset, grid = self.flows
        clock = self.start + self.hour
        return (f"hour {self.hour} of {self.horizon} (day {clock // 24}, {clock % 24:02d}:00): "
                f"battery at {self.soc:.2f}; last period load {load:.0f}, solar {solar:.0f}, "
                f"battery {battery:+.0f}, genset {genset:.0f}, grid {grid:.0f} kWh; cost "
                f"{self.cost:.2f} of {self.target} allowed")

    def __repr__(self):
        return (f"<MicrogridState(hour={self.hour}, soc={self.soc:.2f}, cost={self.cost:.2f}, "
                f"lost={self.lost:.1f})>")


# ------------------------------------------------------------------ reference policies

def _constant(mode, genset):
    """The same battery setting all the way, the genset `on`, `off` or `backup`."""
    return lambda env, state, period, periods: (mode, genset)


def _recharge_at_the_end(last):
    """Cover the load from the battery, then charge it for the `last` periods."""
    return lambda env, state, period, periods: ("charge" if period >= periods - last else "cover",
                                                "backup")


def _charge_off_peak(env, state, period, periods):
    """Charge in the periods whose grid price is the horizon's lowest, cover otherwise."""
    prices = env._period_prices()
    return ("charge" if prices[period] <= min(prices) + 1e-9 else "cover", "backup")


def reference_policies(has_genset, has_grid):
    """The scripted policies the target is set off, as `(name, rule)` pairs.

    A rule gives the battery's setting for a period and the genset `on`, `off` or `backup`,
    which is off unless the period would otherwise lose load. With the grid taking the balance
    and the genset as a backup, `cover` is pymgrid's own rule-based control in merit order:
    solar first, then the battery, then the grid, then the genset.
    """
    policies = [(f"{mode} throughout", _constant(mode, "backup")) for mode in MODES]
    if has_genset:
        policies += [(f"{mode} throughout, genset on", _constant(mode, "on"))
                     for mode in ("store", "cover")]
    policies += [(f"cover, then charge for the last {k}", _recharge_at_the_end(k)) for k in (1, 2, 3)]
    if has_grid:
        policies.append(("charge off-peak, cover otherwise", _charge_off_peak))
    return policies


class MicrogridEnv(Environment):
    """A day or two of a pymgrid microgrid with the load met, the cost under the target and
    the battery back at its reserve."""

    def __init__(self):
        super().__init__("microgrid")
        self.instance = None
        self.index = None
        self.state = None
        self.witness = None
        self.witness_expansions = None
        self.__grid__ = None            # (microgrid number, pymgrid Microgrid, its modules)

    # ------------------------------------------------------------------ instances

    def set_index(self, index):
        if not 0 <= index < len(DAYS):
            raise IndexError(f"Invalid index: {index}. There are {len(DAYS)} instances, so the "
                             f"index must be 0-{len(DAYS) - 1}.")
        self.set_instance(DAYS[index])
        self.index = index

    def set_instance(self, instance):
        """Select an instance: `microgrid` (0-24, pymgrid's benchmark microgrid), `start` (the
        hour of the year it starts at), `horizon` (hours, a multiple of three), `soc` (the
        battery's state of charge at the start), `reserve` (the least it may end at) and
        `target` (the operating cost allowed)."""
        for key in ("microgrid", "start", "horizon", "soc", "reserve", "target"):
            if key not in instance:
                raise ValueError(f"an instance needs `{key}`")
        microgrid, start, horizon = int(instance["microgrid"]), int(instance["start"]), int(instance["horizon"])
        if not 0 <= microgrid < MICROGRIDS:
            raise ValueError(f"microgrid must be 0-{MICROGRIDS - 1}, not {microgrid}")
        if horizon <= 0 or horizon % PERIOD:
            raise ValueError(f"the horizon must be a positive multiple of {PERIOD} hours")
        if not 0 <= start <= HOURS - horizon - FORECAST:
            raise ValueError(f"start {start} leaves no room for {horizon} hours in the year")
        self.instance = {"microgrid": microgrid, "start": start, "horizon": horizon,
                         "soc": float(instance["soc"]), "reserve": float(instance["reserve"]),
                         "target": float(instance["target"])}
        self.index = None
        self.witness = self.witness_expansions = None

    def generate_instance(self, seed=None, microgrid=None, horizon=None, attempts=40):
        """Draw an instance, select it, and return it as the dict `set_instance` takes.

        A draw is one of pymgrid's twenty-five microgrids (or `microgrid`), a day of the year,
        a start at midnight, six, noon or six in the evening, a horizon of 24, 36 or 48 hours
        (or `horizon`) and a starting state of charge, which is also the reserve. The policies
        of `reference_policies` are run; the target is the least cost any of them ends with
        inside the constraints, and its plan is the witness. A draw is thrown back when none
        does, or when the battery left idle throughout does as well as any, since then nothing
        needs deciding.
        """
        random_, _ = rng(seed)
        found = {}

        def draw(attempt):
            hours = horizon or random_.choice(HORIZONS)
            day = random_.randrange((HOURS - hours - FORECAST - max(STARTS)) // 24)
            soc = round(random_.uniform(0.3, 0.8), 2)
            return {"microgrid": random_.randrange(MICROGRIDS) if microgrid is None else microgrid,
                    "start": day * 24 + random_.choice(STARTS), "horizon": hours,
                    "soc": soc, "reserve": soc, "target": 0}

        def accept(instance):
            self.set_instance(instance)
            self.instance["target"] = float("inf")
            outcomes = self.run_references()
            feasible = [o for o in outcomes if o[1] is not None]
            if not feasible:
                return False
            best = min(feasible, key=lambda o: o[1])
            idle = [o for o in feasible if o[0] == "idle throughout"]
            if idle and idle[0][1] <= best[1]:
                return False                    # the battery left alone is already best
            instance["target"] = best[1]
            found["plan"], found["measured"] = best[2], len(outcomes)
            return True

        instance = draw_until(draw, accept, attempts, "microgrid instance")
        self.set_instance(instance)
        self.witness, self.witness_expansions = found["plan"], found["measured"]
        return instance

    def run_references(self):
        """Run every reference policy on the selected instance, as `(name, cost, plan)`, the
        cost `None` for a policy that loses load or ends under the reserve."""
        _, modules = self._microgrid()
        policies = reference_policies("genset" in modules, "grid" in modules)
        periods = self.instance["horizon"] // PERIOD
        outcomes = []
        for name, rule in policies:
            state, _ = self.reset()
            plan = []
            for period in range(periods):
                mode, genset = rule(self, state, period, periods)
                if genset == "backup":
                    child = self.__advance__(state, MicrogridAction(mode, "off"))
                    action = MicrogridAction(mode, "off")
                    if child.lost > 0 and "genset" in modules:
                        action = MicrogridAction(mode, "on")
                        child = self.__advance__(state, action)
                else:
                    action = MicrogridAction(mode, genset if "genset" in modules else "off")
                    child = self.__advance__(state, action)
                plan.append(action)
                state = child
                if state.lost > 0:
                    break
            ok = state.hour >= self.instance["horizon"] and state.lost == 0 and \
                state.charge >= self._reserve_charge()
            outcomes.append((name, state.cost if ok else None, plan))
        return outcomes

    # ------------------------------------------------------------------ the simulator

    def _microgrid(self):
        number = self.instance["microgrid"]
        if self.__grid__ is None or self.__grid__[0] != number:
            pymgrid = _pymgrid()
            grid = pymgrid.Microgrid.from_scenario(microgrid_number=number)
            modules = {name: modules[0] for name, modules in grid.modules.iterdict()}
            self.__grid__ = (number, grid, modules)
        return self.__grid__[1], self.__grid__[2]

    def _capacity(self):
        return float(self._microgrid()[1]["battery"].max_capacity)

    def _reserve_charge(self):
        return round(self.instance["reserve"] * self._capacity(), 1) - 0.05

    def _period_prices(self):
        """The mean import price of each period of the horizon."""
        _, modules = self._microgrid()
        series = modules["grid"].time_series
        start = self.instance["start"]
        return [float(series[start + p:start + p + PERIOD, 0].mean())
                for p in range(0, self.instance["horizon"], PERIOD)]

    def _restore(self, state):
        grid, modules = self._microgrid()
        grid.reset()                                     # flushes pymgrid's logs as well
        for module in modules.values():
            module.current_step = self.instance["start"] + state.hour
        battery = modules["battery"]
        # The state keeps the charge to a tenth of a kWh, which can round it a hair past the
        # battery's limits; pymgrid's own bounds are the ones that hold.
        battery.current_charge = min(max(state.charge, battery.min_capacity), battery.max_capacity)
        return grid, modules

    def _hour(self, grid, modules, mode, genset_on):
        """Dispatch one hour: the battery's setting and the genset's switch are the plan's, the
        grid takes the balance, and pymgrid steps the microgrid and charges its cost."""
        battery, load, solar = modules["battery"], modules["load"].current_load, modules["pv"].current_renewable
        surplus, deficit = max(0.0, solar - load), max(0.0, load - solar)
        charge = discharge = 0.0
        if mode in ("store", "cover"):
            charge = min(surplus, battery.max_consumption)
        if mode == "cover":
            discharge = min(deficit, battery.max_production)
        if mode == "charge":
            charge = battery.max_consumption
        base = load + charge - discharge                 # what solar, the genset and the grid supply
        need = base - solar
        control = {"battery": [discharge - charge]}
        genset = 0.0
        if "genset" in modules:
            unit = modules["genset"]
            if genset_on:
                genset = min(max(need, unit.running_min_production), unit.running_max_production)
            control["genset"] = [[1.0 if genset_on else 0.0, genset]]
        if "grid" in modules:
            line = modules["grid"]
            available = line.max_import if line.time_series[line.current_step, 3] > 0 else 0.0
            exportable = line.max_export if line.time_series[line.current_step, 3] > 0 else 0.0
            imported = min(max(0.0, need - genset), available)
            exported = min(max(0.0, genset - base), exportable)
            control["grid"] = [imported - exported]
        _, reward, _, info = grid.step(control, normalized=False)
        lost = float(info["unbalanced_energy"][0].get("provided_energy", 0.0))
        used = float(info["pv"][0].get("provided_energy", 0.0))
        out = info["battery"][0]
        flow = float(out.get("provided_energy", 0.0)) - float(out.get("absorbed_energy", 0.0))
        produced = float(info["genset"][0].get("provided_energy", 0.0)) if "genset" in info else 0.0
        line_flow = 0.0
        if "grid" in info:
            line_flow = float(info["grid"][0].get("provided_energy", 0.0)) - \
                float(info["grid"][0].get("absorbed_energy", 0.0))
        return -float(reward), lost, (load, used, flow, produced, line_flow)

    def _period(self, state, action):
        grid, modules = self._restore(state)
        cost, lost, flows = state.cost, state.lost, [0.0] * 5
        for _ in range(PERIOD):
            spent, short, hour_flows = self._hour(grid, modules, action.mode, action.genset == "on")
            cost += spent
            lost += short if short > TOLERANCE else 0.0
            flows = [a + b for a, b in zip(flows, hour_flows)]
        return self._read(state.hour + PERIOD, float(modules["battery"].current_charge), cost, lost,
                          flows, state.depth + 1)

    def _read(self, hour, charge, cost, lost, flows, depth):
        inst = self.instance
        return MicrogridState(hour, charge, cost, lost, flows, capacity=self._capacity(),
                              target=inst["target"], reserve=inst["reserve"],
                              horizon=inst["horizon"], start=inst["start"], depth=depth)

    # ------------------------------------------------------------------- contract

    def reset(self):
        if self.instance is None:
            self.set_index(0)
        capacity = self._capacity()
        self.state = self._read(0, self.instance["soc"] * capacity, 0.0, 0.0, (0, 0, 0, 0, 0), 0)
        _, modules = self._microgrid()
        return self.state, {"index": self.index, "microgrid": self.instance["microgrid"],
                            "start": self.instance["start"], "horizon": self.instance["horizon"],
                            "target": self.instance["target"], "reserve": self.instance["reserve"],
                            "genset": "genset" in modules, "grid": "grid" in modules,
                            "generated": self.index is None}

    def is_goal(self, state):
        return (state.hour >= self.instance["horizon"] and state.lost == 0
                and state.cost <= self.instance["target"] and state.charge >= self._reserve_charge())

    def is_terminal(self, state):
        return state.lost > 0 or (state.hour >= self.instance["horizon"] and not self.is_goal(state))

    def get_actions(self, state=None):
        if self.instance is None:
            self.set_index(0)
        switches = ("off", "on") if "genset" in self._microgrid()[1] else ("off",)
        return [MicrogridAction(mode, genset) for mode in MODES for genset in switches]

    def successors(self, state):
        if self.is_goal(state) or self.is_terminal(state):
            return []
        return [(action, self._period(state, action)) for action in self.get_actions()]

    def __advance__(self, state, action):
        if self.is_goal(state) or self.is_terminal(state):
            return state
        if not isinstance(action, MicrogridAction):
            action = MicrogridAction.parse(action)
        return self._period(state, action)

    def close(self):
        self.__grid__ = None


#: The bundled instances: `generate_instance(seed)` for the seed beside each, embedded as the
#: plain data `set_instance` takes, with the plan each was accepted on in
#: `tests/data/microgrid_solutions.json`.
DAYS = (
    # seed 9000; microgrid 2 (genset), 36 hours from day 255, 06:00; 12-step plan
    {"microgrid": 2, "start": 6126, "horizon": 36, "soc": 0.51, "reserve": 0.51, "target": 202446.86},
    # seed 9001; microgrid 23 (genset), 48 hours from day 112, 18:00; 16-step plan
    {"microgrid": 23, "start": 2706, "horizon": 48, "soc": 0.6, "reserve": 0.6, "target": 70258.42},
    # seed 9002; microgrid 0 (grid), 48 hours from day 97, 00:00; 16-step plan
    {"microgrid": 0, "start": 2328, "horizon": 48, "soc": 0.34, "reserve": 0.34, "target": 3574.36},
    # seed 9003; microgrid 12 (grid), 48 hours from day 177, 06:00; 16-step plan
    {"microgrid": 12, "start": 4254, "horizon": 48, "soc": 0.39, "reserve": 0.39, "target": 248275.04},
    # seed 9004; microgrid 0 (grid), 36 hours from day 240, 12:00; 12-step plan
    {"microgrid": 0, "start": 5772, "horizon": 36, "soc": 0.53, "reserve": 0.53, "target": 3860.26},
    # seed 9005; microgrid 8 (genset, grid), 36 hours from day 224, 00:00; 12-step plan
    {"microgrid": 8, "start": 5376, "horizon": 36, "soc": 0.77, "reserve": 0.77, "target": 126935.84},
    # seed 9006; microgrid 1 (genset, grid), 48 hours from day 133, 00:00; 16-step plan
    {"microgrid": 1, "start": 3192, "horizon": 48, "soc": 0.72, "reserve": 0.72, "target": 161431.93},
    # seed 9007; microgrid 2 (genset), 48 hours from day 255, 12:00; 16-step plan
    {"microgrid": 2, "start": 6132, "horizon": 48, "soc": 0.39, "reserve": 0.39, "target": 307121.67},
    # seed 9008; microgrid 24 (genset, grid), 24 hours from day 11, 12:00; 8-step plan
    {"microgrid": 24, "start": 276, "horizon": 24, "soc": 0.75, "reserve": 0.75, "target": 162404.75},
    # seed 9009; microgrid 20 (genset), 36 hours from day 127, 06:00; 12-step plan
    {"microgrid": 20, "start": 3054, "horizon": 36, "soc": 0.78, "reserve": 0.78, "target": 286086.59},
    # seed 9010; microgrid 7 (genset), 48 hours from day 47, 06:00; 16-step plan
    {"microgrid": 7, "start": 1134, "horizon": 48, "soc": 0.78, "reserve": 0.78, "target": 53920.3},
    # seed 9011; microgrid 9 (genset, grid), 48 hours from day 191, 18:00; 16-step plan
    {"microgrid": 9, "start": 4602, "horizon": 48, "soc": 0.76, "reserve": 0.76, "target": 188121.29},
    # seed 9012; microgrid 9 (genset, grid), 48 hours from day 181, 00:00; 16-step plan
    {"microgrid": 9, "start": 4344, "horizon": 48, "soc": 0.78, "reserve": 0.78, "target": 297594.36},
    # seed 9013; microgrid 5 (genset), 48 hours from day 251, 00:00; 16-step plan
    {"microgrid": 5, "start": 6024, "horizon": 48, "soc": 0.74, "reserve": 0.74, "target": 251473.78},
    # seed 9014; microgrid 12 (grid), 48 hours from day 37, 12:00; 16-step plan
    {"microgrid": 12, "start": 900, "horizon": 48, "soc": 0.33, "reserve": 0.33, "target": 242526.42},
    # seed 9015; microgrid 16 (grid), 48 hours from day 322, 12:00; 16-step plan
    {"microgrid": 16, "start": 7740, "horizon": 48, "soc": 0.65, "reserve": 0.65, "target": 148303.24},
    # seed 9016; microgrid 2 (genset), 48 hours from day 120, 06:00; 16-step plan
    {"microgrid": 2, "start": 2886, "horizon": 48, "soc": 0.74, "reserve": 0.74, "target": 277061.29},
    # seed 9017; microgrid 20 (genset), 48 hours from day 114, 18:00; 16-step plan
    {"microgrid": 20, "start": 2754, "horizon": 48, "soc": 0.44, "reserve": 0.44, "target": 269173.72},
    # seed 9018; microgrid 12 (grid), 36 hours from day 231, 00:00; 12-step plan
    {"microgrid": 12, "start": 5544, "horizon": 36, "soc": 0.55, "reserve": 0.55, "target": 167715.26},
    # seed 9019; microgrid 3 (genset), 36 hours from day 209, 12:00; 12-step plan
    {"microgrid": 3, "start": 5028, "horizon": 36, "soc": 0.63, "reserve": 0.63, "target": 134562.23},
    # seed 9020; microgrid 12 (grid), 48 hours from day 189, 06:00; 16-step plan
    {"microgrid": 12, "start": 4542, "horizon": 48, "soc": 0.34, "reserve": 0.34, "target": 231872.54},
    # seed 9021; microgrid 21 (genset), 24 hours from day 181, 12:00; 8-step plan
    {"microgrid": 21, "start": 4356, "horizon": 24, "soc": 0.74, "reserve": 0.74, "target": 429856.42},
    # seed 9022; microgrid 2 (genset), 48 hours from day 151, 12:00; 16-step plan
    {"microgrid": 2, "start": 3636, "horizon": 48, "soc": 0.62, "reserve": 0.62, "target": 258413.37},
    # seed 9023; microgrid 16 (grid), 48 hours from day 40, 18:00; 16-step plan
    {"microgrid": 16, "start": 978, "horizon": 48, "soc": 0.78, "reserve": 0.78, "target": 60546.53},
    # seed 9024; microgrid 16 (grid), 24 hours from day 270, 12:00; 8-step plan
    {"microgrid": 16, "start": 6492, "horizon": 24, "soc": 0.36, "reserve": 0.36, "target": 116857.9},
    # seed 9025; microgrid 16 (grid), 24 hours from day 358, 00:00; 8-step plan
    {"microgrid": 16, "start": 8592, "horizon": 24, "soc": 0.35, "reserve": 0.35, "target": 35840.16},
    # seed 9026; microgrid 20 (genset), 48 hours from day 117, 18:00; 16-step plan
    {"microgrid": 20, "start": 2826, "horizon": 48, "soc": 0.55, "reserve": 0.55, "target": 155876.98},
    # seed 9027; microgrid 23 (genset), 36 hours from day 233, 06:00; 12-step plan
    {"microgrid": 23, "start": 5598, "horizon": 36, "soc": 0.62, "reserve": 0.62, "target": 44681.92},
    # seed 9028; microgrid 16 (grid), 36 hours from day 313, 00:00; 12-step plan
    {"microgrid": 16, "start": 7512, "horizon": 36, "soc": 0.48, "reserve": 0.48, "target": 108552.79},
    # seed 9029; microgrid 17 (genset), 36 hours from day 41, 06:00; 12-step plan
    {"microgrid": 17, "start": 990, "horizon": 36, "soc": 0.79, "reserve": 0.79, "target": 73558.76},
    # seed 9030; microgrid 0 (grid), 36 hours from day 259, 12:00; 12-step plan
    {"microgrid": 0, "start": 6228, "horizon": 36, "soc": 0.71, "reserve": 0.71, "target": 4025.34},
    # seed 9031; microgrid 7 (genset), 48 hours from day 166, 00:00; 16-step plan
    {"microgrid": 7, "start": 3984, "horizon": 48, "soc": 0.56, "reserve": 0.56, "target": 63036.78},
    # seed 9032; microgrid 2 (genset), 48 hours from day 232, 18:00; 16-step plan
    {"microgrid": 2, "start": 5586, "horizon": 48, "soc": 0.79, "reserve": 0.79, "target": 316932.83},
    # seed 9033; microgrid 12 (grid), 24 hours from day 152, 12:00; 8-step plan
    {"microgrid": 12, "start": 3660, "horizon": 24, "soc": 0.51, "reserve": 0.51, "target": 97124.48},
    # seed 9034; microgrid 14 (grid), 48 hours from day 313, 18:00; 16-step plan
    {"microgrid": 14, "start": 7530, "horizon": 48, "soc": 0.71, "reserve": 0.71, "target": 61645.17},
    # seed 9035; microgrid 0 (grid), 48 hours from day 172, 06:00; 16-step plan
    {"microgrid": 0, "start": 4134, "horizon": 48, "soc": 0.38, "reserve": 0.38, "target": 4159.39},
    # seed 9036; microgrid 7 (genset), 48 hours from day 284, 18:00; 16-step plan
    {"microgrid": 7, "start": 6834, "horizon": 48, "soc": 0.47, "reserve": 0.47, "target": 69157.51},
    # seed 9037; microgrid 12 (grid), 48 hours from day 227, 00:00; 16-step plan
    {"microgrid": 12, "start": 5448, "horizon": 48, "soc": 0.36, "reserve": 0.36, "target": 293683.86},
    # seed 9038; microgrid 2 (genset), 24 hours from day 78, 00:00; 8-step plan
    {"microgrid": 2, "start": 1872, "horizon": 24, "soc": 0.76, "reserve": 0.76, "target": 151681.69},
    # seed 9039; microgrid 16 (grid), 24 hours from day 194, 18:00; 8-step plan
    {"microgrid": 16, "start": 4674, "horizon": 24, "soc": 0.45, "reserve": 0.45, "target": 60235.9},
    # seed 9040; microgrid 16 (grid), 48 hours from day 203, 00:00; 16-step plan
    {"microgrid": 16, "start": 4872, "horizon": 48, "soc": 0.45, "reserve": 0.45, "target": 123216.37},
    # seed 9041; microgrid 2 (genset), 48 hours from day 246, 18:00; 16-step plan
    {"microgrid": 2, "start": 5922, "horizon": 48, "soc": 0.31, "reserve": 0.31, "target": 394109.34},
    # seed 9042; microgrid 0 (grid), 36 hours from day 99, 00:00; 12-step plan
    {"microgrid": 0, "start": 2376, "horizon": 36, "soc": 0.51, "reserve": 0.51, "target": 3286.94},
    # seed 9043; microgrid 5 (genset), 24 hours from day 132, 00:00; 8-step plan
    {"microgrid": 5, "start": 3168, "horizon": 24, "soc": 0.61, "reserve": 0.61, "target": 103821.54},
    # seed 9044; microgrid 16 (grid), 48 hours from day 271, 06:00; 16-step plan
    {"microgrid": 16, "start": 6510, "horizon": 48, "soc": 0.69, "reserve": 0.69, "target": 133448.92},
    # seed 9045; microgrid 10 (genset, grid), 24 hours from day 314, 12:00; 8-step plan
    {"microgrid": 10, "start": 7548, "horizon": 24, "soc": 0.69, "reserve": 0.69, "target": 278002.98},
    # seed 9046; microgrid 4 (grid), 24 hours from day 110, 18:00; 8-step plan
    {"microgrid": 4, "start": 2658, "horizon": 24, "soc": 0.79, "reserve": 0.79, "target": 11944.72},
    # seed 9047; microgrid 0 (grid), 36 hours from day 155, 00:00; 12-step plan
    {"microgrid": 0, "start": 3720, "horizon": 36, "soc": 0.66, "reserve": 0.66, "target": 2553.82},
    # seed 9048; microgrid 16 (grid), 48 hours from day 360, 12:00; 16-step plan
    {"microgrid": 16, "start": 8652, "horizon": 48, "soc": 0.78, "reserve": 0.78, "target": 200338.71},
    # seed 9049; microgrid 15 (genset), 36 hours from day 128, 18:00; 12-step plan
    {"microgrid": 15, "start": 3090, "horizon": 36, "soc": 0.68, "reserve": 0.68, "target": 115611.27},
    # seed 9050; microgrid 9 (genset, grid), 36 hours from day 147, 12:00; 12-step plan
    {"microgrid": 9, "start": 3540, "horizon": 36, "soc": 0.41, "reserve": 0.41, "target": 125165.66},
    # seed 9051; microgrid 3 (genset), 24 hours from day 55, 06:00; 8-step plan
    {"microgrid": 3, "start": 1326, "horizon": 24, "soc": 0.53, "reserve": 0.53, "target": 106744.5},
    # seed 9052; microgrid 20 (genset), 48 hours from day 182, 06:00; 16-step plan
    {"microgrid": 20, "start": 4374, "horizon": 48, "soc": 0.32, "reserve": 0.32, "target": 168481.69},
    # seed 9053; microgrid 21 (genset), 48 hours from day 277, 00:00; 16-step plan
    {"microgrid": 21, "start": 6648, "horizon": 48, "soc": 0.31, "reserve": 0.31, "target": 1097583.51},
    # seed 9054; microgrid 3 (genset), 48 hours from day 209, 06:00; 16-step plan
    {"microgrid": 3, "start": 5022, "horizon": 48, "soc": 0.5, "reserve": 0.5, "target": 201232.4},
    # seed 9055; microgrid 16 (grid), 24 hours from day 106, 00:00; 8-step plan
    {"microgrid": 16, "start": 2544, "horizon": 24, "soc": 0.42, "reserve": 0.42, "target": 57743.55},
    # seed 9056; microgrid 16 (grid), 36 hours from day 304, 06:00; 12-step plan
    {"microgrid": 16, "start": 7302, "horizon": 36, "soc": 0.71, "reserve": 0.71, "target": 263663.75},
    # seed 9057; microgrid 5 (genset), 48 hours from day 127, 18:00; 16-step plan
    {"microgrid": 5, "start": 3066, "horizon": 48, "soc": 0.54, "reserve": 0.54, "target": 167037.73},
    # seed 9058; microgrid 15 (genset), 48 hours from day 258, 18:00; 16-step plan
    {"microgrid": 15, "start": 6210, "horizon": 48, "soc": 0.69, "reserve": 0.69, "target": 150701.25},
    # seed 9059; microgrid 0 (grid), 48 hours from day 110, 12:00; 16-step plan
    {"microgrid": 0, "start": 2652, "horizon": 48, "soc": 0.33, "reserve": 0.33, "target": 4221.33},
    # seed 9060; microgrid 4 (grid), 36 hours from day 222, 00:00; 12-step plan
    {"microgrid": 4, "start": 5328, "horizon": 36, "soc": 0.72, "reserve": 0.72, "target": 16563.78},
    # seed 9061; microgrid 11 (grid), 48 hours from day 215, 06:00; 16-step plan
    {"microgrid": 11, "start": 5166, "horizon": 48, "soc": 0.68, "reserve": 0.68, "target": 46114.31},
    # seed 9062; microgrid 16 (grid), 48 hours from day 342, 12:00; 16-step plan
    {"microgrid": 16, "start": 8220, "horizon": 48, "soc": 0.69, "reserve": 0.69, "target": 99150.03},
    # seed 9063; microgrid 12 (grid), 24 hours from day 103, 00:00; 8-step plan
    {"microgrid": 12, "start": 2472, "horizon": 24, "soc": 0.33, "reserve": 0.33, "target": 119270.87},
    # seed 9064; microgrid 12 (grid), 36 hours from day 207, 12:00; 12-step plan
    {"microgrid": 12, "start": 4980, "horizon": 36, "soc": 0.3, "reserve": 0.3, "target": 290493.08},
    # seed 9065; microgrid 20 (genset), 36 hours from day 320, 06:00; 12-step plan
    {"microgrid": 20, "start": 7686, "horizon": 36, "soc": 0.63, "reserve": 0.63, "target": 218826.18},
    # seed 9066; microgrid 21 (genset), 36 hours from day 294, 00:00; 12-step plan
    {"microgrid": 21, "start": 7056, "horizon": 36, "soc": 0.38, "reserve": 0.38, "target": 863832.33},
    # seed 9067; microgrid 4 (grid), 24 hours from day 114, 00:00; 8-step plan
    {"microgrid": 4, "start": 2736, "horizon": 24, "soc": 0.6, "reserve": 0.6, "target": 17791.26},
    # seed 9068; microgrid 9 (genset, grid), 24 hours from day 253, 12:00; 8-step plan
    {"microgrid": 9, "start": 6084, "horizon": 24, "soc": 0.4, "reserve": 0.4, "target": 162197.61},
    # seed 9069; microgrid 4 (grid), 24 hours from day 120, 00:00; 8-step plan
    {"microgrid": 4, "start": 2880, "horizon": 24, "soc": 0.69, "reserve": 0.69, "target": 14322.83},
    # seed 9070; microgrid 1 (genset, grid), 36 hours from day 360, 18:00; 12-step plan
    {"microgrid": 1, "start": 8658, "horizon": 36, "soc": 0.61, "reserve": 0.61, "target": 155048.57},
    # seed 9071; microgrid 12 (grid), 24 hours from day 171, 12:00; 8-step plan
    {"microgrid": 12, "start": 4116, "horizon": 24, "soc": 0.5, "reserve": 0.5, "target": 139445.3},
    # seed 9072; microgrid 21 (genset), 36 hours from day 85, 18:00; 12-step plan
    {"microgrid": 21, "start": 2058, "horizon": 36, "soc": 0.64, "reserve": 0.64, "target": 788438.22},
    # seed 9073; microgrid 16 (grid), 36 hours from day 335, 00:00; 12-step plan
    {"microgrid": 16, "start": 8040, "horizon": 36, "soc": 0.49, "reserve": 0.49, "target": 76856.29},
    # seed 9074; microgrid 1 (genset, grid), 48 hours from day 323, 18:00; 16-step plan
    {"microgrid": 1, "start": 7770, "horizon": 48, "soc": 0.59, "reserve": 0.59, "target": 258027.04},
    # seed 9075; microgrid 21 (genset), 48 hours from day 47, 12:00; 16-step plan
    {"microgrid": 21, "start": 1140, "horizon": 48, "soc": 0.68, "reserve": 0.68, "target": 1014483.76},
    # seed 9076; microgrid 16 (grid), 48 hours from day 119, 12:00; 16-step plan
    {"microgrid": 16, "start": 2868, "horizon": 48, "soc": 0.64, "reserve": 0.64, "target": 107926.89},
    # seed 9077; microgrid 20 (genset), 48 hours from day 45, 12:00; 16-step plan
    {"microgrid": 20, "start": 1092, "horizon": 48, "soc": 0.72, "reserve": 0.72, "target": 337246.05},
    # seed 9078; microgrid 4 (grid), 36 hours from day 114, 06:00; 12-step plan
    {"microgrid": 4, "start": 2742, "horizon": 36, "soc": 0.45, "reserve": 0.45, "target": 16300.7},
    # seed 9079; microgrid 9 (genset, grid), 36 hours from day 260, 06:00; 12-step plan
    {"microgrid": 9, "start": 6246, "horizon": 36, "soc": 0.78, "reserve": 0.78, "target": 159297.76},
    # seed 9080; microgrid 12 (grid), 36 hours from day 173, 18:00; 12-step plan
    {"microgrid": 12, "start": 4170, "horizon": 36, "soc": 0.59, "reserve": 0.59, "target": 271644.19},
    # seed 9081; microgrid 12 (grid), 36 hours from day 268, 00:00; 12-step plan
    {"microgrid": 12, "start": 6432, "horizon": 36, "soc": 0.34, "reserve": 0.34, "target": 295970.73},
    # seed 9082; microgrid 15 (genset), 48 hours from day 160, 12:00; 16-step plan
    {"microgrid": 15, "start": 3852, "horizon": 48, "soc": 0.38, "reserve": 0.38, "target": 111372.72},
    # seed 9083; microgrid 20 (genset), 48 hours from day 180, 18:00; 16-step plan
    {"microgrid": 20, "start": 4338, "horizon": 48, "soc": 0.64, "reserve": 0.64, "target": 147387.75},
    # seed 9084; microgrid 24 (genset, grid), 48 hours from day 286, 06:00; 16-step plan
    {"microgrid": 24, "start": 6870, "horizon": 48, "soc": 0.34, "reserve": 0.34, "target": 91638.37},
    # seed 9085; microgrid 1 (genset, grid), 24 hours from day 221, 06:00; 8-step plan
    {"microgrid": 1, "start": 5310, "horizon": 24, "soc": 0.46, "reserve": 0.46, "target": 164704.24},
    # seed 9086; microgrid 4 (grid), 36 hours from day 169, 06:00; 12-step plan
    {"microgrid": 4, "start": 4062, "horizon": 36, "soc": 0.31, "reserve": 0.31, "target": 22935.25},
    # seed 9087; microgrid 10 (genset, grid), 36 hours from day 245, 12:00; 12-step plan
    {"microgrid": 10, "start": 5892, "horizon": 36, "soc": 0.45, "reserve": 0.45, "target": 260338.34},
    # seed 9088; microgrid 2 (genset), 24 hours from day 148, 18:00; 8-step plan
    {"microgrid": 2, "start": 3570, "horizon": 24, "soc": 0.55, "reserve": 0.55, "target": 128366.39},
    # seed 9089; microgrid 23 (genset), 36 hours from day 129, 06:00; 12-step plan
    {"microgrid": 23, "start": 3102, "horizon": 36, "soc": 0.46, "reserve": 0.46, "target": 43248.92},
    # seed 9090; microgrid 21 (genset), 36 hours from day 253, 06:00; 12-step plan
    {"microgrid": 21, "start": 6078, "horizon": 36, "soc": 0.73, "reserve": 0.73, "target": 611192.43},
    # seed 9091; microgrid 11 (grid), 36 hours from day 153, 06:00; 12-step plan
    {"microgrid": 11, "start": 3678, "horizon": 36, "soc": 0.64, "reserve": 0.64, "target": 19106.61},
    # seed 9092; microgrid 10 (genset, grid), 48 hours from day 157, 12:00; 16-step plan
    {"microgrid": 10, "start": 3780, "horizon": 48, "soc": 0.68, "reserve": 0.68, "target": 464180.01},
    # seed 9093; microgrid 16 (grid), 48 hours from day 43, 06:00; 16-step plan
    {"microgrid": 16, "start": 1038, "horizon": 48, "soc": 0.59, "reserve": 0.59, "target": 162499.76},
    # seed 9094; microgrid 21 (genset), 48 hours from day 255, 06:00; 16-step plan
    {"microgrid": 21, "start": 6126, "horizon": 48, "soc": 0.58, "reserve": 0.58, "target": 1143089.27},
    # seed 9095; microgrid 5 (genset), 48 hours from day 97, 12:00; 16-step plan
    {"microgrid": 5, "start": 2340, "horizon": 48, "soc": 0.75, "reserve": 0.75, "target": 230660.5},
    # seed 9096; microgrid 11 (grid), 48 hours from day 104, 18:00; 16-step plan
    {"microgrid": 11, "start": 2514, "horizon": 48, "soc": 0.33, "reserve": 0.33, "target": 62623.68},
    # seed 9097; microgrid 20 (genset), 36 hours from day 342, 06:00; 12-step plan
    {"microgrid": 20, "start": 8214, "horizon": 36, "soc": 0.59, "reserve": 0.59, "target": 63778.2},
    # seed 9098; microgrid 24 (genset, grid), 24 hours from day 167, 06:00; 8-step plan
    {"microgrid": 24, "start": 4014, "horizon": 24, "soc": 0.32, "reserve": 0.32, "target": 47857.0},
    # seed 9099; microgrid 0 (grid), 24 hours from day 221, 06:00; 8-step plan
    {"microgrid": 0, "start": 5310, "horizon": 24, "soc": 0.55, "reserve": 0.55, "target": 2121.71},
)
