"""Reservoir operations on pywr: two reservoirs in series, a city, a farm and a river to keep
flowing, a release and a farm allocation to set each month, and a year to get through with
the farm's shortfall under the target and both reservoirs above their reserves at the end.

pywr (https://github.com/pywr/pywr, GPL-3.0-or-later; Tomlinson, Arnott and Harou 2020,
https://doi.org/10.1016/j.envsoft.2020.104635) is the water resource system simulator from
the University of Manchester used across the UK water industry: a network of storages, demands
and links whose flows are allocated each step by a linear programme over the nodes' priorities.
It is a dependency installed from PyPI; nothing of it is included here.

The decisions are the operator's: how much to release from the upper reservoir through the
turbine this month, and whether the farm gets its full demand or is rationed to half of it,
which counts as shortfall but keeps water back for the city, the river and the reserves. Everything else is
pywr's allocation: the river's minimum flow first, then the city, then the farm, with whatever
the lower reservoir and the release can give, and spill to the sea only when it is full.

## Trajectory constraints as a goal

The river's minimum flow and the city's supply must hold every month, so a month that shorts
either is `is_terminal`. The farm's shortfall accumulates in the state, and the goal is the
year complete with that shortfall under the target and both reservoirs above their reserves,
which is a plan's final state. The target is the least shortfall any fixed release policy
manages without breaking a constraint, so the planner is asked to do at least as well as the
best rule by varying the release month by month.

## Determinism and state

pywr is deterministic and the year's inflows are the instance's, so a state is what a month
leaves behind: the month, the two volumes and the shortfalls so far. Two release histories that
leave the same volumes are one state, and the environment replays the year from January to
expand one, which takes a few milliseconds.
"""
from planiverse.environments.base import Environment
from planiverse.environments.generation import draw_until, rng

MONTHS = 12
#: The releases a month may be set to (volume units a month) and the farm's allocation.
RELEASES = (2, 4, 6, 8, 10)
SHARES = {"full": 1.0, "half": 0.5}
#: The city must get at least this share of its demand every month.
CITY_FLOOR = 0.9
#: A seasonal shape for the hills' inflow and the farm's demand, January to December.
INFLOW_SHAPE = (9, 8, 7, 6, 4, 3, 2, 2, 3, 5, 7, 8)
FARM_SHAPE = (1, 1, 2, 4, 6, 8, 8, 7, 5, 2, 1, 1)


def _pywr():
    import pywr.core
    import pywr.parameters
    import pywr.recorders
    return pywr


class ReservoirAction:
    """`release(r, share)`: `r` from `RELEASES`, `share` `full` or `half`."""

    def __init__(self, release, share):
        if release not in RELEASES or share not in SHARES:
            raise ValueError(f"unknown setting: release {release}, share {share!r}")
        self.release, self.share = release, share
        self.name = f"release({release}, {share})"

    @classmethod
    def parse(cls, text):
        inside = str(text).strip()[len("release("):-1]
        release, share = (p.strip() for p in inside.split(","))
        return cls(int(release), share)

    def cost(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, ReservoirAction) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    __repr__ = __str__


ACTIONS = tuple(ReservoirAction(r, s) for r in RELEASES for s in SHARES)


class ReservoirState:
    """A month's end: the volumes, the shortfalls so far and the last month's deliveries.
    Identity is the month, the volumes and the shortfalls; `settings` is kept for replay."""

    def __init__(self, settings, upper, lower, farm_short, city_short, river_short, delivered,
                 target=0, reserves=(0, 0), depth=0):
        self.settings = tuple(settings)
        self.month = len(self.settings)
        self.upper, self.lower = round(upper, 2), round(lower, 2)
        self.farm_short, self.city_short, self.river_short = round(farm_short, 2), round(city_short, 2), round(river_short, 2)
        self.delivered = tuple(round(v, 2) for v in delivered)      # city, farm, river, turbine
        self.target, self.reserves, self.depth = target, tuple(reserves), depth
        self.key = (self.month, self.upper, self.lower, self.farm_short, self.city_short, self.river_short)
        literals = [f"month({self.month})", f"upper({int(self.upper) // 5 * 5})", f"lower({int(self.lower) // 5 * 5})",
                    f"farm_short({int(self.farm_short)})"]
        if self.city_short > 0:
            literals.append("city_short")
        if self.river_short > 0:
            literals.append("river_short")
        self.literals = frozenset(literals)

    def __eq__(self, other):
        return isinstance(other, ReservoirState) and self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        city, farm, river, turbine = self.delivered
        return (f"month {self.month}: upper {self.upper:.0f}, lower {self.lower:.0f}; last month released "
                f"{turbine:.0f}, city got {city:.1f}, farm {farm:.1f}, river {river:.1f}; farm short "
                f"{self.farm_short:.1f} of {self.target} allowed")

    def __repr__(self):
        return f"<ReservoirState(month={self.month}, upper={self.upper:.0f}, lower={self.lower:.0f}, farm_short={self.farm_short:.1f})>"


class Setting:
    """A pywr parameter the environment sets before each month."""

    def __new__(cls, model, level):
        pywr = _pywr()

        class _Setting(pywr.parameters.Parameter):
            def __init__(self, model, level):
                super().__init__(model)
                self.level = level

            def value(self, timestep, scenario_index):
                return self.level

        return _Setting(model, level)


def reference_plans():
    """The fixed policies the target is set off: each release all year with the farm on full,
    each with the farm on half through the summer, and the seasonal patterns that release high
    in summer and low in winter, with the farm on full or on half."""
    summer = range(4, 9)
    plans = [[ReservoirAction(r, "full")] * MONTHS for r in RELEASES]
    plans += [[ReservoirAction(r, "half" if m in summer else "full") for m in range(MONTHS)] for r in RELEASES]
    for low, high in ((2, 8), (2, 10), (4, 8), (4, 10)):
        for share in SHARES:
            plans.append([ReservoirAction(high if m in summer else low, share if m in summer else "full")
                          for m in range(MONTHS)])
    return plans


class ReservoirEnv(Environment):
    """A year with the farm's shortfall under the target and the reservoirs above their reserves."""

    def __init__(self):
        super().__init__("reservoir")
        self.instance = None
        self.index = None
        self.state = None
        self.state_history = []
        self.witness = None
        self.witness_expansions = None
        self.__model__ = None
        self.__loaded__ = None

    # ------------------------------------------------------------------ instances

    def set_index(self, index):
        if not 0 <= index < len(YEARS):
            raise IndexError(f"Invalid index: {index}. There are {len(YEARS)} years, so the "
                             f"index must be 0-{len(YEARS) - 1}.")
        self.set_instance(YEARS[index])
        self.index = index

    def set_instance(self, instance):
        """Select a year: `inflow_upper` and `inflow_lower` (twelve months), `city` (a month's
        demand), `farm` (twelve), `river` (minimum flow), `capacity` and `initial` (upper,
        lower), `reserves` (upper, lower, at year end) and `target` (farm shortfall allowed)."""
        for key in ("inflow_upper", "inflow_lower", "city", "farm", "river", "capacity", "initial", "reserves", "target"):
            if key not in instance:
                raise ValueError(f"a year needs `{key}`")
        self.instance = {"inflow_upper": [float(v) for v in instance["inflow_upper"]],
                         "inflow_lower": [float(v) for v in instance["inflow_lower"]],
                         "city": float(instance["city"]), "farm": [float(v) for v in instance["farm"]],
                         "river": float(instance["river"]),
                         "capacity": [float(v) for v in instance["capacity"]],
                         "initial": [float(v) for v in instance["initial"]],
                         "reserves": [float(v) for v in instance["reserves"]],
                         "target": float(instance["target"])}
        self.index = None
        self.witness = self.witness_expansions = None
        self.__model__ = None
        self.__loaded__ = None

    def generate_instance(self, seed=None, attempts=60):
        """Draw a year, select it, and return it as the dict `set_instance` takes.

        Inflows are the seasonal shapes with seeded noise, the demands, capacities, initial
        volumes and reserves are drawn round them, and the fixed policies of `reference_plans`
        are run; the target is the least farm shortfall any of them ends the year with inside
        the constraints. A draw is thrown back when none does, or when a steady release all
        year does as well as any, since then nothing needs deciding.
        """
        random_, _ = rng(seed)
        found = {}

        def draw(attempt):
            scale = random_.uniform(0.8, 1.3)
            return {"inflow_upper": [round(v * scale * random_.uniform(0.7, 1.3), 1) for v in INFLOW_SHAPE],
                    "inflow_lower": [round(v * scale * random_.uniform(0.2, 0.5), 1) for v in INFLOW_SHAPE],
                    "city": round(random_.uniform(3.5, 5.5), 1),
                    "farm": [round(v * random_.uniform(0.8, 1.2), 1) for v in FARM_SHAPE],
                    "river": round(random_.uniform(1.0, 2.5), 1),
                    "capacity": [random_.choice((80, 100, 120)), random_.choice((30, 40, 50))],
                    "initial": [0, 0], "reserves": [0, 0], "target": 0}

        def accept(instance):
            instance["initial"] = [round(instance["capacity"][0] * random_.uniform(0.5, 0.8)),
                                   round(instance["capacity"][1] * random_.uniform(0.5, 0.8))]
            instance["reserves"] = [round(instance["capacity"][0] * random_.uniform(0.3, 0.5)),
                                    round(instance["capacity"][1] * random_.uniform(0.3, 0.5))]
            self.set_instance(instance)
            self.instance["target"] = float("inf")
            plans = reference_plans()
            outcomes = []
            for plan in plans:
                last = self.simulate(plan)[-1]
                ok = (last.month == MONTHS and last.city_short == 0 and last.river_short == 0
                      and last.upper >= instance["reserves"][0] and last.lower >= instance["reserves"][1])
                outcomes.append((last.farm_short, plan) if ok else None)
            feasible = [o for o in outcomes if o is not None]
            if not feasible:
                return False
            best = min(feasible, key=lambda o: o[0])
            steady = [o for o in outcomes[:len(RELEASES)] if o is not None]      # the same release all year
            if steady and min(o[0] for o in steady) <= best[0]:
                return False                     # a steady release is already best: nothing to plan
            instance["target"] = round(best[0], 2)        # the state rounds to two decimals too
            found["plan"], found["measured"] = best[1], len(plans)
            return True

        instance = draw_until(draw, accept, attempts, "reservoir year")
        self.set_instance(instance)
        self.witness, self.witness_expansions = found["plan"], found["measured"]
        return instance

    # ------------------------------------------------------------------ the simulator

    def _build(self):
        pywr = _pywr()
        core, parameters, recorders = pywr.core, pywr.parameters, pywr.recorders
        inst = self.instance
        model = core.Model(start="2020-01-01", end=f"2020-01-{MONTHS:02d}", timestep=1)
        hills = core.Input(model, "hills", max_flow=parameters.ArrayIndexedParameter(model, inst["inflow_upper"]), cost=-1)
        # pywr's costs: a negative cost on a store is a benefit for keeping water in it, so the
        # reservoirs hold what the demands do not take and spill only when full; the turbine's is
        # more negative still, so the release the operator sets is made whenever there is water.
        upper = core.Storage(model, "upper", max_volume=inst["capacity"][0], initial_volume=inst["initial"][0], cost=-5)
        release = Setting(model, RELEASES[0])
        turbine = core.Link(model, "turbine", max_flow=release, cost=-10)
        valley = core.Input(model, "valley", max_flow=parameters.ArrayIndexedParameter(model, inst["inflow_lower"]), cost=-1)
        lower = core.Storage(model, "lower", max_volume=inst["capacity"][1], initial_volume=inst["initial"][1], cost=-3)
        city = core.Output(model, "city", max_flow=inst["city"], cost=-40)
        share = Setting(model, 1.0)
        farm_demand = parameters.ArrayIndexedParameter(model, inst["farm"])
        farm = core.Output(model, "farm", max_flow=parameters.AggregatedParameter(model, [farm_demand, share], agg_func="product"), cost=-20)
        river = core.Output(model, "river", max_flow=inst["river"], cost=-60)
        sea = core.Output(model, "sea", cost=0)
        hills.connect(upper)
        upper.connect(turbine)
        turbine.connect(lower)
        valley.connect(lower)
        for node in (city, farm, river, sea):
            lower.connect(node)
        recs = [recorders.NumpyArrayNodeRecorder(model, node) for node in (city, farm, river, turbine)]
        model.setup()
        self.__model__ = {"model": model, "upper": upper, "lower": lower, "release": release, "share": share, "recs": recs}

    def _start(self):
        if self.__model__ is None:
            self._build()
        self.__model__["model"].reset()
        self.__totals__ = [0.0, 0.0, 0.0]
        self.__loaded__ = ()

    def _apply(self, action):
        m = self.__model__
        m["release"].level = float(action.release)
        m["share"].level = SHARES[action.share]
        m["model"].step()
        k = m["model"].timestepper.current.index
        city, farm, river, turbine = (float(rec.data[k, 0]) for rec in m["recs"])
        self.__totals__[0] += max(0.0, self.instance["farm"][k] - farm)      # against the full demand
        self.__totals__[1] += max(0.0, CITY_FLOOR * self.instance["city"] - city)
        self.__totals__[2] += max(0.0, self.instance["river"] - river - 1e-6)
        self.__last__ = (city, farm, river, turbine)

    def _load(self, settings):
        if self.__loaded__ != settings:
            if self.__loaded__ is None or settings[:len(self.__loaded__)] != self.__loaded__:
                self._start()
            for action in settings[len(self.__loaded__):]:
                self._apply(action)
            self.__loaded__ = settings

    def _read(self, settings, depth):
        m = self.__model__
        return ReservoirState(settings, float(m["upper"].volume[0]), float(m["lower"].volume[0]),
                              *self.__totals__, delivered=self.__last__, target=self.instance["target"],
                              reserves=self.instance["reserves"], depth=depth)

    # ------------------------------------------------------------------- contract

    def reset(self):
        if self.instance is None:
            self.set_index(0)
        self._start()
        self.__last__ = (0.0, 0.0, 0.0, 0.0)
        self.state = self._read((), 0)
        self.state_history = [self.state]
        return self.state, {"year": self.index, "target": self.instance["target"],
                            "reserves": self.instance["reserves"], "generated": self.index is None}

    def is_goal(self, state):
        return (state.month >= MONTHS and state.city_short == 0 and state.river_short == 0
                and state.farm_short <= self.instance["target"]
                and state.upper >= self.instance["reserves"][0] and state.lower >= self.instance["reserves"][1])

    def is_terminal(self, state):
        return state.city_short > 0 or state.river_short > 0 or (state.month >= MONTHS and not self.is_goal(state))

    def get_actions(self, state=None):
        return list(ACTIONS)

    def successors(self, state):
        if self.is_goal(state) or self.is_terminal(state):
            return []
        children = []
        for action in ACTIONS:
            self._load(state.settings)
            self._apply(action)
            self.__loaded__ = state.settings + (action,)
            children.append((action, self._read(self.__loaded__, state.depth + 1)))
        return children

    def __advance__(self, state, action):
        if self.is_goal(state) or self.is_terminal(state):
            return state
        if not isinstance(action, ReservoirAction):
            action = ReservoirAction.parse(action)
        self._load(state.settings)
        self._apply(action)
        self.__loaded__ = state.settings + (action,)
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
        before = self.state.farm_short
        self.state = self.__advance__(self.state, action)
        self.state_history.append(self.state)
        return self.state, before - self.state.farm_short

    def render(self):
        lines = [f"step {k}: {state}" for k, state in enumerate(self.state_history)]
        print("\n".join(lines))
        return lines

    def close(self):
        self.__model__ = None
        self.__loaded__ = None


#: The bundled years: `generate_instance(seed)` for the seed beside each, embedded as the
#: plain data `set_instance` takes, with the plan each was accepted on in
#: `tests/data/reservoir_solutions.json`.
YEARS = (
    # seed 9000; capacity [80, 40], city 5.4, river 1.0, 12-step plan
    {"inflow_upper": [9.5, 6.9, 8.8, 7.0, 4.5, 4.1, 2.4, 1.7, 3.8, 5.4, 7.4, 8.9], "inflow_lower": [2.8, 3.4, 2.6, 2.0, 1.2, 0.9, 0.9, 0.8, 0.8, 1.7, 2.0, 3.0], "city": 5.4, "farm": [1.0, 1.1, 1.8, 3.5, 6.8, 8.7, 7.6, 7.8, 5.4, 2.2, 0.9, 1.2], "river": 1.0, "capacity": [80, 40], "initial": [40, 29], "reserves": [28, 19], "target": 18.15},
    # seed 9001; capacity [80, 50], city 4.7, river 1.9, 12-step plan
    {"inflow_upper": [7.8, 6.9, 9.4, 5.4, 4.5, 4.1, 2.0, 1.9, 2.9, 6.4, 7.9, 10.2], "inflow_lower": [4.3, 3.3, 2.8, 1.8, 2.0, 1.2, 1.1, 1.0, 1.0, 2.1, 2.8, 2.1], "city": 4.7, "farm": [0.9, 0.9, 1.7, 4.2, 5.9, 7.7, 7.3, 6.7, 5.7, 2.4, 1.0, 0.9], "river": 1.9, "capacity": [80, 50], "initial": [41, 38], "reserves": [28, 20], "target": 16.65},
    # seed 9002; capacity [80, 50], city 3.7, river 1.7, 12-step plan
    {"inflow_upper": [9.1, 10.2, 6.5, 6.5, 4.2, 3.5, 2.2, 2.1, 3.1, 3.9, 9.1, 7.9], "inflow_lower": [4.5, 3.5, 2.6, 2.1, 2.1, 1.5, 0.8, 0.4, 0.8, 2.7, 3.7, 3.5], "city": 3.7, "farm": [0.8, 1.0, 2.4, 3.5, 6.8, 8.4, 8.1, 7.0, 5.8, 1.9, 1.1, 0.9], "river": 1.7, "capacity": [80, 50], "initial": [53, 28], "reserves": [39, 23], "target": 18.05},
    # seed 9003; capacity [120, 50], city 4.3, river 2.3, 12-step plan
    {"inflow_upper": [12.2, 6.2, 7.7, 6.8, 4.5, 4.1, 2.2, 1.9, 3.4, 4.9, 8.1, 7.0], "inflow_lower": [4.8, 3.0, 2.3, 2.5, 1.0, 1.3, 0.7, 0.7, 0.7, 1.2, 1.8, 2.5], "city": 4.3, "farm": [1.1, 0.9, 2.3, 3.5, 7.2, 7.3, 8.1, 6.1, 5.7, 1.8, 0.9, 1.1], "river": 2.3, "capacity": [120, 50], "initial": [62, 29], "reserves": [44, 17], "target": 17.2},
    # seed 9004; capacity [120, 50], city 5.2, river 2.4, 12-step plan
    {"inflow_upper": [11.0, 7.1, 8.0, 4.8, 5.6, 3.0, 2.2, 2.7, 2.4, 6.3, 8.5, 8.4], "inflow_lower": [3.7, 2.2, 2.1, 2.7, 2.0, 1.1, 0.6, 0.6, 1.0, 2.7, 2.0, 2.2], "city": 5.2, "farm": [1.1, 1.1, 1.9, 3.8, 5.7, 8.7, 6.8, 6.5, 5.2, 1.8, 1.1, 1.2], "river": 2.4, "capacity": [120, 50], "initial": [95, 25], "reserves": [47, 19], "target": 16.45},
    # seed 9005; capacity [100, 30], city 5.0, river 1.6, 12-step plan
    {"inflow_upper": [11.1, 6.7, 7.0, 5.0, 3.5, 3.2, 1.7, 2.4, 2.6, 5.4, 4.9, 7.5], "inflow_lower": [4.1, 2.9, 2.3, 1.6, 1.7, 1.4, 0.9, 0.9, 0.7, 1.3, 2.5, 2.8], "city": 5.0, "farm": [1.1, 0.9, 2.3, 4.5, 5.7, 7.1, 7.8, 8.4, 6.0, 2.0, 0.9, 0.8], "river": 1.6, "capacity": [100, 30], "initial": [76, 16], "reserves": [32, 10], "target": 17.5},
    # seed 9006; capacity [120, 30], city 4.1, river 2.1, 12-step plan
    {"inflow_upper": [7.3, 9.5, 10.2, 5.6, 5.4, 3.3, 2.0, 2.5, 4.2, 4.6, 8.8, 11.3], "inflow_lower": [2.8, 2.1, 2.2, 3.2, 2.2, 0.9, 0.7, 0.9, 0.9, 2.0, 3.1, 3.4], "city": 4.1, "farm": [0.9, 1.0, 1.8, 3.8, 6.2, 7.1, 8.6, 7.5, 4.1, 2.3, 0.8, 1.0], "river": 2.1, "capacity": [120, 30], "initial": [65, 21], "reserves": [45, 14], "target": 16.75},
    # seed 9007; capacity [80, 30], city 3.6, river 2.0, 12-step plan
    {"inflow_upper": [10.6, 6.4, 7.8, 6.3, 2.7, 2.7, 1.5, 1.6, 3.6, 5.2, 7.9, 7.8], "inflow_lower": [2.6, 2.0, 2.8, 1.6, 1.5, 1.1, 0.5, 1.0, 1.1, 2.3, 2.1, 3.0], "city": 3.6, "farm": [0.9, 1.1, 2.4, 3.7, 5.3, 8.1, 8.6, 8.1, 4.6, 1.7, 1.1, 0.9], "river": 2.0, "capacity": [80, 30], "initial": [40, 18], "reserves": [32, 13], "target": 17.35},
    # seed 9008; capacity [120, 40], city 5.4, river 1.1, 12-step plan
    {"inflow_upper": [9.5, 8.0, 9.4, 7.6, 4.9, 2.8, 1.6, 1.6, 3.7, 6.2, 8.0, 7.8], "inflow_lower": [4.2, 3.8, 2.4, 3.3, 1.2, 0.9, 0.5, 0.8, 1.3, 2.2, 2.1, 3.8], "city": 5.4, "farm": [1.0, 0.8, 1.7, 3.8, 7.0, 9.2, 8.9, 8.1, 4.6, 2.0, 1.0, 0.9], "river": 1.1, "capacity": [120, 40], "initial": [68, 27], "reserves": [46, 16], "target": 18.9},
    # seed 9009; capacity [100, 30], city 3.8, river 1.6, 12-step plan
    {"inflow_upper": [7.2, 7.5, 5.2, 5.4, 4.8, 3.0, 1.8, 1.7, 2.9, 4.7, 4.8, 8.7], "inflow_lower": [2.3, 3.3, 2.0, 2.8, 1.5, 1.0, 0.6, 0.7, 0.7, 2.2, 1.7, 3.1], "city": 3.8, "farm": [1.0, 1.0, 1.9, 3.4, 6.1, 7.8, 8.4, 8.1, 5.3, 1.9, 1.2, 1.0], "river": 1.6, "capacity": [100, 30], "initial": [67, 23], "reserves": [34, 10], "target": 0.0},
    # seed 9010; capacity [100, 30], city 5.0, river 1.5, 12-step plan
    {"inflow_upper": [13.4, 8.4, 7.4, 5.1, 5.1, 3.8, 3.0, 2.8, 3.8, 5.7, 6.6, 9.4], "inflow_lower": [5.0, 3.7, 2.4, 1.9, 1.8, 1.5, 0.6, 0.5, 1.5, 2.0, 1.8, 1.9], "city": 5.0, "farm": [0.9, 1.0, 1.8, 3.6, 7.0, 9.5, 9.2, 7.2, 4.0, 1.8, 0.8, 1.0], "river": 1.5, "capacity": [100, 30], "initial": [60, 21], "reserves": [33, 14], "target": 18.45},
    # seed 9011; capacity [120, 40], city 5.3, river 1.5, 12-step plan
    {"inflow_upper": [11.2, 10.3, 8.1, 7.7, 5.0, 3.7, 2.4, 2.4, 2.8, 4.3, 7.9, 7.5], "inflow_lower": [1.9, 2.2, 2.0, 2.1, 1.6, 1.0, 0.9, 0.5, 1.0, 1.3, 3.4, 3.9], "city": 5.3, "farm": [1.0, 1.0, 2.0, 4.5, 5.3, 8.5, 9.3, 7.6, 4.1, 2.1, 0.9, 1.0], "river": 1.5, "capacity": [120, 40], "initial": [69, 31], "reserves": [56, 17], "target": 17.4},
    # seed 9012; capacity [80, 40], city 3.5, river 1.4, 12-step plan
    {"inflow_upper": [9.5, 8.5, 5.2, 3.8, 3.8, 3.0, 2.0, 2.2, 3.0, 4.7, 5.7, 5.1], "inflow_lower": [2.4, 1.9, 2.7, 2.5, 1.4, 0.7, 0.8, 0.6, 0.9, 2.0, 1.3, 1.6], "city": 3.5, "farm": [0.9, 1.1, 2.2, 4.0, 5.3, 8.4, 9.0, 7.8, 5.9, 2.1, 0.9, 0.9], "river": 1.4, "capacity": [80, 40], "initial": [46, 22], "reserves": [27, 16], "target": 18.2},
    # seed 9013; capacity [80, 50], city 4.9, river 2.5, 12-step plan
    {"inflow_upper": [6.8, 9.1, 5.9, 5.0, 4.0, 3.4, 1.5, 1.5, 2.7, 3.9, 8.4, 6.6], "inflow_lower": [3.6, 3.2, 3.1, 2.6, 1.6, 0.7, 0.8, 0.4, 0.9, 1.4, 2.9, 2.9], "city": 4.9, "farm": [0.9, 0.8, 1.7, 3.9, 7.0, 7.3, 8.5, 7.0, 4.1, 1.8, 0.8, 1.0], "river": 2.5, "capacity": [80, 50], "initial": [62, 40], "reserves": [29, 18], "target": 16.95},
    # seed 9014; capacity [120, 50], city 3.8, river 2.4, 12-step plan
    {"inflow_upper": [8.6, 9.0, 5.4, 6.1, 3.3, 3.5, 1.9, 2.4, 3.3, 5.7, 5.5, 7.2], "inflow_lower": [1.9, 2.1, 2.3, 2.4, 1.7, 0.6, 0.9, 0.6, 0.6, 1.8, 1.4, 2.0], "city": 3.8, "farm": [0.9, 0.8, 1.8, 3.5, 7.2, 8.0, 6.9, 5.9, 4.2, 1.8, 1.1, 1.0], "river": 2.4, "capacity": [120, 50], "initial": [75, 39], "reserves": [47, 18], "target": 16.1},
    # seed 9015; capacity [100, 40], city 3.9, river 1.0, 12-step plan
    {"inflow_upper": [6.7, 10.9, 6.5, 5.0, 3.7, 3.8, 2.5, 2.4, 2.5, 6.5, 8.3, 7.4], "inflow_lower": [3.8, 2.6, 3.5, 3.0, 1.5, 0.9, 0.5, 0.7, 1.2, 1.3, 3.5, 2.6], "city": 3.9, "farm": [1.1, 1.1, 1.8, 4.7, 6.6, 8.3, 8.7, 8.4, 5.3, 2.1, 1.0, 0.8], "river": 1.0, "capacity": [100, 40], "initial": [59, 21], "reserves": [35, 14], "target": 0.0},
    # seed 9016; capacity [100, 30], city 4.2, river 2.0, 12-step plan
    {"inflow_upper": [11.0, 8.5, 7.0, 7.8, 4.4, 3.0, 2.3, 2.6, 3.0, 5.1, 8.8, 6.4], "inflow_lower": [4.2, 3.1, 2.6, 2.0, 1.8, 1.5, 0.7, 0.6, 1.0, 1.7, 2.5, 3.5], "city": 4.2, "farm": [1.0, 0.8, 1.9, 3.2, 5.1, 6.9, 6.4, 6.4, 4.9, 1.7, 1.2, 1.0], "river": 2.0, "capacity": [100, 30], "initial": [60, 23], "reserves": [34, 12], "target": 14.85},
    # seed 9017; capacity [100, 50], city 4.5, river 2.3, 12-step plan
    {"inflow_upper": [11.5, 10.8, 7.8, 6.4, 5.6, 3.7, 2.7, 1.8, 4.4, 7.1, 6.5, 13.0], "inflow_lower": [5.5, 3.8, 4.0, 3.0, 2.3, 1.3, 0.9, 1.0, 1.6, 1.8, 3.5, 4.2], "city": 4.5, "farm": [1.2, 0.9, 2.0, 3.5, 5.4, 9.3, 7.8, 5.8, 5.2, 1.7, 0.9, 0.9], "river": 2.3, "capacity": [100, 50], "initial": [52, 27], "reserves": [46, 15], "target": 16.75},
    # seed 9018; capacity [120, 50], city 4.9, river 1.5, 12-step plan
    {"inflow_upper": [8.4, 9.4, 7.8, 6.8, 4.5, 2.3, 2.3, 1.4, 3.5, 3.4, 5.9, 8.2], "inflow_lower": [2.8, 3.2, 1.8, 2.0, 0.8, 0.8, 0.4, 0.7, 1.0, 1.2, 1.6, 2.9], "city": 4.9, "farm": [0.9, 0.8, 2.3, 3.8, 5.3, 9.1, 7.9, 8.1, 4.4, 2.3, 0.9, 1.2], "river": 1.5, "capacity": [120, 50], "initial": [73, 36], "reserves": [55, 18], "target": 17.4},
    # seed 9019; capacity [80, 30], city 4.6, river 1.4, 12-step plan
    {"inflow_upper": [14.0, 7.9, 10.4, 6.2, 4.7, 2.8, 2.3, 2.2, 3.4, 7.6, 10.0, 11.7], "inflow_lower": [2.5, 2.6, 4.3, 3.5, 1.4, 1.8, 0.8, 1.0, 1.0, 1.3, 3.2, 4.6], "city": 4.6, "farm": [1.1, 0.8, 2.1, 3.5, 7.0, 6.6, 8.2, 7.8, 4.8, 1.7, 1.0, 0.8], "river": 1.4, "capacity": [80, 30], "initial": [42, 16], "reserves": [39, 10], "target": 17.2},
    # seed 9020; capacity [120, 40], city 5.2, river 1.9, 12-step plan
    {"inflow_upper": [8.6, 9.4, 8.3, 6.0, 4.6, 2.4, 2.1, 1.6, 3.0, 5.0, 8.4, 9.0], "inflow_lower": [3.7, 1.9, 1.5, 2.8, 1.7, 1.1, 0.6, 0.8, 0.8, 1.3, 3.2, 2.6], "city": 5.2, "farm": [0.8, 1.1, 1.6, 4.7, 6.0, 7.7, 8.8, 6.3, 5.9, 2.1, 1.2, 1.2], "river": 1.9, "capacity": [120, 40], "initial": [84, 30], "reserves": [49, 20], "target": 17.35},
    # seed 9021; capacity [120, 30], city 5.4, river 1.8, 12-step plan
    {"inflow_upper": [7.0, 7.4, 7.4, 6.3, 2.7, 2.7, 1.5, 1.9, 2.7, 5.5, 6.1, 7.3], "inflow_lower": [2.4, 2.8, 1.9, 1.1, 1.1, 0.8, 0.5, 0.5, 1.0, 1.2, 1.4, 3.4], "city": 5.4, "farm": [0.9, 0.9, 2.3, 4.5, 6.2, 6.6, 7.3, 8.1, 4.0, 2.2, 0.9, 1.0], "river": 1.8, "capacity": [120, 30], "initial": [92, 18], "reserves": [52, 12], "target": 16.1},
    # seed 9022; capacity [80, 50], city 4.6, river 1.5, 12-step plan
    {"inflow_upper": [6.9, 6.2, 5.0, 5.0, 4.4, 2.1, 1.9, 1.9, 2.1, 4.1, 4.5, 8.9], "inflow_lower": [3.3, 2.7, 2.0, 2.2, 0.7, 0.6, 0.6, 0.6, 0.9, 1.5, 1.4, 3.3], "city": 4.6, "farm": [1.1, 1.2, 1.8, 4.3, 5.2, 8.9, 8.1, 7.1, 5.5, 2.0, 1.0, 1.1], "river": 1.5, "capacity": [80, 50], "initial": [58, 36], "reserves": [33, 22], "target": 17.4},
    # seed 9023; capacity [100, 30], city 4.4, river 1.0, 12-step plan
    {"inflow_upper": [7.4, 9.6, 5.5, 6.1, 5.1, 3.6, 2.1, 2.1, 2.6, 4.9, 9.4, 9.8], "inflow_lower": [2.0, 1.9, 1.8, 2.0, 1.0, 1.5, 0.8, 0.5, 0.9, 1.3, 1.9, 3.4], "city": 4.4, "farm": [1.0, 1.1, 1.9, 3.6, 5.2, 9.1, 7.0, 7.1, 4.6, 2.3, 1.0, 1.1], "river": 1.0, "capacity": [100, 30], "initial": [67, 20], "reserves": [42, 13], "target": 16.5},
    # seed 9024; capacity [100, 40], city 3.5, river 1.9, 12-step plan
    {"inflow_upper": [9.3, 8.0, 5.1, 6.1, 3.5, 2.3, 1.4, 1.7, 1.7, 4.9, 5.2, 7.0], "inflow_lower": [3.4, 1.6, 2.2, 2.2, 0.8, 0.9, 0.6, 0.6, 0.7, 0.9, 2.6, 2.6], "city": 3.5, "farm": [0.9, 1.1, 1.8, 3.8, 7.1, 9.3, 7.2, 7.3, 5.0, 1.6, 1.0, 1.1], "river": 1.9, "capacity": [100, 40], "initial": [75, 21], "reserves": [41, 13], "target": 17.95},
    # seed 9025; capacity [120, 50], city 4.1, river 2.4, 12-step plan
    {"inflow_upper": [5.8, 6.0, 8.1, 5.3, 3.5, 2.2, 2.2, 1.7, 2.4, 3.5, 7.4, 7.4], "inflow_lower": [1.8, 3.2, 1.8, 2.6, 1.7, 0.5, 0.7, 0.5, 1.3, 1.1, 1.4, 3.3], "city": 4.1, "farm": [0.8, 1.1, 1.9, 4.0, 5.4, 6.7, 8.0, 7.3, 5.9, 2.0, 1.1, 0.8], "river": 2.4, "capacity": [120, 50], "initial": [80, 30], "reserves": [56, 15], "target": 16.65},
    # seed 9026; capacity [100, 40], city 4.2, river 1.9, 12-step plan
    {"inflow_upper": [8.3, 7.7, 6.8, 5.1, 4.5, 3.5, 1.5, 1.4, 3.0, 4.3, 7.5, 8.1], "inflow_lower": [2.4, 2.0, 1.5, 1.5, 1.8, 0.6, 0.7, 0.4, 0.6, 2.2, 2.5, 2.7], "city": 4.2, "farm": [0.9, 0.9, 2.4, 3.9, 5.5, 6.8, 8.6, 6.3, 4.2, 2.2, 0.8, 0.9], "river": 1.9, "capacity": [100, 40], "initial": [66, 30], "reserves": [48, 13], "target": 15.7},
    # seed 9027; capacity [100, 30], city 5.2, river 2.3, 12-step plan
    {"inflow_upper": [10.0, 11.3, 5.9, 7.6, 5.8, 2.4, 1.9, 2.2, 2.5, 5.0, 6.2, 11.4], "inflow_lower": [4.6, 3.7, 2.4, 3.2, 1.6, 1.4, 0.5, 0.8, 1.2, 1.4, 3.0, 2.3], "city": 5.2, "farm": [0.8, 0.8, 2.0, 4.0, 6.8, 6.9, 7.0, 5.8, 4.2, 2.0, 0.9, 0.9], "river": 2.3, "capacity": [100, 30], "initial": [71, 17], "reserves": [36, 12], "target": 15.35},
    # seed 9028; capacity [80, 50], city 4.8, river 1.8, 12-step plan
    {"inflow_upper": [8.3, 8.0, 8.7, 6.0, 4.5, 2.7, 2.6, 2.2, 3.0, 4.4, 7.3, 6.8], "inflow_lower": [3.3, 3.6, 2.9, 2.8, 1.2, 1.0, 0.7, 0.5, 0.7, 1.7, 1.4, 3.5], "city": 4.8, "farm": [0.9, 1.0, 1.9, 3.3, 4.8, 7.9, 7.5, 6.0, 4.6, 1.7, 0.9, 1.1], "river": 1.8, "capacity": [80, 50], "initial": [48, 38], "reserves": [31, 19], "target": 15.4},
    # seed 9029; capacity [120, 50], city 5.3, river 1.7, 12-step plan
    {"inflow_upper": [10.3, 6.9, 7.6, 5.6, 3.9, 2.4, 2.5, 1.7, 3.2, 3.7, 8.9, 6.5], "inflow_lower": [4.5, 1.7, 1.6, 2.6, 1.0, 1.5, 0.5, 0.5, 1.2, 1.2, 2.3, 3.2], "city": 5.3, "farm": [0.8, 1.0, 2.3, 4.6, 6.9, 6.7, 7.5, 5.9, 5.2, 1.7, 1.2, 1.1], "river": 1.7, "capacity": [120, 50], "initial": [84, 26], "reserves": [50, 17], "target": 16.1},
    # seed 9030; capacity [80, 30], city 3.8, river 1.3, 12-step plan
    {"inflow_upper": [7.9, 6.9, 6.6, 4.2, 2.8, 2.7, 1.8, 2.1, 2.3, 4.4, 7.1, 7.1], "inflow_lower": [3.8, 1.6, 1.9, 1.6, 0.9, 0.7, 0.8, 0.6, 1.1, 1.9, 3.1, 2.0], "city": 3.8, "farm": [0.8, 1.0, 2.1, 4.5, 6.1, 6.9, 7.7, 6.5, 4.4, 1.9, 0.9, 1.0], "river": 1.3, "capacity": [80, 30], "initial": [49, 23], "reserves": [36, 12], "target": 15.8},
    # seed 9031; capacity [120, 50], city 5.1, river 1.1, 12-step plan
    {"inflow_upper": [11.0, 8.8, 5.5, 4.3, 3.6, 2.7, 1.8, 2.5, 3.2, 5.5, 7.9, 9.3], "inflow_lower": [1.8, 3.9, 2.2, 2.5, 1.8, 1.2, 0.6, 0.7, 0.8, 2.3, 2.8, 3.4], "city": 5.1, "farm": [1.2, 1.1, 1.8, 4.8, 6.3, 6.4, 7.0, 8.0, 5.4, 2.3, 0.9, 0.8], "river": 1.1, "capacity": [120, 50], "initial": [82, 37], "reserves": [53, 20], "target": 16.55},
    # seed 9032; capacity [100, 40], city 4.2, river 1.2, 12-step plan
    {"inflow_upper": [6.0, 5.2, 5.2, 3.8, 3.6, 3.1, 1.4, 1.3, 3.3, 3.5, 4.5, 6.4], "inflow_lower": [1.8, 1.4, 2.4, 2.5, 1.1, 0.8, 0.8, 0.7, 1.2, 1.4, 2.5, 2.1], "city": 4.2, "farm": [0.9, 0.9, 2.0, 3.7, 4.9, 6.8, 7.3, 8.1, 5.4, 1.8, 1.1, 1.0], "river": 1.2, "capacity": [100, 40], "initial": [70, 28], "reserves": [42, 14], "target": 16.25},
    # seed 9033; capacity [100, 50], city 4.1, river 1.8, 12-step plan
    {"inflow_upper": [6.4, 7.8, 7.4, 5.0, 4.7, 3.3, 2.3, 1.9, 3.2, 3.7, 7.0, 5.8], "inflow_lower": [1.9, 3.7, 3.2, 1.8, 1.0, 0.9, 0.4, 0.7, 1.2, 1.5, 2.6, 2.0], "city": 4.1, "farm": [0.9, 1.0, 1.7, 3.6, 6.1, 8.9, 7.9, 7.1, 4.5, 2.3, 0.8, 1.0], "river": 1.8, "capacity": [100, 50], "initial": [65, 35], "reserves": [47, 17], "target": 17.25},
    # seed 9034; capacity [80, 50], city 4.5, river 1.2, 12-step plan
    {"inflow_upper": [10.6, 11.7, 6.9, 7.5, 5.2, 3.3, 2.6, 2.2, 3.4, 7.6, 10.3, 10.3], "inflow_lower": [2.8, 4.3, 2.1, 2.2, 2.5, 1.8, 1.0, 0.9, 0.8, 1.8, 2.8, 4.0], "city": 4.5, "farm": [0.8, 0.9, 1.9, 4.6, 6.1, 9.2, 6.8, 8.0, 4.7, 1.8, 1.1, 0.8], "river": 1.2, "capacity": [80, 50], "initial": [41, 28], "reserves": [28, 18], "target": 17.4},
    # seed 9035; capacity [100, 30], city 5.4, river 1.7, 12-step plan
    {"inflow_upper": [8.0, 8.9, 5.9, 6.1, 5.5, 3.0, 2.4, 2.8, 3.8, 4.6, 6.9, 10.3], "inflow_lower": [3.9, 3.5, 2.0, 3.5, 1.5, 1.3, 0.8, 0.9, 1.1, 2.6, 2.3, 4.3], "city": 5.4, "farm": [0.8, 0.8, 2.0, 4.4, 5.1, 9.5, 7.1, 5.6, 4.1, 2.0, 0.9, 1.0], "river": 1.7, "capacity": [100, 30], "initial": [60, 20], "reserves": [46, 9], "target": 15.7},
    # seed 9036; capacity [100, 50], city 4.6, river 2.3, 12-step plan
    {"inflow_upper": [5.5, 5.1, 6.1, 4.0, 4.1, 2.4, 1.4, 2.1, 2.1, 4.9, 6.9, 7.4], "inflow_lower": [3.2, 3.2, 2.0, 1.8, 1.1, 0.6, 0.6, 0.6, 0.9, 1.3, 1.4, 1.3], "city": 4.6, "farm": [1.1, 1.2, 1.8, 4.1, 5.1, 9.1, 7.4, 6.0, 4.8, 1.7, 0.8, 1.2], "river": 2.3, "capacity": [100, 50], "initial": [55, 39], "reserves": [32, 15], "target": 16.2},
    # seed 9037; capacity [80, 50], city 4.9, river 1.3, 12-step plan
    {"inflow_upper": [10.7, 5.7, 5.4, 6.5, 4.0, 2.5, 2.3, 2.3, 2.9, 4.6, 8.1, 8.6], "inflow_lower": [2.1, 2.1, 1.9, 2.6, 0.9, 1.1, 0.6, 0.8, 0.7, 1.3, 3.2, 2.9], "city": 4.9, "farm": [1.0, 1.1, 1.9, 4.0, 5.5, 8.6, 7.7, 6.4, 4.6, 1.6, 1.1, 0.9], "river": 1.3, "capacity": [80, 50], "initial": [57, 35], "reserves": [35, 20], "target": 16.4},
    # seed 9038; capacity [80, 40], city 4.0, river 2.1, 12-step plan
    {"inflow_upper": [7.8, 7.5, 6.4, 5.0, 3.1, 2.1, 2.4, 2.4, 2.6, 5.2, 6.9, 10.0], "inflow_lower": [2.4, 2.6, 2.9, 1.5, 1.8, 1.1, 0.4, 0.9, 1.1, 1.5, 1.6, 2.1], "city": 4.0, "farm": [0.9, 1.2, 2.0, 4.6, 4.9, 7.9, 7.8, 6.4, 4.4, 2.2, 1.0, 1.0], "river": 2.1, "capacity": [80, 40], "initial": [48, 28], "reserves": [26, 12], "target": 15.7},
    # seed 9039; capacity [80, 40], city 5.1, river 1.5, 12-step plan
    {"inflow_upper": [8.4, 6.3, 8.0, 5.8, 2.8, 3.3, 1.5, 2.2, 2.7, 4.1, 4.7, 7.7], "inflow_lower": [3.7, 3.6, 1.8, 1.7, 1.1, 1.3, 0.5, 0.9, 1.2, 1.1, 2.3, 2.9], "city": 5.1, "farm": [1.2, 0.9, 1.8, 3.4, 5.5, 7.9, 9.5, 7.6, 5.9, 2.1, 1.0, 1.1], "river": 1.5, "capacity": [80, 40], "initial": [58, 29], "reserves": [33, 17], "target": 18.2},
    # seed 9040; capacity [120, 30], city 5.3, river 1.1, 12-step plan
    {"inflow_upper": [6.1, 8.4, 6.9, 3.8, 4.0, 1.9, 1.3, 1.6, 2.3, 5.5, 4.4, 5.8], "inflow_lower": [2.5, 3.0, 1.8, 1.7, 0.9, 1.0, 0.5, 0.9, 1.0, 1.0, 2.3, 1.5], "city": 5.3, "farm": [0.8, 1.0, 2.0, 3.6, 6.4, 6.5, 6.5, 7.3, 6.0, 2.0, 0.9, 1.2], "river": 1.1, "capacity": [120, 30], "initial": [85, 22], "reserves": [56, 12], "target": 16.35},
    # seed 9041; capacity [80, 30], city 4.3, river 1.5, 12-step plan
    {"inflow_upper": [10.5, 11.7, 8.1, 7.0, 4.0, 3.4, 2.4, 2.5, 4.4, 4.7, 8.3, 7.5], "inflow_lower": [2.9, 2.5, 2.3, 3.2, 1.5, 1.7, 0.5, 0.9, 1.3, 1.8, 2.4, 2.2], "city": 4.3, "farm": [0.8, 1.0, 2.1, 4.7, 5.6, 7.0, 8.6, 8.4, 4.5, 1.7, 0.9, 0.8], "river": 1.5, "capacity": [80, 30], "initial": [43, 17], "reserves": [36, 12], "target": 17.05},
    # seed 9042; capacity [80, 50], city 5.0, river 1.3, 12-step plan
    {"inflow_upper": [7.4, 7.6, 8.8, 8.4, 5.3, 3.7, 2.6, 2.5, 3.3, 6.4, 8.5, 6.8], "inflow_lower": [4.2, 4.1, 2.8, 1.7, 1.1, 1.1, 0.7, 0.6, 1.4, 1.3, 2.7, 3.6], "city": 5.0, "farm": [1.0, 0.9, 1.8, 3.8, 5.4, 8.8, 7.6, 7.5, 5.1, 2.3, 1.2, 1.1], "river": 1.3, "capacity": [80, 50], "initial": [46, 31], "reserves": [32, 21], "target": 17.2},
    # seed 9043; capacity [100, 40], city 4.2, river 2.4, 12-step plan
    {"inflow_upper": [11.1, 11.1, 6.2, 6.7, 3.4, 2.7, 2.4, 2.7, 3.3, 7.1, 6.0, 7.1], "inflow_lower": [4.5, 4.2, 2.8, 1.5, 0.9, 0.9, 0.8, 0.8, 0.9, 2.4, 2.4, 2.6], "city": 4.2, "farm": [0.9, 1.1, 1.8, 3.6, 6.9, 8.2, 8.3, 6.2, 5.1, 2.2, 0.8, 1.2], "river": 2.4, "capacity": [100, 40], "initial": [68, 32], "reserves": [47, 17], "target": 17.35},
    # seed 9044; capacity [80, 50], city 3.8, river 1.7, 12-step plan
    {"inflow_upper": [8.2, 7.1, 8.3, 6.3, 3.2, 2.9, 1.9, 2.7, 3.2, 6.5, 8.0, 11.8], "inflow_lower": [4.5, 3.9, 3.9, 1.6, 1.9, 1.1, 0.7, 0.5, 1.4, 1.7, 3.3, 3.5], "city": 3.8, "farm": [1.2, 1.2, 1.8, 3.4, 6.7, 7.8, 6.8, 7.9, 4.8, 1.9, 0.9, 1.2], "river": 1.7, "capacity": [80, 50], "initial": [43, 26], "reserves": [31, 19], "target": 0.0},
    # seed 9045; capacity [100, 30], city 5.2, river 1.5, 12-step plan
    {"inflow_upper": [9.2, 9.0, 9.6, 9.0, 6.0, 3.6, 2.5, 2.2, 3.2, 5.7, 10.3, 7.9], "inflow_lower": [2.5, 3.6, 3.4, 3.2, 1.9, 0.8, 0.6, 0.8, 0.9, 2.9, 3.2, 2.5], "city": 5.2, "farm": [0.8, 1.1, 1.7, 3.6, 6.8, 9.4, 9.4, 6.8, 5.6, 2.0, 0.9, 1.1], "river": 1.5, "capacity": [100, 30], "initial": [64, 16], "reserves": [36, 11], "target": 19.0},
    # seed 9046; capacity [100, 50], city 4.5, river 2.4, 12-step plan
    {"inflow_upper": [7.6, 8.0, 5.2, 4.6, 3.9, 3.2, 1.3, 1.8, 3.0, 3.7, 6.9, 8.6], "inflow_lower": [3.3, 2.5, 2.4, 1.4, 1.6, 0.6, 0.6, 0.6, 1.1, 1.4, 1.8, 2.0], "city": 4.5, "farm": [0.9, 1.0, 2.4, 4.5, 5.0, 8.4, 7.9, 8.3, 5.4, 2.1, 0.9, 1.2], "river": 2.4, "capacity": [100, 50], "initial": [75, 29], "reserves": [35, 19], "target": 17.5},
    # seed 9047; capacity [80, 30], city 4.8, river 2.1, 12-step plan
    {"inflow_upper": [6.8, 8.1, 9.1, 5.4, 4.8, 3.9, 2.7, 2.1, 4.0, 5.6, 5.2, 9.9], "inflow_lower": [3.5, 3.3, 1.5, 1.8, 1.2, 1.3, 0.9, 0.6, 1.1, 2.2, 2.2, 4.0], "city": 4.8, "farm": [0.9, 0.9, 1.7, 3.5, 6.0, 7.2, 7.7, 7.6, 5.3, 2.0, 1.1, 0.8], "river": 2.1, "capacity": [80, 30], "initial": [56, 23], "reserves": [35, 9], "target": 16.9},
    # seed 9048; capacity [80, 40], city 4.3, river 1.5, 12-step plan
    {"inflow_upper": [10.3, 7.7, 5.5, 4.3, 4.2, 2.1, 1.9, 2.0, 2.5, 4.7, 7.1, 5.9], "inflow_lower": [3.5, 3.0, 2.5, 2.8, 0.8, 1.3, 0.8, 0.8, 1.2, 2.1, 2.0, 2.0], "city": 4.3, "farm": [1.1, 0.8, 1.9, 3.9, 7.2, 6.9, 8.9, 6.4, 4.8, 1.6, 1.2, 1.1], "river": 1.5, "capacity": [80, 40], "initial": [57, 25], "reserves": [32, 19], "target": 17.1},
    # seed 9049; capacity [80, 40], city 5.3, river 1.2, 12-step plan
    {"inflow_upper": [8.5, 11.5, 8.5, 7.5, 4.5, 2.8, 1.9, 2.4, 3.4, 4.9, 7.5, 7.9], "inflow_lower": [4.4, 4.3, 3.2, 1.7, 1.5, 1.4, 1.0, 0.7, 1.5, 1.4, 3.4, 2.2], "city": 5.3, "farm": [1.0, 0.9, 1.8, 3.8, 5.2, 7.4, 7.4, 8.1, 4.9, 2.0, 0.9, 0.8], "river": 1.2, "capacity": [80, 40], "initial": [49, 27], "reserves": [34, 17], "target": 16.5},
    # seed 9050; capacity [100, 40], city 3.8, river 1.8, 12-step plan
    {"inflow_upper": [11.6, 7.1, 7.5, 5.3, 4.6, 3.4, 2.7, 2.8, 4.2, 6.2, 7.3, 10.3], "inflow_lower": [4.6, 2.1, 3.8, 1.6, 1.9, 1.6, 0.7, 1.0, 1.6, 2.7, 2.8, 2.2], "city": 3.8, "farm": [1.1, 1.0, 1.9, 3.9, 5.8, 8.4, 8.0, 7.7, 4.3, 1.8, 1.0, 1.0], "river": 1.8, "capacity": [100, 40], "initial": [50, 21], "reserves": [31, 12], "target": 0.0},
    # seed 9051; capacity [80, 30], city 4.0, river 1.1, 12-step plan
    {"inflow_upper": [7.4, 5.7, 7.1, 4.1, 2.8, 2.4, 2.0, 2.1, 2.9, 3.8, 6.4, 5.7], "inflow_lower": [1.7, 1.8, 1.8, 2.2, 1.5, 1.2, 0.5, 0.6, 0.6, 2.2, 2.4, 3.3], "city": 4.0, "farm": [0.9, 1.1, 2.3, 4.3, 6.1, 8.7, 7.1, 8.2, 5.6, 2.3, 1.1, 1.2], "river": 1.1, "capacity": [80, 30], "initial": [63, 21], "reserves": [24, 10], "target": 17.85},
    # seed 9052; capacity [120, 30], city 3.8, river 1.3, 12-step plan
    {"inflow_upper": [8.6, 7.2, 6.9, 5.1, 3.2, 2.2, 1.1, 1.6, 2.9, 4.2, 5.1, 5.7], "inflow_lower": [3.4, 1.3, 1.4, 1.8, 0.8, 1.0, 0.7, 0.8, 0.8, 1.6, 2.6, 1.8], "city": 3.8, "farm": [1.0, 0.9, 2.0, 4.4, 6.7, 7.1, 7.5, 8.0, 5.5, 1.9, 0.8, 0.9], "river": 1.3, "capacity": [120, 30], "initial": [75, 21], "reserves": [54, 13], "target": 17.4},
    # seed 9053; capacity [100, 40], city 4.1, river 1.8, 12-step plan
    {"inflow_upper": [10.5, 9.6, 6.7, 5.6, 3.8, 2.2, 2.1, 1.6, 3.1, 6.3, 7.6, 6.3], "inflow_lower": [1.9, 1.6, 1.6, 2.5, 0.8, 1.4, 0.8, 0.5, 1.2, 1.1, 2.5, 3.5], "city": 4.1, "farm": [0.9, 1.0, 2.3, 3.8, 7.2, 9.0, 8.0, 7.8, 4.5, 2.3, 0.9, 0.9], "river": 1.8, "capacity": [100, 40], "initial": [63, 31], "reserves": [36, 18], "target": 18.25},
    # seed 9054; capacity [120, 40], city 4.6, river 2.3, 12-step plan
    {"inflow_upper": [10.2, 7.7, 7.0, 6.2, 3.4, 2.4, 2.4, 2.4, 3.7, 6.5, 9.2, 9.3], "inflow_lower": [2.0, 3.2, 2.2, 2.3, 1.4, 0.9, 0.7, 1.0, 1.1, 2.2, 2.7, 2.4], "city": 4.6, "farm": [1.1, 1.2, 1.7, 3.9, 5.0, 9.2, 8.9, 8.3, 4.9, 2.1, 0.9, 1.2], "river": 2.3, "capacity": [120, 40], "initial": [77, 25], "reserves": [38, 18], "target": 18.15},
    # seed 9055; capacity [100, 40], city 3.8, river 1.2, 12-step plan
    {"inflow_upper": [7.6, 7.0, 7.5, 6.2, 3.9, 3.7, 2.0, 2.1, 2.1, 5.0, 5.4, 7.1], "inflow_lower": [1.8, 3.6, 1.8, 2.6, 1.3, 0.7, 0.7, 0.8, 1.3, 1.4, 2.5, 2.6], "city": 3.8, "farm": [0.8, 0.9, 1.7, 4.2, 5.9, 8.7, 9.1, 6.2, 4.1, 2.0, 1.1, 0.9], "river": 1.2, "capacity": [100, 40], "initial": [73, 24], "reserves": [44, 15], "target": 0.0},
    # seed 9056; capacity [120, 50], city 3.8, river 2.3, 12-step plan
    {"inflow_upper": [9.7, 7.3, 7.8, 5.9, 4.5, 4.0, 2.9, 2.9, 3.7, 6.6, 10.9, 10.5], "inflow_lower": [5.6, 4.2, 3.3, 3.6, 2.4, 0.8, 1.0, 0.5, 1.8, 2.8, 2.3, 4.1], "city": 3.8, "farm": [1.2, 0.9, 2.4, 3.9, 5.9, 9.4, 9.1, 7.4, 5.0, 2.4, 0.9, 1.0], "river": 2.3, "capacity": [120, 50], "initial": [61, 40], "reserves": [55, 23], "target": 0.0},
    # seed 9057; capacity [100, 40], city 4.5, river 1.1, 12-step plan
    {"inflow_upper": [8.1, 5.7, 5.9, 4.3, 2.8, 3.2, 1.6, 1.5, 2.2, 5.0, 5.5, 7.5], "inflow_lower": [2.3, 2.2, 1.6, 2.2, 0.7, 0.7, 0.7, 0.5, 0.9, 1.2, 1.4, 1.9], "city": 4.5, "farm": [0.9, 1.2, 2.1, 4.1, 5.0, 7.4, 9.2, 8.3, 4.1, 2.0, 0.9, 1.2], "river": 1.1, "capacity": [100, 40], "initial": [79, 29], "reserves": [50, 16], "target": 17.0},
    # seed 9058; capacity [100, 40], city 4.7, river 2.2, 12-step plan
    {"inflow_upper": [9.7, 9.4, 6.3, 7.0, 3.2, 4.3, 2.2, 2.3, 3.2, 4.5, 6.2, 6.4], "inflow_lower": [3.8, 4.1, 2.1, 2.3, 1.7, 1.2, 0.7, 0.6, 0.7, 2.2, 2.1, 3.4], "city": 4.7, "farm": [1.2, 1.1, 2.3, 3.9, 5.9, 8.2, 6.8, 8.3, 4.0, 2.3, 1.0, 1.2], "river": 2.2, "capacity": [100, 40], "initial": [52, 26], "reserves": [32, 14], "target": 16.6},
    # seed 9059; capacity [80, 30], city 5.4, river 1.2, 12-step plan
    {"inflow_upper": [8.3, 6.3, 6.5, 5.2, 4.8, 3.5, 2.1, 1.6, 2.4, 4.3, 7.3, 9.4], "inflow_lower": [4.3, 4.4, 2.5, 1.5, 1.2, 1.1, 0.9, 0.5, 0.8, 2.5, 2.8, 2.6], "city": 5.4, "farm": [0.8, 0.8, 2.2, 3.3, 6.6, 7.7, 8.3, 6.6, 5.6, 2.2, 0.9, 1.0], "river": 1.2, "capacity": [80, 30], "initial": [54, 22], "reserves": [25, 9], "target": 17.4},
    # seed 9060; capacity [120, 30], city 4.4, river 1.1, 12-step plan
    {"inflow_upper": [7.9, 6.3, 7.6, 6.3, 4.3, 3.1, 1.7, 1.8, 2.1, 3.3, 6.4, 6.7], "inflow_lower": [2.2, 1.8, 2.5, 1.7, 1.2, 1.1, 0.5, 0.7, 1.1, 1.0, 2.3, 2.9], "city": 4.4, "farm": [0.9, 1.2, 1.8, 3.9, 6.4, 7.6, 8.0, 6.8, 4.3, 2.0, 0.9, 1.0], "river": 1.1, "capacity": [120, 30], "initial": [66, 19], "reserves": [50, 10], "target": 16.55},
    # seed 9061; capacity [80, 40], city 4.4, river 1.3, 12-step plan
    {"inflow_upper": [7.2, 7.7, 7.7, 4.5, 3.2, 2.5, 1.9, 1.4, 2.2, 4.1, 5.7, 7.5], "inflow_lower": [1.5, 1.9, 2.8, 1.8, 0.8, 0.6, 0.8, 0.4, 1.2, 1.3, 2.3, 1.8], "city": 4.4, "farm": [1.0, 1.0, 1.9, 4.0, 5.3, 9.4, 6.8, 7.3, 4.5, 1.7, 1.0, 0.9], "river": 1.3, "capacity": [80, 40], "initial": [63, 27], "reserves": [31, 18], "target": 16.65},
    # seed 9062; capacity [100, 40], city 5.0, river 1.7, 12-step plan
    {"inflow_upper": [8.1, 7.7, 5.5, 6.9, 5.3, 3.9, 2.3, 2.8, 3.4, 6.3, 6.0, 8.7], "inflow_lower": [2.9, 4.0, 2.1, 1.5, 1.3, 1.3, 0.8, 1.0, 1.5, 1.4, 3.6, 4.1], "city": 5.0, "farm": [0.8, 0.9, 1.9, 4.6, 6.2, 7.6, 8.0, 6.1, 4.5, 1.6, 1.2, 0.9], "river": 1.7, "capacity": [100, 40], "initial": [74, 29], "reserves": [48, 20], "target": 16.2},
    # seed 9063; capacity [120, 50], city 5.3, river 1.1, 12-step plan
    {"inflow_upper": [6.9, 6.7, 7.4, 4.9, 3.7, 2.7, 2.3, 1.6, 2.8, 3.8, 5.0, 7.2], "inflow_lower": [3.6, 2.7, 2.6, 1.9, 1.4, 1.2, 0.7, 0.5, 1.0, 2.1, 2.7, 3.6], "city": 5.3, "farm": [1.1, 1.0, 2.3, 3.7, 7.2, 7.3, 7.7, 6.7, 5.3, 1.7, 1.0, 1.0], "river": 1.1, "capacity": [120, 50], "initial": [82, 31], "reserves": [49, 21], "target": 17.1},
    # seed 9064; capacity [80, 50], city 5.1, river 1.6, 12-step plan
    {"inflow_upper": [7.4, 8.0, 5.5, 7.1, 3.9, 2.8, 2.5, 2.6, 2.2, 4.8, 6.2, 6.0], "inflow_lower": [2.2, 1.7, 3.5, 2.5, 2.0, 0.7, 0.4, 0.6, 0.6, 2.0, 3.5, 2.6], "city": 5.1, "farm": [1.0, 0.9, 2.2, 4.2, 6.2, 7.2, 7.5, 7.0, 5.5, 1.8, 1.2, 0.9], "river": 1.6, "capacity": [80, 50], "initial": [54, 38], "reserves": [27, 22], "target": 16.7},
    # seed 9065; capacity [120, 50], city 3.8, river 1.8, 12-step plan
    {"inflow_upper": [8.5, 5.2, 7.6, 4.1, 3.2, 2.6, 1.9, 1.6, 3.0, 5.1, 4.6, 8.7], "inflow_lower": [3.4, 3.4, 2.6, 1.5, 0.9, 0.6, 0.5, 0.8, 1.0, 0.9, 1.7, 3.3], "city": 3.8, "farm": [0.9, 0.8, 2.1, 4.0, 6.9, 9.2, 7.7, 7.7, 4.3, 1.7, 1.1, 0.9], "river": 1.8, "capacity": [120, 50], "initial": [87, 40], "reserves": [57, 21], "target": 0.0},
    # seed 9066; capacity [80, 50], city 5.4, river 1.2, 12-step plan
    {"inflow_upper": [7.2, 7.3, 8.2, 7.2, 2.8, 2.8, 1.7, 1.4, 2.5, 5.9, 6.4, 8.5], "inflow_lower": [3.8, 3.4, 2.3, 2.0, 1.4, 1.2, 0.5, 0.5, 1.0, 1.2, 2.6, 3.6], "city": 5.4, "farm": [1.1, 1.1, 2.2, 4.4, 6.0, 7.5, 6.7, 6.0, 5.0, 1.8, 1.2, 1.1], "river": 1.2, "capacity": [80, 50], "initial": [45, 37], "reserves": [27, 22], "target": 15.6},
    # seed 9067; capacity [120, 40], city 5.2, river 1.8, 12-step plan
    {"inflow_upper": [10.4, 5.3, 7.4, 4.6, 3.0, 3.3, 1.9, 1.4, 2.8, 3.7, 5.3, 8.6], "inflow_lower": [2.8, 2.6, 1.9, 2.5, 1.5, 0.6, 0.6, 0.6, 0.7, 1.9, 2.2, 3.0], "city": 5.2, "farm": [1.0, 1.0, 2.1, 3.9, 6.5, 8.8, 7.3, 5.9, 5.5, 1.6, 0.9, 1.0], "river": 1.8, "capacity": [120, 40], "initial": [95, 27], "reserves": [51, 19], "target": 17.0},
    # seed 9068; capacity [80, 50], city 3.7, river 1.6, 12-step plan
    {"inflow_upper": [10.6, 9.7, 7.7, 5.8, 5.1, 3.8, 2.7, 1.9, 4.0, 4.1, 9.9, 8.6], "inflow_lower": [2.4, 2.5, 1.9, 2.6, 0.9, 1.4, 1.0, 1.1, 0.8, 2.6, 3.7, 2.9], "city": 3.7, "farm": [1.1, 1.2, 2.1, 3.4, 6.6, 6.6, 8.0, 8.1, 4.1, 1.6, 1.1, 0.9], "river": 1.6, "capacity": [80, 50], "initial": [51, 28], "reserves": [30, 25], "target": 16.7},
    # seed 9069; capacity [120, 30], city 4.2, river 2.3, 12-step plan
    {"inflow_upper": [8.9, 5.3, 4.8, 5.5, 3.9, 2.6, 1.8, 1.4, 2.0, 3.1, 4.8, 5.1], "inflow_lower": [3.4, 3.3, 2.0, 1.8, 0.9, 0.7, 0.4, 0.4, 1.1, 0.9, 1.4, 1.9], "city": 4.2, "farm": [1.1, 0.9, 2.1, 3.5, 7.0, 6.7, 7.3, 6.0, 5.4, 2.3, 1.1, 0.9], "river": 2.3, "capacity": [120, 30], "initial": [94, 18], "reserves": [42, 10], "target": 16.2},
    # seed 9070; capacity [120, 50], city 5.3, river 2.1, 12-step plan
    {"inflow_upper": [9.7, 9.7, 8.1, 4.5, 3.0, 2.7, 2.3, 1.9, 3.5, 4.1, 7.7, 6.2], "inflow_lower": [4.5, 3.9, 1.6, 2.1, 1.4, 1.3, 0.6, 0.7, 0.8, 1.9, 2.1, 3.0], "city": 5.3, "farm": [1.1, 1.1, 1.6, 3.3, 7.1, 6.9, 9.1, 7.9, 4.8, 1.7, 1.1, 1.1], "river": 2.1, "capacity": [120, 50], "initial": [92, 37], "reserves": [43, 23], "target": 17.9},
    # seed 9071; capacity [100, 40], city 4.3, river 1.6, 12-step plan
    {"inflow_upper": [8.8, 6.4, 6.9, 7.3, 2.9, 3.0, 2.2, 1.4, 2.7, 5.5, 8.7, 5.5], "inflow_lower": [3.7, 2.7, 1.7, 2.0, 1.9, 1.4, 0.6, 0.7, 1.1, 1.0, 1.8, 3.7], "city": 4.3, "farm": [1.0, 1.0, 2.0, 3.2, 5.8, 7.2, 6.7, 6.6, 4.4, 2.2, 1.1, 0.9], "river": 1.6, "capacity": [100, 40], "initial": [64, 25], "reserves": [30, 12], "target": 0.0},
    # seed 9072; capacity [120, 40], city 3.8, river 1.6, 12-step plan
    {"inflow_upper": [9.9, 5.0, 4.7, 5.7, 3.2, 2.6, 1.4, 1.4, 2.2, 4.3, 7.7, 5.0], "inflow_lower": [2.0, 2.7, 2.0, 2.5, 1.6, 0.8, 0.4, 0.4, 0.8, 1.8, 1.3, 2.3], "city": 3.8, "farm": [1.0, 1.1, 2.4, 4.1, 5.2, 9.6, 9.1, 5.7, 4.4, 1.6, 1.2, 1.1], "river": 1.6, "capacity": [120, 40], "initial": [84, 23], "reserves": [44, 15], "target": 17.0},
    # seed 9073; capacity [120, 40], city 4.3, river 1.2, 12-step plan
    {"inflow_upper": [7.5, 7.7, 5.2, 5.4, 2.4, 2.9, 1.6, 1.5, 2.9, 3.5, 4.9, 5.8], "inflow_lower": [3.6, 2.0, 1.9, 2.4, 1.2, 0.8, 0.8, 0.3, 1.0, 1.8, 2.9, 1.5], "city": 4.3, "farm": [1.2, 0.9, 1.7, 4.2, 6.3, 9.3, 7.2, 7.8, 4.7, 1.9, 0.8, 0.8], "river": 1.2, "capacity": [120, 40], "initial": [93, 27], "reserves": [54, 20], "target": 17.65},
    # seed 9074; capacity [120, 50], city 3.7, river 1.6, 12-step plan
    {"inflow_upper": [8.8, 8.5, 6.3, 5.9, 4.4, 2.3, 1.5, 1.6, 2.6, 5.0, 6.9, 6.6], "inflow_lower": [2.0, 2.6, 2.9, 1.1, 1.3, 0.7, 0.4, 0.8, 0.8, 1.0, 2.2, 2.9], "city": 3.7, "farm": [1.1, 1.1, 2.0, 3.9, 5.2, 8.7, 7.4, 6.0, 4.9, 1.6, 0.9, 1.2], "river": 1.6, "capacity": [120, 50], "initial": [83, 36], "reserves": [49, 21], "target": 0.0},
    # seed 9075; capacity [80, 50], city 3.5, river 1.9, 12-step plan
    {"inflow_upper": [13.0, 7.7, 7.9, 9.2, 3.7, 4.6, 2.4, 2.9, 3.6, 5.4, 9.5, 8.6], "inflow_lower": [3.3, 3.3, 2.6, 3.3, 1.2, 1.3, 0.7, 1.2, 1.0, 2.5, 2.6, 4.4], "city": 3.5, "farm": [0.9, 0.8, 2.4, 4.2, 6.7, 7.5, 9.3, 7.9, 4.9, 1.9, 0.8, 1.0], "river": 1.9, "capacity": [80, 50], "initial": [53, 30], "reserves": [39, 22], "target": 0.0},
    # seed 9076; capacity [120, 30], city 4.4, river 1.3, 12-step plan
    {"inflow_upper": [12.5, 7.4, 9.0, 5.7, 6.1, 3.1, 2.9, 2.2, 4.3, 4.8, 7.5, 11.0], "inflow_lower": [2.4, 2.0, 3.5, 2.3, 1.8, 1.6, 0.5, 0.6, 1.3, 2.2, 2.1, 4.7], "city": 4.4, "farm": [1.0, 0.9, 1.7, 4.5, 5.1, 9.6, 7.8, 7.5, 5.1, 2.1, 1.0, 1.1], "river": 1.3, "capacity": [120, 30], "initial": [65, 19], "reserves": [47, 11], "target": 17.55},
    # seed 9077; capacity [100, 50], city 4.7, river 1.3, 12-step plan
    {"inflow_upper": [9.4, 9.3, 6.3, 6.4, 3.0, 2.3, 2.0, 2.0, 3.0, 5.6, 6.3, 7.0], "inflow_lower": [3.7, 3.1, 2.0, 1.4, 1.4, 1.1, 0.4, 0.5, 1.3, 1.7, 1.8, 1.8], "city": 4.7, "farm": [1.0, 1.0, 1.7, 3.3, 6.6, 7.4, 6.7, 6.5, 5.5, 2.2, 1.0, 1.0], "river": 1.3, "capacity": [100, 50], "initial": [62, 27], "reserves": [31, 16], "target": 16.35},
    # seed 9078; capacity [120, 40], city 4.8, river 1.6, 12-step plan
    {"inflow_upper": [7.5, 9.8, 5.3, 5.7, 4.8, 2.1, 1.6, 1.9, 2.2, 5.5, 7.5, 8.7], "inflow_lower": [3.9, 2.3, 3.3, 1.3, 1.7, 1.0, 0.7, 0.5, 1.1, 1.0, 2.3, 1.7], "city": 4.8, "farm": [1.2, 1.2, 2.1, 3.5, 6.7, 8.2, 7.4, 8.2, 5.4, 2.2, 0.9, 0.9], "river": 1.6, "capacity": [120, 40], "initial": [80, 25], "reserves": [51, 14], "target": 17.95},
    # seed 9079; capacity [100, 50], city 4.9, river 2.2, 12-step plan
    {"inflow_upper": [7.2, 11.0, 5.8, 5.3, 4.1, 3.3, 2.6, 2.1, 4.2, 4.7, 6.1, 9.7], "inflow_lower": [3.9, 3.3, 1.9, 1.7, 1.8, 1.5, 0.8, 0.6, 1.0, 1.4, 2.5, 2.2], "city": 4.9, "farm": [1.1, 1.1, 2.1, 4.8, 6.7, 8.4, 6.8, 5.8, 5.4, 1.7, 0.9, 1.0], "river": 2.2, "capacity": [100, 50], "initial": [73, 35], "reserves": [48, 16], "target": 16.55},
    # seed 9080; capacity [120, 30], city 5.0, river 1.8, 12-step plan
    {"inflow_upper": [8.8, 7.2, 5.0, 5.3, 3.4, 2.8, 1.3, 1.9, 3.1, 5.0, 6.0, 8.3], "inflow_lower": [4.0, 1.5, 2.6, 2.0, 1.4, 0.9, 0.9, 0.5, 0.9, 1.1, 2.7, 2.4], "city": 5.0, "farm": [1.1, 1.2, 2.3, 4.4, 7.2, 7.3, 7.0, 6.3, 5.9, 1.9, 1.1, 1.0], "river": 1.8, "capacity": [120, 30], "initial": [95, 23], "reserves": [56, 13], "target": 16.85},
    # seed 9081; capacity [80, 50], city 5.0, river 1.2, 12-step plan
    {"inflow_upper": [9.9, 6.2, 4.6, 4.8, 2.7, 2.0, 1.7, 1.5, 2.7, 5.5, 8.2, 8.9], "inflow_lower": [1.9, 2.2, 1.7, 1.9, 1.3, 0.6, 0.9, 0.9, 0.8, 1.0, 1.5, 1.5], "city": 5.0, "farm": [1.1, 0.8, 2.2, 3.9, 5.2, 8.1, 7.6, 6.6, 5.0, 1.8, 0.8, 1.1], "river": 1.2, "capacity": [80, 50], "initial": [48, 40], "reserves": [24, 20], "target": 16.25},
    # seed 9082; capacity [120, 30], city 5.4, river 1.4, 12-step plan
    {"inflow_upper": [9.5, 7.6, 10.8, 7.0, 3.9, 3.3, 2.9, 1.9, 4.3, 6.1, 10.6, 9.3], "inflow_lower": [3.8, 2.0, 2.3, 2.1, 1.2, 1.6, 0.6, 1.2, 1.6, 1.4, 2.8, 2.8], "city": 5.4, "farm": [1.1, 0.8, 2.3, 3.4, 6.7, 8.7, 6.9, 6.7, 5.5, 1.8, 1.0, 1.0], "river": 1.4, "capacity": [120, 30], "initial": [63, 23], "reserves": [55, 12], "target": 17.25},
    # seed 9083; capacity [100, 50], city 5.5, river 1.6, 12-step plan
    {"inflow_upper": [7.6, 5.8, 8.9, 6.9, 3.6, 3.6, 2.0, 1.8, 2.3, 4.6, 6.7, 9.6], "inflow_lower": [4.0, 2.2, 1.6, 1.2, 1.0, 0.6, 0.7, 0.5, 0.7, 2.4, 2.9, 3.6], "city": 5.5, "farm": [1.2, 1.1, 2.2, 3.7, 6.6, 8.8, 9.0, 6.2, 5.8, 1.7, 0.8, 0.9], "river": 1.6, "capacity": [100, 50], "initial": [80, 29], "reserves": [35, 21], "target": 18.2},
    # seed 9084; capacity [80, 50], city 4.8, river 2.0, 12-step plan
    {"inflow_upper": [8.3, 10.0, 10.9, 7.2, 6.3, 2.9, 3.0, 2.6, 4.9, 5.4, 10.6, 9.2], "inflow_lower": [4.9, 2.8, 4.4, 1.6, 1.4, 1.0, 0.7, 0.8, 1.1, 3.2, 2.6, 4.4], "city": 4.8, "farm": [1.2, 0.8, 2.1, 4.0, 5.3, 9.6, 8.9, 7.0, 5.2, 2.4, 0.9, 1.0], "river": 2.0, "capacity": [80, 50], "initial": [41, 37], "reserves": [39, 16], "target": 18.0},
    # seed 9085; capacity [120, 50], city 5.2, river 1.7, 12-step plan
    {"inflow_upper": [11.5, 8.0, 8.8, 6.9, 4.4, 3.8, 2.8, 2.6, 3.6, 7.2, 9.5, 9.9], "inflow_lower": [4.5, 3.3, 1.6, 2.8, 1.9, 0.9, 1.1, 0.7, 1.3, 1.4, 1.8, 3.9], "city": 5.2, "farm": [0.8, 1.2, 1.9, 4.1, 6.2, 7.4, 7.4, 6.3, 4.8, 2.0, 1.0, 1.1], "river": 1.7, "capacity": [120, 50], "initial": [61, 28], "reserves": [37, 24], "target": 16.05},
    # seed 9086; capacity [120, 50], city 4.5, river 2.0, 12-step plan
    {"inflow_upper": [9.9, 7.3, 5.8, 4.6, 4.5, 1.9, 1.9, 1.8, 2.3, 4.5, 7.3, 7.5], "inflow_lower": [2.7, 1.5, 2.8, 2.5, 1.6, 0.8, 0.8, 0.6, 0.9, 2.1, 2.3, 2.6], "city": 4.5, "farm": [1.0, 1.0, 1.8, 3.3, 6.7, 6.8, 8.2, 8.0, 5.0, 1.9, 1.1, 1.1], "river": 2.0, "capacity": [120, 50], "initial": [62, 37], "reserves": [51, 19], "target": 17.35},
    # seed 9087; capacity [120, 50], city 4.1, river 2.0, 12-step plan
    {"inflow_upper": [7.6, 5.6, 5.9, 5.6, 4.2, 2.0, 1.4, 1.6, 2.0, 4.3, 7.9, 6.3], "inflow_lower": [2.4, 1.4, 2.9, 1.6, 1.2, 1.1, 0.5, 0.6, 1.1, 1.9, 2.3, 1.5], "city": 4.1, "farm": [0.9, 1.1, 2.4, 3.5, 5.1, 8.9, 8.6, 7.6, 4.3, 1.9, 1.1, 0.8], "river": 2.0, "capacity": [120, 50], "initial": [67, 26], "reserves": [37, 16], "target": 17.25},
    # seed 9088; capacity [120, 50], city 5.2, river 1.6, 12-step plan
    {"inflow_upper": [12.5, 10.7, 8.7, 7.8, 3.6, 2.8, 1.9, 1.9, 2.4, 6.5, 7.8, 11.3], "inflow_lower": [3.4, 3.6, 2.9, 2.7, 1.7, 0.7, 0.8, 0.5, 0.7, 1.5, 1.8, 1.9], "city": 5.2, "farm": [0.8, 0.9, 1.8, 4.7, 5.1, 8.5, 9.4, 6.0, 4.3, 2.1, 0.8, 0.9], "river": 1.6, "capacity": [120, 50], "initial": [78, 29], "reserves": [53, 22], "target": 16.65},
    # seed 9089; capacity [80, 50], city 4.5, river 1.3, 12-step plan
    {"inflow_upper": [8.2, 8.5, 8.1, 6.4, 4.7, 3.6, 2.3, 1.9, 2.0, 5.2, 8.1, 5.3], "inflow_lower": [2.4, 3.3, 3.0, 2.5, 0.9, 1.1, 0.9, 0.9, 0.9, 1.3, 2.4, 3.1], "city": 4.5, "farm": [1.1, 0.9, 2.0, 3.2, 7.2, 8.9, 9.0, 6.5, 4.9, 1.8, 0.8, 0.8], "river": 1.3, "capacity": [80, 50], "initial": [41, 38], "reserves": [25, 22], "target": 0.0},
    # seed 9090; capacity [120, 30], city 3.6, river 1.7, 12-step plan
    {"inflow_upper": [7.8, 9.3, 6.1, 4.1, 4.0, 3.1, 1.7, 1.8, 2.8, 5.0, 7.3, 8.3], "inflow_lower": [1.9, 1.8, 1.5, 1.8, 1.4, 1.4, 0.4, 0.9, 0.8, 1.9, 3.0, 3.1], "city": 3.6, "farm": [1.1, 1.1, 1.9, 4.4, 7.2, 9.6, 7.4, 7.0, 5.2, 1.7, 1.1, 0.9], "river": 1.7, "capacity": [120, 30], "initial": [77, 19], "reserves": [53, 15], "target": 18.2},
    # seed 9091; capacity [120, 50], city 4.2, river 2.4, 12-step plan
    {"inflow_upper": [5.6, 7.7, 7.8, 4.3, 2.7, 2.2, 1.3, 1.9, 2.4, 5.7, 4.9, 8.9], "inflow_lower": [2.3, 3.0, 2.2, 2.6, 1.0, 0.8, 0.6, 0.8, 0.8, 0.9, 2.2, 1.8], "city": 4.2, "farm": [1.0, 1.1, 2.1, 3.7, 6.1, 9.1, 7.2, 7.6, 5.2, 2.0, 0.9, 1.1], "river": 2.4, "capacity": [120, 50], "initial": [91, 30], "reserves": [56, 16], "target": 17.6},
    # seed 9092; capacity [80, 50], city 3.6, river 2.4, 12-step plan
    {"inflow_upper": [7.1, 7.6, 4.9, 3.6, 3.7, 2.2, 1.4, 1.6, 2.9, 3.1, 6.4, 7.8], "inflow_lower": [2.2, 2.0, 2.4, 1.1, 0.9, 0.9, 0.8, 0.7, 0.9, 1.8, 1.6, 2.3], "city": 3.6, "farm": [1.0, 1.0, 1.7, 4.1, 4.9, 7.1, 7.8, 6.9, 5.8, 2.3, 1.0, 1.0], "river": 2.4, "capacity": [80, 50], "initial": [46, 38], "reserves": [26, 25], "target": 16.25},
    # seed 9093; capacity [80, 50], city 3.8, river 1.8, 12-step plan
    {"inflow_upper": [11.0, 9.0, 7.6, 7.5, 4.2, 3.2, 1.6, 2.4, 3.2, 4.2, 7.2, 7.8], "inflow_lower": [4.0, 3.1, 2.9, 1.8, 1.2, 0.8, 0.8, 0.9, 0.7, 1.8, 2.4, 2.6], "city": 3.8, "farm": [1.1, 1.2, 2.1, 4.6, 6.3, 7.2, 7.0, 7.0, 5.2, 2.4, 1.1, 1.0], "river": 1.8, "capacity": [80, 50], "initial": [40, 26], "reserves": [28, 23], "target": 16.35},
    # seed 9094; capacity [80, 50], city 3.9, river 2.1, 12-step plan
    {"inflow_upper": [12.0, 10.5, 8.4, 6.4, 6.0, 2.6, 2.6, 2.2, 4.6, 6.1, 5.9, 7.4], "inflow_lower": [4.9, 3.7, 2.2, 1.6, 1.1, 1.7, 1.0, 0.8, 1.7, 1.4, 3.9, 2.9], "city": 3.9, "farm": [0.9, 1.0, 2.2, 4.2, 6.4, 7.6, 6.8, 5.8, 5.9, 1.9, 1.0, 1.2], "river": 2.1, "capacity": [80, 50], "initial": [44, 30], "reserves": [28, 24], "target": 16.25},
    # seed 9095; capacity [100, 40], city 4.4, river 1.8, 12-step plan
    {"inflow_upper": [12.6, 8.7, 9.7, 6.9, 3.5, 3.2, 2.8, 1.9, 2.7, 4.2, 9.8, 8.8], "inflow_lower": [2.9, 3.3, 3.7, 2.8, 1.7, 1.6, 0.8, 0.7, 1.5, 1.7, 3.2, 2.5], "city": 4.4, "farm": [1.1, 1.1, 2.3, 4.4, 6.7, 6.9, 9.6, 7.7, 5.4, 1.7, 0.9, 1.0], "river": 1.8, "capacity": [100, 40], "initial": [63, 22], "reserves": [50, 14], "target": 18.15},
    # seed 9096; capacity [80, 50], city 5.0, river 1.3, 12-step plan
    {"inflow_upper": [6.8, 10.5, 8.4, 6.4, 3.6, 3.6, 2.5, 2.7, 2.4, 6.1, 6.1, 10.1], "inflow_lower": [3.5, 2.1, 3.7, 2.2, 1.1, 1.3, 0.7, 0.9, 1.3, 1.9, 1.9, 2.4], "city": 5.0, "farm": [1.1, 1.1, 2.3, 3.6, 6.2, 9.0, 6.5, 7.9, 5.7, 2.1, 1.0, 1.0], "river": 1.3, "capacity": [80, 50], "initial": [47, 26], "reserves": [27, 20], "target": 17.65},
    # seed 9097; capacity [80, 50], city 3.9, river 2.1, 12-step plan
    {"inflow_upper": [9.0, 8.3, 5.7, 5.1, 4.4, 2.6, 2.3, 2.3, 2.1, 5.8, 8.1, 8.1], "inflow_lower": [3.6, 2.5, 2.1, 1.7, 1.9, 1.1, 0.4, 0.5, 1.1, 1.6, 2.9, 1.7], "city": 3.9, "farm": [1.1, 1.0, 1.9, 4.3, 5.3, 7.6, 9.0, 7.8, 4.5, 1.6, 1.1, 1.1], "river": 2.1, "capacity": [80, 50], "initial": [59, 37], "reserves": [27, 15], "target": 0.0},
    # seed 9098; capacity [100, 30], city 3.9, river 1.1, 12-step plan
    {"inflow_upper": [8.3, 9.7, 5.7, 5.1, 4.0, 3.4, 1.9, 1.5, 3.2, 3.4, 6.5, 5.3], "inflow_lower": [2.8, 3.4, 3.0, 2.8, 1.9, 0.7, 0.4, 0.9, 0.8, 1.1, 1.7, 3.7], "city": 3.9, "farm": [1.1, 1.0, 2.3, 3.3, 5.9, 9.5, 9.0, 6.6, 5.5, 1.9, 0.9, 1.0], "river": 1.1, "capacity": [100, 30], "initial": [56, 24], "reserves": [49, 12], "target": 18.25},
    # seed 9099; capacity [100, 40], city 4.1, river 1.1, 12-step plan
    {"inflow_upper": [7.3, 4.6, 7.1, 3.9, 3.7, 1.7, 1.4, 2.1, 2.7, 4.2, 5.4, 6.7], "inflow_lower": [1.8, 3.2, 1.5, 1.2, 1.2, 0.8, 0.6, 0.5, 1.1, 0.9, 2.8, 2.3], "city": 4.1, "farm": [1.1, 1.1, 2.2, 4.2, 5.5, 7.8, 8.9, 8.2, 4.4, 2.3, 1.1, 1.0], "river": 1.1, "capacity": [100, 40], "initial": [76, 23], "reserves": [48, 16], "target": 17.4},
)
