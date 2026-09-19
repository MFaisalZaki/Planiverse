"""Protecting a city's roads from floods: which zones to adapt, and when.

A rainstorm floods parts of a city. Where the water stands, roads are damaged in proportion
to its depth, and traffic slows or stops, so trips take longer or are not made at all. The
operator can protect one zone a year, by raising its roads or by making them more resistant,
and each measure costs a share of the value of the roads it protects. The problem is to
decide, over a horizon of years whose storms are drawn from a climate projection, where and
when to spend, so that the sum of damage, delay and adaptation cost stays within a target.

The model follows the MAAT environment of Costa, Petersen, Vandervoort, Drews, Morrissey and
Pereira (*Climate Adaptation with Reinforcement Learning: Experiments with Flooding and
Transportation in Copenhagen*, NeurIPS 2024 workshop on Tackling Climate Change with Machine
Learning, https://github.com/MLSM-at-DTU/floods_transport_rl), and takes from it what the
paper takes from the literature: the damage curves and road values of the European road
network assessment (van Ginkel, Dottori, Alfieri, Feyen and Koks, NHESS 21, 2021), the costs of
each measure, the speed a flooded road allows, the value of an hour of delay, and the rainfall
projections of the Danish Klimaatlas by return period. Two things are this environment's own.
First, the city: MAAT reads Copenhagen's traffic zones, its road network and a hydrodynamic
flood map, and none of that data is redistributable here, so a city is drawn from a seed
(see `draw_city`) and a flood map is a depth per zone. Second, the depth a rainstorm leaves is
a linear ramp from the lightest rain MAAT models (20 mm) to its design storm (160 mm), because
this environment has no terrain to run a hydrodynamic model over. A city exported from MAAT
loads through `set_instance` like any other (`tools/export_maat_city.py`).

## The decision problem

A state is the decisions taken so far, from which follow the year, the measures in place and
the money spent. At each decision the operator either waits or protects one zone that floods
and is not yet protected, with one of the measures the instance offers. A decision covers a
`period` of years (five in the bundled scenarios; MAAT decides yearly over 77 years, which
`period=1` reproduces): each year's storm falls, the roads take damage according to their
depth and their protection, trips are distributed between zones and routed over the slowed
network, and the year's cost is added to the total, along with what the measure cost. The
goal is to reach the end of the horizon with the total at most the target. The total only
grows, so a state that has passed the target is a dead end.

A measure costs a fifth or a half of the value of the roads it protects and saves a few
percent of it a year, so it pays for itself only over decades, and a policy that looks one
period ahead never acts. The target is therefore set, the way the crop environment sets its
yield target, from a reference policy that looks to the end of the horizon: at each decision
it takes the action that leaves the lowest total if nothing more is done afterwards, over the
storms the instance holds, and waits when none improves on waiting. The target is that
policy's cost plus a small slack, so a plan has to do at least about as well, and a draw on
which doing nothing would pass is not an instance.
"""
import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from planiverse.environments.base import Environment
from planiverse.environments.generation import draw_until, rng

# ---------------------------------------------------------------- what MAAT takes from the literature

#: Mean damage ratio of a road as a function of water depth in metres, low flow, from the
#: European road network assessment: C1 is a motorway or trunk with traffic signals, C3 one
#: without, and C5 every other road.
IMPACT = {
    "C1": ((0.0, 0.000), (0.5, 0.010), (1.0, 0.030), (1.5, 0.075), (2.0, 0.100),
           (6.0, 0.200), (10.0, 0.300)),
    "C3": ((0.0, 0.000), (0.5, 0.002), (1.0, 0.004), (1.5, 0.025), (2.0, 0.030),
           (6.0, 0.040), (10.0, 0.050)),
    "C5": ((0.0, 0.000), (0.5, 0.015), (1.0, 0.025), (2.0, 0.035), (6.0, 0.050),
           (10.0, 0.065)),
}
CLASSES = tuple(IMPACT)

#: Value of a kilometre of two-lane road by class, in euros of 2015 (the midpoint of the
#: assessment's range), the factor it adds for signalling on motorways and trunks, and the
#: conversion to Danish kroner of 2023 that MAAT applies (the ratio of GDP per head, then the
#: exchange rate).
VALUE_PER_KM = {"motorway": (35 - 3.5) / 2 * 1e6, "trunk": (7.5 - 2.5) / 2 * 1e6,
                "primary": (3.0 - 1.0) / 2 * 1e6, "secondary": (1.5 - 0.5) / 2 * 1e6,
                "tertiary": (0.6 - 0.2) / 2 * 1e6, "other": (0.3 - 0.1) / 2 * 1e6}
SIGNALLING = {"motorway": 1.22, "trunk": 1.28}
TO_DKK = 7.46 / (45500 / 52510) / (26760 / 45500)

#: The measures: what each does to the depth its roads see (`raise`, metres), to their damage
#: ratio (`resist`, a factor), and what it costs as a share of the value of the roads it
#: protects, per road class, from the assessment. MAAT leaves the two resistance measures at
#: a placeholder cost of one.
MEASURES = {
    "elevate1": {"raise": 1.0, "resist": 1.0, "cost": {"C1": 0.22, "C3": 0.22, "C5": 0.22}},
    "elevate2": {"raise": 2.0, "resist": 1.0, "cost": {"C1": 0.47, "C3": 0.47, "C5": 0.47}},
    "resist25": {"raise": 0.0, "resist": 0.75, "cost": {"C1": 1.00, "C3": 1.00, "C5": 1.00}},
    "resist50": {"raise": 0.0, "resist": 0.50, "cost": {"C1": 1.00, "C3": 1.00, "C5": 1.00}},
}

#: What an hour of delay costs a traveller, and what a trip not made costs, in kroner.
VALUE_OF_DELAY = 213.0
VALUE_OF_NO_TRAVEL = 347.23 * 7.4
#: The speed of a dry road, and the depth above which a road is impassable, in millimetres.
FREE_SPEED = 50.0
IMPASSABLE = 300.0
#: The rainfall MAAT's design storm brings, and the lightest event it models.
DESIGN_STORM = 160
LIGHTEST_STORM = 20

#: Rainfall in millimetres by return period, for three climate periods, from the Danish
#: Klimaatlas as MAAT tabulates it; a year is counted from 2022, MAAT's year zero.
RETURN_PERIODS = (1, 2, 5, 10, 20, 50, 100)
RAINFALL = (
    (18, (35.15, 43.37, 55.67, 65.85, 77.15, 93.72, 107.50)),      # to 2040
    (48, (37.39, 45.05, 57.26, 67.63, 78.89, 95.59, 109.30)),      # to 2070
    (77, (40.65, 48.72, 62.17, 73.64, 86.23, 104.56, 119.95)),     # to 2100
)

#: The bundled scenarios: a seed and the generator's options. Like every bundled instance in
#: this library, each is a draw the generator makes, and the same draw every time.
SCENARIOS = (
    (0, dict(zones=8, years=30, rain="design")),
    (1, dict(zones=8, years=30, rain="klimaatlas")),
    (2, dict(zones=12, years=30, rain="design")),
    (3, dict(zones=12, years=40, rain="klimaatlas")),
    (4, dict(zones=16, years=40, rain="design")),
    (5, dict(zones=16, years=40, rain="klimaatlas")),
    (6, dict(zones=24, years=50, rain="design")),
    (7, dict(zones=24, years=50, rain="klimaatlas")),
    (8, dict(zones=30, years=60, rain="klimaatlas", measures=("elevate1", "elevate2"))),
)


def damage_ratio(road_class, depth, raise_by=0.0, resist=1.0):
    """The share of a road's value a storm takes, at `depth` metres, with a measure."""
    points = IMPACT[road_class]
    return resist * float(np.interp(depth - raise_by, [p[0] for p in points],
                                    [p[1] for p in points]))


def speed(depth):
    """What a road under `depth` metres of water allows, in km/h, after MAAT."""
    mm = depth * 1000
    if mm <= 0:
        return FREE_SPEED
    if mm > IMPASSABLE:
        return 0.0
    return min(FREE_SPEED, 0.0009 * mm ** 2 - 0.5529 * mm + 86.9448)


def storm_depth(design_depth, rain):
    """The depth a storm of `rain` mm leaves where the design storm leaves `design_depth`."""
    return design_depth * max(0.0, (rain - LIGHTEST_STORM) / (DESIGN_STORM - LIGHTEST_STORM))


def sample_rain(random_, year):
    """A year's worst storm in millimetres, drawn from the Klimaatlas return periods by
    inverse-CDF sampling and rounded to 4 mm, as MAAT draws it."""
    for last_year, amounts in RAINFALL:
        if year <= last_year:
            break
    cdf = [1 - 1 / period for period in RETURN_PERIODS]
    return int(round(float(np.interp(random_.random(), cdf, amounts)) / 4) * 4)


def exposure(roads, signalled):
    """The value of a zone's roads by damage class, in kroner, from kilometres by type."""
    values = {road_class: 0.0 for road_class in CLASSES}
    for road_type, km in roads.items():
        value = km * VALUE_PER_KM[road_type] * TO_DKK
        if road_type in SIGNALLING:
            values["C1"] += signalled * value * SIGNALLING[road_type]
            values["C3"] += (1 - signalled) * value
        else:
            values["C5"] += value
    return values


def draw_city(random_, zones, flood_share=0.4, spacing=1.0):
    """A city: zones on a jittered grid, roads and trips in each, and how deep the design
    storm floods it. Everything random comes from `random_`."""
    columns = math.ceil(math.sqrt(zones))
    cells = [(row, column) for row in range(math.ceil(zones / columns))
             for column in range(columns)][:zones]
    drawn = []
    for number, (row, column) in enumerate(cells):
        flooded = random_.random() < flood_share
        drawn.append({
            "id": f"z{number:02d}",
            "x": round(column * spacing + random_.uniform(-0.25, 0.25) * spacing, 3),
            "y": round(row * spacing + random_.uniform(-0.25, 0.25) * spacing, 3),
            "supply": random_.randint(500, 5000),
            "demand": random_.randint(500, 5000),
            "roads": {
                "motorway": round(random_.uniform(0.5, 2.5), 1) if random_.random() < 0.3 else 0.0,
                "trunk": round(random_.uniform(0.5, 3.0), 1) if random_.random() < 0.5 else 0.0,
                "primary": round(random_.uniform(1.0, 4.0), 1),
                "secondary": round(random_.uniform(2.0, 6.0), 1),
                "tertiary": round(random_.uniform(3.0, 8.0), 1),
                "other": round(random_.uniform(5.0, 20.0), 1),
            },
            "signalled": 0.5,
            "design_depth": round(random_.uniform(0.1, 1.2), 2) if flooded else 0.0,
        })
    edges = []
    by_cell = {cell: number for number, cell in enumerate(cells)}
    for (row, column), number in by_cell.items():
        for neighbour, always in (((row, column + 1), True), ((row + 1, column), True),
                                  ((row + 1, column + 1), False), ((row + 1, column - 1), False)):
            if neighbour in by_cell and (always or random_.random() < 0.3):
                other = by_cell[neighbour]
                distance = math.dist((drawn[number]["x"], drawn[number]["y"]),
                                     (drawn[other]["x"], drawn[other]["y"]))
                edges.append([drawn[number]["id"], drawn[other]["id"], round(distance, 3), 0.5, 0.5])
    return {"zones": drawn, "edges": edges}


def trip_distribution(supply, demand, cost, tolerance=1e-6, iterations=1000):
    """Iterative proportional fitting of `cost` to the supply and demand margins, as MAAT
    distributes trips, truncated to whole trips."""
    matrix = np.array(cost, dtype=float)
    supply, demand = np.array(supply, dtype=float), np.array(demand, dtype=float)
    for _ in range(iterations):
        rows = matrix.sum(axis=1)
        matrix = matrix * np.where(rows > 0, supply / np.where(rows > 0, rows, 1), 0)[:, None]
        cols = matrix.sum(axis=0)
        matrix = matrix * np.where(cols > 0, demand / np.where(cols > 0, cols, 1), 0)[None, :]
        if (np.abs(matrix.sum(axis=1) - supply).max() < tolerance
                and np.abs(matrix.sum(axis=0) - demand).max() < tolerance):
            break
    return np.floor(matrix).astype(int)


# ------------------------------------------------------------------------- the environment

class FloodAction:
    """Protect `zone` with `kind`, or `wait` a year."""

    def __init__(self, kind, zone=None):
        if kind != "wait" and kind not in MEASURES:
            raise ValueError(f"unknown measure {kind!r}; choose from {tuple(MEASURES)} or 'wait'")
        self.kind = kind
        self.zone = None if kind == "wait" else zone
        self.name = "wait" if kind == "wait" else f"{kind}({zone})"

    @classmethod
    def parse(cls, text):
        text = str(text)
        if text == "wait":
            return cls("wait")
        kind, _, zone = text.partition("(")
        return cls(kind, zone.rstrip(")"))

    def cost(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, FloodAction) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    __repr__ = __str__


class FloodState:
    """The decisions so far, the measures they put in place, and what the years have cost.

    The path of actions is the identity: one action a period, so two states with the same
    path are the same state, and a path is what `simulate` replays. `depth` is the number of
    decisions taken and is not part of the identity in any other way.
    """

    def __init__(self, path, protected, cost, spent, damage, delay, no_travel, rain, years,
                 period, target):
        self.path = tuple(path)
        self.step = len(self.path)
        self.year = min(years, self.step * period)
        self.steps = -(-years // period)
        self.protected = dict(protected)          # zone id -> measure
        self.cost = cost                          # kroner so far, everything counted
        self.spent = spent                        # of which, on measures
        self.damage, self.delay, self.no_travel = damage, delay, no_travel
        self.rain = rain                          # the worst storm of the last period, mm
        self.years, self.period, self.target = years, period, target
        self.depth = self.step
        self.goal = self.step >= self.steps and cost <= target
        self.terminal = cost > target
        bucket = min(21, int(20 * cost / target)) if target else 0
        literals = [f"step({self.step})", f"spent({bucket})"]
        literals += [f"protected({zone}, {kind})" for zone, kind in sorted(self.protected.items())]
        if self.goal:
            literals.append("goal-reached")
        if self.terminal:
            literals.append("terminal-state")
        self.literals = frozenset(literals)

    def __eq__(self, other):
        return isinstance(other, FloodState) and self.path == other.path

    def __hash__(self):
        return hash(self.path)

    def __lt__(self, other):
        return self.path < other.path

    def __str__(self):
        millions = lambda value: f"{value / 1e6:.2f} M"  # noqa: E731
        protected = ", ".join(f"{zone} {kind}" for zone, kind in sorted(self.protected.items()))
        status = "goal" if self.goal else "over target" if self.terminal else f"{self.years - self.year} years left"
        return "\n".join([
            f"year {self.year} of {self.years}" + (f", worst storm so far {self.rain} mm" if self.year else ""),
            f"protected: {protected or 'nothing'}",
            f"cost {millions(self.cost)} of {millions(self.target)} DKK: damage {millions(self.damage)}, "
            f"delays {millions(self.delay)}, measures {millions(self.spent)}  [{status}]",
        ])

    def __repr__(self):
        return f"<FloodState(step={self.step}, protected={sorted(self.protected)}, cost={self.cost:.0f})>"


class FloodTransportEnv(Environment):
    """Flood adaptation of a city's roads, after MAAT, over a city drawn from a seed."""

    def __init__(self, weights=None, beta=0.5):
        """`weights` scales the four cost components (`damage`, `delay`, `no_travel`,
        `measures`); MAAT's defaults count everything but trips not made. `beta` is the
        distance decay of the trip distribution, per kilometre."""
        super().__init__("flood_transport")
        self.weights = {"damage": 1.0, "delay": 1.0, "no_travel": 0.0, "measures": 1.0,
                        **(weights or {})}
        self.beta = beta
        self.index = 0
        self.instance = None
        self.witness = None
        self.witness_expansions = None
        self.state = None
        self.state_history = []
        self._city = None
        self._years = {}                         # (protected, rain) -> a year's impacts
        self._reference_plan = None

    # ------------------------------------------------------------------ instances

    def set_index(self, index):
        """Select one of the bundled scenarios, a fixed draw of the generator."""
        if not 0 <= index < len(SCENARIOS):
            raise IndexError(f"Invalid index: {index}. There are {len(SCENARIOS)} scenarios, "
                             f"so the index must be 0-{len(SCENARIOS) - 1}.")
        seed, options = SCENARIOS[index]
        self.generate_instance(seed=seed, **options)
        self.index = index
        self.witness = self.witness_expansions = None     # bundled; `reference_plan()` has it

    def set_instance(self, instance):
        """Select an instance: the dict `generate_instance` returns, or a city exported from
        MAAT with `rain`, `years` and `measures` added. `reference` and `do_nothing` are
        measured on reset when absent."""
        instance = dict(instance)
        for key in ("zones", "edges", "rain"):
            if key not in instance:
                raise ValueError(f"an instance needs {key!r}")
        instance.setdefault("years", len(instance["rain"]))
        instance.setdefault("period", 5)
        instance.setdefault("measures", ["elevate1"])
        instance.setdefault("slack", 0.02)
        if instance["period"] < 1 or instance["years"] < 1:
            raise ValueError("an instance needs a horizon of at least a year and a period of at least a year")
        if len(instance["rain"]) < instance["years"]:
            raise ValueError("an instance needs a storm for every year of its horizon")
        for kind in instance["measures"]:
            FloodAction(kind, "")
        self.instance = instance
        self.index = None
        self.witness = self.witness_expansions = None
        self._city = None
        self._years = {}
        self._reference_plan = None

    def generate_instance(self, seed=None, zones=12, years=40, period=5, rain="klimaatlas",
                          measures=("elevate1",), flood_share=0.4, slack=0.02, start_year=0,
                          attempts=50):
        """Draw a city and its storms, select the instance, and return it as a dict.

        `zones` is the size of the city, `years` the horizon, `period` the years one decision
        covers, and `rain` either `"design"` (MAAT's design storm every year) or
        `"klimaatlas"` (a storm a year drawn from the return periods, from `start_year` on).
        `measures` are the kinds offered. A draw is kept only if doing nothing would miss the
        target, so that every instance has a decision in it; the reference policy that sets
        the target is the witness.
        """
        random_, seed = rng(seed)
        found = {}

        def draw(attempt):
            city = draw_city(random_, zones, flood_share=flood_share)
            if rain == "design":
                storms = [DESIGN_STORM] * years
            elif rain == "klimaatlas":
                storms = [sample_rain(random_, start_year + year) for year in range(years)]
            else:
                raise ValueError("rain must be 'design' or 'klimaatlas'")
            return {**city, "rain": storms, "years": years, "period": period,
                    "measures": list(measures), "slack": slack, "seed": seed}

        def accept(instance):
            self.set_instance(instance)
            self.__measure__()
            if self.instance["do_nothing"] <= self.__target__():
                return False
            found.update(instance=self.instance, plan=self._reference_plan)
            return True

        draw_until(draw, accept, attempts, "flood scenario")
        self.set_instance(found["instance"])
        self.__measure__()
        self.witness, self.witness_expansions = found["plan"], 0
        return dict(self.instance)

    # ------------------------------------------------------------------ the contract

    def reset(self):
        if self.instance is None:
            self.set_index(0)
        self.__measure__()
        self.state = self.__initial__()
        self.state_history = [self.state]
        instance = self.instance
        return self.state, {"zones": len(instance["zones"]), "years": instance["years"],
                            "period": instance["period"],
                            "flooded": sum(1 for zone in instance["zones"] if zone["design_depth"] > 0),
                            "rain": list(instance["rain"][:instance["years"]]),
                            "measures": list(instance["measures"]),
                            "target": self.__target__(), "reference": instance["reference"],
                            "do_nothing": instance["do_nothing"],
                            "generated": self.index is None}

    def is_goal(self, state):
        return state.goal

    def is_terminal(self, state):
        return state.terminal and not state.goal

    def successors(self, state):
        if state.goal or state.terminal:
            return []
        return [(action, self.__advance__(state, action)) for action in self.__candidates__(state)]

    def simulate(self, plan):
        state, _ = self.reset()
        trace = [state]
        for action in plan:
            trace.append(self.__advance__(trace[-1], action))
        return trace

    def step(self, action):
        if self.state is None:
            raise ValueError("Game not initialized. Call reset() first.")
        before = self.state.cost
        self.state = self.__advance__(self.state, action)
        self.state_history.append(self.state)
        return self.state, before - self.state.cost

    def get_actions(self):
        """Every action the instance offers, whichever year it is."""
        city = self.__city__()
        return [FloodAction("wait")] + [FloodAction(kind, zone) for kind in self.instance["measures"]
                                        for zone in city["flooded"]]

    def render(self):
        history = [str(state) for state in self.state_history]
        for line in history:
            print(line)
        return history

    # ------------------------------------------------------------------ the model

    def __target__(self):
        return self.instance["reference"] * (1 + self.instance["slack"])

    def __candidates__(self, state):
        if state.step >= state.steps:
            return []
        city = self.__city__()
        actions = [FloodAction("wait")]
        for kind in self.instance["measures"]:
            actions += [FloodAction(kind, zone) for zone in city["flooded"] if zone not in state.protected]
        return actions

    def __initial__(self, target=None):
        return FloodState((), {}, 0.0, 0.0, 0.0, 0.0, 0.0, None, self.instance["years"],
                          self.instance["period"], self.__target__() if target is None else target)

    def __advance__(self, state, action):
        if not isinstance(action, FloodAction):
            action = FloodAction.parse(action)
        if state.goal or state.terminal or state.step >= state.steps:
            return state
        protected = dict(state.protected)
        spent = 0.0
        if action.kind != "wait":
            if action.zone in protected or action.zone not in self.__city__()["flooded"]:
                return state
            protected[action.zone] = action.kind
            spent = self.__measure_cost__(action.zone, action.kind)
        damage = delay = no_travel = 0.0
        worst = 0
        for year in range(state.year, min(state.years, state.year + state.period)):
            rain = self.instance["rain"][year]
            worst = max(worst, rain)
            yearly = self.__year__(protected, rain)
            damage, delay, no_travel = damage + yearly[0], delay + yearly[1], no_travel + yearly[2]
        weights = self.weights
        cost = (state.cost + weights["damage"] * damage + weights["delay"] * delay
                + weights["no_travel"] * no_travel + weights["measures"] * spent)
        return FloodState(state.path + (action.name,), protected, cost, state.spent + spent,
                          state.damage + damage, state.delay + delay, state.no_travel + no_travel,
                          worst, state.years, state.period, state.target)

    def __city__(self):
        """The instance's city as arrays, built once: exposures, depths, the trip table and
        the dry network's travel times."""
        if self._city is not None:
            return self._city
        zones = self.instance["zones"]
        ids = [zone["id"] for zone in zones]
        index = {zone_id: number for number, zone_id in enumerate(ids)}
        n = len(zones)
        values = np.array([[exposure(zone["roads"], zone.get("signalled", 0.5))[road_class]
                            for road_class in CLASSES] for zone in zones])
        design = np.array([zone["design_depth"] for zone in zones], dtype=float)
        xy = np.array([[zone["x"], zone["y"]] for zone in zones], dtype=float)
        distance = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))
        trips = trip_distribution([zone["supply"] for zone in zones],
                                  [zone["demand"] for zone in zones],
                                  np.exp(-self.beta * distance))
        edges = [(index[a], index[b], float(d), float(ra), float(rb))
                 for a, b, d, ra, rb in self.instance["edges"]]
        self._city = {"ids": ids, "index": index, "n": n, "values": values, "design": design,
                      "trips": trips, "edges": edges,
                      "flooded": [zone_id for zone_id, depth in zip(ids, design) if depth > 0]}
        self._city["dry"] = self.__lengths__(np.zeros(n))
        return self._city

    def __lengths__(self, depths):
        """Shortest travel times between every pair of zones, in hours, over the network
        with `depths` metres of water standing in each zone; infinite where no route is
        passable."""
        city = self.__city__()
        n = city["n"]
        weights = np.zeros((n, n))
        for a, b, d, ra, rb in city["edges"]:
            speed_a, speed_b = speed(depths[a]), speed(depths[b])
            if speed_a <= 0 or speed_b <= 0:
                continue
            hours = d * ra / speed_a + d * rb / speed_b
            weights[a, b] = weights[b, a] = hours if hours > 0 else 1e-9
        return dijkstra(csr_matrix(weights), directed=False)

    def __year__(self, protected, rain):
        """What a storm of `rain` mm costs with `protected` in place: damage to the roads,
        the delays it causes, and the trips it prevents, each in kroner."""
        key = (tuple(sorted(protected.items())), rain)
        if key in self._years:
            return self._years[key]
        city = self.__city__()
        depths = np.array([storm_depth(design, rain) for design in city["design"]])
        damage, effective = 0.0, depths.copy()
        for number, zone_id in enumerate(city["ids"]):
            measure = MEASURES.get(protected.get(zone_id))
            raise_by = measure["raise"] if measure else 0.0
            resist = measure["resist"] if measure else 1.0
            effective[number] = max(0.0, depths[number] - raise_by)
            damage += sum(city["values"][number, k] * damage_ratio(road_class, depths[number], raise_by, resist)
                          for k, road_class in enumerate(CLASSES))
        lengths = self.__lengths__(effective)
        trips, dry = city["trips"], city["dry"]
        reachable = np.isfinite(lengths) & np.isfinite(dry)
        delay = float((trips * np.where(reachable, lengths - dry, 0)).sum()) * VALUE_OF_DELAY
        no_travel = float((trips * (~np.isfinite(lengths) & np.isfinite(dry))).sum()) * VALUE_OF_NO_TRAVEL
        self._years[key] = (damage, max(0.0, delay), no_travel)
        return self._years[key]

    def __measure_cost__(self, zone_id, kind):
        city = self.__city__()
        number = city["index"][zone_id]
        return sum(MEASURES[kind]["cost"][road_class] * city["values"][number, k]
                   for k, road_class in enumerate(CLASSES))

    def __wait_out__(self, state):
        """The total if nothing more is done from `state` to the end of the horizon."""
        while state.step < state.steps:
            state = self.__advance__(state, FloodAction("wait"))
        return state.cost

    def __measure__(self):
        """Measure the instance the way the bundled ones are: what doing nothing costs, and
        what the reference policy costs, which sets the target.

        The reference looks to the end of the horizon: at each decision it takes the action
        that leaves the lowest total if nothing more is done afterwards, and waits when
        none beats waiting. It is a solution by construction, and its plan is the witness.
        """
        if self._reference_plan is not None and "reference" in self.instance:
            return
        state = self.__initial__(target=float("inf"))
        self.instance["do_nothing"] = self.__wait_out__(state)
        plan = []
        while state.step < state.steps:
            action, state = min(((action, self.__advance__(state, action))
                                 for action in self.__candidates__(state)),
                                key=lambda pair: (self.__wait_out__(pair[1]), pair[0].name))
            plan.append(action)
        self.instance["reference"] = state.cost
        self._reference_plan = plan

    def reference_plan(self):
        """The reference policy's plan, a solution by construction."""
        self.__measure__()
        return list(self._reference_plan)
