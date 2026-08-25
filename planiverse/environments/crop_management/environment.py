"""Irrigation scheduling across a growing season.

A potato crop goes in the ground in April and comes out in October. Through the season the
farmer decides, every ten days, whether to irrigate and by how much. The yield is not the
sum of those decisions — it is the *integral* of a crop growth model driven by the actual
weather: water applied on day 20 changes the leaf area, which changes how much light the
canopy intercepts for the next eighty days, which changes the tuber weight at harvest.

That delay is the whole point of the environment. An action's effect is not visible when it
is taken and cannot be written down as a set of facts to add and delete. It is a change to
the trajectory of a differential equation whose value only becomes a yield at the end.

Two things make the decision genuinely hard, both measured on the shipped weather:

* **Whether irrigation helps at all depends on weather that has not happened yet.** In 1980
  a reference schedule of four 2 cm applications gains **nothing** — the season was wet
  enough. In 1986 the same schedule gains **2698 kg/ha**. The states look identical on the
  day the first decision is made.
* **Yield is not monotone in water.** Applying more, earlier, is not reliably better; the
  crop's response depends on the growth stage it is in when the water arrives.

    env = CropEnv()
    env.fix_index(0)
    state, info = env.reset()
    for action, successor in env.successors(state):
        print(action, successor.water_used)

Built on PCSE, Wageningen University's Python implementation of the WOFOST crop model:
https://github.com/ajwdewit/pcse
"""
import datetime
import os
from collections import namedtuple

from planiverse.environments.base import Environment

#: A potato crop on a Dutch field, which is what the bundled weather describes.
CROP, VARIETY = "potato", "Potato_701"
SOW_MONTH_DAY = (4, 15)
HARVEST_MONTH_DAY = (10, 15)

#: Decisions are taken every `DECISION_INTERVAL` days, starting `FIRST_DECISION` days after
#: sowing. Ten decision points, so a plan is ten actions long whatever else it does.
FIRST_DECISION = 10
DECISION_INTERVAL = 10
DECISION_COUNT = 10

#: Water the farmer may apply, in cm. The reference schedule below uses exactly this much,
#: so the budget is binding rather than decorative.
WATER_BUDGET_CM = 8.0

#: What one irrigation action may apply, in cm. `0.0` is "wait", which is a real decision
#: here: water not applied now is water still available later.
IRRIGATION_AMOUNTS = (0.0, 1.0, 2.0, 4.0)

#: Applications lose some water to runoff and evaporation before it reaches the root zone.
IRRIGATION_EFFICIENCY = 0.7

#: The schedule every scenario's target is measured against: 2 cm on the 20th, 40th, 60th
#: and 80th day after sowing. It is a witness that each instance is solvable, and it is
#: deliberately naive — a fixed calendar that ignores the weather entirely.
REFERENCE_SCHEDULE = ((20, 2.0), (40, 2.0), (60, 2.0), (80, 2.0))

#: The target is a shade under what the reference achieves, so a planner that finds a
#: different — or cheaper — schedule of the same quality also succeeds.
TARGET_FRACTION = 0.98

Scenario = namedtuple("Scenario", ["year", "rainfed", "reference"])

#: One growing season per scenario: the same field, a different year's weather. Measured by
#: running each year rainfed and then under the reference schedule. 1990 and 1991 are absent
#: because the bundled weather has gaps in them.
#:
#: `rainfed` and `reference` are recorded so a scenario that stops reproducing fails loudly
#: rather than quietly changing difficulty. The spread is the point: in 1980 the reference
#: schedule gains nothing at all and the right plan is to apply no water, while in 1986 it
#: gains 2698 kg/ha and a plan that waits loses most of the crop.
SCENARIOS = (
    Scenario(1976, 4241.2, 5677.6),
    Scenario(1977, 13387.8, 14130.5),
    Scenario(1978, 11971.6, 13542.8),
    Scenario(1979, 13170.0, 14262.2),
    Scenario(1980, 13362.7, 13362.7),
    Scenario(1981, 12611.5, 14010.3),
    Scenario(1982, 7774.4, 9884.9),
    Scenario(1983, 4968.6, 6749.0),
    Scenario(1984, 8314.5, 10132.4),
    Scenario(1985, 14957.2, 14988.5),
    Scenario(1986, 6758.3, 9456.3),
    Scenario(1987, 13471.9, 13856.9),
    Scenario(1988, 13240.8, 14459.5),
    Scenario(1989, 9326.0, 11323.7),
    Scenario(1992, 8115.5, 10452.4),
    Scenario(1993, 14680.0, 15791.2),
    Scenario(1994, 5916.4, 7450.2),
    Scenario(1995, 6555.0, 7352.8),
    Scenario(1996, 7589.7, 9746.3),
    Scenario(1997, 11063.9, 12134.9),
    Scenario(1998, 13411.9, 14052.6),
    Scenario(1999, 10661.8, 12810.9),
)

AGROMANAGEMENT = """
- {start}:
    CropCalendar:
        crop_name: {crop}
        variety_name: {variety}
        crop_start_date: {sow}
        crop_start_type: sowing
        crop_end_date: {harvest}
        crop_end_type: harvest
        max_duration: 300
    TimedEvents: null
    StateEvents: null
"""


def weather_directory():
    """Where PCSE keeps the CABO weather files it ships — 24 years of Dutch seasons."""
    import pcse

    return os.path.join(os.path.dirname(pcse.__file__), "tests", "test_data")


def decision_days():
    """Days after sowing on which the farmer decides."""
    return tuple(FIRST_DECISION + DECISION_INTERVAL * index
                 for index in range(DECISION_COUNT))


class CropAction:
    """Apply `amount` cm of water at this decision point, or wait."""

    def __init__(self, amount):
        self.amount = float(amount)

    def cost(self):
        """Water is the cost. Waiting is free, which is why doing nothing is a real option
        rather than a filler action."""
        return self.amount

    def __eq__(self, other):
        return isinstance(other, CropAction) and self.amount == other.amount

    def __hash__(self):
        return hash(self.amount)

    def __lt__(self, other):
        return self.amount < other.amount

    def __str__(self):
        return "wait" if self.amount == 0 else f"irrigate_{self.amount:g}cm"

    def __repr__(self):
        return str(self)


class CropState:
    """A season part-way through: the decisions taken, and what the crop has done so far.

    The decision tuple is the state. The crop model is deterministic given the weather —
    the same schedule in the same year always produces the same yield, verified by running
    it twice — so replaying a schedule lands in the same place and two schedules that agree
    are the same state.
    """

    def __init__(self, schedule, biomass, yield_kg, water_used, finished, stage):
        self.schedule = tuple(schedule)
        self.biomass = biomass
        self.yield_kg = yield_kg
        self.water_used = water_used
        self.finished = finished
        self.stage = stage
        self.depth = len(self.schedule)

        literals = [f"applied({index},{amount:g})"
                    for index, amount in enumerate(self.schedule) if amount]
        literals.append(f"decisions({self.depth})")
        literals.append(f"water-used({int(water_used)})")
        # Bucketed into quintals per hectare: the raw float would make every state novel.
        literals.append(f"biomass({int(biomass / 100)})")
        if finished:
            literals.append("harvested")
            literals.append(f"yield({int(yield_kg / 100)})")
        self.literals = frozenset(literals)

    def __eq__(self, other):
        return isinstance(other, CropState) and self.schedule == other.schedule

    def __hash__(self):
        return hash(self.schedule)

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        applied = ", ".join(f"day {day}: {amount:g}cm"
                            for day, amount in zip(decision_days(), self.schedule)
                            if amount) or "nothing applied"
        head = (f"harvested: {self.yield_kg:.0f} kg/ha" if self.finished
                else f"growing: {self.biomass:.0f} kg/ha of biomass so far")
        return f"{head}\nwater used: {self.water_used:g} cm\n{applied}"

    def __repr__(self):
        return (f"<CropState(decisions={self.depth}, water={self.water_used:g}cm, "
                f"{'yield' if self.finished else 'biomass'}="
                f"{self.yield_kg if self.finished else self.biomass:.0f})>")


class CropEnv(Environment):
    """Schedule irrigation across a season so the crop makes its target.

    Needs `pcse`. The weather ships inside it and the crop parameters are cached locally by
    PCSE itself, so a season runs offline.
    """

    def __init__(self, budget_cm=WATER_BUDGET_CM, amounts=IRRIGATION_AMOUNTS,
                 target_fraction=TARGET_FRACTION):
        super().__init__("crop_management")
        self.budget_cm = budget_cm
        self.amounts = tuple(amounts)
        self.target_fraction = target_fraction

        self.scenario_index = None
        self.state = None
        self.state_history = []
        self._cache = {}
        self._providers = None

    # ------------------------------------------------------------------ instances

    def fix_index(self, index):
        """Choose the growing season — the same field, a different year's weather."""
        if not 0 <= index < len(SCENARIOS):
            raise IndexError(
                f"Invalid index: {index}. There are {len(SCENARIOS)} seasons, so the index "
                f"must be 0-{len(SCENARIOS) - 1}.")
        self.scenario_index = index
        self._cache = {}

    def scenario(self):
        return SCENARIOS[self.scenario_index]

    def target_yield(self):
        """What this season's plan has to reach, measured off the reference schedule."""
        return self.scenario().reference * self.target_fraction

    # ------------------------------------------------------------------ the crop model

    def __providers__(self):
        """Weather, crop, soil and site parameters. Built once; none of them are per-run."""
        import pcse
        from pcse.base import ParameterProvider
        from pcse.input import (CABOWeatherDataProvider, DummySoilDataProvider,
                                WOFOST72SiteDataProvider, YAMLCropDataProvider)

        if self._providers is None:
            weather = CABOWeatherDataProvider("NL1", weather_directory())
            crops = YAMLCropDataProvider()
            crops.set_active_crop(CROP, VARIETY)
            parameters = ParameterProvider(cropdata=crops,
                                           soildata=DummySoilDataProvider(),
                                           sitedata=WOFOST72SiteDataProvider(WAV=50))
            self._providers = (weather, parameters)
        return self._providers

    def __model__(self, year):
        import yaml
        from pcse.models import Wofost72_WLP_FD

        weather, parameters = self.__providers__()
        agro = yaml.safe_load(AGROMANAGEMENT.format(
            start=f"{year}-01-01", crop=CROP, variety=VARIETY,
            sow=f"{year}-{SOW_MONTH_DAY[0]:02d}-{SOW_MONTH_DAY[1]:02d}",
            harvest=f"{year}-{HARVEST_MONTH_DAY[0]:02d}-{HARVEST_MONTH_DAY[1]:02d}"))
        return Wofost72_WLP_FD(parameters, weather, agro)

    def __run__(self, schedule, to_the_end):
        """Run the season under `schedule`, either to the next decision or to harvest.

        Replaying rather than snapshotting: a PCSE model holds its whole state history, and
        replaying is cheap (a full season is under half a second) and exactly reproducible.
        """
        from pcse import signals

        year = self.scenario().year
        model = self.__model__(year)
        sowing = datetime.date(year, *SOW_MONTH_DAY)
        days = decision_days()

        for index, amount in enumerate(schedule):
            target = (sowing + datetime.timedelta(days=days[index]) - model.day).days
            if target > 0:
                model.run(days=target)
            if amount:
                model._send_signal(signal=signals.irrigate, amount=float(amount),
                                   efficiency=IRRIGATION_EFFICIENCY)

        if to_the_end:
            model.run_till_terminate()
            summary = model.get_summary_output()
            harvest = summary[0] if summary else {}
            return float(harvest.get("TWSO") or 0.0), float(harvest.get("TAGP") or 0.0), True

        # Advance to the day the *next* decision is taken, so the state describes the crop
        # as the farmer sees it when deciding.
        following = len(schedule)
        if following < DECISION_COUNT:
            target = (sowing + datetime.timedelta(days=days[following]) - model.day).days
            if target > 0:
                model.run(days=target)
        variables = model.get_output()
        latest = variables[-1] if variables else {}
        biomass = float(latest.get("TAGP") or 0.0)
        return 0.0, biomass, False

    def __state__(self, schedule):
        """The state a schedule leads to. Memoised — deterministic, so it cannot differ."""
        key = tuple(schedule)
        if key in self._cache:
            return self._cache[key]
        finished = len(key) >= DECISION_COUNT
        yield_kg, biomass, harvested = self.__run__(key, finished)
        state = CropState(key, biomass, yield_kg, sum(key), harvested, len(key))
        self._cache[key] = state
        return state

    # ------------------------------------------------------------------ interface

    def reset(self):
        if self.scenario_index is None:
            self.fix_index(0)
        self.state = self.__state__(())
        self.state_history = [self.state]
        scenario = self.scenario()
        return self.state, {"year": scenario.year,
                            "rainfed": scenario.rainfed,
                            "reference": scenario.reference,
                            "target": self.target_yield(),
                            "budget_cm": self.budget_cm,
                            "decisions": DECISION_COUNT,
                            "decision_days": decision_days()}

    def is_goal(self, state):
        """Harvested, at or above target, without exceeding the water budget.

        The yield is only known at harvest, so no part-grown state is ever a goal — this is
        a finite-horizon problem in which the whole season must be planned before the
        outcome is visible. That is the shape of the domain, not a modelling choice.
        """
        return (state.finished
                and state.yield_kg >= self.target_yield()
                and state.water_used <= self.budget_cm + 1e-9)

    def is_terminal(self, state):
        """The season ended without making the target, or the budget is already spent."""
        if state.water_used > self.budget_cm + 1e-9:
            return True
        return state.finished and not self.is_goal(state)

    def __affordable__(self, state):
        remaining = self.budget_cm - state.water_used
        return [amount for amount in self.amounts if amount <= remaining + 1e-9]

    def successors(self, state):
        """Each affordable amount at the next decision point."""
        if self.is_goal(state) or self.is_terminal(state) or state.depth >= DECISION_COUNT:
            return []
        successors = []
        for amount in self.__affordable__(state):
            successor = self.__state__(state.schedule + (amount,))
            successors.append((CropAction(amount), successor))
        return successors

    def __advance__(self, state, action):
        if self.is_goal(state) or self.is_terminal(state) or state.depth >= DECISION_COUNT:
            return state
        amount = action.amount if isinstance(action, CropAction) else float(action)
        return self.__state__(state.schedule + (amount,))

    def simulate(self, plan):
        state, _ = self.reset()
        trace = [state]
        for action in plan:
            trace.append(self.__advance__(trace[-1], action))
        return trace

    def step(self, action):
        """Stateful play. The reward is the biomass gained since the last decision."""
        if self.state is None:
            raise ValueError("Game not initialized. Call reset() first.")
        before = self.state.biomass
        self.state = self.__advance__(self.state, action)
        self.state_history.append(self.state)
        return self.state, self.state.biomass - before

    def get_actions(self):
        return [CropAction(amount) for amount in self.amounts]

    def reference_plan(self):
        """The naive fixed-calendar schedule every target is measured against.

        Kept as a method because it is a *witness*: every scenario is solvable, and this is
        the plan that proves it. It is also a baseline worth beating — it ignores the
        weather entirely, so in a wet year it spends the whole budget for nothing.
        """
        days = decision_days()
        by_day = dict(REFERENCE_SCHEDULE)
        return [CropAction(by_day.get(day, 0.0)) for day in days]

    def render(self):
        rendered = [str(state) for state in self.state_history]
        for step, text in enumerate(rendered):
            print(f"Step: {step}")
            print(text)
            print("--------------")
        return rendered

    def close(self):
        self._cache = {}
        self._providers = None
