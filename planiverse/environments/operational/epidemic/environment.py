"""An epidemic on Covasim: a week at a time, open the town, ask for distancing or lock it
down, and reach the horizon with the deaths under the target, the hospitals never over
capacity, and the disruption within budget.

Covasim (https://github.com/institutefordiseasemodeling/covasim, MIT; Kerr et al. 2021,
https://doi.org/10.1371/journal.pcbi.1009149) is the Institute for Disease Modeling's
agent-based model of COVID-19: a synthetic population with household, school, work and
community contact layers, an infection with a viral-load course, and a severity ladder from
symptoms to the hospital, intensive care and death. It is a dependency installed from PyPI;
nothing of it is included here.

The decisions are the model's own `beta` (its transmission rate) scaled by a level for the
week: open (1.0), distancing (0.6) or lockdown (0.3), costing 0, 1 or 3 points of disruption
against the instance's budget. Everything else is the model's: who meets whom, who falls ill,
who needs a bed and who dies, and none of it is written down as an action model, which is why
the environment is here.

## Trajectory constraints as a goal

This is a trajectory problem, and it is turned into an initial-to-goal one the way the grid
and flood environments are. The constraint that must hold throughout (the hospital load never
over capacity) is `is_terminal`: the first week that breaks it is a dead end, so every plan
returned respects it. The accumulated quantities (deaths, disruption points) are carried in the
state, and the goal is the horizon reached with deaths under the target and points within the
budget. The target is what the best of a few scripted policies achieves, so the instance is
solvable by construction and the planner is asked to do at least as well.

## Determinism

Covasim replays exactly from its seed, and a copy of a running simulation does not carry the
random stream with it, so a state is its schedule (the levels chosen so far) and every
expansion replays the epidemic from day zero. Twelve weeks of ten thousand people take a
third of a second.
"""
from planiverse.environments.base import Environment
from planiverse.environments.generation import draw_until, rng

#: The levels a week may be set to: the factor on the model's transmission rate, and the
#: disruption each costs against the budget.
LEVELS = {"open": 1.0, "distancing": 0.6, "lockdown": 0.3}
POINTS = {"open": 0, "distancing": 1, "lockdown": 3}
DAYS = 7
#: What the generator draws when not told.
POPULATIONS = (5000, 8000, 10000)
WEEKS = (10, 12, 14)


def _covasim():
    import covasim
    return covasim


class EpidemicAction:
    """`open`, `distancing` or `lockdown`: the level for the coming week."""

    def __init__(self, level):
        if level not in LEVELS:
            raise ValueError(f"unknown level: {level!r}")
        self.level = level
        self.name = level

    @classmethod
    def parse(cls, text):
        return cls(str(text).strip())

    def cost(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, EpidemicAction) and self.level == other.level

    def __hash__(self):
        return hash(self.level)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    __repr__ = __str__


ACTIONS = tuple(EpidemicAction(level) for level in LEVELS)


class EpidemicState:
    """The levels chosen so far and what the epidemic reads at the end of them. Identity is the
    schedule, since the model replays exactly from its seed."""

    def __init__(self, schedule, day, infectious, load, peak, deaths, infections, points,
                 capacity=0, target=0, budget=0, weeks=0, depth=0):
        self.schedule = tuple(schedule)
        self.week = len(self.schedule)
        self.day = day
        self.infectious, self.load, self.peak = infectious, load, peak
        self.deaths, self.infections, self.points = deaths, infections, points
        self.capacity, self.target, self.budget, self.weeks = capacity, target, budget, weeks
        self.depth = depth
        literals = [f"chosen({k}, {level})" for k, level in enumerate(self.schedule)]
        literals += [f"week({self.week})", f"deaths({deaths})", f"points({points})",
                     f"in_hospital({load // 10 * 10})", f"peak({peak // 10 * 10})",
                     f"infectious({infectious // 50 * 50})"]
        if peak > capacity:
            literals.append("over_capacity")
        self.literals = frozenset(literals)

    def __eq__(self, other):
        return isinstance(other, EpidemicState) and self.schedule == other.schedule

    def __hash__(self):
        return hash(self.schedule)

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        chosen = ", ".join(self.schedule) or "nothing decided"
        return (f"week {self.week} of {self.weeks}: {self.infectious} infectious, {self.load} in "
                f"hospital (peak {self.peak} of {self.capacity} beds), {self.deaths} dead of "
                f"{self.target} allowed; {self.points} of {self.budget} points spent; {chosen}")

    def __repr__(self):
        return f"<EpidemicState(week={self.week}, deaths={self.deaths}, peak={self.peak})>"


def replay(instance, schedule):
    """The epidemic from day zero through the weeks of `schedule`: its readings at the end."""
    import numpy as np
    cv = _covasim()
    sim = cv.Sim(dict(pop_size=instance["population"], pop_infected=instance["infected"],
                      n_days=DAYS * instance["weeks"], rand_seed=instance["seed"], verbose=0))
    sim.initialize()
    base = sim["beta"]
    load, peak, points = 0, 0, 0
    for level in schedule:
        sim["beta"] = base * LEVELS[level]
        points += POINTS[level]
        for _ in range(DAYS):
            sim.step()
            load = int(sim.results["n_severe"][sim.t - 1] + sim.results["n_critical"][sim.t - 1])
            peak = max(peak, load)
    people = sim.people
    return dict(day=int(sim.t), infectious=int(people.infectious.sum()), load=load, peak=peak,
                deaths=int(people.dead.sum()),
                infections=int(np.nansum(sim.results["new_infections"][:sim.t])), points=points)


#: Schedules that need no thought, used to set the target: the town left open, distancing
#: throughout, the budget spent on lockdown first, and a lockdown after two open weeks.
def _references(weeks, budget):
    first = min(weeks, budget // POINTS["lockdown"])
    rest = budget - first * POINTS["lockdown"]
    plans = [["open"] * weeks,
             ["lockdown"] * first + ["distancing"] * min(weeks - first, rest) + ["open"] * max(0, weeks - first - min(weeks - first, rest)),
             ["open"] * 2 + ["lockdown"] * first + ["distancing"] * min(weeks - 2 - first, rest)]
    plans[-1] += ["open"] * (weeks - len(plans[-1]))
    if weeks <= budget:
        plans.append(["distancing"] * weeks)
    return [plan[:weeks] for plan in plans]


class EpidemicEnv(Environment):
    """Reach the horizon with the deaths under the target, the hospitals never over capacity
    and the disruption within budget."""

    def __init__(self):
        super().__init__("epidemic")
        self.instance = None
        self.index = None
        self.state = None
        self.state_history = []
        self.witness = None
        self.witness_expansions = None

    # ------------------------------------------------------------------ instances

    def set_index(self, index):
        if not 0 <= index < len(OUTBREAKS):
            raise IndexError(f"Invalid index: {index}. There are {len(OUTBREAKS)} outbreaks, so "
                             f"the index must be 0-{len(OUTBREAKS) - 1}.")
        self.set_instance(OUTBREAKS[index])
        self.index = index

    def set_instance(self, instance):
        """Select an outbreak: `seed`, `population`, `infected`, `weeks`, `capacity` (beds),
        `budget` (disruption points) and `target` (deaths allowed)."""
        for key in ("seed", "population", "infected", "weeks", "capacity", "budget", "target"):
            if key not in instance:
                raise ValueError(f"an outbreak needs `{key}`")
        self.instance = {key: int(instance[key]) for key in
                         ("seed", "population", "infected", "weeks", "capacity", "budget", "target")}
        self.index = None
        self.witness = self.witness_expansions = None

    def generate_instance(self, seed=None, population=None, weeks=None, attempts=40):
        """Draw an outbreak, select it, and return it as the dict `set_instance` takes.

        The population and the horizon come from `POPULATIONS` and `WEEKS` unless given; the
        seed infections, the beds (one to two and a half per cent of the population) and the
        disruption budget are drawn. The scripted schedules are then replayed, the target is
        the fewest deaths any feasible one reaches, and a draw is thrown back when none is
        feasible or when leaving the town open already meets the target.
        """
        random_, _ = rng(seed)
        found = {}

        def draw(attempt):
            people = population or random_.choice(POPULATIONS)
            horizon = weeks or random_.choice(WEEKS)
            return {"seed": random_.randrange(1, 2 ** 30), "population": people,
                    "infected": random_.randint(20, 80), "weeks": horizon,
                    "capacity": int(people * random_.uniform(0.010, 0.025)),
                    "budget": random_.randint(6, 3 * horizon // 2), "target": 0}

        def accept(instance):
            outcomes = []
            for plan in _references(instance["weeks"], instance["budget"]):
                readings = replay(instance, plan)
                feasible = readings["peak"] <= instance["capacity"] and readings["points"] <= instance["budget"]
                outcomes.append((feasible, readings["deaths"], plan))
            feasible = [o for o in outcomes if o[0]]
            if not feasible:
                return False
            best = min(feasible, key=lambda o: o[1])
            if best[1] < 1:
                return False
            open_town = outcomes[0]
            if open_town[0] and open_town[1] <= best[1]:
                return False                    # doing nothing is already good enough
            instance["target"] = best[1]
            found["plan"] = [EpidemicAction(level) for level in best[2]]
            found["measured"] = len(outcomes)
            return True

        instance = draw_until(draw, accept, attempts, "epidemic outbreak")
        self.set_instance(instance)
        self.witness, self.witness_expansions = found["plan"], found["measured"]
        return instance

    # ------------------------------------------------------------------- contract

    def __state__(self, schedule, depth=0):
        readings = replay(self.instance, schedule)
        return EpidemicState(schedule, capacity=self.instance["capacity"], target=self.instance["target"],
                             budget=self.instance["budget"], weeks=self.instance["weeks"], depth=depth,
                             **readings)

    def reset(self):
        if self.instance is None:
            self.set_index(0)
        self.state = self.__state__(())
        self.state_history = [self.state]
        return self.state, {"outbreak": self.index, "weeks": self.instance["weeks"],
                            "capacity": self.instance["capacity"], "budget": self.instance["budget"],
                            "target": self.instance["target"], "generated": self.index is None}

    def is_goal(self, state):
        return (state.week >= self.instance["weeks"] and state.deaths <= self.instance["target"]
                and state.peak <= self.instance["capacity"] and state.points <= self.instance["budget"])

    def is_terminal(self, state):
        return (state.peak > self.instance["capacity"] or state.points > self.instance["budget"]
                or (state.week >= self.instance["weeks"] and not self.is_goal(state)))

    def get_actions(self, state=None):
        state = state or self.state
        left = self.instance["budget"] - state.points
        return [action for action in ACTIONS if POINTS[action.level] <= left]

    def successors(self, state):
        if self.is_goal(state) or self.is_terminal(state):
            return []
        return [(action, self.__advance__(state, action)) for action in self.get_actions(state)]

    def __advance__(self, state, action):
        if self.is_goal(state) or self.is_terminal(state):
            return state
        if not isinstance(action, EpidemicAction):
            action = EpidemicAction.parse(action)
        if action not in self.get_actions(state):
            return state
        return self.__state__(state.schedule + (action.level,), state.depth + 1)

    def simulate(self, plan):
        state, _ = self.reset()
        trace = [state]
        for action in plan:
            trace.append(self.__advance__(trace[-1], action))
        return trace

    def step(self, action):
        if self.state is None:
            raise ValueError("call reset() first")
        before = self.state.deaths
        self.state = self.__advance__(self.state, action)
        self.state_history.append(self.state)
        return self.state, before - self.state.deaths

    def render(self):
        lines = [f"step {k}: {state}" for k, state in enumerate(self.state_history)]
        print("\n".join(lines))
        return lines


#: The bundled outbreaks: `generate_instance(seed)` for the seed beside each, embedded as the
#: plain data `set_instance` takes, with the plan each was accepted on in
#: `tests/data/epidemic_solutions.json`.
OUTBREAKS = (
    # seed 9000; 8000 people, 12 weeks, 144 beds, budget 12, 12-step plan
    {"seed": 639359048, "population": 8000, "infected": 22, "weeks": 12, "capacity": 144, "budget": 12, "target": 14},
    # seed 9001; 5000 people, 12 weeks, 92 beds, budget 15, 12-step plan
    {"seed": 271741907, "population": 5000, "infected": 24, "weeks": 12, "capacity": 92, "budget": 15, "target": 2},
    # seed 9002; 8000 people, 10 weeks, 197 beds, budget 13, 10-step plan
    {"seed": 768659023, "population": 8000, "infected": 22, "weeks": 10, "capacity": 197, "budget": 13, "target": 2},
    # seed 9003; 10000 people, 10 weeks, 241 beds, budget 14, 10-step plan
    {"seed": 611879544, "population": 10000, "infected": 59, "weeks": 10, "capacity": 241, "budget": 14, "target": 3},
    # seed 9004; 10000 people, 14 weeks, 240 beds, budget 21, 14-step plan
    {"seed": 158060733, "population": 10000, "infected": 47, "weeks": 14, "capacity": 240, "budget": 21, "target": 1},
    # seed 9005; 8000 people, 12 weeks, 112 beds, budget 14, 12-step plan
    {"seed": 1016414714, "population": 8000, "infected": 23, "weeks": 12, "capacity": 112, "budget": 14, "target": 1},
    # seed 9006; 5000 people, 12 weeks, 122 beds, budget 15, 12-step plan
    {"seed": 126514070, "population": 5000, "infected": 80, "weeks": 12, "capacity": 122, "budget": 15, "target": 10},
    # seed 9007; 8000 people, 10 weeks, 103 beds, budget 11, 10-step plan
    {"seed": 217043974, "population": 8000, "infected": 67, "weeks": 10, "capacity": 103, "budget": 11, "target": 5},
    # seed 9008; 10000 people, 12 weeks, 150 beds, budget 10, 12-step plan
    {"seed": 455692257, "population": 10000, "infected": 48, "weeks": 12, "capacity": 150, "budget": 10, "target": 2},
    # seed 9009; 8000 people, 10 weeks, 156 beds, budget 12, 10-step plan
    {"seed": 1021599089, "population": 8000, "infected": 51, "weeks": 10, "capacity": 156, "budget": 12, "target": 3},
    # seed 9010; 5000 people, 10 weeks, 78 beds, budget 11, 10-step plan
    {"seed": 231068473, "population": 5000, "infected": 60, "weeks": 10, "capacity": 78, "budget": 11, "target": 10},
    # seed 9011; 10000 people, 14 weeks, 238 beds, budget 15, 14-step plan
    {"seed": 771548876, "population": 10000, "infected": 43, "weeks": 14, "capacity": 238, "budget": 15, "target": 12},
    # seed 9012; 8000 people, 14 weeks, 176 beds, budget 17, 14-step plan
    {"seed": 679190307, "population": 8000, "infected": 56, "weeks": 14, "capacity": 176, "budget": 17, "target": 16},
    # seed 9013; 8000 people, 14 weeks, 115 beds, budget 20, 14-step plan
    {"seed": 1066399490, "population": 8000, "infected": 75, "weeks": 14, "capacity": 115, "budget": 20, "target": 12},
    # seed 9014; 5000 people, 14 weeks, 71 beds, budget 21, 14-step plan
    {"seed": 433114407, "population": 5000, "infected": 36, "weeks": 14, "capacity": 71, "budget": 21, "target": 5},
    # seed 9015; 10000 people, 10 weeks, 111 beds, budget 14, 10-step plan
    {"seed": 320116510, "population": 10000, "infected": 77, "weeks": 10, "capacity": 111, "budget": 14, "target": 1},
    # seed 9016; 8000 people, 10 weeks, 173 beds, budget 8, 10-step plan
    {"seed": 554089690, "population": 8000, "infected": 36, "weeks": 10, "capacity": 173, "budget": 8, "target": 6},
    # seed 9017; 10000 people, 12 weeks, 222 beds, budget 12, 12-step plan
    {"seed": 695950895, "population": 10000, "infected": 37, "weeks": 12, "capacity": 222, "budget": 12, "target": 14},
    # seed 9018; 8000 people, 10 weeks, 146 beds, budget 11, 10-step plan
    {"seed": 729509277, "population": 8000, "infected": 77, "weeks": 10, "capacity": 146, "budget": 11, "target": 10},
    # seed 9019; 8000 people, 10 weeks, 176 beds, budget 9, 10-step plan
    {"seed": 151474544, "population": 8000, "infected": 50, "weeks": 10, "capacity": 176, "budget": 9, "target": 11},
    # seed 9020; 8000 people, 14 weeks, 149 beds, budget 19, 14-step plan
    {"seed": 560165649, "population": 8000, "infected": 75, "weeks": 14, "capacity": 149, "budget": 19, "target": 6},
    # seed 9021; 5000 people, 12 weeks, 99 beds, budget 16, 12-step plan
    {"seed": 954936846, "population": 5000, "infected": 55, "weeks": 12, "capacity": 99, "budget": 16, "target": 6},
    # seed 9022; 8000 people, 12 weeks, 99 beds, budget 15, 12-step plan
    {"seed": 1032699113, "population": 8000, "infected": 31, "weeks": 12, "capacity": 99, "budget": 15, "target": 2},
    # seed 9023; 10000 people, 10 weeks, 176 beds, budget 15, 10-step plan
    {"seed": 1026590899, "population": 10000, "infected": 54, "weeks": 10, "capacity": 176, "budget": 15, "target": 4},
    # seed 9024; 10000 people, 14 weeks, 120 beds, budget 15, 14-step plan
    {"seed": 119707168, "population": 10000, "infected": 73, "weeks": 14, "capacity": 120, "budget": 15, "target": 18},
    # seed 9025; 5000 people, 10 weeks, 75 beds, budget 13, 10-step plan
    {"seed": 235012444, "population": 5000, "infected": 34, "weeks": 10, "capacity": 75, "budget": 13, "target": 3},
    # seed 9026; 8000 people, 12 weeks, 135 beds, budget 18, 12-step plan
    {"seed": 814955787, "population": 8000, "infected": 43, "weeks": 12, "capacity": 135, "budget": 18, "target": 8},
    # seed 9027; 8000 people, 10 weeks, 156 beds, budget 11, 10-step plan
    {"seed": 1038748615, "population": 8000, "infected": 23, "weeks": 10, "capacity": 156, "budget": 11, "target": 3},
    # seed 9028; 10000 people, 14 weeks, 186 beds, budget 14, 14-step plan
    {"seed": 313123336, "population": 10000, "infected": 66, "weeks": 14, "capacity": 186, "budget": 14, "target": 37},
    # seed 9029; 8000 people, 14 weeks, 118 beds, budget 17, 14-step plan
    {"seed": 534852905, "population": 8000, "infected": 36, "weeks": 14, "capacity": 118, "budget": 17, "target": 9},
    # seed 9030; 10000 people, 10 weeks, 216 beds, budget 9, 10-step plan
    {"seed": 584205943, "population": 10000, "infected": 44, "weeks": 10, "capacity": 216, "budget": 9, "target": 2},
    # seed 9031; 10000 people, 14 weeks, 226 beds, budget 21, 14-step plan
    {"seed": 908142938, "population": 10000, "infected": 75, "weeks": 14, "capacity": 226, "budget": 21, "target": 10},
    # seed 9032; 10000 people, 12 weeks, 109 beds, budget 13, 12-step plan
    {"seed": 1046948376, "population": 10000, "infected": 78, "weeks": 12, "capacity": 109, "budget": 13, "target": 18},
    # seed 9033; 8000 people, 10 weeks, 108 beds, budget 11, 10-step plan
    {"seed": 434638249, "population": 8000, "infected": 37, "weeks": 10, "capacity": 108, "budget": 11, "target": 2},
    # seed 9034; 10000 people, 10 weeks, 214 beds, budget 6, 10-step plan
    {"seed": 370997887, "population": 10000, "infected": 49, "weeks": 10, "capacity": 214, "budget": 6, "target": 11},
    # seed 9035; 8000 people, 14 weeks, 119 beds, budget 21, 14-step plan
    {"seed": 850905499, "population": 8000, "infected": 64, "weeks": 14, "capacity": 119, "budget": 21, "target": 6},
    # seed 9036; 10000 people, 14 weeks, 134 beds, budget 21, 14-step plan
    {"seed": 362212790, "population": 10000, "infected": 60, "weeks": 14, "capacity": 134, "budget": 21, "target": 12},
    # seed 9037; 8000 people, 12 weeks, 153 beds, budget 15, 12-step plan
    {"seed": 707625237, "population": 8000, "infected": 77, "weeks": 12, "capacity": 153, "budget": 15, "target": 11},
    # seed 9038; 8000 people, 10 weeks, 185 beds, budget 8, 10-step plan
    {"seed": 90128617, "population": 8000, "infected": 21, "weeks": 10, "capacity": 185, "budget": 8, "target": 8},
    # seed 9039; 5000 people, 12 weeks, 88 beds, budget 16, 12-step plan
    {"seed": 312642918, "population": 5000, "infected": 38, "weeks": 12, "capacity": 88, "budget": 16, "target": 1},
    # seed 9040; 8000 people, 10 weeks, 106 beds, budget 11, 10-step plan
    {"seed": 729573027, "population": 8000, "infected": 48, "weeks": 10, "capacity": 106, "budget": 11, "target": 3},
    # seed 9041; 5000 people, 12 weeks, 82 beds, budget 15, 12-step plan
    {"seed": 643026713, "population": 5000, "infected": 77, "weeks": 12, "capacity": 82, "budget": 15, "target": 15},
    # seed 9042; 8000 people, 12 weeks, 135 beds, budget 16, 12-step plan
    {"seed": 345880655, "population": 8000, "infected": 56, "weeks": 12, "capacity": 135, "budget": 16, "target": 2},
    # seed 9043; 8000 people, 14 weeks, 191 beds, budget 15, 14-step plan
    {"seed": 271527622, "population": 8000, "infected": 76, "weeks": 14, "capacity": 191, "budget": 15, "target": 12},
    # seed 9044; 8000 people, 12 weeks, 185 beds, budget 15, 12-step plan
    {"seed": 362143806, "population": 8000, "infected": 76, "weeks": 12, "capacity": 185, "budget": 15, "target": 7},
    # seed 9045; 10000 people, 12 weeks, 163 beds, budget 18, 12-step plan
    {"seed": 304873120, "population": 10000, "infected": 46, "weeks": 12, "capacity": 163, "budget": 18, "target": 2},
    # seed 9046; 10000 people, 14 weeks, 163 beds, budget 21, 14-step plan
    {"seed": 172947015, "population": 10000, "infected": 80, "weeks": 14, "capacity": 163, "budget": 21, "target": 2},
    # seed 9047; 8000 people, 10 weeks, 187 beds, budget 8, 10-step plan
    {"seed": 807902249, "population": 8000, "infected": 37, "weeks": 10, "capacity": 187, "budget": 8, "target": 3},
    # seed 9048; 10000 people, 12 weeks, 148 beds, budget 13, 12-step plan
    {"seed": 615555230, "population": 10000, "infected": 30, "weeks": 12, "capacity": 148, "budget": 13, "target": 2},
    # seed 9049; 10000 people, 12 weeks, 234 beds, budget 12, 12-step plan
    {"seed": 890312629, "population": 10000, "infected": 58, "weeks": 12, "capacity": 234, "budget": 12, "target": 12},
    # seed 9050; 8000 people, 12 weeks, 152 beds, budget 13, 12-step plan
    {"seed": 952566778, "population": 8000, "infected": 60, "weeks": 12, "capacity": 152, "budget": 13, "target": 18},
    # seed 9051; 8000 people, 12 weeks, 103 beds, budget 17, 12-step plan
    {"seed": 579156667, "population": 8000, "infected": 60, "weeks": 12, "capacity": 103, "budget": 17, "target": 12},
    # seed 9052; 5000 people, 10 weeks, 69 beds, budget 15, 10-step plan
    {"seed": 1009636714, "population": 5000, "infected": 75, "weeks": 10, "capacity": 69, "budget": 15, "target": 5},
    # seed 9053; 8000 people, 12 weeks, 184 beds, budget 13, 12-step plan
    {"seed": 878339593, "population": 8000, "infected": 28, "weeks": 12, "capacity": 184, "budget": 13, "target": 3},
    # seed 9054; 10000 people, 12 weeks, 219 beds, budget 18, 12-step plan
    {"seed": 436622287, "population": 10000, "infected": 68, "weeks": 12, "capacity": 219, "budget": 18, "target": 9},
    # seed 9055; 5000 people, 10 weeks, 87 beds, budget 10, 10-step plan
    {"seed": 266108542, "population": 5000, "infected": 67, "weeks": 10, "capacity": 87, "budget": 10, "target": 8},
    # seed 9056; 10000 people, 14 weeks, 242 beds, budget 12, 14-step plan
    {"seed": 557661796, "population": 10000, "infected": 32, "weeks": 14, "capacity": 242, "budget": 12, "target": 7},
    # seed 9057; 5000 people, 14 weeks, 100 beds, budget 19, 14-step plan
    {"seed": 688839765, "population": 5000, "infected": 45, "weeks": 14, "capacity": 100, "budget": 19, "target": 4},
    # seed 9058; 10000 people, 14 weeks, 219 beds, budget 21, 14-step plan
    {"seed": 832954370, "population": 10000, "infected": 75, "weeks": 14, "capacity": 219, "budget": 21, "target": 8},
    # seed 9059; 10000 people, 10 weeks, 101 beds, budget 11, 10-step plan
    {"seed": 55270718, "population": 10000, "infected": 37, "weeks": 10, "capacity": 101, "budget": 11, "target": 1},
    # seed 9060; 5000 people, 10 weeks, 63 beds, budget 10, 10-step plan
    {"seed": 120188546, "population": 5000, "infected": 77, "weeks": 10, "capacity": 63, "budget": 10, "target": 8},
    # seed 9061; 10000 people, 14 weeks, 190 beds, budget 13, 14-step plan
    {"seed": 114821684, "population": 10000, "infected": 20, "weeks": 14, "capacity": 190, "budget": 13, "target": 7},
    # seed 9062; 10000 people, 14 weeks, 178 beds, budget 14, 14-step plan
    {"seed": 843715022, "population": 10000, "infected": 56, "weeks": 14, "capacity": 178, "budget": 14, "target": 26},
    # seed 9063; 8000 people, 10 weeks, 122 beds, budget 11, 10-step plan
    {"seed": 947688102, "population": 8000, "infected": 24, "weeks": 10, "capacity": 122, "budget": 11, "target": 3},
    # seed 9064; 8000 people, 10 weeks, 136 beds, budget 7, 10-step plan
    {"seed": 176147979, "population": 8000, "infected": 59, "weeks": 10, "capacity": 136, "budget": 7, "target": 2},
    # seed 9065; 8000 people, 12 weeks, 178 beds, budget 10, 12-step plan
    {"seed": 370084460, "population": 8000, "infected": 30, "weeks": 12, "capacity": 178, "budget": 10, "target": 6},
    # seed 9066; 10000 people, 12 weeks, 214 beds, budget 15, 12-step plan
    {"seed": 1058301262, "population": 10000, "infected": 70, "weeks": 12, "capacity": 214, "budget": 15, "target": 11},
    # seed 9067; 5000 people, 14 weeks, 109 beds, budget 13, 14-step plan
    {"seed": 52805802, "population": 5000, "infected": 68, "weeks": 14, "capacity": 109, "budget": 13, "target": 21},
    # seed 9068; 10000 people, 10 weeks, 115 beds, budget 9, 10-step plan
    {"seed": 103557685, "population": 10000, "infected": 25, "weeks": 10, "capacity": 115, "budget": 9, "target": 3},
    # seed 9069; 8000 people, 10 weeks, 158 beds, budget 12, 10-step plan
    {"seed": 1008089559, "population": 8000, "infected": 28, "weeks": 10, "capacity": 158, "budget": 12, "target": 1},
    # seed 9070; 8000 people, 14 weeks, 181 beds, budget 18, 14-step plan
    {"seed": 675897580, "population": 8000, "infected": 69, "weeks": 14, "capacity": 181, "budget": 18, "target": 7},
    # seed 9071; 5000 people, 14 weeks, 92 beds, budget 18, 14-step plan
    {"seed": 1038759296, "population": 5000, "infected": 74, "weeks": 14, "capacity": 92, "budget": 18, "target": 16},
    # seed 9072; 5000 people, 10 weeks, 87 beds, budget 12, 10-step plan
    {"seed": 345210628, "population": 5000, "infected": 32, "weeks": 10, "capacity": 87, "budget": 12, "target": 3},
    # seed 9073; 10000 people, 12 weeks, 130 beds, budget 16, 12-step plan
    {"seed": 191683567, "population": 10000, "infected": 48, "weeks": 12, "capacity": 130, "budget": 16, "target": 6},
    # seed 9074; 10000 people, 14 weeks, 235 beds, budget 19, 14-step plan
    {"seed": 620262960, "population": 10000, "infected": 65, "weeks": 14, "capacity": 235, "budget": 19, "target": 11},
    # seed 9075; 10000 people, 10 weeks, 198 beds, budget 10, 10-step plan
    {"seed": 814518336, "population": 10000, "infected": 80, "weeks": 10, "capacity": 198, "budget": 10, "target": 20},
    # seed 9076; 8000 people, 10 weeks, 135 beds, budget 10, 10-step plan
    {"seed": 474865138, "population": 8000, "infected": 66, "weeks": 10, "capacity": 135, "budget": 10, "target": 8},
    # seed 9077; 5000 people, 10 weeks, 122 beds, budget 6, 10-step plan
    {"seed": 173915038, "population": 5000, "infected": 26, "weeks": 10, "capacity": 122, "budget": 6, "target": 7},
    # seed 9078; 8000 people, 10 weeks, 197 beds, budget 8, 10-step plan
    {"seed": 323851492, "population": 8000, "infected": 68, "weeks": 10, "capacity": 197, "budget": 8, "target": 13},
    # seed 9079; 5000 people, 12 weeks, 82 beds, budget 17, 12-step plan
    {"seed": 970058037, "population": 5000, "infected": 44, "weeks": 12, "capacity": 82, "budget": 17, "target": 6},
    # seed 9080; 5000 people, 14 weeks, 114 beds, budget 15, 14-step plan
    {"seed": 480724003, "population": 5000, "infected": 31, "weeks": 14, "capacity": 114, "budget": 15, "target": 13},
    # seed 9081; 8000 people, 14 weeks, 90 beds, budget 17, 14-step plan
    {"seed": 307489854, "population": 8000, "infected": 27, "weeks": 14, "capacity": 90, "budget": 17, "target": 1},
    # seed 9082; 10000 people, 10 weeks, 139 beds, budget 8, 10-step plan
    {"seed": 105844348, "population": 10000, "infected": 28, "weeks": 10, "capacity": 139, "budget": 8, "target": 4},
    # seed 9083; 8000 people, 14 weeks, 146 beds, budget 18, 14-step plan
    {"seed": 908117095, "population": 8000, "infected": 60, "weeks": 14, "capacity": 146, "budget": 18, "target": 11},
    # seed 9084; 5000 people, 10 weeks, 123 beds, budget 12, 10-step plan
    {"seed": 159696163, "population": 5000, "infected": 48, "weeks": 10, "capacity": 123, "budget": 12, "target": 7},
    # seed 9085; 8000 people, 14 weeks, 199 beds, budget 15, 14-step plan
    {"seed": 439413915, "population": 8000, "infected": 48, "weeks": 14, "capacity": 199, "budget": 15, "target": 18},
    # seed 9086; 5000 people, 14 weeks, 83 beds, budget 21, 14-step plan
    {"seed": 276368755, "population": 5000, "infected": 39, "weeks": 14, "capacity": 83, "budget": 21, "target": 14},
    # seed 9087; 8000 people, 14 weeks, 169 beds, budget 15, 14-step plan
    {"seed": 9516950, "population": 8000, "infected": 62, "weeks": 14, "capacity": 169, "budget": 15, "target": 31},
    # seed 9088; 5000 people, 10 weeks, 88 beds, budget 6, 10-step plan
    {"seed": 475429696, "population": 5000, "infected": 39, "weeks": 10, "capacity": 88, "budget": 6, "target": 6},
    # seed 9089; 8000 people, 10 weeks, 94 beds, budget 10, 10-step plan
    {"seed": 112573549, "population": 8000, "infected": 52, "weeks": 10, "capacity": 94, "budget": 10, "target": 11},
    # seed 9090; 8000 people, 10 weeks, 185 beds, budget 11, 10-step plan
    {"seed": 378099274, "population": 8000, "infected": 28, "weeks": 10, "capacity": 185, "budget": 11, "target": 3},
    # seed 9091; 10000 people, 12 weeks, 139 beds, budget 18, 12-step plan
    {"seed": 976192470, "population": 10000, "infected": 53, "weeks": 12, "capacity": 139, "budget": 18, "target": 4},
    # seed 9092; 8000 people, 10 weeks, 147 beds, budget 9, 10-step plan
    {"seed": 258195947, "population": 8000, "infected": 61, "weeks": 10, "capacity": 147, "budget": 9, "target": 6},
    # seed 9093; 8000 people, 10 weeks, 144 beds, budget 7, 10-step plan
    {"seed": 761630975, "population": 8000, "infected": 35, "weeks": 10, "capacity": 144, "budget": 7, "target": 4},
    # seed 9094; 10000 people, 12 weeks, 198 beds, budget 14, 12-step plan
    {"seed": 599674121, "population": 10000, "infected": 35, "weeks": 12, "capacity": 198, "budget": 14, "target": 9},
    # seed 9095; 8000 people, 12 weeks, 187 beds, budget 8, 12-step plan
    {"seed": 1069315149, "population": 8000, "infected": 27, "weeks": 12, "capacity": 187, "budget": 8, "target": 6},
    # seed 9096; 5000 people, 10 weeks, 81 beds, budget 14, 10-step plan
    {"seed": 642326860, "population": 5000, "infected": 55, "weeks": 10, "capacity": 81, "budget": 14, "target": 2},
    # seed 9097; 8000 people, 14 weeks, 157 beds, budget 14, 14-step plan
    {"seed": 613275094, "population": 8000, "infected": 52, "weeks": 14, "capacity": 157, "budget": 14, "target": 30},
    # seed 9098; 5000 people, 12 weeks, 107 beds, budget 13, 12-step plan
    {"seed": 43941314, "population": 5000, "infected": 29, "weeks": 12, "capacity": 107, "budget": 13, "target": 5},
    # seed 9099; 10000 people, 10 weeks, 246 beds, budget 9, 10-step plan
    {"seed": 158302861, "population": 10000, "infected": 47, "weeks": 10, "capacity": 246, "budget": 9, "target": 9},
)
