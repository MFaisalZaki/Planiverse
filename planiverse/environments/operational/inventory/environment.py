"""A multi-echelon supply chain on or-gym: a retailer, a distributor and a manufacturer, an
order-up-to level to set for the retailer and one for the two stages behind it every period, and
a season to get through with the profit at or above the target and the retailer's lost sales
within what is allowed.

or-gym (https://github.com/hubbs5/or-gym, MIT; Hubbs, Perez, Sarwar, Sahinidis, Grossmann and
Wassick 2020, https://arxiv.org/abs/2008.06319) is a library of operations research problems as
reinforcement learning environments from Carnegie Mellon and Dow. Its multi-echelon inventory
problem (`InvManagement-v1`, the lost-sales variant) is a serial chain: customers buy from the
retailer, the retailer orders from the distributor, the distributor from the manufacturer and
the manufacturer from a raw material supplier with unlimited stock. Each order is shipped as far
as the supplier's capacity and stock allow and arrives after the link's lead time; a sale the
retailer cannot make is lost. It is a dependency installed from PyPI; nothing of it is included
here.

The decisions are the planner's: each period, an order-up-to (base-stock) level for the retailer
and one for the distributor and the manufacturer together, each low, mid or high, from the
levels the instance sets. The orders the levels ask for are or-gym's own `base_stock_action`,
and the period that follows is or-gym's own `_STEP`: the shipments, the arrivals, the demand,
the sales, the lost sales and the period's discounted profit are all its arithmetic.

## Trajectory constraints as a goal

The retailer's lost sales must stay within the instance's allowance throughout, so the first
period that takes them over it is `is_terminal`. The profit accumulates in the state, and the
goal is the season complete with that profit at or above the target, which is a plan's final
state. The target is the most profit any fixed base-stock policy (the same levels every period)
earns inside the allowance, or that policy with its levels dropped to low for the last periods,
so the planner is asked to do at least as well as the best rule by varying the levels period by
period.

## Determinism and state

or-gym draws each period's demand from a scipy distribution on numpy's global random state. Here
the whole season's demand is drawn once per instance from the instance's seed and handed to
or-gym as its user-supplied demand (`dist=5`), so a period's demand is the instance's and
expansion is deterministic. A state is what a period leaves behind: the period, the stock on
hand at each stage, the shipments in the pipeline to each, the profit so far and the lost sales
so far. or-gym's state is plain arrays, so the environment writes a state's numbers into them
and steps one period, and two order histories that leave the same stock, pipeline and
accumulators are one state.
"""
import contextlib
import io
from collections import namedtuple

from planiverse.environments.base import Environment
from planiverse.environments.generation import draw_until, rng
from planiverse.rendering.text import Canvas, bar

#: The three stages that hold stock, downstream first; the fourth is the raw material supplier.
STAGES = ("retailer", "distributor", "manufacturer")
#: Which of an action's two levels each stage takes: the retailer its own, the distributor and
#: the manufacturer the upstream one.
SETTING = (0, 1, 1)
#: The order-up-to levels a stage may be set to, as a share of what its echelon needs to cover
#: its cumulative lead time and the period in hand at mean demand.
LEVELS = {"low": 0.8, "mid": 1.1, "high": 1.4}
#: The demand distributions or-gym offers, drawn here once per season (see `_demand`).
DISTRIBUTIONS = ("poisson", "binomial", "uniform", "geometric")
#: How many of the last periods the run-down reference policies hold both levels at low.
RUN_DOWN = (2, 3, 4)


def _or_gym():
    """or-gym's inventory module, imported on first use.

    or-gym 0.5.0 is unmaintained and was written against numpy 1, and numpy 2 removed two
    aliases it was written with: `np.Inf`, which `_STEP` and `base_stock_action` use for the raw
    material supplier's unlimited stock, and `np.product`. Both are put back before or-gym runs.
    Gym 0.26 also prints a notice at import that it does not support numpy 2; the parts of it
    or-gym uses here (the `Env` base class and `spaces.Box`) work, so the notice is kept off the
    console.
    """
    import numpy as np

    # numpy 2.0 removed these aliases; or-gym still spells them the numpy 1 way.
    if not hasattr(np, "Inf"):
        np.Inf = np.inf
    if not hasattr(np, "product"):
        np.product = np.prod
    with contextlib.redirect_stderr(io.StringIO()):
        from or_gym.envs.supply_chain import inventory_management
    return inventory_management


def _chain_class():
    module = _or_gym()

    class Chain(module.InvManagementLostSalesEnv):
        """or-gym's lost-sales chain with one fix: `base_stock_action` reads the inventory
        position from `_update_base_stock_policy_state`, which stores it on `self.state` but
        returns nothing, so the policy fails with a `TypeError`. Returning it is the fix."""

        def _update_base_stock_policy_state(self):
            super()._update_base_stock_policy_state()
            return self.state

    return Chain


class InventoryAction:
    """`stock(r, u)`: the order-up-to level `r` at the retailer and `u` at the distributor and
    the manufacturer, each `low`, `mid` or `high`."""

    def __init__(self, retailer, upstream):
        if retailer not in LEVELS or upstream not in LEVELS:
            raise ValueError(f"unknown setting: stock({retailer!r}, {upstream!r})")
        self.retailer, self.upstream = retailer, upstream
        self.name = f"stock({retailer}, {upstream})"

    @property
    def levels(self):
        """The level each of `STAGES` is set to."""
        return tuple((self.retailer, self.upstream)[setting] for setting in SETTING)

    @classmethod
    def parse(cls, text):
        inside = str(text).strip()[len("stock("):-1]
        retailer, upstream = (part.strip() for part in inside.split(","))
        return cls(retailer, upstream)

    def cost(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, InventoryAction) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    __repr__ = __str__


ACTIONS = tuple(InventoryAction(r, u) for r in LEVELS for u in LEVELS)


#: What a picture of the chain needs that a state does not hold: the bar scale of each of
#: `STAGES`, its highest order-up-to level. One per season, shared by its states.
ChainMap = namedtuple("ChainMap", ["scale"])

#: The length of a stage's stock bar, in characters.
BAR = 10


class InventoryState:
    """A period's end: the stock on hand and in the pipeline at each stage, the profit and the
    lost sales so far, and the last period's demand, sales, lost sales and retailer's order.
    Identity is the period, the stock, the pipeline and the two accumulators."""

    def readings(self):
        """The state's numbers by name, which its OpenSpiel game gives as the named pieces
        of its observation."""
        demand, sold, _, ordered = self.last
        values = dict(zip(STAGES, self.on_hand))
        values.update({"demand": demand, "sold": sold, "ordered": ordered,
                       "lost sales": self.lost, "allowed": self.allowed,
                       "profit": self.profit, "target": self.target})
        return values

    def __init__(self, period, on_hand, pipeline, profit, lost, last=(0, 0, 0, 0), target=0.0,
                 allowed=0, horizon=0, worth=None, depth=0, chain=None):
        self.period, self.horizon = period, horizon
        #: The chain a picture is drawn on (a `ChainMap`, one per season), outside identity.
        self.chain = chain
        self.on_hand = tuple(int(v) for v in on_hand)
        self.pipeline = tuple(tuple(int(v) for v in shipments) for shipments in pipeline)
        self.profit, self.lost = round(profit, 2), int(lost)
        self.last = tuple(int(v) for v in last)          # demand, sold, lost, retailer's order
        self.target, self.allowed, self.depth = target, allowed, depth
        #: The profit so far with the stock on hand and in transit counted at the raw material
        #: price it cost the chain, so that stock bought is not yet counted as money lost; at
        #: the end of the season unsold stock is worth nothing, and this is the profit alone.
        self.worth = self.profit if worth is None else round(worth, 2)
        self.key = (self.period, self.on_hand, self.pipeline, self.profit, self.lost)
        # The literals name every part of the identity, because a planner's closed list keys
        # on them: a literal that rounded the profit would let it close a state on another
        # that differs only in money, and then report an instance unsolvable that is not.
        literals = [f"period({period})", f"lost({self.lost})", f"profit({self.profit:.2f})"]
        for stage, stock, shipments in zip(STAGES, self.on_hand, self.pipeline):
            literals.append(f"on_hand({stage}, {stock})")
            literals += [f"in_transit({stage}, {due}, {quantity})"
                         for due, quantity in enumerate(shipments, 1)]
        self.literals = frozenset(literals)

    def __eq__(self, other):
        return isinstance(other, InventoryState) and self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        demand, sold, lost, _ = self.last
        stock = ", ".join(f"{stage} {v}" for stage, v in zip(STAGES, self.on_hand))
        transit = ", ".join(str(sum(shipments)) for shipments in self.pipeline)
        return (f"period {self.period}: on hand {stock}; in transit {transit}; last period "
                f"demand {demand}, sold {sold}, lost {lost}; lost {self.lost} of {self.allowed} "
                f"allowed; profit {self.profit:.2f} against {self.target:.2f}")

    def __repr__(self):
        return (f"<InventoryState(period={self.period}, on_hand={self.on_hand}, "
                f"profit={self.profit:.2f}, lost={self.lost})>")

    def picture(self):
        """The supply chain as a row of boxes, then a status line.

        Raw material flows from the supplier through the manufacturer and the distributor to
        the retailer and its customers. Each stage's box holds its stock on hand, as a number
        and as a bar against its highest order-up-to level; the arrow into it carries the
        shipments in transit, the one due soonest nearest the box; the last box is this
        period's demand, what was sold and what was lost.
        """
        demand, sold, lost, _ = self.last
        scale = (self.chain.scale if self.chain is not None
                 else (max(self.on_hand) or 1,) * len(STAGES))
        boxes = [("raw", "supply", "", "")]
        arrows = []
        for stage in reversed(range(len(STAGES))):
            arrows.append(self.pipeline[stage])
            boxes.append((STAGES[stage], f"{self.on_hand[stage]} on hand",
                          bar(self.on_hand[stage], scale[stage], BAR), ""))
        arrows.append(())
        boxes.append(("customers", f"demand {demand}", f"sold   {sold}", f"lost   {lost}"))
        widths = [max(len(text) for text in box) + 2 for box in boxes]
        gaps = [max(4, 4 * len(shipments) + 3) for shipments in arrows]
        canvas = Canvas(sum(widths) + sum(gaps) + 2, 6)
        x = 0
        for number, box in enumerate(boxes):
            width = widths[number]
            canvas.rect(x, 0, x + width - 1, 5)
            for row, text in enumerate(box):
                canvas.write(x + 1, row + 1, text)
            if number < len(arrows):
                start, end = x + width, x + width + gaps[number] - 1
                canvas.hline(start, end - 1, 3, char="─")
                canvas.put(end, 3, "▶")
                slot = start + 1
                for quantity in reversed(arrows[number]):
                    canvas.write(slot, 2, f"{quantity:>3}")
                    slot += 4
                x = end + 1
        status = (f"period {self.period}/{self.horizon} · profit {self.profit:.2f} of "
                  f"{self.target:.2f} · lost sales {self.lost} of {self.allowed}")
        return f"{canvas}\n\n{status}"


def reference_plans(periods):
    """The fixed policies the target is set off: each of the nine settings held every period,
    and each of the other eight held until the last two, three or four periods and then both
    levels at low, which runs the stock down as the season ends."""
    low = InventoryAction("low", "low")
    plans = [[action] * periods for action in ACTIONS]
    for tail in RUN_DOWN:
        plans += [[action] * (periods - tail) + [low] * tail for action in ACTIONS
                  if action != low]
    return plans


def _poisson(random_, mean):
    """Knuth's multiplication method: exact, and quick for the means drawn here."""
    import math

    limit, count, product = math.exp(-mean), 0, random_.random()
    while product > limit:
        count += 1
        product *= random_.random()
    return count


def _demand(random_, distribution, parameters, periods):
    """A season's demand from one of or-gym's four distributions, drawn with `random_`."""
    if distribution == "poisson":
        return [_poisson(random_, parameters["mu"]) for _ in range(periods)]
    if distribution == "binomial":
        n, p = parameters["n"], parameters["p"]
        return [sum(random_.random() < p for _ in range(n)) for _ in range(periods)]
    if distribution == "uniform":            # scipy's randint: low inclusive, high exclusive
        return [random_.randrange(parameters["low"], parameters["high"]) for _ in range(periods)]
    if distribution == "geometric":          # trials to the first success, so at least one
        draws = []
        for _ in range(periods):
            trials = 1
            while random_.random() >= parameters["p"]:
                trials += 1
            draws.append(trials)
        return draws
    raise ValueError(f"unknown distribution {distribution!r}")


def _mean(distribution, parameters):
    if distribution == "poisson":
        return parameters["mu"]
    if distribution == "binomial":
        return parameters["n"] * parameters["p"]
    if distribution == "uniform":
        return (parameters["low"] + parameters["high"] - 1) / 2
    return 1 / parameters["p"]


def order_up_to_levels(lead_times, mean):
    """The low, mid and high echelon level at each stage: `LEVELS`' shares of the mean
    demand over the stage's cumulative lead time and the period in hand."""
    levels, cover = [], 0
    for lead in lead_times:
        cover += lead
        levels.append({name: int(round(share * mean * (cover + 1)))
                       for name, share in LEVELS.items()})
    return levels


class InventoryEnv(Environment):
    """A season with the profit at or above the target and the lost sales within the allowance."""

    def __init__(self):
        super().__init__("inventory")
        self.instance = None
        self.index = None
        self.state = None
        self.witness = None
        self.witness_expansions = None
        self.__chain__ = None

    # ------------------------------------------------------------------ instances

    def set_index(self, index):
        if not 0 <= index < len(SEASONS):
            raise IndexError(f"Invalid index: {index}. There are {len(SEASONS)} seasons, so the "
                             f"index must be 0-{len(SEASONS) - 1}.")
        self.set_instance(SEASONS[index])
        self.index = index

    def set_instance(self, instance):
        """Select a season: `periods`, `demand` (one a period), `initial` (stock at each of the
        three stages), `price`, `costs` (what each stage pays its supplier for a unit, and
        the raw material's), `penalties` (a unit short at each of the four), `holding` (a unit
        held at each of the three),
        `capacity` and `lead_times` (each supplier's), `discount`, `levels` (each stage's
        `low`, `mid` and `high` order-up-to level), `allowed` (lost sales) and `target`
        (profit)."""
        for key in ("periods", "demand", "initial", "price", "costs", "penalties", "holding",
                    "capacity", "lead_times", "discount", "levels", "allowed", "target"):
            if key not in instance:
                raise ValueError(f"a season needs `{key}`")
        periods = int(instance["periods"])
        if len(instance["demand"]) != periods:
            raise ValueError(f"a season of {periods} periods needs {periods} demands")
        if any(int(lead) < 1 for lead in instance["lead_times"]):
            raise ValueError("every lead time must be at least one period")
        self.instance = {"periods": periods, "demand": [int(v) for v in instance["demand"]],
                         "initial": [int(v) for v in instance["initial"]],
                         "price": float(instance["price"]),
                         "costs": [float(v) for v in instance["costs"]],
                         "penalties": [float(v) for v in instance["penalties"]],
                         "holding": [float(v) for v in instance["holding"]],
                         "capacity": [int(v) for v in instance["capacity"]],
                         "lead_times": [int(v) for v in instance["lead_times"]],
                         "discount": float(instance["discount"]),
                         "levels": [{name: int(level[name]) for name in LEVELS}
                                    for level in instance["levels"]],
                         "allowed": int(instance["allowed"]), "target": float(instance["target"])}
        self.index = None
        self.witness = self.witness_expansions = None
        self.__chain__ = None

    def generate_instance(self, seed=None, periods=None, attempts=60):
        """Draw a season, select it, and return it as the dict `set_instance` takes.

        The demand distribution is one of or-gym's four (Poisson, binomial, uniform,
        geometric) with drawn parameters, and the season's demand is drawn from it once.
        The costs, capacities, lead times and initial stock are drawn round or-gym's own
        defaults, scaled to the mean demand, and the lost sales allowed is a share of the
        season's demand. The policies of `reference_plans` are run; the target is the most
        profit any of them ends the season with inside the allowance, and a draw is thrown
        back when none does.
        """
        if periods is not None and periods <= max(RUN_DOWN):
            raise ValueError(f"a season needs more than {max(RUN_DOWN)} periods")
        random_, seed = rng(seed)
        found = {}

        def draw(attempt):
            distribution = random_.choice(DISTRIBUTIONS)
            if distribution == "poisson":
                parameters = {"mu": random_.randint(8, 20)}
            elif distribution == "binomial":
                parameters = {"n": random_.randint(20, 40),
                              "p": round(random_.uniform(0.3, 0.6), 2)}
            elif distribution == "uniform":
                low = random_.randint(4, 12)
                parameters = {"low": low, "high": low + random_.randint(6, 14)}
            else:
                parameters = {"p": round(random_.uniform(0.07, 0.12), 2)}
            mean = _mean(distribution, parameters)
            length = periods or random_.randint(12, 20)
            lead_times = [random_.randint(1, 3) for _ in STAGES]
            supply = round(random_.uniform(0.3, 0.6), 2)
            costs = [supply]
            for _ in STAGES:
                costs.insert(0, round(costs[0] + random_.uniform(0.15, 0.45), 2))
            holding = sorted((round(random_.uniform(0.03, 0.2), 2) for _ in STAGES), reverse=True)
            penalties = sorted((round(random_.uniform(0.02, 0.15), 3) for _ in range(4)),
                               reverse=True)
            demand = _demand(random_, distribution, parameters, length)
            return {"seed": seed, "periods": length, "demand": demand,
                    "distribution": distribution, "parameters": parameters,
                    "initial": [int(round(mean * random_.uniform(1.0, 2.5))) for _ in STAGES],
                    "price": round(costs[0] + random_.uniform(0.5, 1.2), 2), "costs": costs,
                    "penalties": penalties, "holding": holding,
                    "capacity": [int(round(mean * random_.uniform(1.5, 3.0))) for _ in STAGES],
                    "lead_times": lead_times, "discount": round(random_.uniform(0.95, 1.0), 2),
                    "levels": order_up_to_levels(lead_times, mean),
                    "allowed": int(round(sum(demand) * random_.uniform(0.04, 0.12))),
                    "target": 0.0}

        def accept(instance):
            self.set_instance(instance)
            self.instance["target"] = float("-inf")
            plans = reference_plans(instance["periods"])
            best = None
            for plan in plans:
                last = self.simulate(plan)[-1]
                if last.period == instance["periods"] and last.lost <= instance["allowed"]:
                    if best is None or last.profit > best[0]:
                        best = (last.profit, plan)
            if best is None:
                return False
            instance["target"] = best[0]
            found["plan"], found["measured"] = best[1], len(plans)
            return True

        instance = draw_until(draw, accept, attempts, "inventory season")
        self.set_instance(instance)
        self.witness, self.witness_expansions = found["plan"], found["measured"]
        return instance

    # ------------------------------------------------------------------ the simulator

    def _chain(self):
        if self.__chain__ is None:
            inst = self.instance
            self.__chain__ = _chain_class()(
                periods=inst["periods"], I0=list(inst["initial"]), p=inst["price"],
                r=list(inst["costs"]), k=list(inst["penalties"]), h=list(inst["holding"]),
                c=list(inst["capacity"]), L=list(inst["lead_times"]), alpha=inst["discount"],
                dist=5, user_D=list(inst["demand"]))
        return self.__chain__

    def _step(self, state, action):
        """Write `state` into or-gym's arrays, set the order-up-to levels `action` names, and
        let or-gym's `_STEP` run the period."""
        import numpy as np

        chain, n = self._chain(), state.period
        chain.period = n
        chain.I[n, :] = state.on_hand
        chain.T[n, :] = [sum(shipments) for shipments in state.pipeline]
        leads = self.instance["lead_times"]
        for stage, (lead, shipments) in enumerate(zip(leads, state.pipeline)):
            for offset, quantity in enumerate(shipments):      # placed `lead - offset` periods ago
                if n - lead + offset >= 0:
                    chain.R[n - lead + offset, stage] = quantity
        if n >= 1:
            chain.B[n - 1, :] = 0                              # lost sales, so nothing backlogged
        levels = np.array([self.instance["levels"][stage][level]
                           for stage, level in enumerate(action.levels)])
        chain._STEP(chain.base_stock_action(levels))
        pipeline = tuple(shipments[1:] + (int(chain.R[n, stage]),)
                         for stage, shipments in enumerate(state.pipeline))
        last = (chain.D[n], chain.S[n, 0], chain.LS[n, 0], chain.R[n, 0])
        profit = round(state.profit + float(chain.P[n]), 2)
        worth = self._worth(n + 1, chain.I[n + 1, :], pipeline, profit)
        return InventoryState(n + 1, chain.I[n + 1, :], pipeline, profit,
                              state.lost + chain.LS[n, 0], last, state.target, state.allowed,
                              state.horizon, worth, state.depth + 1, chain=state.chain)

    def _worth(self, period, on_hand, pipeline, profit):
        """The profit with every unit in the chain counted at the raw material price. or-gym's
        profit is the chain's: what one stage pays another is the other's revenue, so the two
        cancel, and a unit costs the chain the raw material price when the manufacturer orders
        it. Stock left when the season ends is worth nothing."""
        if period >= self.instance["periods"]:
            return profit
        units = sum(on_hand) + sum(sum(shipments) for shipments in pipeline)
        return profit + self.instance["costs"][-1] * units

    # ------------------------------------------------------------------- contract

    def reset(self):
        if self.instance is None:
            self.set_index(0)
        inst = self.instance
        pipeline = [(0,) * lead for lead in inst["lead_times"]]
        self.state = InventoryState(0, inst["initial"], pipeline, 0.0, 0, target=inst["target"],
                                    allowed=inst["allowed"], horizon=inst["periods"],
                                    worth=self._worth(0, inst["initial"], pipeline, 0.0),
                                    chain=ChainMap(tuple(level["high"]
                                                         for level in inst["levels"])))
        return self.state, {"season": self.index, "target": inst["target"],
                            "allowed": inst["allowed"], "generated": self.index is None}

    def is_goal(self, state):
        return (state.period >= self.instance["periods"] and state.lost <= self.instance["allowed"]
                and state.profit >= self.instance["target"])

    def is_terminal(self, state):
        return state.lost > self.instance["allowed"] or (
            state.period >= self.instance["periods"] and not self.is_goal(state))

    def get_actions(self, state=None):
        return list(ACTIONS)

    def successors(self, state):
        if self.is_goal(state) or self.is_terminal(state):
            return []
        children = []
        for action in ACTIONS:
            child = self._step(state, action)
            if child != state:
                children.append((action, child))
        return children

    def __advance__(self, state, action):
        if self.is_goal(state) or self.is_terminal(state):
            return state
        if not isinstance(action, InventoryAction):
            action = InventoryAction.parse(action)
        return self._step(state, action)

    def close(self):
        self.__chain__ = None


#: The bundled seasons: `generate_instance(seed)` for the seed beside each, embedded as the
#: plain data `set_instance` takes, with the plan each was accepted on in
#: `tests/data/inventory_solutions.json`.
SEASONS = (
    # seed 9000; uniform (low 9, high 17), 20 periods, lead times [2, 1, 3], 20-step plan
    {"seed": 9000, "periods": 20, "demand": [15, 9, 16, 12, 11, 9, 16, 13, 16, 15, 12, 12, 15, 9, 10, 16, 12, 10, 10, 13], "distribution": "uniform", "parameters": {"low": 9, "high": 17}, "initial": [29, 24, 23], "price": 2.69, "costs": [1.52, 1.15, 0.75, 0.6], "penalties": [0.138, 0.103, 0.083, 0.061], "holding": [0.13, 0.11, 0.06], "capacity": [30, 31, 22], "lead_times": [2, 1, 3], "discount": 0.96, "levels": [{"low": 30, "mid": 41, "high": 52}, {"low": 40, "mid": 55, "high": 70}, {"low": 70, "mid": 96, "high": 122}], "allowed": 30, "target": 270.76},
    # seed 9001; binomial (n 39, p 0.33), 18 periods, lead times [1, 3, 2], 18-step plan
    {"seed": 9001, "periods": 18, "demand": [6, 14, 9, 10, 9, 12, 13, 19, 10, 14, 7, 8, 11, 10, 15, 14, 5, 10], "distribution": "binomial", "parameters": {"n": 39, "p": 0.33}, "initial": [26, 28, 31], "price": 2.45, "costs": [1.51, 1.32, 0.97, 0.55], "penalties": [0.102, 0.06, 0.028, 0.021], "holding": [0.14, 0.12, 0.04], "capacity": [30, 33, 22], "lead_times": [1, 3, 2], "discount": 0.98, "levels": [{"low": 21, "mid": 28, "high": 36}, {"low": 51, "mid": 71, "high": 90}, {"low": 72, "mid": 99, "high": 126}], "allowed": 20, "target": 261.57},
    # seed 9002; uniform (low 5, high 11), 18 periods, lead times [1, 1, 1], 18-step plan
    {"seed": 9002, "periods": 18, "demand": [9, 6, 5, 5, 5, 5, 5, 6, 7, 7, 5, 8, 8, 6, 6, 7, 7, 8], "distribution": "uniform", "parameters": {"low": 5, "high": 11}, "initial": [17, 13, 11], "price": 2.23, "costs": [1.19, 0.86, 0.7, 0.4], "penalties": [0.135, 0.12, 0.105, 0.043], "holding": [0.19, 0.18, 0.16], "capacity": [20, 19, 17], "lead_times": [1, 1, 1], "discount": 0.98, "levels": [{"low": 12, "mid": 16, "high": 21}, {"low": 18, "mid": 25, "high": 32}, {"low": 24, "mid": 33, "high": 42}], "allowed": 11, "target": 131.13},
    # seed 9003; poisson (mu 8), 20 periods, lead times [2, 3, 1], 20-step plan
    {"seed": 9003, "periods": 20, "demand": [6, 8, 9, 2, 9, 3, 8, 8, 7, 4, 10, 6, 7, 5, 8, 6, 7, 8, 5, 4], "distribution": "poisson", "parameters": {"mu": 8}, "initial": [19, 17, 12], "price": 2.62, "costs": [1.46, 1.22, 0.9, 0.47], "penalties": [0.146, 0.085, 0.063, 0.043], "holding": [0.13, 0.13, 0.09], "capacity": [12, 16, 17], "lead_times": [2, 3, 1], "discount": 0.98, "levels": [{"low": 19, "mid": 26, "high": 34}, {"low": 38, "mid": 53, "high": 67}, {"low": 45, "mid": 62, "high": 78}], "allowed": 15, "target": 189.18},
    # seed 9004; binomial (n 36, p 0.51), 14 periods, lead times [1, 1, 1], 14-step plan
    {"seed": 9004, "periods": 14, "demand": [19, 17, 17, 14, 19, 19, 19, 18, 16, 15, 16, 20, 14, 18], "distribution": "binomial", "parameters": {"n": 36, "p": 0.51}, "initial": [24, 41, 31], "price": 2.62, "costs": [1.46, 1.21, 0.8, 0.57], "penalties": [0.143, 0.131, 0.076, 0.021], "holding": [0.14, 0.12, 0.08], "capacity": [33, 36, 38], "lead_times": [1, 1, 1], "discount": 1.0, "levels": [{"low": 29, "mid": 40, "high": 51}, {"low": 44, "mid": 61, "high": 77}, {"low": 59, "mid": 81, "high": 103}], "allowed": 26, "target": 454.31},
    # seed 9005; geometric (p 0.07), 18 periods, lead times [1, 2, 1], 18-step plan
    {"seed": 9005, "periods": 18, "demand": [2, 23, 12, 2, 3, 19, 13, 9, 9, 2, 24, 9, 1, 22, 25, 14, 8, 24], "distribution": "geometric", "parameters": {"p": 0.07}, "initial": [18, 31, 29], "price": 2.1, "costs": [1.09, 0.91, 0.73, 0.56], "penalties": [0.107, 0.101, 0.097, 0.043], "holding": [0.14, 0.11, 0.1], "capacity": [33, 36, 30], "lead_times": [1, 2, 1], "discount": 0.97, "levels": [{"low": 23, "mid": 31, "high": 40}, {"low": 46, "mid": 63, "high": 80}, {"low": 57, "mid": 79, "high": 100}], "allowed": 25, "target": 179.38},
    # seed 9006; poisson (mu 11), 12 periods, lead times [1, 1, 1], 12-step plan
    {"seed": 9006, "periods": 12, "demand": [9, 15, 10, 8, 10, 11, 8, 19, 3, 8, 9, 11], "distribution": "poisson", "parameters": {"mu": 11}, "initial": [27, 24, 13], "price": 1.77, "costs": [1.16, 0.79, 0.62, 0.41], "penalties": [0.105, 0.103, 0.045, 0.021], "holding": [0.17, 0.12, 0.04], "capacity": [23, 26, 25], "lead_times": [1, 1, 1], "discount": 0.95, "levels": [{"low": 18, "mid": 24, "high": 31}, {"low": 26, "mid": 36, "high": 46}, {"low": 35, "mid": 48, "high": 62}], "allowed": 13, "target": 108.93},
    # seed 9007; geometric (p 0.09), 17 periods, lead times [1, 1, 2], 17-step plan
    {"seed": 9007, "periods": 17, "demand": [8, 1, 3, 3, 5, 13, 5, 4, 1, 32, 6, 4, 1, 6, 7, 7, 3], "distribution": "geometric", "parameters": {"p": 0.09}, "initial": [12, 26, 26], "price": 2.27, "costs": [1.55, 1.23, 0.84, 0.55], "penalties": [0.082, 0.079, 0.051, 0.043], "holding": [0.12, 0.11, 0.1], "capacity": [30, 19, 26], "lead_times": [1, 1, 2], "discount": 0.97, "levels": [{"low": 18, "mid": 24, "high": 31}, {"low": 27, "mid": 37, "high": 47}, {"low": 44, "mid": 61, "high": 78}], "allowed": 12, "target": 98.66},
    # seed 9008; uniform (low 10, high 17), 20 periods, lead times [2, 1, 1], 20-step plan
    {"seed": 9008, "periods": 20, "demand": [11, 16, 15, 12, 12, 11, 13, 16, 15, 10, 11, 10, 16, 11, 11, 14, 13, 12, 14, 12], "distribution": "uniform", "parameters": {"low": 10, "high": 17}, "initial": [25, 31, 19], "price": 2.32, "costs": [1.29, 0.91, 0.75, 0.57], "penalties": [0.119, 0.052, 0.041, 0.02], "holding": [0.17, 0.14, 0.03], "capacity": [34, 33, 25], "lead_times": [2, 1, 1], "discount": 0.97, "levels": [{"low": 31, "mid": 43, "high": 55}, {"low": 42, "mid": 57, "high": 73}, {"low": 52, "mid": 72, "high": 91}], "allowed": 31, "target": 294.73},
    # seed 9009; poisson (mu 16), 15 periods, lead times [1, 2, 2], 15-step plan
    {"seed": 9009, "periods": 15, "demand": [12, 13, 13, 17, 10, 11, 15, 15, 14, 12, 18, 12, 18, 16, 11], "distribution": "poisson", "parameters": {"mu": 16}, "initial": [25, 17, 32], "price": 2.03, "costs": [1.34, 1.11, 0.75, 0.39], "penalties": [0.149, 0.147, 0.057, 0.025], "holding": [0.12, 0.09, 0.08], "capacity": [42, 29, 33], "lead_times": [1, 2, 2], "discount": 0.99, "levels": [{"low": 26, "mid": 35, "high": 45}, {"low": 51, "mid": 70, "high": 90}, {"low": 77, "mid": 106, "high": 134}], "allowed": 24, "target": 216.58},
    # seed 9010; binomial (n 26, p 0.52), 15 periods, lead times [2, 1, 1], 15-step plan
    {"seed": 9010, "periods": 15, "demand": [14, 13, 13, 14, 11, 11, 15, 17, 12, 15, 11, 11, 10, 11, 13], "distribution": "binomial", "parameters": {"n": 26, "p": 0.52}, "initial": [29, 29, 27], "price": 2.49, "costs": [1.36, 1.18, 0.83, 0.56], "penalties": [0.146, 0.122, 0.098, 0.027], "holding": [0.13, 0.13, 0.04], "capacity": [40, 39, 28], "lead_times": [2, 1, 1], "discount": 0.95, "levels": [{"low": 32, "mid": 45, "high": 57}, {"low": 43, "mid": 59, "high": 76}, {"low": 54, "mid": 74, "high": 95}], "allowed": 19, "target": 247.21},
    # seed 9011; uniform (low 4, high 15), 15 periods, lead times [1, 1, 3], 15-step plan
    {"seed": 9011, "periods": 15, "demand": [4, 11, 14, 6, 9, 5, 6, 10, 4, 4, 14, 7, 13, 6, 11], "distribution": "uniform", "parameters": {"low": 4, "high": 15}, "initial": [13, 22, 16], "price": 2.22, "costs": [1.21, 0.94, 0.72, 0.51], "penalties": [0.112, 0.088, 0.082, 0.023], "holding": [0.14, 0.12, 0.07], "capacity": [25, 24, 16], "lead_times": [1, 1, 3], "discount": 0.95, "levels": [{"low": 14, "mid": 20, "high": 25}, {"low": 22, "mid": 30, "high": 38}, {"low": 43, "mid": 59, "high": 76}], "allowed": 7, "target": 108.69},
    # seed 9012; uniform (low 9, high 22), 17 periods, lead times [1, 2, 1], 17-step plan
    {"seed": 9012, "periods": 17, "demand": [13, 11, 21, 14, 16, 9, 21, 10, 11, 12, 10, 16, 13, 17, 10, 11, 16], "distribution": "uniform", "parameters": {"low": 9, "high": 22}, "initial": [28, 21, 23], "price": 2.2, "costs": [1.44, 1.18, 0.92, 0.55], "penalties": [0.131, 0.048, 0.033, 0.029], "holding": [0.16, 0.09, 0.08], "capacity": [29, 28, 35], "lead_times": [1, 2, 1], "discount": 0.96, "levels": [{"low": 24, "mid": 33, "high": 42}, {"low": 48, "mid": 66, "high": 84}, {"low": 60, "mid": 82, "high": 105}], "allowed": 27, "target": 214.12},
    # seed 9013; binomial (n 40, p 0.41), 12 periods, lead times [2, 2, 1], 12-step plan
    {"seed": 9013, "periods": 12, "demand": [19, 16, 18, 14, 19, 15, 10, 17, 15, 17, 23, 24], "distribution": "binomial", "parameters": {"n": 40, "p": 0.41}, "initial": [33, 39, 32], "price": 2.77, "costs": [1.65, 1.4, 0.97, 0.54], "penalties": [0.102, 0.096, 0.085, 0.033], "holding": [0.18, 0.08, 0.03], "capacity": [25, 26, 41], "lead_times": [2, 2, 1], "discount": 0.96, "levels": [{"low": 39, "mid": 54, "high": 69}, {"low": 66, "mid": 90, "high": 115}, {"low": 79, "mid": 108, "high": 138}], "allowed": 18, "target": 311.02},
    # seed 9014; binomial (n 34, p 0.31), 15 periods, lead times [2, 1, 2], 15-step plan
    {"seed": 9014, "periods": 15, "demand": [14, 5, 12, 9, 9, 11, 13, 13, 8, 7, 10, 11, 11, 9, 12], "distribution": "binomial", "parameters": {"n": 34, "p": 0.31}, "initial": [23, 26, 16], "price": 2.25, "costs": [1.19, 0.92, 0.75, 0.41], "penalties": [0.133, 0.128, 0.075, 0.037], "holding": [0.2, 0.19, 0.12], "capacity": [17, 26, 31], "lead_times": [2, 1, 2], "discount": 0.96, "levels": [{"low": 25, "mid": 35, "high": 44}, {"low": 34, "mid": 46, "high": 59}, {"low": 51, "mid": 70, "high": 89}], "allowed": 15, "target": 145.32},
    # seed 9015; uniform (low 6, high 15), 15 periods, lead times [1, 1, 2], 15-step plan
    {"seed": 9015, "periods": 15, "demand": [12, 7, 12, 13, 7, 8, 12, 10, 12, 10, 6, 9, 12, 10, 7], "distribution": "uniform", "parameters": {"low": 6, "high": 15}, "initial": [12, 23, 21], "price": 2.12, "costs": [1.0, 0.79, 0.49, 0.32], "penalties": [0.144, 0.106, 0.102, 0.029], "holding": [0.13, 0.1, 0.06], "capacity": [20, 18, 23], "lead_times": [1, 1, 2], "discount": 0.99, "levels": [{"low": 16, "mid": 22, "high": 28}, {"low": 24, "mid": 33, "high": 42}, {"low": 40, "mid": 55, "high": 70}], "allowed": 12, "target": 212.94},
    # seed 9016; binomial (n 38, p 0.41), 16 periods, lead times [1, 1, 2], 16-step plan
    {"seed": 9016, "periods": 16, "demand": [10, 21, 12, 17, 17, 12, 14, 18, 20, 16, 15, 16, 17, 15, 21, 16], "distribution": "binomial", "parameters": {"n": 38, "p": 0.41}, "initial": [21, 34, 18], "price": 2.23, "costs": [1.48, 1.17, 0.87, 0.43], "penalties": [0.079, 0.066, 0.05, 0.024], "holding": [0.19, 0.18, 0.03], "capacity": [41, 27, 24], "lead_times": [1, 1, 2], "discount": 0.99, "levels": [{"low": 25, "mid": 34, "high": 44}, {"low": 37, "mid": 51, "high": 65}, {"low": 62, "mid": 86, "high": 109}], "allowed": 30, "target": 314.0},
    # seed 9017; poisson (mu 11), 17 periods, lead times [2, 2, 2], 17-step plan
    {"seed": 9017, "periods": 17, "demand": [6, 11, 9, 10, 8, 15, 15, 8, 12, 9, 10, 17, 5, 5, 9, 10, 10], "distribution": "poisson", "parameters": {"mu": 11}, "initial": [20, 26, 23], "price": 2.41, "costs": [1.33, 1.06, 0.8, 0.38], "penalties": [0.126, 0.105, 0.061, 0.052], "holding": [0.2, 0.04, 0.03], "capacity": [29, 25, 32], "lead_times": [2, 2, 2], "discount": 0.96, "levels": [{"low": 26, "mid": 36, "high": 46}, {"low": 44, "mid": 61, "high": 77}, {"low": 62, "mid": 85, "high": 108}], "allowed": 19, "target": 204.42},
    # seed 9018; uniform (low 11, high 25), 18 periods, lead times [1, 2, 1], 18-step plan
    {"seed": 9018, "periods": 18, "demand": [15, 15, 21, 11, 21, 21, 16, 18, 13, 13, 22, 13, 21, 23, 22, 13, 11, 11], "distribution": "uniform", "parameters": {"low": 11, "high": 25}, "initial": [31, 20, 34], "price": 2.21, "costs": [1.43, 1.08, 0.82, 0.5], "penalties": [0.123, 0.085, 0.067, 0.066], "holding": [0.17, 0.05, 0.03], "capacity": [36, 46, 40], "lead_times": [1, 2, 1], "discount": 1.0, "levels": [{"low": 28, "mid": 38, "high": 49}, {"low": 56, "mid": 77, "high": 98}, {"low": 70, "mid": 96, "high": 122}], "allowed": 32, "target": 431.53},
    # seed 9019; geometric (p 0.08), 12 periods, lead times [2, 2, 2], 12-step plan
    {"seed": 9019, "periods": 12, "demand": [4, 3, 4, 5, 4, 1, 14, 18, 12, 22, 1, 19], "distribution": "geometric", "parameters": {"p": 0.08}, "initial": [13, 27, 30], "price": 2.32, "costs": [1.41, 1.15, 0.9, 0.59], "penalties": [0.054, 0.024, 0.023, 0.021], "holding": [0.14, 0.1, 0.04], "capacity": [31, 34, 30], "lead_times": [2, 2, 2], "discount": 0.97, "levels": [{"low": 30, "mid": 41, "high": 52}, {"low": 50, "mid": 69, "high": 88}, {"low": 70, "mid": 96, "high": 122}], "allowed": 8, "target": 91.93},
    # seed 9020; poisson (mu 14), 15 periods, lead times [2, 1, 1], 15-step plan
    {"seed": 9020, "periods": 15, "demand": [13, 12, 12, 11, 10, 12, 12, 14, 11, 13, 4, 14, 17, 13, 18], "distribution": "poisson", "parameters": {"mu": 14}, "initial": [26, 14, 22], "price": 2.37, "costs": [1.42, 1.12, 0.69, 0.54], "penalties": [0.147, 0.142, 0.14, 0.101], "holding": [0.16, 0.1, 0.07], "capacity": [42, 42, 24], "lead_times": [2, 1, 1], "discount": 0.96, "levels": [{"low": 34, "mid": 46, "high": 59}, {"low": 45, "mid": 62, "high": 78}, {"low": 56, "mid": 77, "high": 98}], "allowed": 17, "target": 195.97},
    # seed 9021; binomial (n 30, p 0.33), 16 periods, lead times [1, 1, 1], 16-step plan
    {"seed": 9021, "periods": 16, "demand": [9, 10, 10, 12, 9, 11, 6, 12, 9, 8, 11, 7, 7, 9, 9, 13], "distribution": "binomial", "parameters": {"n": 30, "p": 0.33}, "initial": [11, 25, 12], "price": 2.31, "costs": [1.33, 1.07, 0.84, 0.51], "penalties": [0.102, 0.101, 0.084, 0.041], "holding": [0.08, 0.07, 0.06], "capacity": [17, 22, 27], "lead_times": [1, 1, 1], "discount": 0.95, "levels": [{"low": 16, "mid": 22, "high": 28}, {"low": 24, "mid": 33, "high": 42}, {"low": 32, "mid": 44, "high": 55}], "allowed": 9, "target": 169.37},
    # seed 9022; binomial (n 30, p 0.54), 15 periods, lead times [1, 2, 2], 15-step plan
    {"seed": 9022, "periods": 15, "demand": [17, 14, 17, 14, 19, 12, 16, 13, 15, 15, 12, 14, 19, 20, 18], "distribution": "binomial", "parameters": {"n": 30, "p": 0.54}, "initial": [34, 23, 39], "price": 2.56, "costs": [1.47, 1.13, 0.79, 0.59], "penalties": [0.146, 0.118, 0.105, 0.077], "holding": [0.1, 0.06, 0.04], "capacity": [47, 30, 40], "lead_times": [1, 2, 2], "discount": 1.0, "levels": [{"low": 26, "mid": 36, "high": 45}, {"low": 52, "mid": 71, "high": 91}, {"low": 78, "mid": 107, "high": 136}], "allowed": 21, "target": 379.42},
    # seed 9023; geometric (p 0.11), 18 periods, lead times [2, 1, 2], 18-step plan
    {"seed": 9023, "periods": 18, "demand": [4, 2, 5, 21, 2, 6, 2, 1, 1, 3, 12, 2, 4, 8, 5, 3, 1, 18], "distribution": "geometric", "parameters": {"p": 0.11}, "initial": [23, 18, 12], "price": 2.49, "costs": [1.4, 1.23, 0.85, 0.43], "penalties": [0.128, 0.116, 0.114, 0.028], "holding": [0.14, 0.06, 0.06], "capacity": [27, 17, 22], "lead_times": [2, 1, 2], "discount": 0.98, "levels": [{"low": 22, "mid": 30, "high": 38}, {"low": 29, "mid": 40, "high": 51}, {"low": 44, "mid": 60, "high": 76}], "allowed": 11, "target": 139.85},
    # seed 9024; geometric (p 0.11), 17 periods, lead times [2, 3, 2], 17-step plan
    {"seed": 9024, "periods": 17, "demand": [7, 1, 9, 1, 11, 2, 6, 1, 4, 2, 3, 34, 1, 4, 1, 26, 2], "distribution": "geometric", "parameters": {"p": 0.11}, "initial": [22, 20, 21], "price": 2.2, "costs": [1.46, 1.16, 0.87, 0.45], "penalties": [0.139, 0.127, 0.12, 0.037], "holding": [0.16, 0.15, 0.05], "capacity": [23, 18, 15], "lead_times": [2, 3, 2], "discount": 0.97, "levels": [{"low": 22, "mid": 30, "high": 38}, {"low": 44, "mid": 60, "high": 76}, {"low": 58, "mid": 80, "high": 102}], "allowed": 10, "target": 77.18},
    # seed 9025; geometric (p 0.1), 17 periods, lead times [2, 1, 1], 17-step plan
    {"seed": 9025, "periods": 17, "demand": [2, 16, 6, 17, 14, 8, 2, 5, 13, 10, 12, 9, 4, 8, 7, 1, 32], "distribution": "geometric", "parameters": {"p": 0.1}, "initial": [23, 17, 18], "price": 2.33, "costs": [1.66, 1.22, 0.87, 0.56], "penalties": [0.093, 0.077, 0.051, 0.035], "holding": [0.16, 0.13, 0.08], "capacity": [16, 29, 18], "lead_times": [2, 1, 1], "discount": 0.99, "levels": [{"low": 24, "mid": 33, "high": 42}, {"low": 32, "mid": 44, "high": 56}, {"low": 40, "mid": 55, "high": 70}], "allowed": 8, "target": 215.4},
    # seed 9026; geometric (p 0.08), 14 periods, lead times [2, 2, 1], 14-step plan
    {"seed": 9026, "periods": 14, "demand": [17, 1, 19, 17, 7, 1, 8, 13, 6, 3, 30, 10, 1, 9], "distribution": "geometric", "parameters": {"p": 0.08}, "initial": [23, 28, 23], "price": 1.82, "costs": [1.1, 0.85, 0.68, 0.36], "penalties": [0.134, 0.109, 0.104, 0.102], "holding": [0.17, 0.16, 0.14], "capacity": [35, 21, 33], "lead_times": [2, 2, 1], "discount": 0.97, "levels": [{"low": 30, "mid": 41, "high": 52}, {"low": 50, "mid": 69, "high": 88}, {"low": 60, "mid": 83, "high": 105}], "allowed": 14, "target": 116.1},
    # seed 9027; geometric (p 0.12), 16 periods, lead times [1, 3, 2], 16-step plan
    {"seed": 9027, "periods": 16, "demand": [1, 1, 3, 4, 1, 8, 8, 5, 6, 11, 2, 4, 4, 6, 2, 2], "distribution": "geometric", "parameters": {"p": 0.12}, "initial": [15, 11, 14], "price": 2.1, "costs": [1.36, 1.09, 0.81, 0.47], "penalties": [0.138, 0.127, 0.082, 0.079], "holding": [0.17, 0.16, 0.09], "capacity": [22, 22, 13], "lead_times": [1, 3, 2], "discount": 0.96, "levels": [{"low": 13, "mid": 18, "high": 23}, {"low": 33, "mid": 46, "high": 58}, {"low": 47, "mid": 64, "high": 82}], "allowed": 5, "target": 25.88},
    # seed 9028; geometric (p 0.07), 12 periods, lead times [2, 2, 3], 12-step plan
    {"seed": 9028, "periods": 12, "demand": [6, 1, 8, 15, 3, 17, 3, 1, 3, 23, 1, 4], "distribution": "geometric", "parameters": {"p": 0.07}, "initial": [29, 20, 30], "price": 2.27, "costs": [1.19, 0.98, 0.63, 0.31], "penalties": [0.113, 0.093, 0.093, 0.089], "holding": [0.16, 0.11, 0.11], "capacity": [23, 42, 31], "lead_times": [2, 2, 3], "discount": 0.98, "levels": [{"low": 34, "mid": 47, "high": 60}, {"low": 57, "mid": 79, "high": 100}, {"low": 91, "mid": 126, "high": 160}], "allowed": 7, "target": 97.31},
    # seed 9029; uniform (low 10, high 23), 18 periods, lead times [2, 2, 2], 18-step plan
    {"seed": 9029, "periods": 18, "demand": [11, 15, 12, 21, 14, 13, 16, 17, 12, 20, 22, 19, 17, 21, 22, 15, 10, 16], "distribution": "uniform", "parameters": {"low": 10, "high": 23}, "initial": [30, 34, 38], "price": 1.8, "costs": [1.26, 1.08, 0.64, 0.4], "penalties": [0.098, 0.082, 0.045, 0.041], "holding": [0.14, 0.11, 0.07], "capacity": [39, 46, 29], "lead_times": [2, 2, 2], "discount": 0.99, "levels": [{"low": 38, "mid": 53, "high": 67}, {"low": 64, "mid": 88, "high": 112}, {"low": 90, "mid": 123, "high": 157}], "allowed": 26, "target": 253.64},
    # seed 9030; poisson (mu 10), 19 periods, lead times [1, 1, 2], 19-step plan
    {"seed": 9030, "periods": 19, "demand": [15, 4, 5, 11, 12, 7, 14, 16, 11, 5, 5, 5, 4, 8, 10, 14, 9, 12, 8], "distribution": "poisson", "parameters": {"mu": 10}, "initial": [15, 20, 16], "price": 2.09, "costs": [1.15, 0.99, 0.67, 0.34], "penalties": [0.143, 0.137, 0.101, 0.083], "holding": [0.18, 0.18, 0.09], "capacity": [26, 24, 20], "lead_times": [1, 1, 2], "discount": 0.98, "levels": [{"low": 16, "mid": 22, "high": 28}, {"low": 24, "mid": 33, "high": 42}, {"low": 40, "mid": 55, "high": 70}], "allowed": 21, "target": 184.53},
    # seed 9031; binomial (n 27, p 0.57), 12 periods, lead times [2, 1, 1], 12-step plan
    {"seed": 9031, "periods": 12, "demand": [8, 11, 16, 12, 19, 12, 17, 18, 12, 14, 17, 18], "distribution": "binomial", "parameters": {"n": 27, "p": 0.57}, "initial": [25, 26, 24], "price": 2.32, "costs": [1.42, 1.15, 0.76, 0.49], "penalties": [0.066, 0.063, 0.058, 0.024], "holding": [0.17, 0.14, 0.05], "capacity": [25, 39, 41], "lead_times": [2, 1, 1], "discount": 0.99, "levels": [{"low": 37, "mid": 51, "high": 65}, {"low": 49, "mid": 68, "high": 86}, {"low": 62, "mid": 85, "high": 108}], "allowed": 11, "target": 254.97},
    # seed 9032; geometric (p 0.08), 14 periods, lead times [1, 2, 2], 14-step plan
    {"seed": 9032, "periods": 14, "demand": [21, 2, 8, 16, 10, 12, 5, 2, 7, 8, 13, 6, 7, 6], "distribution": "geometric", "parameters": {"p": 0.08}, "initial": [17, 30, 26], "price": 2.49, "costs": [1.54, 1.3, 0.86, 0.53], "penalties": [0.128, 0.124, 0.056, 0.026], "holding": [0.2, 0.15, 0.14], "capacity": [36, 31, 23], "lead_times": [1, 2, 2], "discount": 0.98, "levels": [{"low": 20, "mid": 28, "high": 35}, {"low": 40, "mid": 55, "high": 70}, {"low": 60, "mid": 83, "high": 105}], "allowed": 6, "target": 153.11},
    # seed 9033; geometric (p 0.11), 16 periods, lead times [1, 2, 2], 16-step plan
    {"seed": 9033, "periods": 16, "demand": [1, 2, 5, 1, 16, 5, 6, 12, 3, 3, 1, 3, 8, 13, 2, 3], "distribution": "geometric", "parameters": {"p": 0.11}, "initial": [9, 15, 20], "price": 1.97, "costs": [1.07, 0.88, 0.48, 0.32], "penalties": [0.114, 0.063, 0.039, 0.02], "holding": [0.16, 0.15, 0.08], "capacity": [20, 24, 21], "lead_times": [1, 2, 2], "discount": 0.98, "levels": [{"low": 15, "mid": 20, "high": 25}, {"low": 29, "mid": 40, "high": 51}, {"low": 44, "mid": 60, "high": 76}], "allowed": 5, "target": 69.37},
    # seed 9034; uniform (low 10, high 21), 17 periods, lead times [1, 1, 2], 17-step plan
    {"seed": 9034, "periods": 17, "demand": [14, 20, 14, 10, 10, 15, 14, 12, 18, 16, 20, 16, 14, 20, 12, 11, 14], "distribution": "uniform", "parameters": {"low": 10, "high": 21}, "initial": [33, 24, 36], "price": 2.61, "costs": [1.43, 0.99, 0.55, 0.39], "penalties": [0.142, 0.141, 0.12, 0.021], "holding": [0.18, 0.16, 0.11], "capacity": [42, 32, 40], "lead_times": [1, 1, 2], "discount": 0.95, "levels": [{"low": 24, "mid": 33, "high": 42}, {"low": 36, "mid": 50, "high": 63}, {"low": 60, "mid": 82, "high": 105}], "allowed": 26, "target": 308.89},
    # seed 9035; binomial (n 39, p 0.4), 16 periods, lead times [1, 1, 2], 16-step plan
    {"seed": 9035, "periods": 16, "demand": [16, 14, 17, 13, 13, 16, 16, 14, 15, 11, 19, 21, 19, 10, 14, 15], "distribution": "binomial", "parameters": {"n": 39, "p": 0.4}, "initial": [22, 26, 37], "price": 1.91, "costs": [1.4, 1.13, 0.96, 0.58], "penalties": [0.124, 0.102, 0.086, 0.054], "holding": [0.16, 0.07, 0.03], "capacity": [46, 46, 34], "lead_times": [1, 1, 2], "discount": 0.99, "levels": [{"low": 25, "mid": 34, "high": 44}, {"low": 37, "mid": 51, "high": 66}, {"low": 62, "mid": 86, "high": 109}], "allowed": 29, "target": 263.42},
    # seed 9036; uniform (low 7, high 20), 16 periods, lead times [2, 1, 2], 16-step plan
    {"seed": 9036, "periods": 16, "demand": [11, 12, 16, 19, 11, 16, 19, 14, 10, 9, 15, 14, 7, 9, 16, 8], "distribution": "uniform", "parameters": {"low": 7, "high": 20}, "initial": [32, 29, 29], "price": 2.32, "costs": [1.65, 1.27, 0.9, 0.56], "penalties": [0.137, 0.135, 0.095, 0.034], "holding": [0.18, 0.14, 0.08], "capacity": [34, 34, 29], "lead_times": [2, 1, 2], "discount": 0.99, "levels": [{"low": 31, "mid": 43, "high": 55}, {"low": 42, "mid": 57, "high": 73}, {"low": 62, "mid": 86, "high": 109}], "allowed": 15, "target": 251.06},
    # seed 9037; binomial (n 32, p 0.44), 20 periods, lead times [1, 1, 3], 20-step plan
    {"seed": 9037, "periods": 20, "demand": [15, 14, 15, 13, 13, 10, 13, 14, 14, 15, 16, 12, 17, 16, 16, 14, 11, 18, 11, 15], "distribution": "binomial", "parameters": {"n": 32, "p": 0.44}, "initial": [35, 29, 25], "price": 2.25, "costs": [1.18, 0.98, 0.81, 0.39], "penalties": [0.124, 0.124, 0.085, 0.079], "holding": [0.15, 0.13, 0.07], "capacity": [32, 26, 24], "lead_times": [1, 1, 3], "discount": 0.99, "levels": [{"low": 23, "mid": 31, "high": 39}, {"low": 34, "mid": 46, "high": 59}, {"low": 68, "mid": 93, "high": 118}], "allowed": 29, "target": 388.39},
    # seed 9038; poisson (mu 18), 20 periods, lead times [2, 1, 2], 20-step plan
    {"seed": 9038, "periods": 20, "demand": [25, 18, 18, 22, 9, 17, 18, 17, 13, 17, 22, 20, 17, 26, 15, 26, 17, 22, 16, 21], "distribution": "poisson", "parameters": {"mu": 18}, "initial": [27, 39, 36], "price": 2.33, "costs": [1.22, 1.06, 0.79, 0.58], "penalties": [0.124, 0.101, 0.085, 0.064], "holding": [0.09, 0.04, 0.03], "capacity": [46, 48, 54], "lead_times": [2, 1, 2], "discount": 0.98, "levels": [{"low": 43, "mid": 59, "high": 76}, {"low": 58, "mid": 79, "high": 101}, {"low": 86, "mid": 119, "high": 151}], "allowed": 43, "target": 443.14},
    # seed 9039; poisson (mu 19), 12 periods, lead times [1, 2, 2], 12-step plan
    {"seed": 9039, "periods": 12, "demand": [20, 13, 13, 14, 17, 7, 25, 22, 21, 24, 11, 19], "distribution": "poisson", "parameters": {"mu": 19}, "initial": [43, 21, 33], "price": 2.17, "costs": [1.65, 1.22, 0.89, 0.5], "penalties": [0.143, 0.049, 0.035, 0.022], "holding": [0.14, 0.09, 0.03], "capacity": [47, 55, 48], "lead_times": [1, 2, 2], "discount": 0.98, "levels": [{"low": 30, "mid": 42, "high": 53}, {"low": 61, "mid": 84, "high": 106}, {"low": 91, "mid": 125, "high": 160}], "allowed": 21, "target": 235.48},
    # seed 9040; binomial (n 28, p 0.41), 18 periods, lead times [1, 1, 1], 18-step plan
    {"seed": 9040, "periods": 18, "demand": [13, 6, 11, 12, 8, 12, 8, 16, 10, 13, 9, 12, 9, 7, 12, 13, 11, 8], "distribution": "binomial", "parameters": {"n": 28, "p": 0.41}, "initial": [23, 14, 23], "price": 2.14, "costs": [1.45, 1.03, 0.73, 0.41], "penalties": [0.1, 0.084, 0.05, 0.036], "holding": [0.19, 0.18, 0.1], "capacity": [20, 18, 31], "lead_times": [1, 1, 1], "discount": 0.95, "levels": [{"low": 18, "mid": 25, "high": 32}, {"low": 28, "mid": 38, "high": 48}, {"low": 37, "mid": 51, "high": 64}], "allowed": 14, "target": 176.06},
    # seed 9041; uniform (low 9, high 16), 15 periods, lead times [1, 1, 1], 15-step plan
    {"seed": 9041, "periods": 15, "demand": [10, 9, 10, 12, 12, 13, 11, 12, 14, 13, 15, 12, 10, 9, 12], "distribution": "uniform", "parameters": {"low": 9, "high": 16}, "initial": [29, 22, 26], "price": 2.12, "costs": [1.21, 0.91, 0.61, 0.41], "penalties": [0.084, 0.076, 0.059, 0.034], "holding": [0.11, 0.11, 0.1], "capacity": [34, 27, 32], "lead_times": [1, 1, 1], "discount": 0.99, "levels": [{"low": 19, "mid": 26, "high": 34}, {"low": 29, "mid": 40, "high": 50}, {"low": 38, "mid": 53, "high": 67}], "allowed": 18, "target": 233.19},
    # seed 9042; uniform (low 7, high 17), 15 periods, lead times [1, 1, 3], 15-step plan
    {"seed": 9042, "periods": 15, "demand": [13, 8, 14, 10, 7, 12, 10, 11, 8, 10, 10, 9, 13, 16, 7], "distribution": "uniform", "parameters": {"low": 7, "high": 17}, "initial": [24, 12, 26], "price": 2.19, "costs": [1.33, 0.91, 0.5, 0.31], "penalties": [0.148, 0.098, 0.084, 0.065], "holding": [0.2, 0.15, 0.06], "capacity": [33, 33, 34], "lead_times": [1, 1, 3], "discount": 1.0, "levels": [{"low": 18, "mid": 25, "high": 32}, {"low": 28, "mid": 38, "high": 48}, {"low": 55, "mid": 76, "high": 97}], "allowed": 15, "target": 202.22},
    # seed 9043; binomial (n 22, p 0.38), 16 periods, lead times [1, 1, 1], 16-step plan
    {"seed": 9043, "periods": 16, "demand": [8, 9, 9, 9, 8, 4, 6, 5, 10, 6, 8, 12, 7, 8, 7, 9], "distribution": "binomial", "parameters": {"n": 22, "p": 0.38}, "initial": [15, 16, 19], "price": 2.31, "costs": [1.14, 0.92, 0.74, 0.54], "penalties": [0.131, 0.088, 0.047, 0.027], "holding": [0.2, 0.1, 0.07], "capacity": [23, 22, 13], "lead_times": [1, 1, 1], "discount": 0.99, "levels": [{"low": 13, "mid": 18, "high": 23}, {"low": 20, "mid": 28, "high": 35}, {"low": 27, "mid": 37, "high": 47}], "allowed": 7, "target": 181.21},
    # seed 9044; uniform (low 5, high 12), 19 periods, lead times [1, 2, 1], 19-step plan
    {"seed": 9044, "periods": 19, "demand": [5, 5, 7, 8, 9, 7, 8, 8, 6, 6, 5, 9, 5, 8, 11, 5, 9, 10, 11], "distribution": "uniform", "parameters": {"low": 5, "high": 12}, "initial": [12, 16, 12], "price": 2.79, "costs": [1.67, 1.24, 0.86, 0.5], "penalties": [0.146, 0.097, 0.067, 0.064], "holding": [0.2, 0.17, 0.16], "capacity": [21, 14, 19], "lead_times": [1, 2, 1], "discount": 0.96, "levels": [{"low": 13, "mid": 18, "high": 22}, {"low": 26, "mid": 35, "high": 45}, {"low": 32, "mid": 44, "high": 56}], "allowed": 13, "target": 168.03},
    # seed 9045; binomial (n 21, p 0.32), 19 periods, lead times [3, 1, 1], 19-step plan
    {"seed": 9045, "periods": 19, "demand": [4, 3, 8, 9, 6, 7, 5, 4, 5, 5, 7, 6, 8, 7, 5, 4, 10, 4, 9], "distribution": "binomial", "parameters": {"n": 21, "p": 0.32}, "initial": [13, 16, 7], "price": 1.82, "costs": [1.18, 0.76, 0.55, 0.38], "penalties": [0.116, 0.101, 0.077, 0.025], "holding": [0.19, 0.18, 0.12], "capacity": [19, 15, 10], "lead_times": [3, 1, 1], "discount": 0.99, "levels": [{"low": 22, "mid": 30, "high": 38}, {"low": 27, "mid": 37, "high": 47}, {"low": 32, "mid": 44, "high": 56}], "allowed": 7, "target": 88.13},
    # seed 9046; binomial (n 33, p 0.4), 19 periods, lead times [2, 1, 1], 19-step plan
    {"seed": 9046, "periods": 19, "demand": [12, 13, 14, 10, 12, 19, 9, 10, 12, 10, 5, 10, 12, 16, 14, 11, 18, 12, 9], "distribution": "binomial", "parameters": {"n": 33, "p": 0.4}, "initial": [31, 31, 30], "price": 2.66, "costs": [1.5, 1.08, 0.7, 0.53], "penalties": [0.092, 0.076, 0.059, 0.047], "holding": [0.16, 0.07, 0.03], "capacity": [28, 34, 22], "lead_times": [2, 1, 1], "discount": 0.99, "levels": [{"low": 32, "mid": 44, "high": 55}, {"low": 42, "mid": 58, "high": 74}, {"low": 53, "mid": 73, "high": 92}], "allowed": 21, "target": 416.11},
    # seed 9047; geometric (p 0.11), 18 periods, lead times [2, 3, 1], 18-step plan
    {"seed": 9047, "periods": 18, "demand": [4, 3, 3, 1, 18, 10, 4, 5, 7, 3, 30, 4, 5, 6, 10, 18, 3, 17], "distribution": "geometric", "parameters": {"p": 0.11}, "initial": [17, 18, 11], "price": 2.01, "costs": [1.29, 0.92, 0.64, 0.31], "penalties": [0.139, 0.093, 0.066, 0.058], "holding": [0.15, 0.06, 0.06], "capacity": [15, 24, 23], "lead_times": [2, 3, 1], "discount": 1.0, "levels": [{"low": 22, "mid": 30, "high": 38}, {"low": 44, "mid": 60, "high": 76}, {"low": 51, "mid": 70, "high": 89}], "allowed": 15, "target": 186.96},
    # seed 9048; uniform (low 12, high 22), 15 periods, lead times [1, 1, 3], 15-step plan
    {"seed": 9048, "periods": 15, "demand": [14, 12, 12, 20, 18, 16, 18, 17, 20, 15, 18, 13, 14, 18, 18], "distribution": "uniform", "parameters": {"low": 12, "high": 22}, "initial": [28, 41, 21], "price": 2.35, "costs": [1.31, 0.99, 0.68, 0.51], "penalties": [0.115, 0.105, 0.099, 0.039], "holding": [0.14, 0.11, 0.08], "capacity": [31, 35, 25], "lead_times": [1, 1, 3], "discount": 1.0, "levels": [{"low": 26, "mid": 36, "high": 46}, {"low": 40, "mid": 54, "high": 69}, {"low": 79, "mid": 109, "high": 139}], "allowed": 24, "target": 321.37},
    # seed 9049; binomial (n 34, p 0.36), 17 periods, lead times [1, 1, 3], 17-step plan
    {"seed": 9049, "periods": 17, "demand": [11, 8, 17, 12, 9, 10, 9, 12, 14, 12, 9, 12, 9, 14, 12, 14, 8], "distribution": "binomial", "parameters": {"n": 34, "p": 0.36}, "initial": [23, 22, 27], "price": 2.36, "costs": [1.3, 0.87, 0.55, 0.32], "penalties": [0.141, 0.123, 0.116, 0.101], "holding": [0.19, 0.1, 0.03], "capacity": [29, 35, 28], "lead_times": [1, 1, 3], "discount": 0.97, "levels": [{"low": 20, "mid": 27, "high": 34}, {"low": 29, "mid": 40, "high": 51}, {"low": 59, "mid": 81, "high": 103}], "allowed": 19, "target": 261.52},
    # seed 9050; poisson (mu 19), 17 periods, lead times [1, 1, 2], 17-step plan
    {"seed": 9050, "periods": 17, "demand": [17, 16, 15, 16, 15, 21, 15, 12, 13, 16, 27, 15, 14, 11, 20, 15, 26], "distribution": "poisson", "parameters": {"mu": 19}, "initial": [47, 22, 40], "price": 2.02, "costs": [1.19, 0.94, 0.49, 0.32], "penalties": [0.103, 0.09, 0.077, 0.048], "holding": [0.19, 0.19, 0.13], "capacity": [53, 42, 46], "lead_times": [1, 1, 2], "discount": 0.95, "levels": [{"low": 30, "mid": 42, "high": 53}, {"low": 46, "mid": 63, "high": 80}, {"low": 76, "mid": 105, "high": 133}], "allowed": 17, "target": 221.33},
    # seed 9051; uniform (low 7, high 16), 15 periods, lead times [1, 1, 3], 15-step plan
    {"seed": 9051, "periods": 15, "demand": [9, 11, 11, 12, 12, 15, 9, 10, 14, 9, 12, 14, 11, 15, 13], "distribution": "uniform", "parameters": {"low": 7, "high": 16}, "initial": [24, 27, 21], "price": 2.06, "costs": [1.17, 0.91, 0.69, 0.36], "penalties": [0.136, 0.096, 0.094, 0.036], "holding": [0.16, 0.04, 0.03], "capacity": [25, 32, 28], "lead_times": [1, 1, 3], "discount": 0.98, "levels": [{"low": 18, "mid": 24, "high": 31}, {"low": 26, "mid": 36, "high": 46}, {"low": 53, "mid": 73, "high": 92}], "allowed": 16, "target": 218.54},
    # seed 9052; binomial (n 25, p 0.35), 12 periods, lead times [2, 2, 2], 12-step plan
    {"seed": 9052, "periods": 12, "demand": [7, 6, 9, 11, 5, 9, 12, 6, 6, 10, 10, 11], "distribution": "binomial", "parameters": {"n": 25, "p": 0.35}, "initial": [21, 17, 19], "price": 2.55, "costs": [1.46, 1.04, 0.79, 0.47], "penalties": [0.117, 0.114, 0.079, 0.038], "holding": [0.18, 0.15, 0.09], "capacity": [24, 21, 22], "lead_times": [2, 2, 2], "discount": 0.98, "levels": [{"low": 21, "mid": 29, "high": 37}, {"low": 35, "mid": 48, "high": 61}, {"low": 49, "mid": 67, "high": 86}], "allowed": 8, "target": 128.67},
    # seed 9053; uniform (low 10, high 18), 19 periods, lead times [2, 2, 1], 19-step plan
    {"seed": 9053, "periods": 19, "demand": [10, 10, 11, 14, 10, 14, 15, 14, 16, 15, 13, 17, 15, 13, 14, 15, 15, 13, 15], "distribution": "uniform", "parameters": {"low": 10, "high": 18}, "initial": [19, 24, 28], "price": 1.81, "costs": [1.15, 0.95, 0.61, 0.44], "penalties": [0.042, 0.037, 0.027, 0.024], "holding": [0.19, 0.14, 0.13], "capacity": [31, 39, 39], "lead_times": [2, 2, 1], "discount": 0.99, "levels": [{"low": 32, "mid": 45, "high": 57}, {"low": 54, "mid": 74, "high": 94}, {"low": 65, "mid": 89, "high": 113}], "allowed": 26, "target": 211.38},
    # seed 9054; geometric (p 0.09), 13 periods, lead times [1, 2, 3], 13-step plan
    {"seed": 9054, "periods": 13, "demand": [1, 5, 3, 5, 2, 11, 14, 10, 5, 1, 2, 3, 14], "distribution": "geometric", "parameters": {"p": 0.09}, "initial": [17, 16, 25], "price": 2.48, "costs": [1.52, 1.11, 0.94, 0.57], "penalties": [0.118, 0.094, 0.052, 0.041], "holding": [0.17, 0.07, 0.07], "capacity": [26, 22, 21], "lead_times": [1, 2, 3], "discount": 0.99, "levels": [{"low": 18, "mid": 24, "high": 31}, {"low": 36, "mid": 49, "high": 62}, {"low": 62, "mid": 86, "high": 109}], "allowed": 5, "target": 79.56},
    # seed 9055; binomial (n 23, p 0.56), 18 periods, lead times [2, 1, 1], 18-step plan
    {"seed": 9055, "periods": 18, "demand": [13, 16, 15, 10, 15, 8, 15, 16, 13, 12, 12, 11, 12, 13, 14, 14, 12, 13], "distribution": "binomial", "parameters": {"n": 23, "p": 0.56}, "initial": [29, 18, 27], "price": 1.92, "costs": [1.31, 1.14, 0.71, 0.42], "penalties": [0.145, 0.136, 0.124, 0.047], "holding": [0.13, 0.1, 0.05], "capacity": [34, 25, 21], "lead_times": [2, 1, 1], "discount": 0.98, "levels": [{"low": 31, "mid": 43, "high": 54}, {"low": 41, "mid": 57, "high": 72}, {"low": 52, "mid": 71, "high": 90}], "allowed": 10, "target": 256.34},
    # seed 9056; poisson (mu 9), 18 periods, lead times [2, 1, 1], 18-step plan
    {"seed": 9056, "periods": 18, "demand": [11, 4, 4, 5, 8, 7, 10, 7, 13, 9, 13, 5, 9, 12, 4, 13, 8, 8], "distribution": "poisson", "parameters": {"mu": 9}, "initial": [22, 19, 15], "price": 2.51, "costs": [1.32, 1.0, 0.71, 0.53], "penalties": [0.125, 0.083, 0.059, 0.051], "holding": [0.18, 0.12, 0.11], "capacity": [22, 16, 26], "lead_times": [2, 1, 1], "discount": 0.96, "levels": [{"low": 22, "mid": 30, "high": 38}, {"low": 29, "mid": 40, "high": 50}, {"low": 36, "mid": 50, "high": 63}], "allowed": 11, "target": 166.06},
    # seed 9057; poisson (mu 8), 19 periods, lead times [1, 1, 1], 19-step plan
    {"seed": 9057, "periods": 19, "demand": [14, 8, 7, 3, 4, 7, 7, 10, 9, 5, 9, 6, 9, 8, 9, 6, 4, 11, 9], "distribution": "poisson", "parameters": {"mu": 8}, "initial": [13, 19, 13], "price": 2.58, "costs": [1.39, 1.0, 0.62, 0.43], "penalties": [0.135, 0.088, 0.086, 0.077], "holding": [0.2, 0.2, 0.1], "capacity": [23, 23, 15], "lead_times": [1, 1, 1], "discount": 0.99, "levels": [{"low": 13, "mid": 18, "high": 22}, {"low": 19, "mid": 26, "high": 34}, {"low": 26, "mid": 35, "high": 45}], "allowed": 14, "target": 230.03},
    # seed 9058; geometric (p 0.08), 16 periods, lead times [3, 1, 1], 16-step plan
    {"seed": 9058, "periods": 16, "demand": [5, 12, 10, 5, 4, 2, 5, 4, 2, 20, 4, 2, 7, 1, 10, 3], "distribution": "geometric", "parameters": {"p": 0.08}, "initial": [17, 27, 15], "price": 1.77, "costs": [1.23, 0.85, 0.66, 0.38], "penalties": [0.147, 0.089, 0.025, 0.024], "holding": [0.16, 0.14, 0.06], "capacity": [34, 23, 30], "lead_times": [3, 1, 1], "discount": 0.97, "levels": [{"low": 40, "mid": 55, "high": 70}, {"low": 50, "mid": 69, "high": 88}, {"low": 60, "mid": 83, "high": 105}], "allowed": 11, "target": 52.54},
    # seed 9059; poisson (mu 11), 15 periods, lead times [2, 2, 1], 15-step plan
    {"seed": 9059, "periods": 15, "demand": [9, 11, 11, 10, 11, 14, 12, 11, 8, 10, 12, 12, 9, 13, 7], "distribution": "poisson", "parameters": {"mu": 11}, "initial": [17, 27, 24], "price": 1.84, "costs": [1.29, 0.94, 0.69, 0.33], "penalties": [0.145, 0.091, 0.051, 0.044], "holding": [0.19, 0.18, 0.08], "capacity": [28, 30, 27], "lead_times": [2, 2, 1], "discount": 0.99, "levels": [{"low": 26, "mid": 36, "high": 46}, {"low": 44, "mid": 61, "high": 77}, {"low": 53, "mid": 73, "high": 92}], "allowed": 17, "target": 152.65},
    # seed 9060; geometric (p 0.08), 19 periods, lead times [1, 2, 3], 19-step plan
    {"seed": 9060, "periods": 19, "demand": [11, 19, 11, 17, 8, 7, 1, 3, 9, 1, 5, 8, 4, 3, 8, 9, 15, 2, 4], "distribution": "geometric", "parameters": {"p": 0.08}, "initial": [22, 28, 25], "price": 2.13, "costs": [1.5, 1.13, 0.77, 0.34], "penalties": [0.131, 0.1, 0.096, 0.048], "holding": [0.18, 0.11, 0.03], "capacity": [24, 36, 36], "lead_times": [1, 2, 3], "discount": 0.98, "levels": [{"low": 20, "mid": 28, "high": 35}, {"low": 40, "mid": 55, "high": 70}, {"low": 70, "mid": 96, "high": 122}], "allowed": 9, "target": 115.88},
    # seed 9061; uniform (low 7, high 16), 15 periods, lead times [3, 1, 1], 15-step plan
    {"seed": 9061, "periods": 15, "demand": [14, 9, 15, 14, 10, 8, 10, 11, 12, 11, 11, 11, 15, 11, 10], "distribution": "uniform", "parameters": {"low": 7, "high": 16}, "initial": [19, 25, 24], "price": 2.51, "costs": [1.47, 1.25, 0.84, 0.59], "penalties": [0.118, 0.105, 0.104, 0.091], "holding": [0.2, 0.14, 0.09], "capacity": [19, 26, 19], "lead_times": [3, 1, 1], "discount": 0.96, "levels": [{"low": 35, "mid": 48, "high": 62}, {"low": 44, "mid": 61, "high": 77}, {"low": 53, "mid": 73, "high": 92}], "allowed": 19, "target": 179.5},
    # seed 9062; uniform (low 12, high 26), 19 periods, lead times [1, 3, 3], 19-step plan
    {"seed": 9062, "periods": 19, "demand": [15, 22, 15, 19, 13, 13, 24, 20, 15, 23, 22, 18, 12, 19, 22, 25, 12, 23, 20], "distribution": "uniform", "parameters": {"low": 12, "high": 26}, "initial": [46, 43, 42], "price": 2.17, "costs": [1.34, 0.92, 0.75, 0.58], "penalties": [0.135, 0.129, 0.12, 0.071], "holding": [0.18, 0.08, 0.05], "capacity": [49, 50, 30], "lead_times": [1, 3, 3], "discount": 0.97, "levels": [{"low": 30, "mid": 41, "high": 52}, {"low": 74, "mid": 102, "high": 130}, {"low": 118, "mid": 163, "high": 207}], "allowed": 29, "target": 290.08},
    # seed 9063; poisson (mu 8), 15 periods, lead times [2, 1, 1], 15-step plan
    {"seed": 9063, "periods": 15, "demand": [5, 12, 7, 6, 9, 5, 5, 8, 5, 9, 4, 6, 6, 11, 16], "distribution": "poisson", "parameters": {"mu": 8}, "initial": [19, 12, 9], "price": 2.52, "costs": [1.42, 1.1, 0.78, 0.35], "penalties": [0.146, 0.098, 0.044, 0.027], "holding": [0.12, 0.09, 0.06], "capacity": [19, 23, 15], "lead_times": [2, 1, 1], "discount": 1.0, "levels": [{"low": 19, "mid": 26, "high": 34}, {"low": 26, "mid": 35, "high": 45}, {"low": 32, "mid": 44, "high": 56}], "allowed": 11, "target": 210.86},
    # seed 9064; poisson (mu 10), 13 periods, lead times [1, 1, 2], 13-step plan
    {"seed": 9064, "periods": 13, "demand": [6, 6, 10, 7, 12, 12, 13, 6, 7, 10, 15, 6, 8], "distribution": "poisson", "parameters": {"mu": 10}, "initial": [19, 21, 18], "price": 1.8, "costs": [1.16, 1.0, 0.78, 0.59], "penalties": [0.136, 0.123, 0.081, 0.034], "holding": [0.19, 0.09, 0.05], "capacity": [21, 17, 18], "lead_times": [1, 1, 2], "discount": 0.97, "levels": [{"low": 16, "mid": 22, "high": 28}, {"low": 24, "mid": 33, "high": 42}, {"low": 40, "mid": 55, "high": 70}], "allowed": 5, "target": 88.97},
    # seed 9065; geometric (p 0.09), 15 periods, lead times [3, 3, 2], 15-step plan
    {"seed": 9065, "periods": 15, "demand": [3, 18, 7, 6, 1, 11, 1, 9, 1, 20, 23, 6, 12, 17, 14], "distribution": "geometric", "parameters": {"p": 0.09}, "initial": [20, 21, 25], "price": 2.28, "costs": [1.33, 1.0, 0.67, 0.44], "penalties": [0.144, 0.128, 0.126, 0.073], "holding": [0.14, 0.13, 0.09], "capacity": [29, 28, 23], "lead_times": [3, 3, 2], "discount": 0.95, "levels": [{"low": 36, "mid": 49, "high": 62}, {"low": 62, "mid": 86, "high": 109}, {"low": 80, "mid": 110, "high": 140}], "allowed": 14, "target": 101.01},
    # seed 9066; binomial (n 36, p 0.47), 20 periods, lead times [1, 1, 3], 20-step plan
    {"seed": 9066, "periods": 20, "demand": [19, 19, 22, 17, 13, 13, 19, 13, 18, 18, 13, 16, 17, 19, 11, 16, 16, 22, 17, 22], "distribution": "binomial", "parameters": {"n": 36, "p": 0.47}, "initial": [41, 33, 41], "price": 2.06, "costs": [1.31, 1.06, 0.78, 0.5], "penalties": [0.095, 0.06, 0.049, 0.042], "holding": [0.18, 0.13, 0.13], "capacity": [46, 41, 48], "lead_times": [1, 1, 3], "discount": 0.97, "levels": [{"low": 27, "mid": 37, "high": 47}, {"low": 41, "mid": 56, "high": 71}, {"low": 81, "mid": 112, "high": 142}], "allowed": 31, "target": 284.64},
    # seed 9067; binomial (n 37, p 0.31), 15 periods, lead times [1, 1, 1], 15-step plan
    {"seed": 9067, "periods": 15, "demand": [9, 7, 12, 10, 10, 9, 9, 8, 11, 9, 13, 10, 11, 11, 12], "distribution": "binomial", "parameters": {"n": 37, "p": 0.31}, "initial": [14, 13, 19], "price": 2.2, "costs": [1.21, 1.0, 0.67, 0.48], "penalties": [0.133, 0.118, 0.088, 0.063], "holding": [0.17, 0.11, 0.07], "capacity": [28, 19, 25], "lead_times": [1, 1, 1], "discount": 0.96, "levels": [{"low": 18, "mid": 25, "high": 32}, {"low": 28, "mid": 38, "high": 48}, {"low": 37, "mid": 50, "high": 64}], "allowed": 18, "target": 161.82},
    # seed 9068; poisson (mu 20), 19 periods, lead times [1, 1, 2], 19-step plan
    {"seed": 9068, "periods": 19, "demand": [14, 12, 24, 22, 21, 18, 18, 18, 26, 19, 22, 19, 16, 23, 19, 22, 19, 18, 15], "distribution": "poisson", "parameters": {"mu": 20}, "initial": [49, 50, 36], "price": 2.12, "costs": [1.31, 1.13, 0.9, 0.47], "penalties": [0.148, 0.134, 0.051, 0.031], "holding": [0.15, 0.13, 0.03], "capacity": [41, 44, 30], "lead_times": [1, 1, 2], "discount": 0.97, "levels": [{"low": 32, "mid": 44, "high": 56}, {"low": 48, "mid": 66, "high": 84}, {"low": 80, "mid": 110, "high": 140}], "allowed": 32, "target": 398.21},
    # seed 9069; poisson (mu 11), 14 periods, lead times [1, 1, 3], 14-step plan
    {"seed": 9069, "periods": 14, "demand": [12, 5, 10, 5, 12, 10, 8, 16, 6, 13, 13, 7, 8, 11], "distribution": "poisson", "parameters": {"mu": 11}, "initial": [23, 26, 19], "price": 1.83, "costs": [1.18, 0.92, 0.58, 0.32], "penalties": [0.142, 0.074, 0.022, 0.022], "holding": [0.19, 0.14, 0.1], "capacity": [32, 22, 25], "lead_times": [1, 1, 3], "discount": 0.99, "levels": [{"low": 18, "mid": 24, "high": 31}, {"low": 26, "mid": 36, "high": 46}, {"low": 53, "mid": 73, "high": 92}], "allowed": 13, "target": 129.62},
    # seed 9070; poisson (mu 20), 18 periods, lead times [3, 1, 1], 18-step plan
    {"seed": 9070, "periods": 18, "demand": [13, 17, 26, 25, 24, 20, 15, 25, 28, 21, 18, 26, 23, 15, 16, 19, 17, 20], "distribution": "poisson", "parameters": {"mu": 20}, "initial": [34, 43, 50], "price": 2.55, "costs": [1.64, 1.34, 1.04, 0.59], "penalties": [0.135, 0.11, 0.079, 0.05], "holding": [0.11, 0.07, 0.05], "capacity": [32, 53, 48], "lead_times": [3, 1, 1], "discount": 0.98, "levels": [{"low": 64, "mid": 88, "high": 112}, {"low": 80, "mid": 110, "high": 140}, {"low": 96, "mid": 132, "high": 168}], "allowed": 41, "target": 505.52},
    # seed 9071; poisson (mu 20), 17 periods, lead times [2, 2, 2], 17-step plan
    {"seed": 9071, "periods": 17, "demand": [11, 19, 15, 12, 21, 16, 22, 15, 23, 21, 18, 19, 13, 18, 11, 15, 27], "distribution": "poisson", "parameters": {"mu": 20}, "initial": [50, 46, 37], "price": 2.94, "costs": [1.8, 1.41, 1.0, 0.56], "penalties": [0.116, 0.092, 0.062, 0.035], "holding": [0.12, 0.09, 0.07], "capacity": [48, 60, 41], "lead_times": [2, 2, 2], "discount": 0.99, "levels": [{"low": 48, "mid": 66, "high": 84}, {"low": 80, "mid": 110, "high": 140}, {"low": 112, "mid": 154, "high": 196}], "allowed": 17, "target": 564.54},
    # seed 9072; geometric (p 0.12), 15 periods, lead times [1, 1, 2], 15-step plan
    {"seed": 9072, "periods": 15, "demand": [1, 6, 6, 1, 5, 2, 19, 1, 4, 12, 4, 7, 1, 7, 2], "distribution": "geometric", "parameters": {"p": 0.12}, "initial": [11, 19, 18], "price": 1.81, "costs": [1.15, 0.95, 0.67, 0.46], "penalties": [0.149, 0.116, 0.093, 0.068], "holding": [0.19, 0.18, 0.05], "capacity": [18, 13, 15], "lead_times": [1, 1, 2], "discount": 0.97, "levels": [{"low": 13, "mid": 18, "high": 23}, {"low": 20, "mid": 28, "high": 35}, {"low": 33, "mid": 46, "high": 58}], "allowed": 3, "target": 38.63},
    # seed 9073; uniform (low 6, high 19), 15 periods, lead times [2, 3, 1], 15-step plan
    {"seed": 9073, "periods": 15, "demand": [7, 8, 13, 14, 11, 16, 7, 10, 13, 9, 9, 14, 7, 9, 11], "distribution": "uniform", "parameters": {"low": 6, "high": 19}, "initial": [27, 29, 28], "price": 2.11, "costs": [1.25, 0.93, 0.66, 0.41], "penalties": [0.138, 0.087, 0.071, 0.07], "holding": [0.2, 0.13, 0.08], "capacity": [24, 27, 20], "lead_times": [2, 3, 1], "discount": 0.96, "levels": [{"low": 29, "mid": 40, "high": 50}, {"low": 58, "mid": 79, "high": 101}, {"low": 67, "mid": 92, "high": 118}], "allowed": 13, "target": 152.31},
    # seed 9074; binomial (n 27, p 0.57), 12 periods, lead times [1, 1, 1], 12-step plan
    {"seed": 9074, "periods": 12, "demand": [16, 15, 17, 15, 16, 15, 17, 13, 11, 19, 14, 14], "distribution": "binomial", "parameters": {"n": 27, "p": 0.57}, "initial": [26, 35, 21], "price": 1.82, "costs": [1.17, 0.93, 0.54, 0.37], "penalties": [0.118, 0.077, 0.071, 0.035], "holding": [0.18, 0.17, 0.16], "capacity": [38, 32, 41], "lead_times": [1, 1, 1], "discount": 0.97, "levels": [{"low": 25, "mid": 34, "high": 43}, {"low": 37, "mid": 51, "high": 65}, {"low": 49, "mid": 68, "high": 86}], "allowed": 16, "target": 166.52},
    # seed 9075; uniform (low 7, high 14), 15 periods, lead times [1, 1, 2], 15-step plan
    {"seed": 9075, "periods": 15, "demand": [8, 7, 9, 9, 13, 7, 8, 7, 8, 8, 7, 7, 13, 10, 11], "distribution": "uniform", "parameters": {"low": 7, "high": 14}, "initial": [21, 12, 21], "price": 2.42, "costs": [1.36, 1.02, 0.75, 0.53], "penalties": [0.12, 0.086, 0.074, 0.035], "holding": [0.15, 0.07, 0.07], "capacity": [28, 16, 21], "lead_times": [1, 1, 2], "discount": 0.97, "levels": [{"low": 16, "mid": 22, "high": 28}, {"low": 24, "mid": 33, "high": 42}, {"low": 40, "mid": 55, "high": 70}], "allowed": 15, "target": 169.16},
    # seed 9076; uniform (low 4, high 12), 16 periods, lead times [1, 2, 1], 16-step plan
    {"seed": 9076, "periods": 16, "demand": [4, 7, 11, 6, 11, 10, 7, 6, 6, 6, 5, 4, 10, 7, 5, 7], "distribution": "uniform", "parameters": {"low": 4, "high": 12}, "initial": [15, 13, 15], "price": 2.08, "costs": [1.45, 1.24, 0.83, 0.57], "penalties": [0.115, 0.102, 0.073, 0.024], "holding": [0.16, 0.09, 0.05], "capacity": [19, 12, 14], "lead_times": [1, 2, 1], "discount": 0.96, "levels": [{"low": 12, "mid": 16, "high": 21}, {"low": 24, "mid": 33, "high": 42}, {"low": 30, "mid": 41, "high": 52}], "allowed": 8, "target": 106.27},
    # seed 9077; binomial (n 26, p 0.34), 19 periods, lead times [1, 3, 1], 19-step plan
    {"seed": 9077, "periods": 19, "demand": [3, 5, 8, 10, 11, 9, 9, 10, 5, 9, 6, 8, 8, 10, 11, 7, 9, 9, 5], "distribution": "binomial", "parameters": {"n": 26, "p": 0.34}, "initial": [16, 13, 17], "price": 2.25, "costs": [1.27, 1.12, 0.76, 0.38], "penalties": [0.143, 0.137, 0.071, 0.028], "holding": [0.19, 0.1, 0.06], "capacity": [20, 21, 21], "lead_times": [1, 3, 1], "discount": 0.96, "levels": [{"low": 14, "mid": 19, "high": 25}, {"low": 35, "mid": 49, "high": 62}, {"low": 42, "mid": 58, "high": 74}], "allowed": 12, "target": 140.94},
    # seed 9078; geometric (p 0.09), 12 periods, lead times [2, 3, 2], 12-step plan
    {"seed": 9078, "periods": 12, "demand": [1, 19, 3, 10, 1, 8, 3, 6, 6, 2, 7, 7], "distribution": "geometric", "parameters": {"p": 0.09}, "initial": [22, 22, 21], "price": 2.23, "costs": [1.39, 1.12, 0.73, 0.53], "penalties": [0.143, 0.134, 0.117, 0.046], "holding": [0.12, 0.12, 0.11], "capacity": [23, 33, 26], "lead_times": [2, 3, 2], "discount": 0.97, "levels": [{"low": 27, "mid": 37, "high": 47}, {"low": 53, "mid": 73, "high": 93}, {"low": 71, "mid": 98, "high": 124}], "allowed": 3, "target": 79.12},
    # seed 9079; uniform (low 5, high 14), 18 periods, lead times [1, 2, 1], 18-step plan
    {"seed": 9079, "periods": 18, "demand": [5, 12, 6, 5, 8, 5, 9, 13, 7, 12, 9, 8, 6, 7, 12, 8, 13, 9], "distribution": "uniform", "parameters": {"low": 5, "high": 14}, "initial": [9, 14, 10], "price": 2.31, "costs": [1.17, 0.92, 0.68, 0.42], "penalties": [0.111, 0.108, 0.09, 0.024], "holding": [0.15, 0.11, 0.04], "capacity": [21, 20, 14], "lead_times": [1, 2, 1], "discount": 0.96, "levels": [{"low": 14, "mid": 20, "high": 25}, {"low": 29, "mid": 40, "high": 50}, {"low": 36, "mid": 50, "high": 63}], "allowed": 11, "target": 161.27},
    # seed 9080; poisson (mu 11), 19 periods, lead times [2, 2, 1], 19-step plan
    {"seed": 9080, "periods": 19, "demand": [12, 8, 6, 11, 8, 14, 16, 8, 20, 10, 14, 14, 8, 8, 13, 11, 11, 10, 14], "distribution": "poisson", "parameters": {"mu": 11}, "initial": [26, 18, 20], "price": 2.13, "costs": [1.45, 1.14, 0.7, 0.43], "penalties": [0.119, 0.118, 0.056, 0.038], "holding": [0.16, 0.12, 0.11], "capacity": [27, 31, 21], "lead_times": [2, 2, 1], "discount": 0.98, "levels": [{"low": 26, "mid": 36, "high": 46}, {"low": 44, "mid": 61, "high": 77}, {"low": 53, "mid": 73, "high": 92}], "allowed": 21, "target": 221.69},
    # seed 9081; geometric (p 0.1), 13 periods, lead times [1, 1, 2], 13-step plan
    {"seed": 9081, "periods": 13, "demand": [5, 1, 12, 10, 5, 2, 1, 7, 5, 4, 1, 1, 3], "distribution": "geometric", "parameters": {"p": 0.1}, "initial": [21, 17, 13], "price": 2.14, "costs": [1.26, 1.0, 0.74, 0.46], "penalties": [0.147, 0.085, 0.037, 0.035], "holding": [0.16, 0.04, 0.04], "capacity": [20, 20, 22], "lead_times": [1, 1, 2], "discount": 0.96, "levels": [{"low": 16, "mid": 22, "high": 28}, {"low": 24, "mid": 33, "high": 42}, {"low": 40, "mid": 55, "high": 70}], "allowed": 2, "target": 65.75},
    # seed 9082; poisson (mu 12), 13 periods, lead times [3, 1, 1], 13-step plan
    {"seed": 9082, "periods": 13, "demand": [11, 12, 12, 8, 8, 16, 5, 14, 8, 11, 11, 10, 6], "distribution": "poisson", "parameters": {"mu": 12}, "initial": [28, 17, 26], "price": 2.43, "costs": [1.6, 1.21, 0.81, 0.56], "penalties": [0.144, 0.142, 0.123, 0.041], "holding": [0.07, 0.06, 0.05], "capacity": [27, 28, 36], "lead_times": [3, 1, 1], "discount": 0.97, "levels": [{"low": 38, "mid": 53, "high": 67}, {"low": 48, "mid": 66, "high": 84}, {"low": 58, "mid": 79, "high": 101}], "allowed": 11, "target": 183.8},
    # seed 9083; uniform (low 6, high 16), 15 periods, lead times [1, 1, 2], 15-step plan
    {"seed": 9083, "periods": 15, "demand": [7, 10, 9, 11, 14, 9, 12, 9, 15, 7, 10, 11, 13, 13, 6], "distribution": "uniform", "parameters": {"low": 6, "high": 16}, "initial": [21, 11, 25], "price": 2.05, "costs": [1.04, 0.87, 0.53, 0.33], "penalties": [0.077, 0.069, 0.044, 0.022], "holding": [0.16, 0.09, 0.08], "capacity": [19, 21, 31], "lead_times": [1, 1, 2], "discount": 0.99, "levels": [{"low": 17, "mid": 23, "high": 29}, {"low": 25, "mid": 35, "high": 44}, {"low": 42, "mid": 58, "high": 74}], "allowed": 14, "target": 207.08},
    # seed 9084; binomial (n 21, p 0.58), 15 periods, lead times [1, 2, 2], 15-step plan
    {"seed": 9084, "periods": 15, "demand": [13, 13, 14, 11, 15, 12, 11, 12, 9, 11, 10, 15, 11, 14, 14], "distribution": "binomial", "parameters": {"n": 21, "p": 0.58}, "initial": [30, 25, 28], "price": 1.9, "costs": [1.0, 0.77, 0.6, 0.41], "penalties": [0.141, 0.132, 0.099, 0.032], "holding": [0.17, 0.09, 0.04], "capacity": [19, 22, 35], "lead_times": [1, 2, 2], "discount": 0.98, "levels": [{"low": 19, "mid": 27, "high": 34}, {"low": 39, "mid": 54, "high": 68}, {"low": 58, "mid": 80, "high": 102}], "allowed": 10, "target": 193.03},
    # seed 9085; binomial (n 26, p 0.57), 19 periods, lead times [1, 3, 1], 19-step plan
    {"seed": 9085, "periods": 19, "demand": [16, 18, 16, 10, 17, 15, 15, 17, 17, 15, 16, 16, 14, 14, 14, 13, 17, 15, 17], "distribution": "binomial", "parameters": {"n": 26, "p": 0.57}, "initial": [20, 29, 28], "price": 1.91, "costs": [1.08, 0.91, 0.7, 0.43], "penalties": [0.07, 0.046, 0.031, 0.023], "holding": [0.14, 0.12, 0.08], "capacity": [27, 24, 26], "lead_times": [1, 3, 1], "discount": 0.98, "levels": [{"low": 24, "mid": 33, "high": 41}, {"low": 59, "mid": 82, "high": 104}, {"low": 71, "mid": 98, "high": 124}], "allowed": 34, "target": 247.7},
    # seed 9086; geometric (p 0.11), 17 periods, lead times [2, 2, 3], 17-step plan
    {"seed": 9086, "periods": 17, "demand": [2, 3, 3, 4, 9, 4, 6, 14, 1, 1, 1, 19, 9, 2, 9, 2, 25], "distribution": "geometric", "parameters": {"p": 0.11}, "initial": [9, 13, 21], "price": 1.88, "costs": [1.15, 0.75, 0.49, 0.31], "penalties": [0.11, 0.095, 0.09, 0.054], "holding": [0.19, 0.14, 0.12], "capacity": [16, 14, 24], "lead_times": [2, 2, 3], "discount": 0.98, "levels": [{"low": 22, "mid": 30, "high": 38}, {"low": 36, "mid": 50, "high": 64}, {"low": 58, "mid": 80, "high": 102}], "allowed": 11, "target": 47.31},
    # seed 9087; uniform (low 6, high 18), 18 periods, lead times [1, 1, 2], 18-step plan
    {"seed": 9087, "periods": 18, "demand": [17, 6, 9, 10, 9, 13, 14, 7, 14, 10, 17, 6, 6, 6, 17, 15, 7, 17], "distribution": "uniform", "parameters": {"low": 6, "high": 18}, "initial": [29, 20, 21], "price": 2.34, "costs": [1.4, 1.06, 0.73, 0.52], "penalties": [0.146, 0.043, 0.03, 0.023], "holding": [0.15, 0.11, 0.07], "capacity": [30, 22, 28], "lead_times": [1, 1, 2], "discount": 0.98, "levels": [{"low": 18, "mid": 25, "high": 32}, {"low": 28, "mid": 38, "high": 48}, {"low": 46, "mid": 63, "high": 80}], "allowed": 21, "target": 256.49},
    # seed 9088; uniform (low 9, high 17), 19 periods, lead times [3, 1, 1], 19-step plan
    {"seed": 9088, "periods": 19, "demand": [10, 9, 10, 15, 13, 9, 16, 12, 10, 12, 13, 12, 13, 12, 9, 10, 11, 14, 12], "distribution": "uniform", "parameters": {"low": 9, "high": 17}, "initial": [25, 27, 30], "price": 2.72, "costs": [1.65, 1.2, 1.01, 0.58], "penalties": [0.103, 0.089, 0.077, 0.071], "holding": [0.19, 0.16, 0.14], "capacity": [36, 33, 36], "lead_times": [3, 1, 1], "discount": 1.0, "levels": [{"low": 40, "mid": 55, "high": 70}, {"low": 50, "mid": 69, "high": 88}, {"low": 60, "mid": 83, "high": 105}], "allowed": 26, "target": 373.63},
    # seed 9089; poisson (mu 15), 17 periods, lead times [1, 2, 1], 17-step plan
    {"seed": 9089, "periods": 17, "demand": [11, 16, 11, 13, 15, 12, 11, 17, 17, 12, 16, 11, 19, 17, 16, 7, 14], "distribution": "poisson", "parameters": {"mu": 15}, "initial": [26, 37, 16], "price": 2.14, "costs": [1.14, 0.84, 0.67, 0.49], "penalties": [0.078, 0.06, 0.036, 0.029], "holding": [0.11, 0.08, 0.07], "capacity": [28, 28, 25], "lead_times": [1, 2, 1], "discount": 0.97, "levels": [{"low": 24, "mid": 33, "high": 42}, {"low": 48, "mid": 66, "high": 84}, {"low": 60, "mid": 82, "high": 105}], "allowed": 23, "target": 265.07},
    # seed 9090; uniform (low 12, high 26), 14 periods, lead times [3, 1, 2], 14-step plan
    {"seed": 9090, "periods": 14, "demand": [13, 22, 17, 17, 13, 15, 16, 14, 13, 22, 14, 17, 25, 15], "distribution": "uniform", "parameters": {"low": 12, "high": 26}, "initial": [29, 38, 45], "price": 2.15, "costs": [1.51, 1.1, 0.88, 0.45], "penalties": [0.115, 0.114, 0.087, 0.044], "holding": [0.19, 0.15, 0.15], "capacity": [48, 49, 50], "lead_times": [3, 1, 2], "discount": 0.97, "levels": [{"low": 59, "mid": 81, "high": 104}, {"low": 74, "mid": 102, "high": 130}, {"low": 104, "mid": 142, "high": 181}], "allowed": 25, "target": 164.7},
    # seed 9091; uniform (low 12, high 22), 20 periods, lead times [1, 2, 1], 20-step plan
    {"seed": 9091, "periods": 20, "demand": [17, 20, 21, 17, 20, 17, 17, 13, 13, 21, 16, 17, 14, 21, 18, 21, 16, 19, 17, 13], "distribution": "uniform", "parameters": {"low": 12, "high": 22}, "initial": [40, 25, 32], "price": 2.08, "costs": [1.52, 1.16, 0.91, 0.48], "penalties": [0.142, 0.142, 0.137, 0.058], "holding": [0.2, 0.14, 0.12], "capacity": [26, 41, 36], "lead_times": [1, 2, 1], "discount": 0.96, "levels": [{"low": 26, "mid": 36, "high": 46}, {"low": 53, "mid": 73, "high": 92}, {"low": 66, "mid": 91, "high": 115}], "allowed": 34, "target": 285.23},
    # seed 9092; poisson (mu 17), 17 periods, lead times [1, 1, 3], 17-step plan
    {"seed": 9092, "periods": 17, "demand": [19, 16, 15, 12, 12, 16, 15, 14, 18, 17, 19, 19, 11, 16, 21, 16, 20], "distribution": "poisson", "parameters": {"mu": 17}, "initial": [27, 30, 24], "price": 2.9, "costs": [1.73, 1.29, 1.04, 0.59], "penalties": [0.102, 0.099, 0.075, 0.066], "holding": [0.14, 0.11, 0.09], "capacity": [45, 27, 26], "lead_times": [1, 1, 3], "discount": 0.97, "levels": [{"low": 27, "mid": 37, "high": 48}, {"low": 41, "mid": 56, "high": 71}, {"low": 82, "mid": 112, "high": 143}], "allowed": 31, "target": 358.03},
    # seed 9093; poisson (mu 20), 18 periods, lead times [1, 2, 1], 18-step plan
    {"seed": 9093, "periods": 18, "demand": [11, 19, 21, 18, 29, 19, 22, 18, 23, 17, 20, 20, 21, 20, 28, 19, 21, 18], "distribution": "poisson", "parameters": {"mu": 20}, "initial": [23, 43, 38], "price": 2.26, "costs": [1.47, 1.2, 0.92, 0.52], "penalties": [0.09, 0.086, 0.061, 0.039], "holding": [0.15, 0.13, 0.12], "capacity": [43, 43, 49], "lead_times": [1, 2, 1], "discount": 0.96, "levels": [{"low": 32, "mid": 44, "high": 56}, {"low": 64, "mid": 88, "high": 112}, {"low": 80, "mid": 110, "high": 140}], "allowed": 23, "target": 342.25},
    # seed 9094; binomial (n 27, p 0.38), 16 periods, lead times [1, 3, 1], 16-step plan
    {"seed": 9094, "periods": 16, "demand": [9, 7, 13, 4, 11, 8, 7, 11, 14, 12, 10, 10, 11, 8, 13, 11], "distribution": "binomial", "parameters": {"n": 27, "p": 0.38}, "initial": [17, 18, 19], "price": 2.14, "costs": [1.38, 1.06, 0.71, 0.32], "penalties": [0.137, 0.135, 0.106, 0.072], "holding": [0.19, 0.16, 0.14], "capacity": [22, 22, 20], "lead_times": [1, 3, 1], "discount": 0.96, "levels": [{"low": 16, "mid": 23, "high": 29}, {"low": 41, "mid": 56, "high": 72}, {"low": 49, "mid": 68, "high": 86}], "allowed": 18, "target": 136.83},
    # seed 9095; uniform (low 5, high 19), 14 periods, lead times [2, 2, 2], 14-step plan
    {"seed": 9095, "periods": 14, "demand": [6, 12, 6, 18, 7, 7, 14, 11, 10, 17, 14, 14, 18, 15], "distribution": "uniform", "parameters": {"low": 5, "high": 19}, "initial": [20, 24, 29], "price": 2.11, "costs": [1.5, 1.09, 0.92, 0.54], "penalties": [0.125, 0.11, 0.098, 0.054], "holding": [0.17, 0.08, 0.03], "capacity": [34, 19, 34], "lead_times": [2, 2, 2], "discount": 0.96, "levels": [{"low": 28, "mid": 38, "high": 48}, {"low": 46, "mid": 63, "high": 80}, {"low": 64, "mid": 89, "high": 113}], "allowed": 16, "target": 146.43},
    # seed 9096; binomial (n 31, p 0.36), 19 periods, lead times [2, 1, 1], 19-step plan
    {"seed": 9096, "periods": 19, "demand": [12, 14, 8, 11, 9, 11, 11, 7, 14, 11, 14, 9, 15, 10, 13, 11, 14, 7, 9], "distribution": "binomial", "parameters": {"n": 31, "p": 0.36}, "initial": [11, 27, 14], "price": 2.23, "costs": [1.34, 1.14, 0.81, 0.5], "penalties": [0.141, 0.115, 0.077, 0.065], "holding": [0.18, 0.13, 0.08], "capacity": [26, 20, 20], "lead_times": [2, 1, 1], "discount": 0.98, "levels": [{"low": 27, "mid": 37, "high": 47}, {"low": 36, "mid": 49, "high": 62}, {"low": 45, "mid": 61, "high": 78}], "allowed": 25, "target": 216.92},
    # seed 9097; uniform (low 12, high 19), 16 periods, lead times [2, 1, 2], 16-step plan
    {"seed": 9097, "periods": 16, "demand": [16, 14, 15, 17, 16, 12, 15, 17, 14, 16, 16, 14, 18, 12, 12, 15], "distribution": "uniform", "parameters": {"low": 12, "high": 19}, "initial": [16, 31, 33], "price": 2.02, "costs": [1.41, 1.14, 0.79, 0.56], "penalties": [0.106, 0.102, 0.102, 0.081], "holding": [0.17, 0.15, 0.13], "capacity": [43, 34, 35], "lead_times": [2, 1, 2], "discount": 0.98, "levels": [{"low": 36, "mid": 50, "high": 63}, {"low": 48, "mid": 66, "high": 84}, {"low": 72, "mid": 99, "high": 126}], "allowed": 26, "target": 174.54},
    # seed 9098; poisson (mu 18), 18 periods, lead times [2, 1, 1], 18-step plan
    {"seed": 9098, "periods": 18, "demand": [20, 15, 13, 18, 19, 15, 24, 14, 23, 18, 18, 25, 17, 20, 16, 20, 21, 27], "distribution": "poisson", "parameters": {"mu": 18}, "initial": [44, 30, 37], "price": 2.04, "costs": [1.2, 0.77, 0.53, 0.32], "penalties": [0.11, 0.084, 0.057, 0.03], "holding": [0.13, 0.08, 0.06], "capacity": [44, 41, 42], "lead_times": [2, 1, 1], "discount": 0.99, "levels": [{"low": 43, "mid": 59, "high": 76}, {"low": 58, "mid": 79, "high": 101}, {"low": 72, "mid": 99, "high": 126}], "allowed": 31, "target": 457.88},
    # seed 9099; binomial (n 27, p 0.44), 12 periods, lead times [1, 1, 1], 12-step plan
    {"seed": 9099, "periods": 12, "demand": [8, 11, 13, 8, 13, 10, 11, 10, 13, 14, 15, 16], "distribution": "binomial", "parameters": {"n": 27, "p": 0.44}, "initial": [26, 14, 23], "price": 2.27, "costs": [1.23, 0.82, 0.62, 0.42], "penalties": [0.095, 0.078, 0.077, 0.048], "holding": [0.19, 0.17, 0.14], "capacity": [31, 20, 33], "lead_times": [1, 1, 1], "discount": 1.0, "levels": [{"low": 19, "mid": 26, "high": 33}, {"low": 29, "mid": 39, "high": 50}, {"low": 38, "mid": 52, "high": 67}], "allowed": 16, "target": 196.31},
)
