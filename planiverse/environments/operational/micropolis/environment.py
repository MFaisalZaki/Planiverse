"""A city on the Micropolis engine: zone a site a year, and have the population the goal
asks for when the horizon comes.

Micropolis is the GPL release of the original SimCity's simulation (Electronic Arts, 2008;
Micropolis is a registered trademark of Micropolis Corporation (Micropolis GmbH) and is
licensed here as a courtesy of the owner, https://micropolis.com/). Its C++ engine,
MicropolisCore (https://github.com/SimHacker/micropolis), comes with a SWIG binding that
this environment drives headless: `micropolisengine` has to be built from that source (see
`scripts/build_micropolis.sh`), and nothing of it is included here.

The game is the one the engine plays. A map is generated from a seed; a power plant, a road
and the wires are laid out on the largest flat patch of it, with eight sites beside the road;
and once a year the player zones one site residential, commercial or industrial, or waits.
The engine then simulates the year: demand, traffic, power, land value, pollution, growth and
decline, all of it coupled and none of it written down as an action model. Which mix of zones
in which order gives the population the goal asks for by the horizon is only known by running
the years, which is why the environment is here.

## Determinism

The engine reseeds its random numbers from the clock once it has generated a map, so the
environment reseeds them from the instance's seed straight after (which is why the build
script makes `seedRandom` reachable), and every expansion replays the city from the start
(map, layout, decisions and years). A state is therefore its decisions, and expanding it
twice gives the same children. A year of the engine's time is eight hundred ticks and takes
a few milliseconds.

## Instances and generation

`generate_instance(seed, ...)` draws a map and a horizon, lays the city out, measures what a
few fixed zoning plans reach, sets the target above the best of them, and keeps the draw only
if a best-first search over decisions finds a plan that meets it within the budget; the plan
is left in `witness`. The bundled cities are such draws, embedded as plain data with the seed
each came from. The method is generate-and-test (search-based procedural content generation:
Togelius et al. 2011, https://doi.org/10.1109/TCIAIG.2011.2148116).
"""
from collections import namedtuple

from planiverse.environments.base import Environment
from planiverse.environments.generation import bounded_search, draw_until, rng

KINDS = ("R", "C", "I")
#: Ticks of the engine per year, and the sites beside the road.
TICKS_PER_YEAR, SITES = 800, 8
#: The flat patch the layout needs, in tiles.
PATCH_W, PATCH_H = 20, 9


def _engine():
    import micropolisengine
    return micropolisengine


class MicropolisAction:
    """`zone(kind, site)` or `wait`."""

    def __init__(self, kind=None, site=None):
        if kind is not None and kind not in KINDS:
            raise ValueError(f"unknown zone kind: {kind!r}")
        self.kind, self.site = kind, site
        self.name = "wait" if kind is None else f"zone({kind},{site})"

    @classmethod
    def parse(cls, text):
        text = str(text).strip()
        if text == "wait":
            return cls()
        kind, site = text[len("zone("):-1].split(",")
        return cls(kind, int(site))

    def cost(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, MicropolisAction) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    __repr__ = __str__


WAIT = MicropolisAction()


class MicropolisState:
    """The decisions taken so far, one a year, and what the city reads at the end of them.

    Identity is the path: two states with the same decisions are the same city, since the
    engine is deterministic from the seed. The readings (`population`, `funds`, ...) are
    what the decisions produced and are carried for the planner to look at, and `patch`, a
    `CityPatch`, is the engine's map around the layout at the end of them, which `str`
    draws; neither is compared, hashed or in the literals.
    """

    def readings(self):
        """The state's numbers by name, which its OpenSpiel game gives as the named pieces
        of its observation."""
        return {"population": self.population, "residents": self.residents,
                "commerce": self.commerce, "industry": self.industry,
                "funds": self.funds, "score": self.score, "target": self.target}

    def __init__(self, decisions, population, residents, commerce, industry, funds, score,
                 target=0, depth=0, patch=None):
        self.decisions = tuple(decisions)
        self.patch = patch
        self.year = len(self.decisions)
        self.population, self.residents = population, residents
        self.commerce, self.industry = commerce, industry
        self.funds, self.score = funds, score
        self.target = target
        self.depth = depth
        literals = [f"zoned({kind}, {site})" for kind, site in self.decisions if kind is not None]
        literals += [f"year({self.year})", f"population({population // 5 * 5})",
                     f"funds({funds // 1000 * 1000})"]
        self.literals = frozenset(literals)

    def __eq__(self, other):
        return isinstance(other, MicropolisState) and self.decisions == other.decisions

    def __hash__(self):
        return hash(self.decisions)

    def __lt__(self, other):
        return self.depth < other.depth

    def __repr__(self):
        return f"<MicropolisState(year={self.year}, population={self.population}, funds={self.funds})>"

    def __str__(self):
        """The engine's map of the layout and a tile round it, two columns to a tile: each
        zone as a box over its three by three tiles with its letter (R, C or I), filled with ░
        once something is built in it; the power plant as a box marked P; roads in double lines
        and power lines in single ones, ╪ where they cross; each site not yet zoned by its
        number; trees as ♣ and water as ≈."""
        from planiverse.rendering.text import Canvas

        status = (f"year {self.year}   population {self.population} of {self.target}   "
                  f"funds {self.funds}")
        if self.patch is None:
            return status
        me = _engine()
        tiles, (wx, wy) = self.patch.tiles, self.patch.corner
        height, width = len(tiles), len(tiles[0])
        canvas = Canvas(2 * width, height)

        def kind(tx, ty):
            if 0 <= ty < height and 0 <= tx < width:
                return _line_kind(tiles[ty][tx] & me.LOMASK)
            return None

        for ty, row in enumerate(tiles):
            for tx, raw in enumerate(row):
                tile, line = raw & me.LOMASK, kind(tx, ty)
                if line is None:
                    canvas.put(2 * tx, ty, _ground(tile))
                    continue
                if line in ("road-h", "road-v"):
                    canvas.write(2 * tx, ty, ("╪═" if line == "road-h" else "╫ "))
                    continue
                arms = tuple(int(_joins(line, kind(tx + dx, ty + dy), vertical=dx == 0))
                             for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)))
                lines = DOUBLE if line == "road" else SINGLE
                canvas.put(2 * tx, ty, lines.get(arms, lines[(0, 1, 0, 1)] if arms[1] or arms[3]
                                                 else lines[(1, 0, 1, 0)]))
                if arms[1]:
                    canvas.put(2 * tx + 1, ty, lines[(0, 1, 0, 1)])
        zoned = {site for kind_, site in self.decisions if kind_ is not None}
        for site, (sx, sy) in enumerate(site_centres(*self.patch.origin)):
            if site not in zoned:
                canvas.put(2 * (sx - wx), sy - wy, str(site))
        for ty, row in enumerate(tiles):
            for tx, raw in enumerate(row):
                if not raw & me.ZONEBIT:
                    continue
                tile = raw & me.LOMASK
                letter = next((name for name, low, high in ZONES if low <= tile <= high), None)
                if letter is None:
                    continue
                if letter == "P":
                    canvas.fill(2 * tx - 2, ty - 1, 2 * tx + 5, ty + 2, " ")
                    canvas.rect(2 * tx - 2, ty - 1, 2 * tx + 5, ty + 2)
                    canvas.put(2 * tx + 1, ty, "P")
                    continue
                built = tile not in EMPTY_ZONES
                canvas.fill(2 * tx - 2, ty - 1, 2 * tx + 3, ty + 1, "░" if built else " ")
                canvas.rect(2 * tx - 2, ty - 1, 2 * tx + 3, ty + 1)
                canvas.put(2 * tx, ty, letter)
        return str(canvas)


class CityPatch(namedtuple("CityPatch", "origin corner tiles")):
    """The engine's tiles over the layout and one tile round it, as it stood after a state's
    last year: `tiles` rows of raw tile values from the top-left `corner`, and the layout's
    `origin`."""

    __slots__ = ()


#: The engine's tile numbers a picture reads (micropolisengine's `ROADBASE`, `POWERBASE`,
#: `RESBASE` and the rest): zone centres by letter, and the centres of zones nothing is
#: built in yet (`FREEZ`, `COMCLR`, `INDCLR`).
ZONES = (("R", 240, 422), ("C", 423, 611), ("I", 612, 692), ("P", 745, 760), ("P", 811, 826))
EMPTY_ZONES = (244, 427, 616)
#: Box-drawing by (up, right, down, left) arms: double lines for roads, single for wires.
DOUBLE = {(0, 1, 0, 1): "═", (1, 0, 1, 0): "║", (0, 1, 1, 0): "╔", (0, 0, 1, 1): "╗",
          (1, 1, 0, 0): "╚", (1, 0, 0, 1): "╝", (1, 1, 1, 0): "╠", (1, 0, 1, 1): "╣",
          (0, 1, 1, 1): "╦", (1, 1, 0, 1): "╩", (1, 1, 1, 1): "╬"}
SINGLE = {(0, 1, 0, 1): "─", (1, 0, 1, 0): "│", (0, 1, 1, 0): "┌", (0, 0, 1, 1): "┐",
          (1, 1, 0, 0): "└", (1, 0, 0, 1): "┘", (1, 1, 1, 0): "├", (1, 0, 1, 1): "┤",
          (0, 1, 1, 1): "┬", (1, 1, 0, 1): "┴", (1, 1, 1, 1): "┼"}


def _line_kind(tile):
    """What a tile carries: a road, a power line, a road crossed by a power line (the road
    running across, `road-h`, or down, `road-v`), or nothing drawn as a line."""
    if tile == 77:                          # HROADPOWER
        return "road-h"
    if tile == 78:                          # VROADPOWER
        return "road-v"
    if 64 <= tile <= 206:                   # ROADBASE to LASTROAD, bridges and traffic
        return "road"
    if 208 <= tile <= 222:                  # POWERBASE to LASTPOWER
        return "wire"
    return None


def _joins(line, other, vertical):
    """Whether a line reaches a neighbour: a road meets a road, a wire a wire, and either
    meets a crossing along its own direction (a road-h carries the road across, the wire
    down)."""
    if other is None:
        return False
    if other == line:
        return True
    if other in ("road-h", "road-v"):
        road_across = (other == "road-h")
        if line == "road":
            return road_across != vertical
        return road_across == vertical
    return False


def _ground(tile):
    if 2 <= tile <= 20:                     # RIVER to WATER_HIGH
        return "≈"
    if 21 <= tile <= 43:                    # TREEBASE to WOODS5
        return "♣"
    if 44 <= tile <= 47:                    # RUBBLE
        return "·"
    return " "


# ------------------------------------------------------------------------------- the engine

def find_patch(engine):
    """The top-left corner of the first flat patch the layout fits on, or None."""
    me = _engine()
    dirt = {(x, y) for y in range(me.WORLD_H) for x in range(me.WORLD_W)
            if (engine.getTile(x, y) & me.LOMASK) == me.DIRT}
    for y in range(2, me.WORLD_H - PATCH_H - 2):
        for x in range(2, me.WORLD_W - PATCH_W - 2):
            if all((x + dx, y + dy) in dirt for dy in range(PATCH_H) for dx in range(PATCH_W)):
                return x, y
    return None


def lay_out(engine, x0, y0):
    """The power plant, a road between two rows of sites, and wires round them."""
    me = _engine()
    engine.doTool(me.TOOL_COALPOWER, x0 + 2, y0 + 2)
    for x in range(x0 + 5, x0 + PATCH_W - 1):
        engine.doTool(me.TOOL_WIRE, x, y0)
        engine.doTool(me.TOOL_ROAD, x, y0 + 4)
        engine.doTool(me.TOOL_WIRE, x, y0 + 8)
    for y in range(y0, y0 + PATCH_H):
        engine.doTool(me.TOOL_WIRE, x0 + 5, y)
    return site_centres(x0, y0)


def site_centres(x0, y0):
    """The centres of the eight sites, four above the road and four below."""
    return [(x0 + 7 + 3 * k, y0 + 2) for k in range(4)] + [(x0 + 7 + 3 * k, y0 + 6) for k in range(4)]


def replay(seed, origin, decisions):
    """A city from its seed and decisions: the engine's readings after the last year."""
    me = _engine()
    engine = me.Micropolis()
    engine.initGame()
    engine.generateSomeCity(seed)
    engine.seedRandom(seed)      # the engine reseeds from the clock after the map; pin it
    engine.setSpeed(3)
    sites = lay_out(engine, *origin)
    tools = {"R": me.TOOL_RESIDENTIAL, "C": me.TOOL_COMMERCIAL, "I": me.TOOL_INDUSTRIAL}
    for kind, site in decisions:
        if kind is not None:
            engine.doTool(tools[kind], *sites[site])
        for _ in range(TICKS_PER_YEAR):
            engine.simTick()
    x0, y0 = origin
    corner = (x0 - 1, y0 - 1)
    tiles = [[engine.getTile(x, y) for x in range(x0 - 1, x0 + PATCH_W + 1)]
             for y in range(y0 - 1, y0 + PATCH_H + 1)]
    return dict(population=int(engine.totalPop), residents=int(engine.resPop),
                commerce=int(engine.comPop), industry=int(engine.indPop),
                funds=int(engine.totalFunds), score=int(engine.cityScore),
                patch=CityPatch(tuple(origin), corner, tuple(tuple(row) for row in tiles)))


#: Zoning plans that need no thought, used to set a target above what they reach: all
#: residential, alternating residential and industrial, and a mix.
BASELINES = (
    lambda years: [("R", k % SITES) for k in range(min(years, SITES))],
    lambda years: [(("R", "I")[k % 2], k % SITES) for k in range(min(years, SITES))],
    lambda years: [(("R", "C", "I", "R")[k % 4], k % SITES) for k in range(min(years, SITES))],
)


class MicropolisEnv(Environment):
    """Have the population the goal asks for when the horizon comes."""

    def __init__(self):
        super().__init__("micropolis")
        self.instance = None
        self.index = None
        self.state = None
        self.witness = None
        self.witness_expansions = None

    # ------------------------------------------------------------------ instances

    def set_index(self, index):
        if not 0 <= index < len(CITIES):
            raise IndexError(f"Invalid index: {index}. There are {len(CITIES)} cities, so the "
                             f"index must be 0-{len(CITIES) - 1}.")
        self.set_instance(CITIES[index])
        self.index = index

    def set_instance(self, instance):
        """Select a city: `{"seed": s, "origin": [x, y], "years": n, "target": p}`."""
        for key in ("seed", "origin", "years", "target"):
            if key not in instance:
                raise ValueError(f"a city needs `{key}`")
        self.instance = {"seed": int(instance["seed"]), "origin": tuple(int(v) for v in instance["origin"]),
                         "years": int(instance["years"]), "target": int(instance["target"])}
        self.index = None
        self.witness = self.witness_expansions = None

    def generate_instance(self, seed=None, years=None, slack=0.15, search_limit=40, attempts=80):
        """Draw a city, select it, and return it as the dict `set_instance` takes.

        A map is generated from a seed of the draw's, the layout placed on its first flat
        patch (a map with none is thrown back), and the horizon is `years` (ten, fifteen or
        twenty when unset). The target is the best population the baseline zoning plans
        reach, raised by `slack`, and the draw is kept only if a best-first search over
        decisions, guided by the population the city would have at the horizon if nothing
        more were zoned, meets it within `search_limit` expansions; that
        plan is left in `witness` and what the search spent in `witness_expansions`.
        """
        random_, _ = rng(seed)
        me = _engine()
        found = {}

        def draw(attempt):
            map_seed = random_.randint(1, 2 ** 30)
            engine = me.Micropolis()
            engine.initGame()
            engine.generateSomeCity(map_seed)
            origin = find_patch(engine)
            if origin is None:
                return None
            horizon = years or random_.choice((10, 15, 20))
            best = max(replay(map_seed, origin, baseline(horizon) + [(None, None)] * (horizon - min(horizon, SITES)))["population"]
                       for baseline in BASELINES)
            if best <= 0:
                return None
            return {"seed": map_seed, "origin": list(origin), "years": horizon,
                    "target": int(best * (1 + slack)) + 1}

        def accept(instance):
            self.set_instance(instance)
            outcome = bounded_search(self, search_limit, progress=self.__progress__)
            if outcome.plan is None:
                return False
            found["plan"], found["expansions"] = outcome.plan, outcome.expansions
            return True

        instance = draw_until(draw, accept, attempts, "Micropolis city")
        self.set_instance(instance)
        self.witness, self.witness_expansions = found["plan"], found["expansions"]
        return instance

    def __progress__(self, state):
        """The population the city would have at the horizon if nothing more were zoned:
        a rollout, since the year's readings say little about where the city is going."""
        left = self.instance["years"] - state.year
        if left <= 0:
            return -state.population
        return -replay(self.instance["seed"], self.instance["origin"],
                       state.decisions + ((None, None),) * left)["population"]

    # ------------------------------------------------------------------- contract

    def __state__(self, decisions, depth=0):
        readings = replay(self.instance["seed"], self.instance["origin"], decisions)
        return MicropolisState(decisions, target=self.instance["target"], depth=depth, **readings)

    def reset(self):
        if self.instance is None:
            self.set_index(0)
        self.state = self.__state__(())
        return self.state, {"city": self.index, "years": self.instance["years"],
                            "target": self.instance["target"], "sites": SITES,
                            "generated": self.index is None}

    def is_goal(self, state):
        return state.year >= self.instance["years"] and state.population >= self.instance["target"]

    def is_terminal(self, state):
        return state.year >= self.instance["years"] and state.population < self.instance["target"]

    def get_actions(self, state=None):
        state = state or self.state
        taken = {site for kind, site in state.decisions if kind is not None}
        return [MicropolisAction(kind, site) for site in range(SITES) if site not in taken
                for kind in KINDS] + [WAIT]

    def successors(self, state):
        if self.is_goal(state) or self.is_terminal(state):
            return []
        children = []
        for action in self.get_actions(state):
            child = self.__advance__(state, action)
            if child != state:
                children.append((action, child))
        return children

    def __advance__(self, state, action):
        if self.is_goal(state) or self.is_terminal(state):
            return state
        if not isinstance(action, MicropolisAction):
            action = MicropolisAction.parse(action)
        taken = {site for kind, site in state.decisions if kind is not None}
        if action.kind is not None and (action.site in taken or not 0 <= action.site < SITES):
            return state
        return self.__state__(state.decisions + ((action.kind, action.site),), state.depth + 1)

#: The bundled cities: `generate_instance(seed)` for the seed beside each, embedded as the
#: plain data `set_instance` takes, with the plan each was accepted on in
#: `tests/data/micropolis_solutions.json`.
CITIES = (
    # seed 7000; 10 expansions, 10-year plan
    {"seed": 692463416, "origin": [61, 22], "years": 10, "target": 26},
    # seed 7001; 20 expansions, 20-year plan
    {"seed": 688680785, "origin": [49, 2], "years": 20, "target": 18},
    # seed 7002; 15 expansions, 15-year plan
    {"seed": 639525392, "origin": [25, 9], "years": 15, "target": 30},
    # seed 7003; 10 expansions, 10-year plan
    {"seed": 1072880874, "origin": [81, 46], "years": 10, "target": 22},
    # seed 7004; 10 expansions, 10-year plan
    {"seed": 243111630, "origin": [37, 41], "years": 10, "target": 30},
    # seed 7005; 20 expansions, 20-year plan
    {"seed": 874512669, "origin": [74, 26], "years": 20, "target": 15},
    # seed 7006; 10 expansions, 10-year plan
    {"seed": 841155762, "origin": [2, 2], "years": 10, "target": 18},
    # seed 7007; 20 expansions, 20-year plan
    {"seed": 568064233, "origin": [87, 28], "years": 20, "target": 17},
    # seed 7008; 20 expansions, 20-year plan
    {"seed": 575556508, "origin": [34, 39], "years": 20, "target": 15},
    # seed 7009; 12 expansions, 10-year plan
    {"seed": 568266217, "origin": [40, 65], "years": 10, "target": 12},
    # seed 7010; 20 expansions, 20-year plan
    {"seed": 202660203, "origin": [59, 39], "years": 20, "target": 26},
    # seed 7011; 20 expansions, 20-year plan
    {"seed": 764032050, "origin": [14, 14], "years": 20, "target": 19},
    # seed 7012; 20 expansions, 20-year plan
    {"seed": 371520921, "origin": [17, 12], "years": 20, "target": 26},
    # seed 7013; 15 expansions, 15-year plan
    {"seed": 80736956, "origin": [3, 67], "years": 15, "target": 29},
    # seed 7014; 11 expansions, 10-year plan
    {"seed": 130542058, "origin": [6, 81], "years": 10, "target": 18},
    # seed 7015; 22 expansions, 20-year plan
    {"seed": 532303452, "origin": [26, 10], "years": 20, "target": 18},
    # seed 7016; 12 expansions, 10-year plan
    {"seed": 244602847, "origin": [20, 46], "years": 10, "target": 20},
    # seed 7017; 15 expansions, 15-year plan
    {"seed": 1048653960, "origin": [91, 73], "years": 15, "target": 21},
    # seed 7018; 40 expansions, 20-year plan
    {"seed": 727461025, "origin": [51, 19], "years": 20, "target": 21},
    # seed 7019; 20 expansions, 20-year plan
    {"seed": 733476396, "origin": [91, 58], "years": 20, "target": 19},
    # seed 7020; 15 expansions, 15-year plan
    {"seed": 148279070, "origin": [15, 33], "years": 15, "target": 28},
    # seed 7021; 20 expansions, 20-year plan
    {"seed": 759555117, "origin": [18, 40], "years": 20, "target": 20},
    # seed 7022; 15 expansions, 15-year plan
    {"seed": 133495549, "origin": [10, 44], "years": 15, "target": 19},
    # seed 7023; 27 expansions, 20-year plan
    {"seed": 814948069, "origin": [30, 2], "years": 20, "target": 21},
    # seed 7024; 10 expansions, 10-year plan
    {"seed": 813943220, "origin": [54, 75], "years": 10, "target": 14},
    # seed 7025; 10 expansions, 10-year plan
    {"seed": 294859630, "origin": [2, 2], "years": 10, "target": 15},
    # seed 7026; 16 expansions, 15-year plan
    {"seed": 750711609, "origin": [33, 2], "years": 15, "target": 13},
    # seed 7027; 16 expansions, 15-year plan
    {"seed": 66213408, "origin": [51, 33], "years": 15, "target": 27},
    # seed 7028; 18 expansions, 10-year plan
    {"seed": 919359005, "origin": [2, 2], "years": 10, "target": 9},
    # seed 7029; 22 expansions, 15-year plan
    {"seed": 615180193, "origin": [2, 2], "years": 15, "target": 18},
    # seed 7030; 20 expansions, 20-year plan
    {"seed": 71742729, "origin": [73, 26], "years": 20, "target": 22},
    # seed 7031; 10 expansions, 10-year plan
    {"seed": 32595939, "origin": [39, 11], "years": 10, "target": 26},
    # seed 7032; 16 expansions, 15-year plan
    {"seed": 453865980, "origin": [89, 14], "years": 15, "target": 14},
    # seed 7033; 10 expansions, 10-year plan
    {"seed": 7765428, "origin": [74, 83], "years": 10, "target": 13},
    # seed 7100; 10 expansions, 10-year plan
    {"seed": 128887149, "origin": [21, 21], "years": 10, "target": 27},
    # seed 7101; 21 expansions, 20-year plan
    {"seed": 1057760496, "origin": [67, 31], "years": 20, "target": 20},
    # seed 7102; 15 expansions, 15-year plan
    {"seed": 453471395, "origin": [31, 13], "years": 15, "target": 30},
    # seed 7103; 21 expansions, 15-year plan
    {"seed": 983289976, "origin": [9, 81], "years": 15, "target": 28},
    # seed 7104; 37 expansions, 20-year plan
    {"seed": 873603864, "origin": [2, 2], "years": 20, "target": 15},
    # seed 7105; 20 expansions, 20-year plan
    {"seed": 1001974512, "origin": [93, 86], "years": 20, "target": 15},
    # seed 7106; 16 expansions, 15-year plan
    {"seed": 1070815447, "origin": [2, 2], "years": 15, "target": 20},
    # seed 7107; 15 expansions, 15-year plan
    {"seed": 59360224, "origin": [97, 77], "years": 15, "target": 26},
    # seed 7108; 20 expansions, 20-year plan
    {"seed": 837800045, "origin": [69, 23], "years": 20, "target": 34},
    # seed 7109; 21 expansions, 20-year plan
    {"seed": 72847068, "origin": [8, 49], "years": 20, "target": 3},
    # seed 7110; 16 expansions, 15-year plan
    {"seed": 353416251, "origin": [62, 2], "years": 15, "target": 17},
    # seed 7111; 12 expansions, 10-year plan
    {"seed": 391921156, "origin": [9, 2], "years": 10, "target": 22},
    # seed 7112; 21 expansions, 20-year plan
    {"seed": 467749341, "origin": [73, 28], "years": 20, "target": 24},
    # seed 7113; 17 expansions, 15-year plan
    {"seed": 502014890, "origin": [97, 18], "years": 15, "target": 5},
    # seed 7114; 19 expansions, 10-year plan
    {"seed": 797338681, "origin": [62, 38], "years": 10, "target": 21},
    # seed 7115; 10 expansions, 10-year plan
    {"seed": 94638757, "origin": [37, 4], "years": 10, "target": 26},
    # seed 7116; 26 expansions, 15-year plan
    {"seed": 404412207, "origin": [24, 19], "years": 15, "target": 13},
    # seed 7117; 10 expansions, 10-year plan
    {"seed": 170054036, "origin": [2, 2], "years": 10, "target": 18},
    # seed 7118; 15 expansions, 15-year plan
    {"seed": 352956802, "origin": [89, 67], "years": 15, "target": 33},
    # seed 7119; 21 expansions, 20-year plan
    {"seed": 750491905, "origin": [91, 35], "years": 20, "target": 17},
    # seed 7120; 10 expansions, 10-year plan
    {"seed": 467675726, "origin": [52, 11], "years": 10, "target": 19},
    # seed 7121; 11 expansions, 10-year plan
    {"seed": 309642681, "origin": [84, 86], "years": 10, "target": 18},
    # seed 7122; 17 expansions, 15-year plan
    {"seed": 292790035, "origin": [2, 2], "years": 15, "target": 15},
    # seed 7123; 15 expansions, 15-year plan
    {"seed": 1065896097, "origin": [2, 2], "years": 15, "target": 17},
    # seed 7124; 24 expansions, 20-year plan
    {"seed": 650220339, "origin": [70, 7], "years": 20, "target": 7},
    # seed 7125; 11 expansions, 10-year plan
    {"seed": 759183766, "origin": [61, 56], "years": 10, "target": 26},
    # seed 7126; 11 expansions, 10-year plan
    {"seed": 592974808, "origin": [49, 2], "years": 10, "target": 27},
    # seed 7127; 10 expansions, 10-year plan
    {"seed": 245329377, "origin": [73, 78], "years": 10, "target": 28},
    # seed 7128; 20 expansions, 20-year plan
    {"seed": 46468250, "origin": [67, 77], "years": 20, "target": 27},
    # seed 7129; 15 expansions, 15-year plan
    {"seed": 917221031, "origin": [85, 27], "years": 15, "target": 14},
    # seed 7130; 21 expansions, 20-year plan
    {"seed": 895814153, "origin": [44, 2], "years": 20, "target": 15},
    # seed 7131; 15 expansions, 15-year plan
    {"seed": 156516182, "origin": [91, 19], "years": 15, "target": 26},
    # seed 7132; 21 expansions, 20-year plan
    {"seed": 399462917, "origin": [22, 13], "years": 20, "target": 15},
    # seed 7200; 21 expansions, 20-year plan
    {"seed": 641114446, "origin": [96, 22], "years": 20, "target": 19},
    # seed 7201; 21 expansions, 20-year plan
    {"seed": 3635365, "origin": [90, 32], "years": 20, "target": 13},
    # seed 7202; 10 expansions, 10-year plan
    {"seed": 124778390, "origin": [19, 2], "years": 10, "target": 27},
    # seed 7203; 11 expansions, 10-year plan
    {"seed": 165121456, "origin": [51, 27], "years": 10, "target": 27},
    # seed 7204; 15 expansions, 15-year plan
    {"seed": 697586777, "origin": [61, 2], "years": 15, "target": 25},
    # seed 7205; 20 expansions, 20-year plan
    {"seed": 459684923, "origin": [60, 35], "years": 20, "target": 17},
    # seed 7206; 15 expansions, 15-year plan
    {"seed": 373252162, "origin": [7, 31], "years": 15, "target": 26},
    # seed 7207; 20 expansions, 20-year plan
    {"seed": 310997089, "origin": [43, 76], "years": 20, "target": 27},
    # seed 7208; 15 expansions, 15-year plan
    {"seed": 703238321, "origin": [19, 25], "years": 15, "target": 21},
    # seed 7209; 20 expansions, 20-year plan
    {"seed": 971102158, "origin": [51, 31], "years": 20, "target": 25},
    # seed 7210; 10 expansions, 10-year plan
    {"seed": 333834436, "origin": [19, 43], "years": 10, "target": 26},
    # seed 7211; 26 expansions, 15-year plan
    {"seed": 24795300, "origin": [96, 21], "years": 15, "target": 18},
    # seed 7212; 21 expansions, 15-year plan
    {"seed": 556973088, "origin": [53, 60], "years": 15, "target": 18},
    # seed 7213; 10 expansions, 10-year plan
    {"seed": 1007614398, "origin": [17, 2], "years": 10, "target": 20},
    # seed 7214; 39 expansions, 20-year plan
    {"seed": 709638021, "origin": [8, 2], "years": 20, "target": 20},
    # seed 7215; 23 expansions, 20-year plan
    {"seed": 1014949790, "origin": [84, 43], "years": 20, "target": 15},
    # seed 7216; 20 expansions, 20-year plan
    {"seed": 619576549, "origin": [60, 62], "years": 20, "target": 9},
    # seed 7217; 15 expansions, 15-year plan
    {"seed": 567578043, "origin": [29, 62], "years": 15, "target": 21},
    # seed 7218; 16 expansions, 15-year plan
    {"seed": 187291627, "origin": [34, 37], "years": 15, "target": 19},
    # seed 7219; 11 expansions, 10-year plan
    {"seed": 419512601, "origin": [68, 11], "years": 10, "target": 17},
    # seed 7220; 22 expansions, 20-year plan
    {"seed": 369696613, "origin": [32, 13], "years": 20, "target": 15},
    # seed 7221; 10 expansions, 10-year plan
    {"seed": 669402921, "origin": [2, 40], "years": 10, "target": 20},
    # seed 7222; 10 expansions, 10-year plan
    {"seed": 911095858, "origin": [25, 15], "years": 10, "target": 21},
    # seed 7223; 10 expansions, 10-year plan
    {"seed": 315370096, "origin": [90, 23], "years": 10, "target": 20},
    # seed 7224; 21 expansions, 20-year plan
    {"seed": 782163027, "origin": [9, 7], "years": 20, "target": 10},
    # seed 7225; 12 expansions, 10-year plan
    {"seed": 1072132728, "origin": [22, 64], "years": 10, "target": 19},
    # seed 7226; 10 expansions, 10-year plan
    {"seed": 350829691, "origin": [86, 73], "years": 10, "target": 13},
    # seed 7227; 12 expansions, 10-year plan
    {"seed": 163577741, "origin": [30, 66], "years": 10, "target": 17},
    # seed 7228; 16 expansions, 15-year plan
    {"seed": 304910282, "origin": [90, 40], "years": 15, "target": 21},
    # seed 7229; 15 expansions, 15-year plan
    {"seed": 532953898, "origin": [71, 22], "years": 15, "target": 21},
    # seed 7230; 15 expansions, 15-year plan
    {"seed": 506902021, "origin": [93, 2], "years": 15, "target": 24},
    # seed 7231; 20 expansions, 20-year plan
    {"seed": 971077735, "origin": [11, 36], "years": 20, "target": 20},
    # seed 7232; 10 expansions, 10-year plan
    {"seed": 491710463, "origin": [2, 2], "years": 10, "target": 20},
)
