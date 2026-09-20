"""A cellular fluid puzzle: water pours from a spring into a cave of rock and earth, and the
player digs channels, a cell at a time and only so many, to bring enough of it to a basin
before the spring runs dry, past drains that swallow whatever falls into them.

Water is a cellular automaton in the falling-sand family: each tick every water cell falls
if it can, else slides diagonally, else moves sideways, in a fixed scan order with the side
tried first alternating by tick, so the flow is deterministic and the same on every
machine. A pool on a floor keeps shifting about, as the genre's water does; what matters
to the puzzle is where it can fall next. That automaton is the transition function.
A dig is one cell of earth removed, after which the water runs for a period; what reaches
the basin depends on the whole cave and on everything dug before, and no add or delete list
carries a flow. The mechanic is the one the falling-sand games and the water-routing puzzles
share; nothing here is taken from any published title, and the environment needs no
dependency.

## Instances and generation

`generate_instance(seed, ...)` draws a cave, a spring, a basin and a drain, and keeps the draw
only if waiting without digging fails and a best-first search over digs, guided by how much
water the basin still needs, finds a plan within the budget; the plan is left in `witness`.
The bundled caves are such draws, embedded as plain text with the seed each came from. The
method is generate-and-test (search-based procedural content generation: Togelius et al.
2011, https://doi.org/10.1109/TCIAIG.2011.2148116).
"""
from planiverse.environments.base import Environment
from planiverse.environments.generation import bounded_search, draw_until, rng

WIDTH, HEIGHT = 16, 10
ROCK, EARTH, AIR, WATER, SPRING, BASIN, DRAIN = "#", ".", " ", "~", "S", "T", "X"
#: Ticks the water runs after each action, and the most ticks a game may take.
PERIOD, MAX_TICKS = 12, 240


class FluidAction:
    """`dig(x, y)` or `wait`."""

    def __init__(self, x=None, y=None):
        self.x, self.y = x, y
        self.name = "wait" if x is None else f"dig({x},{y})"

    @classmethod
    def parse(cls, text):
        text = str(text).strip()
        if text == "wait":
            return cls()
        x, y = (int(part) for part in text[len("dig("):-1].split(","))
        return cls(x, y)

    def cost(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, FluidAction) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    __repr__ = __str__


WAIT = FluidAction()


class FluidState:
    """The cave as a tuple of row strings, the water the spring has left, the water the basin
    has taken, the digs left, and the tick."""

    def __init__(self, rows, supply, filled, digs_left, tick, need=0, depth=0):
        self.rows = tuple(rows)
        self.supply, self.filled, self.digs_left, self.tick = supply, filled, digs_left, tick
        self.need = need                         # the basin's demand, carried for the measures
        self.depth = depth
        literals = [f"cell({x}, {y}, {'water' if cell == WATER else 'air' if cell == AIR else 'earth'})"
                    for y, row in enumerate(self.rows) for x, cell in enumerate(row)
                    if cell in (WATER, AIR, EARTH)]
        literals += [f"filled({filled})", f"supply({supply})", f"digs_left({digs_left})"]
        self.literals = frozenset(literals)

    @property
    def afloat(self):
        return sum(row.count(WATER) for row in self.rows)

    def __eq__(self, other):
        return (isinstance(other, FluidState) and self.rows == other.rows
                and self.supply == other.supply and self.filled == other.filled
                and self.digs_left == other.digs_left and self.tick == other.tick)

    def __hash__(self):
        return hash((self.rows, self.supply, self.filled, self.digs_left, self.tick))

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        return "\n".join(self.rows) + (f"\nfilled {self.filled}, spring {self.supply} left, "
                                       f"{self.digs_left} digs left, tick {self.tick}")

    def __repr__(self):
        return f"<FluidState(filled={self.filled}, supply={self.supply}, digs={self.digs_left}, tick={self.tick})>"


# ---------------------------------------------------------------------------- the automaton

def tick(rows, supply, filled, step):
    """One tick of the automaton: the spring emits a unit if it can, and every water cell
    falls, slides or spreads. Returns `(rows, supply, filled)`."""
    grid = [list(row) for row in rows]
    height, width = len(grid), len(grid[0])
    moved = set()
    if supply > 0:
        for y in range(height):
            for x in range(width):
                if grid[y][x] == SPRING and y + 1 < height and grid[y + 1][x] == AIR:
                    grid[y + 1][x] = WATER
                    moved.add((x, y + 1))
                    supply -= 1
                    break
            else:
                continue
            break
    first, second = (-1, 1) if step % 2 == 0 else (1, -1)
    for y in range(height - 2, -1, -1):
        columns = range(width) if step % 2 == 0 else range(width - 1, -1, -1)
        for x in columns:
            if grid[y][x] != WATER or (x, y) in moved:
                continue
            for dx, dy in ((0, 1), (first, 1), (second, 1), (first, 0), (second, 0)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                target = grid[ny][nx]
                if target == AIR:
                    grid[ny][nx], grid[y][x] = WATER, AIR
                    moved.add((nx, ny))
                    break
                if target == BASIN:
                    grid[y][x] = AIR
                    filled += 1
                    break
                if target == DRAIN:
                    grid[y][x] = AIR
                    break
    return tuple("".join(row) for row in grid), supply, filled


def run(rows, supply, filled, start, ticks):
    for step in range(start, start + ticks):
        rows, supply, filled = tick(rows, supply, filled, step)
    return rows, supply, filled


def diggable(rows):
    """Earth cells with water against them: the only digs that move water now, so the
    player follows the flow rather than tunnelling anywhere."""
    cells = []
    for y, row in enumerate(rows):
        for x, cell in enumerate(row):
            if cell != EARTH:
                continue
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = x + dx, y + dy
                if 0 <= ny < len(rows) and 0 <= nx < len(row) and rows[ny][nx] == WATER:
                    cells.append((x, y))
                    break
    return cells


def distance_to_basin(rows):
    """How far the nearest water is from the basin, as the crow flies over cells; the
    width of the cave when there is no water afloat."""
    basin = next((x, y) for y, row in enumerate(rows) for x, cell in enumerate(row) if cell == BASIN)
    water = [(x, y) for y, row in enumerate(rows) for x, cell in enumerate(row) if cell == WATER]
    if not water:
        return len(rows[0])
    return min(abs(x - basin[0]) + abs(y - basin[1]) for x, y in water)


# ------------------------------------------------------------------------------- the caves

def draw_cave(random_, need=None, digs=None):
    """A random cave as the instance dict `set_instance` takes: earth inside a rock border,
    pockets of air, a spring near the top left, a basin near the bottom right with air over
    it, and a drain in between."""
    grid = [[ROCK if x in (0, WIDTH - 1) or y in (0, HEIGHT - 1) else EARTH
             for x in range(WIDTH)] for y in range(HEIGHT)]
    for _ in range(random_.randint(2, 4)):                      # pockets of air
        cx, cy = random_.randint(3, WIDTH - 4), random_.randint(2, HEIGHT - 3)
        w, h = random_.randint(1, 3), random_.randint(1, 2)
        for y in range(max(1, cy - h), min(HEIGHT - 1, cy + h + 1)):
            for x in range(max(1, cx - w), min(WIDTH - 1, cx + w + 1)):
                grid[y][x] = AIR
    for _ in range(random_.randint(1, 3)):                      # veins of rock
        x, y = random_.randint(2, WIDTH - 3), random_.randint(2, HEIGHT - 3)
        for _ in range(random_.randint(2, 5)):
            grid[y][x] = ROCK
            x += random_.choice((-1, 0, 1))
            y += random_.choice((0, 1))
            if not (1 <= x < WIDTH - 1 and 1 <= y < HEIGHT - 1):
                break
    sx = random_.randint(2, 5)
    grid[1][sx] = SPRING
    grid[2][sx] = AIR
    bx = random_.randint(WIDTH - 5, WIDTH - 3)
    grid[HEIGHT - 2][bx] = BASIN
    grid[HEIGHT - 3][bx] = AIR
    dx = random_.randint(sx + 2, bx - 3)
    grid[HEIGHT - 2][dx] = DRAIN
    grid[HEIGHT - 3][dx] = AIR
    need = need or random_.randint(8, 12)
    return {"rows": ["".join(row) for row in grid], "need": need, "supply": need + random_.randint(3, 6),
            "digs": digs or random_.randint(6, 9)}


class FluidEnv(Environment):
    """Bring enough water to the basin within the digs and the time given."""

    def __init__(self):
        super().__init__("fluid")
        self.instance = None
        self.index = None
        self.state = None
        self.state_history = []
        self.witness = None
        self.witness_expansions = None

    # ------------------------------------------------------------------ instances

    def set_index(self, index):
        if not 0 <= index < len(CAVES):
            raise IndexError(f"Invalid index: {index}. There are {len(CAVES)} caves, so the "
                             f"index must be 0-{len(CAVES) - 1}.")
        self.set_instance(CAVES[index])
        self.index = index

    def set_instance(self, instance):
        """Select a cave: `{"rows": [...], "need": n, "supply": s, "digs": d}`."""
        for key in ("rows", "need", "supply", "digs"):
            if key not in instance:
                raise ValueError(f"a cave needs `{key}`")
        rows = [str(row) for row in instance["rows"]]
        if len({len(row) for row in rows}) != 1:
            raise ValueError("a cave's rows are all the same width")
        self.instance = {"rows": rows, "need": int(instance["need"]),
                         "supply": int(instance["supply"]), "digs": int(instance["digs"])}
        self.index = None
        self.witness = self.witness_expansions = None

    def generate_instance(self, seed=None, need=None, digs=None, search_limit=200, attempts=40):
        """Draw a cave, select it, and return it as the dict `set_instance` takes.

        `need` (water units the basin must take, 8 to 12 when unset) and `digs` (6 to 9)
        are `draw_cave`'s. A draw is kept only if waiting without digging fails and a
        best-first search over digs, guided by what the basin still needs and then by how
        far the nearest water is from it, finds a plan
        within `search_limit` expansions; that plan is left in `witness` and what the search
        spent in `witness_expansions`.
        """
        random_, _ = rng(seed)
        found = {}

        def draw(attempt):
            return draw_cave(random_, need=need, digs=digs)

        def accept(instance):
            self.set_instance(instance)
            idle = self.simulate([WAIT] * (MAX_TICKS // PERIOD))[-1]
            if self.is_goal(idle):
                return False
            outcome = bounded_search(self, search_limit, progress=self.__progress__)
            if outcome.plan is None:
                return False
            found["plan"], found["expansions"] = outcome.plan, outcome.expansions
            return True

        instance = draw_until(draw, accept, attempts, "fluid cave")
        self.set_instance(instance)
        self.witness, self.witness_expansions = found["plan"], found["expansions"]
        return instance

    def __progress__(self, state):
        """What the basin still needs, then how far the nearest water is from it."""
        return (self.instance["need"] - state.filled) * 100 + distance_to_basin(state.rows)

    # ------------------------------------------------------------------- contract

    def reset(self):
        if self.instance is None:
            self.set_index(0)
        self.state = FluidState(self.instance["rows"], self.instance["supply"], 0,
                                self.instance["digs"], 0, self.instance["need"])
        self.state_history = [self.state]
        return self.state, {"cave": self.index, "need": self.instance["need"],
                            "supply": self.instance["supply"], "digs": self.instance["digs"],
                            "generated": self.index is None}

    def is_goal(self, state):
        return state.filled >= self.instance["need"]

    def is_terminal(self, state):
        if self.is_goal(state):
            return False
        short = self.instance["need"] - state.filled
        return state.tick >= MAX_TICKS or state.supply + state.afloat < short

    def get_actions(self, state=None):
        state = state or self.state
        return [FluidAction(x, y) for x, y in diggable(state.rows)] + [WAIT]

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
        if not isinstance(action, FluidAction):
            action = FluidAction.parse(action)
        rows = state.rows
        digs_left = state.digs_left
        if action.x is not None:
            if digs_left == 0 or (action.x, action.y) not in diggable(rows):
                return state
            row = list(rows[action.y])
            row[action.x] = AIR
            rows = rows[:action.y] + ("".join(row),) + rows[action.y + 1:]
            digs_left -= 1
        rows, supply, filled = run(rows, state.supply, state.filled, state.tick, PERIOD)
        return FluidState(rows, supply, filled, digs_left, state.tick + PERIOD, state.need,
                          state.depth + 1)

    def simulate(self, plan):
        state, _ = self.reset()
        trace = [state]
        for action in plan:
            trace.append(self.__advance__(trace[-1], action))
        return trace

    def step(self, action):
        if self.state is None:
            raise ValueError("call reset() first")
        before = self.state.filled
        self.state = self.__advance__(self.state, action)
        self.state_history.append(self.state)
        return self.state, self.state.filled - before

    def render(self):
        lines = [f"step {k}:\n{state}" for k, state in enumerate(self.state_history)]
        print("\n".join(lines))
        return lines


#: The bundled caves: `generate_instance(seed)` for the seed beside each, embedded as the
#: plain data `set_instance` takes, with the plan each was accepted on in
#: `tests/data/fluid_solutions.json`.
CAVES = (
    # seed 4000; 3 expansions, 3-action plan
    {"rows": ["################",
             "#....S.........#",
             "#....       ...#",
             "#.#..       ...#",
             "##...       ...#",
             "#....###  .....#",
             "#...###   .....#",
             "#.....#   . ...#",
             "#......X# .T...#",
             "################"],
     "need": 12, "supply": 15, "digs": 9},
    # seed 4001; 7 expansions, 7-action plan
    {"rows": ["################",
             "#....S.........#",
             "#....        ..#",
             "#..#.        ..#",
             "#...##       ..#",
             "#..............#",
             "#..............#",
             "#........ #.. .#",
             "#........X#..T.#",
             "################"],
     "need": 10, "supply": 14, "digs": 9},
    # seed 4002; 7 expansions, 7-action plan
    {"rows": ["################",
             "#..S...     ...#",
             "#.. ...#    ...#",
             "#...... #   ...#",
             "#......#    ...#",
             "#......     ...#",
             "#..............#",
             "#..... ..... ..#",
             "#.....X.....T..#",
             "################"],
     "need": 8, "supply": 11, "digs": 9},
    # seed 4003; 6 expansions, 6-action plan
    {"rows": ["################",
             "#...S..........#",
             "#... .....     #",
             "#...     .#    #",
             "#...     .#    #",
             "#...     .#....#",
             "#..............#",
             "#..... ..... ..#",
             "#.....X.....T..#",
             "################"],
     "need": 9, "supply": 13, "digs": 6},
    # seed 4004; 9 expansions, 9-action plan
    {"rows": ["################",
             "#.S..   .......#",
             "#. ..   .......#",
             "#....  #       #",
             "#....   #      #",
             "#....   ##     #",
             "#..............#",
             "#.... ..... ...#",
             "#....X.....T...#",
             "################"],
     "need": 8, "supply": 14, "digs": 8},
    # seed 4005; 5 expansions, 5-action plan
    {"rows": ["################",
             "#...S..        #",
             "#..# ..        #",
             "#....#.        #",
             "#.........     #",
             "#.........     #",
             "#.........     #",
             "#.##... ..     #",
             "#......X.....T.#",
             "################"],
     "need": 8, "supply": 12, "digs": 6},
    # seed 4006; 7 expansions, 7-action plan
    {"rows": ["################",
             "#...S..........#",
             "#... ....   ...#",
             "#........      #",
             "#........      #",
             "#........      #",
             "#.....#..#  ...#",
             "#..... ###. ...#",
             "#.....X....T...#",
             "################"],
     "need": 10, "supply": 14, "digs": 6},
    # seed 4007; 10 expansions, 10-action plan
    {"rows": ["################",
             "#..S...........#",
             "#.. ......   ..#",
             "#.........   ..#",
             "#.........   ..#",
             "#.........#    #",
             "#.........##   #",
             "#.... .... #   #",
             "#....X....#..T.#",
             "################"],
     "need": 8, "supply": 14, "digs": 9},
    # seed 4008; 9 expansions, 9-action plan
    {"rows": ["################",
             "#..S...........#",
             "#.. .......#...#",
             "#........##....#",
             "#.........     #",
             "#........      #",
             "#........      #",
             "#.... ...      #",
             "#....X...   T  #",
             "################"],
     "need": 10, "supply": 13, "digs": 9},
    # seed 4009; 7 expansions, 7-action plan
    {"rows": ["################",
             "#  S  .   .....#",
             "#     .   .....#",
             "#     .   .....#",
             "#...#..        #",
             "#...#.....     #",
             "#.........     #",
             "#.... ..... ...#",
             "#....X.....T...#",
             "################"],
     "need": 12, "supply": 15, "digs": 8},
    # seed 4010; 7 expansions, 7-action plan
    {"rows": ["################",
             "#...S..........#",
             "#... ..........#",
             "#.......     ..#",
             "#.......     ..#",
             "#.......    #  #",
             "#........ ##   #",
             "#..... ..      #",
             "#.....X..    T #",
             "################"],
     "need": 9, "supply": 12, "digs": 7},
    # seed 4011; 8 expansions, 8-action plan
    {"rows": ["################",
             "#..S....      .#",
             "#.. ....      .#",
             "#....... #    .#",
             "#....... ##....#",
             "#....#..   #...#",
             "#....#.........#",
             "#...# ..... ...#",
             "#....X.....T...#",
             "################"],
     "need": 12, "supply": 17, "digs": 8},
    # seed 4012; 5 expansions, 5-action plan
    {"rows": ["################",
             "#.   S ..      #",
             "#.     ..      #",
             "#.     ..      #",
             "#.     ..      #",
             "#........      #",
             "#............#.#",
             "#....... .. ..##",
             "#.......X..T...#",
             "################"],
     "need": 12, "supply": 18, "digs": 8},
    # seed 4013; 7 expansions, 7-action plan
    {"rows": ["################",
             "#..S....   ....#",
             "#.. ....   ....#",
             "#.......####...#",
             "#..#....       #",
             "#..##...       #",
             "###......      #",
             "#..... .... ...#",
             "#.....X....T...#",
             "################"],
     "need": 9, "supply": 13, "digs": 8},
    # seed 4014; 9 expansions, 9-action plan
    {"rows": ["################",
             "#.S...   .     #",
             "#. ...   .   ###",
             "#.....   .   # #",
             "#.........  #  #",
             "#..#......   ###",
             "#..............#",
             "#....... ... ..#",
             "#.......X...T..#",
             "################"],
     "need": 10, "supply": 13, "digs": 9},
    # seed 4015; 8 expansions, 8-action plan
    {"rows": ["################",
             "#.S....     ...#",
             "#. ....     .###",
             "#......        #",
             "#......        #",
             "#........      #",
             "#..............#",
             "#... ...... ...#",
             "#...X......T...#",
             "################"],
     "need": 11, "supply": 17, "digs": 8},
    # seed 4016; 6 expansions, 6-action plan
    {"rows": ["################",
             "#....S.........#",
             "#.... ..     ..#",
             "#.......     ..#",
             "#.......     ..#",
             "#.......#......#",
             "#......#..#....#",
             "#......# #. ...#",
             "#.......X..T...#",
             "################"],
     "need": 10, "supply": 14, "digs": 6},
    # seed 4017; 4 expansions, 4-action plan
    {"rows": ["################",
             "#... S        .#",
             "#...          .#",
             "#...          .#",
             "#..............#",
             "#..........   .#",
             "#..........   .#",
             "#.....# ...   .#",
             "#....##X... T .#",
             "################"],
     "need": 12, "supply": 16, "digs": 7},
    # seed 4018; 7 expansions, 7-action plan
    {"rows": ["################",
             "#...S...   ....#",
             "#... ...      .#",
             "#.......      .#",
             "#......#.     .#",
             "#.......#     .#",
             "#......#.     .#",
             "#...... ..# ...#",
             "#......X..#T...#",
             "################"],
     "need": 11, "supply": 14, "digs": 7},
    # seed 4019; 6 expansions, 6-action plan
    {"rows": ["################",
             "#...S...   ....#",
             "#... ...       #",
             "#.....         #",
             "#.....         #",
             "#.##..     ....#",
             "##.............#",
             "#.#..... #.. ..#",
             "#.......X#..T..#",
             "################"],
     "need": 9, "supply": 14, "digs": 9},
    # seed 4020; 4 expansions, 4-action plan
    {"rows": ["################",
             "#.S.       ....#",
             "#. .       ....#",
             "#...        ...#",
             "#...        # .#",
             "#........  ## .#",
             "#.   ......   .#",
             "#.    .....   .#",
             "#.   X..... T .#",
             "################"],
     "need": 9, "supply": 12, "digs": 8},
    # seed 4021; 7 expansions, 7-action plan
    {"rows": ["################",
             "#.S   .........#",
             "#. #  .#.......#",
             "#..  ###  .....#",
             "#......   .....#",
             "#......   .....#",
             "#..............#",
             "#...... ... ...#",
             "#......X...T...#",
             "################"],
     "need": 11, "supply": 15, "digs": 8},
    # seed 4022; 7 expansions, 7-action plan
    {"rows": ["################",
             "#.  S..........#",
             "#.   ...     ..#",
             "#.   ...     ..#",
             "#.   ...     ..#",
             "#...........#..#",
             "#...........##.#",
             "#....... .. .###",
             "#.......X..T.###",
             "################"],
     "need": 12, "supply": 16, "digs": 6},
    # seed 4023; 10 expansions, 9-action plan
    {"rows": ["################",
             "#...S....   ...#",
             "#.#. ....   ...#",
             "#.#...... # ...#",
             "#.#........#...#",
             "##....    #    #",
             "#.....    #    #",
             "#.....         #",
             "#.......X# T   #",
             "################"],
     "need": 12, "supply": 16, "digs": 8},
    # seed 4024; 5 expansions, 5-action plan
    {"rows": ["################",
             "#....S.........#",
             "#.... .........#",
             "#.......     ..#",
             "###.....     ..#",
             "#..   ..     ..#",
             "#..   .........#",
             "#..   .. ..#. .#",
             "#..   ..X.##.T.#",
             "################"],
     "need": 12, "supply": 18, "digs": 8},
    # seed 4025; 8 expansions, 8-action plan
    {"rows": ["################",
             "#...S.....   ..#",
             "#.     ...   ..#",
             "#.     ...    .#",
             "#.     ....   .#",
             "#......#...   .#",
             "#.....#....   .#",
             "#..... ....   .#",
             "#.....X.....T..#",
             "################"],
     "need": 9, "supply": 13, "digs": 9},
    # seed 4026; 3 expansions, 3-action plan
    {"rows": ["################",
             "#.. S   .......#",
             "#..     .......#",
             "#..          ..#",
             "#.....       ..#",
             "#.....       ..#",
             "#..............#",
             "#.....#. ... ..#",
             "#......#X...T..#",
             "################"],
     "need": 11, "supply": 16, "digs": 8},
    # seed 4027; 4 expansions, 4-action plan
    {"rows": ["################",
             "#...S          #",
             "#...           #",
             "#...           #",
             "#.......#      #",
             "#......#       #",
             "#......#.......#",
             "#..... ..... ..#",
             "#....#X.....T..#",
             "################"],
     "need": 9, "supply": 14, "digs": 8},
    # seed 4028; 7 expansions, 7-action plan
    {"rows": ["################",
             "#...S...   ....#",
             "#... ...   ....#",
             "#.......     ..#",
             "#....#..     ..#",
             "#.....#.     ..#",
             "#     .#..   ..#",
             "#      ...    .#",
             "#     X......T.#",
             "################"],
     "need": 11, "supply": 14, "digs": 9},
    # seed 4029; 4 expansions, 4-action plan
    {"rows": ["################",
             "#....S.........#",
             "#.... ..     ..#",
             "#.......#     .#",
             "#....... #    .#",
             "#.......      .#",
             "#.......     ..#",
             "#.......    ...#",
             "#.......X..T...#",
             "################"],
     "need": 9, "supply": 15, "digs": 7},
    # seed 4030; 10 expansions, 9-action plan
    {"rows": ["################",
             "#....S.........#",
             "#.... .........#",
             "#..............#",
             "#..............#",
             "#..............#",
             "#.      #     .#",
             "#.      ###   .#",
             "#.     X  # T .#",
             "################"],
     "need": 11, "supply": 15, "digs": 8},
    # seed 4031; 6 expansions, 6-action plan
    {"rows": ["################",
             "#.S.......   ..#",
             "#. .......   ..#",
             "#..     ..   ..#",
             "#..   # ..   ..#",
             "#..     .......#",
             "#..............#",
             "#... ...... ...#",
             "#...X......T...#",
             "################"],
     "need": 9, "supply": 13, "digs": 7},
    # seed 4032; 7 expansions, 6-action plan
    {"rows": ["################",
             "#.S............#",
             "#. ..       ...#",
             "#....      #...#",
             "#....       #..#",
             "#..........#...#",
             "#.....       ..#",
             "#.....       ..#",
             "#.....  X  T ..#",
             "################"],
     "need": 11, "supply": 16, "digs": 7},
    # seed 4033; 9 expansions, 7-action plan
    {"rows": ["################",
             "#..S..     ....#",
             "#.. ..     ..#.#",
             "#.....     .#..#",
             "#..............#",
             "#      ........#",
             "#      ........#",
             "#      ..... ..#",
             "#....X......T..#",
             "################"],
     "need": 8, "supply": 14, "digs": 6},
    # seed 4034; 5 expansions, 5-action plan
    {"rows": ["################",
             "#.S  ..........#",
             "#.      .....#.#",
             "#. ##   ....#..#",
             "#....#  .......#",
             "#...#..........#",
             "#.........     #",
             "#...# ....     #",
             "#...#X....  T  #",
             "################"],
     "need": 12, "supply": 17, "digs": 6},
    # seed 4035; 11 expansions, 9-action plan
    {"rows": ["################",
             "#....S...     .#",
             "#.... ...     .#",
             "#........     .#",
             "#..............#",
             "#........   ...#",
             "#........   ...#",
             "#..##....   . .#",
             "#.###....X  .T.#",
             "################"],
     "need": 8, "supply": 12, "digs": 8},
    # seed 4036; 6 expansions, 6-action plan
    {"rows": ["################",
             "#.S.   ........#",
             "#. .       ....#",
             "#..#       ....#",
             "#..#       ....#",
             "#..#     ......#",
             "#.....   ......#",
             "#... .....#. ..#",
             "#...X......#T..#",
             "################"],
     "need": 9, "supply": 13, "digs": 6},
    # seed 4037; 7 expansions, 7-action plan
    {"rows": ["################",
             "#... S   ......#",
             "#...  #  ......#",
             "#... #   ....#.#",
             "#...   ......###",
             "#..............#",
             "#..............#",
             "#...... .... ..#",
             "#......X....T..#",
             "################"],
     "need": 9, "supply": 15, "digs": 9},
    # seed 4038; 68 expansions, 9-action plan
    {"rows": ["################",
             "# S       .....#",
             "#         .....#",
             "###       .....#",
             "# ##      .....#",
             "#     ...   ...#",
             "#........   ...#",
             "#... ....   . .#",
             "#...X........T.#",
             "################"],
     "need": 12, "supply": 16, "digs": 9},
    # seed 4039; 9 expansions, 9-action plan
    {"rows": ["################",
             "#.S      ......#",
             "#.      ##.....#",
             "#.       #.....#",
             "#.       ......#",
             "#.   #   ......#",
             "#.###..........#",
             "#..# ...... ...#",
             "#..#X......T...#",
             "################"],
     "need": 11, "supply": 16, "digs": 9},
    # seed 4040; 5 expansions, 5-action plan
    {"rows": ["################",
             "#..S...........#",
             "#.. ....       #",
             "#.....         #",
             "#.....#        #",
             "#..... #       #",
             "#....###       #",
             "#..... ...... .#",
             "#.....X......T.#",
             "################"],
     "need": 8, "supply": 11, "digs": 8},
    # seed 4041; 10 expansions, 10-action plan
    {"rows": ["################",
             "#..S...........#",
             "#.. ..   ......#",
             "#.....   ......#",
             "#.....   ......#",
             "#..............#",
             "#     .....#...#",
             "#     .... .. .#",
             "#     ....X..T.#",
             "################"],
     "need": 12, "supply": 15, "digs": 8},
    # seed 4042; 5 expansions, 5-action plan
    {"rows": ["################",
             "#.S     .......#",
             "#.      .......#",
             "#..     .......#",
             "#..#    .......#",
             "#..#    ...   .#",
             "#.#........   .#",
             "#..#... ...   .#",
             "#.#....X....T..#",
             "################"],
     "need": 9, "supply": 13, "digs": 8},
    # seed 4043; 4 expansions, 4-action plan
    {"rows": ["################",
             "#....S.........#",
             "#.... ..       #",
             "#.......       #",
             "#..#..#.       #",
             "#...##.... # ..#",
             "#....##...#  ..#",
             "#......# ..# ..#",
             "#.......X..#T..#",
             "################"],
     "need": 8, "supply": 11, "digs": 9},
    # seed 4044; 8 expansions, 8-action plan
    {"rows": ["################",
             "#....S.........#",
             "#...  #  ......#",
             "#... #   ......#",
             "#...     ......#",
             "#..............#",
             "#.   .#........#",
             "#.  ###.. ... .#",
             "#.   ##..X...T.#",
             "################"],
     "need": 12, "supply": 18, "digs": 9},
    # seed 4045; 6 expansions, 6-action plan
    {"rows": ["################",
             "#.. S      ....#",
             "#..        ....#",
             "#..        ....#",
             "#.##....   ....#",
             "#..............#",
             "#..............#",
             "#...... ... ...#",
             "#......X...T...#",
             "################"],
     "need": 12, "supply": 18, "digs": 7},
    # seed 4046; 6 expansions, 6-action plan
    {"rows": ["################",
             "#.S...     ....#",
             "#. ...      ...#",
             "#.....      ...#",
             "#.....      ...#",
             "#........   ...#",
             "#........   .#.#",
             "#...... ... ..##",
             "#......X...T...#",
             "################"],
     "need": 8, "supply": 13, "digs": 8},
    # seed 4047; 4 expansions, 4-action plan
    {"rows": ["################",
             "#..S...........#",
             "#.. .       ...#",
             "#...##      ...#",
             "#           ...#",
             "#      #    ...#",
             "#     #     ...#",
             "#.... #.... ...#",
             "#....X.....T...#",
             "################"],
     "need": 10, "supply": 15, "digs": 7},
    # seed 4048; 4 expansions, 4-action plan
    {"rows": ["################",
             "#..S...........#",
             "#.. .       ...#",
             "#....    #     #",
             "#....    #     #",
             "#......        #",
             "#..........   .#",
             "#.... .....   .#",
             "#....X.....  T.#",
             "################"],
     "need": 8, "supply": 11, "digs": 8},
    # seed 4049; 6 expansions, 5-action plan
    {"rows": ["################",
             "#....S.........#",
             "#.... ..     ..#",
             "#.......     ..#",
             "#.......     ..#",
             "#.......##...#.#",
             "#      ......#.#",
             "#      .. ... ##",
             "#      ..X...T.#",
             "################"],
     "need": 9, "supply": 13, "digs": 6},
    # seed 4050; 4 expansions, 4-action plan
    {"rows": ["################",
             "#.  S  ........#",
             "#.     ........#",
             "#. #   .       #",
             "#.  #  .       #",
             "#.......       #",
             "#......... #   #",
             "#..... ..##    #",
             "#.....X......T.#",
             "################"],
     "need": 11, "supply": 14, "digs": 7},
    # seed 4051; 3 expansions, 3-action plan
    {"rows": ["################",
             "#...S        ..#",
             "#...         ..#",
             "#.#..        ..#",
             "###.....   ....#",
             "#.......   ....#",
             "#..............#",
             "#...... .... ..#",
             "#......X....T..#",
             "################"],
     "need": 9, "supply": 12, "digs": 8},
    # seed 4053; 3 expansions, 3-action plan
    {"rows": ["################",
             "#..S.       ...#",
             "#.. .   ##  ...#",
             "#....    ## ...#",
             "#........   ...#",
             "#........#  ...#",
             "#........ # ...#",
             "#........ .# ..#",
             "#........X#.T..#",
             "################"],
     "need": 8, "supply": 12, "digs": 9},
    # seed 4054; 10 expansions, 10-action plan
    {"rows": ["################",
             "#.S.....   ....#",
             "#. .....   ....#",
             "#.......       #",
             "#.......       #",
             "#....##.       #",
             "#.....#...     #",
             "#.... ....     #",
             "#....X.......T.#",
             "################"],
     "need": 10, "supply": 15, "digs": 8},
    # seed 4055; 4 expansions, 4-action plan
    {"rows": ["################",
             "#..S...........#",
             "#.. ...........#",
             "#..       .....#",
             "#..           .#",
             "#..  ##       .#",
             "#....##       .#",
             "#.... .   . ...#",
             "#....X.....T...#",
             "################"],
     "need": 12, "supply": 18, "digs": 7},
    # seed 4056; 7 expansions, 7-action plan
    {"rows": ["################",
             "#..S..         #",
             "#.. ..  #      #",
             "#..... # ##    #",
             "#.....       ..#",
             "#..............#",
             "#....     ..#..#",
             "#....     ... .#",
             "#....    X...T.#",
             "################"],
     "need": 12, "supply": 17, "digs": 7},
    # seed 4057; 8 expansions, 8-action plan
    {"rows": ["################",
             "#.S............#",
             "#. ....       .#",
             "#......       .#",
             "#......  #  # .#",
             "#......##   # .#",
             "#......   ##  .#",
             "#......       .#",
             "#......X   T  .#",
             "################"],
     "need": 10, "supply": 14, "digs": 9},
    # seed 4058; 10 expansions, 10-action plan
    {"rows": ["################",
             "# S    ...   ..#",
             "#      ...   ..#",
             "# ##   ...  #..#",
             "##     .....#..#",
             "#.#..........#.#",
             "#.............##",
             "#..... ..... ..#",
             "#.....X.....T..#",
             "################"],
     "need": 12, "supply": 18, "digs": 8},
    # seed 4059; 8 expansions, 8-action plan
    {"rows": ["################",
             "#.S....       .#",
             "#. ....       .#",
             "#......       .#",
             "#...... #     .#",
             "#......  #    .#",
             "#........#.....#",
             "#........ #.. .#",
             "#........X#..T.#",
             "################"],
     "need": 9, "supply": 15, "digs": 8},
    # seed 4060; 5 expansions, 5-action plan
    {"rows": ["################",
             "#..S...........#",
             "#.. ...........#",
             "#.....     ....#",
             "#.....     ....#",
             "#.....         #",
             "#.........     #",
             "#...... #.     #",
             "#.....#X..  T  #",
             "################"],
     "need": 10, "supply": 13, "digs": 7},
    # seed 4061; 8 expansions, 8-action plan
    {"rows": ["################",
             "#...S...   ....#",
             "#... .      ...#",
             "#.....      ...#",
             "#.....##    ...#",
             "#..............#",
             "#..............#",
             "#........ ... .#",
             "#........X...T.#",
             "################"],
     "need": 11, "supply": 16, "digs": 8},
    # seed 4062; 5 expansions, 5-action plan
    {"rows": ["################",
             "#..S...........#",
             "#.. ..       ..#",
             "#.....         #",
             "#.....#        #",
             "#.....##       #",
             "#..##...#      #",
             "#.... ..       #",
             "#....X......T..#",
             "################"],
     "need": 8, "supply": 11, "digs": 9},
    # seed 4063; 43 expansions, 9-action plan
    {"rows": ["################",
             "# S   .........#",
             "#     .........#",
             "#     .....##..#",
             "#..........#...#",
             "#.......  #  ..#",
             "#.......     ..#",
             "#... ...     ..#",
             "#...X...    T..#",
             "################"],
     "need": 9, "supply": 14, "digs": 9},
    # seed 4064; 4 expansions, 4-action plan
    {"rows": ["################",
             "#.  S  ...     #",
             "#.     ...     #",
             "#.             #",
             "#...       ....#",
             "#...       ....#",
             "#.......   .##.#",
             "#..... .   .. ##",
             "#.....X......T.#",
             "################"],
     "need": 9, "supply": 14, "digs": 9},
    # seed 4065; 8 expansions, 8-action plan
    {"rows": ["################",
             "# S       .....#",
             "#         ..##.#",
             "#         ..#..#",
             "#         ..##.#",
             "#......   ...#.#",
             "#..............#",
             "#... .#..... ..#",
             "#...X..#....T..#",
             "################"],
     "need": 11, "supply": 17, "digs": 9},
    # seed 4066; 5 expansions, 5-action plan
    {"rows": ["################",
             "#.S............#",
             "#. ............#",
             "#..  ##   .....#",
             "#.. ##    .....#",
             "#..       .....#",
             "#..............#",
             "#..# #...... ..#",
             "#...X.......T..#",
             "################"],
     "need": 12, "supply": 17, "digs": 9},
    # seed 4067; 8 expansions, 8-action plan
    {"rows": ["################",
             "#.S.......   ..#",
             "#. ......     .#",
             "#.....##       #",
             "#......        #",
             "#......        #",
             "#......#       #",
             "#... ...#      #",
             "#...X......T...#",
             "################"],
     "need": 10, "supply": 15, "digs": 6},
    # seed 4068; 6 expansions, 6-action plan
    {"rows": ["################",
             "#..S..       ..#",
             "#.. ..       ..#",
             "#.....       ..#",
             "#.....       ..#",
             "#.....       ###",
             "#.......     ..#",
             "#.... .#      .#",
             "#....X..###..T.#",
             "################"],
     "need": 12, "supply": 17, "digs": 9},
    # seed 4069; 7 expansions, 7-action plan
    {"rows": ["################",
             "#....S.........#",
             "#.... #..   ...#",
             "#........   ...#",
             "#........   ...#",
             "#..............#",
             "#......      #.#",
             "#......   # # .#",
             "#......X  #  T.#",
             "################"],
     "need": 11, "supply": 15, "digs": 9},
    # seed 4070; 4 expansions, 4-action plan
    {"rows": ["################",
             "#..S.          #",
             "#.. .          #",
             "#....          #",
             "#..... ##      #",
             "#.....         #",
             "#..............#",
             "#...... ..... .#",
             "#......X.....T.#",
             "################"],
     "need": 11, "supply": 15, "digs": 7},
    # seed 4071; 5 expansions, 5-action plan
    {"rows": ["################",
             "#....S..       #",
             "#.... ..       #",
             "#.......       #",
             "#.........#....#",
             "#........#   ..#",
             "#.........#  ..#",
             "#....... . # ..#",
             "#.......X...T..#",
             "################"],
     "need": 12, "supply": 17, "digs": 6},
    # seed 4072; 7 expansions, 7-action plan
    {"rows": ["################",
             "# S    ........#",
             "#              #",
             "#              #",
             "#......        #",
             "#.......   ##  #",
             "#.......   #   #",
             "#....# ..... ..#",
             "#.....X#....T..#",
             "################"],
     "need": 10, "supply": 13, "digs": 6},
    # seed 4073; 9 expansions, 9-action plan
    {"rows": ["################",
             "#..S.......   .#",
             "#.. .......   .#",
             "#........     .#",
             "#......###  ...#",
             "#......#       #",
             "#......        #",
             "#.... .        #",
             "#....X....  T  #",
             "################"],
     "need": 11, "supply": 16, "digs": 8},
    # seed 4074; 3 expansions, 3-action plan
    {"rows": ["################",
             "#....S    .....#",
             "#....     .....#",
             "#....     #  ..#",
             "#.....    #  ..#",
             "#.....       ..#",
             "#..............#",
             "#...... ... ...#",
             "#......X...T...#",
             "################"],
     "need": 10, "supply": 15, "digs": 9},
    # seed 4075; 5 expansions, 5-action plan
    {"rows": ["################",
             "#...S.       ..#",
             "#...         ..#",
             "#...   #     ..#",
             "#...         ..#",
             "#..............#",
             "#..............#",
             "#..##..... .. .#",
             "#.........X..T.#",
             "################"],
     "need": 10, "supply": 15, "digs": 9},
    # seed 4076; 140 expansions, 5-action plan
    {"rows": ["################",
             "#....S      ...#",
             "#.... ##    ...#",
             "#....  #    ...#",
             "#....##     ...#",
             "#....       ...#",
             "#..#.....#.....#",
             "#.#.....# ... .#",
             "##.....#.X...T.#",
             "################"],
     "need": 8, "supply": 14, "digs": 7},
    # seed 4077; 4 expansions, 4-action plan
    {"rows": ["################",
             "#.S............#",
             "#. ............#",
             "#...     ......#",
             "#...     ......#",
             "#...##     ....#",
             "#...#      ....#",
             "#...       . ..#",
             "#...X.     .T..#",
             "################"],
     "need": 9, "supply": 12, "digs": 9},
    # seed 4078; 5 expansions, 5-action plan
    {"rows": ["################",
             "#....S.........#",
             "#.... .........#",
             "#.......     ..#",
             "#.......     ..#",
             "#.......     ..#",
             "#.....#........#",
             "#..#.#... ... .#",
             "#..#..##.X...T.#",
             "################"],
     "need": 12, "supply": 17, "digs": 6},
    # seed 4079; 12 expansions, 12-action plan
    {"rows": ["################",
             "#.S  ....     .#",
             "#.   ....     .#",
             "#.   ....     .#",
             "#.   ..........#",
             "#..............#",
             "###............#",
             "#.#.. ...... ..#",
             "#....X......T..#",
             "################"],
     "need": 12, "supply": 18, "digs": 9},
    # seed 4080; 4 expansions, 4-action plan
    {"rows": ["################",
             "#....S.........#",
             "#.... .........#",
             "#......   .....#",
             "#......        #",
             "#......        #",
             "#........  #   #",
             "#...... .   #  #",
             "#......X. #T   #",
             "################"],
     "need": 12, "supply": 16, "digs": 6},
    # seed 4081; 5 expansions, 5-action plan
    {"rows": ["################",
             "#.S............#",
             "#. .... #     .#",
             "#..     #     .#",
             "#..           .#",
             "#..           .#",
             "#.....        .#",
             "#... .   ... ..#",
             "#...X.   ...T..#",
             "################"],
     "need": 10, "supply": 14, "digs": 7},
    # seed 4082; 7 expansions, 7-action plan
    {"rows": ["################",
             "#....S.........#",
             "#.... .......###",
             "#..............#",
             "#..............#",
             "#........     .#",
             "#.......# #   .#",
             "#....... #    .#",
             "#.......X  T ..#",
             "################"],
     "need": 9, "supply": 13, "digs": 8},
    # seed 4083; 9 expansions, 9-action plan
    {"rows": ["################",
             "#.S.....       #",
             "#. .....       #",
             "#.......       #",
             "#.....      ...#",
             "#.....      #..#",
             "#.....     ..###",
             "#... ........ .#",
             "#...X........T.#",
             "################"],
     "need": 8, "supply": 13, "digs": 8},
    # seed 4084; 9 expansions, 8-action plan
    {"rows": ["################",
             "#...S...   ....#",
             "#... ..#   ....#",
             "#.#.....#      #",
             "#.#....##      #",
             "#.##....       #",
             "#..#..... ##   #",
             "#..... ..      #",
             "#.....X.....T..#",
             "################"],
     "need": 9, "supply": 13, "digs": 9},
    # seed 4085; 6 expansions, 6-action plan
    {"rows": ["################",
             "#.S  ..........#",
             "#.       ......#",
             "#.       ......#",
             "#...     ......#",
             "#...#..#.......#",
             "#..#.....#.....#",
             "#.#.. ..#... ..#",
             "#....X...#..T..#",
             "################"],
     "need": 9, "supply": 15, "digs": 7},
    # seed 4086; 7 expansions, 7-action plan
    {"rows": ["################",
             "#.S............#",
             "#. ............#",
             "#..... # ......#",
             "#.....  #      #",
             "#...#.  #  ##  #",
             "#..##...    #  #",
             "#.... ..   ##  #",
             "#....X..     T #",
             "################"],
     "need": 10, "supply": 14, "digs": 6},
    # seed 4087; 11 expansions, 8-action plan
    {"rows": ["################",
             "#..S....     ..#",
             "#.. ....    #  #",
             "#.......   #   #",
             "#.......       #",
             "#....          #",
             "#.##           #",
             "##          ...#",
             "#.   X     T...#",
             "################"],
     "need": 9, "supply": 15, "digs": 8},
    # seed 4088; 4 expansions, 4-action plan
    {"rows": ["################",
             "#...S...       #",
             "#... ...       #",
             "#..            #",
             "#..       ##   #",
             "#..       ##...#",
             "#..............#",
             "#...... ... ...#",
             "#......X...T...#",
             "################"],
     "need": 10, "supply": 14, "digs": 7},
    # seed 4089; 3 expansions, 3-action plan
    {"rows": ["################",
             "#....S.........#",
             "#.... ....#....#",
             "#.... ##  #    #",
             "#....      #   #",
             "#.##.          #",
             "#...#..........#",
             "#...... .... ..#",
             "#......X....T..#",
             "################"],
     "need": 8, "supply": 12, "digs": 7},
    # seed 4090; 37 expansions, 11-action plan
    {"rows": ["################",
             "#.S.....   ....#",
             "#. .....   ....#",
             "#.......# ##   #",
             "#.......##     #",
             "#..   ..       #",
             "#..   ..       #",
             "#..   ..       #",
             "#....X.... T   #",
             "################"],
     "need": 11, "supply": 16, "digs": 9},
    # seed 4091; 3 expansions, 3-action plan
    {"rows": ["################",
             "#....S   ......#",
             "#....    ......#",
             "#.....####  ...#",
             "#.....       ..#",
             "#.....       ..#",
             "#....      ##..#",
             "#....        ..#",
             "#....  X   T ..#",
             "################"],
     "need": 8, "supply": 14, "digs": 6},
    # seed 4092; 3 expansions, 3-action plan
    {"rows": ["################",
             "#..S           #",
             "#..            #",
             "#..            #",
             "#..            #",
             "#..            #",
             "#..............#",
             "#...... #... ..#",
             "#......X....T..#",
             "################"],
     "need": 9, "supply": 15, "digs": 9},
    # seed 4093; 3 expansions, 3-action plan
    {"rows": ["################",
             "#.. S      ....#",
             "#..        ....#",
             "#..     ## ....#",
             "#......       .#",
             "#......       .#",
             "#......       .#",
             "#....# .    ...#",
             "#.....X#...T...#",
             "################"],
     "need": 8, "supply": 13, "digs": 7},
    # seed 4094; 9 expansions, 8-action plan
    {"rows": ["################",
             "#.S............#",
             "#. ............#",
             "#.....       ..#",
             "#.....       ..#",
             "#.....       ..#",
             "#.......#    ..#",
             "#....... #   ..#",
             "#.......X..T...#",
             "################"],
     "need": 12, "supply": 15, "digs": 8},
    # seed 4095; 7 expansions, 7-action plan
    {"rows": ["################",
             "#..S...........#",
             "#.   .#........#",
             "#.   .#........#",
             "#.   ..........#",
             "#.....#   # ...#",
             "#....#.   ##...#",
             "#.... .     . .#",
             "#....X.......T.#",
             "################"],
     "need": 12, "supply": 18, "digs": 9},
    # seed 4096; 4 expansions, 4-action plan
    {"rows": ["################",
             "#...S.       ..#",
             "#... .   #   ..#",
             "#..##.   #   ..#",
             "#...#... #   ..#",
             "#.......     ..#",
             "#......#.     .#",
             "#..... ..     .#",
             "#.....X..   T .#",
             "################"],
     "need": 12, "supply": 17, "digs": 6},
    # seed 4097; 5 expansions, 5-action plan
    {"rows": ["################",
             "#...S..       .#",
             "#... ..       .#",
             "#...##.       .#",
             "#......       .#",
             "#......       .#",
             "#............#.#",
             "#........ .. ..#",
             "#........X..T#.#",
             "################"],
     "need": 11, "supply": 17, "digs": 8},
    # seed 4098; 10 expansions, 6-action plan
    {"rows": ["################",
             "#...S    ......#",
             "#...     ......#",
             "#... ### ......#",
             "#...  #  ......#",
             "#....##........#",
             "#.....   ......#",
             "#.....    .. #.#",
             "#.....   X..T#.#",
             "################"],
     "need": 9, "supply": 15, "digs": 8},
    # seed 4099; 6 expansions, 6-action plan
    {"rows": ["################",
             "#..S   ........#",
             "#..         ...#",
             "#...#       ...#",
             "#..##     ##...#",
             "#..#.......#...#",
             "#..#...........#",
             "#....... ..# ..#",
             "#.......X..#T..#",
             "################"],
     "need": 11, "supply": 17, "digs": 8},
    # seed 4100; 6 expansions, 6-action plan
    {"rows": ["################",
             "#....S.........#",
             "#.          ...#",
             "#.          ...#",
             "#.  #       ...#",
             "#...#.......##.#",
             "#..............#",
             "###.... ... ...#",
             "#......X...T...#",
             "################"],
     "need": 11, "supply": 16, "digs": 6},
)
