"""A Pipe Dream-like: pipe pieces arrive one at a time from a queue, the player lays each on a
board ahead of a flow that starts after a countdown and then advances a cell at a time, and the
level is won when the flow has run through the distance it asks for before spilling out of an
open end.

The game is The Assembly Line's *Pipe Mania* (1989), which Lucasfilm Games published as *Pipe
Dream* and Bullet-Proof Software brought to the Game Boy in 1990. What is here is a
re-implementation of its rules, written from the game as played rather than from its program,
at the granularity a planner can use: a placement is one tick, the flow advances every `pace`
ticks once the `countdown` has run, and `flush` lets it run to the end. The queue is fixed by
the instance, so the whole future of the pieces is known and the problem is deterministic; the
difficulty is that a piece has to be laid before the flow reaches its cell, and a piece that
does not fit where the flow is going has to be laid where it will fit later or discarded, at a
tick's cost either way. Nothing of the
original's code, art or level data is here, and the environment needs no dependency.

## Instances and generation

`generate_instance(seed, ...)` draws a board with a few walls, a start piece, a queue of pieces
and a distance, and keeps the draw only if a thoughtless policy (lay the piece at the end of
the pipe laid when it fits there, else discard it) fails and a best-first search over
placements, guided by the distance still to carry less the pipe already laid ahead of the flow
with a point against every stray piece, finds a plan within the budget; the plan is left in
`witness`. The bundled levels are such
draws, embedded as plain data with the seed each came from. The method is generate-and-test
(search-based procedural content generation: Togelius et al. 2011,
https://doi.org/10.1109/TCIAIG.2011.2148116).
"""
from planiverse.environments.base import Environment
from planiverse.environments.generation import bounded_search, draw_until, rng

WIDTH, HEIGHT = 7, 7
EMPTY, WALL = ".", "#"

#: The pieces: a glyph on the board, a name for actions and literals, and the sides it opens.
N, S, E, W = "n", "s", "e", "w"
DELTA = {N: (0, -1), S: (0, 1), E: (1, 0), W: (-1, 0)}
OPPOSITE = {N: S, S: N, E: W, W: E}
AXIS = {N: "ns", S: "ns", E: "ew", W: "ew"}
KINDS = {"ns": frozenset("ns"), "ew": frozenset("ew"), "ne": frozenset("ne"),
         "nw": frozenset("nw"), "se": frozenset("se"), "sw": frozenset("sw"),
         "x": frozenset("nsew")}
GLYPH = {"ns": "|", "ew": "-", "ne": "L", "nw": "J", "se": "r", "sw": "7", "x": "+"}
KIND = {glyph: kind for kind, glyph in GLYPH.items()}
CROSS = "x"
#: The start piece, drawn as the direction the flow leaves it in.
START = {">": E, "<": W, "^": N, "v": S}
#: How the queue is weighted: straights most, the cross least, as the original deals them.
WEIGHTS = {"ns": 3, "ew": 3, "ne": 2, "nw": 2, "se": 2, "sw": 2, "x": 1}
#: Ticks a placement takes, and a replacement of a piece already laid.
PLACE_TICKS, REPLACE_TICKS = 1, 2
#: How the board is drawn: single lines for pipe laid, double lines for pipe the flow has run.
DRAW = {"ns": "│", "ew": "─", "ne": "└", "nw": "┘", "se": "┌", "sw": "┐", "x": "┼"}
DRAW_FLOODED = {"ns": "║", "ew": "═", "ne": "╚", "nw": "╝", "se": "╔", "sw": "╗", "x": "╬"}


class PipeAction:
    """`place(x, y)` lays the next piece of the queue at `(x, y)`; `discard` lays it out of
    the way, on a cell that will never matter; `flush` lets the flow run to the end."""

    def __init__(self, x=None, y=None, name=None):
        self.x, self.y = x, y
        self.name = name or (f"place({x},{y})" if x is not None else "flush")
        if self.name not in ("flush", "discard") and x is None:
            raise ValueError("a placement needs a cell")

    @classmethod
    def parse(cls, text):
        text = str(text).strip()
        if text in ("flush", "discard"):
            return cls(name=text)
        x, y = (int(part) for part in text[len("place("):-1].split(","))
        return cls(x, y)

    def cost(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, PipeAction) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    __repr__ = __str__


FLUSH = PipeAction(name="flush")
DISCARD = PipeAction(name="discard")


class PipeState:
    """The board as a tuple of row strings, the cells the flow has run through (with the axis
    it ran along, so a cross may be crossed twice), the flow's head, the segments carried, the
    tick, the pieces of the queue used, and whether the flow has spilled."""

    def __init__(self, rows, flooded, head, carried, tick, placed, spilled, queue=(),
                 distance=0, countdown=0, pace=1, depth=0):
        self.rows = tuple(rows)
        self.flooded = frozenset(flooded)
        self.head = tuple(head)
        self.carried, self.tick, self.placed, self.spilled = carried, tick, placed, spilled
        self.queue, self.distance, self.countdown, self.pace = tuple(queue), distance, countdown, pace
        self.depth = depth
        self.ahead, self.reach = run_ahead(self.rows, self.flooded, self.head,
                                           max(0, distance - carried))
        literals = [f"pipe({x}, {y}, {KIND[cell]})" for y, row in enumerate(self.rows)
                    for x, cell in enumerate(row) if cell in KIND]
        literals += [f"flooded({x}, {y})" for x, y in {(x, y) for x, y, _ in self.flooded}]
        literals += [f"head({self.head[0]}, {self.head[1]}, {self.head[2]})",
                     f"carried({carried})", f"due({self.due})"]
        literals.append(f"next({self.next})" if self.next else "queue_empty")
        if spilled:
            literals.append("spilled")
        self.literals = frozenset(literals)

    @property
    def stray(self):
        """Pipe laid that the flow has neither run through nor will reach as things stand."""
        wet = {(x, y) for x, y, _ in self.flooded}
        return sum(1 for y, row in enumerate(self.rows) for x, cell in enumerate(row)
                   if cell in KIND and (x, y) not in wet and (x, y) not in self.reach)

    @property
    def next(self):
        """The kind of the next piece to lay, or None once the queue is used up."""
        return self.queue[self.placed] if self.placed < len(self.queue) else None

    @property
    def due(self):
        """Ticks until the flow next advances: the countdown left, then the pace."""
        if self.tick < self.countdown:
            return self.countdown - self.tick
        return self.pace - (self.tick - self.countdown) % self.pace

    def __eq__(self, other):
        return (isinstance(other, PipeState) and self.rows == other.rows
                and self.flooded == other.flooded and self.head == other.head
                and self.carried == other.carried and self.tick == other.tick
                and self.placed == other.placed and self.spilled == other.spilled)

    def __hash__(self):
        return hash((self.rows, self.flooded, self.head, self.carried, self.tick, self.placed,
                     self.spilled))

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        wet = {(x, y) for x, y, _ in self.flooded}
        lines = []
        for y, row in enumerate(self.rows):
            line = ""
            for x, cell in enumerate(row):
                if cell in KIND:
                    line += (DRAW_FLOODED if (x, y) in wet else DRAW)[KIND[cell]]
                else:
                    line += cell
            lines.append(line)
        coming = " ".join(DRAW[kind] for kind in self.queue[self.placed:self.placed + 5]) or "none"
        status = ("spilled" if self.spilled else
                  f"{self.due} tick{'s' if self.due != 1 else ''} before the flow moves")
        lines.append(f"carried {self.carried} of {self.distance}; next {coming}; {status}")
        return "\n".join(lines)

    def __repr__(self):
        return (f"<PipeState(carried={self.carried}/{self.distance}, tick={self.tick}, "
                f"placed={self.placed}, spilled={self.spilled})>")


# --------------------------------------------------------------------------------- the flow

def advance(rows, flooded, head):
    """One step of the flow from `head` `(x, y, direction)`. Returns the new flooded set and
    head, or `None` when the flow spills: off the board, into an empty cell or a wall, into a
    piece with no opening on that side, or back into pipe it has already run through."""
    x, y, direction = head
    dx, dy = DELTA[direction]
    nx, ny = x + dx, y + dy
    if not (0 <= ny < len(rows) and 0 <= nx < len(rows[ny])):
        return None
    cell = rows[ny][nx]
    if cell not in KIND:
        return None
    kind = KIND[cell]
    entry = OPPOSITE[direction]
    if entry not in KINDS[kind]:
        return None
    axis = AXIS[direction]
    if (nx, ny, axis) in flooded:
        return None
    if kind != CROSS and any(fx == nx and fy == ny for fx, fy, _ in flooded):
        return None
    exit_ = direction if kind == CROSS else next(iter(KINDS[kind] - {entry}))
    return flooded | {(nx, ny, axis)}, (nx, ny, exit_)


def run_ahead(rows, flooded, head, limit):
    """How many segments the flow would still run through the pipe already laid, up to
    `limit`, and the cells it would run through."""
    count, cells = 0, set()
    while count < limit:
        step = advance(rows, flooded, head)
        if step is None:
            break
        flooded, head = step
        cells.add(head[:2])
        count += 1
    return count, frozenset(cells)


def end_of_pipe(state):
    """Where the flow will be once it has run the pipe already laid: the last head."""
    flooded, head = state.flooded, state.head
    for _ in range(state.ahead):
        flooded, head = advance(state.rows, flooded, head)
    return head


def advances_due(tick, countdown, pace):
    """How many times the flow has advanced by `tick`: none during the countdown, then once
    every `pace` ticks."""
    if tick < countdown:
        return 0
    return 1 + (tick - countdown) // pace


# ------------------------------------------------------------------------------- the levels

def draw_level(random_, width=WIDTH, height=HEIGHT, distance=None, countdown=None, pace=3,
               walls=None):
    """A random level as the instance dict `set_instance` takes: a board with a few walls, a
    start piece with room ahead of it, a weighted queue of pieces, and the distance to carry."""
    grid = [[EMPTY] * width for _ in range(height)]
    for _ in range(random_.randint(0, 3) if walls is None else walls):
        grid[random_.randrange(height)][random_.randrange(width)] = WALL
    while True:
        glyph = random_.choice(list(START))
        sx, sy = random_.randrange(width), random_.randrange(height)
        dx, dy = DELTA[START[glyph]]
        ahead = [(sx + k * dx, sy + k * dy) for k in range(1, 3)]
        if all(0 <= ax < width and 0 <= ay < height and grid[ay][ax] == EMPTY for ax, ay in ahead):
            break
    grid[sy][sx] = glyph
    distance = distance or random_.randint(8, 12)
    countdown = countdown or random_.randint(5, 8)
    kinds, weights = zip(*WEIGHTS.items())
    queue = random_.choices(kinds, weights=weights, k=countdown + 3 * distance)
    return {"rows": ["".join(row) for row in grid], "queue": list(queue),
            "distance": distance, "countdown": countdown, "pace": pace}


class PipeDreamEnv(Environment):
    """Lay the queue's pieces ahead of the flow so that it runs the level's distance."""

    def __init__(self):
        super().__init__("pipe_dream")
        self.instance = None
        self.index = None
        self.state = None
        self.state_history = []
        self.witness = None
        self.witness_expansions = None

    # ------------------------------------------------------------------ instances

    def set_index(self, index):
        if not 0 <= index < len(LEVELS):
            raise IndexError(f"Invalid index: {index}. There are {len(LEVELS)} levels, so the "
                             f"index must be 0-{len(LEVELS) - 1}.")
        self.set_instance(LEVELS[index])
        self.index = index

    def set_instance(self, instance):
        """Select a level: `{"rows": [...], "queue": [kinds...], "distance": d, "countdown": c,
        "pace": p}`. The rows hold `.` for a free cell, `#` for a wall, and one start piece
        drawn as `>`, `<`, `^` or `v`, the way the flow leaves it."""
        for key in ("rows", "queue", "distance"):
            if key not in instance:
                raise ValueError(f"a level needs `{key}`")
        rows = [str(row) for row in instance["rows"]]
        if len({len(row) for row in rows}) != 1:
            raise ValueError("a level's rows are all the same width")
        starts = [(x, y) for y, row in enumerate(rows) for x, cell in enumerate(row) if cell in START]
        if len(starts) != 1:
            raise ValueError("a level has exactly one start piece")
        for cell in "".join(rows):
            if cell not in (EMPTY, WALL) and cell not in START and cell not in KIND:
                raise ValueError(f"unknown cell {cell!r}")
        queue = [str(kind) for kind in instance["queue"]]
        for kind in queue:
            if kind not in KINDS:
                raise ValueError(f"unknown piece {kind!r}; the kinds are {sorted(KINDS)}")
        self.instance = {"rows": rows, "queue": queue, "distance": int(instance["distance"]),
                         "countdown": int(instance.get("countdown", 5)),
                         "pace": max(1, int(instance.get("pace", 2)))}
        self.index = None
        self.witness = self.witness_expansions = None

    def generate_instance(self, seed=None, width=WIDTH, height=HEIGHT, distance=None,
                          countdown=None, pace=3, walls=None, min_plan_length=6,
                          search_limit=600, attempts=40):
        """Draw a level, select it, and return it as the dict `set_instance` takes.

        `width`, `height`, `distance` (8 to 12 when unset), `countdown` (5 to 8), `pace` and
        `walls` (0 to 3) are `draw_level`'s. A draw is kept only if the thoughtless policy
        (`greedy_plan`) fails and a best-first search over placements, guided by the distance
        still to carry less the pipe already laid ahead of the flow, with a point against every
        stray piece, finds a plan of at least `min_plan_length` actions within `search_limit`
        expansions; that plan is left in `witness` and what the search spent in
        `witness_expansions`.
        """
        random_, _ = rng(seed)
        found = {}

        def draw(attempt):
            return draw_level(random_, width, height, distance, countdown, pace, walls)

        def accept(instance):
            self.set_instance(instance)
            if self.validate(self.greedy_plan()):
                return False
            outcome = bounded_search(self, search_limit, progress=self.__progress__)
            if outcome.plan is None or len(outcome.plan) < min_plan_length:
                return False
            found["plan"], found["expansions"] = outcome.plan, outcome.expansions
            return True

        instance = draw_until(draw, accept, attempts, "pipe dream level")
        self.set_instance(instance)
        self.witness, self.witness_expansions = found["plan"], found["expansions"]
        return instance

    def greedy_plan(self):
        """The thoughtless policy: lay each piece at the end of the pipe already laid when it
        fits there, else discard it, and flush once the pipe laid reaches the distance or the
        queue is used up."""
        state, _ = self.reset()
        plan = []
        while not self.is_goal(state) and not self.is_terminal(state):
            if state.next is None or state.carried + state.ahead >= state.distance:
                action = FLUSH
            else:
                action = DISCARD
                end = end_of_pipe(state)
                dx, dy = DELTA[end[2]]
                target = (end[0] + dx, end[1] + dy)
                if self.__free__(state, *target) and state.rows[target[1]][target[0]] == EMPTY:
                    laid = self.__lay__(state.rows, target, state.next)
                    if run_ahead(laid, state.flooded, state.head, state.ahead + 1)[0] > state.ahead:
                        action = PipeAction(*target)
            plan.append(action)
            state = self.__advance__(state, action)
        return plan

    def __progress__(self, state):
        """The distance still to carry, less the pipe already laid ahead of the flow, with a
        point against every stray piece so that a piece is discarded rather than dumped."""
        return 10 * max(0, state.distance - state.carried - state.ahead) + state.stray

    # ------------------------------------------------------------------- contract

    def reset(self):
        if self.instance is None:
            self.set_index(0)
        rows = self.instance["rows"]
        (sx, sy), glyph = next(((x, y), cell) for y, row in enumerate(rows)
                               for x, cell in enumerate(row) if cell in START)
        self.state = PipeState(rows, (), (sx, sy, START[glyph]), 0, 0, 0, False,
                               self.instance["queue"], self.instance["distance"],
                               self.instance["countdown"], self.instance["pace"])
        self.state_history = [self.state]
        return self.state, {"level": self.index, "distance": self.instance["distance"],
                            "countdown": self.instance["countdown"], "pace": self.instance["pace"],
                            "queue": len(self.instance["queue"]), "generated": self.index is None}

    def is_goal(self, state):
        return state.carried >= state.distance

    def is_terminal(self, state):
        return state.spilled and not self.is_goal(state)

    def get_actions(self, state=None):
        state = state or self.state
        actions = []
        if state.next is not None:
            actions = [PipeAction(x, y) for y, row in enumerate(state.rows)
                       for x, cell in enumerate(row) if self.__free__(state, x, y)]
            actions.append(DISCARD)
        return actions + [FLUSH]

    def successors(self, state):
        if self.is_goal(state) or self.is_terminal(state):
            return []
        children = []
        for action in self.get_actions(state):
            child = self.__advance__(state, action)
            if child != state:
                children.append((action, child))
        return children

    @staticmethod
    def __free__(state, x, y):
        """A cell a piece may be laid on: empty, or pipe the flow has not run through."""
        if not (0 <= y < len(state.rows) and 0 <= x < len(state.rows[y])):
            return False
        cell = state.rows[y][x]
        if cell == EMPTY:
            return True
        return cell in KIND and not any(fx == x and fy == y for fx, fy, _ in state.flooded)

    @staticmethod
    def __lay__(rows, cell, kind):
        x, y = cell
        row = list(rows[y])
        row[x] = GLYPH[kind]
        return rows[:y] + ("".join(row),) + rows[y + 1:]

    def __advance__(self, state, action):
        if self.is_goal(state) or self.is_terminal(state):
            return state
        if not isinstance(action, PipeAction):
            action = PipeAction.parse(action)
        rows, flooded, head = state.rows, state.flooded, state.head
        carried, tick, placed, spilled = state.carried, state.tick, state.placed, False
        if action.name == "flush":
            due = state.distance - carried
        elif action.name == "discard":
            if state.next is None:
                return state
            placed += 1
            tick += PLACE_TICKS
            due = advances_due(tick, state.countdown, state.pace) - carried
        else:
            if state.next is None or not self.__free__(state, action.x, action.y):
                return state
            replacing = rows[action.y][action.x] != EMPTY
            rows = self.__lay__(rows, (action.x, action.y), state.next)
            placed += 1
            tick += REPLACE_TICKS if replacing else PLACE_TICKS
            due = advances_due(tick, state.countdown, state.pace) - carried
        while due > 0 and carried < state.distance:
            step = advance(rows, flooded, head)
            if step is None:
                spilled = True
                break
            flooded, head = step
            carried += 1
            due -= 1
        return PipeState(rows, flooded, head, carried, tick, placed, spilled, state.queue,
                         state.distance, state.countdown, state.pace, state.depth + 1)

    def simulate(self, plan):
        state, _ = self.reset()
        trace = [state]
        for action in plan:
            trace.append(self.__advance__(trace[-1], action))
        return trace

    def step(self, action):
        if self.state is None:
            raise ValueError("call reset() first")
        before = self.state.carried
        self.state = self.__advance__(self.state, action)
        self.state_history.append(self.state)
        return self.state, self.state.carried - before

    def render(self):
        lines = [f"step {k}:\n{state}" for k, state in enumerate(self.state_history)]
        print("\n".join(lines))
        return lines


#: The bundled levels: `generate_instance(seed)` for the seed beside each, embedded as the
#: plain data `set_instance` takes, with the plan each was accepted on in
#: `tests/data/pipe_dream_solutions.json`.
LEVELS = (
    # seed 11000; 36 expansions, 24-action plan
    {"rows": ["....#..",
             ".......",
             "....#..",
             ".......",
             "..#...^",
             ".......",
             "......."],
     "queue": ["x", "ew", "x", "se", "ne", "sw", "sw", "se", "ew", "nw", "se", "nw", "x", "se", "se", "x", "nw", "ew", "se", "ew", "nw", "nw", "ne", "x", "sw", "ne", "nw", "ew", "se", "sw", "sw", "ew", "ns", "ne", "ew", "ew", "ne", "ew", "se"],
     "distance": 11, "countdown": 6, "pace": 3},
    # seed 11001; 34 expansions, 24-action plan
    {"rows": [".......",
             ".......",
             ".....<.",
             "...#...",
             ".......",
             "...#...",
             "......."],
     "queue": ["se", "ew", "sw", "ew", "sw", "ew", "nw", "se", "ew", "ne", "ns", "sw", "x", "ew", "se", "sw", "ew", "ew", "se", "ns", "se", "ns", "ew", "ne", "nw", "ns", "ns", "ns", "ew", "ns", "ew", "se", "nw", "ew", "ns", "ne"],
     "distance": 10, "countdown": 6, "pace": 3},
    # seed 11002; 22 expansions, 19-action plan
    {"rows": [".......",
             "......v",
             ".#.....",
             ".......",
             ".......",
             ".......",
             "......."],
     "queue": ["ne", "ns", "ns", "ne", "se", "ns", "ne", "ew", "nw", "ew", "ns", "se", "nw", "ns", "sw", "nw", "x", "ew", "se", "sw", "ns", "ne", "nw", "ns", "ew", "sw", "se", "sw", "ns", "nw", "ew", "ns", "ns"],
     "distance": 9, "countdown": 6, "pace": 3},
    # seed 11003; 34 expansions, 28-action plan
    {"rows": [".......",
             ".....#.",
             ".......",
             ".......",
             "v......",
             ".......",
             "......."],
     "queue": ["ns", "ew", "ns", "sw", "sw", "ew", "sw", "ew", "ne", "ne", "sw", "ew", "ns", "ew", "ne", "ew", "sw", "sw", "sw", "ew", "ew", "ns", "ne", "ne", "nw", "nw", "ew", "ns", "nw", "ne", "nw", "ew", "ns", "ew"],
     "distance": 9, "countdown": 7, "pace": 3},
    # seed 11004; 61 expansions, 26-action plan
    {"rows": [".......",
             "....#..",
             ".......",
             "......<",
             ".......",
             ".......",
             "...#..."],
     "queue": ["x", "nw", "x", "nw", "nw", "ew", "ew", "nw", "ne", "ne", "x", "sw", "ns", "ew", "ne", "ns", "ne", "sw", "ns", "ew", "sw", "ew", "se", "ne", "nw", "ns", "ne", "sw", "ew", "ns", "sw", "ew", "nw", "ns", "ns", "sw", "ew"],
     "distance": 10, "countdown": 7, "pace": 3},
    # seed 11005; 23 expansions, 21-action plan
    {"rows": [".......",
             ".......",
             ".......",
             ".......",
             "v......",
             "...#...",
             "......."],
     "queue": ["ew", "ew", "se", "ne", "nw", "se", "nw", "ns", "ew", "ns", "nw", "ew", "ns", "x", "se", "ew", "ew", "ns", "ne", "ew", "nw", "nw", "ew", "x", "se", "ne", "se", "nw", "sw", "sw", "sw", "ne", "nw", "ew", "nw", "x", "se", "ew", "ne", "ew"],
     "distance": 11, "countdown": 7, "pace": 3},
    # seed 11006; 28 expansions, 22-action plan
    {"rows": [".....#.",
             "#......",
             ".......",
             ".....<.",
             ".......",
             ".......",
             "......."],
     "queue": ["sw", "ew", "se", "ew", "ew", "nw", "ew", "ne", "ns", "se", "ns", "ns", "nw", "ew", "ew", "ne", "sw", "nw", "x", "ne", "x", "ew", "ns", "sw", "sw", "se", "sw", "se", "ne", "se", "ew", "ne", "ns", "ew", "sw", "ns", "ns", "se", "se"],
     "distance": 11, "countdown": 6, "pace": 3},
    # seed 11007; 52 expansions, 28-action plan
    {"rows": [".......",
             ".......",
             ".......",
             ".......",
             ">......",
             ".......",
             "......."],
     "queue": ["ne", "x", "se", "se", "sw", "nw", "x", "x", "ns", "se", "x", "sw", "ew", "nw", "nw", "x", "ne", "ew", "se", "ns", "x", "nw", "ew", "ne", "nw", "se", "x", "se", "ew", "ns", "x", "nw", "ew", "sw", "se", "se", "x", "sw", "ns", "ew", "se"],
     "distance": 12, "countdown": 5, "pace": 3},
    # seed 11008; 307 expansions, 28-action plan
    {"rows": ["#......",
             "......#",
             ".......",
             ".......",
             "..v....",
             ".......",
             "#......"],
     "queue": ["ne", "ne", "ne", "nw", "nw", "ne", "ne", "ne", "ns", "se", "nw", "se", "se", "ew", "se", "sw", "ew", "nw", "ns", "ne", "ns", "sw", "x", "se", "nw", "ns", "ne", "ne", "ns", "se", "ew", "ew", "se", "ew", "ew", "sw", "sw", "sw", "ew", "ns", "ew"],
     "distance": 12, "countdown": 5, "pace": 3},
    # seed 11009; 33 expansions, 24-action plan
    {"rows": [".......",
             ".......",
             ".......",
             ".......",
             "..<....",
             ".......",
             "......."],
     "queue": ["ew", "ew", "sw", "se", "ew", "se", "se", "ew", "ns", "se", "ne", "ne", "sw", "ew", "ew", "ns", "ew", "ns", "nw", "ne", "nw", "sw", "ew", "ns", "sw", "ew", "nw", "se", "sw", "nw", "ns", "ew", "x", "ew", "ns", "se"],
     "distance": 10, "countdown": 6, "pace": 3},
    # seed 11010; 27 expansions, 22-action plan
    {"rows": [".......",
             ".......",
             "......<",
             ".......",
             ".......",
             ".......",
             ".#.#..."],
     "queue": ["nw", "se", "ne", "se", "ns", "ns", "se", "nw", "nw", "ew", "ew", "ew", "se", "ew", "ne", "sw", "sw", "ew", "ew", "ns", "ns", "ns", "sw", "x", "ew", "sw", "ew", "x", "se", "ew", "ns", "ew", "ew", "nw", "x", "ns", "ew", "sw", "nw", "sw"],
     "distance": 11, "countdown": 7, "pace": 3},
    # seed 11011; 54 expansions, 27-action plan
    {"rows": [".......",
             ".......",
             ".......",
             "..#....",
             "^......",
             "...#...",
             "......."],
     "queue": ["ns", "se", "sw", "se", "x", "nw", "ne", "ns", "ns", "sw", "ne", "sw", "se", "ew", "sw", "nw", "ne", "ne", "x", "ns", "sw", "ns", "se", "nw", "se", "ew", "ew", "ew", "nw", "x", "ns", "ns", "ew", "ns", "ew", "sw", "ew", "ne", "ns", "ne", "ns", "ns", "nw"],
     "distance": 12, "countdown": 7, "pace": 3},
    # seed 11012; 39 expansions, 30-action plan
    {"rows": [".......",
             ".......",
             "..v....",
             ".......",
             ".......",
             ".......",
             "......."],
     "queue": ["ew", "ew", "ew", "nw", "se", "x", "ew", "ns", "se", "ew", "ns", "ns", "ns", "nw", "ew", "nw", "sw", "ne", "se", "ns", "sw", "ns", "x", "ne", "ew", "ew", "x", "ew", "se", "ne", "ns", "nw", "nw", "ns", "sw", "ns", "ne", "se", "ew", "ew", "se"],
     "distance": 11, "countdown": 8, "pace": 3},
    # seed 11013; 51 expansions, 22-action plan
    {"rows": ["......v",
             ".......",
             ".......",
             ".......",
             ".......",
             ".....#.",
             "......."],
     "queue": ["ew", "ew", "ne", "ne", "nw", "ne", "sw", "ew", "se", "ew", "se", "ne", "se", "ew", "ew", "nw", "ew", "ne", "x", "sw", "ns", "ns", "sw", "se", "ne", "se", "ew", "nw", "ew", "se", "ew", "nw", "sw", "se", "sw", "ew", "ns", "sw"],
     "distance": 10, "countdown": 8, "pace": 3},
    # seed 11014; 46 expansions, 25-action plan
    {"rows": [".......",
             ".......",
             ".......",
             ".......",
             "...#...",
             ".......",
             "...>..."],
     "queue": ["nw", "x", "ns", "nw", "ne", "ns", "ew", "se", "ns", "ew", "ew", "nw", "ew", "nw", "ns", "ns", "x", "ns", "nw", "sw", "se", "nw", "nw", "ew", "se", "ns", "sw", "ns", "ew", "sw", "ew", "ns", "sw", "ns", "se", "nw", "ew", "x", "x", "nw", "ns"],
     "distance": 12, "countdown": 5, "pace": 3},
    # seed 11015; 21 expansions, 21-action plan
    {"rows": [".......",
             "......<",
             ".......",
             ".......",
             ".......",
             ".......",
             "......."],
     "queue": ["nw", "sw", "nw", "ne", "ns", "sw", "ne", "ns", "ns", "se", "ns", "ns", "x", "ns", "ew", "ne", "ne", "ew", "nw", "ns", "nw", "ew", "ew", "ne", "x", "ne", "nw", "sw", "ns", "sw", "sw", "ns", "nw", "x", "nw", "nw", "se", "ns", "ns", "se", "ew"],
     "distance": 11, "countdown": 8, "pace": 3},
    # seed 11016; 173 expansions, 17-action plan
    {"rows": [".......",
             "v...#..",
             ".......",
             ".......",
             ".......",
             ".......",
             "....#.."],
     "queue": ["sw", "nw", "sw", "nw", "ew", "ne", "se", "ns", "ns", "ew", "ew", "ne", "x", "sw", "ne", "ew", "ns", "se", "ns", "ew", "se", "x", "ns", "nw", "ew", "ew", "sw", "ew", "ns", "ns"],
     "distance": 8, "countdown": 6, "pace": 3},
    # seed 11017; 13 expansions, 12-action plan
    {"rows": [".......",
             ".......",
             ".......",
             ".......",
             ".....<.",
             ".......",
             "......."],
     "queue": ["ne", "x", "sw", "ew", "se", "ne", "ns", "sw", "ne", "ns", "nw", "se", "sw", "ew", "x", "ne", "ne", "x", "se", "ns", "ew", "ew", "ne", "ew", "ew", "sw", "ne", "ew", "se", "ns", "se", "nw", "nw", "ns"],
     "distance": 9, "countdown": 7, "pace": 3},
    # seed 11018; 22 expansions, 18-action plan
    {"rows": [".......",
             "......#",
             "v......",
             ".......",
             ".......",
             ".......",
             "......."],
     "queue": ["nw", "ne", "ns", "sw", "se", "ne", "nw", "nw", "se", "sw", "ne", "nw", "ne", "ew", "se", "nw", "sw", "sw", "ns", "ns", "ew", "se", "ew", "se", "ew", "ew", "ns", "nw", "ne", "nw", "sw", "sw", "se", "sw", "se", "ne", "se", "ne", "ew"],
     "distance": 11, "countdown": 6, "pace": 3},
    # seed 11019; 39 expansions, 24-action plan
    {"rows": [".......",
             ".......",
             "......#",
             ".......",
             ".......",
             "...^...",
             ".#....."],
     "queue": ["sw", "ns", "ns", "ns", "ne", "nw", "se", "ns", "ew", "sw", "sw", "nw", "ns", "ne", "nw", "ns", "ne", "ew", "ew", "ns", "sw", "ns", "ne", "ne", "ew", "ns", "sw", "ew", "se", "sw", "se", "sw", "sw", "ew", "ew", "sw", "nw", "sw"],
     "distance": 11, "countdown": 5, "pace": 3},
    # seed 11020; 26 expansions, 19-action plan
    {"rows": [".......",
             ".......",
             ".......",
             ".......",
             ".#.....",
             ".......",
             "#..^..."],
     "queue": ["se", "ne", "se", "nw", "sw", "x", "sw", "ne", "ew", "ne", "ne", "sw", "x", "x", "ne", "se", "ew", "nw", "ew", "ns", "ew", "nw", "sw", "nw", "ew", "nw", "se", "ew", "ew", "ew", "ew", "se", "x", "ew", "sw", "ew", "ne", "ne", "ew", "ew", "se"],
     "distance": 11, "countdown": 8, "pace": 3},
    # seed 11021; 17 expansions, 16-action plan
    {"rows": [".......",
             ".......",
             ".......",
             ".......",
             ".......",
             "..>....",
             "......."],
     "queue": ["se", "ne", "x", "ew", "se", "ns", "nw", "nw", "ns", "se", "ns", "ew", "nw", "ns", "se", "ew", "nw", "ew", "ne", "x", "se", "ew", "ne", "ew", "se", "nw", "ew", "sw", "sw", "x", "sw", "se"],
     "distance": 8, "countdown": 8, "pace": 3},
    # seed 11022; 52 expansions, 23-action plan
    {"rows": [".......",
             ".......",
             "..^....",
             ".......",
             ".......",
             ".......",
             "......."],
     "queue": ["ns", "ns", "nw", "ns", "ns", "x", "ns", "ns", "se", "ew", "ne", "sw", "ne", "ns", "ns", "nw", "sw", "ns", "sw", "nw", "ew", "ne", "ew", "ew", "nw", "se", "se", "ew", "sw", "ew", "ew", "ew", "ew", "ns", "sw", "nw", "ne", "x"],
     "distance": 10, "countdown": 8, "pace": 3},
    # seed 11023; 18 expansions, 17-action plan
    {"rows": [".......",
             ".......",
             ".......",
             "...#...",
             ".......",
             "...<...",
             "......."],
     "queue": ["se", "ew", "ne", "ne", "x", "ne", "se", "nw", "ne", "sw", "sw", "ne", "ne", "ew", "ne", "ns", "ew", "ns", "ew", "nw", "x", "ew", "nw", "ns", "se", "ew", "ne", "nw", "ne", "ew", "ew", "se", "se", "x", "se"],
     "distance": 9, "countdown": 8, "pace": 3},
    # seed 11024; 22 expansions, 21-action plan
    {"rows": [".....<.",
             ".......",
             ".......",
             ".......",
             ".......",
             ".......",
             "......."],
     "queue": ["ne", "nw", "x", "x", "nw", "ew", "ns", "se", "ne", "nw", "ns", "sw", "ne", "ne", "x", "sw", "ew", "sw", "se", "ne", "ns", "nw", "ew", "ew", "x", "ew", "ew", "nw", "x", "ns", "se", "sw", "ns", "se", "ew", "ns", "ne", "se", "sw"],
     "distance": 11, "countdown": 6, "pace": 3},
    # seed 11025; 22 expansions, 20-action plan
    {"rows": [".......",
             ".......",
             ".......",
             "......#",
             "....^..",
             ".......",
             "......."],
     "queue": ["x", "nw", "sw", "ns", "ns", "ns", "ns", "se", "sw", "ew", "ns", "sw", "ns", "se", "ns", "ne", "ns", "sw", "ns", "nw", "nw", "nw", "x", "ne", "sw", "ns", "se", "x", "ns", "ns", "ns", "ns", "ns", "ns", "nw", "ns", "se", "nw"],
     "distance": 11, "countdown": 5, "pace": 3},
    # seed 11026; 23 expansions, 20-action plan
    {"rows": [".......",
             ".#.....",
             ".......",
             "..#....",
             ".......",
             "...>...",
             "......."],
     "queue": ["sw", "x", "ne", "sw", "se", "sw", "x", "nw", "se", "nw", "ne", "ns", "ns", "ns", "sw", "ne", "ns", "ew", "se", "sw", "ns", "ne", "ew", "ew", "ne", "ne", "sw", "ne", "ne", "ew", "sw", "x", "ne", "ne", "sw", "nw", "ne", "se", "se", "ns"],
     "distance": 11, "countdown": 7, "pace": 3},
    # seed 11027; 26 expansions, 21-action plan
    {"rows": ["#......",
             ".......",
             ".......",
             ".......",
             ".......",
             ".......",
             "...^..."],
     "queue": ["ew", "se", "x", "ns", "se", "x", "se", "x", "nw", "ew", "sw", "se", "ne", "ne", "ew", "ew", "nw", "ne", "ew", "sw", "ew", "ew", "se", "ne", "ew", "ns", "ne", "se", "ew", "nw", "ew", "ew", "se"],
     "distance": 9, "countdown": 6, "pace": 3},
    # seed 11028; 27 expansions, 25-action plan
    {"rows": [".......",
             ".......",
             ".......",
             ".......",
             ".....^.",
             ".......",
             "......."],
     "queue": ["ne", "ew", "sw", "se", "ew", "ew", "nw", "se", "ew", "ns", "ns", "sw", "se", "ew", "sw", "ew", "ne", "ne", "nw", "se", "ew", "nw", "ew", "ew", "nw", "ne", "ns", "ns", "se", "ne", "ew", "ew", "nw", "x", "ns", "se", "sw", "sw", "sw", "sw", "ew"],
     "distance": 12, "countdown": 5, "pace": 3},
    # seed 11029; 77 expansions, 25-action plan
    {"rows": [".......",
             ".......",
             "#....<.",
             ".......",
             ".......",
             ".......",
             "......."],
     "queue": ["sw", "ns", "sw", "se", "nw", "ew", "nw", "ns", "ew", "x", "ne", "ne", "ew", "ew", "ew", "ns", "se", "ew", "ne", "ew", "sw", "sw", "ew", "ns", "x", "se", "ns", "ew", "ns", "sw", "sw", "ew", "se", "ew", "nw", "se", "sw", "ew"],
     "distance": 10, "countdown": 8, "pace": 3},
    # seed 11030; 21 expansions, 18-action plan
    {"rows": [".......",
             ".......",
             ".......",
             "....>..",
             ".......",
             ".......",
             "......."],
     "queue": ["ew", "ew", "nw", "sw", "sw", "ns", "ne", "ew", "ns", "sw", "x", "ew", "ew", "sw", "ew", "se", "ns", "se", "ew", "nw", "sw", "se", "x", "ew", "ns", "ew", "ew", "se", "ns", "ew", "sw", "se", "ew", "ns", "ew", "sw", "x", "ew", "ns", "x", "x", "sw"],
     "distance": 12, "countdown": 6, "pace": 3},
    # seed 11031; 22 expansions, 19-action plan
    {"rows": [".......",
             "......#",
             ".v#....",
             ".......",
             ".......",
             ".......",
             "...#..."],
     "queue": ["ew", "ns", "sw", "x", "ns", "se", "ns", "ne", "ns", "ew", "ew", "se", "nw", "se", "ew", "x", "x", "nw", "x", "ns", "sw", "se", "ne", "ne", "ne", "ns", "ns", "ns", "se", "sw", "ew", "ns", "ne", "nw", "ew", "ew", "se", "ns"],
     "distance": 10, "countdown": 8, "pace": 3},
    # seed 11032; 33 expansions, 19-action plan
    {"rows": ["....#..",
             "#......",
             ".......",
             ".......",
             ".......",
             ".......",
             "^......"],
     "queue": ["x", "se", "ns", "sw", "se", "nw", "nw", "ns", "se", "nw", "ne", "sw", "se", "nw", "ns", "ne", "ns", "nw", "sw", "ne", "ns", "ns", "ew", "ew", "sw", "ew", "x", "x", "nw", "sw", "ew", "se", "sw", "ew", "sw", "ne", "ew", "ns"],
     "distance": 10, "countdown": 8, "pace": 3},
    # seed 11033; 26 expansions, 20-action plan
    {"rows": [".......",
             ".......",
             ".......",
             "..v....",
             "...#..#",
             ".......",
             "......."],
     "queue": ["ew", "ne", "ew", "nw", "ns", "ew", "ew", "ne", "ne", "ne", "se", "ne", "ns", "nw", "ne", "sw", "ns", "ne", "se", "ew", "sw", "nw", "ew", "nw", "ew", "se", "ne", "sw", "ns", "ns"],
     "distance": 8, "countdown": 6, "pace": 3},
    # seed 11034; 26 expansions, 18-action plan
    {"rows": [".......",
             ".......",
             ".......",
             ".......",
             ".......",
             ".....<.",
             "......."],
     "queue": ["x", "ew", "ne", "ew", "nw", "se", "sw", "ew", "ns", "ne", "sw", "nw", "sw", "ew", "x", "ns", "nw", "sw", "sw", "nw", "ns", "se", "ne", "ns", "ns", "ew", "ne", "ns", "ew", "se", "ew", "se"],
     "distance": 9, "countdown": 5, "pace": 3},
    # seed 11035; 24 expansions, 15-action plan
    {"rows": [".......",
             ".......",
             ".......",
             "..#....",
             "......#",
             ".......",
             "...^..."],
     "queue": ["ew", "x", "x", "ns", "ew", "x", "nw", "x", "ns", "sw", "ns", "sw", "x", "se", "ne", "se", "se", "x", "sw", "ew", "se", "ew", "ns", "ne", "sw", "ew", "nw", "sw", "se", "x", "nw"],
     "distance": 8, "countdown": 7, "pace": 3},
    # seed 11036; 22 expansions, 18-action plan
    {"rows": [".......",
             "#......",
             ".......",
             "......<",
             ".......",
             ".......",
             "......#"],
     "queue": ["ne", "ns", "ne", "ne", "ne", "sw", "ne", "ew", "nw", "sw", "ne", "x", "ne", "ew", "ne", "se", "nw", "ew", "ns", "nw", "se", "ns", "ns", "sw", "ew", "nw", "ns", "ne", "se", "sw", "nw", "se", "x", "ns", "se"],
     "distance": 9, "countdown": 8, "pace": 3},
    # seed 11037; 33 expansions, 22-action plan
    {"rows": [".......",
             ".......",
             "..<....",
             ".......",
             ".......",
             ".......",
             "......."],
     "queue": ["se", "ns", "ew", "ns", "ne", "ns", "x", "sw", "ne", "ns", "ns", "ew", "ne", "ns", "ne", "ne", "sw", "sw", "nw", "ew", "ew", "nw", "ns", "ns", "se", "se", "nw", "ew", "ne", "ne", "ew", "se", "nw", "ne", "ns", "se"],
     "distance": 10, "countdown": 6, "pace": 3},
    # seed 11038; 46 expansions, 22-action plan
    {"rows": [".>.....",
             "...#...",
             ".......",
             ".......",
             ".......",
             ".......",
             "......."],
     "queue": ["ns", "x", "ns", "ew", "ne", "ew", "se", "ne", "sw", "se", "nw", "nw", "ne", "sw", "nw", "ew", "se", "sw", "se", "ne", "ew", "nw", "nw", "nw", "ew", "nw", "ne", "se", "ew", "ew", "ew"],
     "distance": 8, "countdown": 7, "pace": 3},
    # seed 11039; 35 expansions, 21-action plan
    {"rows": [".......",
             ".......",
             "...#...",
             "....^..",
             ".......",
             ".......",
             "......."],
     "queue": ["sw", "sw", "nw", "ns", "ns", "sw", "ns", "sw", "ns", "ns", "se", "ew", "sw", "se", "x", "ne", "se", "nw", "ns", "se", "sw", "ns", "ne", "ew", "se", "nw", "ne", "ns", "ew", "sw", "ns", "nw"],
     "distance": 8, "countdown": 8, "pace": 3},
    # seed 11040; 31 expansions, 19-action plan
    {"rows": [".......",
             ".......",
             ".......",
             ".......",
             "....^..",
             ".......",
             "......."],
     "queue": ["ew", "x", "nw", "ns", "sw", "ew", "ns", "nw", "ns", "ns", "nw", "nw", "ne", "ne", "ew", "se", "nw", "ew", "ne", "se", "ew", "ne", "ew", "ns", "ne", "ew", "sw", "ne", "x", "ns", "sw", "ne"],
     "distance": 8, "countdown": 8, "pace": 3},
    # seed 11041; 42 expansions, 30-action plan
    {"rows": [".#.....",
             ">......",
             "..#....",
             ".......",
             ".......",
             ".......",
             "......."],
     "queue": ["ew", "ne", "sw", "ns", "sw", "ns", "ew", "ns", "sw", "ew", "x", "ew", "ne", "ns", "ew", "ns", "nw", "ew", "ne", "ew", "se", "nw", "ew", "ne", "sw", "ew", "ns", "ne", "ew", "se", "ns", "ns", "nw", "nw", "se", "nw", "ew", "ne", "nw", "x", "sw"],
     "distance": 12, "countdown": 5, "pace": 3},
    # seed 11042; 78 expansions, 21-action plan
    {"rows": ["..>....",
             ".......",
             ".......",
             ".......",
             ".......",
             ".......",
             "......."],
     "queue": ["ns", "ns", "ew", "ew", "x", "x", "ew", "ne", "se", "ew", "ew", "nw", "ew", "sw", "nw", "ew", "ns", "ew", "sw", "x", "ns", "ne", "ew", "ne", "se", "ns", "sw", "x", "ne", "ns", "sw"],
     "distance": 8, "countdown": 7, "pace": 3},
    # seed 11043; 39 expansions, 28-action plan
    {"rows": [".......",
             ".......",
             ".......",
             ".......",
             ">......",
             ".......",
             "......."],
     "queue": ["ns", "se", "ns", "ew", "ew", "se", "x", "ew", "ne", "ew", "sw", "ne", "ew", "sw", "ew", "ew", "ns", "nw", "ew", "ew", "se", "nw", "ns", "se", "sw", "ew", "ne", "se", "ns", "sw", "ew", "ns", "ew", "ne", "x", "x", "ne", "ew", "ew", "sw", "sw"],
     "distance": 12, "countdown": 5, "pace": 3},
    # seed 11044; 19 expansions, 17-action plan
    {"rows": [".......",
             ".......",
             ".......",
             ".......",
             "...v...",
             "#......",
             "......."],
     "queue": ["ne", "nw", "ew", "se", "se", "ns", "se", "ns", "sw", "nw", "x", "ns", "ne", "ew", "ew", "ew", "se", "ns", "sw", "sw", "ew", "se", "se", "ne", "se", "se", "ne", "ns", "nw", "se", "ew", "ew", "ns", "ne", "nw"],
     "distance": 9, "countdown": 8, "pace": 3},
    # seed 11045; 41 expansions, 28-action plan
    {"rows": ["......#",
             ".......",
             ".......",
             "....<#.",
             ".......",
             ".......",
             "......."],
     "queue": ["sw", "sw", "ns", "se", "se", "nw", "nw", "se", "nw", "ns", "ew", "ns", "x", "ns", "ew", "sw", "nw", "ne", "ew", "ns", "ns", "ew", "x", "se", "ns", "ew", "x", "ew", "ew", "sw", "ns", "nw", "se", "ne", "nw", "nw", "se", "nw", "sw", "ne", "sw", "x", "ns", "ew"],
     "distance": 12, "countdown": 8, "pace": 3},
    # seed 11046; 16 expansions, 15-action plan
    {"rows": ["....>..",
             ".......",
             ".......",
             ".......",
             ".......",
             ".......",
             "......."],
     "queue": ["se", "ew", "sw", "ew", "ns", "ne", "se", "ew", "x", "x", "ns", "ns", "nw", "x", "nw", "ew", "ew", "ns", "sw", "x", "x", "ew", "nw", "ns", "sw", "ew", "se", "ew", "ns", "ne", "se", "ne", "se", "sw", "nw"],
     "distance": 9, "countdown": 8, "pace": 3},
    # seed 11047; 16 expansions, 15-action plan
    {"rows": [".......",
             ".......",
             ".......",
             "....<..",
             ".......",
             ".......",
             "......."],
     "queue": ["ne", "se", "ne", "ns", "ns", "ew", "se", "sw", "ns", "ne", "sw", "ns", "ew", "ns", "ns", "nw", "ne", "se", "ew", "ns", "se", "ne", "nw", "ne", "ns", "se", "sw", "se", "ns", "ns", "ew", "se", "ew", "x", "se", "se", "ew"],
     "distance": 10, "countdown": 7, "pace": 3},
    # seed 11048; 24 expansions, 20-action plan
    {"rows": ["..v....",
             ".......",
             ".......",
             "#......",
             ".......",
             "...#...",
             "......."],
     "queue": ["nw", "se", "se", "ew", "se", "sw", "sw", "nw", "ne", "ew", "ew", "ew", "se", "ns", "sw", "ew", "se", "se", "ns", "ne", "ne", "nw", "ns", "sw", "ew", "ew", "nw", "se", "ns", "ne", "ns", "se"],
     "distance": 9, "countdown": 5, "pace": 3},
    # seed 11049; 27 expansions, 27-action plan
    {"rows": [".......",
             ".......",
             "....>..",
             ".....#.",
             ".....#.",
             ".......",
             "......."],
     "queue": ["ne", "nw", "ew", "nw", "ns", "ne", "ne", "ne", "se", "ne", "ne", "sw", "x", "ns", "nw", "sw", "sw", "ew", "ew", "ns", "ew", "sw", "ns", "nw", "se", "nw", "ns", "sw", "nw", "ns", "ew", "se", "ns", "sw", "nw", "nw", "ns"],
     "distance": 10, "countdown": 7, "pace": 3},
    # seed 11500; 55 expansions, 24-action plan
    {"rows": [".......",
             ".......",
             ".......",
             ".......",
             ".......",
             ".......",
             "#.....<"],
     "queue": ["ew", "ne", "ns", "nw", "ns", "nw", "ew", "sw", "nw", "se", "se", "ns", "ns", "ns", "sw", "sw", "ns", "ne", "ns", "ew", "nw", "nw", "ew", "x", "ne", "ne", "sw", "ns", "ns", "x", "x", "nw", "ew", "ne", "ew"],
     "distance": 10, "countdown": 5, "pace": 3},
    # seed 11501; 39 expansions, 25-action plan
    {"rows": [".......",
             ".#..#..",
             ".....v.",
             ".......",
             ".......",
             ".......",
             "......."],
     "queue": ["ns", "ne", "nw", "ew", "ns", "ns", "se", "ew", "ns", "ns", "ns", "se", "ns", "sw", "x", "ns", "sw", "nw", "ew", "ns", "ns", "sw", "ns", "ne", "ew", "ne", "ew", "ew", "nw", "ew", "nw", "ne", "ns", "nw", "ew", "ne", "sw", "se"],
     "distance": 10, "countdown": 8, "pace": 3},
    # seed 11502; 28 expansions, 27-action plan
    {"rows": [".......",
             ".......",
             ".......",
             "...#...",
             ".....^.",
             ".......",
             "......."],
     "queue": ["x", "ew", "ns", "nw", "ne", "sw", "ew", "sw", "ns", "ns", "sw", "ew", "x", "ns", "sw", "ew", "nw", "nw", "ns", "sw", "sw", "ns", "ew", "ne", "se", "ew", "ew", "ew", "ns", "ew", "ns", "se", "ne", "ne", "ns"],
     "distance": 10, "countdown": 5, "pace": 3},
    # seed 11503; 33 expansions, 21-action plan
    {"rows": ["......#",
             "..>....",
             ".......",
             ".......",
             "....#..",
             ".......",
             "......."],
     "queue": ["nw", "se", "se", "x", "nw", "ew", "ns", "ew", "sw", "x", "ne", "ne", "ns", "ne", "nw", "x", "ne", "ew", "nw", "ns", "ns", "ne", "ew", "ew", "x", "se", "x", "nw", "se", "sw", "nw", "ne", "ew", "nw", "se"],
     "distance": 10, "countdown": 5, "pace": 3},
    # seed 11504; 19 expansions, 16-action plan
    {"rows": [".......",
             "...v...",
             ".....#.",
             ".......",
             ".......",
             "......#",
             "......."],
     "queue": ["ne", "ns", "se", "ew", "ns", "ne", "sw", "ne", "ns", "ns", "ns", "nw", "ew", "ne", "ns", "ew", "nw", "ne", "ne", "se", "ew", "sw", "nw", "nw", "sw", "ns", "sw", "ne", "ns", "x", "sw", "ne", "ne"],
     "distance": 9, "countdown": 6, "pace": 3},
    # seed 11505; 57 expansions, 19-action plan
    {"rows": [".......",
             ".......",
             "....#..",
             ".......",
             ".....#.",
             "...<.#.",
             "......."],
     "queue": ["ne", "nw", "x", "x", "ns", "ns", "sw", "ne", "ns", "nw", "ns", "sw", "sw", "ne", "ew", "sw", "se", "ns", "nw", "ne", "ne", "sw", "ns", "x", "ew", "se", "ne", "ne", "sw", "nw", "nw", "ns"],
     "distance": 9, "countdown": 5, "pace": 3},
    # seed 11506; 15 expansions, 15-action plan
    {"rows": ["#......",
             ".......",
             "...#...",
             ".....<.",
             ".......",
             ".......",
             "....#.."],
     "queue": ["ns", "ns", "ew", "se", "ne", "se", "sw", "nw", "se", "ew", "ne", "ns", "sw", "se", "sw", "ns", "nw", "x", "nw", "nw", "se", "ns", "ne", "ew", "nw", "nw", "ew", "sw", "nw", "se", "ns", "ns", "ew", "ew", "se", "sw", "ns"],
     "distance": 10, "countdown": 7, "pace": 3},
    # seed 11507; 30 expansions, 26-action plan
    {"rows": [".......",
             ".......",
             ".....#.",
             ".....<#",
             ".......",
             ".......",
             "......."],
     "queue": ["ew", "ew", "ew", "ew", "ns", "nw", "x", "ns", "sw", "sw", "ne", "ew", "sw", "nw", "sw", "ew", "ns", "ns", "ne", "ns", "ew", "nw", "se", "sw", "ne", "ew", "ns", "ew", "ew", "ne", "ns", "ew", "nw", "ns", "nw", "sw", "sw", "sw"],
     "distance": 10, "countdown": 8, "pace": 3},
    # seed 11508; 29 expansions, 22-action plan
    {"rows": [".......",
             ".......",
             "....#..",
             ".#.....",
             "..v....",
             ".......",
             "......."],
     "queue": ["ew", "se", "ns", "ew", "ne", "ns", "ew", "nw", "ne", "nw", "nw", "nw", "ew", "nw", "sw", "ew", "sw", "ns", "ne", "se", "x", "ne", "nw", "ew", "ne", "se", "ew", "x", "ew", "nw", "nw", "se", "nw", "ne", "ne", "x", "ns", "sw"],
     "distance": 10, "countdown": 8, "pace": 3},
    # seed 11509; 33 expansions, 25-action plan
    {"rows": [".......",
             ".......",
             "^..#..#",
             ".......",
             ".......",
             ".......",
             "...#..."],
     "queue": ["ne", "sw", "ns", "se", "se", "ew", "ew", "ne", "sw", "ew", "ew", "ne", "ns", "ns", "ne", "ew", "nw", "se", "ne", "ew", "se", "ew", "sw", "ne", "ne", "ns", "se", "sw", "nw", "ew", "sw", "ne", "ns", "se"],
     "distance": 9, "countdown": 7, "pace": 3},
    # seed 11510; 16 expansions, 15-action plan
    {"rows": [".#.....",
             ".......",
             ".......",
             "..^....",
             "....#..",
             "#......",
             "......."],
     "queue": ["se", "ns", "ns", "ew", "ew", "se", "se", "se", "ew", "ew", "sw", "ns", "ew", "ne", "ns", "nw", "sw", "nw", "sw", "x", "sw", "ns", "nw", "ns", "ew", "ew", "ne", "ns", "x", "ew"],
     "distance": 8, "countdown": 6, "pace": 3},
    # seed 11511; 15 expansions, 15-action plan
    {"rows": [".......",
             ".......",
             ".......",
             ".#.....",
             ".......",
             ".......",
             ".....^."],
     "queue": ["x", "ns", "ew", "x", "ns", "x", "ew", "ns", "ew", "nw", "ne", "se", "sw", "nw", "nw", "nw", "se", "sw", "se", "se", "ns", "ew", "ew", "nw", "ns", "nw", "x", "ns", "sw", "nw", "se"],
     "distance": 8, "countdown": 7, "pace": 3},
    # seed 11512; 23 expansions, 18-action plan
    {"rows": [".......",
             ".......",
             ".......",
             ".......",
             ".......",
             "...^...",
             "....#.."],
     "queue": ["ew", "ns", "ew", "nw", "sw", "ew", "nw", "x", "ne", "ew", "nw", "sw", "se", "ne", "x", "ns", "ew", "ns", "se", "ew", "ns", "nw", "ew", "ew", "sw", "ew", "ew", "sw", "ew"],
     "distance": 8, "countdown": 5, "pace": 3},
    # seed 11513; 29 expansions, 24-action plan
    {"rows": [".......",
             ".......",
             ".......",
             "v......",
             ".......",
             ".......",
             "......."],
     "queue": ["ne", "se", "x", "ne", "ew", "se", "x", "sw", "se", "sw", "ns", "ne", "se", "nw", "sw", "ns", "sw", "x", "ne", "sw", "nw", "nw", "se", "nw", "ne", "ne", "ne", "se", "ne", "ns", "nw", "ew", "ne", "sw", "ew", "ne", "ne", "ns", "ns", "se", "ne", "sw", "ew", "ew"],
     "distance": 12, "countdown": 8, "pace": 3},
    # seed 11514; 22 expansions, 19-action plan
    {"rows": [".......",
             ".......",
             ".......",
             ".......",
             ".......",
             "..^....",
             "....#.."],
     "queue": ["ne", "ns", "sw", "ew", "ew", "ne", "ew", "se", "ew", "ne", "ne", "ew", "x", "nw", "sw", "x", "nw", "ew", "ns", "sw", "se", "ew", "ns", "x", "ns", "se", "ns", "ew", "ew", "sw", "ns", "se", "se", "ne", "nw", "ne", "se", "ne", "nw", "se", "ew", "ew"],
     "distance": 12, "countdown": 6, "pace": 3},
    # seed 11515; 18 expansions, 16-action plan
    {"rows": [".......",
             ".......",
             "..>....",
             ".......",
             ".......",
             ".......",
             "......."],
     "queue": ["ew", "ew", "sw", "ns", "x", "ew", "se", "ne", "se", "nw", "nw", "ns", "nw", "sw", "x", "ne", "ne", "ne", "nw", "ew", "ew", "ew", "se", "sw", "se", "sw", "ns", "ew", "ew", "sw", "sw", "ns", "ew", "ne", "x", "se", "ew", "ne"],
     "distance": 10, "countdown": 8, "pace": 3},
    # seed 11516; 41 expansions, 26-action plan
    {"rows": [".......",
             ">......",
             ".......",
             ".......",
             ".......",
             ".......",
             "......."],
     "queue": ["ew", "sw", "ne", "se", "ns", "ne", "x", "ns", "ew", "ew", "sw", "sw", "sw", "ns", "sw", "ew", "ns", "ns", "ns", "sw", "ew", "ne", "ew", "nw", "ew", "ns", "ew", "ns", "ns", "ns", "ew", "se", "sw", "ne", "nw", "sw", "x", "ew", "nw", "se", "ew", "nw", "ne"],
     "distance": 12, "countdown": 7, "pace": 3},
    # seed 11517; 97 expansions, 20-action plan
    {"rows": [".......",
             ".......",
             ".......",
             "...v...",
             ".......",
             ".......",
             "......."],
     "queue": ["ne", "ew", "sw", "x", "ne", "sw", "ns", "ew", "ew", "ns", "se", "se", "nw", "ne", "nw", "ew", "ew", "nw", "sw", "sw", "se", "sw", "nw", "se", "se", "nw", "ns", "se", "se"],
     "distance": 8, "countdown": 5, "pace": 3},
    # seed 11518; 96 expansions, 31-action plan
    {"rows": [".......",
             "......v",
             "....##.",
             ".......",
             ".......",
             ".......",
             "......."],
     "queue": ["se", "ew", "ew", "ns", "ns", "ns", "x", "se", "ew", "ne", "x", "ne", "ew", "x", "nw", "ns", "ne", "nw", "ns", "ne", "nw", "se", "ns", "x", "ns", "ne", "sw", "se", "x", "ns", "sw", "ns", "se", "sw", "sw", "ew", "ew", "nw", "ew", "ne", "nw", "nw", "sw", "sw"],
     "distance": 12, "countdown": 8, "pace": 3},
    # seed 11519; 21 expansions, 20-action plan
    {"rows": [".v.#...",
             ".......",
             ".......",
             "......#",
             ".......",
             ".......",
             "......."],
     "queue": ["ew", "ne", "sw", "ew", "ne", "x", "ne", "sw", "ne", "sw", "se", "ew", "ne", "se", "ns", "ns", "ne", "se", "sw", "ne", "nw", "sw", "sw", "sw", "ns", "ew", "nw", "ne", "x", "ne", "se", "nw", "ew", "ew", "se", "ew", "ns", "se", "se", "ns"],
     "distance": 11, "countdown": 7, "pace": 3},
    # seed 11520; 27 expansions, 22-action plan
    {"rows": ["v......",
             "...#..#",
             ".......",
             ".......",
             ".......",
             ".......",
             "......."],
     "queue": ["se", "se", "se", "nw", "ne", "sw", "se", "ew", "ne", "ew", "ew", "nw", "sw", "ew", "se", "ne", "sw", "ne", "ew", "sw", "x", "ew", "ns", "sw", "ew", "ns", "se", "ew", "se", "ns", "se", "se", "nw", "sw", "ew", "ew", "ew", "nw"],
     "distance": 11, "countdown": 5, "pace": 3},
    # seed 11521; 35 expansions, 23-action plan
    {"rows": ["......#",
             "#......",
             ".......",
             ".......",
             "..<....",
             ".......",
             "......."],
     "queue": ["nw", "nw", "ne", "ns", "sw", "se", "sw", "nw", "sw", "ne", "ns", "ne", "ns", "se", "ne", "nw", "nw", "se", "ew", "ns", "ew", "ew", "ns", "ns", "ew", "nw", "nw", "ns", "ns", "ne", "ew", "ew", "ne", "ns", "se", "nw", "ew"],
     "distance": 10, "countdown": 7, "pace": 3},
    # seed 11522; 24 expansions, 20-action plan
    {"rows": [".......",
             ".......",
             ".......",
             "..#....",
             ".......",
             ".......",
             ".....<."],
     "queue": ["x", "ew", "ne", "ne", "ns", "ew", "ew", "ns", "ns", "ns", "sw", "x", "nw", "nw", "ne", "ns", "ns", "se", "nw", "ne", "nw", "ew", "nw", "nw", "ew", "x", "ew", "ne", "se", "ew", "se", "sw", "ns", "ne", "ne", "sw", "ew", "ns"],
     "distance": 11, "countdown": 5, "pace": 3},
    # seed 11523; 33 expansions, 20-action plan
    {"rows": [".......",
             ">......",
             ".#.#.#.",
             ".......",
             ".......",
             ".......",
             "......."],
     "queue": ["ns", "nw", "se", "ew", "sw", "ns", "ew", "ew", "sw", "nw", "se", "ew", "ne", "ew", "nw", "nw", "sw", "x", "x", "ns", "ew", "se", "ns", "nw", "ns", "ew", "ew", "ew", "ne"],
     "distance": 8, "countdown": 5, "pace": 3},
    # seed 11524; 31 expansions, 27-action plan
    {"rows": [".>.....",
             ".......",
             ".......",
             ".......",
             ".......",
             ".......",
             "......."],
     "queue": ["x", "nw", "ne", "se", "ew", "ew", "ns", "ew", "nw", "ns", "ne", "ne", "ns", "ne", "sw", "ew", "ew", "ns", "ew", "ns", "ns", "ns", "ew", "nw", "ns", "x", "ew", "ew", "se", "ns", "nw", "x", "sw", "ew", "sw", "nw", "sw", "ns"],
     "distance": 11, "countdown": 5, "pace": 3},
    # seed 11525; 45 expansions, 27-action plan
    {"rows": [".......",
             ".......",
             "...#...",
             "..#....",
             ".......",
             ".......",
             "^......"],
     "queue": ["sw", "ns", "ew", "ew", "se", "ew", "ew", "ew", "ne", "se", "ns", "nw", "se", "nw", "nw", "ne", "ew", "ns", "ns", "ns", "ns", "sw", "sw", "ns", "ns", "ew", "ns", "ew", "ns", "ew", "ew", "sw", "sw", "ne", "ew", "nw", "ew", "ns", "ne", "ew", "ns", "sw"],
     "distance": 12, "countdown": 6, "pace": 3},
    # seed 11526; 20 expansions, 17-action plan
    {"rows": [".......",
             ".......",
             "#......",
             ".....<#",
             ".......",
             ".......",
             "......."],
     "queue": ["ne", "sw", "se", "se", "sw", "ns", "ns", "ns", "ew", "ns", "ns", "ew", "ne", "nw", "ew", "sw", "x", "ew", "ns", "ne", "nw", "x", "se", "nw", "ew", "sw", "ew", "ne", "ew", "se", "nw", "sw", "x", "nw"],
     "distance": 9, "countdown": 7, "pace": 3},
    # seed 11527; 24 expansions, 20-action plan
    {"rows": [".......",
             ".......",
             ".......",
             ".......",
             ".>.....",
             ".....#.",
             "......."],
     "queue": ["nw", "ew", "ew", "ns", "se", "sw", "ne", "ew", "se", "ew", "ns", "ns", "ns", "ne", "ew", "ns", "nw", "x", "ns", "sw", "ns", "ne", "ew", "ns", "se", "ne", "ew", "se", "sw", "ns", "sw", "x", "sw"],
     "distance": 9, "countdown": 6, "pace": 3},
    # seed 11528; 17 expansions, 16-action plan
    {"rows": [".......",
             ".......",
             ".......",
             ".......",
             "......v",
             "...#.#.",
             "......."],
     "queue": ["ew", "se", "ew", "se", "ns", "sw", "nw", "x", "ew", "se", "ew", "ne", "sw", "ns", "ew", "ns", "ns", "se", "ns", "se", "se", "ew", "x", "ew", "ew", "ne", "se", "ne", "ne", "se", "nw"],
     "distance": 8, "countdown": 7, "pace": 3},
    # seed 11529; 26 expansions, 23-action plan
    {"rows": [".......",
             ".......",
             ".......",
             ".......",
             ".......",
             ".......",
             "....>.."],
     "queue": ["ns", "ew", "nw", "ew", "sw", "ne", "ew", "se", "ew", "ns", "ne", "ne", "sw", "ne", "ne", "ns", "ew", "ne", "ew", "ns", "se", "ew", "sw", "nw", "ne", "se", "sw", "ne", "x", "sw", "ns", "se", "ns", "nw", "sw", "se", "ne", "ew", "x", "ew", "ns"],
     "distance": 12, "countdown": 5, "pace": 3},
    # seed 11530; 24 expansions, 23-action plan
    {"rows": ["..#....",
             ".......",
             ".......",
             "...<...",
             ".......",
             ".......",
             "......."],
     "queue": ["ns", "ne", "ew", "ew", "ns", "sw", "ns", "sw", "nw", "se", "ne", "x", "ew", "ew", "ns", "nw", "ns", "ns", "ns", "nw", "ew", "ns", "ew", "sw", "ns", "ew", "ew", "ns", "ns", "ew", "ew", "se", "nw", "ew", "ew", "ns", "ns", "ew", "se", "x", "ns", "ns", "sw"],
     "distance": 12, "countdown": 7, "pace": 3},
    # seed 11531; 173 expansions, 23-action plan
    {"rows": [".......",
             ".v.....",
             ".......",
             ".......",
             ".......",
             ".......",
             "......."],
     "queue": ["nw", "ew", "x", "x", "sw", "ns", "nw", "se", "ne", "ns", "x", "se", "ns", "nw", "ew", "x", "ew", "ne", "ne", "ne", "ns", "nw", "ne", "x", "ns", "se", "nw", "ns", "ne", "ns", "x", "ew"],
     "distance": 8, "countdown": 8, "pace": 3},
    # seed 11532; 24 expansions, 21-action plan
    {"rows": [".......",
             ".......",
             ".......",
             "....<..",
             "...#...",
             "#......",
             "......#"],
     "queue": ["ne", "ne", "x", "x", "ew", "ns", "ew", "se", "nw", "se", "se", "sw", "ns", "nw", "nw", "x", "x", "ew", "se", "ns", "ne", "ew", "ew", "ns", "nw", "sw", "se", "sw", "ns", "ew", "ew", "ew", "nw", "ns", "nw", "ne", "ew", "ns", "ne", "se", "sw", "x", "ne", "ne"],
     "distance": 12, "countdown": 8, "pace": 3},
    # seed 11533; 38 expansions, 22-action plan
    {"rows": ["..#....",
             ".......",
             "..#....",
             ".......",
             ".....#.",
             "^......",
             "......."],
     "queue": ["x", "ne", "se", "nw", "ne", "sw", "ns", "ew", "ns", "se", "se", "ne", "ew", "ns", "ne", "ne", "sw", "se", "ns", "ne", "x", "x", "sw", "nw", "ns", "x", "ne", "nw", "ns", "se", "sw"],
     "distance": 8, "countdown": 7, "pace": 3},
    # seed 11534; 26 expansions, 24-action plan
    {"rows": [".......",
             "..v....",
             ".#.....",
             ".......",
             "...#...",
             ".......",
             "......."],
     "queue": ["ew", "ns", "se", "se", "ne", "ns", "sw", "ne", "ew", "ew", "ew", "se", "ne", "ew", "se", "ns", "ne", "nw", "ns", "nw", "ns", "ew", "ns", "se", "ew", "ew", "ew", "nw", "se", "ns", "ew", "ne"],
     "distance": 9, "countdown": 5, "pace": 3},
    # seed 11535; 19 expansions, 17-action plan
    {"rows": [".....v.",
             ".......",
             ".......",
             "...#...",
             ".......",
             ".....#.",
             "......."],
     "queue": ["sw", "ew", "ne", "sw", "nw", "ew", "ns", "se", "ns", "se", "ew", "ne", "sw", "nw", "nw", "ns", "ns", "x", "x", "nw", "ns", "nw", "x", "ne", "ew", "ew", "ew", "sw", "ne", "se", "ne", "ew"],
     "distance": 9, "countdown": 5, "pace": 3},
    # seed 11536; 13 expansions, 12-action plan
    {"rows": [".......",
             "....#..",
             "..v....",
             ".......",
             ".#.....",
             ".......",
             "......."],
     "queue": ["ne", "ne", "nw", "sw", "x", "ew", "x", "se", "ew", "sw", "x", "nw", "ns", "x", "ew", "ns", "ns", "ns", "ns", "se", "ne", "ew", "nw", "ns", "ew", "ns", "se", "sw", "nw", "ne", "ns", "sw"],
     "distance": 8, "countdown": 8, "pace": 3},
    # seed 11537; 39 expansions, 27-action plan
    {"rows": [".....<.",
             ".#.....",
             ".......",
             "....#..",
             "....#..",
             ".......",
             "......."],
     "queue": ["x", "ns", "ns", "ns", "ew", "sw", "ns", "ne", "ew", "ew", "x", "ns", "se", "ew", "se", "sw", "x", "ns", "ns", "x", "ns", "x", "se", "ne", "se", "ew", "ne", "x", "se", "sw", "ns", "se", "nw", "nw", "nw", "ne", "sw", "se", "ne", "ew", "se", "ne"],
     "distance": 12, "countdown": 6, "pace": 3},
    # seed 11538; 19 expansions, 17-action plan
    {"rows": [".......",
             ".......",
             "#......",
             ".......",
             ".>.....",
             ".......",
             "..#...."],
     "queue": ["ns", "se", "sw", "ns", "ne", "nw", "ew", "ns", "se", "ew", "ew", "ew", "sw", "nw", "ns", "ew", "ew", "ew", "x", "sw", "ew", "ns", "se", "x", "sw", "ew", "ew", "nw", "x", "ns", "ew", "ew", "ne", "ns", "ns", "ne"],
     "distance": 10, "countdown": 6, "pace": 3},
    # seed 11539; 162 expansions, 23-action plan
    {"rows": [".......",
             "...v...",
             ".......",
             ".......",
             ".......",
             ".......",
             "......."],
     "queue": ["se", "ns", "x", "x", "ew", "se", "sw", "ns", "ne", "se", "ns", "x", "x", "sw", "ew", "sw", "sw", "ew", "ns", "nw", "ew", "nw", "se", "se", "ns", "ne", "se", "ew", "ne", "ne", "se", "ns"],
     "distance": 9, "countdown": 5, "pace": 3},
    # seed 11540; 42 expansions, 30-action plan
    {"rows": [".......",
             ".......",
             ".......",
             ".....#.",
             "..>...#",
             ".......",
             "......."],
     "queue": ["ne", "ns", "ne", "ns", "sw", "ew", "ns", "x", "nw", "nw", "nw", "ne", "sw", "ns", "se", "nw", "se", "ne", "ne", "nw", "ns", "nw", "sw", "ew", "ne", "nw", "sw", "ns", "ew", "ns", "sw", "ew", "x", "ew", "ns", "nw", "nw", "ns", "se", "se"],
     "distance": 11, "countdown": 7, "pace": 3},
    # seed 11541; 25 expansions, 21-action plan
    {"rows": [".......",
             ".......",
             ".....^.",
             ".......",
             ".......",
             ".......",
             "....#.."],
     "queue": ["x", "ew", "ns", "ew", "ew", "se", "ns", "nw", "sw", "se", "ew", "nw", "sw", "ns", "ns", "ne", "ne", "ns", "sw", "se", "nw", "ns", "ne", "ew", "ns", "ew", "x", "ns", "ns", "ew"],
     "distance": 8, "countdown": 6, "pace": 3},
    # seed 11542; 17 expansions, 15-action plan
    {"rows": [".......",
             ".......",
             ".......",
             ".......",
             "..>....",
             ".......",
             "......."],
     "queue": ["ns", "ew", "nw", "x", "sw", "se", "se", "ew", "ns", "se", "ne", "ns", "ne", "ew", "ew", "nw", "x", "ns", "ns", "ew", "ns", "ne", "nw", "nw", "ns", "sw", "nw", "se", "nw", "ew", "nw", "ne"],
     "distance": 8, "countdown": 8, "pace": 3},
    # seed 11543; 20 expansions, 18-action plan
    {"rows": [".......",
             "....#..",
             ".......",
             ".....<.",
             ".......",
             ".......",
             "......."],
     "queue": ["sw", "ns", "sw", "sw", "ne", "ew", "se", "se", "ew", "nw", "ne", "ew", "ne", "ew", "ns", "sw", "ne", "ew", "ns", "sw", "se", "ns", "x", "ns", "sw", "se", "nw", "se", "ew", "sw"],
     "distance": 8, "countdown": 6, "pace": 3},
    # seed 11544; 66 expansions, 28-action plan
    {"rows": [".......",
             ".#.....",
             "#......",
             "..<....",
             ".......",
             ".......",
             "......."],
     "queue": ["nw", "ew", "ew", "ne", "ew", "ns", "ew", "nw", "sw", "ns", "se", "ns", "nw", "ew", "nw", "ns", "ne", "nw", "ew", "nw", "sw", "ew", "ne", "nw", "ne", "nw", "sw", "se", "nw", "se", "se", "x", "ne", "ne", "ns", "ne", "se", "ns"],
     "distance": 10, "countdown": 8, "pace": 3},
    # seed 11545; 39 expansions, 25-action plan
    {"rows": [".......",
             ".......",
             ".......",
             ".......",
             ".......",
             ".......",
             "#.....<"],
     "queue": ["x", "sw", "se", "sw", "ew", "ns", "nw", "ns", "nw", "ew", "sw", "nw", "ns", "x", "ns", "ne", "se", "ew", "sw", "ne", "ns", "nw", "ew", "se", "ns", "nw", "nw", "sw", "nw", "ns", "ne", "x", "se", "nw", "ew"],
     "distance": 9, "countdown": 8, "pace": 3},
    # seed 11546; 24 expansions, 21-action plan
    {"rows": [".......",
             "..v....",
             "...#...",
             "......#",
             ".......",
             ".......",
             "......."],
     "queue": ["ne", "ew", "se", "x", "ew", "sw", "sw", "x", "ew", "ew", "nw", "ew", "sw", "ew", "ne", "ns", "ns", "nw", "se", "nw", "se", "nw", "se", "se", "sw", "ne", "ne", "ns", "sw", "nw", "ew", "x", "ew", "sw"],
     "distance": 9, "countdown": 7, "pace": 3},
    # seed 11547; 24 expansions, 22-action plan
    {"rows": ["......<",
             ".....#.",
             ".......",
             ".......",
             "......#",
             ".#.....",
             "......."],
     "queue": ["ns", "ne", "nw", "nw", "nw", "ew", "nw", "ew", "x", "ns", "se", "x", "se", "sw", "ns", "ew", "sw", "ew", "nw", "se", "ns", "sw", "ew", "ew", "nw", "se", "ne", "ns", "ew", "nw", "nw", "ne", "ns", "ne"],
     "distance": 9, "countdown": 7, "pace": 3},
    # seed 11548; 36 expansions, 21-action plan
    {"rows": [".......",
             ".......",
             ".......",
             ".......",
             ".......",
             ".....^.",
             "......."],
     "queue": ["ns", "se", "nw", "ew", "se", "sw", "ne", "ew", "ew", "ne", "se", "se", "ns", "ne", "se", "se", "ns", "ns", "ns", "nw", "sw", "ew", "ne", "ew", "ew", "ns", "ne", "nw", "ns", "ns", "ne", "sw", "ne", "se", "sw", "nw", "ns"],
     "distance": 10, "countdown": 7, "pace": 3},
    # seed 11549; 18 expansions, 17-action plan
    {"rows": [".......",
             ".......",
             ".....^.",
             ".......",
             ".......",
             "....#..",
             "......."],
     "queue": ["nw", "ne", "ns", "ns", "sw", "ne", "sw", "se", "x", "x", "ew", "sw", "nw", "se", "se", "ns", "ew", "nw", "x", "x", "x", "ew", "ne", "ns", "nw", "se", "x", "x", "se", "ew", "ns", "se"],
     "distance": 8, "countdown": 8, "pace": 3},
)
