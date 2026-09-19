"""Artillery: a gun on the left of a hilly field, targets dug in behind the hills, a wind
that blows the same way all game, and a few shells to destroy every target.

The genre is Scorched Earth's and Worms's; the model here is the textbook one. A shell
leaves the gun at an angle and a speed from a small set and flies under gravity and the
wind until it meets the ground or leaves the field. Where it lands it blows a crater out of
the terrain, destroys any target within its radius, and drops a target whose ground was blown
away, which kills it if the fall is long. The effect of a shot is therefore a flight through a
wind field onto a surface that earlier shots have already reshaped, and it is only known by
integrating it. That is what keeps the environment out of PDDL: a crater is not a fact, and a
lob's landing point moves with every hill it clears.

## Determinism

The flight is integrated with fixed steps in plain floating point, so the same shot from the
same state lands in the same place on any machine that rounds the same way, which every one
running the same Python does.

## Instances and generation

`generate_instance(seed, ...)` draws the terrain (a sum of hills and a random walk), sets the
targets in hollows behind hills so that a flat shot is blocked, draws the wind, and keeps the
draw only if a breadth-first search over shots finds a plan of at least two shells within the
budget; the plan is left in `witness`. The bundled instances are such draws, embedded as plain
data with the seed each came from. The method is generate-and-test (search-based procedural
content generation: Togelius et al. 2011, https://doi.org/10.1109/TCIAIG.2011.2148116).
"""
import math

from planiverse.environments.base import Environment
from planiverse.environments.generation import bounded_search, draw_until, rng

#: The field: `WIDTH` columns, heights in units; the gun stands on column `GUN`.
WIDTH, GUN, MAX_HEIGHT = 80, 3, 28
GRAVITY = -20.0
#: Integration step and the most steps a flight may take.
DT, MAX_STEPS = 0.05, 2000
#: The shot alphabet: `angle` degrees above the horizontal, `power` units per second.
ANGLES = (15, 25, 35, 45, 55, 65, 75)
POWERS = (30, 40, 50)
#: A shell's crater radius, and the fall that kills a target whose ground was blown away.
RADIUS, FATAL_FALL = 3.0, 4


class ArtilleryAction:
    """`fire(angle, power)`."""

    def __init__(self, angle, power):
        if angle not in ANGLES or power not in POWERS:
            raise ValueError(f"unknown shot: angle {angle}, power {power}")
        self.angle, self.power = angle, power
        self.name = f"fire({angle},{power})"

    @classmethod
    def parse(cls, text):
        inside = str(text).strip()[len("fire("):-1]
        angle, power = (int(part) for part in inside.split(","))
        return cls(angle, power)

    def cost(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, ArtilleryAction) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    __repr__ = __str__


ACTIONS = tuple(ArtilleryAction(angle, power) for angle in ANGLES for power in POWERS)


class ArtilleryState:
    """The terrain as a tuple of column heights, the targets as `(column, height)` pairs, and
    the shells left. `depth` is bookkeeping and stays out of equality."""

    def __init__(self, heights, targets, shells_left, depth=0):
        self.heights = tuple(heights)
        self.targets = tuple(sorted(targets))
        self.shells_left = shells_left
        self.depth = depth
        literals = [f"height({column}, {height})" for column, height in enumerate(self.heights)]
        literals += [f"target({column}, {height})" for column, height in self.targets]
        literals += [f"shells_left({shells_left})", f"targets_left({len(self.targets)})"]
        self.literals = frozenset(literals)

    @property
    def targets_left(self):
        return len(self.targets)

    def __eq__(self, other):
        return (isinstance(other, ArtilleryState) and self.heights == other.heights
                and self.targets == other.targets and self.shells_left == other.shells_left)

    def __hash__(self):
        return hash((self.heights, self.targets, self.shells_left))

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        rows = []
        top = max(max(self.heights), max((h for _, h in self.targets), default=0)) + 1
        for level in range(top, -1, -1):
            row = []
            for column, height in enumerate(self.heights):
                if (column, level) in self.targets:
                    row.append("T")
                elif column == GUN and level == height + 1:
                    row.append("G")
                else:
                    row.append("#" if level <= height else " ")
            rows.append("".join(row).rstrip())
        rows.append(f"shells left: {self.shells_left}, targets: {len(self.targets)}")
        return "\n".join(rows)

    def __repr__(self):
        return f"<ArtilleryState(targets={len(self.targets)}, shells={self.shells_left})>"


# ------------------------------------------------------------------------------ the physics

def flight(heights, wind, action):
    """Where a shell fired from the gun lands: `(x, y)`, or None if it leaves the field."""
    x, y = GUN + 0.5, heights[GUN] + 1.5
    radians = math.radians(action.angle)
    vx, vy = action.power * math.cos(radians), action.power * math.sin(radians)
    for _ in range(MAX_STEPS):
        vx += wind * DT
        vy += GRAVITY * DT
        x += vx * DT
        y += vy * DT
        if not 0.0 <= x < WIDTH or y < 0.0:
            return None
        if y <= heights[int(x)] + 0.5 and vy < 0:
            return x, y
    return None


def fire(heights, targets, wind, action):
    """Apply one shot: the terrain and targets after the shell lands, or the same ones if it
    left the field."""
    landing = flight(heights, wind, action)
    if landing is None:
        return tuple(heights), tuple(targets)
    x, y = landing
    heights = list(heights)
    for column in range(max(0, int(x - RADIUS)), min(WIDTH, int(x + RADIUS) + 1)):
        dx = column + 0.5 - x
        if abs(dx) <= RADIUS:
            floor = int(y - math.sqrt(RADIUS * RADIUS - dx * dx))
            heights[column] = max(0, min(heights[column], floor))
    survivors = []
    for column, height in targets:
        if math.hypot(column + 0.5 - x, height + 0.5 - y) <= RADIUS:
            continue                                     # blown up
        ground = heights[column] + 1
        if height - ground >= FATAL_FALL:
            continue                                     # fell too far
        survivors.append((column, min(height, ground)))
    return tuple(heights), tuple(survivors)


# ------------------------------------------------------------------------------- the levels

def draw_field(random_, targets=2, shells=None):
    """A random field as the instance dict `set_instance` takes: rolling terrain from a
    sum of hills and a random walk, targets in hollows behind hills, and a wind."""
    heights = []
    phase = [random_.uniform(0, 2 * math.pi) for _ in range(3)]
    walk = 0.0
    for column in range(WIDTH):
        walk += random_.uniform(-0.6, 0.6)
        walk = max(-3.0, min(3.0, walk))
        height = 8.0 + walk
        height += 5.0 * math.sin(column / 9.0 + phase[0])
        height += 3.0 * math.sin(column / 4.5 + phase[1])
        height += 2.0 * math.sin(column / 2.5 + phase[2])
        heights.append(max(2, min(MAX_HEIGHT, int(round(height)))))
    # a hollow is a column lower than the ridge between it and the gun: shots have to come
    # over the top
    hollows = [column for column in range(28, WIDTH - 2)
               if max(heights[column - 8:column]) >= heights[column] + 3]
    random_.shuffle(hollows)
    seats = sorted(hollows[:targets])
    while len(seats) < targets:                          # a flat field: take what there is
        seats.append(random_.randint(28, WIDTH - 3))
    wind = round(random_.uniform(-6.0, 6.0), 1)
    return {"heights": heights, "targets": [[column, heights[column] + 1] for column in seats],
            "wind": wind, "shells": shells or targets + 2}


class ArtilleryEnv(Environment):
    """Destroy every target within the shells given."""

    def __init__(self):
        super().__init__("artillery")
        self.instance = None
        self.index = None
        self.state = None
        self.state_history = []
        self.witness = None
        self.witness_expansions = None

    # ------------------------------------------------------------------ instances

    def set_index(self, index):
        if not 0 <= index < len(FIELDS):
            raise IndexError(f"Invalid index: {index}. There are {len(FIELDS)} fields, so the "
                             f"index must be 0-{len(FIELDS) - 1}.")
        self.set_instance(FIELDS[index])
        self.index = index

    def set_instance(self, instance):
        """Select a field: `{"heights": [...], "targets": [[column, height], ...], "wind": w,
        "shells": n}`."""
        for key in ("heights", "targets", "wind", "shells"):
            if key not in instance:
                raise ValueError(f"a field needs `{key}`")
        if len(instance["heights"]) != WIDTH:
            raise ValueError(f"a field is {WIDTH} columns wide")
        self.instance = {"heights": [int(h) for h in instance["heights"]],
                         "targets": [tuple(int(v) for v in target) for target in instance["targets"]],
                         "wind": float(instance["wind"]), "shells": int(instance["shells"])}
        self.index = None
        self.witness = self.witness_expansions = None

    def generate_instance(self, seed=None, targets=None, shells=None, min_plan_length=2,
                          search_limit=300, attempts=60):
        """Draw a field, select it, and return it as the dict `set_instance` takes.

        `targets` (two or three, drawn when unset) and `shells` (two more than the targets
        when unset) are `draw_field`'s. A draw is kept only if a breadth-first search over
        shots finds a plan within `search_limit` expansions that is at least
        `min_plan_length` shells long, so a field one shell clears is thrown back; the plan
        is left in `witness` and what the search spent in `witness_expansions`.
        """
        random_, _ = rng(seed)
        found = {}

        def draw(attempt):
            return draw_field(random_, targets=targets or random_.randint(2, 3), shells=shells)

        def accept(instance):
            self.set_instance(instance)
            outcome = bounded_search(self, search_limit)
            if outcome.plan is None or len(outcome.plan) < min_plan_length:
                return False
            found["plan"], found["expansions"] = outcome.plan, outcome.expansions
            return True

        instance = draw_until(draw, accept, attempts, "artillery field")
        self.set_instance(instance)
        self.witness, self.witness_expansions = found["plan"], found["expansions"]
        return instance

    # ------------------------------------------------------------------- contract

    def reset(self):
        if self.instance is None:
            self.set_index(0)
        self.state = ArtilleryState(self.instance["heights"], self.instance["targets"],
                                    self.instance["shells"])
        self.state_history = [self.state]
        return self.state, {"field": self.index, "targets": len(self.instance["targets"]),
                            "shells": self.instance["shells"], "wind": self.instance["wind"],
                            "generated": self.index is None}

    def is_goal(self, state):
        return not state.targets

    def is_terminal(self, state):
        return bool(state.targets) and state.shells_left == 0

    def successors(self, state):
        if self.is_goal(state) or self.is_terminal(state):
            return []
        children = []
        for action in ACTIONS:
            child = self.__advance__(state, action)
            if child != state:
                children.append((action, child))
        return children

    def __advance__(self, state, action):
        if self.is_goal(state) or self.is_terminal(state):
            return state
        if not isinstance(action, ArtilleryAction):
            action = ArtilleryAction.parse(action)
        heights, targets = fire(state.heights, state.targets, self.instance["wind"], action)
        return ArtilleryState(heights, targets, state.shells_left - 1, state.depth + 1)

    def simulate(self, plan):
        state, _ = self.reset()
        trace = [state]
        for action in plan:
            trace.append(self.__advance__(trace[-1], action))
        return trace

    def step(self, action):
        if self.state is None:
            raise ValueError("call reset() first")
        before = len(self.state.targets)
        self.state = self.__advance__(self.state, action)
        self.state_history.append(self.state)
        return self.state, before - len(self.state.targets)

    def get_actions(self):
        return list(ACTIONS)

    def render(self):
        lines = [f"step {k}:\n{state}" for k, state in enumerate(self.state_history)]
        print("\n".join(lines))
        return lines


#: The bundled fields: `generate_instance(seed)` for the seed beside each, embedded as the
#: plain data `set_instance` takes, with the plan each was accepted on in
#: `tests/data/artillery_solutions.json`.
FIELDS = (
    # seed 2000; 11 expansions, 2-shell plan
    {"heights": [15, 16, 16, 16, 16, 15, 14, 13, 11, 11, 9, 9, 8, 8, 8, 7, 7, 7, 6, 4, 3, 3, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 7, 7, 6, 5, 5, 4, 2, 2, 2, 2, 2, 2, 2, 2, 3, 5, 6, 7, 8, 8, 8, 9, 9, 10, 10, 12, 12, 14, 15, 16, 17, 17, 16, 15, 14, 12, 10, 8, 6, 4, 3, 3, 2, 2, 2, 2, 2, 3],
     "targets": [[67, 13], [69, 9], [75, 3]], "wind": -2.3, "shells": 5},
    # seed 2001; 48 expansions, 3-shell plan
    {"heights": [8, 9, 11, 12, 13, 13, 14, 14, 13, 12, 12, 12, 11, 11, 11, 10, 11, 11, 11, 10, 9, 7, 5, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8, 7, 6, 5, 5, 5, 5, 5, 6, 8, 9, 10, 12, 13, 13, 14, 13, 13, 11, 10, 10, 8, 7, 5, 5, 4, 3],
     "targets": [[73, 11], [75, 8], [76, 6]], "wind": -2.3, "shells": 5},
    # seed 2002; 140 expansions, 4-shell plan
    {"heights": [13, 14, 15, 16, 16, 15, 15, 14, 13, 13, 12, 12, 12, 12, 12, 12, 11, 10, 9, 8, 6, 4, 3, 2, 2, 2, 2, 2, 2, 3, 5, 7, 8, 10, 10, 11, 11, 11, 11, 10, 9, 9, 10, 10, 11, 12, 12, 13, 13, 14, 13, 13, 13, 13, 13, 13, 12, 13, 13, 14, 15, 17, 18, 18, 19, 18, 17, 16, 15, 13, 11, 9, 8, 7, 6, 6, 5, 5, 6, 5],
     "targets": [[67, 17], [72, 9]], "wind": 4.0, "shells": 4},
    # seed 2003; 4 expansions, 2-shell plan
    {"heights": [4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 6, 8, 10, 11, 12, 13, 14, 14, 14, 13, 13, 13, 12, 12, 12, 12, 13, 13, 14, 14, 13, 13, 12, 10, 9, 9, 8, 8, 8, 9, 10, 11, 13, 13, 13, 13, 13, 12, 11, 10, 8, 6, 5, 4, 3, 2, 3, 3, 4, 4, 5, 6, 6, 6, 6, 6, 6, 7, 7, 8, 9, 10, 12, 13, 14, 15, 16, 16],
     "targets": [[35, 11], [38, 9]], "wind": -3.0, "shells": 4},
    # seed 2004; 129 expansions, 3-shell plan
    {"heights": [8, 7, 7, 7, 6, 5, 5, 6, 6, 7, 8, 9, 9, 10, 10, 10, 11, 10, 10, 9, 10, 10, 11, 12, 13, 15, 17, 17, 18, 19, 18, 18, 16, 15, 13, 12, 10, 9, 8, 7, 7, 8, 8, 8, 8, 8, 7, 6, 6, 6, 5, 5, 5, 5, 6, 7, 8, 9, 10, 11, 11, 11, 11, 10, 9, 9, 8, 7, 7, 7, 7, 8, 9, 10, 12, 13, 14, 15, 16, 16],
     "targets": [[51, 6], [67, 8]], "wind": 2.8, "shells": 4},
    # seed 2005; 3 expansions, 2-shell plan
    {"heights": [2, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 4, 5, 5, 7, 8, 9, 11, 12, 14, 15, 16, 17, 16, 16, 16, 14, 12, 11, 10, 10, 9, 8, 8, 9, 9, 10, 10, 10, 10, 9, 9, 8, 7, 7, 6, 6, 7, 7, 8, 8, 9, 9, 8, 7, 6, 5, 3, 2, 2, 2, 2, 2, 2, 2, 3, 5, 7, 8, 10, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13],
     "targets": [[28, 12], [43, 8]], "wind": 4.8, "shells": 4},
    # seed 2006; 217 expansions, 4-shell plan
    {"heights": [13, 14, 15, 15, 15, 15, 14, 13, 13, 12, 11, 10, 10, 10, 11, 12, 12, 12, 12, 11, 11, 10, 9, 9, 9, 8, 8, 9, 9, 9, 10, 10, 10, 10, 9, 8, 7, 5, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 5, 6, 6, 7, 8, 8, 9, 9, 10, 11, 12, 14, 15, 15, 15, 16, 15, 15, 14, 13, 11, 10, 9, 8, 8, 8, 9, 9, 9, 10],
     "targets": [[70, 12], [71, 11]], "wind": 1.8, "shells": 4},
    # seed 2007; 237 expansions, 4-shell plan
    {"heights": [3, 3, 3, 3, 3, 4, 5, 5, 6, 7, 7, 7, 7, 7, 5, 5, 3, 3, 3, 2, 3, 4, 5, 6, 9, 10, 12, 13, 14, 14, 14, 13, 13, 12, 12, 12, 12, 12, 13, 13, 12, 12, 11, 10, 8, 7, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 5, 5, 5, 4, 3, 3, 3, 3, 2, 3, 3, 4, 4, 5, 6, 6, 6, 5, 5, 5],
     "targets": [[43, 11], [48, 3]], "wind": 5.4, "shells": 4},
    # seed 2008; 11 expansions, 2-shell plan
    {"heights": [10, 9, 9, 8, 8, 7, 5, 3, 2, 2, 2, 2, 2, 2, 3, 4, 6, 8, 9, 10, 11, 12, 12, 12, 11, 11, 12, 12, 12, 13, 13, 14, 15, 15, 15, 14, 14, 13, 12, 11, 10, 10, 10, 11, 11, 13, 13, 15, 15, 16, 16, 15, 15, 14, 12, 11, 10, 8, 7, 6, 5, 4, 3, 4, 4, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 3, 4, 6, 7, 9],
     "targets": [[42, 11], [56, 11]], "wind": 2.0, "shells": 4},
    # seed 2009; 9 expansions, 2-shell plan
    {"heights": [5, 6, 6, 6, 5, 5, 4, 4, 3, 2, 2, 2, 2, 2, 2, 2, 3, 5, 7, 8, 9, 10, 11, 12, 11, 12, 12, 12, 11, 11, 12, 12, 12, 13, 13, 12, 11, 10, 8, 6, 4, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 4, 5, 5, 5, 5, 4, 4, 2, 2, 2, 2, 2, 2, 3, 4, 4, 5, 4, 5, 4, 4, 3, 4, 4, 5, 5, 6, 8, 10],
     "targets": [[43, 3], [61, 3]], "wind": 3.5, "shells": 4},
    # seed 2010; 129 expansions, 4-shell plan
    {"heights": [11, 12, 12, 11, 10, 9, 7, 6, 5, 4, 3, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 5, 6, 9, 10, 12, 13, 14, 14, 14, 13, 11, 10, 10, 10, 9, 9, 8, 8, 9, 9, 9, 9, 8, 8, 8, 7, 6, 6, 6, 5, 5, 6, 6, 6, 7, 7, 8, 7, 6, 6, 5, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
     "targets": [[38, 11], [67, 6]], "wind": 3.0, "shells": 4},
    # seed 2011; 49 expansions, 3-shell plan
    {"heights": [5, 6, 7, 9, 10, 11, 10, 10, 9, 8, 7, 6, 6, 6, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 15, 13, 12, 10, 7, 6, 5, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 9, 9, 8, 8, 8, 8, 8, 8, 9, 9, 10, 11, 11, 11, 12, 12, 12, 11, 10, 10, 10, 10, 10, 11, 13, 14],
     "targets": [[38, 14], [41, 8]], "wind": 3.4, "shells": 4},
    # seed 2012; 72 expansions, 3-shell plan
    {"heights": [11, 10, 10, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 4, 5, 7, 9, 10, 11, 11, 12, 11, 11, 11, 11, 11, 12, 13, 13, 13, 14, 14, 15, 15, 15, 14, 13, 12, 12, 12, 11, 11, 12, 13, 13, 14, 15, 15, 15, 15, 15, 14, 12, 10, 9, 7, 6, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 4, 4, 5, 5],
     "targets": [[61, 13], [64, 8]], "wind": -0.4, "shells": 4},
    # seed 2013; 156 expansions, 4-shell plan
    {"heights": [15, 16, 15, 15, 14, 14, 13, 12, 11, 10, 10, 10, 9, 8, 8, 7, 6, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 5, 6, 7, 7, 7, 6, 6, 6, 5, 5, 5, 5, 6, 6, 7, 7, 8, 8, 8, 8, 7, 7, 6, 6, 6, 7, 8, 9, 10, 11, 12, 13, 14, 13, 13, 11, 10, 8, 7, 6, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2],
     "targets": [[75, 3], [76, 3]], "wind": 1.0, "shells": 4},
    # seed 2014; 51 expansions, 3-shell plan
    {"heights": [8, 8, 7, 7, 7, 8, 9, 9, 9, 9, 8, 6, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 5, 6, 8, 9, 9, 10, 9, 9, 10, 10, 10, 10, 10, 11, 12, 12, 12, 13, 12, 11, 10, 9, 8, 7, 7, 8, 7, 8, 8, 9, 10, 11, 12, 12, 12, 11, 10, 8, 7, 6, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
     "targets": [[44, 11], [51, 9]], "wind": -1.4, "shells": 4},
    # seed 2015; 13 expansions, 2-shell plan
    {"heights": [13, 13, 14, 13, 13, 11, 10, 8, 7, 8, 7, 8, 9, 10, 11, 13, 13, 14, 13, 14, 13, 11, 10, 9, 8, 7, 6, 6, 6, 6, 5, 5, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 5, 6, 8, 9, 9, 9, 9, 9, 8, 7, 7, 8, 7, 8, 9, 9, 10, 10, 11, 11, 10, 9, 9, 9, 8, 8, 7, 8, 8, 9, 10, 11, 12, 12],
     "targets": [[30, 6], [36, 3]], "wind": -2.7, "shells": 4},
    # seed 2016; 9 expansions, 2-shell plan
    {"heights": [13, 14, 16, 17, 17, 18, 17, 16, 15, 13, 11, 9, 8, 7, 6, 5, 5, 5, 4, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 6, 6, 8, 8, 8, 8, 8, 7, 6, 5, 4, 4, 4, 5, 5, 7, 7, 9, 10, 11, 12, 12, 12, 12, 11, 11, 11, 11, 12, 12, 12, 13, 13, 13, 13, 13, 12, 10, 9, 6, 4, 3, 2, 2, 2, 2, 2],
     "targets": [[72, 7], [76, 3]], "wind": -1.9, "shells": 4},
    # seed 2017; 139 expansions, 4-shell plan
    {"heights": [14, 16, 16, 17, 17, 16, 15, 14, 14, 12, 12, 12, 11, 11, 10, 10, 10, 9, 8, 7, 6, 4, 3, 2, 2, 2, 2, 3, 5, 6, 8, 9, 10, 11, 12, 12, 11, 11, 10, 10, 10, 10, 10, 9, 10, 11, 12, 12, 13, 13, 12, 12, 12, 12, 12, 12, 12, 12, 13, 14, 16, 17, 18, 18, 18, 17, 16, 15, 14, 12, 11, 9, 7, 6, 6, 5, 5, 5, 6, 6],
     "targets": [[72, 8], [75, 6]], "wind": 3.6, "shells": 4},
    # seed 2018; 98 expansions, 3-shell plan
    {"heights": [8, 7, 7, 6, 5, 5, 5, 4, 5, 7, 8, 9, 10, 13, 14, 15, 17, 17, 17, 17, 18, 17, 16, 15, 15, 15, 15, 15, 15, 15, 14, 13, 11, 10, 9, 7, 5, 5, 4, 4, 4, 5, 6, 6, 7, 8, 10, 10, 10, 9, 9, 8, 7, 6, 6, 5, 6, 7, 7, 7, 9, 9, 10, 10, 9, 9, 9, 9, 10, 10, 11, 13, 14, 16, 17, 19, 20, 20, 20, 20],
     "targets": [[33, 11], [37, 6], [55, 6]], "wind": -3.8, "shells": 5},
    # seed 2019; 38 expansions, 3-shell plan
    {"heights": [11, 10, 9, 8, 7, 7, 7, 6, 7, 7, 8, 9, 9, 9, 9, 8, 8, 6, 5, 3, 2, 2, 2, 2, 2, 2, 2, 2, 4, 5, 7, 8, 8, 8, 9, 9, 10, 11, 11, 13, 13, 15, 16, 18, 18, 19, 19, 18, 17, 16, 15, 13, 12, 11, 10, 10, 10, 10, 11, 12, 12, 13, 13, 13, 12, 11, 11, 10, 9, 8, 7, 7, 6, 6, 6, 6, 6, 6, 5, 4],
     "targets": [[50, 16], [51, 14]], "wind": 3.3, "shells": 4},
    # seed 2020; 11 expansions, 2-shell plan
    {"heights": [9, 8, 7, 6, 5, 4, 5, 4, 4, 4, 4, 5, 5, 4, 4, 2, 2, 2, 2, 2, 2, 3, 5, 7, 9, 11, 13, 14, 16, 16, 16, 15, 15, 15, 14, 13, 13, 13, 12, 12, 12, 12, 12, 12, 11, 10, 9, 8, 7, 7, 7, 6, 6, 7, 7, 8, 9, 9, 9, 9, 8, 6, 5, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5],
     "targets": [[39, 13], [51, 7]], "wind": 4.3, "shells": 4},
    # seed 2021; 11 expansions, 2-shell plan
    {"heights": [9, 9, 8, 7, 7, 6, 5, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 5, 7, 7, 8, 8, 8, 7, 6, 5, 4, 4, 4, 4, 4, 4, 5, 6, 7, 7, 8, 8, 8, 7, 7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 13, 12, 11, 9, 8, 5, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6],
     "targets": [[30, 5], [65, 3]], "wind": -4.8, "shells": 4},
    # seed 2022; 123 expansions, 3-shell plan
    {"heights": [9, 11, 12, 14, 15, 17, 18, 18, 19, 19, 18, 17, 15, 14, 12, 11, 9, 8, 7, 6, 7, 6, 7, 7, 6, 7, 7, 6, 5, 5, 4, 4, 4, 3, 4, 4, 4, 5, 6, 7, 6, 6, 4, 4, 3, 2, 2, 2, 2, 2, 2, 3, 5, 7, 10, 11, 12, 13, 13, 13, 13, 13, 13, 13, 14, 14, 13, 13, 13, 13, 13, 13, 11, 9, 7, 6, 4, 2, 2, 2],
     "targets": [[30, 5], [76, 5]], "wind": -3.1, "shells": 4},
    # seed 2023; 59 expansions, 3-shell plan
    {"heights": [6, 5, 3, 2, 2, 2, 2, 2, 2, 3, 4, 6, 7, 8, 9, 10, 10, 10, 10, 11, 11, 11, 12, 12, 13, 13, 14, 14, 15, 14, 13, 11, 10, 9, 7, 6, 5, 4, 4, 4, 5, 6, 6, 7, 8, 8, 8, 8, 7, 6, 5, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 7, 9, 11, 13, 13, 14, 14],
     "targets": [[35, 7], [50, 6], [52, 4]], "wind": -3.5, "shells": 5},
    # seed 2024; 11 expansions, 2-shell plan
    {"heights": [9, 8, 8, 9, 9, 10, 9, 10, 9, 8, 7, 6, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 5, 5, 5, 6, 6, 6, 7, 8, 8, 9, 10, 11, 12, 12, 12, 13, 12, 11, 10, 8, 7, 7, 6, 6, 6, 6, 7, 7, 8, 9, 9, 9, 9, 9, 7, 6, 5, 5, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2],
     "targets": [[45, 9], [63, 6], [64, 5]], "wind": -5.7, "shells": 5},
    # seed 2025; 30 expansions, 3-shell plan
    {"heights": [4, 4, 5, 5, 7, 8, 9, 10, 10, 9, 8, 8, 7, 6, 5, 4, 5, 5, 6, 7, 9, 10, 12, 12, 14, 15, 15, 14, 14, 13, 12, 12, 11, 10, 9, 9, 8, 8, 8, 7, 6, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 4, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 5, 6, 6, 7, 7, 7, 7, 7, 6, 6, 6, 6, 7],
     "targets": [[33, 11], [34, 10]], "wind": 0.8, "shells": 4},
    # seed 2026; 42 expansions, 3-shell plan
    {"heights": [11, 11, 12, 13, 13, 14, 15, 16, 16, 16, 15, 14, 12, 10, 7, 5, 4, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 3, 4, 5, 5, 7, 8, 9, 10, 12, 13, 14, 14, 14, 14, 13, 12, 11, 11, 10, 11, 11, 12, 13, 14, 15, 16, 16, 17, 17, 17, 16, 15, 15, 14, 13, 13, 13, 13, 12, 12, 11, 11, 10, 9, 8, 6, 4, 3, 2, 2],
     "targets": [[48, 11], [63, 15]], "wind": -0.5, "shells": 4},
    # seed 2027; 176 expansions, 4-shell plan
    {"heights": [10, 12, 13, 14, 14, 15, 14, 14, 14, 14, 14, 15, 16, 16, 17, 17, 16, 15, 13, 12, 10, 8, 7, 5, 4, 4, 4, 4, 5, 6, 6, 7, 8, 8, 8, 8, 7, 7, 5, 5, 5, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 6, 6, 6, 6, 6, 7, 9, 10, 12, 13, 15, 17, 17, 18, 19, 19, 19, 18, 16, 16, 14, 12, 12, 11, 10, 10, 9, 9, 8],
     "targets": [[74, 12], [76, 11]], "wind": 4.6, "shells": 4},
    # seed 2028; 43 expansions, 3-shell plan
    {"heights": [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5, 7, 7, 8, 9, 9, 10, 10, 10, 9, 8, 7, 6, 5, 5, 5, 5, 6, 8, 9, 10, 10, 12, 12, 12, 13, 12, 11, 10, 9, 9, 8, 7, 7, 6, 6, 5, 4, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8],
     "targets": [[33, 7], [50, 7]], "wind": 2.8, "shells": 4},
    # seed 2029; 122 expansions, 4-shell plan
    {"heights": [13, 14, 15, 15, 15, 15, 14, 13, 12, 10, 10, 9, 8, 8, 8, 8, 8, 9, 9, 9, 8, 7, 6, 6, 6, 5, 5, 4, 4, 5, 5, 6, 6, 6, 6, 5, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9, 10, 11, 11, 13, 14, 14, 13, 12, 12, 11, 10, 8, 6, 5, 5, 5, 6, 6, 7, 7],
     "targets": [[72, 7], [73, 6], [76, 7]], "wind": 4.9, "shells": 5},
    # seed 2030; 57 expansions, 4-shell plan
    {"heights": [10, 11, 12, 12, 13, 13, 13, 12, 11, 9, 8, 7, 6, 6, 5, 5, 4, 4, 4, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 6, 7, 9, 11, 12, 13, 13, 13, 12, 11, 10, 9, 9, 9, 10, 9, 9, 9, 9, 9, 9, 9, 8, 7, 6, 6, 5, 6, 5, 6, 7, 7, 8, 9, 9, 9, 8, 7, 6, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2],
     "targets": [[42, 10], [45, 11], [77, 3]], "wind": 6.0, "shells": 5},
    # seed 2031; 123 expansions, 3-shell plan
    {"heights": [2, 4, 5, 5, 6, 6, 6, 7, 7, 8, 9, 10, 12, 14, 15, 16, 16, 17, 17, 17, 16, 15, 15, 13, 12, 10, 10, 10, 10, 11, 11, 12, 12, 12, 12, 12, 12, 11, 10, 9, 9, 9, 8, 7, 8, 8, 9, 9, 9, 8, 8, 7, 5, 4, 3, 2, 2, 2, 2, 3, 5, 6, 8, 9, 10, 11, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15],
     "targets": [[30, 12], [42, 9], [54, 4]], "wind": -1.6, "shells": 5},
    # seed 2032; 11 expansions, 2-shell plan
    {"heights": [8, 9, 9, 9, 10, 10, 11, 11, 12, 13, 14, 14, 14, 15, 15, 15, 14, 12, 11, 10, 8, 7, 5, 5, 4, 5, 6, 6, 7, 8, 8, 8, 9, 8, 8, 6, 5, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 5, 7, 9, 11, 12, 14, 14, 13, 13, 12, 11, 10, 9, 9, 9, 9, 9, 10, 10, 9, 9, 9],
     "targets": [[42, 4], [69, 11]], "wind": 0.4, "shells": 4},
    # seed 2033; 126 expansions, 3-shell plan
    {"heights": [13, 14, 14, 15, 16, 16, 17, 17, 18, 18, 17, 16, 15, 14, 13, 11, 9, 8, 8, 7, 6, 6, 6, 8, 8, 8, 9, 9, 9, 9, 8, 8, 8, 7, 7, 6, 7, 7, 7, 7, 8, 8, 8, 7, 6, 6, 5, 5, 5, 5, 6, 7, 9, 11, 13, 15, 17, 19, 19, 20, 20, 19, 19, 18, 16, 15, 14, 13, 13, 12, 12, 12, 11, 11, 11, 10, 9, 8, 6, 6],
     "targets": [[72, 12], [73, 12], [75, 11]], "wind": 0.3, "shells": 5},
    # seed 2034; 187 expansions, 3-shell plan
    {"heights": [2, 2, 2, 3, 4, 5, 5, 6, 6, 7, 6, 6, 5, 4, 3, 2, 2, 2, 2, 2, 2, 3, 5, 7, 7, 9, 10, 10, 10, 10, 10, 11, 12, 12, 12, 13, 14, 15, 15, 15, 15, 14, 13, 11, 10, 7, 5, 4, 3, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 4, 3, 3, 2, 2, 2, 2, 2, 2, 3, 3, 4, 5, 4, 4, 4, 3, 2, 2, 2, 2],
     "targets": [[43, 12], [45, 8], [50, 3]], "wind": -1.9, "shells": 5},
    # seed 2035; 4 expansions, 2-shell plan
    {"heights": [10, 9, 7, 6, 5, 4, 3, 3, 3, 3, 3, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 3, 4, 5, 7, 9, 11, 12, 14, 14, 14, 14, 13, 13, 11, 11, 11, 10, 10, 10, 11, 11, 12, 12, 12, 12, 12, 11, 10, 9, 9, 9, 9, 10, 10, 10, 11, 11, 11, 10, 9, 8, 6, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 5],
     "targets": [[34, 12], [39, 11]], "wind": -0.9, "shells": 4},
    # seed 2036; 10 expansions, 2-shell plan
    {"heights": [13, 13, 14, 15, 16, 16, 16, 15, 13, 11, 9, 6, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 5, 7, 7, 9, 9, 9, 9, 8, 8, 7, 6, 6, 5, 5, 5, 6, 7, 8, 8, 10, 10, 11, 11, 10, 10, 10, 9, 9, 9, 9, 9, 10, 10, 10, 10, 9, 8, 7, 5, 3, 2, 2, 2, 2, 2, 2, 2, 2],
     "targets": [[71, 4], [75, 3]], "wind": 0.3, "shells": 4},
    # seed 2037; 90 expansions, 3-shell plan
    {"heights": [2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 13, 14, 13, 12, 12, 11, 10, 8, 8, 7, 7, 8, 8, 9, 9, 10, 11, 11, 11, 11, 11, 10, 9, 9, 8, 8, 7, 7, 8, 8, 9, 8, 7, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 5, 7, 7, 8, 9, 10, 10, 10, 10, 11, 11, 11, 12, 12, 13, 14],
     "targets": [[30, 9], [53, 4]], "wind": 4.3, "shells": 4},
    # seed 2038; 14 expansions, 2-shell plan
    {"heights": [11, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 14, 12, 10, 8, 6, 5, 4, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4, 5, 5, 7, 8, 10, 12, 13, 14, 14, 15, 14, 14, 13, 12, 11, 10, 10, 10, 11, 11, 12, 14, 14, 15, 16, 16, 16, 16, 15, 15, 14, 14, 13, 12, 12, 12, 12, 11],
     "targets": [[60, 12], [61, 12]], "wind": 1.8, "shells": 4},
    # seed 2039; 5 expansions, 2-shell plan
    {"heights": [10, 11, 11, 12, 12, 12, 12, 13, 15, 15, 16, 16, 17, 17, 17, 16, 15, 14, 12, 10, 8, 7, 6, 5, 6, 7, 8, 8, 10, 10, 11, 11, 10, 10, 9, 8, 8, 7, 6, 6, 5, 6, 6, 7, 7, 8, 7, 7, 7, 6, 5, 5, 5, 6, 6, 7, 9, 11, 13, 15, 17, 18, 19, 19, 19, 18, 18, 17, 16, 15, 14, 14, 14, 13, 13, 13, 12, 12, 11, 10],
     "targets": [[37, 8], [42, 7]], "wind": -5.4, "shells": 4},
    # seed 2040; 50 expansions, 3-shell plan
    {"heights": [10, 9, 8, 8, 7, 6, 6, 6, 7, 8, 9, 9, 10, 10, 10, 9, 8, 6, 5, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 4, 5, 5, 5, 5, 6, 7, 8, 9, 10, 12, 13, 13, 15, 15, 15, 14, 13, 11, 9, 8, 7, 6, 6, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 6, 5, 5, 5, 4, 5, 5, 6, 6, 6, 6, 5, 4, 4, 2],
     "targets": [[50, 10], [53, 7], [56, 6]], "wind": -1.6, "shells": 5},
    # seed 2041; 109 expansions, 4-shell plan
    {"heights": [13, 14, 15, 16, 16, 16, 15, 14, 12, 10, 8, 7, 6, 6, 5, 5, 6, 6, 7, 7, 8, 8, 8, 7, 6, 6, 5, 4, 4, 4, 3, 3, 3, 3, 4, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 6, 7, 8, 10, 11, 11, 12, 12, 12, 12, 12, 13, 12, 12, 12, 13, 14, 14, 13, 13, 12, 10, 9, 8, 8, 8, 8, 7, 8, 9, 9],
     "targets": [[70, 11], [72, 9], [75, 9]], "wind": 5.1, "shells": 5},
    # seed 2042; 40 expansions, 3-shell plan
    {"heights": [9, 10, 9, 8, 7, 6, 4, 3, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 9, 10, 11, 12, 12, 14, 14, 15, 15, 15, 14, 13, 12, 11, 10, 8, 7, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 7, 7, 6, 5, 5, 5, 5, 6, 6, 6, 7, 6, 5, 4, 3, 3, 2, 2, 2, 3, 3, 4, 6, 8, 9, 11, 12],
     "targets": [[39, 9], [56, 6], [58, 6]], "wind": 3.0, "shells": 5},
    # seed 2043; 12 expansions, 2-shell plan
    {"heights": [8, 8, 9, 9, 10, 10, 12, 12, 13, 13, 12, 12, 11, 11, 10, 9, 9, 8, 9, 9, 10, 12, 13, 14, 14, 14, 14, 13, 12, 10, 9, 7, 6, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 6, 8, 9, 10, 11, 11, 10, 9, 8, 8, 7, 7, 7, 7, 7, 8, 8, 8, 9, 9, 10, 10, 10, 10, 10, 10, 9],
     "targets": [[39, 3], [62, 9], [63, 8]], "wind": -3.3, "shells": 5},
    # seed 2044; 256 expansions, 4-shell plan
    {"heights": [3, 3, 3, 5, 5, 7, 8, 8, 8, 8, 7, 6, 5, 4, 4, 3, 3, 3, 4, 5, 7, 9, 11, 13, 14, 14, 14, 15, 15, 14, 15, 15, 14, 14, 14, 14, 14, 14, 13, 12, 11, 9, 7, 5, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 5, 5, 4, 4, 4, 3, 3, 3, 3, 4, 5, 5, 6, 6, 6, 5, 4, 4, 4, 4, 4, 4, 5],
     "targets": [[43, 6], [45, 3]], "wind": 4.2, "shells": 4},
    # seed 2045; 2 expansions, 2-shell plan
    {"heights": [14, 15, 16, 16, 16, 16, 15, 14, 12, 10, 8, 6, 4, 3, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 4, 3, 2, 2, 2, 2, 3, 3, 3, 3, 5, 5, 5, 5, 4, 4, 2, 2, 2, 2, 2, 2, 2, 3, 6, 7, 8, 10, 12, 13, 13, 13, 13, 12, 12, 11, 11, 11, 10, 10, 10, 9, 9, 8, 8, 7, 5, 4, 2, 2, 2, 2, 2, 2, 2, 2],
     "targets": [[45, 3], [71, 5]], "wind": -2.2, "shells": 4},
    # seed 2046; 114 expansions, 4-shell plan
    {"heights": [14, 16, 18, 19, 19, 18, 17, 16, 14, 13, 12, 11, 10, 9, 8, 8, 8, 8, 9, 8, 8, 8, 8, 7, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 10, 10, 11, 12, 12, 13, 14, 14, 13, 13, 11, 9, 7, 6, 5, 4, 3, 3, 3, 4, 5, 5],
     "targets": [[71, 7], [77, 5]], "wind": 5.0, "shells": 4},
    # seed 2047; 3 expansions, 2-shell plan
    {"heights": [12, 10, 8, 6, 4, 3, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 2, 2, 2, 3, 3, 4, 5, 6, 7, 7, 8, 9, 9, 9, 8, 8, 7, 6, 6, 5, 6, 6, 7, 8, 10, 11, 11, 13, 14, 15, 15, 14, 13, 12, 11, 10, 9, 9, 9, 9, 9, 8, 8, 7, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 6, 6],
     "targets": [[38, 7], [65, 3]], "wind": 5.6, "shells": 4},
    # seed 2048; 150 expansions, 3-shell plan
    {"heights": [4, 3, 3, 3, 4, 5, 6, 6, 6, 7, 6, 6, 5, 5, 4, 4, 4, 4, 5, 5, 6, 8, 10, 13, 14, 15, 17, 18, 17, 17, 17, 16, 14, 13, 12, 11, 11, 10, 10, 10, 9, 9, 8, 7, 5, 4, 2, 2, 2, 2, 2, 2, 2, 4, 4, 5, 6, 8, 8, 8, 8, 7, 5, 5, 4, 4, 4, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 9, 9, 10],
     "targets": [[40, 10], [45, 5]], "wind": -0.1, "shells": 4},
    # seed 2049; 57 expansions, 3-shell plan
    {"heights": [3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 12, 12, 12, 11, 10, 9, 9, 9, 9, 9, 8, 9, 9, 10, 9, 8, 8, 7, 7, 7, 6, 6, 6, 6, 7, 8, 8, 9, 9, 9, 9, 8, 6, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 5, 5, 6, 7, 8, 9, 11, 11, 12, 12],
     "targets": [[36, 8], [51, 7]], "wind": -4.1, "shells": 4},
    # seed 2050; 49 expansions, 3-shell plan
    {"heights": [2, 3, 5, 7, 9, 10, 11, 12, 13, 13, 13, 12, 12, 12, 11, 12, 12, 13, 14, 14, 14, 15, 14, 14, 14, 14, 13, 12, 13, 13, 13, 14, 14, 15, 16, 17, 17, 17, 16, 15, 14, 12, 10, 8, 7, 5, 5, 4, 4, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 12, 11, 11, 10, 9, 8, 7, 6, 6, 6, 7],
     "targets": [[41, 13], [72, 11], [73, 10]], "wind": -0.7, "shells": 5},
    # seed 2051; 40 expansions, 3-shell plan
    {"heights": [3, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 8, 8, 8, 8, 9, 9, 9, 10, 12, 14, 15, 16, 17, 18, 18, 18, 17, 15, 14, 12, 11, 10, 9, 9, 9, 9, 10, 11, 11, 11, 11, 12, 12, 11, 11, 10, 9, 8, 8, 7, 7, 6, 7, 7, 7, 7, 6, 6, 5, 4, 4, 3, 2, 2, 3, 4, 5, 7, 9, 11, 12, 14, 15, 16, 17, 16, 16, 16],
     "targets": [[29, 16], [64, 3]], "wind": -3.7, "shells": 4},
    # seed 2052; 67 expansions, 3-shell plan
    {"heights": [10, 9, 9, 8, 7, 8, 7, 8, 8, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 16, 17, 17, 17, 17, 17, 16, 14, 13, 11, 10, 7, 6, 4, 3, 3, 3, 4, 4, 5, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 9, 9, 10, 11, 12, 13, 13, 13, 13, 12, 12, 10, 10, 9, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 18, 17, 17],
     "targets": [[30, 15], [69, 11]], "wind": 5.5, "shells": 4},
    # seed 2053; 37 expansions, 3-shell plan
    {"heights": [8, 8, 8, 8, 8, 7, 7, 6, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 4, 6, 8, 10, 11, 12, 13, 13, 13, 14, 14, 14, 14, 14, 14, 15, 15, 15, 16, 15, 15, 14, 13, 12, 12, 10, 10, 10, 9, 10, 11, 12, 13, 15, 15, 15, 15, 14, 13, 12, 10, 9, 7, 6, 4, 4, 4, 4, 4, 4, 4, 5, 4, 3, 3, 3, 3, 3, 4, 5, 7, 8],
     "targets": [[41, 13], [42, 13], [57, 13]], "wind": 2.2, "shells": 5},
    # seed 2054; 14 expansions, 2-shell plan
    {"heights": [3, 2, 2, 2, 2, 2, 3, 4, 4, 6, 6, 7, 8, 8, 8, 7, 6, 5, 4, 3, 3, 4, 5, 6, 7, 9, 10, 11, 12, 12, 13, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 12, 11, 10, 8, 7, 6, 4, 3, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 8, 9, 9, 9, 9, 9, 9, 9, 9],
     "targets": [[48, 5], [50, 3], [55, 3]], "wind": -1.5, "shells": 5},
    # seed 2055; 8 expansions, 2-shell plan
    {"heights": [5, 4, 5, 5, 5, 6, 6, 6, 8, 9, 10, 12, 14, 15, 17, 17, 18, 18, 17, 15, 14, 12, 11, 10, 9, 9, 9, 8, 9, 9, 9, 9, 8, 8, 7, 6, 5, 5, 5, 4, 4, 5, 6, 6, 7, 7, 6, 6, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 5, 6, 8, 10, 12, 12, 13, 13, 13, 13, 14, 14, 15, 15, 16, 16, 16, 16, 16, 14],
     "targets": [[37, 6], [41, 6]], "wind": -5.1, "shells": 4},
    # seed 2056; 205 expansions, 3-shell plan
    {"heights": [3, 2, 2, 3, 3, 4, 4, 4, 4, 3, 3, 3, 4, 4, 4, 5, 7, 9, 10, 11, 12, 12, 12, 11, 11, 10, 9, 9, 9, 9, 10, 10, 11, 12, 13, 14, 15, 15, 16, 16, 16, 16, 16, 15, 15, 15, 14, 14, 15, 15, 14, 13, 12, 10, 8, 7, 5, 4, 2, 2, 2, 2, 3, 4, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9, 9, 8, 8, 9, 9],
     "targets": [[58, 3], [63, 5]], "wind": -3.6, "shells": 4},
    # seed 2057; 9 expansions, 2-shell plan
    {"heights": [9, 8, 7, 6, 6, 6, 6, 6, 7, 7, 8, 8, 8, 9, 8, 8, 8, 8, 8, 9, 10, 11, 11, 12, 13, 13, 13, 12, 11, 9, 8, 6, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 5, 5, 6, 7, 8, 8, 9, 8, 8, 7, 6, 5, 5, 5, 5, 6, 7, 8, 9, 10, 11, 12, 12, 12, 12, 11, 11],
     "targets": [[34, 3], [65, 6]], "wind": -3.9, "shells": 4},
    # seed 2058; 117 expansions, 3-shell plan
    {"heights": [7, 8, 10, 11, 13, 15, 16, 17, 18, 18, 18, 18, 16, 15, 13, 12, 11, 11, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 8, 7, 6, 6, 7, 7, 8, 8, 9, 10, 10, 10, 10, 10, 9, 8, 6, 5, 3, 3, 2, 3, 3, 4, 5, 6, 8, 9, 11, 12, 12, 13, 13, 13, 14, 14, 14, 15, 16, 16, 17, 17, 17, 17, 16, 15, 14, 12, 10, 8, 7, 6],
     "targets": [[31, 7], [44, 7]], "wind": -4.4, "shells": 4},
    # seed 2059; 27 expansions, 3-shell plan
    {"heights": [13, 13, 13, 12, 11, 10, 9, 7, 6, 5, 5, 5, 6, 7, 8, 9, 10, 11, 12, 12, 11, 10, 9, 8, 7, 6, 5, 4, 4, 4, 4, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2, 3, 4, 5, 7, 9, 11, 13, 15, 16, 17, 17, 17, 16, 16, 14, 13, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 12, 11, 10, 10, 10, 10, 10, 10, 10, 11, 12, 12],
     "targets": [[39, 3], [60, 13], [71, 11]], "wind": -3.2, "shells": 5},
    # seed 2060; 59 expansions, 3-shell plan
    {"heights": [12, 12, 13, 14, 14, 15, 15, 15, 15, 14, 13, 11, 9, 6, 5, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 3, 3, 4, 4, 5, 6, 7, 7, 7, 8, 8, 8, 7, 7, 6, 6, 5, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 16, 16, 16, 15, 14, 13, 12, 11, 11, 10, 9, 9, 8, 7, 7, 6, 5, 4, 2, 2, 2, 2, 2],
     "targets": [[62, 14], [68, 10]], "wind": -5.6, "shells": 4},
    # seed 2061; 4 expansions, 2-shell plan
    {"heights": [7, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 5, 6, 7, 8, 9, 9, 9, 9, 10, 11, 11, 11, 12, 13, 12, 13, 13, 13, 13, 11, 10, 9, 8, 7, 7, 8, 9, 10, 11, 12, 13, 13, 14, 14, 14, 13, 12, 11, 9, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 3, 2, 2, 2, 3, 3, 4, 6, 7, 9, 10, 12, 13, 13, 13],
     "targets": [[36, 9], [37, 8], [39, 9]], "wind": -3.8, "shells": 5},
    # seed 2062; 125 expansions, 3-shell plan
    {"heights": [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 6, 8, 9, 11, 12, 12, 13, 13, 13, 13, 11, 11, 10, 9, 8, 8, 8, 8, 9, 10, 11, 11, 12, 13, 12, 13, 12, 12, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 8, 7, 6, 5, 3, 2, 2, 2, 2, 2, 2, 3, 5, 6, 8, 9, 10, 11, 10, 11, 10, 10, 11, 11],
     "targets": [[48, 11], [63, 3], [66, 3]], "wind": 2.8, "shells": 5},
    # seed 2063; 4 expansions, 2-shell plan
    {"heights": [4, 4, 4, 4, 4, 5, 5, 5, 6, 7, 9, 10, 12, 13, 15, 16, 17, 16, 16, 15, 14, 13, 11, 10, 10, 10, 9, 9, 10, 10, 9, 10, 10, 10, 9, 8, 7, 7, 7, 6, 6, 7, 7, 8, 8, 9, 9, 10, 9, 8, 6, 5, 3, 3, 2, 2, 2, 2, 3, 3, 5, 6, 8, 10, 10, 11, 12, 12, 12, 12, 13, 14, 15, 16, 17, 18, 18, 19, 18, 18],
     "targets": [[56, 3], [58, 4]], "wind": 3.6, "shells": 4},
    # seed 2064; 15 expansions, 2-shell plan
    {"heights": [6, 6, 5, 6, 5, 5, 5, 6, 6, 7, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 7, 9, 10, 12, 14, 16, 17, 18, 19, 19, 19, 19, 18, 17, 16, 16, 15, 15, 15, 15, 15, 14, 14, 13, 11, 9, 8, 6, 5, 3, 2, 3, 3, 4, 5, 7, 7, 9, 9, 9, 9, 8, 7, 6, 5, 4, 3, 3, 4, 4, 5, 6, 7, 7, 7, 7, 7, 8, 8],
     "targets": [[38, 16], [46, 10], [48, 7]], "wind": -2.5, "shells": 5},
    # seed 2065; 91 expansions, 3-shell plan
    {"heights": [8, 9, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 13, 14, 14, 15, 15, 15, 14, 13, 12, 10, 7, 6, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 6, 7, 8, 9, 9, 10, 9, 9, 8, 7, 6, 6, 5, 6, 7, 8, 9, 11, 13, 14, 16, 17, 17, 16, 16, 16, 15, 14, 13, 13, 13, 13, 13, 13, 12],
     "targets": [[54, 8], [55, 7], [74, 14]], "wind": -0.2, "shells": 5},
    # seed 2066; 101 expansions, 3-shell plan
    {"heights": [5, 6, 7, 8, 9, 9, 10, 10, 10, 10, 11, 12, 12, 14, 15, 16, 17, 18, 18, 18, 18, 17, 16, 14, 12, 11, 9, 8, 7, 6, 6, 6, 7, 7, 8, 7, 8, 7, 7, 7, 6, 6, 7, 7, 7, 7, 8, 9, 9, 10, 10, 10, 10, 9, 8, 7, 6, 6, 6, 6, 8, 9, 11, 13, 14, 15, 16, 17, 18, 18, 18, 17, 16, 16, 15, 15, 15, 15, 14, 14],
     "targets": [[28, 8], [31, 7], [59, 7]], "wind": -2.0, "shells": 5},
    # seed 2067; 5 expansions, 2-shell plan
    {"heights": [7, 6, 6, 5, 5, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 5, 6, 7, 8, 9, 9, 9, 9, 7, 6, 6, 4, 5, 5, 5, 6, 6, 7, 8, 8, 9, 10, 10, 10, 11, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 14, 13, 12, 10, 8, 7, 5, 4, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 5, 7, 7],
     "targets": [[30, 7], [62, 3]], "wind": -2.5, "shells": 4},
    # seed 2068; 13 expansions, 2-shell plan
    {"heights": [4, 4, 3, 3, 4, 4, 5, 6, 7, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 11, 11, 12, 13, 14, 15, 14, 14, 13, 12, 10, 7, 6, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 5, 5, 4, 4, 3, 2, 2, 2, 2, 2, 3, 5, 7, 9, 11, 12, 13, 14, 14, 14, 14, 14, 13],
     "targets": [[30, 8], [34, 3], [35, 3]], "wind": -4.3, "shells": 5},
    # seed 2069; 45 expansions, 3-shell plan
    {"heights": [10, 10, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 14, 14, 14, 14, 13, 12, 11, 10, 8, 6, 5, 4, 4, 4, 4, 5, 7, 7, 8, 9, 9, 8, 7, 6, 5, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 5, 7, 9, 11, 13, 14, 15, 15, 14, 14, 13, 13, 11, 11, 9, 9, 8, 8, 7, 8, 7, 7, 8],
     "targets": [[39, 3], [69, 12]], "wind": -0.7, "shells": 4},
    # seed 2070; 234 expansions, 4-shell plan
    {"heights": [17, 18, 17, 16, 15, 14, 13, 12, 11, 10, 10, 9, 10, 10, 9, 8, 8, 7, 7, 6, 5, 4, 3, 2, 2, 3, 5, 6, 6, 7, 8, 9, 9, 8, 7, 5, 4, 4, 2, 2, 2, 2, 3, 4, 5, 6, 6, 7, 8, 9, 9, 10, 10, 10, 11, 12, 12, 13, 15, 16, 16, 17, 16, 15, 14, 13, 12, 10, 8, 7, 6, 5, 5, 5, 6, 6, 7, 7, 7, 8],
     "targets": [[42, 4], [65, 14], [68, 9]], "wind": -4.0, "shells": 5},
    # seed 2071; 131 expansions, 3-shell plan
    {"heights": [14, 15, 14, 14, 13, 13, 12, 12, 12, 12, 11, 11, 10, 9, 7, 6, 5, 4, 3, 2, 2, 2, 2, 2, 2, 3, 4, 6, 8, 9, 9, 9, 9, 9, 9, 8, 9, 9, 9, 9, 10, 11, 11, 12, 13, 13, 12, 12, 11, 11, 11, 10, 11, 12, 12, 13, 14, 16, 16, 17, 18, 17, 17, 16, 13, 12, 10, 9, 7, 6, 6, 5, 4, 4, 4, 4, 3, 3, 3, 2],
     "targets": [[72, 5], [77, 4]], "wind": -2.3, "shells": 4},
    # seed 2072; 12 expansions, 2-shell plan
    {"heights": [5, 7, 8, 9, 9, 10, 10, 10, 9, 8, 7, 7, 7, 7, 8, 8, 9, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 13, 13, 14, 15, 16, 16, 16, 15, 14, 13, 11, 9, 7, 5, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 4, 5, 6, 7, 8, 9, 11, 11, 11, 10, 10, 9, 7, 7, 6, 5, 5, 6, 6, 7, 9, 10],
     "targets": [[44, 3], [72, 7]], "wind": -1.6, "shells": 4},
    # seed 2073; 17 expansions, 2-shell plan
    {"heights": [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 6, 7, 9, 11, 12, 12, 12, 12, 12, 11, 9, 9, 8, 8, 8, 9, 10, 11, 12, 13, 13, 13, 14, 13, 13, 13, 12, 12, 12, 12, 12, 12, 12, 11, 11, 11, 10, 9, 8, 6, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 8],
     "targets": [[61, 3], [63, 3]], "wind": -3.5, "shells": 4},
    # seed 2074; 95 expansions, 3-shell plan
    {"heights": [6, 4, 3, 2, 2, 2, 2, 2, 3, 3, 4, 5, 5, 4, 4, 4, 5, 5, 7, 7, 9, 10, 12, 13, 15, 17, 18, 17, 18, 17, 16, 15, 14, 13, 12, 11, 11, 11, 11, 12, 13, 13, 13, 12, 12, 11, 10, 10, 9, 9, 9, 8, 9, 9, 9, 9, 9, 9, 8, 7, 5, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 6, 8, 9, 10, 10, 10, 10, 11],
     "targets": [[46, 11], [47, 11]], "wind": 5.7, "shells": 4},
    # seed 2075; 9 expansions, 2-shell plan
    {"heights": [6, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 4, 5, 6, 7, 8, 8, 8, 8, 8, 7, 6, 5, 6, 6, 6, 7, 7, 8, 8, 8, 9, 8, 8, 7, 7, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 12, 11, 8, 7, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8, 8, 7],
     "targets": [[48, 13], [54, 3], [56, 3]], "wind": -4.5, "shells": 5},
    # seed 2076; 233 expansions, 4-shell plan
    {"heights": [8, 7, 7, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 3, 4, 6, 7, 8, 10, 10, 11, 12, 13, 14, 14, 15, 16, 18, 18, 19, 19, 19, 19, 18, 16, 14, 11, 10, 9, 7, 7, 6, 6, 7, 7, 8, 9, 9, 9, 8, 7, 7, 6, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 5, 5, 5, 4, 4, 3, 3, 3, 4, 6, 7, 9, 11, 13, 14, 15, 17],
     "targets": [[36, 12], [39, 8], [40, 8]], "wind": 4.5, "shells": 5},
    # seed 2077; 62 expansions, 3-shell plan
    {"heights": [7, 8, 8, 8, 8, 7, 7, 6, 5, 4, 4, 4, 4, 6, 7, 9, 10, 11, 12, 13, 13, 13, 14, 13, 14, 14, 14, 15, 16, 17, 17, 17, 17, 17, 16, 15, 13, 11, 8, 7, 5, 4, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 5, 6, 7, 7, 8, 9, 9, 9, 9, 9, 8, 8, 8, 7, 7, 7, 8, 8, 9, 10, 12, 13, 15, 17],
     "targets": [[38, 9], [40, 6]], "wind": 0.2, "shells": 4},
    # seed 2078; 37 expansions, 3-shell plan
    {"heights": [8, 9, 11, 11, 12, 12, 12, 13, 13, 14, 14, 15, 16, 17, 17, 17, 16, 15, 14, 13, 11, 9, 8, 6, 6, 5, 5, 4, 5, 5, 5, 6, 7, 7, 7, 6, 6, 6, 6, 5, 4, 4, 4, 4, 5, 6, 6, 7, 6, 6, 6, 5, 5, 4, 3, 4, 4, 5, 6, 7, 9, 11, 12, 14, 15, 15, 15, 15, 15, 14, 13, 11, 10, 9, 8, 8, 9, 9, 8, 7],
     "targets": [[71, 12], [76, 10], [77, 10]], "wind": 3.8, "shells": 5},
    # seed 2079; 5 expansions, 2-shell plan
    {"heights": [15, 16, 17, 17, 17, 17, 15, 14, 12, 10, 8, 6, 4, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 3, 2, 2, 2, 2, 2, 3, 4, 4, 5, 5, 5, 5, 5, 4, 3, 3, 2, 2, 2, 2, 2, 3, 5, 6, 8, 9, 11, 12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 10, 9, 8, 7, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2],
     "targets": [[68, 9], [69, 8], [71, 5]], "wind": 5.7, "shells": 5},
    # seed 2080; 129 expansions, 4-shell plan
    {"heights": [12, 11, 10, 8, 7, 7, 7, 7, 6, 7, 6, 6, 5, 4, 3, 3, 2, 2, 2, 2, 2, 3, 4, 6, 7, 9, 10, 11, 11, 12, 12, 11, 11, 10, 10, 9, 9, 9, 9, 9, 10, 10, 10, 11, 11, 10, 9, 9, 8, 7, 7, 7, 7, 8, 8, 9, 10, 10, 10, 10, 9, 7, 6, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3],
     "targets": [[37, 10], [38, 10], [63, 5]], "wind": 4.5, "shells": 5},
    # seed 2081; 149 expansions, 3-shell plan
    {"heights": [6, 6, 7, 7, 8, 9, 10, 11, 11, 10, 10, 8, 6, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 15, 14, 13, 12, 10, 8, 7, 7, 6, 6, 6, 7, 7, 7, 8, 8, 8, 7, 6, 6, 5, 5, 5, 5, 5, 6, 6, 6, 7, 6, 6, 5, 4, 3, 2, 2, 2, 2, 2],
     "targets": [[52, 8], [73, 5]], "wind": -1.7, "shells": 4},
    # seed 2082; 132 expansions, 3-shell plan
    {"heights": [11, 11, 12, 12, 13, 13, 13, 12, 11, 11, 11, 10, 10, 9, 10, 10, 11, 11, 12, 13, 14, 14, 14, 13, 13, 12, 10, 9, 7, 6, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 6, 7, 9, 10, 12, 13, 14, 14, 13, 13, 11, 10, 9, 8, 8, 7, 8, 9, 9, 10, 10, 10, 10, 10, 11, 10, 10, 9, 9, 9, 9, 9, 9],
     "targets": [[36, 3], [60, 9]], "wind": -5.4, "shells": 4},
    # seed 2083; 5 expansions, 2-shell plan
    {"heights": [9, 10, 10, 11, 13, 14, 16, 17, 18, 18, 18, 17, 16, 15, 13, 12, 10, 9, 8, 7, 7, 7, 7, 7, 8, 7, 6, 6, 5, 5, 4, 4, 4, 5, 6, 6, 7, 9, 10, 10, 11, 11, 12, 10, 9, 8, 7, 7, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 15, 15, 15, 16, 16, 17, 17, 18, 17, 18, 18, 17, 16, 14, 12, 10, 9, 7, 5, 4],
     "targets": [[47, 8], [49, 8]], "wind": 2.9, "shells": 4},
    # seed 2084; 228 expansions, 4-shell plan
    {"heights": [4, 5, 6, 6, 8, 9, 11, 12, 13, 15, 15, 15, 14, 14, 13, 12, 10, 10, 9, 9, 9, 10, 11, 11, 12, 13, 13, 12, 12, 11, 11, 10, 9, 8, 8, 8, 7, 7, 7, 7, 7, 6, 5, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 5, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 10, 11, 12, 12, 12, 11, 11, 10, 9, 8, 7, 7],
     "targets": [[34, 9], [36, 8]], "wind": 0.4, "shells": 4},
    # seed 2085; 10 expansions, 2-shell plan
    {"heights": [6, 7, 8, 9, 10, 11, 13, 14, 14, 15, 14, 14, 13, 13, 12, 11, 11, 11, 11, 12, 13, 13, 14, 15, 16, 17, 17, 17, 16, 16, 14, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 7, 5, 4, 3, 2, 2, 2, 2, 2, 2, 4, 5, 7, 8, 9, 10, 11, 11, 11, 11, 10, 10, 9, 9, 9, 9, 9, 10, 11, 11, 11, 12, 12, 12, 11, 10, 9, 9, 9],
     "targets": [[31, 14], [35, 12]], "wind": -5.1, "shells": 4},
    # seed 2086; 45 expansions, 3-shell plan
    {"heights": [12, 11, 10, 9, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 12, 11, 10, 9, 8, 7, 6, 6, 6, 5, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 5, 7, 8, 10, 11, 11, 11, 12, 11, 11, 11, 11, 11, 11, 11, 12, 13, 14, 14, 13, 12, 11, 9, 8, 8, 8, 8, 7, 8, 9, 10, 11, 12, 13, 13, 12, 11, 10, 8, 7],
     "targets": [[62, 10], [68, 9]], "wind": 0.3, "shells": 4},
    # seed 2087; 69 expansions, 3-shell plan
    {"heights": [8, 7, 7, 6, 6, 5, 5, 5, 5, 7, 7, 9, 11, 13, 14, 15, 16, 16, 16, 17, 17, 17, 17, 16, 16, 16, 17, 17, 17, 16, 16, 15, 14, 12, 11, 9, 8, 6, 5, 4, 4, 4, 5, 6, 7, 8, 9, 9, 9, 8, 8, 8, 7, 7, 7, 6, 6, 6, 6, 7, 8, 8, 8, 8, 7, 7, 6, 6, 6, 6, 7, 8, 10, 11, 12, 14, 16, 17, 17, 18],
     "targets": [[34, 12], [37, 7], [56, 7]], "wind": -0.7, "shells": 5},
    # seed 2088; 116 expansions, 4-shell plan
    {"heights": [11, 10, 11, 10, 10, 10, 10, 10, 9, 8, 7, 5, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 5, 6, 7, 8, 9, 9, 9, 8, 9, 8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 10, 9, 8, 7, 7, 6, 6, 6, 8, 8, 9, 10, 11, 12, 13, 12, 11, 10, 9, 7, 5, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
     "targets": [[46, 8], [47, 7], [68, 3]], "wind": 4.3, "shells": 5},
    # seed 2089; 40 expansions, 3-shell plan
    {"heights": [5, 6, 6, 5, 5, 5, 4, 4, 3, 2, 2, 2, 2, 2, 2, 4, 6, 8, 10, 11, 13, 14, 16, 16, 16, 15, 15, 15, 15, 15, 16, 15, 16, 16, 16, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 7, 8, 8, 9, 10, 11, 12, 12, 12, 11, 10, 9, 7, 6, 5, 4, 4, 4, 4, 4, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 11, 13, 14],
     "targets": [[41, 11], [46, 9]], "wind": 5.9, "shells": 4},
    # seed 2090; 13 expansions, 2-shell plan
    {"heights": [9, 9, 9, 8, 8, 8, 9, 10, 10, 11, 12, 12, 12, 12, 12, 12, 11, 12, 11, 12, 13, 13, 14, 15, 16, 17, 16, 16, 15, 13, 11, 8, 6, 4, 3, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 11, 11, 11, 10, 9, 8, 7, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 14, 14],
     "targets": [[34, 4], [35, 3]], "wind": -3.7, "shells": 4},
    # seed 2091; 274 expansions, 4-shell plan
    {"heights": [10, 9, 8, 8, 6, 5, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 14, 14, 13, 12, 11, 10, 9, 9, 8, 8, 8, 8, 8, 7, 7, 7, 7, 6, 6, 5, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 5, 6],
     "targets": [[39, 10], [53, 5]], "wind": 2.1, "shells": 4},
    # seed 2092; 13 expansions, 2-shell plan
    {"heights": [8, 7, 6, 5, 4, 3, 4, 4, 5, 6, 7, 8, 10, 11, 13, 13, 15, 14, 14, 14, 13, 13, 12, 12, 12, 12, 11, 10, 10, 9, 9, 8, 7, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 5, 6, 6, 5, 4, 4, 3, 2, 2, 2, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 5, 6, 7, 7, 8, 9, 11, 12, 13, 14, 15, 15, 15],
     "targets": [[32, 8], [57, 3]], "wind": -3.8, "shells": 4},
    # seed 2093; 176 expansions, 3-shell plan
    {"heights": [10, 11, 12, 11, 11, 10, 10, 10, 10, 10, 10, 10, 11, 12, 12, 11, 11, 10, 9, 8, 8, 7, 7, 6, 6, 7, 8, 9, 9, 10, 10, 11, 10, 9, 7, 5, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 5, 7, 9, 10, 11, 12, 12, 13, 11, 10, 9, 8, 7, 6, 6, 6, 6, 7, 8, 8, 9, 10, 9],
     "targets": [[43, 3], [69, 8]], "wind": -3.2, "shells": 4},
    # seed 2094; 4 expansions, 2-shell plan
    {"heights": [8, 7, 6, 5, 3, 3, 3, 4, 4, 5, 6, 7, 8, 8, 9, 10, 11, 11, 11, 11, 12, 12, 13, 14, 15, 16, 17, 17, 17, 17, 16, 15, 15, 13, 11, 9, 7, 6, 5, 4, 4, 4, 5, 5, 6, 6, 6, 6, 5, 5, 5, 5, 4, 5, 5, 5, 5, 6, 6, 7, 7, 6, 5, 5, 3, 3, 2, 2, 2, 2, 3, 4, 6, 7, 9, 11, 12, 13, 13, 13],
     "targets": [[34, 12], [64, 4]], "wind": -5.0, "shells": 4},
    # seed 2095; 48 expansions, 3-shell plan
    {"heights": [12, 12, 12, 12, 10, 9, 9, 8, 7, 7, 7, 7, 6, 6, 6, 5, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 5, 7, 7, 7, 8, 9, 9, 9, 8, 8, 9, 9, 10, 11, 11, 11, 11, 11, 10, 9, 9, 7, 7, 6, 6, 6, 6, 7, 8, 10, 11, 12, 12, 12, 12, 11, 10, 8, 7, 5, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2],
     "targets": [[55, 7], [56, 7], [68, 8]], "wind": 5.8, "shells": 5},
    # seed 2096; 13 expansions, 2-shell plan
    {"heights": [11, 10, 9, 9, 8, 8, 9, 10, 11, 11, 12, 12, 12, 12, 12, 11, 10, 10, 8, 7, 6, 6, 5, 4, 4, 4, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 5, 7, 9, 11, 13, 14, 15, 16, 16, 16, 15, 15, 15, 14, 14, 14, 14, 13, 13, 13, 13, 12, 12, 11, 11, 10, 9, 8, 8, 8, 8, 9, 9, 11, 11, 11, 11, 10, 10, 8, 7, 5],
     "targets": [[30, 3], [61, 12], [62, 12]], "wind": -3.2, "shells": 5},
    # seed 2097; 13 expansions, 2-shell plan
    {"heights": [11, 9, 8, 6, 5, 4, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 9, 8, 6, 5, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 6, 7, 9, 10, 12, 14, 15, 17, 17, 17, 16, 15, 14, 12, 10, 9, 9, 7, 7, 7, 7, 8, 8, 8, 8, 7, 7, 7, 7, 7, 6, 5, 5, 6, 6, 7, 7, 8, 7, 8, 8, 7],
     "targets": [[53, 10], [69, 6], [70, 6]], "wind": -1.6, "shells": 5},
    # seed 2098; 4 expansions, 2-shell plan
    {"heights": [8, 6, 4, 3, 2, 2, 2, 2, 2, 2, 3, 3, 4, 5, 5, 5, 6, 6, 6, 6, 7, 8, 10, 11, 13, 14, 15, 16, 17, 17, 16, 15, 13, 11, 10, 9, 8, 7, 7, 7, 8, 8, 8, 8, 8, 8, 7, 7, 6, 6, 5, 4, 4, 4, 4, 3, 4, 4, 4, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 6, 7, 9, 10, 11, 11, 12],
     "targets": [[35, 10], [40, 9]], "wind": -2.6, "shells": 4},
    # seed 2099; 40 expansions, 3-shell plan
    {"heights": [2, 2, 2, 2, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 7, 7, 8, 9, 10, 12, 13, 14, 16, 17, 17, 16, 15, 14, 12, 11, 9, 8, 7, 6, 5, 5, 5, 6, 6, 6, 6, 6, 5, 4, 4, 3, 3, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 5, 6, 8, 9, 10, 10, 10, 11, 11, 11, 11],
     "targets": [[28, 13], [32, 8], [33, 7]], "wind": -4.4, "shells": 5},
)
