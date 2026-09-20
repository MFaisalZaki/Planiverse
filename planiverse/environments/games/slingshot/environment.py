"""A slingshot physics puzzle in the Angry Birds mould, on pymunk.

A wall of blocks stands on the ground with a few targets on or among them; the player has a
handful of shots from a slingshot on the left, each an angle and a power from a small set,
and wins when every target is down. What a shot does is decided by a rigid-body simulation
(pymunk, the Python binding of Chipmunk2D): the bird flies under gravity, strikes the wall,
and blocks topple, slide and break according to the collision energies the engine reports.
No add or delete list carries that, which is why the environment is here.

## What an action is

`shoot(angle, power)`: a bird leaves the slingshot at `power` units per second, `angle`
degrees above the horizontal, and the world is stepped until everything is at rest again
(or ten simulated seconds pass). A target breaks when a collision dissipates more than
`TARGET_BREAKS_AT` of kinetic energy in it, a wooden block above `WOOD_BREAKS_AT`; stone
never breaks and only topples. Between shots the world is at rest, so a state is the set of
bodies still standing, where they lie, and how many shots are left.

## Determinism

pymunk is deterministic for the same sequence of operations on the same platform, and a
state is rebuilt from its record before each expansion, so expanding a state twice gives
the same children. Across platforms floating point may differ in the last places; an
instance's stored witness plan is the test of that.

## Instances and generation

`generate_instance(seed, ...)` draws towers of blocks, sets targets on and among them, and
keeps the draw only if a bounded breadth-first search finds a plan within the shots given
that needs more than one shot; the plan is left in `witness`. The bundled instances are such
draws, embedded as plain data with the seed each came from. The method is generate-and-test
(search-based procedural content generation: Togelius et al. 2011,
https://doi.org/10.1109/TCIAIG.2011.2148116), and the game is the genre's own, after the
open Science Birds clone used by the AIBirds competition (https://aibirds.org/).
"""
import math

from planiverse.environments.base import Environment
from planiverse.environments.generation import bounded_search, draw_until, rng

#: The world: `WIDTH` units wide, the ground at y = 0, a wall at the far right.
WIDTH = 100.0
#: Where a bird leaves the slingshot.
LAUNCH = (8.0, 6.0)
GRAVITY = -60.0
#: The shot alphabet: `angle` degrees above the horizontal, `power` units per second.
ANGLES = (20, 30, 40, 50, 60, 70)
POWERS = (45, 60, 75)
#: Simulation: the step, how many steps a shot may run, and the speed below which a body
#: counts as at rest.
DT, MAX_STEPS, AT_REST = 1.0 / 60.0, 600, 0.6
BIRD_RADIUS, BIRD_MASS = 1.0, 3.0
TARGET_RADIUS, TARGET_MASS = 0.9, 1.0
#: Kinetic energy a collision must dissipate to break a target or a wooden block.
TARGET_BREAKS_AT, WOOD_BREAKS_AT = 60.0, 400.0
DENSITY = {"wood": 0.4, "stone": 1.0}
#: Collision types, so the handler knows what it is looking at.
STATIC, BLOCK, TARGET, BIRD = 0, 1, 2, 3
#: The grid the literals name positions on.
CELL = 2.0


def _pymunk():
    import pymunk
    return pymunk


class SlingshotAction:
    """`shoot(angle, power)`."""

    def __init__(self, angle, power):
        if angle not in ANGLES or power not in POWERS:
            raise ValueError(f"unknown shot: angle {angle}, power {power}")
        self.angle, self.power = angle, power
        self.name = f"shoot({angle},{power})"

    @classmethod
    def parse(cls, text):
        inside = str(text).strip()[len("shoot("):-1]
        angle, power = (int(part) for part in inside.split(","))
        return cls(angle, power)

    def cost(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, SlingshotAction) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    __repr__ = __str__


ACTIONS = tuple(SlingshotAction(angle, power) for angle in ANGLES for power in POWERS)


class SlingshotState:
    """The bodies still standing, at rest, and the shots left.

    `bodies` is a tuple of records, `("block", material, w, h, x, y, angle)` or
    `("target", x, y)`, positions rounded so that a state is the same state wherever it was
    reached from. `depth` is bookkeeping and stays out of equality.
    """

    def __init__(self, bodies, shots_left, depth=0):
        self.bodies = tuple(bodies)
        self.shots_left = shots_left
        self.depth = depth
        self.targets_left = sum(1 for body in self.bodies if body[0] == "target")
        literals = []
        for index, body in enumerate(self.bodies):
            x, y = (body[4], body[5]) if body[0] == "block" else (body[1], body[2])
            literals.append(f"at({body[0]}-{index}, {int(x // CELL)}, {int(y // CELL)})")
        literals.append(f"shots_left({shots_left})")
        literals.append(f"targets_left({self.targets_left})")
        self.literals = frozenset(literals)

    def __eq__(self, other):
        return (isinstance(other, SlingshotState) and self.bodies == other.bodies
                and self.shots_left == other.shots_left)

    def __hash__(self):
        return hash((self.bodies, self.shots_left))

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        lines = [f"shots left: {self.shots_left}, targets standing: {self.targets_left}"]
        for body in self.bodies:
            if body[0] == "block":
                lines.append(f"  {body[1]} {body[2]:g}x{body[3]:g} at ({body[4]:.1f}, {body[5]:.1f}) "
                             f"turned {math.degrees(body[6]):.0f}°")
            else:
                lines.append(f"  target at ({body[1]:.1f}, {body[2]:.1f})")
        return "\n".join(lines)

    def __repr__(self):
        return f"<SlingshotState(bodies={len(self.bodies)}, targets={self.targets_left}, shots={self.shots_left})>"


# ----------------------------------------------------------------------------- the physics

def build_space(bodies):
    """A pymunk space holding `bodies` at rest, with the ground and the far wall."""
    pymunk = _pymunk()
    space = pymunk.Space()
    space.gravity = (0.0, GRAVITY)
    ground = pymunk.Segment(space.static_body, (-10.0, 0.0), (WIDTH + 10.0, 0.0), 0.5)
    wall = pymunk.Segment(space.static_body, (WIDTH, 0.0), (WIDTH, 200.0), 0.5)
    for shape in (ground, wall):
        shape.friction, shape.elasticity, shape.collision_type = 0.8, 0.1, STATIC
    space.add(ground, wall)
    for record in bodies:
        if record[0] == "block":
            _, material, w, h, x, y, angle = record
            mass = DENSITY[material] * w * h
            body = pymunk.Body(mass, pymunk.moment_for_box(mass, (w, h)))
            body.position, body.angle = (x, y), angle
            shape = pymunk.Poly.create_box(body, (w, h))
            shape.collision_type = BLOCK
            shape.breaks_at = WOOD_BREAKS_AT if material == "wood" else None
            shape.record = (material, w, h)
        else:
            _, x, y = record
            body = pymunk.Body(TARGET_MASS, pymunk.moment_for_circle(TARGET_MASS, 0, TARGET_RADIUS))
            body.position = (x, y)
            shape = pymunk.Circle(body, TARGET_RADIUS)
            shape.collision_type = TARGET
            shape.breaks_at = TARGET_BREAKS_AT
            shape.record = None
        shape.friction, shape.elasticity = 0.7, 0.1
        space.add(body, shape)
    return space


def records_of(space):
    """The bodies in `space` as state records, rounded, birds left out."""
    pymunk = _pymunk()
    records = []
    for shape in space.shapes:
        if shape.collision_type == BLOCK:
            material, w, h = shape.record
            x, y = shape.body.position
            records.append(("block", material, w, h, round(x, 2), round(y, 2),
                            round(shape.body.angle, 3)))
        elif shape.collision_type == TARGET:
            x, y = shape.body.position
            records.append(("target", round(x, 2), round(y, 2)))
    records.sort(key=lambda record: (record[0], record[1:]))
    return tuple(records)


def shoot(bodies, action):
    """Fire one shot at the world `bodies` describes and return the records at rest after."""
    pymunk = _pymunk()
    space = build_space(bodies)
    bird = pymunk.Body(BIRD_MASS, pymunk.moment_for_circle(BIRD_MASS, 0, BIRD_RADIUS))
    bird.position = LAUNCH
    radians = math.radians(action.angle)
    bird.velocity = (action.power * math.cos(radians), action.power * math.sin(radians))
    bird_shape = pymunk.Circle(bird, BIRD_RADIUS)
    bird_shape.friction, bird_shape.elasticity, bird_shape.collision_type = 0.7, 0.2, BIRD
    bird_shape.breaks_at = None
    space.add(bird, bird_shape)

    broken = set()

    def post_solve(arbiter, _space, _data):
        energy = arbiter.total_ke
        for shape in arbiter.shapes:
            breaks_at = getattr(shape, "breaks_at", None)
            if breaks_at is not None and energy >= breaks_at:
                broken.add(shape)

    space.on_collision(post_solve=post_solve)
    for step in range(MAX_STEPS):
        space.step(DT)
        if broken:
            for shape in broken:
                if shape in space.shapes:
                    space.remove(shape, shape.body)
            broken.clear()
        if step > 30 and all(body.velocity.length < AT_REST and abs(body.angular_velocity) < 0.3
                             for body in space.bodies):
            break
    return records_of(space)


# ------------------------------------------------------------------------------ the levels

def draw_level(random_, structures=3, targets=2, shots=3, min_height=2, max_height=4):
    """A random wall as the instance dict `set_instance` takes: `structures` stand on the
    ground between the slingshot and the far wall, each a tower of blocks, a shelter (a roof
    on two pillars) or a stone wall in front of the ground, and `targets` are set on top of,
    under or behind them, so that some need a lob over, some a roof broken and some a tower
    toppled."""
    bodies, seats = [], []
    spacing = 45.0 / (structures - 1) if structures > 1 else 0.0
    for k in range(structures):
        x = 45.0 + k * spacing + random_.uniform(-3.0, 3.0)
        kind = random_.choice(("tower", "tower", "shelter", "wall"))
        if kind == "tower":
            y = 0.5
            for _ in range(random_.randint(min_height, max_height)):
                material = random_.choice(("wood", "wood", "stone"))
                w, h = random_.choice(((2.0, 6.0), (4.0, 3.0), (6.0, 2.0)))
                bodies.append(("block", material, w, h, round(x, 2), round(y + h / 2, 2), 0.0))
                y += h
            seats.append((x, y))                                   # on top
        elif kind == "shelter":
            material = random_.choice(("wood", "stone"))
            for dx in (-3.0, 3.0):
                bodies.append(("block", material, 2.0, 6.0, round(x + dx, 2), 3.5, 0.0))
            bodies.append(("block", "wood", 8.0, 2.0, round(x, 2), 7.5, 0.0))     # the roof
            seats.append((x, 0.5))                                 # under it
        else:
            bodies.append(("block", "stone", 2.0, 8.0, round(x, 2), 4.5, 0.0))
            seats.append((x + 4.0, 0.5))                           # behind it
    random_.shuffle(seats)
    for k in range(targets):
        x, y = seats[k % len(seats)]
        if k >= len(seats):
            x += random_.choice((-4.0, 4.0))
            y = 0.5
        bodies.append(("target", round(x, 2), round(y + TARGET_RADIUS, 2)))
    return {"bodies": bodies, "shots": shots}


def settle(instance):
    """Let a drawn level come to rest before it is played, so its opening is a fixed point."""
    space = build_space(instance["bodies"])
    for _ in range(240):
        space.step(DT)
    return {"bodies": [list(record) for record in records_of(space)], "shots": instance["shots"]}


class SlingshotEnv(Environment):
    """Knock every target down within the shots given."""

    def __init__(self):
        super().__init__("slingshot")
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
        """Select a level: `{"bodies": [...records...], "shots": n}`."""
        if "bodies" not in instance or "shots" not in instance:
            raise ValueError("a level is a dict with `bodies` and `shots`")
        self.instance = {"bodies": [tuple(record) for record in instance["bodies"]],
                         "shots": int(instance["shots"])}
        self.index = None
        self.witness = self.witness_expansions = None

    def generate_instance(self, seed=None, structures=None, targets=None, shots=None,
                          min_plan_length=3, search_limit=600, attempts=60):
        """Draw a level, select it, and return it as the dict `set_instance` takes.

        `structures`, `targets` and `shots` are `draw_level`'s; each left unset is drawn from
        the bundled levels' range (two to four structures, three or four targets, and as many
        shots as targets, so no shot may be wasted). A draw is kept only if a breadth-first
        search over shots finds a plan within `search_limit` expansions and that plan, a
        shortest one, has at least `min_plan_length` shots: the search has then shown that no
        shorter plan exists, so a level two shots flatten is thrown back. The plan is left in
        `witness` and what the search spent in `witness_expansions`.
        """
        random_, _ = rng(seed)
        found = {}

        def draw(attempt):
            options = dict(structures=structures or random_.randint(2, 4),
                           targets=targets or random_.randint(3, 4))
            options["shots"] = shots or options["targets"]
            return settle(draw_level(random_, **options))

        def accept(instance):
            self.set_instance(instance)
            outcome = bounded_search(self, search_limit)
            if outcome.plan is None or len(outcome.plan) < min_plan_length:
                return False
            found["plan"], found["expansions"] = outcome.plan, outcome.expansions
            return True

        instance = draw_until(draw, accept, attempts, "slingshot level")
        self.set_instance(instance)
        self.witness, self.witness_expansions = found["plan"], found["expansions"]
        return instance

    # ------------------------------------------------------------------- contract

    def reset(self):
        if self.instance is None:
            self.set_index(0)
        self.state = SlingshotState(self.instance["bodies"], self.instance["shots"])
        self.state_history = [self.state]
        return self.state, {"level": self.index, "targets": self.state.targets_left,
                            "shots": self.instance["shots"], "generated": self.index is None}

    def is_goal(self, state):
        return state.targets_left == 0

    def is_terminal(self, state):
        return state.targets_left > 0 and state.shots_left == 0

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
        if not isinstance(action, SlingshotAction):
            action = SlingshotAction.parse(action)
        bodies = shoot(state.bodies, action)
        return SlingshotState(bodies, state.shots_left - 1, state.depth + 1)

    def simulate(self, plan):
        state, _ = self.reset()
        trace = [state]
        for action in plan:
            trace.append(self.__advance__(trace[-1], action))
        return trace

    def step(self, action):
        if self.state is None:
            raise ValueError("call reset() first")
        before = self.state.targets_left
        self.state = self.__advance__(self.state, action)
        self.state_history.append(self.state)
        return self.state, before - self.state.targets_left

    def get_actions(self):
        return list(ACTIONS)

    def render(self):
        lines = [f"step {k}:\n{state}" for k, state in enumerate(self.state_history)]
        print("\n".join(lines))
        return lines


#: The bundled levels: `generate_instance(seed, targets=n)` for the seed and target count
#: beside each, embedded as the plain data `set_instance` takes, with the plan each was
#: accepted on in `tests/data/slingshot_solutions.json`.
LEVELS = (
    # seed 9000, targets=3; 24 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 61.74, 4.5, -0.0), ('block', 'stone', 6.0, 2.0, 77.72, 1.48, 0.005), ('block', 'wood', 2.0, 6.0, 40.58, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 46.58, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 77.64, 13.45, 0.008), ('block', 'wood', 2.0, 6.0, 77.68, 5.48, 0.008), ('block', 'wood', 2.0, 6.0, 88.82, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 94.82, 3.5, -0.001), ('block', 'wood', 6.0, 2.0, 77.66, 9.46, -0.001), ('block', 'wood', 8.0, 2.0, 43.58, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 91.82, 7.49, -0.002), ('target', 43.58, 1.38), ('target', 71.39, 2.33), ('target', 91.82, 1.38)],
     "shots": 3},
    # seed 9001, targets=4; 10 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 87.69, 3.5, 0.0), ('block', 'stone', 2.0, 6.0, 93.59, 3.49, -0.007), ('block', 'stone', 4.0, 3.0, 43.76, 7.94, 0.005), ('block', 'stone', 6.0, 2.0, 43.8, 3.46, 0.002), ('block', 'wood', 6.0, 2.0, 43.78, 5.45, 0.004), ('block', 'wood', 6.0, 2.0, 43.81, 1.49, 0.001), ('block', 'wood', 8.0, 2.0, 90.67, 7.5, -0.0), ('target', 39.81, 1.38), ('target', 42.14, 10.32), ('target', 90.69, 1.38), ('target', 95.44, 1.38)],
     "shots": 4},
    # seed 9002, targets=3; 15 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 54.21, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 60.21, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 86.36, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 92.36, 3.5, -0.001), ('block', 'stone', 4.0, 3.0, 74.63, 4.97, -0.005), ('block', 'wood', 2.0, 6.0, 42.14, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 48.14, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 74.62, 1.99, 0.0), ('block', 'wood', 4.0, 3.0, 74.66, 7.95, -0.01), ('block', 'wood', 4.0, 3.0, 74.67, 10.94, -0.011), ('block', 'wood', 8.0, 2.0, 45.14, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 57.21, 7.5, -0.0), ('block', 'wood', 8.0, 2.0, 89.36, 7.49, -0.002), ('target', 45.14, 1.38), ('target', 57.21, 1.38), ('target', 78.14, 10.83)],
     "shots": 3},
    # seed 9003, targets=4; 33 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 43.11, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 49.11, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 69.5, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 75.5, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 87.59, 4.5, -0.0), ('block', 'stone', 6.0, 2.0, 59.86, 9.46, 0.011), ('block', 'wood', 2.0, 6.0, 59.9, 3.48, 0.002), ('block', 'wood', 6.0, 2.0, 59.89, 7.47, 0.01), ('block', 'wood', 8.0, 2.0, 46.11, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 72.5, 7.49, -0.002), ('target', 46.11, 1.38), ('target', 56.64, 11.3), ('target', 72.5, 1.38), ('target', 91.59, 1.38)],
     "shots": 4},
    # seed 9004, targets=3; 19 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 70.57, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 76.57, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 45.49, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 60.64, 3.96, 0.008), ('block', 'wood', 2.0, 6.0, 89.88, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 95.88, 3.5, -0.001), ('block', 'wood', 6.0, 2.0, 60.59, 8.45, 0.011), ('block', 'wood', 6.0, 2.0, 60.62, 6.45, 0.011), ('block', 'wood', 6.0, 2.0, 60.65, 1.49, 0.003), ('block', 'wood', 8.0, 2.0, 73.57, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 92.88, 7.49, -0.002), ('target', 49.49, 1.38), ('target', 73.57, 1.38), ('target', 92.88, 1.38)],
     "shots": 3},
    # seed 9005, targets=4; 45 expansions, 4-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 84.24, 3.49, 0.009), ('block', 'stone', 2.0, 6.0, 90.14, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 43.69, 4.5, -0.0), ('block', 'wood', 8.0, 2.0, 87.15, 7.49, 0.002), ('target', 47.69, 1.38), ('target', 51.69, 1.38), ('target', 82.37, 1.38), ('target', 87.14, 1.38)],
     "shots": 4},
    # seed 9006, targets=3; 15 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 46.09, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 65.14, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 71.14, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 87.66, 4.5, -0.0), ('block', 'wood', 4.0, 3.0, 46.09, 9.99, 0.004), ('block', 'wood', 6.0, 2.0, 46.09, 7.49, 0.0), ('block', 'wood', 8.0, 2.0, 68.14, 7.49, -0.002), ('target', 44.9, 12.36), ('target', 68.14, 1.38), ('target', 91.66, 1.38)],
     "shots": 3},
    # seed 9007, targets=4; 17 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 46.98, 3.5, -0.001), ('block', 'stone', 4.0, 3.0, 46.99, 7.99, -0.001), ('block', 'stone', 4.0, 3.0, 91.12, 1.99, 0.004), ('block', 'wood', 2.0, 6.0, 61.57, 3.5, -0.0), ('block', 'wood', 2.0, 6.0, 67.57, 3.48, 0.0), ('block', 'wood', 2.0, 6.0, 91.1, 6.49, 0.005), ('block', 'wood', 4.0, 3.0, 91.06, 12.98, 0.007), ('block', 'wood', 6.0, 2.0, 91.08, 10.48, 0.007), ('block', 'wood', 8.0, 2.0, 64.57, 7.49, -0.002), ('target', 47.29, 10.37), ('target', 64.57, 1.38), ('target', 88.69, 15.28), ('target', 95.13, 1.38)],
     "shots": 4},
    # seed 9008, targets=3; 15 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 85.74, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 91.74, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 43.39, 4.5, -0.0), ('block', 'wood', 8.0, 2.0, 88.74, 7.49, -0.002), ('target', 47.39, 1.38), ('target', 51.39, 1.38), ('target', 88.74, 1.38)],
     "shots": 3},
    # seed 9009, targets=4; 19 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 4.0, 3.0, 42.2, 9.97, 0.002), ('block', 'stone', 4.0, 3.0, 42.21, 4.98, 0.002), ('block', 'stone', 4.0, 3.0, 42.22, 1.99, 0.002), ('block', 'stone', 6.0, 2.0, 42.21, 7.47, 0.002), ('block', 'stone', 6.0, 2.0, 89.67, 3.48, 0.004), ('block', 'wood', 6.0, 2.0, 89.68, 1.49, 0.003), ('target', 41.47, 12.36), ('target', 46.22, 1.38), ('target', 88.34, 5.36), ('target', 93.68, 1.38)],
     "shots": 4},
    # seed 9010, targets=3; 17 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 87.28, 3.5, 0.0), ('block', 'stone', 2.0, 6.0, 93.18, 3.49, -0.007), ('block', 'stone', 2.0, 8.0, 45.54, 4.5, -0.0), ('block', 'wood', 8.0, 2.0, 90.26, 7.5, -0.0), ('target', 49.54, 1.38), ('target', 90.28, 1.38), ('target', 95.03, 1.38)],
     "shots": 3},
    # seed 9011, targets=4; 24 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 39.55, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 45.55, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 69.76, 15.47, 0.009), ('block', 'stone', 2.0, 6.0, 69.84, 3.48, 0.001), ('block', 'stone', 2.0, 6.0, 87.68, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 93.68, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 69.81, 9.47, 0.007), ('block', 'wood', 6.0, 2.0, 69.73, 19.47, 0.008), ('block', 'wood', 8.0, 2.0, 42.55, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 90.68, 7.49, -0.002), ('target', 42.55, 1.38), ('target', 65.84, 1.38), ('target', 67.49, 21.35), ('target', 90.68, 1.38)],
     "shots": 4},
    # seed 9012, targets=3; 19 expansions, 3-shot plan
    {"bodies": [('block', 'wood', 2.0, 6.0, 39.56, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 45.56, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 64.27, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 70.27, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 86.35, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 92.35, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 42.56, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 67.27, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 89.35, 7.49, -0.002), ('target', 42.56, 1.38), ('target', 67.27, 1.38), ('target', 89.35, 1.38)],
     "shots": 3},
    # seed 9013, targets=4; 13 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 59.12, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 44.27, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 50.27, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 85.51, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 91.51, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 77.29, 1.99, 0.001), ('block', 'wood', 6.0, 2.0, 77.28, 4.49, -0.0), ('block', 'wood', 8.0, 2.0, 47.27, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 88.51, 7.49, -0.002), ('target', 47.27, 1.38), ('target', 63.12, 1.38), ('target', 77.39, 6.37), ('target', 88.51, 1.38)],
     "shots": 4},
    # seed 9014, targets=3; 44 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 75.66, 11.48, 0.007), ('block', 'stone', 2.0, 8.0, 89.57, 4.5, -0.0), ('block', 'stone', 6.0, 2.0, 60.68, 1.48, -0.0), ('block', 'stone', 6.0, 2.0, 75.73, 1.48, 0.005), ('block', 'wood', 2.0, 6.0, 39.41, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 45.41, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 75.71, 5.48, 0.007), ('block', 'wood', 4.0, 3.0, 60.68, 3.96, 0.005), ('block', 'wood', 8.0, 2.0, 42.41, 7.49, -0.002), ('target', 42.41, 1.38), ('target', 71.07, 1.11), ('target', 93.57, 1.38)],
     "shots": 3},
    # seed 9015, targets=4; 20 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 58.68, 6.49, 0.004), ('block', 'stone', 2.0, 6.0, 75.72, 3.48, 0.001), ('block', 'stone', 4.0, 3.0, 58.7, 1.99, 0.004), ('block', 'stone', 6.0, 2.0, 75.68, 11.46, 0.008), ('block', 'stone', 6.0, 2.0, 75.7, 9.47, 0.007), ('block', 'stone', 6.0, 2.0, 75.71, 7.48, 0.005), ('block', 'wood', 2.0, 6.0, 43.85, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 49.85, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 58.66, 12.49, 0.005), ('block', 'wood', 2.0, 6.0, 88.4, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 94.4, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 58.63, 16.98, 0.005), ('block', 'wood', 8.0, 2.0, 46.85, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 91.4, 7.49, -0.002), ('target', 46.85, 1.38), ('target', 57.31, 19.38), ('target', 73.25, 13.33), ('target', 91.4, 1.38)],
     "shots": 4},
    # seed 9016, targets=3; 15 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 87.94, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 93.94, 3.5, -0.001), ('block', 'stone', 6.0, 2.0, 42.1, 9.46, 0.005), ('block', 'wood', 2.0, 6.0, 42.13, 3.48, 0.002), ('block', 'wood', 4.0, 3.0, 42.09, 11.96, 0.007), ('block', 'wood', 6.0, 2.0, 42.11, 7.47, 0.003), ('block', 'wood', 8.0, 2.0, 90.94, 7.49, -0.002), ('target', 38.14, 1.38), ('target', 40.27, 14.33), ('target', 90.94, 1.38)],
     "shots": 3},
    # seed 9017, targets=4; 22 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 40.38, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 46.38, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 65.97, 11.47, 0.003), ('block', 'wood', 2.0, 6.0, 65.95, 17.47, 0.003), ('block', 'wood', 2.0, 6.0, 65.99, 3.48, 0.001), ('block', 'wood', 2.0, 6.0, 89.8, 17.46, 0.009), ('block', 'wood', 2.0, 6.0, 89.86, 11.47, 0.01), ('block', 'wood', 2.0, 6.0, 89.92, 5.47, 0.009), ('block', 'wood', 6.0, 2.0, 65.98, 7.47, 0.003), ('block', 'wood', 6.0, 2.0, 89.95, 1.48, 0.006), ('block', 'wood', 8.0, 2.0, 43.38, 7.49, -0.002), ('target', 43.38, 1.38), ('target', 62.0, 1.38), ('target', 63.88, 20.29), ('target', 84.61, 1.4)],
     "shots": 4},
    # seed 9018, targets=3; 12 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 88.58, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 94.58, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 47.12, 4.5, -0.0), ('block', 'stone', 6.0, 2.0, 68.94, 1.48, -0.0), ('block', 'wood', 4.0, 3.0, 68.94, 3.98, 0.0), ('block', 'wood', 8.0, 2.0, 91.58, 7.49, -0.002), ('target', 51.12, 1.38), ('target', 68.79, 6.36), ('target', 91.58, 1.38)],
     "shots": 3},
    # seed 9019, targets=4; 21 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 47.32, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 85.01, 3.5, -0.0), ('block', 'wood', 2.0, 6.0, 90.8, 3.49, -0.008), ('block', 'wood', 4.0, 3.0, 67.14, 1.99, 0.001), ('block', 'wood', 6.0, 2.0, 67.13, 4.49, -0.0), ('block', 'wood', 8.0, 2.0, 87.97, 7.5, -0.001), ('target', 51.32, 1.38), ('target', 67.24, 6.37), ('target', 88.01, 1.38), ('target', 92.59, 1.38)],
     "shots": 4},
    # seed 9020, targets=3; 21 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 84.27, 3.49, 0.009), ('block', 'stone', 2.0, 6.0, 90.17, 3.5, -0.001), ('block', 'stone', 4.0, 3.0, 42.65, 6.97, -0.0), ('block', 'wood', 4.0, 3.0, 42.66, 1.99, 0.001), ('block', 'wood', 6.0, 2.0, 42.65, 4.48, -0.001), ('block', 'wood', 8.0, 2.0, 87.18, 7.49, 0.002), ('target', 42.73, 9.36), ('target', 82.4, 1.38), ('target', 87.17, 1.38)],
     "shots": 3},
    # seed 9021, targets=4; 48 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 69.86, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 87.95, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 93.95, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 40.84, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 46.84, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 69.87, 7.99, -0.002), ('block', 'wood', 8.0, 2.0, 43.84, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 90.95, 7.49, -0.002), ('target', 43.84, 1.38), ('target', 70.49, 10.37), ('target', 73.86, 1.38), ('target', 90.95, 1.38)],
     "shots": 4},
    # seed 9022, targets=3; 20 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 41.49, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 47.49, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 88.05, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 94.05, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 74.05, 4.5, -0.0), ('block', 'stone', 6.0, 2.0, 61.08, 1.49, 0.004), ('block', 'wood', 6.0, 2.0, 61.08, 3.49, 0.004), ('block', 'wood', 8.0, 2.0, 44.49, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 91.05, 7.49, -0.002), ('target', 44.49, 1.38), ('target', 78.05, 1.38), ('target', 91.05, 1.38)],
     "shots": 3},
    # seed 9023, targets=4; 23 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 85.86, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 91.86, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 42.47, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 61.32, 11.48, 0.01), ('block', 'wood', 2.0, 6.0, 61.35, 3.5, -0.0), ('block', 'wood', 2.0, 6.0, 74.14, 3.5, 0.0), ('block', 'wood', 6.0, 2.0, 61.35, 7.49, 0.005), ('block', 'wood', 6.0, 2.0, 74.14, 7.49, -0.0), ('block', 'wood', 8.0, 2.0, 88.86, 7.49, -0.002), ('target', 46.47, 1.38), ('target', 52.56, 1.36), ('target', 74.23, 9.37), ('target', 88.86, 1.38)],
     "shots": 4},
    # seed 9024, targets=3; 11 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 75.59, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 59.19, 4.98, -0.001), ('block', 'wood', 2.0, 6.0, 47.99, 6.48, -0.0), ('block', 'wood', 2.0, 6.0, 84.11, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 90.11, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 47.99, 1.98, -0.0), ('block', 'wood', 4.0, 3.0, 47.99, 10.98, -0.0), ('block', 'wood', 4.0, 3.0, 59.21, 1.99, 0.002), ('block', 'wood', 8.0, 2.0, 87.11, 7.49, -0.002), ('target', 48.19, 13.36), ('target', 59.46, 7.37), ('target', 87.11, 1.38)],
     "shots": 3},
    # seed 9025, targets=4; 15 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 76.98, 3.5, 0.0), ('block', 'stone', 2.0, 6.0, 86.51, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 92.51, 3.5, -0.001), ('block', 'stone', 4.0, 3.0, 59.17, 1.98, -0.0), ('block', 'stone', 6.0, 2.0, 76.97, 7.48, 0.002), ('block', 'wood', 2.0, 6.0, 43.54, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 49.54, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 59.18, 6.48, -0.001), ('block', 'wood', 8.0, 2.0, 46.54, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 89.51, 7.49, -0.002), ('target', 46.54, 1.38), ('target', 59.47, 10.37), ('target', 76.48, 9.36), ('target', 89.51, 1.38)],
     "shots": 4},
    # seed 9026, targets=3; 21 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 64.14, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 70.14, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 87.77, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 42.48, 4.98, -0.001), ('block', 'stone', 4.0, 3.0, 42.49, 1.99, 0.003), ('block', 'wood', 2.0, 6.0, 42.49, 9.48, -0.002), ('block', 'wood', 4.0, 3.0, 42.5, 13.97, -0.003), ('block', 'wood', 8.0, 2.0, 67.14, 7.49, -0.002), ('target', 43.08, 16.37), ('target', 67.14, 1.38), ('target', 91.77, 1.38)],
     "shots": 3},
    # seed 9027, targets=4; 19 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 73.82, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 79.82, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 86.84, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 92.84, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 43.09, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 57.14, 4.5, -0.0), ('block', 'wood', 8.0, 2.0, 76.82, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 89.84, 7.49, -0.002), ('target', 47.09, 1.38), ('target', 61.14, 1.38), ('target', 76.82, 1.38), ('target', 89.84, 1.38)],
     "shots": 4},
    # seed 9028, targets=3; 12 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 88.22, 3.5, 0.0), ('block', 'stone', 2.0, 6.0, 94.12, 3.49, -0.007), ('block', 'stone', 2.0, 8.0, 47.13, 4.5, -0.0), ('block', 'wood', 8.0, 2.0, 91.2, 7.5, -0.0), ('target', 51.13, 1.38), ('target', 91.22, 1.38), ('target', 95.97, 1.38)],
     "shots": 3},
    # seed 9029, targets=4; 13 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 92.96, 5.47, 0.008), ('block', 'stone', 6.0, 2.0, 47.79, 1.48, -0.0), ('block', 'wood', 2.0, 6.0, 47.79, 5.48, 0.001), ('block', 'wood', 2.0, 6.0, 92.91, 11.47, 0.009), ('block', 'wood', 6.0, 2.0, 93.0, 1.48, 0.006), ('target', 43.79, 1.38), ('target', 47.6, 9.36), ('target', 87.49, 1.27), ('target', 97.0, 1.38)],
     "shots": 4},
    # seed 9030, targets=3; 40 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 44.48, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 50.48, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 72.23, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 78.23, 3.5, -0.001), ('block', 'stone', 6.0, 2.0, 88.8, 4.48, -0.004), ('block', 'wood', 2.0, 6.0, 57.09, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 63.09, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 88.8, 1.99, -0.002), ('block', 'wood', 8.0, 2.0, 47.48, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 60.09, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 75.23, 7.49, -0.002), ('target', 47.48, 1.38), ('target', 60.09, 1.38), ('target', 75.23, 1.38)],
     "shots": 3},
    # seed 9031, targets=4; 27 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 44.64, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 50.64, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 58.34, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 72.42, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 78.42, 3.5, -0.001), ('block', 'wood', 6.0, 2.0, 87.07, 1.49, 0.004), ('block', 'wood', 6.0, 2.0, 87.07, 3.49, 0.004), ('block', 'wood', 8.0, 2.0, 47.64, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 75.42, 7.49, -0.002), ('target', 47.64, 1.38), ('target', 62.34, 1.38), ('target', 75.42, 1.38), ('target', 85.8, 5.37)],
     "shots": 4},
    # seed 9032, targets=3; 11 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 86.56, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 92.56, 3.5, -0.001), ('block', 'stone', 4.0, 3.0, 44.67, 1.99, 0.0), ('block', 'wood', 4.0, 3.0, 44.67, 4.98, 0.001), ('block', 'wood', 8.0, 2.0, 89.56, 7.49, -0.002), ('target', 44.24, 7.36), ('target', 48.67, 1.38), ('target', 89.56, 1.38)],
     "shots": 3},
    # seed 9033, targets=4; 15 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 42.27, 3.5, 0.0), ('block', 'stone', 2.0, 6.0, 48.17, 3.49, -0.007), ('block', 'stone', 2.0, 8.0, 88.76, 1.43, 1.578), ('block', 'wood', 8.0, 2.0, 45.25, 7.5, -0.0), ('target', 45.27, 1.38), ('target', 50.02, 1.38), ('target', 95.65, 1.3), ('target', 97.43, 1.38)],
     "shots": 4},
    # seed 9034, targets=3; 15 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 88.75, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 94.75, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 46.18, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 65.15, 3.48, 0.002), ('block', 'wood', 4.0, 3.0, 65.12, 9.97, 0.007), ('block', 'wood', 6.0, 2.0, 65.14, 7.47, 0.006), ('block', 'wood', 8.0, 2.0, 91.75, 7.49, -0.002), ('target', 50.18, 1.38), ('target', 63.18, 12.34), ('target', 91.75, 1.38)],
     "shots": 3},
    # seed 9035, targets=4; 19 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 54.38, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 60.38, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 75.67, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 89.45, 11.46, 0.01), ('block', 'stone', 2.0, 6.0, 89.51, 5.47, 0.01), ('block', 'stone', 2.0, 8.0, 42.4, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 75.64, 12.47, 0.006), ('block', 'wood', 4.0, 3.0, 75.66, 7.98, 0.001), ('block', 'wood', 6.0, 2.0, 89.54, 1.47, 0.006), ('block', 'wood', 8.0, 2.0, 57.38, 7.49, -0.002), ('target', 46.4, 1.38), ('target', 57.38, 1.38), ('target', 71.98, 4.19), ('target', 83.29, 1.29)],
     "shots": 4},
    # seed 9036, targets=3; 29 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 59.37, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 65.37, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 45.34, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 76.01, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 87.1, 4.5, -0.0), ('block', 'wood', 8.0, 2.0, 62.37, 7.49, -0.002), ('target', 49.34, 1.38), ('target', 62.37, 1.38), ('target', 91.1, 1.38)],
     "shots": 3},
    # seed 9037, targets=4; 140 expansions, 4-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 44.14, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 44.14, 9.49, 0.003), ('block', 'stone', 2.0, 6.0, 85.2, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 91.2, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 44.12, 15.49, 0.002), ('block', 'wood', 2.0, 6.0, 64.62, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 70.62, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 67.62, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 88.2, 7.49, -0.002), ('target', 40.14, 1.38), ('target', 43.36, 19.39), ('target', 67.62, 1.38), ('target', 88.2, 1.38)],
     "shots": 4},
    # seed 9038, targets=3; 28 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 40.29, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 46.29, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 92.63, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 63.11, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 69.11, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 43.29, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 66.11, 7.49, -0.002), ('target', 43.29, 1.38), ('target', 66.11, 1.38), ('target', 96.63, 1.38)],
     "shots": 3},
    # seed 9039, targets=4; 103 expansions, 4-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 42.1, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 48.1, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 54.96, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 60.96, 3.5, -0.0), ('block', 'stone', 2.0, 8.0, 90.91, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 73.82, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 79.82, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 45.1, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 57.96, 7.5, -0.0), ('block', 'wood', 8.0, 2.0, 76.82, 7.49, -0.002), ('target', 45.1, 1.38), ('target', 57.96, 1.38), ('target', 76.82, 1.38), ('target', 94.91, 1.38)],
     "shots": 4},
    # seed 9040, targets=3; 20 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 70.66, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 76.66, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 42.19, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 62.36, 3.96, 0.008), ('block', 'stone', 6.0, 2.0, 62.37, 1.49, 0.004), ('block', 'wood', 2.0, 6.0, 84.19, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 90.19, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 73.66, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 87.19, 7.49, -0.002), ('target', 46.19, 1.38), ('target', 73.66, 1.38), ('target', 87.19, 1.38)],
     "shots": 3},
    # seed 9041, targets=4; 11 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 85.53, 3.49, 0.009), ('block', 'stone', 2.0, 6.0, 91.43, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 43.69, 9.49, 0.004), ('block', 'wood', 2.0, 6.0, 43.7, 3.5, -0.001), ('block', 'wood', 6.0, 2.0, 43.67, 13.48, 0.011), ('block', 'wood', 8.0, 2.0, 88.44, 7.49, 0.002), ('target', 39.65, 14.4), ('target', 39.7, 1.38), ('target', 83.66, 1.38), ('target', 88.43, 1.38)],
     "shots": 4},
    # seed 9042, targets=3; 13 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 85.37, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 91.37, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 43.75, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 64.61, 1.99, 0.004), ('block', 'wood', 2.0, 6.0, 64.56, 14.48, 0.004), ('block', 'wood', 2.0, 6.0, 64.59, 6.49, 0.004), ('block', 'wood', 6.0, 2.0, 64.57, 10.48, 0.004), ('block', 'wood', 8.0, 2.0, 88.37, 7.49, -0.002), ('target', 47.75, 1.38), ('target', 63.06, 18.22), ('target', 88.37, 1.38)],
     "shots": 3},
    # seed 9043, targets=4; 24 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 44.02, 15.47, 0.009), ('block', 'stone', 2.0, 6.0, 44.07, 9.48, 0.006), ('block', 'stone', 2.0, 6.0, 44.09, 3.48, 0.001), ('block', 'stone', 2.0, 6.0, 85.57, 3.5, 0.0), ('block', 'stone', 2.0, 6.0, 91.47, 3.49, -0.007), ('block', 'wood', 6.0, 2.0, 43.98, 19.47, 0.009), ('block', 'wood', 8.0, 2.0, 88.55, 7.5, -0.0), ('target', 41.62, 21.35), ('target', 48.09, 1.38), ('target', 88.57, 1.38), ('target', 93.32, 1.38)],
     "shots": 4},
    # seed 9044, targets=3; 22 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 47.68, 11.47, 0.004), ('block', 'stone', 2.0, 6.0, 76.19, 6.48, -0.0), ('block', 'stone', 4.0, 3.0, 58.06, 12.96, -0.0), ('block', 'wood', 2.0, 6.0, 47.72, 3.49, 0.006), ('block', 'wood', 2.0, 6.0, 58.06, 3.48, 0.001), ('block', 'wood', 2.0, 6.0, 86.36, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 92.36, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 58.06, 7.98, -0.001), ('block', 'wood', 4.0, 3.0, 76.2, 1.98, -0.002), ('block', 'wood', 6.0, 2.0, 47.71, 7.47, 0.007), ('block', 'wood', 6.0, 2.0, 58.06, 10.47, -0.001), ('block', 'wood', 8.0, 2.0, 89.36, 7.49, -0.002), ('target', 37.68, 1.29), ('target', 58.21, 15.35), ('target', 89.36, 1.38)],
     "shots": 3},
    # seed 9045, targets=4; 21 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 74.95, 3.48, 0.001), ('block', 'stone', 2.0, 8.0, 87.47, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 74.92, 19.97, 0.0), ('block', 'wood', 2.0, 6.0, 42.08, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 48.08, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 56.46, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 62.46, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 74.92, 15.47, 0.002), ('block', 'wood', 2.0, 6.0, 74.94, 9.48, 0.003), ('block', 'wood', 8.0, 2.0, 45.08, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 59.46, 7.49, -0.002), ('target', 45.08, 1.38), ('target', 59.46, 1.38), ('target', 75.08, 22.37), ('target', 91.47, 1.38)],
     "shots": 4},
    # seed 9046, targets=3; 29 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 59.08, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 65.08, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 87.37, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 93.37, 3.5, -0.001), ('block', 'stone', 4.0, 3.0, 44.13, 4.98, -0.001), ('block', 'stone', 6.0, 2.0, 77.95, 1.48, 0.005), ('block', 'wood', 2.0, 6.0, 77.82, 17.47, 0.008), ('block', 'wood', 2.0, 6.0, 77.87, 11.47, 0.008), ('block', 'wood', 2.0, 6.0, 77.92, 5.48, 0.007), ('block', 'wood', 4.0, 3.0, 44.15, 1.99, 0.002), ('block', 'wood', 8.0, 2.0, 62.08, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 90.37, 7.49, -0.002), ('target', 44.4, 7.37), ('target', 73.32, 1.19), ('target', 90.37, 1.38)],
     "shots": 3},
    # seed 9047, targets=4; 26 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 39.68, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 45.68, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 62.29, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 87.18, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 72.96, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 78.96, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 42.68, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 75.96, 7.49, -0.002), ('target', 42.68, 1.38), ('target', 66.29, 1.38), ('target', 75.96, 1.38), ('target', 91.18, 1.38)],
     "shots": 4},
    # seed 9048, targets=3; 16 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 46.05, 9.47, 0.01), ('block', 'stone', 2.0, 6.0, 63.41, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 69.41, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 89.48, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 46.0, 13.97, 0.01), ('block', 'wood', 2.0, 6.0, 46.09, 3.48, 0.003), ('block', 'wood', 8.0, 2.0, 66.41, 7.49, -0.002), ('target', 41.8, 9.2), ('target', 66.41, 1.38), ('target', 93.48, 1.38)],
     "shots": 3},
    # seed 9049, targets=4; 14 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 42.01, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 48.01, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 74.21, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 60.76, 3.96, 0.007), ('block', 'stone', 4.0, 3.0, 92.83, 1.98, -0.002), ('block', 'wood', 2.0, 6.0, 92.81, 12.47, 0.007), ('block', 'wood', 2.0, 6.0, 92.83, 6.48, -0.0), ('block', 'wood', 4.0, 3.0, 60.71, 9.95, 0.005), ('block', 'wood', 4.0, 3.0, 60.74, 6.95, 0.004), ('block', 'wood', 6.0, 2.0, 60.78, 1.48, 0.003), ('block', 'wood', 8.0, 2.0, 45.01, 7.49, -0.002), ('target', 45.01, 1.38), ('target', 59.12, 12.33), ('target', 78.21, 1.38), ('target', 88.3, 1.16)],
     "shots": 4},
    # seed 9500, targets=4; 29 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 74.8, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 80.8, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 44.72, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 50.72, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 57.58, 9.46, -0.009), ('block', 'wood', 2.0, 6.0, 90.74, 3.48, 0.003), ('block', 'wood', 4.0, 3.0, 57.52, 1.99, -0.0), ('block', 'wood', 4.0, 3.0, 57.54, 4.96, -0.007), ('block', 'wood', 4.0, 3.0, 57.62, 13.95, -0.01), ('block', 'wood', 4.0, 3.0, 90.73, 7.97, 0.004), ('block', 'wood', 6.0, 2.0, 90.72, 10.46, 0.0), ('block', 'wood', 8.0, 2.0, 47.72, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 77.8, 7.49, -0.002), ('target', 47.72, 1.38), ('target', 59.58, 16.33), ('target', 77.8, 1.38), ('target', 90.61, 12.34)],
     "shots": 4},
    # seed 9501, targets=3; 20 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 45.33, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 85.44, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 91.44, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 68.6, 4.98, 0.002), ('block', 'wood', 4.0, 3.0, 68.61, 1.99, 0.002), ('block', 'wood', 6.0, 2.0, 68.59, 7.47, 0.002), ('block', 'wood', 6.0, 2.0, 68.59, 9.47, 0.003), ('block', 'wood', 8.0, 2.0, 88.44, 7.49, -0.002), ('target', 49.33, 1.38), ('target', 67.69, 11.35), ('target', 88.44, 1.38)],
     "shots": 3},
    # seed 9502, targets=4; 26 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 41.95, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 47.95, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 70.69, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 76.69, 3.5, -0.001), ('block', 'stone', 6.0, 2.0, 60.97, 1.48, -0.0), ('block', 'wood', 2.0, 6.0, 60.97, 5.48, 0.001), ('block', 'wood', 2.0, 6.0, 89.68, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 95.68, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 44.95, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 73.69, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 92.68, 7.49, -0.002), ('target', 44.95, 1.38), ('target', 60.78, 9.36), ('target', 73.69, 1.38), ('target', 92.68, 1.38)],
     "shots": 4},
    # seed 9503, targets=3; 19 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 88.0, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 94.0, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 58.04, 4.5, -0.0), ('block', 'stone', 6.0, 2.0, 72.7, 4.47, -0.005), ('block', 'wood', 2.0, 6.0, 41.91, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 47.91, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 72.72, 8.46, -0.006), ('block', 'wood', 4.0, 3.0, 72.7, 1.99, -0.001), ('block', 'wood', 6.0, 2.0, 72.75, 12.46, -0.007), ('block', 'wood', 8.0, 2.0, 44.91, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 91.0, 7.49, -0.002), ('target', 44.91, 1.38), ('target', 62.04, 1.38), ('target', 91.0, 1.38)],
     "shots": 3},
    # seed 9504, targets=4; 16 expansions, 3-shot plan
    {"bodies": [('block', 'wood', 2.0, 6.0, 41.93, 3.5, -0.0), ('block', 'wood', 2.0, 6.0, 47.72, 3.49, -0.008), ('block', 'wood', 2.0, 6.0, 85.95, 3.5, -0.0), ('block', 'wood', 2.0, 6.0, 91.74, 3.49, -0.008), ('block', 'wood', 8.0, 2.0, 44.88, 7.5, -0.001), ('block', 'wood', 8.0, 2.0, 88.91, 7.5, -0.001), ('target', 44.93, 1.38), ('target', 49.51, 1.38), ('target', 88.95, 1.38), ('target', 93.53, 1.38)],
     "shots": 4},
    # seed 9505, targets=3; 6 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 4.0, 3.0, 45.45, 7.94, 0.005), ('block', 'stone', 6.0, 2.0, 45.49, 3.46, 0.002), ('block', 'stone', 6.0, 2.0, 91.33, 1.49, 0.004), ('block', 'wood', 6.0, 2.0, 45.47, 5.45, 0.004), ('block', 'wood', 6.0, 2.0, 45.5, 1.49, 0.001), ('block', 'wood', 6.0, 2.0, 91.33, 3.49, 0.004), ('target', 43.83, 10.32), ('target', 49.5, 1.38), ('target', 90.11, 5.37)],
     "shots": 3},
    # seed 9506, targets=4; 69 expansions, 4-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 56.44, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 62.44, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 71.89, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 77.89, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 91.16, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 43.36, 2.0, 0.002), ('block', 'wood', 4.0, 3.0, 43.36, 4.99, 0.001), ('block', 'wood', 6.0, 2.0, 43.35, 7.49, 0.001), ('block', 'wood', 8.0, 2.0, 59.44, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 74.89, 7.49, -0.002), ('target', 43.04, 9.37), ('target', 59.44, 1.38), ('target', 74.89, 1.38), ('target', 95.16, 1.38)],
     "shots": 4},
    # seed 9507, targets=3; 13 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 44.64, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 50.64, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 58.96, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 72.67, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 78.67, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 88.02, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 94.02, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 47.64, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 75.67, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 91.02, 7.49, -0.002), ('target', 47.64, 1.38), ('target', 62.96, 1.38), ('target', 75.67, 1.38)],
     "shots": 3},
    # seed 9508, targets=4; 44 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 41.91, 3.5, 0.0), ('block', 'stone', 2.0, 6.0, 47.81, 3.49, -0.007), ('block', 'stone', 2.0, 6.0, 87.13, 3.49, 0.009), ('block', 'stone', 2.0, 6.0, 93.03, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 44.89, 7.5, -0.0), ('block', 'wood', 8.0, 2.0, 90.04, 7.49, 0.002), ('target', 44.91, 1.38), ('target', 49.64, 1.38), ('target', 85.26, 1.38), ('target', 90.03, 1.38)],
     "shots": 4},
    # seed 9509, targets=3; 36 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 41.31, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 47.31, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 69.24, 9.48, 0.01), ('block', 'stone', 2.0, 6.0, 87.74, 12.47, 0.009), ('block', 'wood', 2.0, 6.0, 69.16, 15.47, 0.015), ('block', 'wood', 2.0, 6.0, 69.27, 3.48, 0.003), ('block', 'wood', 2.0, 6.0, 87.8, 3.48, 0.002), ('block', 'wood', 4.0, 3.0, 87.78, 7.97, 0.01), ('block', 'wood', 8.0, 2.0, 44.31, 7.49, -0.002), ('target', 44.31, 1.38), ('target', 61.35, 1.3), ('target', 81.71, 1.39)],
     "shots": 3},
    # seed 9510, targets=4; 26 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 44.3, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 66.45, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 72.45, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 84.98, 3.49, 0.009), ('block', 'wood', 2.0, 6.0, 90.79, 3.5, -0.0), ('block', 'wood', 8.0, 2.0, 69.45, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 87.83, 7.49, -0.002), ('target', 48.3, 1.38), ('target', 69.45, 1.38), ('target', 82.88, 1.38), ('target', 87.79, 1.38)],
     "shots": 4},
    # seed 9511, targets=3; 23 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 42.1, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 48.1, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 64.51, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 70.51, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 86.24, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 92.24, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 45.1, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 67.51, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 89.24, 7.49, -0.002), ('target', 45.1, 1.38), ('target', 67.51, 1.38), ('target', 89.24, 1.38)],
     "shots": 3},
    # seed 9512, targets=4; 26 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 54.18, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 60.18, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 88.37, 18.48, 0.007), ('block', 'stone', 2.0, 6.0, 88.45, 6.49, 0.006), ('block', 'stone', 2.0, 8.0, 43.05, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 88.48, 1.99, 0.004), ('block', 'wood', 2.0, 6.0, 75.46, 5.48, 0.004), ('block', 'wood', 2.0, 6.0, 88.41, 12.48, 0.007), ('block', 'wood', 6.0, 2.0, 75.48, 1.48, 0.005), ('block', 'wood', 8.0, 2.0, 57.18, 7.5, -0.0), ('target', 47.05, 1.38), ('target', 57.18, 1.38), ('target', 69.86, 1.38), ('target', 84.71, 8.95)],
     "shots": 4},
    # seed 9513, targets=3; 18 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 44.05, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 50.05, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 59.14, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 65.14, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 86.28, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 92.28, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 73.11, 4.5, -0.0), ('block', 'wood', 8.0, 2.0, 47.05, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 62.14, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 89.28, 7.49, -0.002), ('target', 62.14, 1.38), ('target', 77.11, 1.38), ('target', 89.28, 1.38)],
     "shots": 3},
    # seed 9514, targets=4; 168 expansions, 4-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 72.17, 9.48, 0.01), ('block', 'stone', 2.0, 6.0, 87.94, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 93.94, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 44.06, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 62.8, 8.94, -0.002), ('block', 'stone', 6.0, 2.0, 62.81, 1.49, 0.003), ('block', 'wood', 2.0, 6.0, 72.09, 15.47, 0.015), ('block', 'wood', 2.0, 6.0, 72.2, 3.48, 0.003), ('block', 'wood', 4.0, 3.0, 62.81, 5.95, -0.003), ('block', 'wood', 6.0, 2.0, 62.81, 3.47, 0.001), ('block', 'wood', 8.0, 2.0, 90.94, 7.49, -0.002), ('target', 48.06, 1.38), ('target', 63.22, 11.33), ('target', 66.7, 1.3), ('target', 90.94, 1.38)],
     "shots": 4},
    # seed 9515, targets=3; 25 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 44.87, 9.46, 0.002), ('block', 'stone', 2.0, 6.0, 85.68, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 91.68, 3.5, -0.001), ('block', 'stone', 6.0, 2.0, 44.89, 1.49, 0.002), ('block', 'stone', 6.0, 2.0, 44.89, 3.47, 0.0), ('block', 'wood', 6.0, 2.0, 44.88, 5.46, 0.002), ('block', 'wood', 8.0, 2.0, 88.68, 7.49, -0.002), ('target', 44.33, 13.35), ('target', 48.89, 1.38), ('target', 88.68, 1.38)],
     "shots": 3},
    # seed 9516, targets=4; 39 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 54.91, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 60.91, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 91.24, 8.44, -0.011), ('block', 'stone', 4.0, 3.0, 91.29, 12.94, -0.012), ('block', 'wood', 2.0, 6.0, 43.93, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 49.93, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 74.17, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 80.17, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 91.2, 1.99, 0.001), ('block', 'wood', 6.0, 2.0, 91.2, 4.45, -0.007), ('block', 'wood', 8.0, 2.0, 46.93, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 57.91, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 77.17, 7.49, -0.002), ('target', 46.93, 1.38), ('target', 57.91, 1.38), ('target', 77.17, 1.38), ('target', 93.51, 15.28)],
     "shots": 4},
    # seed 9517, targets=3; 13 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 65.16, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 92.9, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 41.65, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 47.65, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 44.65, 7.49, -0.002), ('target', 44.65, 1.38), ('target', 69.16, 1.38), ('target', 96.9, 1.38)],
     "shots": 3},
    # seed 9518, targets=4; 15 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 73.73, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 79.73, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 59.7, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 43.07, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 49.07, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 89.53, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 95.53, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 46.07, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 76.73, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 92.53, 7.49, -0.002), ('target', 46.07, 1.38), ('target', 63.7, 1.38), ('target', 76.73, 1.38), ('target', 92.53, 1.38)],
     "shots": 4},
    # seed 9519, targets=3; 14 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 39.41, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 45.41, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 73.89, 3.5, -0.0), ('block', 'stone', 2.0, 8.0, 58.3, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 73.88, 9.48, 0.001), ('block', 'wood', 2.0, 6.0, 87.06, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 93.06, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 42.41, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 90.06, 7.49, -0.002), ('target', 62.3, 1.38), ('target', 73.5, 13.36), ('target', 90.06, 1.38)],
     "shots": 3},
    # seed 9520, targets=4; 25 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 41.55, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 47.55, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 66.69, 1.43, 1.578), ('block', 'wood', 2.0, 6.0, 87.31, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 93.31, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 44.55, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 90.31, 7.49, -0.002), ('target', 44.55, 1.38), ('target', 73.58, 1.3), ('target', 75.36, 1.38), ('target', 90.31, 1.38)],
     "shots": 4},
    # seed 9521, targets=3; 27 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 44.51, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 57.27, 13.95, -0.003), ('block', 'wood', 2.0, 6.0, 57.26, 9.45, -0.002), ('block', 'wood', 2.0, 6.0, 71.3, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 77.3, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 87.48, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 93.48, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 57.24, 1.99, 0.001), ('block', 'wood', 4.0, 3.0, 57.25, 4.96, -0.004), ('block', 'wood', 8.0, 2.0, 74.3, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 90.48, 7.49, -0.002), ('target', 48.51, 1.38), ('target', 74.3, 1.38), ('target', 90.48, 1.38)],
     "shots": 3},
    # seed 9522, targets=4; 9 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 44.48, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 50.48, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 88.69, 8.46, -0.007), ('block', 'stone', 2.0, 8.0, 59.31, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 73.09, 4.5, -0.0), ('block', 'stone', 6.0, 2.0, 88.67, 4.46, -0.003), ('block', 'wood', 4.0, 3.0, 88.67, 1.99, 0.001), ('block', 'wood', 8.0, 2.0, 47.48, 7.49, -0.002), ('target', 47.48, 1.38), ('target', 63.31, 1.38), ('target', 77.09, 1.38), ('target', 91.65, 7.0)],
     "shots": 4},
    # seed 9523, targets=3; 23 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 73.33, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 79.33, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 43.79, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 59.74, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 85.44, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 91.44, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 76.33, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 88.44, 7.49, -0.002), ('target', 47.79, 1.38), ('target', 63.74, 1.38), ('target', 88.44, 1.38)],
     "shots": 3},
    # seed 9524, targets=4; 30 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 39.91, 3.5, 0.0), ('block', 'stone', 2.0, 6.0, 45.81, 3.49, -0.007), ('block', 'wood', 2.0, 6.0, 62.56, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 68.56, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 89.19, 5.48, 0.004), ('block', 'wood', 6.0, 2.0, 89.21, 1.48, 0.005), ('block', 'wood', 8.0, 2.0, 42.89, 7.5, -0.0), ('block', 'wood', 8.0, 2.0, 65.56, 7.49, -0.002), ('target', 42.91, 1.38), ('target', 47.66, 1.38), ('target', 65.56, 1.38), ('target', 83.59, 1.38)],
     "shots": 4},
    # seed 9525, targets=3; 15 expansions, 3-shot plan
    {"bodies": [('block', 'wood', 2.0, 6.0, 42.25, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 48.25, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 64.17, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 70.17, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 89.56, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 95.56, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 45.25, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 67.17, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 92.56, 7.49, -0.002), ('target', 45.25, 1.38), ('target', 67.17, 1.38), ('target', 92.56, 1.38)],
     "shots": 3},
    # seed 9526, targets=4; 38 expansions, 4-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 84.39, 3.5, 0.0), ('block', 'stone', 2.0, 6.0, 90.29, 3.49, -0.007), ('block', 'stone', 2.0, 8.0, 45.27, 4.5, -0.0), ('block', 'wood', 8.0, 2.0, 87.37, 7.5, -0.0), ('target', 49.27, 1.38), ('target', 53.27, 1.38), ('target', 87.39, 1.38), ('target', 92.14, 1.38)],
     "shots": 4},
    # seed 9527, targets=3; 19 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 56.1, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 62.1, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 42.3, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 89.47, 6.48, 0.003), ('block', 'wood', 4.0, 3.0, 89.48, 1.98, 0.0), ('block', 'wood', 6.0, 2.0, 76.9, 7.45, 0.005), ('block', 'wood', 6.0, 2.0, 76.92, 5.46, 0.004), ('block', 'wood', 6.0, 2.0, 76.93, 3.47, 0.002), ('block', 'wood', 6.0, 2.0, 76.94, 1.49, 0.002), ('block', 'wood', 8.0, 2.0, 59.1, 7.49, -0.002), ('target', 46.3, 1.38), ('target', 59.1, 1.38), ('target', 75.49, 9.34)],
     "shots": 3},
    # seed 9528, targets=4; 22 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 41.1, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 47.1, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 67.28, 4.5, -0.0), ('block', 'wood', 4.0, 3.0, 92.69, 1.99, 0.001), ('block', 'wood', 6.0, 2.0, 92.68, 4.49, -0.0), ('block', 'wood', 8.0, 2.0, 44.1, 7.49, -0.002), ('target', 44.1, 1.38), ('target', 71.28, 1.38), ('target', 92.79, 6.37), ('target', 96.69, 1.38)],
     "shots": 4},
    # seed 9529, targets=3; 31 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 42.3, 3.5, 0.0), ('block', 'stone', 2.0, 6.0, 48.2, 3.49, -0.007), ('block', 'stone', 2.0, 6.0, 85.98, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 91.98, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 45.28, 7.5, -0.0), ('block', 'wood', 8.0, 2.0, 88.98, 7.49, -0.002), ('target', 45.3, 1.38), ('target', 50.03, 1.38), ('target', 88.98, 1.38)],
     "shots": 3},
    # seed 9530, targets=4; 124 expansions, 4-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 39.27, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 45.27, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 56.23, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 62.23, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 88.48, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 94.48, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 72.11, 3.99, 0.004), ('block', 'wood', 6.0, 2.0, 72.11, 1.49, 0.004), ('block', 'wood', 8.0, 2.0, 42.27, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 59.23, 7.5, -0.0), ('block', 'wood', 8.0, 2.0, 91.48, 7.49, -0.002), ('target', 42.27, 1.38), ('target', 59.23, 1.38), ('target', 70.85, 6.37), ('target', 91.48, 1.38)],
     "shots": 4},
    # seed 9531, targets=3; 17 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 62.39, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 68.39, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 92.88, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 39.97, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 45.97, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 42.97, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 65.39, 7.49, -0.002), ('target', 42.97, 1.38), ('target', 65.39, 1.38), ('target', 96.88, 1.38)],
     "shots": 3},
    # seed 9532, targets=4; 11 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 44.63, 3.5, 0.0), ('block', 'stone', 2.0, 6.0, 50.53, 3.49, -0.007), ('block', 'wood', 2.0, 6.0, 87.68, 3.49, 0.009), ('block', 'wood', 2.0, 6.0, 93.49, 3.5, -0.0), ('block', 'wood', 8.0, 2.0, 47.61, 7.5, -0.0), ('block', 'wood', 8.0, 2.0, 90.53, 7.49, -0.002), ('target', 47.63, 1.38), ('target', 52.36, 1.38), ('target', 85.58, 1.38), ('target', 90.49, 1.38)],
     "shots": 4},
    # seed 9533, targets=3; 22 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 71.94, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 77.94, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 87.37, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 93.37, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 43.17, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 59.43, 12.48, 0.008), ('block', 'wood', 2.0, 6.0, 59.48, 6.49, 0.007), ('block', 'wood', 4.0, 3.0, 59.5, 1.99, 0.004), ('block', 'wood', 8.0, 2.0, 74.94, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 90.37, 7.49, -0.002), ('target', 55.67, 4.05), ('target', 74.94, 1.38), ('target', 90.37, 1.38)],
     "shots": 3},
    # seed 9534, targets=4; 35 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 59.12, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 65.12, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 70.35, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 76.35, 3.5, -0.001), ('block', 'stone', 4.0, 3.0, 46.89, 10.98, 0.006), ('block', 'stone', 4.0, 3.0, 46.96, 1.99, 0.005), ('block', 'stone', 4.0, 3.0, 89.7, 3.98, 0.004), ('block', 'wood', 2.0, 6.0, 46.93, 6.48, 0.007), ('block', 'wood', 6.0, 2.0, 46.88, 13.48, 0.006), ('block', 'wood', 6.0, 2.0, 89.7, 1.49, 0.004), ('block', 'wood', 8.0, 2.0, 62.12, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 73.35, 7.49, -0.002), ('target', 44.97, 15.35), ('target', 62.12, 1.38), ('target', 73.35, 1.38), ('target', 88.45, 6.36)],
     "shots": 4},
    # seed 9535, targets=3; 16 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 40.93, 3.5, 0.0), ('block', 'stone', 2.0, 6.0, 46.83, 3.49, -0.007), ('block', 'stone', 2.0, 6.0, 87.09, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 93.09, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 43.91, 7.5, -0.0), ('block', 'wood', 8.0, 2.0, 90.09, 7.49, -0.002), ('target', 43.93, 1.38), ('target', 48.66, 1.38), ('target', 90.09, 1.38)],
     "shots": 3},
    # seed 9536, targets=4; 14 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 42.12, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 48.12, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 60.06, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 77.44, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 90.95, 9.49, 0.004), ('block', 'wood', 2.0, 6.0, 90.96, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 60.06, 9.99, 0.004), ('block', 'wood', 6.0, 2.0, 60.06, 7.49, 0.0), ('block', 'wood', 6.0, 2.0, 90.93, 13.48, 0.011), ('block', 'wood', 8.0, 2.0, 45.12, 7.49, -0.002), ('target', 45.12, 1.38), ('target', 58.87, 12.36), ('target', 81.44, 1.38), ('target', 86.91, 14.4)],
     "shots": 4},
    # seed 9537, targets=3; 12 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 44.98, 3.49, 0.009), ('block', 'stone', 2.0, 6.0, 50.88, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 90.02, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 90.02, 7.99, 0.002), ('block', 'wood', 6.0, 2.0, 90.01, 10.49, 0.004), ('block', 'wood', 8.0, 2.0, 47.89, 7.49, 0.002), ('target', 43.11, 1.38), ('target', 47.88, 1.38), ('target', 88.79, 12.36)],
     "shots": 3},
    # seed 9538, targets=4; 22 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 56.33, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 62.33, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 43.23, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 91.1, 1.99, 0.004), ('block', 'stone', 6.0, 2.0, 74.67, 7.47, 0.006), ('block', 'wood', 2.0, 6.0, 74.69, 3.48, 0.003), ('block', 'wood', 2.0, 6.0, 91.08, 6.49, 0.006), ('block', 'wood', 4.0, 3.0, 74.66, 9.97, 0.006), ('block', 'wood', 4.0, 3.0, 91.03, 13.98, 0.006), ('block', 'wood', 4.0, 3.0, 91.05, 10.98, 0.005), ('block', 'wood', 8.0, 2.0, 59.33, 7.49, -0.002), ('target', 47.23, 1.38), ('target', 59.33, 1.38), ('target', 72.73, 12.34), ('target', 89.44, 16.37)],
     "shots": 4},
    # seed 9539, targets=3; 11 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 44.69, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 50.69, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 65.75, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 91.38, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 65.75, 9.49, 0.001), ('block', 'wood', 4.0, 3.0, 65.74, 13.98, 0.011), ('block', 'wood', 8.0, 2.0, 47.69, 7.49, -0.002), ('target', 47.69, 1.38), ('target', 61.02, 4.76), ('target', 95.38, 1.38)],
     "shots": 3},
    # seed 9540, targets=4; 15 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 91.24, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 42.79, 5.48, 0.004), ('block', 'wood', 2.0, 6.0, 65.71, 3.5, -0.0), ('block', 'wood', 2.0, 6.0, 71.5, 3.49, -0.008), ('block', 'wood', 6.0, 2.0, 42.81, 1.48, 0.005), ('block', 'wood', 8.0, 2.0, 68.67, 7.5, -0.001), ('target', 37.19, 1.38), ('target', 68.71, 1.38), ('target', 73.29, 1.38), ('target', 95.24, 1.38)],
     "shots": 4},
    # seed 9541, targets=3; 32 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 40.84, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 46.84, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 60.96, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 70.75, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 76.75, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 88.61, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 94.61, 3.5, -0.001), ('block', 'wood', 6.0, 2.0, 60.95, 7.48, 0.002), ('block', 'wood', 8.0, 2.0, 43.84, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 73.75, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 91.61, 7.49, -0.002), ('target', 43.84, 1.38), ('target', 73.75, 1.38), ('target', 91.61, 1.38)],
     "shots": 3},
    # seed 9542, targets=4; 26 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 54.44, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 60.44, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 91.99, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 46.34, 6.48, 0.0), ('block', 'wood', 2.0, 6.0, 69.46, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 75.46, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 46.34, 1.98, -0.0), ('block', 'wood', 8.0, 2.0, 57.44, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 72.46, 7.49, -0.002), ('target', 46.26, 10.36), ('target', 57.44, 1.38), ('target', 72.46, 1.38), ('target', 95.99, 1.38)],
     "shots": 4},
    # seed 9543, targets=3; 25 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 59.68, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 65.68, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 44.01, 4.5, -0.0), ('block', 'stone', 6.0, 2.0, 77.22, 1.48, -0.0), ('block', 'wood', 2.0, 6.0, 89.73, 11.47, 0.001), ('block', 'wood', 4.0, 3.0, 77.22, 3.98, 0.0), ('block', 'wood', 4.0, 3.0, 89.73, 6.98, 0.002), ('block', 'wood', 4.0, 3.0, 89.74, 1.99, 0.002), ('block', 'wood', 6.0, 2.0, 89.73, 4.48, 0.001), ('block', 'wood', 8.0, 2.0, 62.68, 7.49, -0.002), ('target', 48.01, 1.38), ('target', 62.68, 1.38), ('target', 77.07, 6.36)],
     "shots": 3},
    # seed 9544, targets=4; 71 expansions, 4-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 71.45, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 77.45, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 57.76, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 92.55, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 43.79, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 49.79, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 46.79, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 74.45, 7.49, -0.002), ('target', 46.79, 1.38), ('target', 61.76, 1.38), ('target', 74.45, 1.38), ('target', 96.55, 1.38)],
     "shots": 4},
    # seed 9545, targets=3; 23 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 58.86, 11.47, 0.005), ('block', 'stone', 2.0, 6.0, 70.83, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 76.83, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 58.82, 17.47, 0.009), ('block', 'wood', 2.0, 6.0, 58.9, 3.48, 0.002), ('block', 'wood', 2.0, 6.0, 87.82, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 93.82, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 46.45, 1.99, 0.002), ('block', 'wood', 6.0, 2.0, 46.43, 6.48, 0.001), ('block', 'wood', 6.0, 2.0, 46.44, 4.49, 0.0), ('block', 'wood', 6.0, 2.0, 58.88, 7.47, 0.005), ('block', 'wood', 8.0, 2.0, 73.83, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 90.82, 7.49, -0.002), ('target', 46.18, 8.37), ('target', 73.83, 1.38), ('target', 90.82, 1.38)],
     "shots": 3},
    # seed 9546, targets=4; 26 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 72.81, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 84.77, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 90.77, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 45.08, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 57.94, 7.97, 0.002), ('block', 'wood', 4.0, 3.0, 57.94, 4.98, 0.0), ('block', 'wood', 4.0, 3.0, 57.95, 1.99, 0.001), ('block', 'wood', 6.0, 2.0, 72.82, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 87.77, 7.49, -0.002), ('target', 49.08, 1.38), ('target', 57.33, 10.35), ('target', 73.35, 9.37), ('target', 87.77, 1.38)],
     "shots": 4},
    # seed 9547, targets=3; 27 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 58.45, 3.48, 0.001), ('block', 'stone', 2.0, 6.0, 72.39, 9.47, 0.01), ('block', 'stone', 2.0, 6.0, 87.88, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 93.88, 3.5, -0.001), ('block', 'stone', 4.0, 3.0, 72.34, 13.97, 0.01), ('block', 'wood', 2.0, 6.0, 39.13, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 45.13, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 72.43, 3.48, 0.003), ('block', 'wood', 4.0, 3.0, 58.43, 12.96, -0.0), ('block', 'wood', 4.0, 3.0, 58.44, 9.96, -0.001), ('block', 'wood', 6.0, 2.0, 58.44, 7.47, 0.004), ('block', 'wood', 8.0, 2.0, 42.13, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 90.88, 7.49, -0.002), ('target', 42.13, 1.38), ('target', 68.14, 9.2), ('target', 90.88, 1.38)],
     "shots": 3},
    # seed 9548, targets=4; 18 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 41.9, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 47.9, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 62.57, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 68.57, 3.5, -0.0), ('block', 'stone', 2.0, 8.0, 85.96, 1.43, 1.578), ('block', 'wood', 8.0, 2.0, 44.9, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 65.57, 7.5, -0.0), ('target', 44.9, 1.38), ('target', 65.57, 1.38), ('target', 92.85, 1.3), ('target', 94.63, 1.38)],
     "shots": 4},
    # seed 9549, targets=3; 24 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 39.93, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 45.93, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 71.24, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 77.24, 3.5, -0.001), ('block', 'stone', 6.0, 2.0, 87.1, 1.49, 0.004), ('block', 'wood', 2.0, 6.0, 56.45, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 62.45, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 87.06, 8.46, 0.005), ('block', 'wood', 4.0, 3.0, 87.09, 3.97, 0.007), ('block', 'wood', 8.0, 2.0, 42.93, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 59.45, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 74.24, 7.49, -0.002), ('target', 42.93, 1.38), ('target', 59.45, 1.38), ('target', 79.89, 1.24)],
     "shots": 3},
)
