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
                          min_plan_length=2, search_limit=400, attempts=60):
        """Draw a level, select it, and return it as the dict `set_instance` takes.

        `structures`, `targets` and `shots` are `draw_level`'s; each left unset is drawn from
        the bundled levels' range (two to four structures, two or three targets, one shot
        more than targets). A draw is kept only if a breadth-first search over shots finds a
        plan within `search_limit` expansions and that plan, a shortest one, has at least
        `min_plan_length` shots, so a level that one shot flattens is thrown back. The plan
        is left in `witness` and what the search spent in `witness_expansions`.
        """
        random_, _ = rng(seed)
        found = {}

        def draw(attempt):
            options = dict(structures=structures or random_.randint(2, 4),
                           targets=targets or random_.randint(2, 3))
            options["shots"] = shots or options["targets"] + 1
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


#: The bundled levels: `generate_instance(seed)` for the seed beside each, embedded as the
#: plain data `set_instance` takes, with the plan each was accepted on in
#: `tests/data/slingshot_solutions.json`.
LEVELS = (
    # seed 1000; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 41.91, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 47.91, 3.5, -0.001), ('block', 'stone', 6.0, 2.0, 66.47, 4.48, -0.004), ('block', 'wood', 2.0, 6.0, 84.45, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 90.45, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 66.47, 1.99, -0.001), ('block', 'wood', 8.0, 2.0, 44.91, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 87.45, 7.49, -0.002), ('target', 44.91, 1.38), ('target', 67.66, 6.35), ('target', 87.45, 1.38)],
     "shots": 4},
    # seed 1001; 4 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 42.0, 14.46, 0.008), ('block', 'stone', 2.0, 6.0, 89.4, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 60.26, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 42.07, 5.47, 0.008), ('block', 'wood', 2.0, 6.0, 69.54, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 75.54, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 42.03, 9.96, 0.007), ('block', 'wood', 4.0, 3.0, 89.41, 7.99, -0.002), ('block', 'wood', 6.0, 2.0, 42.1, 1.47, 0.006), ('block', 'wood', 8.0, 2.0, 72.54, 7.49, -0.002), ('target', 64.26, 1.38), ('target', 90.03, 10.37)],
     "shots": 3},
    # seed 1002; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 92.63, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 75.38, 6.96, 0.001), ('block', 'wood', 2.0, 6.0, 60.34, 17.46, 0.009), ('block', 'wood', 2.0, 6.0, 60.4, 11.47, 0.01), ('block', 'wood', 2.0, 6.0, 60.46, 5.47, 0.009), ('block', 'wood', 4.0, 3.0, 47.4, 1.99, 0.001), ('block', 'wood', 4.0, 3.0, 75.39, 1.99, 0.001), ('block', 'wood', 6.0, 2.0, 47.39, 4.49, -0.0), ('block', 'wood', 6.0, 2.0, 60.49, 1.48, 0.006), ('block', 'wood', 6.0, 2.0, 75.37, 9.46, 0.001), ('block', 'wood', 6.0, 2.0, 75.38, 4.47, 0.0), ('target', 47.5, 6.37), ('target', 55.15, 1.4), ('target', 96.63, 1.38)],
     "shots": 4},
    # seed 1003; 4 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 43.33, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 67.18, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 86.42, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 92.42, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 89.42, 7.49, -0.002), ('target', 47.33, 1.38), ('target', 71.18, 1.38), ('target', 89.42, 1.38)],
     "shots": 4},
    # seed 1004; 2 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 4.0, 3.0, 43.03, 7.99, 0.007), ('block', 'wood', 2.0, 6.0, 43.04, 3.5, -0.0), ('block', 'wood', 2.0, 6.0, 87.33, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 93.33, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 90.33, 7.49, -0.002), ('target', 41.02, 10.36), ('target', 47.04, 1.38), ('target', 90.33, 1.38)],
     "shots": 4},
    # seed 1005; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 42.22, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 48.22, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 67.77, 8.46, 0.003), ('block', 'wood', 2.0, 6.0, 90.89, 12.48, 0.008), ('block', 'wood', 2.0, 6.0, 90.94, 6.49, 0.007), ('block', 'wood', 4.0, 3.0, 67.79, 3.96, 0.007), ('block', 'wood', 4.0, 3.0, 90.96, 1.99, 0.004), ('block', 'wood', 6.0, 2.0, 67.8, 1.49, 0.004), ('block', 'wood', 8.0, 2.0, 45.22, 7.49, -0.002), ('target', 45.22, 1.38), ('target', 59.49, 1.33), ('target', 87.13, 4.05)],
     "shots": 4},
    # seed 1006; 4 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 43.36, 6.48, 0.003), ('block', 'stone', 2.0, 8.0, 90.52, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 43.37, 1.98, 0.0), ('block', 'wood', 2.0, 6.0, 66.1, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 72.1, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 69.1, 7.49, -0.002), ('target', 69.1, 1.38), ('target', 94.52, 1.38)],
     "shots": 3},
    # seed 1007; 5 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 42.2, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 72.8, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 78.8, 3.5, -0.001), ('block', 'stone', 4.0, 3.0, 87.55, 1.99, 0.0), ('block', 'stone', 6.0, 2.0, 62.5, 1.49, 0.004), ('block', 'wood', 2.0, 6.0, 62.46, 8.47, 0.008), ('block', 'wood', 4.0, 3.0, 62.49, 3.97, 0.007), ('block', 'wood', 4.0, 3.0, 87.55, 4.98, 0.0), ('block', 'wood', 6.0, 2.0, 42.21, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 75.8, 7.49, -0.002), ('target', 42.74, 9.37), ('target', 75.8, 1.38)],
     "shots": 3},
    # seed 1008; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 57.96, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 63.96, 3.5, -0.001), ('block', 'stone', 4.0, 3.0, 43.79, 4.97, -0.001), ('block', 'stone', 4.0, 3.0, 43.79, 7.96, -0.0), ('block', 'wood', 2.0, 6.0, 71.49, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 77.49, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 90.25, 6.48, -0.0), ('block', 'wood', 4.0, 3.0, 43.8, 1.99, 0.002), ('block', 'wood', 4.0, 3.0, 90.25, 1.98, -0.0), ('block', 'wood', 4.0, 3.0, 90.25, 10.98, -0.0), ('block', 'wood', 8.0, 2.0, 60.96, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 74.49, 7.49, -0.002), ('target', 43.8, 10.35), ('target', 60.96, 1.38), ('target', 90.45, 13.36)],
     "shots": 4},
    # seed 1009; 5 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 86.24, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 92.24, 3.5, -0.001), ('block', 'stone', 6.0, 2.0, 44.97, 9.46, 0.009), ('block', 'wood', 2.0, 6.0, 44.93, 13.46, 0.009), ('block', 'wood', 2.0, 6.0, 45.02, 5.47, 0.01), ('block', 'wood', 4.0, 3.0, 67.05, 7.98, 0.001), ('block', 'wood', 4.0, 3.0, 67.06, 4.99, 0.001), ('block', 'wood', 4.0, 3.0, 67.07, 1.99, 0.002), ('block', 'wood', 6.0, 2.0, 45.05, 1.47, 0.006), ('block', 'wood', 8.0, 2.0, 89.24, 7.49, -0.002), ('target', 66.61, 10.37), ('target', 89.24, 1.38)],
     "shots": 3},
    # seed 1010; 5 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 42.76, 15.45, 0.013), ('block', 'stone', 2.0, 6.0, 91.94, 6.48, 0.008), ('block', 'stone', 4.0, 3.0, 91.91, 10.98, 0.008), ('block', 'stone', 6.0, 2.0, 42.71, 19.45, 0.013), ('block', 'stone', 6.0, 2.0, 57.17, 1.49, 0.003), ('block', 'wood', 2.0, 6.0, 42.82, 9.47, 0.007), ('block', 'wood', 2.0, 6.0, 42.86, 3.48, 0.003), ('block', 'wood', 2.0, 6.0, 57.08, 14.46, 0.007), ('block', 'wood', 2.0, 6.0, 57.12, 8.46, 0.006), ('block', 'wood', 2.0, 6.0, 73.08, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 79.08, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 57.16, 3.96, 0.006), ('block', 'wood', 4.0, 3.0, 91.97, 1.99, 0.005), ('block', 'wood', 8.0, 2.0, 76.08, 7.49, -0.002), ('target', 39.41, 21.25), ('target', 76.08, 1.38)],
     "shots": 3},
    # seed 1011; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 86.05, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 92.05, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 58.07, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 76.5, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 43.54, 6.48, 0.003), ('block', 'wood', 4.0, 3.0, 43.55, 1.98, 0.0), ('block', 'wood', 8.0, 2.0, 89.05, 7.49, -0.002), ('target', 80.5, 1.38), ('target', 89.05, 1.38)],
     "shots": 3},
    # seed 1012; 6 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 89.11, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 95.11, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 46.3, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 67.87, 11.47, 0.009), ('block', 'wood', 2.0, 6.0, 67.92, 3.48, 0.003), ('block', 'wood', 6.0, 2.0, 67.83, 15.47, 0.008), ('block', 'wood', 6.0, 2.0, 67.9, 7.47, 0.01), ('block', 'wood', 8.0, 2.0, 92.11, 7.49, -0.002), ('target', 50.3, 1.38), ('target', 65.69, 17.35), ('target', 92.11, 1.38)],
     "shots": 4},
    # seed 1013; 4 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 40.94, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 46.94, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 86.61, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 92.61, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 67.5, 4.5, -0.0), ('block', 'wood', 8.0, 2.0, 43.94, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 89.61, 7.49, -0.002), ('target', 71.5, 1.38), ('target', 89.61, 1.38)],
     "shots": 3},
    # seed 1014; 4 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 65.32, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 71.32, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 45.15, 3.5, -0.0), ('block', 'wood', 2.0, 6.0, 84.52, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 90.52, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 45.16, 7.99, -0.002), ('block', 'wood', 8.0, 2.0, 68.32, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 87.52, 7.49, -0.002), ('target', 45.7, 10.37), ('target', 68.32, 1.38), ('target', 87.52, 1.38)],
     "shots": 4},
    # seed 1015; 17 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 91.48, 17.45, 0.006), ('block', 'stone', 2.0, 6.0, 91.51, 11.45, 0.006), ('block', 'stone', 2.0, 8.0, 57.86, 4.5, -0.0), ('block', 'stone', 6.0, 2.0, 45.11, 1.48, 0.005), ('block', 'wood', 2.0, 6.0, 45.04, 11.48, 0.008), ('block', 'wood', 2.0, 6.0, 45.09, 5.48, 0.007), ('block', 'wood', 2.0, 6.0, 74.56, 3.5, 0.001), ('block', 'wood', 2.0, 6.0, 91.56, 5.46, 0.008), ('block', 'wood', 4.0, 3.0, 74.56, 7.99, 0.001), ('block', 'wood', 6.0, 2.0, 45.0, 15.47, 0.008), ('block', 'wood', 6.0, 2.0, 91.59, 1.47, 0.007), ('target', 42.89, 17.36), ('target', 61.86, 1.38), ('target', 86.71, 1.01)],
     "shots": 4},
    # seed 1016; 100 expansions, 4-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 40.08, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 46.08, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 56.03, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 62.03, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 74.05, 6.48, 0.009), ('block', 'stone', 2.0, 6.0, 85.41, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 91.41, 3.5, -0.001), ('block', 'stone', 4.0, 3.0, 74.01, 10.97, 0.009), ('block', 'wood', 2.0, 6.0, 73.96, 15.47, 0.01), ('block', 'wood', 4.0, 3.0, 74.09, 1.98, 0.005), ('block', 'wood', 8.0, 2.0, 43.08, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 59.03, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 88.41, 7.49, -0.002), ('target', 43.08, 1.38), ('target', 59.03, 1.38), ('target', 88.41, 1.38)],
     "shots": 4},
    # seed 1017; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 66.21, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 92.11, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 43.13, 3.48, 0.003), ('block', 'wood', 4.0, 3.0, 43.12, 7.97, 0.004), ('block', 'wood', 6.0, 2.0, 43.11, 10.46, 0.0), ('target', 43.0, 12.34), ('target', 70.21, 1.38), ('target', 96.11, 1.38)],
     "shots": 4},
    # seed 1018; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 40.21, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 46.21, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 64.41, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 70.41, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 92.45, 9.48, 0.002), ('block', 'wood', 2.0, 6.0, 92.46, 3.5, -0.0), ('block', 'wood', 8.0, 2.0, 43.21, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 67.41, 7.49, -0.002), ('target', 43.21, 1.38), ('target', 67.41, 1.38)],
     "shots": 3},
    # seed 1019; 2 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 59.94, 12.47, 0.001), ('block', 'stone', 4.0, 3.0, 44.45, 4.97, -0.003), ('block', 'stone', 4.0, 3.0, 59.96, 1.99, 0.002), ('block', 'stone', 4.0, 3.0, 89.67, 10.96, 0.004), ('block', 'stone', 4.0, 3.0, 89.69, 1.99, 0.002), ('block', 'wood', 2.0, 6.0, 44.46, 9.47, -0.003), ('block', 'wood', 2.0, 6.0, 69.61, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 75.61, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 44.45, 1.99, 0.0), ('block', 'wood', 4.0, 3.0, 59.95, 4.98, 0.002), ('block', 'wood', 4.0, 3.0, 59.95, 7.97, 0.003), ('block', 'wood', 4.0, 3.0, 89.68, 4.99, 0.003), ('block', 'wood', 4.0, 3.0, 89.68, 7.97, 0.004), ('block', 'wood', 8.0, 2.0, 72.61, 7.49, -0.002), ('target', 45.11, 13.35), ('target', 72.61, 1.38), ('target', 88.52, 13.35)],
     "shots": 4},
    # seed 1020; 2 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 44.77, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 87.49, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 93.49, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 90.49, 7.49, -0.002), ('target', 48.77, 1.38), ('target', 90.49, 1.38)],
     "shots": 3},
    # seed 1021; 19 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 66.84, 5.48, 0.005), ('block', 'stone', 2.0, 6.0, 84.29, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 90.29, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 44.73, 4.5, -0.0), ('block', 'wood', 6.0, 2.0, 66.86, 1.48, 0.006), ('block', 'wood', 8.0, 2.0, 87.29, 7.49, -0.002), ('target', 48.73, 1.38), ('target', 62.69, 2.03), ('target', 87.29, 1.38)],
     "shots": 4},
    # seed 1022; 2 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 4.0, 3.0, 91.81, 16.96, 0.006), ('block', 'wood', 2.0, 6.0, 44.58, 3.5, -0.0), ('block', 'wood', 2.0, 6.0, 50.37, 3.49, -0.008), ('block', 'wood', 2.0, 6.0, 91.85, 9.47, 0.004), ('block', 'wood', 2.0, 6.0, 91.87, 3.48, 0.002), ('block', 'wood', 4.0, 3.0, 91.83, 13.96, 0.005), ('block', 'wood', 8.0, 2.0, 47.53, 7.5, -0.001), ('target', 47.58, 1.38), ('target', 52.16, 1.38), ('target', 90.45, 19.35)],
     "shots": 4},
    # seed 1023; 21 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 85.65, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 91.65, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 66.7, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 44.84, 7.97, 0.006), ('block', 'wood', 2.0, 6.0, 44.86, 3.48, 0.003), ('block', 'wood', 4.0, 3.0, 44.83, 10.96, 0.001), ('block', 'wood', 8.0, 2.0, 88.65, 7.49, -0.002), ('target', 44.31, 13.34), ('target', 70.7, 1.38), ('target', 88.65, 1.38)],
     "shots": 4},
    # seed 1024; 5 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 86.67, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 92.67, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 44.29, 3.49, 0.009), ('block', 'wood', 2.0, 6.0, 50.1, 3.5, -0.0), ('block', 'wood', 8.0, 2.0, 47.14, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 89.67, 7.49, -0.002), ('target', 42.19, 1.38), ('target', 47.1, 1.38), ('target', 89.67, 1.38)],
     "shots": 4},
    # seed 1025; 5 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 4.0, 3.0, 90.81, 2.0, 0.002), ('block', 'wood', 2.0, 6.0, 39.27, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 45.27, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 90.81, 4.99, 0.001), ('block', 'wood', 6.0, 2.0, 90.8, 7.49, 0.001), ('block', 'wood', 8.0, 2.0, 42.27, 7.49, -0.002), ('target', 42.27, 1.38), ('target', 90.49, 9.37)],
     "shots": 3},
    # seed 1026; 2 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 90.46, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 40.6, 3.5, -0.0), ('block', 'wood', 2.0, 6.0, 46.39, 3.49, -0.008), ('block', 'wood', 8.0, 2.0, 43.56, 7.5, -0.001), ('target', 43.6, 1.38), ('target', 48.18, 1.38), ('target', 94.46, 1.38)],
     "shots": 4},
    # seed 1027; 4 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 43.16, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 49.16, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 61.43, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 76.12, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 92.36, 7.94, 0.005), ('block', 'wood', 6.0, 2.0, 92.39, 5.45, 0.005), ('block', 'wood', 6.0, 2.0, 92.41, 3.46, 0.003), ('block', 'wood', 6.0, 2.0, 92.42, 1.49, 0.001), ('block', 'wood', 8.0, 2.0, 46.16, 7.49, -0.002), ('target', 80.12, 1.38), ('target', 90.75, 10.32)],
     "shots": 3},
    # seed 1028; 2 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 67.3, 4.5, -0.0), ('block', 'stone', 6.0, 2.0, 92.09, 3.48, 0.004), ('block', 'wood', 2.0, 6.0, 44.2, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 50.2, 3.5, -0.001), ('block', 'wood', 6.0, 2.0, 92.1, 1.49, 0.003), ('block', 'wood', 8.0, 2.0, 47.2, 7.49, -0.002), ('target', 47.2, 1.38), ('target', 71.3, 1.38), ('target', 90.76, 5.36)],
     "shots": 4},
    # seed 1029; 16 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 55.64, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 61.64, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 76.0, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 88.95, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 46.2, 3.98, 0.007), ('block', 'stone', 6.0, 2.0, 46.21, 1.49, 0.003), ('block', 'wood', 2.0, 6.0, 46.14, 11.47, 0.009), ('block', 'wood', 4.0, 3.0, 46.18, 6.97, 0.008), ('block', 'wood', 8.0, 2.0, 58.64, 7.49, -0.002), ('target', 58.64, 1.38), ('target', 80.0, 1.38), ('target', 92.95, 1.38)],
     "shots": 4},
    # seed 1030; 23 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 42.85, 5.47, 0.007), ('block', 'stone', 2.0, 6.0, 74.73, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 80.73, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 86.08, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 92.08, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 61.73, 4.5, -0.0), ('block', 'wood', 4.0, 3.0, 42.82, 9.97, 0.007), ('block', 'wood', 6.0, 2.0, 42.88, 1.48, 0.006), ('block', 'wood', 8.0, 2.0, 77.73, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 89.08, 7.49, -0.002), ('target', 40.8, 12.34), ('target', 65.73, 1.38), ('target', 89.08, 1.38)],
     "shots": 4},
    # seed 1031; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 44.81, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 50.81, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 54.28, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 60.28, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 74.79, 4.5, -0.0), ('block', 'stone', 6.0, 2.0, 89.64, 13.47, 0.01), ('block', 'wood', 2.0, 6.0, 89.68, 9.47, 0.01), ('block', 'wood', 2.0, 6.0, 89.72, 3.48, 0.002), ('block', 'wood', 8.0, 2.0, 47.81, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 57.28, 7.49, -0.002), ('target', 57.28, 1.38), ('target', 86.95, 15.33)],
     "shots": 3},
    # seed 1032; 24 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 54.24, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 60.24, 3.5, -0.0), ('block', 'stone', 2.0, 8.0, 89.75, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 42.4, 11.49, 0.008), ('block', 'wood', 2.0, 6.0, 42.43, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 73.93, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 79.93, 3.5, -0.001), ('block', 'wood', 6.0, 2.0, 42.43, 7.49, 0.004), ('block', 'wood', 8.0, 2.0, 57.24, 7.5, -0.0), ('block', 'wood', 8.0, 2.0, 76.93, 7.49, -0.002), ('target', 57.24, 1.38), ('target', 76.93, 1.38), ('target', 93.75, 1.38)],
     "shots": 4},
    # seed 1033; 34 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 57.83, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 63.83, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 46.49, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 69.49, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 75.49, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 86.77, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 92.77, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 60.83, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 72.49, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 89.77, 7.49, -0.002), ('target', 60.83, 1.38), ('target', 72.49, 1.38), ('target', 89.77, 1.38)],
     "shots": 4},
    # seed 1034; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 45.9, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 86.76, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 92.76, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 89.76, 7.49, -0.002), ('target', 49.9, 1.38), ('target', 89.76, 1.38)],
     "shots": 3},
    # seed 1035; 4 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 72.04, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 58.17, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 64.17, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 85.82, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 91.82, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 42.84, 1.99, 0.001), ('block', 'wood', 6.0, 2.0, 42.83, 4.49, -0.0), ('block', 'wood', 8.0, 2.0, 61.17, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 88.82, 7.49, -0.002), ('target', 42.94, 6.37), ('target', 76.04, 1.38), ('target', 88.82, 1.38)],
     "shots": 4},
    # seed 1036; 4 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 73.97, 9.45, -0.003), ('block', 'stone', 4.0, 3.0, 57.15, 2.0, 0.002), ('block', 'stone', 4.0, 3.0, 87.56, 1.99, 0.0), ('block', 'stone', 6.0, 2.0, 73.98, 13.44, -0.004), ('block', 'wood', 2.0, 6.0, 39.81, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 45.81, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 57.15, 4.99, 0.001), ('block', 'wood', 4.0, 3.0, 73.94, 1.99, 0.002), ('block', 'wood', 4.0, 3.0, 73.95, 4.95, -0.003), ('block', 'wood', 6.0, 2.0, 57.14, 7.49, 0.001), ('block', 'wood', 6.0, 2.0, 87.56, 4.48, 0.0), ('block', 'wood', 8.0, 2.0, 42.81, 7.49, -0.002), ('target', 42.81, 1.38), ('target', 56.83, 9.37), ('target', 74.35, 15.33)],
     "shots": 4},
    # seed 1037; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 46.68, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 86.02, 1.43, 1.578), ('target', 50.68, 1.38), ('target', 92.91, 1.3), ('target', 94.69, 1.38)],
     "shots": 4},
    # seed 1038; 4 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 44.46, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 62.99, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 88.02, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 73.46, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 79.46, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 76.46, 7.49, -0.002), ('target', 76.46, 1.38), ('target', 92.02, 1.38)],
     "shots": 3},
    # seed 1039; 4 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 45.17, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 67.28, 4.5, -0.0), ('block', 'stone', 6.0, 2.0, 89.36, 1.48, -0.0), ('block', 'wood', 2.0, 6.0, 89.36, 5.48, 0.001), ('target', 49.17, 1.38), ('target', 71.28, 1.38), ('target', 89.17, 9.36)],
     "shots": 4},
    # seed 1040; 22 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 88.27, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 94.27, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 70.11, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 46.24, 4.98, -0.001), ('block', 'stone', 4.0, 3.0, 46.25, 1.99, 0.003), ('block', 'stone', 4.0, 3.0, 46.25, 13.97, -0.002), ('block', 'wood', 2.0, 6.0, 46.25, 9.47, -0.001), ('block', 'wood', 8.0, 2.0, 91.27, 7.49, -0.002), ('target', 46.62, 16.36), ('target', 74.11, 1.38), ('target', 91.27, 1.38)],
     "shots": 4},
    # seed 1041; 4 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 43.66, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 58.3, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 72.13, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 78.13, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 89.39, 8.96, 0.002), ('block', 'wood', 4.0, 3.0, 89.41, 1.99, 0.001), ('block', 'wood', 6.0, 2.0, 89.4, 6.47, 0.001), ('block', 'wood', 6.0, 2.0, 89.41, 4.48, 0.0), ('block', 'wood', 8.0, 2.0, 75.13, 7.49, -0.002), ('target', 75.13, 1.38), ('target', 88.82, 11.35)],
     "shots": 3},
    # seed 1042; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 41.13, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 47.13, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 86.94, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 92.94, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 44.13, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 89.94, 7.49, -0.002), ('target', 44.13, 1.38), ('target', 89.94, 1.38)],
     "shots": 3},
    # seed 1043; 67 expansions, 4-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 55.39, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 61.39, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 76.59, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 43.93, 3.48, 0.002), ('block', 'wood', 2.0, 6.0, 86.56, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 92.56, 3.5, -0.001), ('block', 'wood', 6.0, 2.0, 43.88, 11.46, 0.009), ('block', 'wood', 6.0, 2.0, 43.9, 9.47, 0.008), ('block', 'wood', 6.0, 2.0, 43.92, 7.47, 0.006), ('block', 'wood', 8.0, 2.0, 58.39, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 89.56, 7.49, -0.002), ('target', 41.25, 13.32), ('target', 58.39, 1.38), ('target', 89.56, 1.38)],
     "shots": 4},
    # seed 1044; 8 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 42.75, 11.47, 0.009), ('block', 'stone', 2.0, 6.0, 56.73, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 62.73, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 73.95, 9.46, -0.003), ('block', 'wood', 2.0, 6.0, 42.81, 3.48, 0.003), ('block', 'wood', 4.0, 3.0, 73.94, 1.99, 0.001), ('block', 'wood', 4.0, 3.0, 73.94, 4.97, -0.001), ('block', 'wood', 4.0, 3.0, 89.31, 1.99, 0.001), ('block', 'wood', 6.0, 2.0, 42.79, 7.47, 0.01), ('block', 'wood', 6.0, 2.0, 89.3, 4.49, -0.0), ('block', 'wood', 8.0, 2.0, 59.73, 7.5, -0.0), ('target', 59.73, 1.38), ('target', 89.41, 6.37)],
     "shots": 3},
    # seed 1045; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 56.31, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 62.31, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 75.83, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 87.0, 4.5, -0.0), ('block', 'wood', 4.0, 3.0, 45.58, 4.99, -0.001), ('block', 'wood', 4.0, 3.0, 45.59, 1.99, 0.001), ('block', 'wood', 8.0, 2.0, 59.31, 7.49, -0.002), ('target', 59.31, 1.38), ('target', 79.83, 1.38)],
     "shots": 3},
    # seed 1046; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 41.11, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 47.11, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 67.54, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 90.52, 4.5, -0.0), ('block', 'wood', 8.0, 2.0, 44.11, 7.49, -0.002), ('target', 44.11, 1.38), ('target', 71.54, 1.38)],
     "shots": 3},
    # seed 1047; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 89.42, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 44.25, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 89.4, 15.49, 0.005), ('block', 'wood', 2.0, 6.0, 89.42, 9.49, 0.002), ('target', 48.25, 1.38), ('target', 87.21, 17.88), ('target', 93.42, 1.38)],
     "shots": 4},
    # seed 1048; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 43.03, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 49.03, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 84.61, 3.5, -0.0), ('block', 'wood', 2.0, 6.0, 90.4, 3.49, -0.008), ('block', 'wood', 8.0, 2.0, 46.03, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 87.57, 7.5, -0.001), ('target', 46.03, 1.38), ('target', 87.61, 1.38), ('target', 92.19, 1.38)],
     "shots": 4},
    # seed 1049; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 69.2, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 92.71, 4.5, -0.0), ('block', 'wood', 4.0, 3.0, 44.98, 5.95, -0.006), ('block', 'wood', 6.0, 2.0, 44.98, 3.47, 0.001), ('block', 'wood', 6.0, 2.0, 44.99, 1.49, 0.003), ('target', 46.85, 8.33), ('target', 73.2, 1.38), ('target', 96.71, 1.38)],
     "shots": 4},
    # seed 1050; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 69.4, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 75.4, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 44.87, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 62.35, 7.97, 0.002), ('block', 'stone', 6.0, 2.0, 91.54, 3.48, 0.004), ('block', 'wood', 4.0, 3.0, 62.35, 4.98, 0.0), ('block', 'wood', 4.0, 3.0, 62.36, 1.99, 0.001), ('block', 'wood', 6.0, 2.0, 91.54, 5.48, 0.005), ('block', 'wood', 6.0, 2.0, 91.55, 1.49, 0.003), ('block', 'wood', 8.0, 2.0, 72.4, 7.49, -0.002), ('target', 48.87, 1.38), ('target', 61.74, 10.35), ('target', 72.4, 1.38)],
     "shots": 4},
    # seed 1051; 2 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 91.12, 6.48, 0.002), ('block', 'wood', 2.0, 6.0, 43.56, 11.48, 0.01), ('block', 'wood', 2.0, 6.0, 43.59, 3.5, -0.0), ('block', 'wood', 4.0, 3.0, 91.12, 1.99, 0.0), ('block', 'wood', 6.0, 2.0, 43.59, 7.49, 0.005), ('block', 'wood', 6.0, 2.0, 91.11, 10.48, 0.004), ('target', 34.8, 1.36), ('target', 90.17, 12.36)],
     "shots": 3},
    # seed 1052; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 43.88, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 86.73, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 92.73, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 89.73, 7.49, -0.002), ('target', 47.88, 1.38), ('target', 89.73, 1.38)],
     "shots": 3},
    # seed 1053; 11 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 88.61, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 94.61, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 47.09, 4.5, -0.0), ('block', 'wood', 8.0, 2.0, 91.61, 7.49, -0.002), ('target', 51.09, 1.38), ('target', 91.61, 1.38)],
     "shots": 3},
    # seed 1054; 2 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 44.41, 5.48, 0.005), ('block', 'stone', 2.0, 6.0, 87.6, 6.48, 0.002), ('block', 'wood', 2.0, 6.0, 62.15, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 68.15, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 87.6, 1.99, 0.0), ('block', 'wood', 6.0, 2.0, 44.43, 1.48, 0.006), ('block', 'wood', 6.0, 2.0, 87.59, 10.48, 0.004), ('block', 'wood', 8.0, 2.0, 65.15, 7.49, -0.002), ('target', 40.26, 2.03), ('target', 65.15, 1.38), ('target', 86.65, 12.36)],
     "shots": 4},
    # seed 1055; 6 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 44.59, 9.48, 0.001), ('block', 'stone', 2.0, 6.0, 59.93, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 65.93, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 86.19, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 92.19, 3.5, -0.001), ('block', 'stone', 4.0, 3.0, 44.6, 1.99, 0.002), ('block', 'stone', 6.0, 2.0, 76.66, 8.45, 0.003), ('block', 'stone', 6.0, 2.0, 76.67, 4.47, 0.0), ('block', 'wood', 4.0, 3.0, 44.59, 4.98, 0.0), ('block', 'wood', 4.0, 3.0, 76.68, 1.99, 0.001), ('block', 'wood', 6.0, 2.0, 76.67, 6.46, 0.002), ('block', 'wood', 8.0, 2.0, 62.93, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 89.19, 7.49, -0.002), ('target', 75.79, 10.34), ('target', 89.19, 1.38)],
     "shots": 3},
    # seed 1056; 6 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 42.4, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 89.59, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 61.55, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 67.55, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 64.55, 7.49, -0.002), ('target', 46.4, 1.38), ('target', 64.55, 1.38), ('target', 93.59, 1.38)],
     "shots": 4},
    # seed 1057; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 44.06, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 69.72, 1.99, 0.0), ('block', 'wood', 2.0, 6.0, 88.93, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 94.93, 3.5, -0.001), ('block', 'wood', 6.0, 2.0, 69.72, 4.48, 0.001), ('block', 'wood', 8.0, 2.0, 91.93, 7.49, -0.002), ('target', 48.06, 1.38), ('target', 69.32, 6.37), ('target', 91.93, 1.38)],
     "shots": 4},
    # seed 1058; 2 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 87.35, 9.48, 0.006), ('block', 'stone', 2.0, 8.0, 69.78, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 43.2, 1.99, 0.004), ('block', 'wood', 2.0, 6.0, 43.07, 18.48, 0.009), ('block', 'wood', 2.0, 6.0, 43.13, 12.48, 0.009), ('block', 'wood', 2.0, 6.0, 43.18, 6.49, 0.006), ('block', 'wood', 2.0, 6.0, 87.38, 3.48, 0.003), ('block', 'wood', 6.0, 2.0, 87.33, 13.47, 0.007), ('target', 38.15, 1.38), ('target', 73.78, 1.38)],
     "shots": 3},
    # seed 1059; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 39.53, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 45.53, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 92.85, 4.5, -0.0), ('block', 'wood', 8.0, 2.0, 42.53, 7.49, -0.002), ('target', 42.53, 1.38), ('target', 96.85, 1.38)],
     "shots": 3},
    # seed 1060; 4 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 47.86, 5.47, 0.008), ('block', 'stone', 2.0, 6.0, 61.21, 9.47, 0.01), ('block', 'stone', 2.0, 6.0, 86.24, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 92.24, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 77.58, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 61.16, 13.97, 0.01), ('block', 'wood', 2.0, 6.0, 61.25, 3.48, 0.003), ('block', 'wood', 6.0, 2.0, 47.89, 1.48, 0.005), ('block', 'wood', 8.0, 2.0, 89.24, 7.49, -0.002), ('target', 56.96, 9.2), ('target', 81.58, 1.38), ('target', 89.24, 1.38)],
     "shots": 4},
    # seed 1061; 4 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 47.8, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 60.94, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 92.64, 10.95, 0.004), ('block', 'wood', 2.0, 6.0, 73.64, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 79.64, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 92.65, 7.96, 0.003), ('block', 'wood', 4.0, 3.0, 92.66, 4.98, 0.002), ('block', 'wood', 4.0, 3.0, 92.67, 1.99, 0.003), ('block', 'wood', 8.0, 2.0, 76.64, 7.49, -0.002), ('target', 64.94, 1.38), ('target', 76.64, 1.38)],
     "shots": 3},
    # seed 1062; 7 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 86.38, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 92.38, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 44.03, 13.45, 0.007), ('block', 'wood', 2.0, 6.0, 44.08, 5.47, 0.009), ('block', 'wood', 6.0, 2.0, 44.06, 9.46, 0.004), ('block', 'wood', 6.0, 2.0, 44.11, 1.48, 0.005), ('block', 'wood', 8.0, 2.0, 89.38, 7.49, -0.002), ('target', 37.3, 1.11), ('target', 89.38, 1.38)],
     "shots": 3},
    # seed 1063; 4 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 55.52, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 61.52, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 90.32, 5.48, 0.004), ('block', 'stone', 4.0, 3.0, 46.64, 6.95, 0.003), ('block', 'stone', 6.0, 2.0, 77.4, 1.49, 0.003), ('block', 'stone', 6.0, 2.0, 90.34, 1.49, 0.004), ('block', 'wood', 2.0, 6.0, 77.29, 14.45, 0.01), ('block', 'wood', 2.0, 6.0, 77.35, 8.46, 0.009), ('block', 'wood', 4.0, 3.0, 46.67, 3.96, 0.006), ('block', 'wood', 4.0, 3.0, 77.38, 3.96, 0.008), ('block', 'wood', 6.0, 2.0, 46.68, 1.49, 0.003), ('block', 'wood', 8.0, 2.0, 58.52, 7.5, -0.0), ('target', 45.7, 9.33), ('target', 58.52, 1.38)],
     "shots": 3},
    # seed 1064; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 88.8, 3.49, 0.009), ('block', 'stone', 2.0, 6.0, 94.7, 3.5, -0.001), ('block', 'stone', 6.0, 2.0, 47.6, 1.48, -0.0), ('block', 'wood', 4.0, 3.0, 47.6, 3.98, 0.0), ('block', 'wood', 8.0, 2.0, 91.71, 7.49, 0.002), ('target', 47.45, 6.36), ('target', 86.93, 1.38), ('target', 91.7, 1.38)],
     "shots": 4},
    # seed 1065; 2 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 6.0, 2.0, 92.65, 1.48, -0.0), ('block', 'wood', 2.0, 6.0, 43.92, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 49.92, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 92.65, 3.98, 0.0), ('block', 'wood', 8.0, 2.0, 46.92, 7.49, -0.002), ('target', 46.92, 1.38), ('target', 92.5, 6.36)],
     "shots": 3},
    # seed 1066; 56 expansions, 4-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 85.73, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 91.73, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 64.81, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 42.41, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 48.41, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 45.41, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 88.73, 7.49, -0.002), ('target', 45.41, 1.38), ('target', 68.81, 1.38), ('target', 88.73, 1.38)],
     "shots": 4},
    # seed 1067; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 6.0, 2.0, 47.59, 1.48, -0.0), ('block', 'wood', 2.0, 6.0, 84.78, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 90.78, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 47.6, 3.96, 0.002), ('block', 'wood', 6.0, 2.0, 47.58, 6.45, 0.006), ('block', 'wood', 8.0, 2.0, 87.78, 7.49, -0.002), ('target', 45.74, 8.33), ('target', 51.59, 1.38), ('target', 87.78, 1.38)],
     "shots": 4},
    # seed 1068; 4 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 40.24, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 46.24, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 84.46, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 90.46, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 68.86, 3.99, 0.004), ('block', 'wood', 6.0, 2.0, 68.86, 1.49, 0.004), ('block', 'wood', 8.0, 2.0, 43.24, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 87.46, 7.49, -0.002), ('target', 43.24, 1.38), ('target', 67.6, 6.37), ('target', 87.46, 1.38)],
     "shots": 4},
    # seed 1069; 2 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 57.74, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 88.65, 1.99, 0.0), ('block', 'stone', 6.0, 2.0, 77.65, 7.49, 0.002), ('block', 'wood', 2.0, 6.0, 41.17, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 47.17, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 77.65, 3.5, -0.001), ('block', 'wood', 6.0, 2.0, 88.65, 4.48, 0.001), ('block', 'wood', 8.0, 2.0, 44.17, 7.49, -0.002), ('target', 44.17, 1.38), ('target', 88.25, 6.37)],
     "shots": 3},
    # seed 1070; 2 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 89.43, 5.48, 0.007), ('block', 'stone', 4.0, 3.0, 65.69, 1.99, 0.0), ('block', 'stone', 6.0, 2.0, 89.46, 1.48, 0.005), ('block', 'wood', 2.0, 6.0, 41.88, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 47.88, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 89.39, 11.48, 0.008), ('block', 'wood', 6.0, 2.0, 65.69, 4.48, 0.001), ('block', 'wood', 8.0, 2.0, 44.88, 7.49, -0.002), ('target', 44.88, 1.38), ('target', 65.29, 6.37)],
     "shots": 3},
    # seed 1071; 4 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 46.83, 3.48, 0.001), ('block', 'stone', 2.0, 6.0, 87.68, 3.49, 0.009), ('block', 'stone', 2.0, 6.0, 93.58, 3.5, -0.001), ('block', 'stone', 6.0, 2.0, 46.73, 16.46, 0.01), ('block', 'wood', 2.0, 6.0, 46.8, 9.47, 0.008), ('block', 'wood', 4.0, 3.0, 46.76, 13.97, 0.008), ('block', 'wood', 8.0, 2.0, 90.59, 7.49, 0.002), ('target', 44.04, 18.33), ('target', 85.81, 1.38), ('target', 90.58, 1.38)],
     "shots": 4},
    # seed 1072; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 42.13, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 90.7, 4.5, -0.0), ('block', 'stone', 6.0, 2.0, 66.07, 9.46, 0.011), ('block', 'wood', 2.0, 6.0, 66.11, 3.48, 0.002), ('block', 'wood', 6.0, 2.0, 66.1, 7.47, 0.01), ('target', 46.13, 1.38), ('target', 62.85, 11.3), ('target', 94.7, 1.38)],
     "shots": 4},
    # seed 1073; 2 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 57.99, 7.46, -0.001), ('block', 'stone', 2.0, 6.0, 89.42, 3.48, -0.0), ('block', 'stone', 4.0, 3.0, 89.41, 12.97, 0.001), ('block', 'stone', 6.0, 2.0, 57.99, 3.46, -0.0), ('block', 'stone', 6.0, 2.0, 58.0, 1.49, 0.003), ('block', 'wood', 2.0, 6.0, 75.33, 5.48, 0.004), ('block', 'wood', 4.0, 3.0, 58.0, 11.96, -0.002), ('block', 'wood', 4.0, 3.0, 89.42, 7.98, -0.001), ('block', 'wood', 6.0, 2.0, 47.41, 1.49, 0.003), ('block', 'wood', 6.0, 2.0, 47.41, 3.49, 0.004), ('block', 'wood', 6.0, 2.0, 75.35, 1.48, 0.005), ('block', 'wood', 6.0, 2.0, 89.42, 10.47, 0.0), ('target', 69.73, 1.38), ('target', 89.13, 15.36)],
     "shots": 3},
    # seed 1074; 2 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 43.82, 14.45, 0.011), ('block', 'stone', 2.0, 6.0, 43.9, 3.48, 0.001), ('block', 'stone', 2.0, 8.0, 77.18, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 59.77, 6.95, 0.003), ('block', 'stone', 4.0, 3.0, 88.83, 9.98, 0.006), ('block', 'stone', 6.0, 2.0, 43.87, 10.46, 0.011), ('block', 'stone', 6.0, 2.0, 88.88, 1.48, 0.005), ('block', 'wood', 2.0, 6.0, 88.86, 5.48, 0.006), ('block', 'wood', 4.0, 3.0, 43.89, 7.98, 0.004), ('block', 'wood', 4.0, 3.0, 59.8, 3.96, 0.006), ('block', 'wood', 6.0, 2.0, 59.81, 1.49, 0.003), ('target', 37.19, 1.19), ('target', 58.83, 9.33), ('target', 81.18, 1.38)],
     "shots": 4},
    # seed 1075; 8 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 42.95, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 48.95, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 84.34, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 90.34, 3.5, -0.001), ('block', 'stone', 4.0, 3.0, 72.23, 6.97, -0.0), ('block', 'wood', 2.0, 6.0, 58.02, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 64.02, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 72.24, 1.99, 0.001), ('block', 'wood', 6.0, 2.0, 72.23, 4.48, -0.001), ('block', 'wood', 8.0, 2.0, 45.95, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 61.02, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 87.34, 7.49, -0.002), ('target', 72.31, 9.36), ('target', 87.34, 1.38)],
     "shots": 3},
    # seed 1076; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 6.0, 2.0, 68.28, 1.48, 0.005), ('block', 'wood', 2.0, 6.0, 39.19, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 45.19, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 68.19, 14.48, 0.006), ('block', 'wood', 2.0, 6.0, 68.25, 5.48, 0.006), ('block', 'wood', 2.0, 6.0, 87.17, 3.5, 0.0), ('block', 'wood', 4.0, 3.0, 68.22, 9.98, 0.007), ('block', 'wood', 6.0, 2.0, 87.16, 7.48, 0.002), ('block', 'wood', 8.0, 2.0, 42.19, 7.49, -0.002), ('target', 63.84, 1.22), ('target', 86.62, 9.36)],
     "shots": 3},
    # seed 1077; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 40.31, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 46.31, 3.5, -0.001), ('block', 'stone', 2.0, 6.0, 87.59, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 93.59, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 58.32, 4.5, -0.0), ('block', 'stone', 6.0, 2.0, 75.52, 6.47, 0.0), ('block', 'wood', 4.0, 3.0, 75.53, 1.99, 0.001), ('block', 'wood', 6.0, 2.0, 75.52, 4.48, -0.001), ('block', 'wood', 8.0, 2.0, 43.31, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 90.59, 7.49, -0.002), ('target', 43.31, 1.38), ('target', 90.59, 1.38)],
     "shots": 3},
    # seed 1078; 2 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 4.0, 3.0, 88.52, 3.97, 0.011), ('block', 'stone', 6.0, 2.0, 88.53, 1.49, 0.004), ('block', 'wood', 2.0, 6.0, 41.67, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 47.67, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 88.47, 8.46, 0.011), ('block', 'wood', 8.0, 2.0, 44.67, 7.49, -0.002), ('target', 44.67, 1.38), ('target', 69.62, 1.37)],
     "shots": 3},
    # seed 1079; 10 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 42.3, 3.48, 0.0), ('block', 'stone', 2.0, 6.0, 88.34, 5.46, 0.008), ('block', 'stone', 2.0, 8.0, 59.78, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 42.25, 12.96, 0.01), ('block', 'stone', 4.0, 3.0, 88.27, 11.95, 0.013), ('block', 'wood', 2.0, 6.0, 72.41, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 78.41, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 42.28, 9.97, 0.007), ('block', 'wood', 6.0, 2.0, 42.3, 7.48, 0.004), ('block', 'wood', 6.0, 2.0, 88.3, 9.46, 0.013), ('block', 'wood', 6.0, 2.0, 88.37, 1.47, 0.007), ('block', 'wood', 8.0, 2.0, 75.41, 7.49, -0.002), ('target', 63.78, 1.38), ('target', 75.41, 1.38), ('target', 81.11, 0.95)],
     "shots": 4},
    # seed 1080; 28 expansions, 4-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 84.12, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 90.12, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 47.91, 9.48, 0.002), ('block', 'wood', 2.0, 6.0, 47.93, 3.5, 0.0), ('block', 'wood', 8.0, 2.0, 87.12, 7.49, -0.002), ('target', 47.37, 13.36), ('target', 51.93, 1.38), ('target', 87.12, 1.38)],
     "shots": 4},
    # seed 1081; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 42.77, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 48.77, 3.5, -0.001), ('block', 'stone', 6.0, 2.0, 68.13, 1.48, -0.0), ('block', 'wood', 2.0, 6.0, 84.65, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 90.65, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 68.13, 3.98, 0.0), ('block', 'wood', 8.0, 2.0, 45.77, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 87.65, 7.49, -0.002), ('target', 45.77, 1.38), ('target', 67.98, 6.36), ('target', 87.65, 1.38)],
     "shots": 4},
    # seed 1082; 28 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 43.56, 3.5, 0.001), ('block', 'stone', 2.0, 8.0, 92.61, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 43.56, 7.99, 0.001), ('block', 'wood', 2.0, 6.0, 56.08, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 62.08, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 72.88, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 78.88, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 59.08, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 75.88, 7.49, -0.002), ('target', 59.08, 1.38), ('target', 75.88, 1.38), ('target', 96.61, 1.38)],
     "shots": 4},
    # seed 1083; 2 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 88.01, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 42.06, 3.49, 0.009), ('block', 'wood', 2.0, 6.0, 47.87, 3.5, -0.0), ('block', 'wood', 8.0, 2.0, 44.91, 7.49, -0.002), ('target', 39.96, 1.38), ('target', 44.87, 1.38), ('target', 92.01, 1.38)],
     "shots": 4},
    # seed 1084; 2 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 87.51, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 41.81, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 47.81, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 44.81, 7.49, -0.002), ('target', 44.81, 1.38), ('target', 91.51, 1.38), ('target', 95.51, 1.38)],
     "shots": 4},
    # seed 1085; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 58.79, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 88.07, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 94.07, 3.5, -0.001), ('block', 'stone', 4.0, 3.0, 45.09, 1.99, 0.001), ('block', 'stone', 6.0, 2.0, 45.08, 4.49, -0.0), ('block', 'wood', 2.0, 6.0, 58.79, 11.48, -0.002), ('block', 'wood', 2.0, 6.0, 72.24, 5.48, 0.004), ('block', 'wood', 6.0, 2.0, 58.78, 7.48, 0.0), ('block', 'wood', 6.0, 2.0, 72.26, 1.49, 0.004), ('block', 'wood', 8.0, 2.0, 91.07, 7.49, -0.002), ('target', 59.23, 15.36), ('target', 91.07, 1.38)],
     "shots": 3},
    # seed 1086; 14 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 74.35, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 80.35, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 43.94, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 90.36, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 59.64, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 65.64, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 62.64, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 77.35, 7.49, -0.002), ('target', 47.94, 1.38), ('target', 77.35, 1.38), ('target', 94.36, 1.38)],
     "shots": 4},
    # seed 1087; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 8.0, 43.28, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 73.18, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 90.57, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 57.96, 11.47, 0.0), ('block', 'wood', 4.0, 3.0, 57.97, 4.98, 0.001), ('block', 'wood', 4.0, 3.0, 57.98, 1.99, 0.002), ('block', 'wood', 6.0, 2.0, 57.97, 7.47, 0.002), ('target', 47.28, 1.38), ('target', 57.69, 15.36), ('target', 94.57, 1.38)],
     "shots": 4},
    # seed 1088; 2 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 4.0, 3.0, 43.98, 6.95, 0.003), ('block', 'stone', 6.0, 2.0, 89.2, 4.47, -0.003), ('block', 'wood', 2.0, 6.0, 59.6, 3.5, -0.0), ('block', 'wood', 2.0, 6.0, 65.6, 3.48, 0.0), ('block', 'wood', 2.0, 6.0, 75.84, 6.48, 0.0), ('block', 'wood', 4.0, 3.0, 44.01, 3.96, 0.006), ('block', 'wood', 4.0, 3.0, 75.84, 1.98, -0.0), ('block', 'wood', 4.0, 3.0, 89.2, 1.99, 0.0), ('block', 'wood', 6.0, 2.0, 44.02, 1.49, 0.003), ('block', 'wood', 6.0, 2.0, 89.19, 6.47, -0.002), ('block', 'wood', 8.0, 2.0, 62.6, 7.49, -0.002), ('target', 43.04, 9.33), ('target', 75.76, 10.36), ('target', 89.76, 8.35)],
     "shots": 4},
    # seed 1089; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 87.31, 3.5, -0.0), ('block', 'stone', 2.0, 8.0, 45.11, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 67.48, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 73.48, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 87.3, 9.48, 0.001), ('block', 'wood', 8.0, 2.0, 70.48, 7.49, -0.002), ('target', 49.11, 1.38), ('target', 70.48, 1.38), ('target', 86.92, 13.36)],
     "shots": 4},
    # seed 1090; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 61.69, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 67.69, 3.5, -0.001), ('block', 'stone', 6.0, 2.0, 44.35, 1.48, -0.0), ('block', 'wood', 2.0, 6.0, 88.09, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 94.09, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 44.35, 3.98, 0.0), ('block', 'wood', 8.0, 2.0, 64.69, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 91.09, 7.49, -0.002), ('target', 44.2, 6.36), ('target', 91.09, 1.38)],
     "shots": 3},
    # seed 1091; 4 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 6.0, 2.0, 46.09, 1.49, 0.003), ('block', 'stone', 6.0, 2.0, 46.09, 3.49, 0.004), ('block', 'stone', 6.0, 2.0, 69.42, 1.49, 0.003), ('block', 'stone', 6.0, 2.0, 69.42, 3.47, -0.0), ('block', 'wood', 2.0, 6.0, 69.42, 7.47, -0.001), ('block', 'wood', 2.0, 6.0, 86.76, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 92.76, 3.5, -0.001), ('block', 'wood', 6.0, 2.0, 46.08, 5.48, 0.004), ('block', 'wood', 8.0, 2.0, 89.76, 7.49, -0.002), ('target', 69.45, 11.35), ('target', 89.76, 1.38)],
     "shots": 3},
    # seed 1092; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 43.78, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 49.78, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 75.82, 4.5, -0.0), ('block', 'stone', 6.0, 2.0, 60.89, 3.48, 0.004), ('block', 'wood', 2.0, 6.0, 88.86, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 94.86, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 60.89, 5.98, 0.005), ('block', 'wood', 6.0, 2.0, 60.9, 1.49, 0.003), ('block', 'wood', 8.0, 2.0, 46.78, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 91.86, 7.49, -0.002), ('target', 46.78, 1.38), ('target', 79.82, 1.38), ('target', 91.86, 1.38)],
     "shots": 4},
    # seed 1093; 2 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 67.14, 11.48, 0.007), ('block', 'stone', 2.0, 6.0, 67.18, 5.48, 0.007), ('block', 'stone', 6.0, 2.0, 67.21, 1.48, 0.005), ('block', 'stone', 6.0, 2.0, 91.11, 7.47, 0.005), ('block', 'wood', 2.0, 6.0, 43.29, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 49.29, 3.5, -0.001), ('block', 'wood', 2.0, 6.0, 91.12, 3.48, 0.003), ('block', 'wood', 6.0, 2.0, 91.1, 9.46, 0.001), ('block', 'wood', 8.0, 2.0, 46.29, 7.49, -0.002), ('target', 46.29, 1.38), ('target', 61.05, 1.19), ('target', 90.65, 11.34)],
     "shots": 4},
    # seed 1094; 4 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 71.09, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 77.09, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 61.15, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 45.95, 5.48, 0.006), ('block', 'wood', 2.0, 6.0, 89.0, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 95.0, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 45.92, 9.97, 0.006), ('block', 'wood', 6.0, 2.0, 45.98, 1.48, 0.005), ('block', 'wood', 8.0, 2.0, 74.09, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 92.0, 7.49, -0.002), ('target', 44.17, 12.35), ('target', 74.09, 1.38), ('target', 92.0, 1.38)],
     "shots": 4},
    # seed 1095; 22 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 46.83, 6.49, 0.004), ('block', 'stone', 2.0, 6.0, 70.92, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 76.92, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 59.92, 4.5, -0.0), ('block', 'stone', 4.0, 3.0, 46.85, 1.99, 0.004), ('block', 'wood', 2.0, 6.0, 46.77, 18.48, 0.005), ('block', 'wood', 2.0, 6.0, 46.81, 12.48, 0.005), ('block', 'wood', 2.0, 6.0, 91.45, 6.48, 0.0), ('block', 'wood', 4.0, 3.0, 91.45, 1.98, -0.0), ('block', 'wood', 8.0, 2.0, 73.92, 7.49, -0.002), ('target', 63.92, 1.38), ('target', 73.92, 1.38), ('target', 91.37, 10.36)],
     "shots": 4},
    # seed 1096; 22 expansions, 3-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 84.13, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 90.13, 3.5, -0.001), ('block', 'stone', 6.0, 2.0, 59.44, 1.49, 0.004), ('block', 'stone', 6.0, 2.0, 74.71, 13.47, 0.008), ('block', 'wood', 2.0, 6.0, 47.52, 8.46, 0.004), ('block', 'wood', 2.0, 6.0, 59.44, 7.47, -0.002), ('block', 'wood', 2.0, 6.0, 74.75, 9.47, 0.009), ('block', 'wood', 2.0, 6.0, 74.79, 3.48, 0.002), ('block', 'wood', 4.0, 3.0, 47.54, 3.96, 0.007), ('block', 'wood', 6.0, 2.0, 47.55, 1.49, 0.004), ('block', 'wood', 6.0, 2.0, 59.43, 3.47, -0.001), ('block', 'wood', 8.0, 2.0, 87.13, 7.49, -0.002), ('target', 38.87, 1.34), ('target', 59.96, 11.35), ('target', 87.13, 1.38)],
     "shots": 4},
    # seed 1097; 5 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 89.94, 3.5, -0.0), ('block', 'stone', 2.0, 6.0, 95.94, 3.5, -0.001), ('block', 'stone', 2.0, 8.0, 42.13, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 60.7, 4.5, -0.0), ('block', 'wood', 2.0, 6.0, 73.93, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 79.93, 3.5, -0.001), ('block', 'wood', 8.0, 2.0, 76.93, 7.49, -0.002), ('block', 'wood', 8.0, 2.0, 92.94, 7.49, -0.002), ('target', 64.7, 1.38), ('target', 76.93, 1.38)],
     "shots": 3},
    # seed 1098; 3 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 2.0, 6.0, 68.02, 7.46, 0.0), ('block', 'stone', 2.0, 8.0, 44.34, 4.5, -0.0), ('block', 'stone', 2.0, 8.0, 92.45, 4.5, -0.0), ('block', 'stone', 6.0, 2.0, 68.03, 1.49, 0.003), ('block', 'wood', 6.0, 2.0, 68.02, 3.46, -0.001), ('target', 67.95, 11.35), ('target', 96.45, 1.38)],
     "shots": 3},
    # seed 1099; 4 expansions, 2-shot plan
    {"bodies": [('block', 'stone', 6.0, 2.0, 43.43, 9.46, 0.011), ('block', 'wood', 2.0, 6.0, 43.47, 3.48, 0.002), ('block', 'wood', 2.0, 6.0, 68.82, 6.48, 0.0), ('block', 'wood', 2.0, 6.0, 84.98, 3.5, 0.0), ('block', 'wood', 2.0, 6.0, 90.98, 3.5, -0.001), ('block', 'wood', 4.0, 3.0, 68.82, 1.98, -0.0), ('block', 'wood', 4.0, 3.0, 68.82, 10.98, -0.0), ('block', 'wood', 6.0, 2.0, 43.46, 7.47, 0.01), ('block', 'wood', 8.0, 2.0, 87.98, 7.49, -0.002), ('target', 40.21, 11.3), ('target', 87.98, 1.38)],
     "shots": 3},
)
