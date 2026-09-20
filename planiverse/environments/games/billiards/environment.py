"""Billiards on pooltool: a cue ball and a few object balls on a pool table, a handful of
shots, and every object ball to pot without sinking the cue ball.

The physics is pooltool's (Kiefl, *Pooltool: A Python package for realistic billiards
simulation*, JOSS 2024, https://doi.org/10.21105/joss.07301; Apache-2.0): an event-based
simulation of sliding, rolling and spinning balls, ball-ball and ball-cushion collisions and
pockets. A shot here is a choice of object ball to aim at, a cut angle and a speed, and what
it does is a cascade of collisions the engine resolves. No add or delete list carries a
carom, which is why the environment is here.

## Determinism

pooltool's simulation is deterministic for the same balls in the same places, and every
expansion rebuilds the table from the state's record, so expanding a state twice gives the
same children. pooltool's own rack placement is randomised, so the environment places the
balls itself, from the seed.

## Instances and generation

`generate_instance(seed, ...)` scatters the balls over the table and keeps the draw only if a
best-first search over shots, fewest balls left first, pots every object ball within the
shots given and needs at least two of them; the plan is left in `witness`. The bundled
tables are such draws, embedded as plain data with the seed each came from. The method is
generate-and-test (search-based procedural content generation: Togelius et al. 2011,
https://doi.org/10.1109/TCIAIG.2011.2148116).
"""
from planiverse.environments.base import Environment
from planiverse.environments.generation import bounded_search, draw_until, rng

#: The shot alphabet: cut angles in degrees either side of a full hit, and cue speeds in
#: metres per second.
CUTS = (-30, 0, 30)
SPEEDS = (2.0, 3.5)
#: Ball radius and the table's playing surface, in metres, as pooltool's default table.
RADIUS = 0.028575
TABLE_W, TABLE_L = 0.9906, 1.9812
#: Positions are rounded to a millimetre for the state record.
MM = 3


def _pooltool():
    import pooltool
    return pooltool


class BilliardsAction:
    """`shot(ball, cut, speed)`: aim the cue ball at `ball`, `cut` degrees off full, at `speed`."""

    def __init__(self, ball, cut, speed):
        if cut not in CUTS or speed not in SPEEDS:
            raise ValueError(f"unknown shot: cut {cut}, speed {speed}")
        self.ball, self.cut, self.speed = str(ball), cut, speed
        self.name = f"shot({self.ball},{cut},{speed:g})"

    @classmethod
    def parse(cls, text):
        ball, cut, speed = str(text).strip()[len("shot("):-1].split(",")
        return cls(ball, int(cut), float(speed))

    def cost(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, BilliardsAction) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    __repr__ = __str__


class BilliardsState:
    """Where every ball still on the table lies, and the shots left.

    `balls` maps a ball id to its `(x, y)` in metres, rounded to the millimetre; a potted
    ball is absent. The cue ball is `"cue"`; its absence means it was sunk, which is the end.
    """

    def __init__(self, balls, shots_left, depth=0):
        self.balls = tuple(sorted((ball, (round(x, MM), round(y, MM))) for ball, (x, y) in dict(balls).items()))
        self.shots_left = shots_left
        self.depth = depth
        self.object_balls = tuple(ball for ball, _ in self.balls if ball != "cue")
        self.scratched = all(ball != "cue" for ball, _ in self.balls)
        literals = [f"at({ball}, {int(x * 20)}, {int(y * 20)})" for ball, (x, y) in self.balls]
        literals += [f"shots_left({shots_left})", f"balls_left({len(self.object_balls)})"]
        if self.scratched:
            literals.append("scratched()")
        self.literals = frozenset(literals)

    def position(self, ball):
        return dict(self.balls).get(ball)

    def __eq__(self, other):
        return (isinstance(other, BilliardsState) and self.balls == other.balls
                and self.shots_left == other.shots_left)

    def __hash__(self):
        return hash((self.balls, self.shots_left))

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        lines = [f"shots left: {self.shots_left}, balls to pot: {len(self.object_balls)}"
                 + (", cue ball sunk" if self.scratched else "")]
        lines += [f"  {ball} at ({x:.3f}, {y:.3f})" for ball, (x, y) in self.balls]
        return "\n".join(lines)

    def __repr__(self):
        return f"<BilliardsState(balls={len(self.object_balls)}, shots={self.shots_left})>"


# ------------------------------------------------------------------------------ the physics

def build_system(balls):
    """A pooltool system with `balls` (id to `(x, y)`) at rest on the default table."""
    pt = _pooltool()
    table = pt.Table.default()
    made = {ball: pt.Ball.create(ball, xy=(x, y)) for ball, (x, y) in dict(balls).items()}
    return pt.System(table=table, balls=made, cue=pt.Cue(cue_ball_id="cue"))


def strike(balls, action):
    """Play one shot on the table `balls` describes and return the balls left, at rest."""
    pt = _pooltool()
    balls = dict(balls)
    if "cue" not in balls or action.ball not in balls:
        return None
    system = build_system(balls)
    phi = pt.aim.at_ball(system, action.ball, cut=action.cut)
    system.cue.set_state(V0=action.speed, phi=phi)
    try:
        pt.simulate(system, inplace=True)
    except (AssertionError, ArithmeticError, ValueError):
        # pooltool's collision models assert on the odd geometry (a ball resting against a
        # pocket jaw with no closing speed); a shot the physics cannot resolve is not offered.
        return None
    after = {}
    for ball_id, ball in system.balls.items():
        if ball.state.s == pt.constants.pocketed:
            continue
        x, y = ball.state.rvw[0][0], ball.state.rvw[0][1]
        after[ball_id] = (float(x), float(y))
    return after


# ------------------------------------------------------------------------------- the tables

def draw_table(random_, balls=3, shots=None):
    """A random table as the instance dict `set_instance` takes: the cue ball and `balls`
    object balls scattered over the cloth, none touching."""
    placed = {}
    ids = ["cue"] + [str(k) for k in range(1, balls + 1)]
    for ball in ids:
        for _ in range(200):
            x = random_.uniform(3 * RADIUS, TABLE_W - 3 * RADIUS)
            y = random_.uniform(3 * RADIUS, TABLE_L - 3 * RADIUS)
            if all((x - px) ** 2 + (y - py) ** 2 > (4 * RADIUS) ** 2 for px, py in placed.values()):
                placed[ball] = (round(x, MM), round(y, MM))
                break
    return {"balls": {ball: list(xy) for ball, xy in placed.items()}, "shots": shots or balls}


class BilliardsEnv(Environment):
    """Pot every object ball within the shots given, without sinking the cue ball."""

    def __init__(self):
        super().__init__("billiards")
        self.instance = None
        self.index = None
        self.state = None
        self.state_history = []
        self.witness = None
        self.witness_expansions = None

    # ------------------------------------------------------------------ instances

    def set_index(self, index):
        if not 0 <= index < len(TABLES):
            raise IndexError(f"Invalid index: {index}. There are {len(TABLES)} tables, so the "
                             f"index must be 0-{len(TABLES) - 1}.")
        self.set_instance(TABLES[index])
        self.index = index

    def set_instance(self, instance):
        """Select a table: `{"balls": {"cue": [x, y], "1": [x, y], ...}, "shots": n}`."""
        if "balls" not in instance or "shots" not in instance or "cue" not in instance["balls"]:
            raise ValueError("a table is a dict with `balls` (including `cue`) and `shots`")
        self.instance = {"balls": {str(ball): (float(x), float(y)) for ball, (x, y) in instance["balls"].items()},
                         "shots": int(instance["shots"])}
        self.index = None
        self.witness = self.witness_expansions = None

    def generate_instance(self, seed=None, balls=None, shots=None, min_plan_length=3,
                          search_limit=80, attempts=30):
        """Draw a table, select it, and return it as the dict `set_instance` takes.

        `balls` (three or four object balls, drawn when unset) and `shots` (as many as the
        balls when unset, so no shot may be wasted) are `draw_table`'s. A draw is kept only if
        a breadth-first search over shots finds a plan within `search_limit` expansions and
        that plan, a shortest one, is at least `min_plan_length` shots long: the search has
        then shown that no shorter plan exists, so a table two shots clear is thrown back. The
        plan is left in `witness` and what the search spent in `witness_expansions`.
        """
        random_, _ = rng(seed)
        found = {}

        def draw(attempt):
            count = balls or random_.randint(3, 4)
            return draw_table(random_, balls=count, shots=shots or count)

        def accept(instance):
            self.set_instance(instance)
            outcome = bounded_search(self, search_limit)
            if outcome.plan is None or len(outcome.plan) < min_plan_length:
                return False
            found["plan"], found["expansions"] = outcome.plan, outcome.expansions
            return True

        instance = draw_until(draw, accept, attempts, "billiards table")
        self.set_instance(instance)
        self.witness, self.witness_expansions = found["plan"], found["expansions"]
        return instance

    # ------------------------------------------------------------------- contract

    def reset(self):
        if self.instance is None:
            self.set_index(0)
        self.state = BilliardsState(self.instance["balls"], self.instance["shots"])
        self.state_history = [self.state]
        return self.state, {"table": self.index, "balls": len(self.state.object_balls),
                            "shots": self.instance["shots"], "generated": self.index is None}

    def is_goal(self, state):
        return not state.object_balls and not state.scratched

    def is_terminal(self, state):
        return state.scratched or (bool(state.object_balls) and state.shots_left == 0)

    def get_actions(self, state=None):
        state = state or self.state
        return [BilliardsAction(ball, cut, speed) for ball in state.object_balls
                for cut in CUTS for speed in SPEEDS]

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
        if not isinstance(action, BilliardsAction):
            action = BilliardsAction.parse(action)
        after = strike(dict(state.balls), action)
        if after is None:
            return state
        return BilliardsState(after, state.shots_left - 1, state.depth + 1)

    def simulate(self, plan):
        state, _ = self.reset()
        trace = [state]
        for action in plan:
            trace.append(self.__advance__(trace[-1], action))
        return trace

    def step(self, action):
        if self.state is None:
            raise ValueError("call reset() first")
        before = len(self.state.object_balls)
        self.state = self.__advance__(self.state, action)
        self.state_history.append(self.state)
        return self.state, before - len(self.state.object_balls)

    def render(self):
        lines = [f"step {k}:\n{state}" for k, state in enumerate(self.state_history)]
        print("\n".join(lines))
        return lines


#: The bundled tables: `generate_instance(seed, balls=n)` for the seed and ball count beside
#: each, embedded as the plain data `set_instance` takes, with the plan each was accepted on
#: in `tests/data/billiards_solutions.json`.
TABLES = (
    # seed 9000, balls=3; 26 expansions, 3-shot plan
    {"balls": {"cue": [0.437, 1.829], "1": [0.139, 1.701], "2": [0.343, 0.822], "3": [0.454, 0.731]}, "shots": 3},
    # seed 9001, balls=4; 27 expansions, 3-shot plan
    {"balls": {"cue": [0.157, 1.693], "1": [0.6, 1.38], "2": [0.682, 1.191], "3": [0.892, 0.283], "4": [0.863, 0.948]}, "shots": 4},
    # seed 9002, balls=3; 37 expansions, 3-shot plan
    {"balls": {"cue": [0.311, 0.713], "1": [0.559, 1.712], "2": [0.286, 1.87], "3": [0.115, 0.145]}, "shots": 3},
    # seed 9003, balls=4; 53 expansions, 3-shot plan
    {"balls": {"cue": [0.552, 1.754], "1": [0.111, 1.034], "2": [0.559, 1.117], "3": [0.858, 1.088], "4": [0.334, 1.131]}, "shots": 4},
    # seed 9004, balls=3; 56 expansions, 3-shot plan
    {"balls": {"cue": [0.897, 0.937], "1": [0.775, 0.627], "2": [0.766, 0.118], "3": [0.133, 1.568]}, "shots": 3},
    # seed 9005, balls=4; 78 expansions, 3-shot plan
    {"balls": {"cue": [0.373, 1.799], "1": [0.309, 1.049], "2": [0.288, 0.648], "3": [0.643, 0.61], "4": [0.832, 0.692]}, "shots": 4},
    # seed 9006, balls=3; 20 expansions, 3-shot plan
    {"balls": {"cue": [0.649, 0.75], "1": [0.268, 1.162], "2": [0.84, 0.155], "3": [0.089, 1.298]}, "shots": 3},
    # seed 9007, balls=4; 39 expansions, 3-shot plan
    {"balls": {"cue": [0.202, 1.732], "1": [0.849, 1.25], "2": [0.376, 1.043], "3": [0.894, 0.566], "4": [0.883, 1.008]}, "shots": 4},
    # seed 9008, balls=3; 40 expansions, 3-shot plan
    {"balls": {"cue": [0.581, 0.854], "1": [0.364, 1.635], "2": [0.694, 1.302], "3": [0.299, 0.114]}, "shots": 3},
    # seed 9009, balls=4; 46 expansions, 3-shot plan
    {"balls": {"cue": [0.545, 1.176], "1": [0.616, 1.661], "2": [0.852, 0.296], "3": [0.894, 0.123], "4": [0.355, 0.798]}, "shots": 4},
    # seed 9010, balls=3; 26 expansions, 3-shot plan
    {"balls": {"cue": [0.178, 0.533], "1": [0.699, 0.856], "2": [0.787, 0.394], "3": [0.902, 1.261]}, "shots": 3},
    # seed 9011, balls=4; 73 expansions, 3-shot plan
    {"balls": {"cue": [0.791, 0.899], "1": [0.526, 0.117], "2": [0.276, 0.775], "3": [0.12, 0.137], "4": [0.752, 0.101]}, "shots": 4},
    # seed 9012, balls=3; 30 expansions, 3-shot plan
    {"balls": {"cue": [0.496, 1.296], "1": [0.543, 1.833], "2": [0.847, 0.296], "3": [0.809, 0.659]}, "shots": 3},
    # seed 9013, balls=4; 73 expansions, 3-shot plan
    {"balls": {"cue": [0.356, 0.807], "1": [0.126, 1.316], "2": [0.262, 1.752], "3": [0.519, 0.448], "4": [0.444, 1.001]}, "shots": 4},
    # seed 9014, balls=3; 16 expansions, 3-shot plan
    {"balls": {"cue": [0.348, 0.972], "1": [0.736, 0.419], "2": [0.586, 0.563], "3": [0.79, 0.981]}, "shots": 3},
    # seed 9015, balls=4; 45 expansions, 3-shot plan
    {"balls": {"cue": [0.475, 1.614], "1": [0.242, 1.161], "2": [0.206, 0.446], "3": [0.54, 0.816], "4": [0.688, 0.523]}, "shots": 4},
    # seed 9016, balls=3; 33 expansions, 3-shot plan
    {"balls": {"cue": [0.736, 0.13], "1": [0.492, 0.586], "2": [0.403, 1.076], "3": [0.142, 0.223]}, "shots": 3},
    # seed 9017, balls=4; 46 expansions, 3-shot plan
    {"balls": {"cue": [0.672, 0.174], "1": [0.883, 1.781], "2": [0.578, 1.592], "3": [0.441, 0.95], "4": [0.67, 1.411]}, "shots": 4},
    # seed 9018, balls=3; 18 expansions, 3-shot plan
    {"balls": {"cue": [0.119, 0.718], "1": [0.625, 0.92], "2": [0.709, 0.535], "3": [0.508, 0.142]}, "shots": 3},
    # seed 9019, balls=4; 31 expansions, 3-shot plan
    {"balls": {"cue": [0.784, 1.417], "1": [0.728, 0.228], "2": [0.236, 1.863], "3": [0.797, 0.568], "4": [0.82, 0.864]}, "shots": 4},
    # seed 9020, balls=3; 46 expansions, 3-shot plan
    {"balls": {"cue": [0.83, 0.755], "1": [0.645, 1.296], "2": [0.305, 1.03], "3": [0.799, 1.207]}, "shots": 3},
    # seed 9021, balls=4; 24 expansions, 3-shot plan
    {"balls": {"cue": [0.823, 0.434], "1": [0.169, 1.274], "2": [0.853, 0.652], "3": [0.361, 0.601], "4": [0.512, 1.216]}, "shots": 4},
    # seed 9022, balls=3; 23 expansions, 3-shot plan
    {"balls": {"cue": [0.608, 0.401], "1": [0.427, 0.224], "2": [0.878, 1.451], "3": [0.62, 0.874]}, "shots": 3},
    # seed 9023, balls=4; 44 expansions, 3-shot plan
    {"balls": {"cue": [0.478, 1.585], "1": [0.405, 0.192], "2": [0.171, 1.457], "3": [0.841, 1.17], "4": [0.194, 1.616]}, "shots": 4},
    # seed 9024, balls=3; 23 expansions, 3-shot plan
    {"balls": {"cue": [0.102, 0.314], "1": [0.516, 0.789], "2": [0.527, 0.287], "3": [0.196, 1.233]}, "shots": 3},
    # seed 9025, balls=4; 32 expansions, 3-shot plan
    {"balls": {"cue": [0.253, 0.114], "1": [0.271, 1.863], "2": [0.477, 0.919], "3": [0.237, 1.676], "4": [0.404, 0.699]}, "shots": 4},
    # seed 9026, balls=3; 46 expansions, 3-shot plan
    {"balls": {"cue": [0.343, 1.109], "1": [0.893, 0.788], "2": [0.117, 0.94], "3": [0.527, 0.319]}, "shots": 3},
    # seed 9027, balls=4; 44 expansions, 3-shot plan
    {"balls": {"cue": [0.772, 1.829], "1": [0.373, 0.839], "2": [0.506, 1.136], "3": [0.426, 0.126], "4": [0.532, 0.952]}, "shots": 4},
    # seed 9028, balls=3; 47 expansions, 3-shot plan
    {"balls": {"cue": [0.408, 0.753], "1": [0.505, 1.714], "2": [0.502, 1.36], "3": [0.356, 1.894]}, "shots": 3},
    # seed 9029, balls=4; 69 expansions, 3-shot plan
    {"balls": {"cue": [0.703, 1.579], "1": [0.537, 1.185], "2": [0.288, 1.104], "3": [0.57, 0.668], "4": [0.639, 0.184]}, "shots": 4},
    # seed 9030, balls=3; 36 expansions, 3-shot plan
    {"balls": {"cue": [0.331, 0.779], "1": [0.378, 0.975], "2": [0.127, 0.167], "3": [0.661, 0.404]}, "shots": 3},
    # seed 9031, balls=4; 49 expansions, 3-shot plan
    {"balls": {"cue": [0.393, 0.548], "1": [0.27, 0.443], "2": [0.457, 0.836], "3": [0.28, 0.912], "4": [0.807, 0.109]}, "shots": 4},
    # seed 9032, balls=3; 38 expansions, 3-shot plan
    {"balls": {"cue": [0.843, 1.506], "1": [0.884, 0.204], "2": [0.45, 0.192], "3": [0.67, 1.307]}, "shots": 3},
    # seed 9033, balls=4; 80 expansions, 3-shot plan
    {"balls": {"cue": [0.38, 0.413], "1": [0.29, 0.707], "2": [0.167, 0.413], "3": [0.63, 1.198], "4": [0.615, 0.9]}, "shots": 4},
    # seed 9034, balls=3; 37 expansions, 3-shot plan
    {"balls": {"cue": [0.2, 1.121], "1": [0.713, 0.276], "2": [0.568, 0.711], "3": [0.458, 1.182]}, "shots": 3},
    # seed 9035, balls=4; 36 expansions, 3-shot plan
    {"balls": {"cue": [0.292, 1.012], "1": [0.42, 0.263], "2": [0.201, 1.291], "3": [0.656, 0.739], "4": [0.49, 0.123]}, "shots": 4},
    # seed 9036, balls=3; 31 expansions, 3-shot plan
    {"balls": {"cue": [0.886, 1.244], "1": [0.362, 0.503], "2": [0.726, 1.202], "3": [0.302, 1.638]}, "shots": 3},
    # seed 9037, balls=4; 22 expansions, 3-shot plan
    {"balls": {"cue": [0.253, 1.875], "1": [0.595, 0.721], "2": [0.162, 1.354], "3": [0.651, 1.075], "4": [0.16, 1.759]}, "shots": 4},
    # seed 9038, balls=3; 22 expansions, 3-shot plan
    {"balls": {"cue": [0.105, 0.393], "1": [0.545, 0.241], "2": [0.834, 0.224], "3": [0.087, 0.766]}, "shots": 3},
    # seed 9039, balls=4; 48 expansions, 3-shot plan
    {"balls": {"cue": [0.13, 0.848], "1": [0.811, 0.547], "2": [0.622, 0.477], "3": [0.713, 0.87], "4": [0.163, 0.709]}, "shots": 4},
    # seed 9040, balls=3; 49 expansions, 3-shot plan
    {"balls": {"cue": [0.668, 0.624], "1": [0.507, 0.988], "2": [0.128, 1.3], "3": [0.544, 0.587]}, "shots": 3},
    # seed 9041, balls=4; 67 expansions, 3-shot plan
    {"balls": {"cue": [0.648, 1.889], "1": [0.293, 0.361], "2": [0.355, 1.536], "3": [0.228, 1.407], "4": [0.811, 0.55]}, "shots": 4},
    # seed 9042, balls=3; 34 expansions, 3-shot plan
    {"balls": {"cue": [0.164, 1.19], "1": [0.435, 1.493], "2": [0.366, 0.66], "3": [0.618, 0.597]}, "shots": 3},
    # seed 9043, balls=4; 37 expansions, 3-shot plan
    {"balls": {"cue": [0.182, 0.872], "1": [0.398, 0.86], "2": [0.202, 1.3], "3": [0.451, 0.419], "4": [0.513, 1.856]}, "shots": 4},
    # seed 9044, balls=3; 41 expansions, 3-shot plan
    {"balls": {"cue": [0.728, 1.6], "1": [0.514, 1.592], "2": [0.543, 1.303], "3": [0.439, 1.728]}, "shots": 3},
    # seed 9045, balls=4; 44 expansions, 3-shot plan
    {"balls": {"cue": [0.445, 1.772], "1": [0.276, 0.323], "2": [0.589, 1.379], "3": [0.781, 1.301], "4": [0.524, 0.471]}, "shots": 4},
    # seed 9046, balls=3; 24 expansions, 3-shot plan
    {"balls": {"cue": [0.527, 0.377], "1": [0.43, 0.965], "2": [0.866, 0.178], "3": [0.704, 0.21]}, "shots": 3},
    # seed 9047, balls=4; 31 expansions, 3-shot plan
    {"balls": {"cue": [0.691, 0.734], "1": [0.855, 1.228], "2": [0.375, 0.45], "3": [0.178, 0.412], "4": [0.48, 0.551]}, "shots": 4},
    # seed 9048, balls=3; 34 expansions, 3-shot plan
    {"balls": {"cue": [0.275, 0.825], "1": [0.674, 0.238], "2": [0.416, 0.892], "3": [0.889, 0.382]}, "shots": 3},
    # seed 9049, balls=4; 78 expansions, 3-shot plan
    {"balls": {"cue": [0.608, 1.586], "1": [0.82, 0.808], "2": [0.179, 1.111], "3": [0.244, 0.904], "4": [0.113, 0.268]}, "shots": 4},
    # seed 9250, balls=3; 27 expansions, 3-shot plan
    {"balls": {"cue": [0.691, 1.728], "1": [0.207, 1.159], "2": [0.89, 0.282], "3": [0.25, 1.768]}, "shots": 3},
    # seed 9251, balls=4; 34 expansions, 3-shot plan
    {"balls": {"cue": [0.683, 1.672], "1": [0.628, 0.286], "2": [0.727, 1.053], "3": [0.441, 1.125], "4": [0.236, 0.73]}, "shots": 4},
    # seed 9252, balls=3; 25 expansions, 3-shot plan
    {"balls": {"cue": [0.789, 0.811], "1": [0.765, 0.165], "2": [0.559, 1.075], "3": [0.126, 1.08]}, "shots": 3},
    # seed 9253, balls=4; 22 expansions, 3-shot plan
    {"balls": {"cue": [0.151, 0.403], "1": [0.346, 1.387], "2": [0.518, 1.149], "3": [0.448, 0.539], "4": [0.198, 1.474]}, "shots": 4},
    # seed 9254, balls=3; 70 expansions, 3-shot plan
    {"balls": {"cue": [0.428, 0.98], "1": [0.268, 0.193], "2": [0.088, 1.491], "3": [0.255, 1.449]}, "shots": 3},
    # seed 9255, balls=4; 32 expansions, 3-shot plan
    {"balls": {"cue": [0.867, 0.089], "1": [0.302, 0.758], "2": [0.182, 1.022], "3": [0.146, 0.19], "4": [0.268, 1.771]}, "shots": 4},
    # seed 9256, balls=3; 72 expansions, 3-shot plan
    {"balls": {"cue": [0.128, 1.111], "1": [0.365, 0.51], "2": [0.091, 0.46], "3": [0.409, 1.884]}, "shots": 3},
    # seed 9257, balls=4; 26 expansions, 3-shot plan
    {"balls": {"cue": [0.305, 0.145], "1": [0.513, 0.605], "2": [0.128, 1.423], "3": [0.615, 1.741], "4": [0.466, 1.726]}, "shots": 4},
    # seed 9258, balls=3; 18 expansions, 3-shot plan
    {"balls": {"cue": [0.322, 0.733], "1": [0.517, 1.066], "2": [0.614, 1.662], "3": [0.659, 1.482]}, "shots": 3},
    # seed 9259, balls=4; 80 expansions, 3-shot plan
    {"balls": {"cue": [0.39, 0.781], "1": [0.721, 0.336], "2": [0.379, 0.97], "3": [0.639, 1.664], "4": [0.155, 0.622]}, "shots": 4},
    # seed 9260, balls=3; 33 expansions, 3-shot plan
    {"balls": {"cue": [0.187, 0.496], "1": [0.426, 1.277], "2": [0.553, 0.953], "3": [0.757, 0.832]}, "shots": 3},
    # seed 9261, balls=4; 57 expansions, 3-shot plan
    {"balls": {"cue": [0.748, 0.604], "1": [0.094, 0.505], "2": [0.143, 1.021], "3": [0.392, 1.66], "4": [0.802, 1.826]}, "shots": 4},
    # seed 9262, balls=3; 42 expansions, 3-shot plan
    {"balls": {"cue": [0.627, 0.297], "1": [0.419, 1.649], "2": [0.579, 1.361], "3": [0.256, 1.395]}, "shots": 3},
    # seed 9263, balls=4; 76 expansions, 3-shot plan
    {"balls": {"cue": [0.843, 0.405], "1": [0.097, 0.911], "2": [0.273, 1.392], "3": [0.22, 0.647], "4": [0.435, 1.07]}, "shots": 4},
    # seed 9264, balls=3; 43 expansions, 3-shot plan
    {"balls": {"cue": [0.892, 1.392], "1": [0.694, 1.846], "2": [0.872, 0.388], "3": [0.263, 1.149]}, "shots": 3},
    # seed 9265, balls=4; 56 expansions, 3-shot plan
    {"balls": {"cue": [0.234, 1.437], "1": [0.834, 0.867], "2": [0.523, 1.16], "3": [0.883, 1.631], "4": [0.661, 1.552]}, "shots": 4},
    # seed 9266, balls=3; 56 expansions, 3-shot plan
    {"balls": {"cue": [0.636, 0.373], "1": [0.26, 0.896], "2": [0.705, 1.237], "3": [0.149, 0.829]}, "shots": 3},
    # seed 9267, balls=4; 26 expansions, 3-shot plan
    {"balls": {"cue": [0.656, 0.969], "1": [0.095, 0.779], "2": [0.817, 1.783], "3": [0.281, 1.312], "4": [0.086, 0.555]}, "shots": 4},
    # seed 9268, balls=3; 50 expansions, 3-shot plan
    {"balls": {"cue": [0.54, 1.507], "1": [0.427, 0.729], "2": [0.145, 0.35], "3": [0.648, 1.258]}, "shots": 3},
    # seed 9269, balls=4; 80 expansions, 3-shot plan
    {"balls": {"cue": [0.512, 1.413], "1": [0.285, 1.44], "2": [0.903, 1.234], "3": [0.185, 0.092], "4": [0.303, 0.63]}, "shots": 4},
    # seed 9270, balls=3; 49 expansions, 3-shot plan
    {"balls": {"cue": [0.683, 0.236], "1": [0.582, 0.143], "2": [0.317, 0.724], "3": [0.751, 1.37]}, "shots": 3},
    # seed 9271, balls=4; 34 expansions, 3-shot plan
    {"balls": {"cue": [0.534, 0.172], "1": [0.663, 0.998], "2": [0.191, 1.699], "3": [0.306, 1.062], "4": [0.404, 1.885]}, "shots": 4},
    # seed 9272, balls=3; 29 expansions, 3-shot plan
    {"balls": {"cue": [0.653, 0.185], "1": [0.39, 1.272], "2": [0.892, 0.711], "3": [0.306, 0.835]}, "shots": 3},
    # seed 9273, balls=4; 70 expansions, 3-shot plan
    {"balls": {"cue": [0.369, 1.573], "1": [0.789, 1.836], "2": [0.763, 1.082], "3": [0.829, 0.933], "4": [0.107, 1.112]}, "shots": 4},
    # seed 9274, balls=3; 62 expansions, 3-shot plan
    {"balls": {"cue": [0.662, 1.538], "1": [0.145, 0.272], "2": [0.678, 1.184], "3": [0.488, 1.504]}, "shots": 3},
    # seed 9500, balls=4; 76 expansions, 3-shot plan
    {"balls": {"cue": [0.222, 1.095], "1": [0.638, 1.026], "2": [0.622, 0.86], "3": [0.128, 0.2], "4": [0.666, 1.726]}, "shots": 4},
    # seed 9501, balls=3; 69 expansions, 3-shot plan
    {"balls": {"cue": [0.806, 0.907], "1": [0.174, 1.076], "2": [0.575, 0.094], "3": [0.257, 0.803]}, "shots": 3},
    # seed 9502, balls=4; 31 expansions, 3-shot plan
    {"balls": {"cue": [0.112, 1.573], "1": [0.724, 1.299], "2": [0.162, 0.603], "3": [0.53, 0.666], "4": [0.167, 0.45]}, "shots": 4},
    # seed 9503, balls=3; 57 expansions, 3-shot plan
    {"balls": {"cue": [0.682, 0.266], "1": [0.373, 1.094], "2": [0.089, 0.127], "3": [0.401, 1.677]}, "shots": 3},
    # seed 9504, balls=4; 26 expansions, 3-shot plan
    {"balls": {"cue": [0.395, 0.604], "1": [0.876, 0.713], "2": [0.728, 1.618], "3": [0.117, 1.538], "4": [0.189, 1.44]}, "shots": 4},
    # seed 9505, balls=3; 54 expansions, 3-shot plan
    {"balls": {"cue": [0.271, 0.737], "1": [0.333, 0.11], "2": [0.731, 1.824], "3": [0.581, 1.613]}, "shots": 3},
    # seed 9506, balls=4; 38 expansions, 3-shot plan
    {"balls": {"cue": [0.181, 0.477], "1": [0.825, 1.577], "2": [0.159, 1.742], "3": [0.156, 1.529], "4": [0.16, 0.972]}, "shots": 4},
    # seed 9507, balls=3; 68 expansions, 3-shot plan
    {"balls": {"cue": [0.135, 1.042], "1": [0.528, 0.336], "2": [0.136, 0.452], "3": [0.52, 0.647]}, "shots": 3},
    # seed 9508, balls=4; 26 expansions, 3-shot plan
    {"balls": {"cue": [0.509, 1.5], "1": [0.85, 1.034], "2": [0.41, 1.012], "3": [0.726, 0.192], "4": [0.299, 0.19]}, "shots": 4},
    # seed 9509, balls=3; 28 expansions, 3-shot plan
    {"balls": {"cue": [0.784, 1.37], "1": [0.177, 0.804], "2": [0.309, 0.298], "3": [0.474, 0.74]}, "shots": 3},
    # seed 9510, balls=4; 57 expansions, 3-shot plan
    {"balls": {"cue": [0.708, 1.65], "1": [0.601, 0.186], "2": [0.638, 0.779], "3": [0.89, 0.167], "4": [0.279, 0.94]}, "shots": 4},
    # seed 9511, balls=3; 56 expansions, 3-shot plan
    {"balls": {"cue": [0.379, 1.844], "1": [0.904, 0.708], "2": [0.657, 0.11], "3": [0.637, 1.426]}, "shots": 3},
    # seed 9512, balls=4; 46 expansions, 3-shot plan
    {"balls": {"cue": [0.151, 1.272], "1": [0.236, 0.8], "2": [0.308, 1.671], "3": [0.228, 1.816], "4": [0.431, 1.861]}, "shots": 4},
    # seed 9513, balls=3; 51 expansions, 3-shot plan
    {"balls": {"cue": [0.802, 1.878], "1": [0.587, 1.166], "2": [0.258, 0.771], "3": [0.328, 1.36]}, "shots": 3},
    # seed 9514, balls=4; 35 expansions, 3-shot plan
    {"balls": {"cue": [0.446, 0.506], "1": [0.585, 0.608], "2": [0.826, 1.541], "3": [0.62, 1.738], "4": [0.842, 1.085]}, "shots": 4},
    # seed 9515, balls=3; 68 expansions, 3-shot plan
    {"balls": {"cue": [0.849, 0.504], "1": [0.206, 0.606], "2": [0.365, 1.498], "3": [0.665, 0.092]}, "shots": 3},
    # seed 9516, balls=4; 33 expansions, 3-shot plan
    {"balls": {"cue": [0.891, 0.605], "1": [0.136, 1.449], "2": [0.887, 1.563], "3": [0.606, 0.524], "4": [0.681, 1.729]}, "shots": 4},
    # seed 9517, balls=3; 36 expansions, 3-shot plan
    {"balls": {"cue": [0.119, 1.651], "1": [0.38, 1.381], "2": [0.633, 0.25], "3": [0.549, 0.691]}, "shots": 3},
    # seed 9518, balls=4; 66 expansions, 3-shot plan
    {"balls": {"cue": [0.144, 0.714], "1": [0.709, 0.607], "2": [0.591, 0.64], "3": [0.355, 0.902], "4": [0.816, 0.765]}, "shots": 4},
    # seed 9519, balls=3; 73 expansions, 3-shot plan
    {"balls": {"cue": [0.885, 0.523], "1": [0.682, 0.343], "2": [0.166, 0.854], "3": [0.175, 0.442]}, "shots": 3},
    # seed 9521, balls=4; 25 expansions, 3-shot plan
    {"balls": {"cue": [0.097, 0.579], "1": [0.705, 1.467], "2": [0.526, 0.917], "3": [0.148, 0.843], "4": [0.428, 1.56]}, "shots": 4},
    # seed 9522, balls=3; 76 expansions, 3-shot plan
    {"balls": {"cue": [0.551, 0.469], "1": [0.485, 1.816], "2": [0.465, 0.838], "3": [0.314, 0.267]}, "shots": 3},
    # seed 9523, balls=4; 76 expansions, 3-shot plan
    {"balls": {"cue": [0.485, 1.227], "1": [0.553, 1.649], "2": [0.37, 1.314], "3": [0.185, 0.925], "4": [0.178, 1.112]}, "shots": 4},
    # seed 9524, balls=3; 16 expansions, 3-shot plan
    {"balls": {"cue": [0.469, 0.613], "1": [0.629, 0.623], "2": [0.309, 1.503], "3": [0.833, 1.569]}, "shots": 3},
    # seed 9525, balls=4; 44 expansions, 3-shot plan
    {"balls": {"cue": [0.79, 0.346], "1": [0.512, 0.13], "2": [0.336, 1.343], "3": [0.535, 1.09], "4": [0.128, 1.216]}, "shots": 4},
)
