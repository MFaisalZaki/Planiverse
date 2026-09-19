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
    # seed 5000; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.386, 0.804], "1": [0.177, 0.811], "2": [0.122, 1.846]}, "shots": 3},
    # seed 5001; 4 expansions, 3-shot plan
    {"balls": {"cue": [0.6, 1.422], "1": [0.549, 0.302], "2": [0.19, 1.123], "3": [0.529, 0.925]}, "shots": 4},
    # seed 5002; 5 expansions, 4-shot plan
    {"balls": {"cue": [0.797, 0.155], "1": [0.71, 1.384], "2": [0.454, 0.572], "3": [0.548, 0.646]}, "shots": 4},
    # seed 5003; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.714, 0.818], "1": [0.524, 1.127], "2": [0.571, 0.88]}, "shots": 3},
    # seed 5004; 4 expansions, 4-shot plan
    {"balls": {"cue": [0.819, 0.693], "1": [0.564, 0.97], "2": [0.137, 1.262], "3": [0.235, 0.241]}, "shots": 4},
    # seed 5005; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.241, 0.476], "1": [0.483, 0.579], "2": [0.158, 0.209]}, "shots": 3},
    # seed 5006; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.905, 1.324], "1": [0.682, 0.235], "2": [0.507, 0.388]}, "shots": 3},
    # seed 5007; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.368, 1.866], "1": [0.824, 1.012], "2": [0.401, 1.018]}, "shots": 3},
    # seed 5008; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.562, 1.399], "1": [0.282, 1.673], "2": [0.095, 0.803]}, "shots": 3},
    # seed 5009; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.67, 1.433], "1": [0.524, 0.444], "2": [0.294, 1.394], "3": [0.899, 0.667]}, "shots": 4},
    # seed 5010; 3 expansions, 2-shot plan
    {"balls": {"cue": [0.093, 1.876], "1": [0.616, 1.663], "2": [0.554, 1.5], "3": [0.62, 0.884]}, "shots": 4},
    # seed 5011; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.624, 0.396], "1": [0.147, 0.971], "2": [0.25, 0.751]}, "shots": 3},
    # seed 5012; 4 expansions, 3-shot plan
    {"balls": {"cue": [0.612, 0.252], "1": [0.539, 1.866], "2": [0.626, 1.171]}, "shots": 3},
    # seed 5013; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.504, 0.888], "1": [0.398, 1.055], "2": [0.145, 1.072], "3": [0.138, 1.373]}, "shots": 4},
    # seed 5014; 3 expansions, 2-shot plan
    {"balls": {"cue": [0.279, 1.372], "1": [0.774, 1.834], "2": [0.25, 1.143]}, "shots": 3},
    # seed 5015; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.472, 0.092], "1": [0.868, 0.11], "2": [0.678, 0.289], "3": [0.765, 1.083]}, "shots": 4},
    # seed 5016; 3 expansions, 2-shot plan
    {"balls": {"cue": [0.519, 0.692], "1": [0.86, 0.778], "2": [0.123, 1.452]}, "shots": 3},
    # seed 5017; 4 expansions, 4-shot plan
    {"balls": {"cue": [0.276, 0.857], "1": [0.773, 0.341], "2": [0.326, 1.465], "3": [0.857, 1.433]}, "shots": 4},
    # seed 5018; 3 expansions, 2-shot plan
    {"balls": {"cue": [0.659, 0.615], "1": [0.12, 1.332], "2": [0.196, 0.348]}, "shots": 3},
    # seed 5019; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.546, 1.799], "1": [0.68, 1.199], "2": [0.718, 0.994]}, "shots": 3},
    # seed 5020; 4 expansions, 3-shot plan
    {"balls": {"cue": [0.568, 1.111], "1": [0.641, 1.767], "2": [0.765, 0.211], "3": [0.728, 1.684]}, "shots": 4},
    # seed 5021; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.759, 1.481], "1": [0.089, 1.089], "2": [0.325, 1.534], "3": [0.565, 1.706]}, "shots": 4},
    # seed 5022; 4 expansions, 2-shot plan
    {"balls": {"cue": [0.886, 0.713], "1": [0.635, 0.126], "2": [0.535, 0.703]}, "shots": 3},
    # seed 5023; 4 expansions, 3-shot plan
    {"balls": {"cue": [0.381, 1.018], "1": [0.422, 1.237], "2": [0.305, 1.795], "3": [0.638, 1.037]}, "shots": 4},
    # seed 5024; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.399, 0.201], "1": [0.886, 1.065], "2": [0.486, 0.633], "3": [0.33, 1.797]}, "shots": 4},
    # seed 5025; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.154, 0.515], "1": [0.88, 1.845], "2": [0.555, 1.55], "3": [0.751, 1.194]}, "shots": 4},
    # seed 5026; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.713, 1.003], "1": [0.366, 1.739], "2": [0.359, 1.362]}, "shots": 3},
    # seed 5027; 3 expansions, 2-shot plan
    {"balls": {"cue": [0.352, 0.196], "1": [0.209, 1.488], "2": [0.672, 1.489]}, "shots": 3},
    # seed 5028; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.391, 0.216], "1": [0.902, 1.855], "2": [0.536, 1.0]}, "shots": 3},
    # seed 5029; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.475, 0.613], "1": [0.61, 1.887], "2": [0.807, 0.402]}, "shots": 3},
    # seed 5030; 4 expansions, 3-shot plan
    {"balls": {"cue": [0.567, 1.692], "1": [0.277, 1.763], "2": [0.186, 1.301], "3": [0.857, 0.12]}, "shots": 4},
    # seed 5031; 4 expansions, 3-shot plan
    {"balls": {"cue": [0.13, 0.9], "1": [0.288, 1.83], "2": [0.642, 0.209], "3": [0.591, 0.72]}, "shots": 4},
    # seed 5032; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.705, 1.342], "1": [0.724, 1.215], "2": [0.569, 1.886]}, "shots": 3},
    # seed 5033; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.664, 1.789], "1": [0.595, 1.142], "2": [0.529, 0.608], "3": [0.9, 1.281]}, "shots": 4},
    # seed 5034; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.705, 0.134], "1": [0.463, 0.6], "2": [0.175, 1.6], "3": [0.221, 0.184]}, "shots": 4},
    # seed 5035; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.329, 0.844], "1": [0.841, 1.545], "2": [0.601, 1.02]}, "shots": 3},
    # seed 5036; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.19, 0.473], "1": [0.862, 1.864], "2": [0.544, 0.614]}, "shots": 3},
    # seed 5037; 3 expansions, 2-shot plan
    {"balls": {"cue": [0.779, 0.471], "1": [0.547, 1.406], "2": [0.571, 0.863]}, "shots": 3},
    # seed 5038; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.445, 1.807], "1": [0.342, 0.537], "2": [0.566, 0.922]}, "shots": 3},
    # seed 5039; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.839, 0.179], "1": [0.386, 0.345], "2": [0.109, 1.021]}, "shots": 3},
    # seed 5040; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.431, 0.273], "1": [0.5, 1.535], "2": [0.564, 0.224]}, "shots": 3},
    # seed 5041; 4 expansions, 3-shot plan
    {"balls": {"cue": [0.787, 1.089], "1": [0.323, 0.59], "2": [0.201, 1.345], "3": [0.726, 1.84]}, "shots": 4},
    # seed 5042; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.706, 1.432], "1": [0.119, 1.719], "2": [0.169, 0.933]}, "shots": 3},
    # seed 5043; 5 expansions, 4-shot plan
    {"balls": {"cue": [0.469, 1.113], "1": [0.252, 1.471], "2": [0.408, 0.135], "3": [0.656, 1.313]}, "shots": 4},
    # seed 5044; 11 expansions, 3-shot plan
    {"balls": {"cue": [0.893, 1.373], "1": [0.42, 0.472], "2": [0.207, 0.211]}, "shots": 3},
    # seed 5045; 6 expansions, 3-shot plan
    {"balls": {"cue": [0.09, 0.339], "1": [0.87, 0.29], "2": [0.451, 1.063]}, "shots": 3},
    # seed 5046; 4 expansions, 3-shot plan
    {"balls": {"cue": [0.714, 0.45], "1": [0.403, 1.502], "2": [0.537, 0.852]}, "shots": 3},
    # seed 5047; 3 expansions, 2-shot plan
    {"balls": {"cue": [0.56, 0.742], "1": [0.808, 0.701], "2": [0.49, 1.371]}, "shots": 3},
    # seed 5048; 4 expansions, 3-shot plan
    {"balls": {"cue": [0.581, 1.582], "1": [0.591, 0.646], "2": [0.465, 1.729], "3": [0.839, 1.287]}, "shots": 4},
    # seed 5049; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.323, 0.491], "1": [0.188, 0.193], "2": [0.389, 0.79]}, "shots": 3},
    # seed 5050; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.169, 1.618], "1": [0.547, 0.452], "2": [0.876, 0.736]}, "shots": 3},
    # seed 5051; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.203, 1.73], "1": [0.839, 0.289], "2": [0.159, 1.33]}, "shots": 3},
    # seed 5052; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.452, 0.979], "1": [0.8, 0.592], "2": [0.339, 0.184], "3": [0.197, 1.622]}, "shots": 4},
    # seed 5053; 7 expansions, 3-shot plan
    {"balls": {"cue": [0.594, 1.643], "1": [0.669, 0.162], "2": [0.216, 1.002]}, "shots": 3},
    # seed 5054; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.468, 1.334], "1": [0.572, 0.189], "2": [0.356, 0.553]}, "shots": 3},
    # seed 5055; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.858, 1.454], "1": [0.591, 0.938], "2": [0.433, 0.165], "3": [0.766, 1.889]}, "shots": 4},
    # seed 5056; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.553, 1.665], "1": [0.582, 1.055], "2": [0.363, 1.064], "3": [0.713, 1.698]}, "shots": 4},
    # seed 5057; 6 expansions, 3-shot plan
    {"balls": {"cue": [0.89, 1.06], "1": [0.639, 1.355], "2": [0.584, 0.877], "3": [0.334, 0.28]}, "shots": 4},
    # seed 5058; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.754, 1.708], "1": [0.278, 1.744], "2": [0.516, 0.386], "3": [0.454, 0.144]}, "shots": 4},
    # seed 5059; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.672, 0.576], "1": [0.468, 0.687], "2": [0.323, 0.817], "3": [0.443, 0.56]}, "shots": 4},
    # seed 5060; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.377, 0.601], "1": [0.596, 1.056], "2": [0.135, 1.731]}, "shots": 3},
    # seed 5061; 4 expansions, 3-shot plan
    {"balls": {"cue": [0.221, 0.143], "1": [0.571, 1.856], "2": [0.623, 0.706], "3": [0.835, 1.819]}, "shots": 4},
    # seed 5062; 4 expansions, 3-shot plan
    {"balls": {"cue": [0.741, 0.972], "1": [0.109, 0.807], "2": [0.747, 0.1], "3": [0.711, 1.317]}, "shots": 4},
    # seed 5063; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.686, 0.748], "1": [0.883, 0.686], "2": [0.852, 1.719]}, "shots": 3},
    # seed 5064; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.188, 0.713], "1": [0.865, 1.336], "2": [0.284, 0.566], "3": [0.627, 0.104]}, "shots": 4},
    # seed 5065; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.118, 0.64], "1": [0.442, 0.573], "2": [0.349, 0.853], "3": [0.561, 0.591]}, "shots": 4},
    # seed 5066; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.522, 1.817], "1": [0.737, 1.442], "2": [0.401, 1.013]}, "shots": 3},
    # seed 5067; 5 expansions, 2-shot plan
    {"balls": {"cue": [0.687, 0.979], "1": [0.362, 1.111], "2": [0.264, 0.987]}, "shots": 3},
    # seed 5068; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.328, 0.541], "1": [0.172, 1.407], "2": [0.785, 1.298], "3": [0.473, 0.098]}, "shots": 4},
    # seed 5069; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.606, 0.563], "1": [0.231, 1.472], "2": [0.848, 0.592]}, "shots": 3},
    # seed 5070; 4 expansions, 4-shot plan
    {"balls": {"cue": [0.318, 0.953], "1": [0.121, 0.479], "2": [0.787, 0.963], "3": [0.617, 1.885]}, "shots": 4},
    # seed 5071; 5 expansions, 4-shot plan
    {"balls": {"cue": [0.545, 1.849], "1": [0.84, 0.667], "2": [0.728, 0.605], "3": [0.753, 1.297]}, "shots": 4},
    # seed 5072; 4 expansions, 3-shot plan
    {"balls": {"cue": [0.587, 1.013], "1": [0.851, 0.872], "2": [0.672, 0.721]}, "shots": 3},
    # seed 5073; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.334, 1.017], "1": [0.091, 0.728], "2": [0.728, 1.847]}, "shots": 3},
    # seed 5074; 5 expansions, 4-shot plan
    {"balls": {"cue": [0.464, 1.768], "1": [0.531, 1.607], "2": [0.608, 0.193], "3": [0.719, 0.516]}, "shots": 4},
    # seed 5075; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.634, 1.435], "1": [0.286, 0.278], "2": [0.84, 0.312]}, "shots": 3},
    # seed 5076; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.413, 0.249], "1": [0.413, 1.414], "2": [0.574, 0.723], "3": [0.307, 1.293]}, "shots": 4},
    # seed 5077; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.516, 0.131], "1": [0.097, 1.155], "2": [0.128, 0.729]}, "shots": 3},
    # seed 5078; 8 expansions, 4-shot plan
    {"balls": {"cue": [0.795, 1.661], "1": [0.603, 0.467], "2": [0.585, 0.248], "3": [0.121, 1.372]}, "shots": 4},
    # seed 5079; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.833, 0.382], "1": [0.183, 0.481], "2": [0.784, 1.113]}, "shots": 3},
    # seed 5080; 4 expansions, 3-shot plan
    {"balls": {"cue": [0.396, 1.127], "1": [0.725, 1.857], "2": [0.758, 0.861]}, "shots": 3},
    # seed 5081; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.657, 1.89], "1": [0.769, 0.946], "2": [0.711, 1.413]}, "shots": 3},
    # seed 5082; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.411, 1.18], "1": [0.721, 1.004], "2": [0.446, 0.827], "3": [0.134, 0.229]}, "shots": 4},
    # seed 5083; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.894, 0.268], "1": [0.225, 0.767], "2": [0.329, 1.13], "3": [0.568, 0.094]}, "shots": 4},
    # seed 5084; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.322, 0.601], "1": [0.687, 0.349], "2": [0.1, 0.64]}, "shots": 3},
    # seed 5085; 4 expansions, 3-shot plan
    {"balls": {"cue": [0.393, 0.971], "1": [0.303, 0.402], "2": [0.398, 0.324], "3": [0.871, 1.619]}, "shots": 4},
    # seed 5086; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.209, 1.641], "1": [0.639, 0.278], "2": [0.454, 1.839], "3": [0.466, 0.936]}, "shots": 4},
    # seed 5087; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.679, 1.756], "1": [0.759, 0.373], "2": [0.281, 0.142], "3": [0.478, 1.539]}, "shots": 4},
    # seed 5088; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.184, 1.638], "1": [0.553, 1.843], "2": [0.75, 0.473], "3": [0.527, 0.476]}, "shots": 4},
    # seed 5089; 3 expansions, 2-shot plan
    {"balls": {"cue": [0.324, 0.902], "1": [0.429, 0.797], "2": [0.24, 0.501]}, "shots": 3},
    # seed 5090; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.414, 0.14], "1": [0.328, 0.291], "2": [0.468, 1.04], "3": [0.788, 1.213]}, "shots": 4},
    # seed 5091; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.112, 0.164], "1": [0.152, 1.295], "2": [0.53, 1.507], "3": [0.581, 0.672]}, "shots": 4},
    # seed 5092; 3 expansions, 2-shot plan
    {"balls": {"cue": [0.567, 0.213], "1": [0.563, 0.982], "2": [0.211, 0.928]}, "shots": 3},
    # seed 5093; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.801, 0.346], "1": [0.626, 1.805], "2": [0.792, 0.972], "3": [0.579, 1.16]}, "shots": 4},
    # seed 5094; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.555, 0.271], "1": [0.094, 0.48], "2": [0.24, 0.562]}, "shots": 3},
    # seed 5095; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.831, 0.931], "1": [0.134, 1.563], "2": [0.597, 0.607]}, "shots": 3},
    # seed 5096; 3 expansions, 3-shot plan
    {"balls": {"cue": [0.437, 0.95], "1": [0.462, 1.393], "2": [0.224, 1.196], "3": [0.684, 1.708]}, "shots": 4},
    # seed 5097; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.346, 1.017], "1": [0.648, 1.216], "2": [0.346, 0.747], "3": [0.903, 1.664]}, "shots": 4},
    # seed 5098; 5 expansions, 3-shot plan
    {"balls": {"cue": [0.274, 0.755], "1": [0.402, 1.646], "2": [0.85, 1.814]}, "shots": 3},
    # seed 5099; 2 expansions, 2-shot plan
    {"balls": {"cue": [0.857, 1.852], "1": [0.481, 1.809], "2": [0.477, 1.317], "3": [0.717, 0.139]}, "shots": 4},
)
