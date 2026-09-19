"""A Lemmings-like: a crowd of walkers is let into a cave one at a time, walks without
being told, and has to be steered to the exit with a few skills handed out at the right
moments, so that enough of them are saved.

Each tick every lemming acts on its own: a walker walks, climbs a step of one, turns at a
wall of two and falls off an edge; a faller dies if it falls too far; a digger digs down
through earth, a basher tunnels forward through it, a builder lays a stair of bricks up and
forward, and a blocker stands and turns the others back. Rock stops every tool. Between
decisions the crowd runs for a fixed number of ticks, so what a skill is worth depends on
where everyone will be by then, on the terrain the earlier tools have already changed, and
on the order the lemmings were released in. That is the whole puzzle, and it is what keeps
the environment out of PDDL: a crowd walking a cave is a simulation, not an action model.

The mechanics are the genre's, at the resolution of a cell rather than a pixel, and the
levels are this environment's own; the public-domain Lix (https://www.lixgame.com/) is the
reference for what the skills do. The environment needs no dependency.

## Instances and generation

`generate_instance(seed, ...)` draws platforms, gaps and walls between an entrance and an
exit, hands out a few skills, and keeps the draw only if letting the crowd walk unaided
saves too few and a best-first search over decisions, guided by the lemmings saved and how
near the rest are to the exit, finds a plan within the budget; the plan is left in
`witness`. The bundled levels are such draws, embedded as plain data with the seed each
came from. The method is generate-and-test (search-based procedural content generation:
Togelius et al. 2011, https://doi.org/10.1109/TCIAIG.2011.2148116).
"""
from planiverse.environments.base import Environment
from planiverse.environments.generation import bounded_search, draw_until, rng

WIDTH, HEIGHT = 32, 16
ROCK, EARTH, AIR, ENTRANCE, EXIT = "#", ".", " ", "E", "X"
SKILLS = ("blocker", "digger", "basher", "builder")
#: Ticks between two lemmings entering, ticks between two decisions, the most ticks a level
#: may take, the fall that kills, and how many bricks a builder lays.
RELEASE, DECISION, MAX_TICKS, FATAL_FALL, BRICKS = 6, 8, 320, 6, 8


class LemmingsAction:
    """`assign(skill, lemming)` or `wait`."""

    def __init__(self, skill=None, lemming=None):
        if skill is not None and skill not in SKILLS:
            raise ValueError(f"unknown skill: {skill!r}")
        self.skill, self.lemming = skill, lemming
        self.name = "wait" if skill is None else f"assign({skill},{lemming})"

    @classmethod
    def parse(cls, text):
        text = str(text).strip()
        if text == "wait":
            return cls()
        skill, lemming = text[len("assign("):-1].split(",")
        return cls(skill, int(lemming))

    def cost(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, LemmingsAction) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    __repr__ = __str__


WAIT = LemmingsAction()


class LemmingsState:
    """The terrain, every lemming out in the cave as `(x, y, direction, state, counter)`,
    the counts of lemmings released, saved and lost, the skills left, and the tick."""

    def __init__(self, rows, lemmings, released, saved, lost, skills, tick, quota=0, depth=0):
        self.rows = tuple(rows)
        self.quota = quota                        # carried for the benchmark's measure
        self.lemmings = tuple(tuple(lemming) for lemming in lemmings)
        self.released, self.saved, self.lost = released, saved, lost
        self.skills = tuple(sorted(dict(skills).items()))
        self.tick = tick
        self.depth = depth
        literals = [f"earth({x}, {y})" for y, row in enumerate(self.rows)
                    for x, cell in enumerate(row) if cell == EARTH]
        literals += [f"lemming({k}, {x}, {y}, {state})" for k, (x, y, _, state, _) in enumerate(self.lemmings)]
        literals += [f"{skill}_left({count})" for skill, count in self.skills]
        literals += [f"saved({saved})", f"lost({lost})", f"released({released})"]
        self.literals = frozenset(literals)

    @property
    def alive(self):
        return [k for k, lemming in enumerate(self.lemmings) if lemming[3] != "blocker"]

    def __eq__(self, other):
        return (isinstance(other, LemmingsState) and self.rows == other.rows
                and self.lemmings == other.lemmings and self.released == other.released
                and self.saved == other.saved and self.lost == other.lost
                and self.skills == other.skills and self.tick == other.tick)

    def __hash__(self):
        return hash((self.rows, self.lemmings, self.released, self.saved, self.lost, self.skills, self.tick))

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        grid = [list(row) for row in self.rows]
        for x, y, direction, state, _ in self.lemmings:
            grid[y][x] = {"blocker": "B", "digger": "D", "basher": "H", "builder": "U"}.get(
                state, ">" if direction > 0 else "<")
        skills = ", ".join(f"{count} {skill}" for skill, count in self.skills)
        return "\n".join("".join(row) for row in grid) + (
            f"\ntick {self.tick}: {self.released} out, {self.saved} saved, {self.lost} lost; {skills}")

    def __repr__(self):
        return f"<LemmingsState(tick={self.tick}, out={len(self.lemmings)}, saved={self.saved}, lost={self.lost})>"


# ------------------------------------------------------------------------------- the crowd

def solid(rows, x, y):
    return not (0 <= x < len(rows[0]) and 0 <= y < len(rows)) or rows[y][x] in (ROCK, EARTH)


def run_ticks(rows, lemmings, released, saved, lost, tick, count, entrance, exit_, total, ticks):
    """Run the crowd for `ticks` ticks. Returns the new terrain, lemmings and counts."""
    grid = [list(row) for row in rows]
    crowd = [list(lemming) for lemming in lemmings]
    for _ in range(ticks):
        if released < total and tick % RELEASE == 0:
            crowd.append([entrance[0], entrance[1], 1, "faller", 0])
            released += 1
        blockers = {(x, y) for x, y, _, state, _ in crowd if state == "blocker"}
        survivors = []
        for lemming in crowd:
            x, y, direction, state, counter = lemming
            if state == "blocker":
                survivors.append(lemming)
                continue
            if state == "faller":
                if not solid(grid, x, y + 1):
                    if y + 1 >= len(grid):
                        lost += 1
                        continue
                    lemming[1], lemming[4] = y + 1, counter + 1
                else:
                    if counter > FATAL_FALL:
                        lost += 1
                        continue
                    lemming[3], lemming[4] = "walker", 0
                survivors.append(lemming)
                continue
            if not solid(grid, x, y + 1):                       # the ground went, or an edge
                lemming[3], lemming[4] = "faller", 0
                survivors.append(lemming)
                continue
            if state == "digger":
                if grid[y + 1][x] == EARTH:
                    grid[y + 1][x] = AIR
                    lemming[1] = y + 1
                else:
                    lemming[3] = "walker"
                survivors.append(lemming)
                continue
            if state == "basher":
                if 0 <= x + direction < len(grid[0]) and grid[y][x + direction] == EARTH:
                    grid[y][x + direction] = AIR
                    if y > 0 and grid[y - 1][x + direction] == EARTH:
                        grid[y - 1][x + direction] = AIR
                    lemming[0] = x + direction
                else:
                    lemming[3] = "walker"
                survivors.append(lemming)
                continue
            if state == "builder":
                nx, ny = x + direction, y - 1
                if counter >= BRICKS or solid(grid, nx, ny) or solid(grid, nx, y) or ny < 0:
                    lemming[3], lemming[4] = "walker", 0
                else:
                    grid[y][nx] = EARTH                          # the brick, under the next step
                    lemming[0], lemming[1], lemming[4] = nx, ny, counter + 1
                survivors.append(lemming)
                continue
            # a walker
            if (x, y) == exit_:
                saved += 1
                continue
            nx = x + direction
            if (nx, y) in blockers or nx < 0 or nx >= len(grid[0]):
                lemming[2] = -direction
            elif not solid(grid, nx, y):
                lemming[0] = nx
            elif not solid(grid, nx, y - 1) and y > 0:
                lemming[0], lemming[1] = nx, y - 1                # a step of one
            else:
                lemming[2] = -direction                          # a wall of two: turn
            survivors.append(lemming)
        crowd = survivors
        tick += 1
    return tuple("".join(row) for row in grid), crowd, released, saved, lost, tick


# ------------------------------------------------------------------------------- the levels

def draw_level(random_, lemmings=None, quota=None):
    """A random level as the instance dict `set_instance` takes: earth platforms at
    different heights with gaps and walls between them, the entrance over the first, the
    exit on the last, and a handful of skills."""
    grid = [[AIR] * WIDTH for _ in range(HEIGHT)]
    for x in range(WIDTH):
        grid[HEIGHT - 1][x] = ROCK
    platforms = []
    x, y = 1, random_.randint(3, 6)
    while x < WIDTH - 4:
        length = random_.randint(4, 8)
        end = min(WIDTH - 2, x + length)
        for px in range(x, end):
            grid[y][px] = EARTH
            for py in range(y + 1, HEIGHT - 1):                  # earth beneath each platform
                grid[py][px] = EARTH if random_.random() < 0.85 else ROCK
        platforms.append((x, end - 1, y))
        step = random_.choice((-3, -2, 2, 3, 4))
        gap = random_.choice((0, 0, 1, 2))
        x = end + gap
        y = max(2, min(HEIGHT - 3, y + step))
    for px, (_, end, y) in enumerate(platforms[:-1]):           # a wall at some platform ends
        if random_.random() < 0.5 and y - 2 >= 1:
            for wy in (y - 1, y - 2):
                grid[wy][end] = EARTH
    first, last = platforms[0], platforms[-1]
    entrance = (first[0] + 1, max(0, first[2] - 3))
    exit_ = (last[1] - 1, last[2] - 1)
    grid[entrance[1]][entrance[0]] = ENTRANCE
    grid[exit_[1]][exit_[0]] = EXIT
    lemmings = lemmings or random_.randint(5, 8)
    return {"rows": ["".join(row) for row in grid], "entrance": list(entrance), "exit": list(exit_),
            "lemmings": lemmings, "quota": quota or max(2, lemmings - random_.randint(1, 3)),
            "skills": {"blocker": random_.randint(0, 1), "digger": random_.randint(1, 2),
                       "basher": random_.randint(1, 2), "builder": random_.randint(1, 2)}}


class LemmingsEnv(Environment):
    """Save the quota."""

    def __init__(self):
        super().__init__("lemmings")
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
        """Select a level: `{"rows": [...], "entrance": [x, y], "exit": [x, y], "lemmings": n,
        "quota": q, "skills": {"blocker": b, "digger": d, "basher": h, "builder": u}}`."""
        for key in ("rows", "entrance", "exit", "lemmings", "quota", "skills"):
            if key not in instance:
                raise ValueError(f"a level needs `{key}`")
        self.instance = {"rows": [str(row) for row in instance["rows"]],
                         "entrance": tuple(int(v) for v in instance["entrance"]),
                         "exit": tuple(int(v) for v in instance["exit"]),
                         "lemmings": int(instance["lemmings"]), "quota": int(instance["quota"]),
                         "skills": {skill: int(instance["skills"].get(skill, 0)) for skill in SKILLS}}
        self.index = None
        self.witness = self.witness_expansions = None

    def generate_instance(self, seed=None, lemmings=None, quota=None, search_limit=400, attempts=40):
        """Draw a level, select it, and return it as the dict `set_instance` takes.

        `lemmings` (five to eight when unset) and `quota` (one to three fewer) are
        `draw_level`'s. A draw is kept only if letting the crowd walk unaided saves fewer
        than the quota and a best-first search over decisions, guided by the lemmings saved
        and how near the rest are to the exit, finds a plan within `search_limit`
        expansions; that plan is left in `witness` and what the search spent in
        `witness_expansions`.
        """
        random_, _ = rng(seed)
        found = {}

        def draw(attempt):
            return draw_level(random_, lemmings=lemmings, quota=quota)

        def accept(instance):
            self.set_instance(instance)
            unaided = self.simulate([WAIT] * (MAX_TICKS // DECISION))[-1]
            if self.is_goal(unaided):
                return False
            outcome = bounded_search(self, search_limit, progress=self.__progress__)
            if outcome.plan is None:
                return False
            found["plan"], found["expansions"] = outcome.plan, outcome.expansions
            return True

        instance = draw_until(draw, accept, attempts, "lemmings level")
        self.set_instance(instance)
        self.witness, self.witness_expansions = found["plan"], found["expansions"]
        return instance

    def __progress__(self, state):
        ex, ey = self.instance["exit"]
        near = min((abs(x - ex) + abs(y - ey) for x, y, _, s, _ in state.lemmings if s != "blocker"),
                   default=WIDTH + HEIGHT)
        return (self.instance["quota"] - state.saved) * 100 + near

    # ------------------------------------------------------------------- contract

    def reset(self):
        if self.instance is None:
            self.set_index(0)
        self.state = LemmingsState(self.instance["rows"], (), 0, 0, 0, self.instance["skills"], 0,
                                   self.instance["quota"])
        self.state_history = [self.state]
        return self.state, {"level": self.index, "lemmings": self.instance["lemmings"],
                            "quota": self.instance["quota"], "skills": dict(self.instance["skills"]),
                            "generated": self.index is None}

    def is_goal(self, state):
        return state.saved >= self.instance["quota"]

    def is_terminal(self, state):
        if self.is_goal(state):
            return False
        could_still = state.saved + len(state.alive) + (self.instance["lemmings"] - state.released)
        return state.tick >= MAX_TICKS or could_still < self.instance["quota"]

    def get_actions(self, state=None):
        state = state or self.state
        skills = dict(state.skills)
        return [LemmingsAction(skill, k) for k in state.alive for skill in SKILLS
                if skills[skill] > 0 and state.lemmings[k][3] == "walker"] + [WAIT]

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
        if not isinstance(action, LemmingsAction):
            action = LemmingsAction.parse(action)
        lemmings = [list(lemming) for lemming in state.lemmings]
        skills = dict(state.skills)
        if action.skill is not None:
            k = action.lemming
            if not (0 <= k < len(lemmings)) or lemmings[k][3] != "walker" or skills[action.skill] <= 0:
                return state
            lemmings[k][3], lemmings[k][4] = action.skill, 0
            skills[action.skill] -= 1
        rows, crowd, released, saved, lost, tick = run_ticks(
            state.rows, lemmings, state.released, state.saved, state.lost, state.tick,
            None, self.instance["entrance"], self.instance["exit"], self.instance["lemmings"], DECISION)
        return LemmingsState(rows, crowd, released, saved, lost, skills, tick, state.quota,
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
        before = self.state.saved
        self.state = self.__advance__(self.state, action)
        self.state_history.append(self.state)
        return self.state, self.state.saved - before

    def render(self):
        lines = [f"step {k}:\n{state}" for k, state in enumerate(self.state_history)]
        print("\n".join(lines))
        return lines


#: The bundled levels: `generate_instance(seed)` for the seed beside each, embedded as the
#: plain data `set_instance` takes, with the plan each was accepted on in
#: `tests/data/lemmings_solutions.json`.
LEVELS = (
    # seed 6000; 357 expansions, 14-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "    .                           ",
             "    .                           ",
             " ....                           ",
             " ...#                           ",
             " ....                           ",
             " ....                           ",
             " ..#.........                   ",
             " ............     .         X   ",
             " .#.........#     .      .....  ",
             " #...#.............      .#...  ",
             " .#..#...........#.      .#.##  ",
             " #.....###........#...........  ",
             " ....#.....#..................  ",
             "################################"],
     "entrance": [2, 1], "exit": [28, 9], "lemmings": 7, "quota": 5,
     "skills": {"blocker": 0, "digger": 2, "basher": 2, "builder": 1}},
    # seed 6001; 260 expansions, 9-decision plan
    {"rows": ["  E                             ",
             "                                ",
             "        ...............         ",
             " ..........#...#....#.#     X   ",
             " .................#.... ......  ",
             " ...............##.#... ..#.#.  ",
             " ........#...........#. ......  ",
             " ...#.............#.... ......  ",
             " .....#......#.......## .#...#  ",
             " ..#...#............... #....#  ",
             " ......#.#.......#..... ......  ",
             " ...#.................. ...#..  ",
             " .....#..#..........#.. ...#..  ",
             " ......#............... .....#  ",
             " ...........#.....#.... ....#.  ",
             "################################"],
     "entrance": [2, 0], "exit": [28, 3], "lemmings": 8, "quota": 6,
     "skills": {"blocker": 1, "digger": 2, "basher": 1, "builder": 1}},
    # seed 6002; 30 expansions, 10-decision plan
    {"rows": ["  E                             ",
             "                                ",
             "                                ",
             " ........                       ",
             " ........                       ",
             " .....#.......                  ",
             " ..........##.     .            ",
             " .............     .            ",
             " .#.#......#........            ",
             " ...........#.......            ",
             " .#.........#.....##     X      ",
             " #..........#.....#........     ",
             " .###....##.........##.#...     ",
             " #...#.#.......#..#.#......     ",
             " .#...#....#.#.........#...     ",
             "################################"],
     "entrance": [2, 0], "exit": [25, 10], "lemmings": 7, "quota": 6,
     "skills": {"blocker": 0, "digger": 2, "basher": 2, "builder": 2}},
    # seed 6004; 21 expansions, 11-decision plan
    {"rows": ["                                ",
             "                                ",
             "  E  ..............             ",
             "    ...........#...             ",
             "    ....................        ",
             " .....#.................        ",
             " ...#...#...............        ",
             " #..............#.....#.  X     ",
             " ..#............#...........    ",
             " .#.....#.......#...#.......    ",
             " ............###.....#...#..    ",
             " ......#...........##.#...##    ",
             " ..#.#.....#..............##    ",
             " ...#.....##........#.#.....    ",
             " .............#.............    ",
             "################################"],
     "entrance": [2, 2], "exit": [26, 7], "lemmings": 8, "quota": 6,
     "skills": {"blocker": 0, "digger": 2, "basher": 2, "builder": 2}},
    # seed 6005; 15 expansions, 9-decision plan
    {"rows": ["                                ",
             "                                ",
             "                ....            ",
             "  E      ........##.            ",
             "        ............            ",
             "        ..........#.            ",
             " ............#.............     ",
             " .#........................     ",
             " #.............#.#...#....# X   ",
             " ......#....#.........#.#.....  ",
             " ...........#......#...#......  ",
             " ..............#..........#...  ",
             " ..#.#......##.##...........#.  ",
             " ..#............#..#.##.......  ",
             " ..........#.#..#........#....  ",
             "################################"],
     "entrance": [2, 3], "exit": [28, 8], "lemmings": 5, "quota": 3,
     "skills": {"blocker": 1, "digger": 1, "basher": 1, "builder": 2}},
    # seed 6006; 155 expansions, 13-decision plan
    {"rows": ["                                ",
             "                                ",
             "                                ",
             "  E                             ",
             "        .                       ",
             "        .                   X   ",
             " ........              . .....  ",
             " ..#.....              . .....  ",
             " ..#....#        ....... .....  ",
             " ..#....#        #...#.. .#...  ",
             " ...#...............#... ..#..  ",
             " ...#.#................. ..#..  ",
             " .....#.#............#.. .....  ",
             " ...#.............#....# #..#.  ",
             " .....#..........##..... .##..  ",
             "################################"],
     "entrance": [2, 3], "exit": [28, 5], "lemmings": 7, "quota": 5,
     "skills": {"blocker": 1, "digger": 2, "basher": 1, "builder": 2}},
    # seed 6007; 35 expansions, 11-decision plan
    {"rows": ["                                ",
             "                                ",
             "              .....             ",
             "  E           .....             ",
             "        ..... #...#             ",
             "        ..... #....             ",
             " ........#.#. .............     ",
             " #........... ....#........     ",
             " .#..#....... #......#....# X   ",
             " #........... ......#.#.......  ",
             " .#..#.#..#.. ...#............  ",
             " ..#..#..#... #..#.......#....  ",
             " ............ ....#.....#...#.  ",
             " ..#...#....# .#....#..#......  ",
             " ...#..#..... ..............#.  ",
             "################################"],
     "entrance": [2, 3], "exit": [28, 8], "lemmings": 8, "quota": 7,
     "skills": {"blocker": 0, "digger": 2, "basher": 2, "builder": 2}},
    # seed 6008; 15 expansions, 8-decision plan
    {"rows": ["  E                             ",
             "                            X   ",
             "         .............    ....  ",
             " .....................    ....  ",
             " #...#..........#.....    ....  ",
             " ....#...............#........  ",
             " .......#..#.....#......#.....  ",
             " .....#...#....#.........#....  ",
             " ..#.##............#..........  ",
             " ......#.....#.....#....#.....  ",
             " .......#....#.......#....#...  ",
             " ...........#.....#...#....#..  ",
             " ...............#...#......#..  ",
             " .......#.#............##.#..#  ",
             " .....###.#...........#..#....  ",
             "################################"],
     "entrance": [2, 0], "exit": [28, 1], "lemmings": 6, "quota": 3,
     "skills": {"blocker": 0, "digger": 2, "basher": 2, "builder": 2}},
    # seed 6009; 19 expansions, 10-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "        .                       ",
             "        .                       ",
             " ........                       ",
             " .#....#.                       ",
             " .#...#..     .             X   ",
             " ##....#.     .         ......  ",
             " ....#.#.......       . #.....  ",
             " ...........#..       . ...#..  ",
             " ....#................. #.....  ",
             " ....#..#....#..#...... ......  ",
             " ....#..........###...# .#....  ",
             " ........##.......#.... .....#  ",
             " .......#......#....#.. ......  ",
             "################################"],
     "entrance": [2, 1], "exit": [28, 6], "lemmings": 7, "quota": 4,
     "skills": {"blocker": 0, "digger": 1, "basher": 2, "builder": 2}},
    # seed 6010; 51 expansions, 10-decision plan
    {"rows": ["                                ",
             "                                ",
             "                                ",
             "  E      ......                 ",
             "         #.....                 ",
             "         ......                 ",
             " ....... ..##.#.......          ",
             " ....... ..#......#...          ",
             " ....... ......#..#...    X     ",
             " ....... ............#......    ",
             " #...... ##..#.....#.....#..    ",
             " #...... .#......#.#........    ",
             " .##.... .##....#....##..#..    ",
             " ....... .................##    ",
             " ....#.. ...#.....#.........    ",
             "################################"],
     "entrance": [2, 3], "exit": [26, 8], "lemmings": 7, "quota": 5,
     "skills": {"blocker": 1, "digger": 2, "basher": 1, "builder": 1}},
    # seed 6011; 60 expansions, 8-decision plan
    {"rows": ["                                ",
             "  E                         X   ",
             "          ................ ...  ",
             "          #...#.#......... ...  ",
             " ........ ...#........#.#. .#.  ",
             " ........ ..#..#.......... ...  ",
             " .#...... ..........##...# ...  ",
             " .#...... ......#.......#. ..#  ",
             " .......# ..##...#........ #..  ",
             " ........ .#...#.....#.... .#.  ",
             " #.#..#.. .##......#.#.... ...  ",
             " .....#.. ...#.#.......... ..#  ",
             " ........ .......#.#...... ...  ",
             " .....#.. ....#.#....#.... #..  ",
             " .#...... ................ ...  ",
             "################################"],
     "entrance": [2, 1], "exit": [28, 1], "lemmings": 5, "quota": 2,
     "skills": {"blocker": 1, "digger": 1, "basher": 1, "builder": 2}},
    # seed 6012; 108 expansions, 11-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "                                ",
             "                                ",
             " ....                           ",
             " ....                           ",
             " ............                   ",
             " ........#.#.                   ",
             " .#...#....## ....              ",
             " .......#.... ....              ",
             " .....#...... ....              ",
             " ......#..... ...#              ",
             " .#.....#.... .#.......     X   ",
             " .....#....#. .......#........  ",
             " .....##...#. ......#.........  ",
             "################################"],
     "entrance": [2, 1], "exit": [28, 12], "lemmings": 6, "quota": 5,
     "skills": {"blocker": 0, "digger": 2, "basher": 1, "builder": 2}},
    # seed 6013; 48 expansions, 12-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "        ........                ",
             "        ....#...   .            ",
             " ...............   .            ",
             " .#.................            ",
             " #..................        X   ",
             " .....#.............  ........  ",
             " ......#.#.#.#....#.  ..#...#.  ",
             " .....#.............  ....#...  ",
             " ......#..........#.  ........  ",
             " ....#.#........#...  ..##....  ",
             " ........#.#......#.  ....#..#  ",
             " ..........#.....#.#  ........  ",
             " ...........#.......  ...#.#..  ",
             "################################"],
     "entrance": [2, 1], "exit": [28, 6], "lemmings": 8, "quota": 7,
     "skills": {"blocker": 1, "digger": 1, "basher": 1, "builder": 2}},
    # seed 6014; 22 expansions, 7-decision plan
    {"rows": ["  E                             ",
             "                                ",
             "                                ",
             " ........                       ",
             " #.......                       ",
             " ......##.....            X     ",
             " .....#......#       .......    ",
             " ...........#.       .#.....    ",
             " ....#..#.#.................    ",
             " ....#....#............#....    ",
             " ......#.#....##....#.......    ",
             " .......#...................    ",
             " #.#........................    ",
             " .....#.#..........#..#.....    ",
             " .#...................#.....    ",
             "################################"],
     "entrance": [2, 0], "exit": [26, 5], "lemmings": 5, "quota": 2,
     "skills": {"blocker": 0, "digger": 1, "basher": 2, "builder": 1}},
    # seed 6015; 51 expansions, 10-decision plan
    {"rows": ["                                ",
             "                                ",
             "  E                             ",
             "                           X    ",
             "                     . ......   ",
             " ......              . .#....   ",
             " ..#.#.       ........ .#.#..   ",
             " ......      ......... ....#.   ",
             " .....#      ......#.. ....#.   ",
             " ....................# ......   ",
             " .##.#............#... ......   ",
             " ...#................. ......   ",
             " .......#.#........... #....#   ",
             " .#................... ....#.   ",
             " ....#....#..#........ ....#.   ",
             "################################"],
     "entrance": [2, 2], "exit": [27, 3], "lemmings": 6, "quota": 4,
     "skills": {"blocker": 0, "digger": 1, "basher": 1, "builder": 2}},
    # seed 6016; 248 expansions, 10-decision plan
    {"rows": ["  E                             ",
             "    .                           ",
             "    .                           ",
             " ....                   X       ",
             " ...#   .....      .  ....      ",
             " ..#.   .....      .  ....      ",
             " ...................  ....      ",
             " .#.........#...#...  ...#      ",
             " ....#.....#.....#..  ....      ",
             " ........#.##......#  ....      ",
             " ...............#...  ....      ",
             " ..#.#...........#..  .#..      ",
             " ..#.#..............  ...#      ",
             " .....#.............  ..#.      ",
             " ...................  #.#.      ",
             "################################"],
     "entrance": [2, 0], "exit": [24, 3], "lemmings": 5, "quota": 3,
     "skills": {"blocker": 1, "digger": 1, "basher": 2, "builder": 2}},
    # seed 6017; 383 expansions, 12-decision plan
    {"rows": ["  E                             ",
             "                                ",
             "          ...........           ",
             " ........ ........#..    .      ",
             " ..#..... .#......##.    .      ",
             " ..#.#.#. .#...##.........      ",
             " ...#.#.# #.......#.....#.  X   ",
             " .#...... .....#......#.......  ",
             " .#...... #....####..#......#.  ",
             " .#.#.#.. ...#.......#........  ",
             " .......# .....#..#...........  ",
             " ....#.#. ..#.##........#.....  ",
             " .#...... ...#.......##.#.....  ",
             " #....... ....#..............#  ",
             " ........ #.#.....##..##.#....  ",
             "################################"],
     "entrance": [2, 0], "exit": [28, 6], "lemmings": 8, "quota": 7,
     "skills": {"blocker": 1, "digger": 1, "basher": 1, "builder": 1}},
    # seed 6018; 10 expansions, 9-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "        .                   X   ",
             "        .             ........  ",
             " ........             ........  ",
             " ........         ....##..#...  ",
             " ....#.#.         .#..........  ",
             " #..#....         #........#..  ",
             " ......#......... ....#....#..  ",
             " #.......#.#.##.. ............  ",
             " #..###.#....#... ....#...##..  ",
             " #.#....#........ ......##....  ",
             " .......#........ .........#..  ",
             " .#.......#.....# ......#.#.#.  ",
             " ..#..#..#....#.. ............  ",
             "################################"],
     "entrance": [2, 1], "exit": [28, 2], "lemmings": 5, "quota": 4,
     "skills": {"blocker": 1, "digger": 1, "basher": 2, "builder": 2}},
    # seed 6019; 24 expansions, 11-decision plan
    {"rows": ["  E                             ",
             "                                ",
             "                                ",
             " ......                         ",
             " ......        ......           ",
             " ##..#.        ..#.##           ",
             " ....##        .#....           ",
             " ...#.#....... ....#.    X      ",
             " .......##..#. .....#......     ",
             " ..........#.. .##.........     ",
             " ...###..#.... .#.......#..     ",
             " .......#..... ............     ",
             " ......##...#. .#...#..#...     ",
             " ..#...#.#.... ........#.#.     ",
             " ....#....#... .#.#...#....     ",
             "################################"],
     "entrance": [2, 0], "exit": [25, 7], "lemmings": 8, "quota": 7,
     "skills": {"blocker": 0, "digger": 1, "basher": 2, "builder": 2}},
    # seed 6020; 34 expansions, 14-decision plan
    {"rows": ["                                ",
             "                                ",
             "                                ",
             "  E                             ",
             "                                ",
             "                                ",
             " .....                          ",
             " .#...                          ",
             " .....  .......                 ",
             " .....  ...###.                 ",
             " ..#..  .#.....                 ",
             " ...#.  ..#....                 ",
             " .....  ....... .....     X     ",
             " .....  .#..... ............    ",
             " ..#..  ....... ......#.#.#.    ",
             "################################"],
     "entrance": [2, 3], "exit": [26, 12], "lemmings": 5, "quota": 3,
     "skills": {"blocker": 1, "digger": 2, "basher": 2, "builder": 2}},
    # seed 6021; 27 expansions, 12-decision plan
    {"rows": ["  E                             ",
             "    .                           ",
             "    .                           ",
             " ....                           ",
             " ....                           ",
             " ....      .                    ",
             " ....      .         .          ",
             " ...........         .          ",
             " .##....#...      ....          ",
             " #.#....#..#      ....          ",
             " .....#...#.........#.          ",
             " ......#..........#...     X    ",
             " ....#.......#...#.#.........   ",
             " ......#....#.......#.##.#...   ",
             " ...#.##.......#..#.#........   ",
             "################################"],
     "entrance": [2, 0], "exit": [27, 11], "lemmings": 6, "quota": 3,
     "skills": {"blocker": 0, "digger": 1, "basher": 2, "builder": 2}},
    # seed 6022; 30 expansions, 10-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "    .                           ",
             "    .                           ",
             " ....        ......             ",
             " ...#       .......             ",
             " #.##       ..##...        X    ",
             " .#................  ........   ",
             " .##...............  ..#....#   ",
             " ..................  ..#....#   ",
             " .#.#....#..####.#.  ..#.....   ",
             " .......##.........  ........   ",
             " ......##.......#..  ......#.   ",
             " #...#.............  ........   ",
             " ..................  ........   ",
             "################################"],
     "entrance": [2, 1], "exit": [27, 6], "lemmings": 8, "quota": 5,
     "skills": {"blocker": 1, "digger": 1, "basher": 1, "builder": 2}},
    # seed 6023; 18 expansions, 8-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "         ....           X       ",
             "         ....      .......      ",
             " ....... ....      ...#.#.      ",
             " .#....# .............#...      ",
             " ....... .##......#.......      ",
             " #....#. .....##..........      ",
             " ....... .......#.........      ",
             " ....#.. ......#..#...#...      ",
             " ....... .#..........#....      ",
             " ....... ##.#......#.#....      ",
             " ....#.. ......#..#......#      ",
             " #...... #........##....#.      ",
             " ....... .....#.........#.      ",
             "################################"],
     "entrance": [2, 1], "exit": [24, 2], "lemmings": 8, "quota": 6,
     "skills": {"blocker": 0, "digger": 2, "basher": 1, "builder": 2}},
    # seed 6024; 18 expansions, 8-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "       ..............           ",
             "       .....#.....#..           ",
             " ....................     .     ",
             " ....#...............     .     ",
             " .#........................     ",
             " .....##....#...........#..     ",
             " ..................#.......     ",
             " ....#..........#....#..... X   ",
             " .#..#.#.....#.#.........#....  ",
             " .#...........#....#..#......#  ",
             " ......#....#.............#...  ",
             " .#.........#.#....#..........  ",
             " ....#.#.....#.......#...#....  ",
             "################################"],
     "entrance": [2, 1], "exit": [28, 9], "lemmings": 6, "quota": 5,
     "skills": {"blocker": 0, "digger": 1, "basher": 1, "builder": 1}},
    # seed 6025; 45 expansions, 11-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "      .                         ",
             "      .                         ",
             " ......        ......           ",
             " .#...#        ......           ",
             " ....#.        #..#..   X       ",
             " #..#.....................      ",
             " ....#....................      ",
             " .........#....#.#..#.....      ",
             " .#..............#........      ",
             " #....#.#.......#..#......      ",
             " ..........#...#.....##...      ",
             " ........#.#..............      ",
             " #.....#...........#.#....      ",
             "################################"],
     "entrance": [2, 1], "exit": [24, 6], "lemmings": 5, "quota": 3,
     "skills": {"blocker": 0, "digger": 2, "basher": 1, "builder": 2}},
    # seed 6026; 92 expansions, 10-decision plan
    {"rows": ["  E                             ",
             "       .                        ",
             "       .                        ",
             " .......       .....            ",
             " ...#...       ....#      X     ",
             " ....##.       ...#. .......    ",
             " ............  ..... .......    ",
             " ......#...#.  ..... .......    ",
             " ..#.###.....  ..... .......    ",
             " ....#.......  ..... ...#...    ",
             " ....#.......  #.##. ...#...    ",
             " ..#..##.....  ..... ..#....    ",
             " .#.#........  ..... ...#...    ",
             " .#.##...##.#  ..#.. .......    ",
             " #...........  ..... ..#....    ",
             "################################"],
     "entrance": [2, 0], "exit": [26, 4], "lemmings": 7, "quota": 5,
     "skills": {"blocker": 0, "digger": 2, "basher": 1, "builder": 2}},
    # seed 6027; 48 expansions, 12-decision plan
    {"rows": ["  E                             ",
             "      .                         ",
             "      .........                 ",
             " ..............                 ",
             " ..............       .         ",
             " ....#...#.#...       .         ",
             " .........##..#........         ",
             " #.............##.#..#.         ",
             " ........#.....#..#....         ",
             " .........#....#..#....     X   ",
             " ...................#.........  ",
             " ...................#.###.....  ",
             " #...#........................  ",
             " ##.......#..........##.......  ",
             " .......#...#.......#..#......  ",
             "################################"],
     "entrance": [2, 0], "exit": [28, 9], "lemmings": 8, "quota": 6,
     "skills": {"blocker": 1, "digger": 2, "basher": 2, "builder": 2}},
    # seed 6028; 119 expansions, 10-decision plan
    {"rows": ["                                ",
             "               .                ",
             "  E            .                ",
             "       .........                ",
             "       ......#..                ",
             " ...............                ",
             " .....#....##.........          ",
             " #....#............#..      X   ",
             " ..##.##......#.....#.........  ",
             " .............................  ",
             " ..................#....#.....  ",
             " .....#.........#..........#..  ",
             " .#..............#...#........  ",
             " #..#...#...........#.#.......  ",
             " #..###.#.....##.#...#.......#  ",
             "################################"],
     "entrance": [2, 2], "exit": [28, 7], "lemmings": 8, "quota": 5,
     "skills": {"blocker": 1, "digger": 2, "basher": 1, "builder": 1}},
    # seed 6029; 153 expansions, 13-decision plan
    {"rows": ["                                ",
             "                                ",
             "                                ",
             "  E                             ",
             "      .                         ",
             "      .                         ",
             " ......                         ",
             " ...#..                         ",
             " #.....                         ",
             " ..#...                     X   ",
             " .............           .....  ",
             " .............   .      ......  ",
             " .............   .      ......  ",
             " .....#.##..#.................  ",
             " #..#...#.......#.#...........  ",
             "################################"],
     "entrance": [2, 3], "exit": [28, 9], "lemmings": 6, "quota": 4,
     "skills": {"blocker": 1, "digger": 1, "basher": 2, "builder": 2}},
    # seed 6030; 19 expansions, 10-decision plan
    {"rows": ["                                ",
             "  E                         X   ",
             "        .             ........  ",
             "        .             ..#....#  ",
             " ........       ....  ..#.....  ",
             " ........       .#..  ...#....  ",
             " .....#..       ....  #......#  ",
             " ...#.......... ...#  ........  ",
             " .##........... ....  .#.##...  ",
             " ..#........... .#..  #.......  ",
             " ....#.###..... ....  ........  ",
             " ...#.......... ..#.  .....#.#  ",
             " ###....#...#.. #...  .#......  ",
             " .............. .#..  ......#.  ",
             " #.#.....#..... ###.  ....#...  ",
             "################################"],
     "entrance": [2, 1], "exit": [28, 1], "lemmings": 8, "quota": 6,
     "skills": {"blocker": 1, "digger": 1, "basher": 1, "builder": 2}},
    # seed 6031; 127 expansions, 11-decision plan
    {"rows": ["                                ",
             "                                ",
             "  E                             ",
             "                                ",
             "                                ",
             " ....                           ",
             " .#.#                           ",
             " ....                     .     ",
             " ........                 .     ",
             " ...#....            ......     ",
             " ................   ...#...     ",
             " .#..#..#...#....   .......     ",
             " .........#.##..#......#.#. X   ",
             " .....#...###.................  ",
             " ........#.......#..#.#....##.  ",
             "################################"],
     "entrance": [2, 2], "exit": [28, 12], "lemmings": 8, "quota": 5,
     "skills": {"blocker": 0, "digger": 1, "basher": 1, "builder": 2}},
    # seed 6032; 56 expansions, 8-decision plan
    {"rows": ["                                ",
             "                                ",
             "  E      ..............         ",
             "         #........#....    X    ",
             "         ....#...............   ",
             " ....... ......#.............   ",
             " ....... ...#....#.....##....   ",
             " ..#.... #.#....#..##........   ",
             " ....... ...#...#........#...   ",
             " .....#. ..#.....#...#...#..#   ",
             " ....... .#....#.......#....#   ",
             " .#..#.. ...#.###..####......   ",
             " .....#. #.....#..####.#...#.   ",
             " ...#... ....#...#..........#   ",
             " #...##. ..#.......#..#......   ",
             "################################"],
     "entrance": [2, 2], "exit": [27, 3], "lemmings": 5, "quota": 2,
     "skills": {"blocker": 1, "digger": 2, "basher": 1, "builder": 2}},
    # seed 6033; 319 expansions, 13-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "                                ",
             "                  .             ",
             " ......           .             ",
             " ...##.    ........             ",
             " ......    ......##             ",
             " .#.........#....#.             ",
             " ..#..#..........#.      X      ",
             " ......##..................     ",
             " ..#.......#....##...#.....     ",
             " .#....#..#..#...#..#....#.     ",
             " .#..#.............#.#.....     ",
             " .......#......#....#...#..     ",
             " ....##..#......##......#..     ",
             "################################"],
     "entrance": [2, 1], "exit": [25, 8], "lemmings": 8, "quota": 6,
     "skills": {"blocker": 1, "digger": 1, "basher": 1, "builder": 2}},
    # seed 6034; 81 expansions, 10-decision plan
    {"rows": ["  E                             ",
             "        .                       ",
             "        .                       ",
             " ........                       ",
             " ........                       ",
             " #.......                       ",
             " #.....##                       ",
             " ...#...#........               ",
             " .......##.......      .        ",
             " #..#..#.........      .        ",
             " ...........#...........        ",
             " #........#...#..#...#..        ",
             " ......#.#......##...#..    X   ",
             " ....##.......................  ",
             " ...........#.......#.........  ",
             "################################"],
     "entrance": [2, 0], "exit": [28, 12], "lemmings": 8, "quota": 6,
     "skills": {"blocker": 0, "digger": 2, "basher": 2, "builder": 1}},
    # seed 6035; 43 expansions, 11-decision plan
    {"rows": ["  E                             ",
             "      .                         ",
             "      .                         ",
             " ......                         ",
             " ...#..                         ",
             " ......                     X   ",
             " ......  ........    .  ......  ",
             " ....#.  ...#....    .  ......  ",
             " .....#  .....#.......  ......  ",
             " ......  #....##......  ..#...  ",
             " ......  .............  ..#.#.  ",
             " ..#.#.  ....#........  ..#...  ",
             " .#....  ..##...#.#...  ......  ",
             " .#...#  ..........#..  ......  ",
             " .#.##.  .........#...  ..##..  ",
             "################################"],
     "entrance": [2, 0], "exit": [28, 5], "lemmings": 6, "quota": 3,
     "skills": {"blocker": 1, "digger": 2, "basher": 2, "builder": 2}},
    # seed 6036; 165 expansions, 12-decision plan
    {"rows": ["                                ",
             "                                ",
             "                                ",
             "  E                             ",
             "     .                          ",
             "     .                          ",
             " .....                          ",
             " .....                          ",
             " ..#.#.......                   ",
             " ....#..#..#.             X     ",
             " .....##.....        .......    ",
             " #........#..        .#.#...    ",
             " ...........................    ",
             " .#.#.......................    ",
             " ............##......#......    ",
             "################################"],
     "entrance": [2, 3], "exit": [26, 9], "lemmings": 6, "quota": 5,
     "skills": {"blocker": 1, "digger": 2, "basher": 1, "builder": 2}},
    # seed 6037; 371 expansions, 12-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "                                ",
             "                                ",
             " .....                          ",
             " .#...                          ",
             " .....                      X   ",
             " .............           .....  ",
             " .........#...           ....#  ",
             " ##...#.#..... ........  .....  ",
             " .....#...#... .##.....  ....#  ",
             " .....#....... ...#....  .....  ",
             " ....#....#... ....##..  #....  ",
             " ......#...... .##.....  .....  ",
             " .......#..### .#.....#  #....  ",
             "################################"],
     "entrance": [2, 1], "exit": [28, 6], "lemmings": 8, "quota": 6,
     "skills": {"blocker": 0, "digger": 1, "basher": 2, "builder": 2}},
    # seed 6038; 347 expansions, 9-decision plan
    {"rows": ["  E                             ",
             "                                ",
             "      ....                      ",
             " .........                      ",
             " ........#                      ",
             " #............                  ",
             " .............                  ",
             " ..#..........                  ",
             " #............                  ",
             " ..#..#.##............          ",
             " ...#........##.......          ",
             " .#...##.....##.......          ",
             " ...##............#...   X      ",
             " ...........#......... ....     ",
             " ..#.....#............ ....     ",
             "################################"],
     "entrance": [2, 0], "exit": [25, 12], "lemmings": 5, "quota": 3,
     "skills": {"blocker": 0, "digger": 1, "basher": 2, "builder": 2}},
    # seed 6039; 237 expansions, 11-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "       .                        ",
             "       .        .....           ",
             " .......        .....           ",
             " ...#...        .....           ",
             " .#................#.     X     ",
             " #...#.#....##.##...........    ",
             " ...............##..#......#    ",
             " ....#.....#.........#..#..#    ",
             " .......#..#..........###...    ",
             " ..#.....##...##............    ",
             " ..............#........##..    ",
             " .......#..........#........    ",
             " ...##.......#....#...#.#...    ",
             "################################"],
     "entrance": [2, 1], "exit": [26, 6], "lemmings": 7, "quota": 5,
     "skills": {"blocker": 0, "digger": 1, "basher": 2, "builder": 1}},
    # seed 6040; 12 expansions, 12-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "       .                        ",
             "       .                        ",
             " .......     .                  ",
             " ...#...     .                  ",
             " .#....#......                  ",
             " .#...........     .            ",
             " .....#.#.....     .            ",
             " ......#.....#......            ",
             " ..#........#.....#.            ",
             " #...#.....###.#....            ",
             " ..............#....      X     ",
             " ..#.........##..##.........    ",
             " .#..#......................    ",
             "################################"],
     "entrance": [2, 1], "exit": [26, 12], "lemmings": 8, "quota": 7,
     "skills": {"blocker": 1, "digger": 1, "basher": 2, "builder": 1}},
    # seed 6041; 89 expansions, 12-decision plan
    {"rows": ["                                ",
             "                                ",
             "          .                     ",
             "  E       .                     ",
             "    .......                     ",
             "    .......                     ",
             " ..........                     ",
             " .#..#.....                     ",
             " .#....#...  .......            ",
             " ..........  .#.....            ",
             " .....##...  ..##...            ",
             " .#........  .......     X      ",
             " #.#.......  ..............     ",
             " ..........  ....#........#     ",
             " ...#......  ..#....#.#....     ",
             "################################"],
     "entrance": [2, 3], "exit": [25, 11], "lemmings": 8, "quota": 6,
     "skills": {"blocker": 0, "digger": 2, "basher": 2, "builder": 2}},
    # seed 6042; 332 expansions, 10-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "                                ",
             "                      .         ",
             " .......              .         ",
             " .......         ......         ",
             " ..#..#.      .  ......         ",
             " ......#      .  #.....         ",
             " ...#...  .....  ...#..    X    ",
             " .......  .#...  .#.#........   ",
             " ..#....  ....#  .........###   ",
             " .....#.  .....  ............   ",
             " #......  .#.#.  ............   ",
             " ...##..  .....  ..........#.   ",
             " ....#..  #....  #..#........   ",
             "################################"],
     "entrance": [2, 1], "exit": [27, 8], "lemmings": 5, "quota": 3,
     "skills": {"blocker": 1, "digger": 1, "basher": 2, "builder": 2}},
    # seed 6043; 15 expansions, 8-decision plan
    {"rows": ["                                ",
             "                                ",
             "                                ",
             "  E                             ",
             "                                ",
             "                                ",
             " .....                          ",
             " .....    .                     ",
             " #..##    .                     ",
             " ..........                     ",
             " ..........                     ",
             " .#.#....#.                     ",
             " .......#.#              X      ",
             " #.....#...  ..............     ",
             " ...#......  .#.#..........     ",
             "################################"],
     "entrance": [2, 3], "exit": [25, 12], "lemmings": 5, "quota": 3,
     "skills": {"blocker": 0, "digger": 1, "basher": 1, "builder": 2}},
    # seed 6044; 38 expansions, 9-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "                                ",
             "                                ",
             " ....      .                    ",
             " ....      .                    ",
             " ...........                    ",
             " #......#...                    ",
             " ..#........      .             ",
             " #..#....#..      .             ",
             " ..#...#...........             ",
             " ..#..#............             ",
             " #..#..#..##.......     X       ",
             " ..............#...  .....      ",
             " ..................  #....      ",
             "################################"],
     "entrance": [2, 1], "exit": [24, 12], "lemmings": 6, "quota": 5,
     "skills": {"blocker": 1, "digger": 2, "basher": 1, "builder": 1}},
    # seed 6045; 153 expansions, 10-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "        .                       ",
             "        .                       ",
             " ........                       ",
             " ........                       ",
             " .......#                       ",
             " ..#.....  ........             ",
             " ........  ...##..#             ",
             " .#...#.#  ........             ",
             " .......#  .....#.#     X       ",
             " .#......  .#..#.#........      ",
             " ........  #..#..#....##.#      ",
             " .....#..  .#.#...........      ",
             " ........  ....#..#.##....      ",
             "################################"],
     "entrance": [2, 1], "exit": [24, 10], "lemmings": 6, "quota": 3,
     "skills": {"blocker": 1, "digger": 1, "basher": 2, "builder": 2}},
    # seed 6046; 93 expansions, 10-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "                                ",
             "                                ",
             " ........                       ",
             " .......#                       ",
             " .#...... ....                  ",
             " ........ #...    .             ",
             " ........ ....    .             ",
             " ........ .........             ",
             " #....... ..#..#...       .     ",
             " .#.#.... .........       .     ",
             " ........ ...#............. X   ",
             " ......#. ....#..#............  ",
             " .##.#..# ....................  ",
             "################################"],
     "entrance": [2, 1], "exit": [28, 12], "lemmings": 6, "quota": 5,
     "skills": {"blocker": 0, "digger": 1, "basher": 2, "builder": 2}},
    # seed 6047; 8 expansions, 8-decision plan
    {"rows": ["                                ",
             "                                ",
             "                                ",
             "  E                             ",
             "        .                X      ",
             "        .           .......     ",
             " ........           ....#.#     ",
             " .....#..      .... ......#     ",
             " ...#....      ##.. ....#..     ",
             " ........      .... .......     ",
             " .................# .......     ",
             " .............#..#. ..##...     ",
             " .................. ....#..     ",
             " .....#............ .......     ",
             " .###..........#... ##.....     ",
             "################################"],
     "entrance": [2, 3], "exit": [25, 4], "lemmings": 8, "quota": 5,
     "skills": {"blocker": 0, "digger": 1, "basher": 1, "builder": 2}},
    # seed 6048; 53 expansions, 10-decision plan
    {"rows": ["  E                             ",
             "        .                       ",
             "        ...............         ",
             " ..................#...     X   ",
             " .......#.....##.#............  ",
             " .......#.....#.#......#...#..  ",
             " .....#.#.......#.....#..#....  ",
             " ..#.............#........#.#.  ",
             " ..#......#.#.......#..#..#...  ",
             " .........#...................  ",
             " .....#..........#...#.......#  ",
             " ..#....#....#................  ",
             " ..........#..###....#........  ",
             " ......#..............#....#.#  ",
             " ....##..........#............  ",
             "################################"],
     "entrance": [2, 0], "exit": [28, 3], "lemmings": 8, "quota": 5,
     "skills": {"blocker": 1, "digger": 1, "basher": 1, "builder": 1}},
    # seed 6049; 33 expansions, 8-decision plan
    {"rows": ["  E                             ",
             "     .                      X   ",
             "     ........              ...  ",
             " .....#......             ....  ",
             " .......#..........       ....  ",
             " #....#............    ....#..  ",
             " #.................   ...#.##.  ",
             " ...........#..#...   ........  ",
             " ..#.......#..#...............  ",
             " .........#.....#...........#.  ",
             " .#........#.......#...#.#...#  ",
             " .#..#................#.##..#.  ",
             " #......................#..#.#  ",
             " .#....#......##..............  ",
             " #.......#..................#.  ",
             "################################"],
     "entrance": [2, 0], "exit": [28, 1], "lemmings": 5, "quota": 2,
     "skills": {"blocker": 1, "digger": 1, "basher": 1, "builder": 2}},
    # seed 6050; 14 expansions, 10-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "    .                           ",
             "    .                           ",
             " ....                           ",
             " #...       .                   ",
             " .#..       .                   ",
             " ............                   ",
             " ...#...#...#                   ",
             " ........#...                   ",
             " ..#.#...#..........            ",
             " ...##..#.....#.....            ",
             " ......#.#...#...#..        X   ",
             " #.##..........#..............  ",
             " ..#............##....#.....#.  ",
             "################################"],
     "entrance": [2, 1], "exit": [28, 12], "lemmings": 5, "quota": 3,
     "skills": {"blocker": 1, "digger": 1, "basher": 2, "builder": 2}},
    # seed 6051; 16 expansions, 11-decision plan
    {"rows": ["                                ",
             "                                ",
             "                                ",
             "  E                             ",
             "       .                        ",
             "       .                        ",
             " .......                        ",
             " .......                        ",
             " ...#...                        ",
             " ...#...........            X   ",
             " ....#...#...#..     .........  ",
             " ...............     .#.......  ",
             " ........#...................#  ",
             " .#.##.#........#.....#.##...#  ",
             " ..........#..................  ",
             "################################"],
     "entrance": [2, 3], "exit": [28, 9], "lemmings": 8, "quota": 7,
     "skills": {"blocker": 0, "digger": 1, "basher": 1, "builder": 2}},
    # seed 6052; 140 expansions, 11-decision plan
    {"rows": ["                                ",
             "                                ",
             "                                ",
             "  E                      .      ",
             "                         .      ",
             "                     .....      ",
             " ....    ....        #....  X   ",
             " ....    ....        #........  ",
             " ..##    #.#...........#...#.#  ",
             " #........#...#....#..........  ",
             " #.......###...#..............  ",
             " ......#......#...#..#.......#  ",
             " .........#....#.......#......  ",
             " #...............#.......#.#.#  ",
             " .......#......#...........#..  ",
             "################################"],
     "entrance": [2, 3], "exit": [28, 6], "lemmings": 8, "quota": 5,
     "skills": {"blocker": 1, "digger": 1, "basher": 1, "builder": 2}},
    # seed 6053; 68 expansions, 7-decision plan
    {"rows": ["  E                             ",
             "        .                       ",
             "        ......... ........      ",
             " ...........#...# ..#.....  X   ",
             " ...#.#.#..#..#.. ............  ",
             " .#.#............ ..##....#.#.  ",
             " ................ .......#...#  ",
             " .......#......#. ........#..#  ",
             " ......#.##...... ............  ",
             " ....#....###.... .......#..#.  ",
             " .......#.......# ......#..#..  ",
             " .#..#.....#..#.. #.####......  ",
             " ................ ........#...  ",
             " ..........#..... ...#.#.#.#..  ",
             " .#..#.....#...#. ..#.........  ",
             "################################"],
     "entrance": [2, 0], "exit": [28, 3], "lemmings": 6, "quota": 3,
     "skills": {"blocker": 0, "digger": 2, "basher": 2, "builder": 1}},
    # seed 6054; 21 expansions, 10-decision plan
    {"rows": ["                                ",
             "                                ",
             "  E                             ",
             "                                ",
             "                                ",
             " ......                         ",
             " ...#.#                         ",
             " #.....                         ",
             " #.#.#.                         ",
             " #..#......    ........         ",
             " ....#.....    .......#         ",
             " .#.#...#.......#.#...#  X      ",
             " ............#.#..#........     ",
             " .............#....#......#     ",
             " ..#.#...#...#...#..#.....#     ",
             "################################"],
     "entrance": [2, 2], "exit": [25, 11], "lemmings": 8, "quota": 7,
     "skills": {"blocker": 1, "digger": 1, "basher": 2, "builder": 2}},
    # seed 6055; 76 expansions, 9-decision plan
    {"rows": ["                                ",
             "                                ",
             "                                ",
             "  E                             ",
             "       .                        ",
             "       .                        ",
             " .......                        ",
             " ...#...                        ",
             " .......                  .     ",
             " .#...#......             .     ",
             " ..........##       .......     ",
             " #.....#.##..       .....#. X   ",
             " #.......##..........#........  ",
             " ..##...#..#...#.......#......  ",
             " ..............#.....#......#.  ",
             "################################"],
     "entrance": [2, 3], "exit": [28, 11], "lemmings": 5, "quota": 4,
     "skills": {"blocker": 0, "digger": 1, "basher": 1, "builder": 1}},
    # seed 6056; 281 expansions, 12-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "                                ",
             "                                ",
             " ........                       ",
             " ..#.....                       ",
             " .#......                       ",
             " ........ .......               ",
             " .......# .......               ",
             " #....... #......               ",
             " ....#... ...............       ",
             " .....##. ..#.#.#.#......       ",
             " ....##.# .......#..###..   X   ",
             " ..#.#... #....##.............  ",
             " ..##.#.. .............#......  ",
             "################################"],
     "entrance": [2, 1], "exit": [28, 12], "lemmings": 8, "quota": 7,
     "skills": {"blocker": 0, "digger": 2, "basher": 2, "builder": 2}},
    # seed 6057; 348 expansions, 8-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "      .... ....             X   ",
             "      .... ....          .....  ",
             " .......## .#..          .....  ",
             " ......... ....          .....  ",
             " #.......# .#..........  ..#..  ",
             " ......#.. .......#..##  ...#.  ",
             " ......... ...........#  #...#  ",
             " ...#...#. ........#..#  .....  ",
             " #.......# ....#....#..  .#...  ",
             " ....##... ..#.#..###..  .....  ",
             " ...#..... #.....###...  .#...  ",
             " .......#. ....#....#..  .#.#.  ",
             " #.....#.. ..#.........  .....  ",
             "################################"],
     "entrance": [2, 1], "exit": [28, 2], "lemmings": 5, "quota": 2,
     "skills": {"blocker": 1, "digger": 1, "basher": 2, "builder": 2}},
    # seed 6058; 146 expansions, 14-decision plan
    {"rows": ["                                ",
             "                                ",
             "  E                             ",
             "       .                        ",
             "       .                        ",
             " .......                        ",
             " #......                        ",
             " .......     .                  ",
             " .....##     .           .      ",
             " ....#........           .      ",
             " .......#.....      ......      ",
             " ..........#..      ....#.      ",
             " #....#.....##............  X   ",
             " ..........#...#.#.#...#......  ",
             " ..#.#.....#.#....#.....#.....  ",
             "################################"],
     "entrance": [2, 2], "exit": [28, 12], "lemmings": 6, "quota": 5,
     "skills": {"blocker": 1, "digger": 1, "basher": 2, "builder": 1}},
    # seed 6059; 251 expansions, 8-decision plan
    {"rows": ["  E                             ",
             "                                ",
             "     ........       .           ",
             " ............       .           ",
             " ...##...#...........           ",
             " ..........#.....#...      X    ",
             " .....#..#......#............   ",
             " ...#.......#..............##   ",
             " #.#..........##..##.........   ",
             " #.......................#...   ",
             " .....#.....#............##..   ",
             " .........##......#.#........   ",
             " ......##..#............#....   ",
             " .......#.........#..#....##.   ",
             " ..........#.###..#........##   ",
             "################################"],
     "entrance": [2, 0], "exit": [27, 5], "lemmings": 7, "quota": 5,
     "skills": {"blocker": 1, "digger": 1, "basher": 1, "builder": 1}},
    # seed 6060; 55 expansions, 9-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "        ....                    ",
             "        ....                    ",
             " ...........                    ",
             " #..........  ........          ",
             " .....#.....  ........          ",
             " .....#....#  ........          ",
             " .......#.#.  ..#....#    X     ",
             " ...#.......  ....#.#.......    ",
             " .#.........  .....#....#.#.    ",
             " #.......##.  ............#.    ",
             " .......#..#  ..#.#.#.......    ",
             " ....#......  .#......#....#    ",
             " ...#.......  ..............    ",
             "################################"],
     "entrance": [2, 1], "exit": [26, 8], "lemmings": 6, "quota": 3,
     "skills": {"blocker": 1, "digger": 2, "basher": 2, "builder": 2}},
    # seed 6061; 53 expansions, 9-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "               .......          ",
             "               #......      X   ",
             " ....     .... ......#........  ",
             " #...    ..... ...##...##....#  ",
             " ....    ...#. ....#..#..#....  ",
             " ............. .#........##...  ",
             " #...#.##..... .###...........  ",
             " ..##.....#... #.##.#.........  ",
             " ...#.....#.#. #....#..#......  ",
             " ...........#. ...##.........#  ",
             " .#..#..#..... .......#.......  ",
             " .##......#.#. ..#.........##.  ",
             " ......#...... ..........##...  ",
             "################################"],
     "entrance": [2, 1], "exit": [28, 3], "lemmings": 5, "quota": 2,
     "skills": {"blocker": 1, "digger": 1, "basher": 2, "builder": 2}},
    # seed 6062; 51 expansions, 9-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "        .                       ",
             "        .                       ",
             " ........                       ",
             " ..#.....                       ",
             " #...#.#.......          X      ",
             " ....#......#..    ........     ",
             " ...##.#......#    ##......     ",
             " ...................#.#.#..     ",
             " .........................#     ",
             " ..#.......#...............     ",
             " .#.................#......     ",
             " ...#..#..#.....#..........     ",
             " .##............#........#.     ",
             "################################"],
     "entrance": [2, 1], "exit": [25, 6], "lemmings": 8, "quota": 5,
     "skills": {"blocker": 1, "digger": 1, "basher": 2, "builder": 2}},
    # seed 6063; 74 expansions, 11-decision plan
    {"rows": ["                                ",
             "                                ",
             "                .               ",
             "  E             .               ",
             "        . .......               ",
             "        . .#.#...               ",
             " ........ ............          ",
             " #...#... .....#.#....      X   ",
             " ......## ....................  ",
             " ..#..... ...........#.......#  ",
             " ....#... #.....##..#....#....  ",
             " ........ #....#.............#  ",
             " ......#. ....................  ",
             " #.#...#. ...#....#........#..  ",
             " ##.#.... .....##....#......#.  ",
             "################################"],
     "entrance": [2, 3], "exit": [28, 7], "lemmings": 7, "quota": 4,
     "skills": {"blocker": 1, "digger": 1, "basher": 2, "builder": 2}},
    # seed 6064; 137 expansions, 8-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "       .......                  ",
             "       .......            X     ",
             " ...........#.        ......    ",
             " .#.#......#.#        ......    ",
             " ..#.#..............  .#..#.    ",
             " .#...#.#.#......#..  .....#    ",
             " .......#...........  ......    ",
             " #..................  ...#.#    ",
             " ....##..#.#........  ...#..    ",
             " ...#.......#.......  ......    ",
             " #...#.........#....  ......    ",
             " .##.....###.#......  ...#..    ",
             " .#......#..........  ......    ",
             "################################"],
     "entrance": [2, 1], "exit": [26, 3], "lemmings": 5, "quota": 2,
     "skills": {"blocker": 1, "digger": 1, "basher": 1, "builder": 1}},
    # seed 6065; 341 expansions, 13-decision plan
    {"rows": ["                                ",
             "                                ",
             "                                ",
             "  E                             ",
             "                            X   ",
             "                           ...  ",
             " ....                      ...  ",
             " ...#               ...... #..  ",
             " ..........       . #..... #..  ",
             " ..........       . ....#. ...  ",
             " #...#............. ....#. #.#  ",
             " ..#............#.. ....#. ...  ",
             " ....##........#..# ...... ...  ",
             " ...............#.. ...... #..  ",
             " ..#.......#..#...# .#.... ...  ",
             "################################"],
     "entrance": [2, 3], "exit": [28, 4], "lemmings": 5, "quota": 2,
     "skills": {"blocker": 1, "digger": 1, "basher": 1, "builder": 2}},
    # seed 6066; 36 expansions, 11-decision plan
    {"rows": ["                                ",
             "                                ",
             "  E                             ",
             "                                ",
             "                                ",
             " .....                          ",
             " ..#..                          ",
             " .....                          ",
             " ...........                    ",
             " ...#..#....                    ",
             " ..........#....                ",
             " ........#......       .        ",
             " ..#........#...       .  X     ",
             " ......#....................    ",
             " .....#...........#......#..    ",
             "################################"],
     "entrance": [2, 2], "exit": [26, 12], "lemmings": 7, "quota": 6,
     "skills": {"blocker": 1, "digger": 2, "basher": 1, "builder": 2}},
    # seed 6067; 274 expansions, 12-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "                                ",
             "                                ",
             " ........   .                   ",
             " #.......   .                   ",
             " ......#.....                   ",
             " ......#...#.                   ",
             " ........##.#     .             ",
             " #.##.....#.#     .             ",
             " #..#...#..........             ",
             " ...............##.             ",
             " ................#.        X    ",
             " ............................   ",
             " .....#.............#........   ",
             "################################"],
     "entrance": [2, 1], "exit": [27, 12], "lemmings": 6, "quota": 4,
     "skills": {"blocker": 1, "digger": 1, "basher": 1, "builder": 2}},
    # seed 6068; 46 expansions, 11-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "                                ",
             "                                ",
             " ....                           ",
             " ....                           ",
             " ....                           ",
             " #..#                           ",
             " ...#....                       ",
             " #.......                       ",
             " .#......                   X   ",
             " .....#.........       . .....  ",
             " .....##.#......       . .....  ",
             " ......#................ .....  ",
             " ...##....#............# ...#.  ",
             "################################"],
     "entrance": [2, 1], "exit": [28, 10], "lemmings": 7, "quota": 4,
     "skills": {"blocker": 0, "digger": 2, "basher": 1, "builder": 2}},
    # seed 6069; 13 expansions, 10-decision plan
    {"rows": ["                                ",
             "                                ",
             "  E    ..........      .        ",
             "       ..#.......      .        ",
             "       .................        ",
             " .......#..#..........#.        ",
             " ....#..##.....#.#.##...    X   ",
             " .........##..................  ",
             " ...................#.........  ",
             " .#......................#.#.#  ",
             " ....#....#...............#...  ",
             " .......#.....##.........##.##  ",
             " .......#...........#....#.#..  ",
             " #......#.........#.#.........  ",
             " .....#.....#.#.....#.#..#..#.  ",
             "################################"],
     "entrance": [2, 2], "exit": [28, 6], "lemmings": 7, "quota": 6,
     "skills": {"blocker": 1, "digger": 1, "basher": 2, "builder": 2}},
    # seed 6070; 14 expansions, 8-decision plan
    {"rows": ["                                ",
             "  E                     X       ",
             "    .              .......      ",
             "    .              .....#.      ",
             " ....              ......#      ",
             " #...      ...............      ",
             " .#..      .#.#...........      ",
             " .#.......................      ",
             " #...............#........      ",
             " ............#............      ",
             " ................##..#....      ",
             " ..................#...#..      ",
             " #.........#..#.#........#      ",
             " .......#.....#.........#.      ",
             " ......#.........#......#.      ",
             "################################"],
     "entrance": [2, 1], "exit": [24, 1], "lemmings": 5, "quota": 4,
     "skills": {"blocker": 1, "digger": 1, "basher": 1, "builder": 2}},
    # seed 6071; 16 expansions, 9-decision plan
    {"rows": ["  E                             ",
             "                                ",
             "                                ",
             " .....                          ",
             " .....          ......          ",
             " .....          ....#.          ",
             " ..... ........ ..#..#          ",
             " ..... .....#.. ......   X      ",
             " .#.#. ........ #....#.....     ",
             " ..... ....##.. .......#..#     ",
             " ..... ......#. #..........     ",
             " ..... #....#.. .#......##.     ",
             " ....# .##..#.. #..........     ",
             " ..#.. .#.....# .#.........     ",
             " #.#.. #..##... ...#.......     ",
             "################################"],
     "entrance": [2, 0], "exit": [25, 7], "lemmings": 6, "quota": 3,
     "skills": {"blocker": 0, "digger": 1, "basher": 1, "builder": 2}},
    # seed 6072; 136 expansions, 12-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "                                ",
             "                                ",
             " ......                         ",
             " ...#..                         ",
             " ......                         ",
             " ......                         ",
             " ..##......                     ",
             " ..........                     ",
             " .......#..                     ",
             " .#....#...........   .         ",
             " ...##.............   .   X     ",
             " ......#...##.#.............    ",
             " .......#..........#.....#..    ",
             "################################"],
     "entrance": [2, 1], "exit": [26, 12], "lemmings": 6, "quota": 4,
     "skills": {"blocker": 1, "digger": 2, "basher": 2, "builder": 2}},
    # seed 6073; 9 expansions, 9-decision plan
    {"rows": ["  E                             ",
             "    .                           ",
             "    .......                     ",
             " .....#....                     ",
             " ..........  ........           ",
             " .....#....  #.....#.     X     ",
             " ..........  .....#.........    ",
             " ..#.......  #.....#........    ",
             " .....#.#..  .#....#........    ",
             " ....##....  #.....#......#.    ",
             " ..#.......  #.........#....    ",
             " #.........  ..............#    ",
             " ...#......  ...............    ",
             " .#.#..#...  .#..........##.    ",
             " .#........  .##...#........    ",
             "################################"],
     "entrance": [2, 0], "exit": [26, 5], "lemmings": 6, "quota": 3,
     "skills": {"blocker": 0, "digger": 1, "basher": 2, "builder": 2}},
    # seed 6074; 17 expansions, 9-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "       ..............           ",
             "       ..............           ",
             " .......###......#.........     ",
             " #...#.#...#...#.......#.#. X   ",
             " ...#..#...#........#.........  ",
             " ........#......#.............  ",
             " ...............##........#...  ",
             " ..................#..........  ",
             " .......##................#...  ",
             " ...#....#..........#..##.....  ",
             " .#..#...#.....#..........##..  ",
             " #...#..#..............#....#.  ",
             " .....#..##.........###..#...#  ",
             "################################"],
     "entrance": [2, 1], "exit": [28, 5], "lemmings": 6, "quota": 3,
     "skills": {"blocker": 1, "digger": 2, "basher": 1, "builder": 1}},
    # seed 6075; 24 expansions, 10-decision plan
    {"rows": ["                                ",
             "                                ",
             "                                ",
             "  E    ......                   ",
             "      .#.....                   ",
             "      .......     .             ",
             " ......##....     .             ",
             " ....#.......  ....             ",
             " ..........#.  ....             ",
             " .......#.#..  #...             ",
             " ............  ...#     X       ",
             " ...##.#.#...  ...........      ",
             " ............  .........#.      ",
             " #...##.##...  .....##....      ",
             " #...#....#..  .#.........      ",
             "################################"],
     "entrance": [2, 3], "exit": [24, 10], "lemmings": 5, "quota": 4,
     "skills": {"blocker": 1, "digger": 2, "basher": 2, "builder": 1}},
    # seed 6076; 23 expansions, 10-decision plan
    {"rows": ["                                ",
             "                                ",
             "  E                             ",
             "                                ",
             "                                ",
             " .....                          ",
             " ..#..                          ",
             " .....       .                  ",
             " ..#.#       .                  ",
             " .............                  ",
             " .............                  ",
             " ....#.#......      .           ",
             " .......#..#..      .      X    ",
             " ......#.............  ......   ",
             " ..#.....#...#.......  ...#..   ",
             "################################"],
     "entrance": [2, 2], "exit": [27, 12], "lemmings": 7, "quota": 6,
     "skills": {"blocker": 0, "digger": 1, "basher": 1, "builder": 2}},
    # seed 6077; 66 expansions, 15-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "                                ",
             "                                ",
             " .......            .           ",
             " .#.....            .           ",
             " .......      .......           ",
             " #.##...      .......           ",
             " ......#.............           ",
             " ........##..#.#...#.   X       ",
             " .............#...........      ",
             " #...#....#...#......#....      ",
             " ........##.....#.......#.      ",
             " ....#...........#........      ",
             " ......#..#..#...#....##..      ",
             "################################"],
     "entrance": [2, 1], "exit": [24, 9], "lemmings": 6, "quota": 5,
     "skills": {"blocker": 0, "digger": 2, "basher": 2, "builder": 2}},
    # seed 6078; 367 expansions, 13-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "      .                         ",
             "      .                         ",
             " ......                         ",
             " .....#                         ",
             " ..#...........                 ",
             " ............##      .          ",
             " .....#........      .          ",
             " .##.#......##........          ",
             " .........#...#....#..          ",
             " ..........#..........          ",
             " ...........#.......#.....  X   ",
             " ......................#......  ",
             " ......................#.....#  ",
             "################################"],
     "entrance": [2, 1], "exit": [28, 12], "lemmings": 5, "quota": 3,
     "skills": {"blocker": 0, "digger": 1, "basher": 1, "builder": 1}},
    # seed 6079; 25 expansions, 9-decision plan
    {"rows": ["                                ",
             "                                ",
             "                                ",
             "  E                             ",
             "        .                       ",
             "        .                       ",
             " ........            .          ",
             " .##.#.#.            .          ",
             " ##.##...     ........          ",
             " ......#.     ........    X     ",
             " ......#........##..........    ",
             " ......#..#...............##    ",
             " .....#.##.......#.........#    ",
             " ...#.......#..#..##...#....    ",
             " ...........#...............    ",
             "################################"],
     "entrance": [2, 3], "exit": [26, 9], "lemmings": 7, "quota": 4,
     "skills": {"blocker": 0, "digger": 1, "basher": 2, "builder": 2}},
    # seed 6080; 15 expansions, 7-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "     ..............             ",
             "     .........#.#..             ",
             " .....#...##.#.....     X       ",
             " .........#....#.#........      ",
             " ...............#..#...##.      ",
             " .....#.......###....##.#.      ",
             " ..#.............#..#.##..      ",
             " ...........##............      ",
             " ...#.#........#..#......#      ",
             " #........###....#.......#      ",
             " #...#........#.#.........      ",
             " .#.#..#...#...#.#........      ",
             " ##........#..............      ",
             "################################"],
     "entrance": [2, 1], "exit": [24, 4], "lemmings": 5, "quota": 2,
     "skills": {"blocker": 0, "digger": 1, "basher": 2, "builder": 1}},
    # seed 6081; 30 expansions, 11-decision plan
    {"rows": ["                                ",
             "                                ",
             "                                ",
             "  E                             ",
             "      .                         ",
             "      .                         ",
             " ......                         ",
             " #.....                         ",
             " ......                         ",
             " ......  ......                 ",
             " ...#..  ......                 ",
             " ..#...  ..#...                 ",
             " ......  ......           X     ",
             " ...#..  ...#....... .......    ",
             " ......  .......#... .......    ",
             "################################"],
     "entrance": [2, 3], "exit": [26, 12], "lemmings": 5, "quota": 2,
     "skills": {"blocker": 0, "digger": 1, "basher": 2, "builder": 2}},
    # seed 6082; 11 expansions, 8-decision plan
    {"rows": ["  E                             ",
             "                           X    ",
             "     ..... ...... ...........   ",
             " ........# ..#... ..........#   ",
             " ...#..#.. .#.... ...........   ",
             " #........ ...... .#.#..#....   ",
             " .#...#... ...... ........#..   ",
             " .....#..# #..... ...........   ",
             " .#....... .....# .........#.   ",
             " .......## ..#... ..#..#...##   ",
             " ......#.. ..#... .......#...   ",
             " #........ .....# .........#.   ",
             " ......... ...... ..#.#......   ",
             " ...##.... .#.... ..#.#......   ",
             " ....###.. .#..## ...#.......   ",
             "################################"],
     "entrance": [2, 0], "exit": [27, 1], "lemmings": 5, "quota": 3,
     "skills": {"blocker": 1, "digger": 2, "basher": 2, "builder": 2}},
    # seed 6083; 29 expansions, 9-decision plan
    {"rows": ["  E                             ",
             "      .                         ",
             "      .                         ",
             " ......                         ",
             " ......                         ",
             " .#........                     ",
             " ....#...#.                     ",
             " .#..#.#.#.       .             ",
             " ....##....       .             ",
             " ..................             ",
             " #....#....#.......      X      ",
             " ................#.........     ",
             " .#......#...........##....     ",
             " ........#.............#.#.     ",
             " ##.#...........#.#.....#..     ",
             "################################"],
     "entrance": [2, 0], "exit": [25, 10], "lemmings": 5, "quota": 3,
     "skills": {"blocker": 1, "digger": 1, "basher": 1, "builder": 1}},
    # seed 6084; 53 expansions, 9-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "        .....                   ",
             "        .....                   ",
             " .........###                   ",
             " ............                   ",
             " ..##...#....  ......           ",
             " ..........#.  ##....           ",
             " #....#......  #..##.           ",
             " #...#.......  ......     X     ",
             " ............  .#...#.......    ",
             " ............  ..##.#......#    ",
             " #...#.......  .##..........    ",
             " ...#.......#  ......#...#.#    ",
             " #.#......#..  ..#.#..#.....    ",
             "################################"],
     "entrance": [2, 1], "exit": [26, 9], "lemmings": 5, "quota": 3,
     "skills": {"blocker": 0, "digger": 1, "basher": 1, "builder": 2}},
    # seed 6085; 253 expansions, 12-decision plan
    {"rows": ["                                ",
             "                                ",
             "                                ",
             "  E                             ",
             "                                ",
             "                                ",
             " .......                        ",
             " #......                        ",
             " .......                        ",
             " ......#                        ",
             " ....... ........               ",
             " ...#... ..#.....      .        ",
             " #...#.. ..#.....      .    X   ",
             " ....... ............... .....  ",
             " ......# .............#. .....  ",
             "################################"],
     "entrance": [2, 3], "exit": [28, 12], "lemmings": 8, "quota": 7,
     "skills": {"blocker": 1, "digger": 2, "basher": 2, "builder": 1}},
    # seed 6086; 51 expansions, 11-decision plan
    {"rows": ["  E                             ",
             "                                ",
             "        ................        ",
             " ...... ..##............        ",
             " #..... ................        ",
             " ....#. ................  X     ",
             " ...... ............#.......    ",
             " ...... #....##.....#.....#.    ",
             " ...... ...#....#..........#    ",
             " ..#.#. ...#.#...#.#........    ",
             " ..#... .....#..........##..    ",
             " #..... #..............#..#.    ",
             " ...... ............#.#.....    ",
             " ...... ......#.............    ",
             " ##...# ..##.#..#...........    ",
             "################################"],
     "entrance": [2, 0], "exit": [26, 5], "lemmings": 8, "quota": 6,
     "skills": {"blocker": 1, "digger": 1, "basher": 2, "builder": 2}},
    # seed 6087; 34 expansions, 9-decision plan
    {"rows": ["                                ",
             "                                ",
             "                                ",
             "  E                             ",
             "                                ",
             "                                ",
             " ....                           ",
             " ....       .                   ",
             " #...       .                   ",
             " ............                   ",
             " #.....#.#...                   ",
             " ..#......... ........          ",
             " ............ ........    X     ",
             " ............ ..#...........    ",
             " ..#....#.... .......#..##..    ",
             "################################"],
     "entrance": [2, 3], "exit": [26, 12], "lemmings": 7, "quota": 5,
     "skills": {"blocker": 1, "digger": 1, "basher": 2, "builder": 1}},
    # seed 6088; 43 expansions, 11-decision plan
    {"rows": ["                                ",
             "                                ",
             "  E                             ",
             "                     .          ",
             "                     .          ",
             " .......      ........          ",
             " #......      ........          ",
             " .......  ..........#.     X    ",
             " #......  ..........#........   ",
             " .....#.  .#....#.#.#.....#..   ",
             " ..#....  .............#.....   ",
             " .......  ............#.#....   ",
             " ....##.  ##...#....#........   ",
             " ...#..#  ..........#.#......   ",
             " .##..#.  ...#...............   ",
             "################################"],
     "entrance": [2, 2], "exit": [27, 7], "lemmings": 8, "quota": 5,
     "skills": {"blocker": 0, "digger": 1, "basher": 1, "builder": 1}},
    # seed 6089; 7 expansions, 7-decision plan
    {"rows": ["  E                             ",
             "        .                   X   ",
             "        .             ........  ",
             " ........             #.......  ",
             " #.#..#..             ........  ",
             " .....#..    .......  ..#..#..  ",
             " #.......    .....#.  #.......  ",
             " .................#.  #....##.  ",
             " ...#.....#..#.....#  ........  ",
             " ......#......#.....  ......#.  ",
             " ......#........#..#  ........  ",
             " ....#.....#.....##.  #....#..  ",
             " #...#.#............  .#......  ",
             " .#....#............  ........  ",
             " .#....#......#.##.#  #.......  ",
             "################################"],
     "entrance": [2, 0], "exit": [28, 1], "lemmings": 5, "quota": 3,
     "skills": {"blocker": 1, "digger": 1, "basher": 1, "builder": 2}},
    # seed 6090; 150 expansions, 8-decision plan
    {"rows": ["                                ",
             "                                ",
             "  E                             ",
             "                                ",
             "                                ",
             " ......                         ",
             " ...#..                         ",
             " ......                         ",
             " .............                  ",
             " #....#.......           X      ",
             " ........#...#       ......     ",
             " .............       ......     ",
             " #.....##........... ...#..     ",
             " ..........#........ ..#..#     ",
             " ...#............... ##....     ",
             "################################"],
     "entrance": [2, 2], "exit": [25, 9], "lemmings": 7, "quota": 5,
     "skills": {"blocker": 1, "digger": 1, "basher": 1, "builder": 2}},
    # seed 6091; 288 expansions, 12-decision plan
    {"rows": ["  E                             ",
             "                                ",
             "                                ",
             " ......                     X   ",
             " ......                  .....  ",
             " .#....                 ..#..#  ",
             " ......                 ......  ",
             " .#.... ......    ............  ",
             " ...... ......    ............  ",
             " .#.... ...............#......  ",
             " ..#... .#....................  ",
             " #....# ............#...#....#  ",
             " ...... ..#....#..............  ",
             " ##.#.. .#..........#..#......  ",
             " .#.... ......................  ",
             "################################"],
     "entrance": [2, 0], "exit": [28, 3], "lemmings": 7, "quota": 4,
     "skills": {"blocker": 1, "digger": 2, "basher": 2, "builder": 2}},
    # seed 6093; 99 expansions, 14-decision plan
    {"rows": ["                                ",
             "                                ",
             "  E                             ",
             "                                ",
             "                                ",
             " ........                       ",
             " .#....#.             .....     ",
             " ....####.......     ...#..     ",
             " #.#...##.......     ...... X   ",
             " ..#...##.#..##.......#.#.....  ",
             " ..#......#...............#...  ",
             " ..........#.........#.#...#..  ",
             " ....#....##..........#.....#.  ",
             " ...#..............#....#.....  ",
             " .........#....#....####......  ",
             "################################"],
     "entrance": [2, 2], "exit": [28, 8], "lemmings": 7, "quota": 6,
     "skills": {"blocker": 1, "digger": 1, "basher": 2, "builder": 2}},
    # seed 6094; 237 expansions, 13-decision plan
    {"rows": ["                                ",
             "                                ",
             "  E       ........              ",
             "          ........       .      ",
             "          ........       .      ",
             " ........ ................      ",
             " #..#...# .....##....#....  X   ",
             " #....... ....................  ",
             " #....... ............##..#...  ",
             " .#.....# #.........##.....#..  ",
             " #..#.... ......#....#..#.....  ",
             " ...#.... ..............#..#..  ",
             " ........ ...#.....#.....#.#..  ",
             " ........ .............#.##...  ",
             " ....#... #..#................  ",
             "################################"],
     "entrance": [2, 2], "exit": [28, 6], "lemmings": 5, "quota": 3,
     "skills": {"blocker": 0, "digger": 2, "basher": 2, "builder": 2}},
    # seed 6095; 35 expansions, 10-decision plan
    {"rows": ["                                ",
             "                                ",
             "  E                             ",
             "                                ",
             "                                ",
             " .......       .                ",
             " .......       .                ",
             " .....#.........                ",
             " ..............#                ",
             " ...............     .          ",
             " ....###....#...     .          ",
             " ..#...#....#.........          ",
             " ...#...#...........#.   X      ",
             " #...#..#..................     ",
             " ...#.##............#.....#     ",
             "################################"],
     "entrance": [2, 2], "exit": [25, 12], "lemmings": 7, "quota": 6,
     "skills": {"blocker": 0, "digger": 1, "basher": 1, "builder": 2}},
    # seed 6096; 154 expansions, 10-decision plan
    {"rows": ["  E                             ",
             "     .                          ",
             "     .......... ........        ",
             " .....#..#..... #.......        ",
             " ..#......##... ..#.#..#        ",
             " .#..#.#.....#. ...#....    X   ",
             " .#.#.......... ...#..........  ",
             " ......#...##.. ......#......#  ",
             " ......#....... .#............  ",
             " #.#...#...#... ..............  ",
             " .#...#...#.... .........#..#.  ",
             " ....#........# ..............  ",
             " .###........#. ....#...#.....  ",
             " .............. ..#....#..#.#.  ",
             " .........##.#. ..#........#..  ",
             "################################"],
     "entrance": [2, 0], "exit": [28, 5], "lemmings": 6, "quota": 4,
     "skills": {"blocker": 1, "digger": 1, "basher": 1, "builder": 2}},
    # seed 6097; 297 expansions, 7-decision plan
    {"rows": ["  E                             ",
             "                                ",
             "         ...........            ",
             " .............#.....    X       ",
             " #....##...#.#.#.... .....      ",
             " #.......#......#... .....      ",
             " ..##..........#.... .#.##      ",
             " ................#.# .#.#.      ",
             " .#.#...#......#.... ....#      ",
             " .#......#.#........ .....      ",
             " #...........##..... ....#      ",
             " ......#.....#...... ..#.#      ",
             " ...#..#..#.#.#.#... #..#.      ",
             " .#....#..#..####... ...#.      ",
             " .#..........#..#..# .....      ",
             "################################"],
     "entrance": [2, 0], "exit": [24, 3], "lemmings": 5, "quota": 4,
     "skills": {"blocker": 0, "digger": 2, "basher": 2, "builder": 2}},
    # seed 6098; 14 expansions, 9-decision plan
    {"rows": ["                                ",
             "                                ",
             "                                ",
             "  E                             ",
             "      .                         ",
             "      .                         ",
             " ......                         ",
             " ....#.                         ",
             " .#........                     ",
             " ..#.##..##                     ",
             " .##...#.#.           ....      ",
             " ....#.....           #...      ",
             " ...............      ....  X   ",
             " ...#...#...........  ........  ",
             " ..#...#.....#..#...  .......#  ",
             "################################"],
     "entrance": [2, 3], "exit": [28, 12], "lemmings": 5, "quota": 4,
     "skills": {"blocker": 1, "digger": 2, "basher": 2, "builder": 2}},
    # seed 6099; 43 expansions, 10-decision plan
    {"rows": ["  E                             ",
             "                                ",
             "                                ",
             " ........                       ",
             " ........                       ",
             " ..#.....                  X    ",
             " ....#... ........    . .....   ",
             " ........ ...#...#    . .....   ",
             " #...#.#. #.....##..... .#..#   ",
             " ........ .......#.##.. ....#   ",
             " ......#. .#........... .....   ",
             " .#.....# #............ ...#.   ",
             " ....#... ............# ...#.   ",
             " ..#..... .#.........#. .#.#.   ",
             " #....### .....#..#.... .....   ",
             "################################"],
     "entrance": [2, 0], "exit": [27, 5], "lemmings": 7, "quota": 5,
     "skills": {"blocker": 1, "digger": 1, "basher": 1, "builder": 2}},
    # seed 6100; 17 expansions, 11-decision plan
    {"rows": ["                                ",
             "                                ",
             "                                ",
             "  E                             ",
             "                                ",
             "                                ",
             " .....                          ",
             " .....                          ",
             " ...#.                          ",
             " ...#.           ........       ",
             " ..#.........   ..#......   X   ",
             " ............   .##...#.......  ",
             " ...#....................##...  ",
             " ........#........##.......#..  ",
             " ..............#..............  ",
             "################################"],
     "entrance": [2, 3], "exit": [28, 10], "lemmings": 8, "quota": 7,
     "skills": {"blocker": 1, "digger": 1, "basher": 1, "builder": 2}},
    # seed 6101; 62 expansions, 10-decision plan
    {"rows": ["                                ",
             "  E                             ",
             "                                ",
             "                                ",
             " .......                        ",
             " #....#.                        ",
             " ....#..        .......         ",
             " .......        .......   X     ",
             " ...............#...........    ",
             " ..............#.#..#......#    ",
             " ......##.......##.........#    ",
             " ###.....#.#....#.#.......#.    ",
             " .........#...............#.    ",
             " ..##..#..................#.    ",
             " .............##..##.......#    ",
             "################################"],
     "entrance": [2, 1], "exit": [26, 7], "lemmings": 8, "quota": 7,
     "skills": {"blocker": 0, "digger": 1, "basher": 2, "builder": 1}},
)
