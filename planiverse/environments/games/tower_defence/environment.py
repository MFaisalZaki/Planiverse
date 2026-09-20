"""Tower defence: enemies march along a path, towers beside it shoot them, and the player
decides between waves what to build with the gold the last wave paid.

The rules are the genre's own, written as small as they can be. A map is a path of cells
from an entrance to an exit with a few building slots beside it. Two kinds of tower are on
offer, an arrow tower that fires often and short, and a cannon that fires slowly, far and
hard. Between waves the player may build, as long as the gold holds out, and then starts the
next wave, which runs to its end: enemies enter at intervals and walk the path at their
speed, each tower fires at the enemy furthest along within its range whenever it has
reloaded, a kill pays a bounty, and an enemy that reaches the exit costs a life. The goal is
to be alive after the last wave.

What a placement is worth is only known by running the wave. Range, rate, damage, speed,
hit points and the timing of arrivals all interact, and the same tower in the same slot is
worth a lot against a slow thick wave and nothing against a fast thin one that is past it
before it reloads. That is what keeps the environment out of PDDL.

## Determinism

The wave is integer arithmetic and fixed-order loops, so the same decisions give the same
outcome anywhere.

## Instances and generation

`generate_instance(seed, ...)` draws the path, the slots and the waves, and keeps the draw
only if starting every wave without building loses and a best-first search over decisions
finds a way to win within the budget; the plan is left in `witness`. The bundled maps are such
draws, embedded as plain data with the seed each came from. The method is generate-and-test
(search-based procedural content generation: Togelius et al. 2011,
https://doi.org/10.1109/TCIAIG.2011.2148116).
"""
import math

from planiverse.environments.base import Environment
from planiverse.environments.generation import bounded_search, draw_until, rng

#: The map: `WIDTH` by `HEIGHT` cells.
WIDTH, HEIGHT = 12, 8
#: The towers on offer: cost, range in cells, damage per shot, ticks to reload.
TOWERS = {"arrow": dict(cost=60, range=1.6, damage=4, reload=2),
          "cannon": dict(cost=140, range=2.8, damage=24, reload=6)}
#: Speed is in hundredths of a cell per tick; a wave is over when its last enemy is dead or
#: through.
SPEED_UNIT = 100
MAX_TICKS = 4000


class TowerAction:
    """`build(kind, slot)` or `start`."""

    def __init__(self, kind=None, slot=None):
        if kind is None:
            self.name = "start"
        else:
            if kind not in TOWERS:
                raise ValueError(f"unknown tower: {kind!r}")
            self.name = f"build({kind},{slot})"
        self.kind, self.slot = kind, slot

    @classmethod
    def parse(cls, text):
        text = str(text).strip()
        if text == "start":
            return cls()
        inside = text[len("build("):-1]
        kind, slot = inside.split(",")
        return cls(kind, int(slot))

    def cost(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, TowerAction) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    __repr__ = __str__


START = TowerAction()


class TowerState:
    """Which slots hold which towers, the gold, the lives, and the waves fought so far."""

    def __init__(self, towers, gold, lives, wave, depth=0):
        self.towers = tuple(sorted(towers))
        self.gold, self.lives, self.wave = gold, lives, wave
        self.depth = depth
        literals = [f"tower({kind}, {slot})" for slot, kind in self.towers]
        literals += [f"gold({gold // 20 * 20})", f"lives({lives})", f"waves_fought({wave})"]
        self.literals = frozenset(literals)

    def __eq__(self, other):
        return (isinstance(other, TowerState) and self.towers == other.towers
                and self.gold == other.gold and self.lives == other.lives and self.wave == other.wave)

    def __hash__(self):
        return hash((self.towers, self.gold, self.lives, self.wave))

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        built = ", ".join(f"{kind} at {slot}" for slot, kind in self.towers) or "nothing built"
        return f"wave {self.wave} fought, {self.lives} lives, {self.gold} gold; {built}"

    def __repr__(self):
        return f"<TowerState(wave={self.wave}, lives={self.lives}, gold={self.gold}, towers={len(self.towers)})>"


# ------------------------------------------------------------------------------- the wave

def run_wave(path, slots, towers, wave):
    """Fight one wave to its end and return `(kills, leaks)`.

    `wave` is `(count, hp, speed, spacing)`: enemies with `hp` hit points enter `spacing`
    ticks apart and walk `speed` hundredths of a cell a tick. Each tick every tower with a
    loaded shot fires at the enemy furthest along the path within its range.
    """
    count, hp, speed, spacing = wave
    guns = [(slots[slot], TOWERS[kind]) for slot, kind in towers]
    reload = [0] * len(guns)
    enemies = []                                  # [position in hundredths of a cell, hp]
    entered = kills = leaks = 0
    length = len(path) * SPEED_UNIT
    for tick in range(MAX_TICKS):
        if entered < count and tick % spacing == 0:
            enemies.append([0, hp])
            entered += 1
        for enemy in enemies:
            enemy[0] += speed
        alive = []
        for enemy in enemies:
            if enemy[0] >= length:
                leaks += 1
            else:
                alive.append(enemy)
        enemies = alive
        for index, ((sx, sy), gun) in enumerate(guns):
            if reload[index] > 0:
                reload[index] -= 1
                continue
            best = None
            for enemy in enemies:
                if enemy[1] <= 0:
                    continue
                cell = path[min(len(path) - 1, enemy[0] // SPEED_UNIT)]
                if math.hypot(cell[0] - sx, cell[1] - sy) <= gun["range"]:
                    if best is None or enemy[0] > best[0]:
                        best = enemy
            if best is not None:
                best[1] -= gun["damage"]
                reload[index] = gun["reload"]
        survivors = []
        for enemy in enemies:
            if enemy[1] <= 0:
                kills += 1
            else:
                survivors.append(enemy)
        enemies = survivors
        if entered == count and not enemies:
            break
    return kills, leaks


# ------------------------------------------------------------------------------- the maps

def draw_map(random_, waves=4, slots=8):
    """A random map as the instance dict `set_instance` takes: a path from the left edge to
    the right that wanders up and down, building slots beside it, and escalating waves."""
    y = random_.randint(2, HEIGHT - 3)
    path, x = [(0, y)], 0
    while x < WIDTH - 1:
        if random_.random() < 0.45:
            step = random_.choice((-1, 1))
            if 0 <= y + step < HEIGHT and (x, y + step) not in path:
                y += step
                path.append((x, y))
                continue
        x += 1
        path.append((x, y))
    beside = sorted({(px + dx, py + dy) for px, py in path for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))
                     if 0 <= px + dx < WIDTH and 0 <= py + dy < HEIGHT} - set(path))
    random_.shuffle(beside)
    chosen = sorted(beside[:slots])
    hp, speed = random_.randint(10, 16), random_.randint(8, 12)
    table = []
    for k in range(waves):
        table.append([4 + 2 * k + random_.randint(0, 2), hp, speed, random_.choice((10, 14, 18))])
        hp = int(hp * random_.uniform(1.4, 1.8))
        speed = min(20, speed + random_.randint(0, 3))
    return {"path": [list(cell) for cell in path], "slots": [list(cell) for cell in chosen],
            "waves": table, "gold": random_.choice((100, 120, 140)), "lives": random_.choice((2, 3, 4)),
            "bounty": random_.choice((4, 5, 6))}


class TowerDefenceEnv(Environment):
    """Be alive after the last wave."""

    def __init__(self):
        super().__init__("tower_defence")
        self.instance = None
        self.index = None
        self.state = None
        self.state_history = []
        self.witness = None
        self.witness_expansions = None

    # ------------------------------------------------------------------ instances

    def set_index(self, index):
        if not 0 <= index < len(MAPS):
            raise IndexError(f"Invalid index: {index}. There are {len(MAPS)} maps, so the "
                             f"index must be 0-{len(MAPS) - 1}.")
        self.set_instance(MAPS[index])
        self.index = index

    def set_instance(self, instance):
        """Select a map: `{"path": [[x, y], ...], "slots": [[x, y], ...], "waves": [[count,
        hp, speed, spacing], ...], "gold": g, "lives": n, "bounty": b}`."""
        for key in ("path", "slots", "waves", "gold", "lives", "bounty"):
            if key not in instance:
                raise ValueError(f"a map needs `{key}`")
        self.instance = {"path": [tuple(cell) for cell in instance["path"]],
                         "slots": [tuple(cell) for cell in instance["slots"]],
                         "waves": [tuple(int(v) for v in wave) for wave in instance["waves"]],
                         "gold": int(instance["gold"]), "lives": int(instance["lives"]),
                         "bounty": int(instance["bounty"])}
        self.index = None
        self.witness = self.witness_expansions = None

    def generate_instance(self, seed=None, waves=None, slots=None, search_limit=600, attempts=40):
        """Draw a map, select it, and return it as the dict `set_instance` takes.

        `waves` (three to five, drawn when unset) and `slots` (six to eight) are `draw_map`'s.
        A draw is kept only if starting every wave without building loses, so that building
        is the game, and a best-first search over decisions (waves left to fight, then lives
        lost) finds a way to win within `search_limit` expansions; that plan is left in
        `witness` and what the search spent in `witness_expansions`.
        """
        random_, _ = rng(seed)
        found = {}

        def draw(attempt):
            return draw_map(random_, waves=waves or random_.randint(3, 5),
                            slots=slots or random_.randint(6, 8))

        def accept(instance):
            self.set_instance(instance)
            if any(self.simulate(plan)[-1].lives > 0 for plan in self.__baselines__()):
                return False                     # no thought needed: not a puzzle
            outcome = bounded_search(self, search_limit, progress=self.__progress__)
            if outcome.plan is None:
                return False
            found["plan"], found["expansions"] = outcome.plan, outcome.expansions
            return True

        instance = draw_until(draw, accept, attempts, "tower defence map")
        self.set_instance(instance)
        self.witness, self.witness_expansions = found["plan"], found["expansions"]
        return instance

    def __baselines__(self):
        """The plans that need no thought: start every wave building nothing, and, for each
        kind of tower, fill the slots in order with it whenever the gold allows. A map that
        one of these wins is thrown back."""
        plans = [[START] * len(self.instance["waves"])]
        for kind in TOWERS:
            plan, state = [], self.reset()[0]
            while not self.is_goal(state) and not self.is_terminal(state):
                taken = {slot for slot, _ in state.towers}
                free = [slot for slot in range(len(self.instance["slots"])) if slot not in taken]
                if free and state.gold >= TOWERS[kind]["cost"]:
                    action = TowerAction(kind, free[0])
                else:
                    action = START
                plan.append(action)
                state = self.__advance__(state, action)
            plans.append(plan)
        return plans

    def __progress__(self, state):
        return (len(self.instance["waves"]) - state.wave) * 10 + (self.instance["lives"] - state.lives)

    # ------------------------------------------------------------------- contract

    def reset(self):
        if self.instance is None:
            self.set_index(0)
        self.state = TowerState((), self.instance["gold"], self.instance["lives"], 0)
        self.state_history = [self.state]
        return self.state, {"map": self.index, "waves": len(self.instance["waves"]),
                            "slots": len(self.instance["slots"]), "gold": self.instance["gold"],
                            "lives": self.instance["lives"], "generated": self.index is None}

    def is_goal(self, state):
        return state.lives > 0 and state.wave == len(self.instance["waves"])

    def is_terminal(self, state):
        return state.lives <= 0

    def get_actions(self):
        return [TowerAction(kind, slot) for slot in range(len(self.instance["slots"]))
                for kind in TOWERS] + [START]

    def successors(self, state):
        if self.is_goal(state) or self.is_terminal(state):
            return []
        children = []
        for action in self.get_actions():
            child = self.__advance__(state, action)
            if child != state:
                children.append((action, child))
        return children

    def __advance__(self, state, action):
        if self.is_goal(state) or self.is_terminal(state):
            return state
        if not isinstance(action, TowerAction):
            action = TowerAction.parse(action)
        if action.kind is not None:
            taken = {slot for slot, _ in state.towers}
            cost = TOWERS[action.kind]["cost"]
            if action.slot in taken or not 0 <= action.slot < len(self.instance["slots"]) \
                    or state.gold < cost:
                return state
            return TowerState(state.towers + ((action.slot, action.kind),), state.gold - cost,
                              state.lives, state.wave, state.depth + 1)
        wave = self.instance["waves"][state.wave]
        kills, leaks = run_wave(self.instance["path"], self.instance["slots"], state.towers, wave)
        return TowerState(state.towers, state.gold + kills * self.instance["bounty"],
                          max(0, state.lives - leaks), state.wave + 1, state.depth + 1)

    def simulate(self, plan):
        state, _ = self.reset()
        trace = [state]
        for action in plan:
            trace.append(self.__advance__(trace[-1], action))
        return trace

    def step(self, action):
        if self.state is None:
            raise ValueError("call reset() first")
        before = self.state.lives
        self.state = self.__advance__(self.state, action)
        self.state_history.append(self.state)
        return self.state, before - self.state.lives

    def render(self):
        lines = [f"step {k}: {state}" for k, state in enumerate(self.state_history)]
        print("\n".join(lines))
        return lines


#: The bundled maps: `generate_instance(seed)` for the seed beside each, embedded as the
#: plain data `set_instance` takes, with the plan each was accepted on in
#: `tests/data/tower_defence_solutions.json`.
MAPS = (
    # seed 3000; 10 expansions, 7-decision plan
    {"path": [[0, 4], [0, 3], [1, 3], [2, 3], [3, 3], [3, 4], [3, 5], [4, 5], [4, 4], [5, 4], [6, 4], [7, 4], [8, 4], [9, 4], [9, 3], [10, 3], [10, 2], [11, 2]],
     "slots": [[0, 5], [4, 6], [5, 3], [6, 3], [6, 5], [7, 3], [10, 1], [10, 4]], "waves": [[6, 13, 12, 10], [8, 19, 14, 10], [9, 28, 14, 10], [10, 43, 17, 18]],
     "gold": 100, "lives": 4, "bounty": 5},
    # seed 3001; 10 expansions, 8-decision plan
    {"path": [[0, 2], [0, 3], [0, 4], [1, 4], [2, 4], [2, 5], [3, 5], [3, 6], [3, 7], [4, 7], [4, 6], [5, 6], [6, 6], [7, 6], [8, 6], [9, 6], [9, 5], [10, 5], [10, 6], [10, 7], [11, 7]],
     "slots": [[2, 7], [5, 5], [5, 7], [7, 5], [9, 4], [9, 7], [10, 4]], "waves": [[4, 11, 9, 10], [6, 18, 9, 10], [9, 27, 11, 18], [11, 41, 13, 18], [14, 68, 16, 18]],
     "gold": 100, "lives": 3, "bounty": 4},
    # seed 3002; 109 expansions, 8-decision plan
    {"path": [[0, 5], [1, 5], [2, 5], [3, 5], [4, 5], [4, 4], [4, 3], [5, 3], [5, 2], [5, 1], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0]],
     "slots": [[0, 4], [2, 4], [4, 1], [7, 1], [8, 1], [10, 1]], "waves": [[6, 16, 12, 14], [7, 25, 14, 10], [10, 35, 16, 14], [12, 49, 18, 14], [12, 79, 19, 14]],
     "gold": 140, "lives": 4, "bounty": 4},
    # seed 3003; 16 expansions, 9-decision plan
    {"path": [[0, 5], [0, 4], [1, 4], [2, 4], [3, 4], [4, 4], [4, 5], [5, 5], [6, 5], [7, 5], [7, 4], [8, 4], [9, 4], [10, 4], [11, 4]],
     "slots": [[3, 3], [3, 5], [4, 3], [6, 4], [9, 3], [9, 5], [11, 3], [11, 5]], "waves": [[4, 11, 12, 10], [8, 19, 13, 18], [10, 30, 15, 14], [10, 52, 15, 14], [14, 91, 17, 18]],
     "gold": 120, "lives": 4, "bounty": 4},
    # seed 3004; 14 expansions, 9-decision plan
    {"path": [[0, 2], [1, 2], [1, 3], [2, 3], [3, 3], [4, 3], [4, 4], [4, 5], [5, 5], [6, 5], [7, 5], [7, 6], [8, 6], [9, 6], [9, 7], [10, 7], [11, 7]],
     "slots": [[1, 1], [1, 4], [3, 4], [3, 5], [4, 2], [5, 4], [5, 6], [11, 6]], "waves": [[5, 15, 9, 18], [7, 21, 9, 10], [8, 32, 10, 18], [12, 53, 12, 18], [13, 77, 14, 10]],
     "gold": 140, "lives": 2, "bounty": 6},
    # seed 3005; 25 expansions, 7-decision plan
    {"path": [[0, 4], [0, 5], [1, 5], [2, 5], [3, 5], [3, 4], [4, 4], [5, 4], [5, 5], [5, 6], [6, 6], [7, 6], [8, 6], [8, 7], [9, 7], [9, 6], [9, 5], [10, 5], [10, 4], [11, 4]],
     "slots": [[5, 7], [6, 4], [6, 7], [7, 7], [10, 3], [11, 3]], "waves": [[6, 16, 12, 18], [8, 24, 15, 14], [8, 39, 18, 18], [12, 60, 19, 10]],
     "gold": 140, "lives": 4, "bounty": 6},
    # seed 3006; 15 expansions, 10-decision plan
    {"path": [[0, 4], [0, 3], [1, 3], [2, 3], [3, 3], [4, 3], [4, 2], [5, 2], [5, 1], [6, 1], [6, 0], [7, 0], [8, 0], [8, 1], [9, 1], [9, 2], [9, 3], [10, 3], [10, 4], [11, 4]],
     "slots": [[1, 4], [2, 4], [4, 4], [5, 0], [10, 1], [10, 2], [11, 5]], "waves": [[5, 11, 9, 18], [6, 19, 11, 18], [10, 28, 13, 14], [11, 50, 16, 10], [12, 89, 16, 14]],
     "gold": 140, "lives": 3, "bounty": 5},
    # seed 3007; 11 expansions, 5-decision plan
    {"path": [[0, 2], [0, 1], [1, 1], [2, 1], [3, 1], [3, 2], [3, 3], [4, 3], [4, 2], [5, 2], [5, 1], [6, 1], [6, 0], [7, 0], [7, 1], [8, 1], [8, 0], [9, 0], [10, 0], [10, 1], [11, 1]],
     "slots": [[0, 3], [2, 2], [3, 0], [4, 1], [6, 2], [7, 2], [8, 2]], "waves": [[4, 15, 8, 18], [8, 25, 8, 10], [9, 41, 9, 14]],
     "gold": 100, "lives": 2, "bounty": 4},
    # seed 3008; 116 expansions, 6-decision plan
    {"path": [[0, 2], [0, 1], [1, 1], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [5, 3], [6, 3], [7, 3], [8, 3], [9, 3], [9, 4], [10, 4], [11, 4]],
     "slots": [[0, 0], [3, 1], [4, 1], [6, 2], [6, 4], [7, 4], [11, 5]], "waves": [[4, 15, 10, 10], [6, 22, 13, 14], [9, 37, 14, 14], [12, 53, 15, 10]],
     "gold": 140, "lives": 3, "bounty": 4},
    # seed 3009; 8 expansions, 7-decision plan
    {"path": [[0, 4], [0, 3], [1, 3], [2, 3], [2, 4], [3, 4], [3, 3], [4, 3], [5, 3], [6, 3], [7, 3], [7, 2], [8, 2], [9, 2], [9, 3], [10, 3], [10, 2], [11, 2]],
     "slots": [[0, 2], [0, 5], [5, 4], [6, 2], [6, 4], [7, 1], [11, 3]], "waves": [[4, 13, 8, 14], [6, 18, 8, 14], [9, 28, 8, 10], [10, 41, 10, 10]],
     "gold": 120, "lives": 3, "bounty": 5},
    # seed 3010; 15 expansions, 7-decision plan
    {"path": [[0, 4], [1, 4], [2, 4], [2, 5], [3, 5], [3, 4], [4, 4], [5, 4], [6, 4], [7, 4], [8, 4], [9, 4], [9, 5], [9, 6], [10, 6], [10, 7], [11, 7]],
     "slots": [[2, 3], [2, 6], [4, 5], [6, 3], [6, 5], [8, 3], [8, 6]], "waves": [[5, 15, 9, 10], [6, 24, 12, 10], [9, 35, 15, 14], [11, 57, 18, 18]],
     "gold": 120, "lives": 2, "bounty": 5},
    # seed 3011; 118 expansions, 9-decision plan
    {"path": [[0, 2], [0, 1], [1, 1], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [5, 1], [6, 1], [7, 1], [7, 0], [8, 0], [8, 1], [9, 1], [9, 2], [10, 2], [11, 2]],
     "slots": [[0, 0], [1, 2], [3, 1], [6, 0], [7, 2], [8, 2], [9, 0], [11, 1]], "waves": [[5, 14, 8, 18], [7, 20, 8, 18], [9, 35, 10, 14], [11, 53, 10, 10], [12, 84, 10, 14]],
     "gold": 120, "lives": 2, "bounty": 4},
    # seed 3012; 112 expansions, 8-decision plan
    {"path": [[0, 2], [1, 2], [2, 2], [3, 2], [3, 3], [4, 3], [5, 3], [5, 4], [5, 5], [6, 5], [6, 6], [7, 6], [8, 6], [9, 6], [10, 6], [11, 6]],
     "slots": [[1, 1], [3, 1], [4, 2], [6, 7], [9, 7], [11, 5]], "waves": [[6, 14, 9, 18], [6, 24, 9, 10], [9, 38, 10, 18], [10, 56, 13, 10], [14, 80, 13, 14]],
     "gold": 140, "lives": 3, "bounty": 5},
    # seed 3013; 7 expansions, 6-decision plan
    {"path": [[0, 3], [0, 2], [0, 1], [1, 1], [1, 0], [2, 0], [3, 0], [3, 1], [4, 1], [5, 1], [5, 2], [6, 2], [7, 2], [7, 3], [7, 4], [8, 4], [8, 3], [9, 3], [10, 3], [11, 3]],
     "slots": [[0, 4], [4, 2], [6, 1], [6, 4], [7, 1], [8, 5]], "waves": [[4, 14, 10, 18], [8, 23, 11, 10], [9, 39, 13, 14], [12, 63, 13, 14]],
     "gold": 140, "lives": 2, "bounty": 4},
    # seed 3014; 9 expansions, 8-decision plan
    {"path": [[0, 5], [0, 6], [1, 6], [2, 6], [3, 6], [4, 6], [4, 7], [5, 7], [6, 7], [7, 7], [8, 7], [9, 7], [10, 7], [10, 6], [10, 5], [11, 5]],
     "slots": [[1, 7], [2, 5], [4, 5], [5, 6], [9, 5], [9, 6]], "waves": [[5, 11, 9, 14], [6, 18, 12, 10], [8, 27, 14, 18], [10, 44, 15, 18], [14, 69, 16, 18]],
     "gold": 100, "lives": 3, "bounty": 4},
    # seed 3015; 8 expansions, 7-decision plan
    {"path": [[0, 5], [1, 5], [2, 5], [2, 6], [3, 6], [4, 6], [5, 6], [6, 6], [6, 7], [7, 7], [8, 7], [9, 7], [10, 7], [11, 7]],
     "slots": [[0, 4], [1, 4], [2, 7], [3, 5], [4, 5], [4, 7], [10, 6], [11, 6]], "waves": [[4, 15, 9, 14], [8, 21, 12, 14], [9, 36, 14, 18], [11, 53, 16, 14]],
     "gold": 100, "lives": 4, "bounty": 5},
    # seed 3016; 90 expansions, 9-decision plan
    {"path": [[0, 5], [1, 5], [2, 5], [2, 6], [3, 6], [3, 7], [4, 7], [4, 6], [5, 6], [5, 7], [6, 7], [7, 7], [8, 7], [9, 7], [9, 6], [10, 6], [10, 7], [11, 7]],
     "slots": [[0, 6], [2, 4], [2, 7], [3, 5], [6, 6], [8, 6], [9, 5], [11, 6]], "waves": [[5, 13, 12, 10], [8, 22, 13, 10], [9, 32, 14, 18], [12, 54, 17, 14], [14, 84, 18, 18]],
     "gold": 120, "lives": 3, "bounty": 4},
    # seed 3017; 191 expansions, 7-decision plan
    {"path": [[0, 5], [1, 5], [2, 5], [2, 6], [2, 7], [3, 7], [3, 6], [3, 5], [4, 5], [5, 5], [6, 5], [7, 5], [7, 4], [8, 4], [9, 4], [9, 3], [10, 3], [11, 3]],
     "slots": [[0, 4], [1, 6], [5, 4], [7, 3], [8, 3], [9, 2], [10, 4], [11, 4]], "waves": [[6, 11, 10, 10], [6, 18, 12, 14], [9, 28, 12, 14], [10, 50, 14, 10], [13, 87, 15, 18]],
     "gold": 140, "lives": 2, "bounty": 4},
    # seed 3018; 9 expansions, 8-decision plan
    {"path": [[0, 3], [1, 3], [1, 4], [2, 4], [3, 4], [3, 5], [3, 6], [4, 6], [5, 6], [6, 6], [6, 5], [7, 5], [8, 5], [8, 4], [9, 4], [10, 4], [11, 4]],
     "slots": [[0, 4], [1, 5], [4, 4], [8, 6], [10, 3], [11, 3]], "waves": [[5, 11, 12, 10], [6, 16, 12, 14], [8, 27, 15, 18], [10, 42, 15, 18], [12, 65, 18, 14]],
     "gold": 120, "lives": 3, "bounty": 6},
    # seed 3019; 54 expansions, 7-decision plan
    {"path": [[0, 4], [1, 4], [2, 4], [2, 5], [3, 5], [4, 5], [5, 5], [6, 5], [7, 5], [7, 6], [8, 6], [9, 6], [10, 6], [11, 6]],
     "slots": [[1, 3], [3, 6], [4, 6], [6, 6], [9, 5], [11, 5]], "waves": [[6, 16, 11, 14], [8, 25, 11, 10], [9, 36, 12, 10], [11, 64, 14, 10]],
     "gold": 140, "lives": 3, "bounty": 6},
    # seed 3020; 31 expansions, 8-decision plan
    {"path": [[0, 2], [0, 3], [1, 3], [2, 3], [3, 3], [3, 4], [3, 5], [4, 5], [5, 5], [6, 5], [7, 5], [7, 4], [8, 4], [8, 3], [9, 3], [10, 3], [10, 2], [11, 2]],
     "slots": [[1, 2], [2, 4], [2, 5], [4, 4], [7, 3], [10, 1], [11, 1]], "waves": [[5, 12, 9, 14], [6, 17, 10, 10], [8, 26, 11, 10], [10, 42, 14, 14], [14, 70, 15, 18]],
     "gold": 100, "lives": 4, "bounty": 4},
    # seed 3021; 191 expansions, 9-decision plan
    {"path": [[0, 3], [1, 3], [2, 3], [3, 3], [4, 3], [4, 2], [5, 2], [5, 3], [6, 3], [7, 3], [7, 4], [7, 5], [7, 6], [8, 6], [9, 6], [9, 5], [10, 5], [10, 6], [11, 6]],
     "slots": [[0, 4], [3, 4], [4, 1], [8, 4], [8, 7], [9, 7], [10, 7]], "waves": [[4, 16, 8, 10], [8, 26, 11, 18], [8, 42, 14, 10], [11, 65, 17, 14], [13, 102, 20, 14]],
     "gold": 140, "lives": 4, "bounty": 6},
    # seed 3022; 108 expansions, 7-decision plan
    {"path": [[0, 5], [0, 4], [1, 4], [1, 3], [2, 3], [3, 3], [4, 3], [4, 4], [5, 4], [6, 4], [7, 4], [8, 4], [9, 4], [10, 4], [11, 4]],
     "slots": [[1, 2], [3, 2], [4, 2], [5, 5], [7, 3], [7, 5]], "waves": [[4, 10, 10, 18], [7, 16, 11, 10], [10, 27, 14, 10], [10, 44, 17, 10]],
     "gold": 100, "lives": 3, "bounty": 5},
    # seed 3023; 8 expansions, 6-decision plan
    {"path": [[0, 3], [0, 2], [1, 2], [2, 2], [2, 3], [2, 4], [3, 4], [3, 5], [4, 5], [5, 5], [6, 5], [7, 5], [8, 5], [9, 5], [10, 5], [10, 4], [10, 3], [11, 3]],
     "slots": [[0, 4], [1, 1], [4, 4], [4, 6], [7, 6], [11, 2], [11, 4], [11, 5]], "waves": [[4, 10, 12, 10], [7, 14, 13, 14], [8, 19, 15, 10], [10, 27, 17, 18]],
     "gold": 100, "lives": 4, "bounty": 4},
    # seed 3024; 19 expansions, 5-decision plan
    {"path": [[0, 3], [1, 3], [1, 4], [2, 4], [3, 4], [3, 3], [3, 2], [4, 2], [5, 2], [5, 3], [6, 3], [6, 2], [7, 2], [8, 2], [9, 2], [10, 2], [11, 2]],
     "slots": [[1, 5], [2, 2], [2, 5], [8, 1], [9, 3], [10, 3], [11, 3]], "waves": [[4, 15, 11, 10], [6, 24, 11, 14], [9, 42, 11, 14]],
     "gold": 120, "lives": 2, "bounty": 4},
    # seed 3025; 7 expansions, 6-decision plan
    {"path": [[0, 4], [1, 4], [1, 5], [2, 5], [3, 5], [3, 6], [3, 7], [4, 7], [5, 7], [6, 7], [7, 7], [8, 7], [9, 7], [10, 7], [11, 7]],
     "slots": [[0, 5], [2, 4], [4, 5], [4, 6], [5, 6], [6, 6], [11, 6]], "waves": [[5, 16, 12, 10], [6, 27, 13, 18], [9, 46, 13, 10]],
     "gold": 120, "lives": 4, "bounty": 6},
    # seed 3026; 42 expansions, 9-decision plan
    {"path": [[0, 4], [0, 5], [0, 6], [0, 7], [1, 7], [2, 7], [3, 7], [4, 7], [5, 7], [6, 7], [7, 7], [8, 7], [9, 7], [10, 7], [11, 7]],
     "slots": [[1, 4], [1, 5], [3, 6], [5, 6], [6, 6], [7, 6]], "waves": [[5, 14, 11, 10], [6, 21, 11, 18], [8, 32, 13, 10], [11, 57, 15, 18], [12, 84, 16, 18]],
     "gold": 100, "lives": 3, "bounty": 5},
    # seed 3027; 7 expansions, 5-decision plan
    {"path": [[0, 5], [0, 4], [1, 4], [1, 3], [2, 3], [2, 4], [3, 4], [4, 4], [5, 4], [6, 4], [7, 4], [8, 4], [8, 5], [9, 5], [9, 4], [9, 3], [10, 3], [11, 3]],
     "slots": [[0, 6], [2, 2], [3, 3], [7, 5], [10, 2], [10, 4], [10, 5], [11, 4]], "waves": [[6, 14, 12, 18], [6, 20, 13, 14], [9, 33, 14, 18]],
     "gold": 100, "lives": 3, "bounty": 6},
    # seed 3028; 136 expansions, 6-decision plan
    {"path": [[0, 3], [1, 3], [2, 3], [3, 3], [3, 4], [4, 4], [5, 4], [5, 3], [6, 3], [7, 3], [8, 3], [9, 3], [10, 3], [11, 3]],
     "slots": [[3, 2], [4, 3], [4, 5], [7, 2], [7, 4], [8, 4], [10, 2]], "waves": [[6, 15, 10, 10], [6, 25, 11, 18], [8, 38, 14, 14], [12, 60, 15, 10]],
     "gold": 140, "lives": 3, "bounty": 5},
    # seed 3029; 26 expansions, 5-decision plan
    {"path": [[0, 3], [0, 2], [1, 2], [2, 2], [3, 2], [3, 3], [3, 4], [4, 4], [5, 4], [6, 4], [6, 3], [7, 3], [7, 4], [8, 4], [9, 4], [10, 4], [10, 5], [11, 5]],
     "slots": [[0, 4], [1, 3], [2, 1], [3, 1], [5, 3], [5, 5], [8, 5], [9, 3]], "waves": [[6, 13, 12, 10], [8, 22, 15, 10], [8, 31, 17, 10]],
     "gold": 100, "lives": 4, "bounty": 5},
    # seed 3030; 65 expansions, 8-decision plan
    {"path": [[0, 3], [0, 4], [1, 4], [1, 3], [2, 3], [3, 3], [4, 3], [4, 4], [5, 4], [6, 4], [7, 4], [8, 4], [8, 5], [9, 5], [9, 6], [10, 6], [11, 6]],
     "slots": [[0, 5], [1, 2], [4, 2], [4, 5], [5, 5], [6, 3]], "waves": [[4, 15, 9, 14], [7, 24, 9, 10], [10, 36, 9, 18], [10, 56, 10, 10], [12, 87, 10, 14]],
     "gold": 140, "lives": 3, "bounty": 5},
    # seed 3031; 9 expansions, 5-decision plan
    {"path": [[0, 2], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [5, 0], [6, 0], [7, 0], [8, 0], [8, 1], [8, 2], [9, 2], [9, 3], [10, 3], [11, 3]],
     "slots": [[0, 3], [1, 2], [3, 2], [4, 0], [6, 1], [8, 3], [9, 1], [11, 4]], "waves": [[6, 10, 12, 18], [7, 14, 14, 18], [10, 22, 16, 10]],
     "gold": 120, "lives": 2, "bounty": 4},
    # seed 3032; 9 expansions, 8-decision plan
    {"path": [[0, 2], [0, 1], [1, 1], [2, 1], [2, 2], [2, 3], [3, 3], [3, 4], [4, 4], [5, 4], [6, 4], [7, 4], [7, 5], [8, 5], [9, 5], [9, 4], [10, 4], [11, 4]],
     "slots": [[1, 0], [4, 5], [5, 3], [9, 3], [9, 6], [10, 5]], "waves": [[4, 16, 9, 18], [7, 28, 9, 14], [10, 42, 10, 18], [11, 62, 10, 10], [14, 102, 11, 18]],
     "gold": 140, "lives": 3, "bounty": 6},
    # seed 3033; 15 expansions, 7-decision plan
    {"path": [[0, 4], [1, 4], [1, 3], [2, 3], [2, 4], [3, 4], [3, 5], [3, 6], [3, 7], [4, 7], [5, 7], [6, 7], [7, 7], [8, 7], [9, 7], [10, 7], [11, 7]],
     "slots": [[0, 3], [0, 5], [2, 7], [4, 5], [4, 6], [9, 6], [11, 6]], "waves": [[6, 13, 11, 14], [7, 22, 14, 14], [8, 30, 17, 18], [12, 50, 20, 14]],
     "gold": 120, "lives": 2, "bounty": 4},
    # seed 3034; 58 expansions, 6-decision plan
    {"path": [[0, 5], [1, 5], [2, 5], [3, 5], [4, 5], [4, 6], [5, 6], [5, 7], [6, 7], [7, 7], [8, 7], [9, 7], [10, 7], [11, 7]],
     "slots": [[0, 6], [2, 6], [3, 4], [4, 4], [7, 6], [11, 6]], "waves": [[5, 16, 8, 10], [7, 23, 8, 18], [9, 32, 8, 10], [10, 52, 9, 10]],
     "gold": 140, "lives": 4, "bounty": 4},
    # seed 3035; 10 expansions, 9-decision plan
    {"path": [[0, 4], [0, 5], [1, 5], [2, 5], [3, 5], [4, 5], [5, 5], [6, 5], [7, 5], [7, 4], [7, 3], [8, 3], [8, 2], [8, 1], [8, 0], [9, 0], [10, 0], [11, 0]],
     "slots": [[2, 4], [4, 4], [4, 6], [5, 4], [7, 2], [8, 5], [11, 1]], "waves": [[4, 13, 11, 14], [6, 20, 12, 14], [9, 31, 12, 14], [10, 49, 13, 14], [13, 80, 15, 14]],
     "gold": 100, "lives": 4, "bounty": 6},
    # seed 3036; 338 expansions, 7-decision plan
    {"path": [[0, 2], [0, 1], [1, 1], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0]],
     "slots": [[0, 0], [0, 3], [2, 1], [4, 1], [5, 1], [10, 1], [11, 1]], "waves": [[6, 16, 11, 18], [7, 28, 13, 18], [10, 43, 14, 14], [10, 75, 16, 10]],
     "gold": 140, "lives": 4, "bounty": 6},
    # seed 3037; 82 expansions, 9-decision plan
    {"path": [[0, 5], [0, 4], [0, 3], [1, 3], [2, 3], [3, 3], [3, 4], [4, 4], [4, 5], [4, 6], [5, 6], [5, 7], [6, 7], [7, 7], [8, 7], [8, 6], [9, 6], [10, 6], [10, 7], [11, 7]],
     "slots": [[0, 2], [0, 6], [3, 2], [4, 3], [8, 5], [9, 7], [11, 6]], "waves": [[6, 15, 11, 14], [6, 24, 13, 18], [8, 36, 14, 14], [10, 55, 15, 18], [14, 91, 16, 18]],
     "gold": 100, "lives": 4, "bounty": 5},
    # seed 3038; 276 expansions, 8-decision plan
    {"path": [[0, 3], [0, 2], [1, 2], [1, 3], [1, 4], [2, 4], [2, 5], [3, 5], [4, 5], [5, 5], [6, 5], [6, 4], [7, 4], [7, 3], [8, 3], [9, 3], [9, 2], [9, 1], [10, 1], [11, 1]],
     "slots": [[1, 1], [1, 5], [2, 3], [2, 6], [3, 6], [4, 6], [8, 1], [10, 3]], "waves": [[6, 13, 9, 10], [8, 22, 10, 10], [9, 32, 10, 10], [10, 57, 13, 18], [14, 87, 15, 14]],
     "gold": 140, "lives": 2, "bounty": 4},
    # seed 3039; 7 expansions, 5-decision plan
    {"path": [[0, 5], [0, 6], [0, 7], [1, 7], [1, 6], [2, 6], [3, 6], [3, 5], [4, 5], [4, 4], [4, 3], [5, 3], [5, 4], [5, 5], [6, 5], [7, 5], [7, 6], [8, 6], [8, 5], [8, 4], [9, 4], [10, 4], [10, 5], [11, 5]],
     "slots": [[0, 4], [5, 2], [5, 6], [6, 4], [9, 6], [10, 6]], "waves": [[5, 10, 10, 18], [8, 15, 13, 10], [9, 25, 16, 14]],
     "gold": 100, "lives": 2, "bounty": 4},
    # seed 3040; 9 expansions, 8-decision plan
    {"path": [[0, 2], [0, 1], [1, 1], [2, 1], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [8, 1], [9, 1], [9, 2], [10, 2], [11, 2]],
     "slots": [[0, 0], [0, 3], [1, 0], [3, 1], [6, 1], [9, 0], [11, 1], [11, 3]], "waves": [[5, 16, 9, 10], [8, 27, 10, 10], [10, 42, 10, 14], [10, 68, 10, 18]],
     "gold": 120, "lives": 2, "bounty": 6},
    # seed 3041; 13 expansions, 5-decision plan
    {"path": [[0, 5], [1, 5], [2, 5], [3, 5], [4, 5], [5, 5], [5, 4], [6, 4], [7, 4], [7, 5], [7, 6], [7, 7], [8, 7], [9, 7], [10, 7], [11, 7]],
     "slots": [[0, 4], [0, 6], [2, 6], [5, 6], [6, 5], [7, 3]], "waves": [[6, 16, 12, 10], [7, 23, 14, 14], [8, 39, 14, 14]],
     "gold": 100, "lives": 2, "bounty": 6},
    # seed 3042; 29 expansions, 7-decision plan
    {"path": [[0, 3], [1, 3], [1, 2], [2, 2], [3, 2], [4, 2], [4, 1], [5, 1], [6, 1], [7, 1], [7, 2], [8, 2], [9, 2], [9, 3], [10, 3], [10, 2], [11, 2]],
     "slots": [[1, 4], [4, 0], [4, 3], [5, 2], [6, 0], [7, 0], [8, 3]], "waves": [[6, 13, 12, 14], [7, 21, 15, 14], [9, 31, 17, 18], [11, 53, 18, 14]],
     "gold": 100, "lives": 4, "bounty": 6},
    # seed 3043; 138 expansions, 8-decision plan
    {"path": [[0, 3], [1, 3], [2, 3], [2, 4], [3, 4], [4, 4], [5, 4], [6, 4], [7, 4], [7, 3], [8, 3], [9, 3], [9, 4], [10, 4], [11, 4]],
     "slots": [[0, 2], [1, 2], [2, 2], [2, 5], [4, 5], [6, 3], [7, 5], [11, 5]], "waves": [[5, 10, 10, 10], [6, 15, 12, 18], [9, 24, 13, 10], [10, 37, 16, 14], [12, 60, 16, 10]],
     "gold": 140, "lives": 2, "bounty": 4},
    # seed 3044; 7 expansions, 5-decision plan
    {"path": [[0, 4], [1, 4], [1, 5], [2, 5], [3, 5], [4, 5], [4, 4], [5, 4], [6, 4], [6, 3], [6, 2], [7, 2], [8, 2], [9, 2], [10, 2], [10, 1], [11, 1]],
     "slots": [[3, 4], [6, 1], [7, 4], [9, 1], [9, 3], [10, 3]], "waves": [[5, 11, 10, 14], [6, 18, 11, 14], [9, 29, 13, 10]],
     "gold": 100, "lives": 3, "bounty": 4},
    # seed 3045; 11 expansions, 5-decision plan
    {"path": [[0, 2], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [4, 2], [4, 3], [4, 4], [5, 4], [5, 3], [6, 3], [6, 2], [7, 2], [7, 3], [8, 3], [8, 4], [9, 4], [10, 4], [11, 4]],
     "slots": [[0, 3], [3, 4], [5, 5], [6, 4], [10, 3], [11, 3], [11, 5]], "waves": [[4, 12, 9, 14], [6, 18, 12, 10], [10, 29, 12, 18]],
     "gold": 100, "lives": 4, "bounty": 4},
    # seed 3046; 20 expansions, 9-decision plan
    {"path": [[0, 3], [1, 3], [1, 4], [2, 4], [2, 5], [3, 5], [3, 6], [4, 6], [4, 5], [5, 5], [6, 5], [7, 5], [8, 5], [9, 5], [10, 5], [10, 4], [10, 3], [11, 3]],
     "slots": [[0, 2], [1, 2], [2, 6], [3, 4], [6, 4], [7, 6], [9, 4], [10, 2]], "waves": [[4, 11, 12, 14], [8, 19, 12, 14], [8, 29, 15, 10], [11, 43, 18, 14], [12, 65, 20, 14]],
     "gold": 100, "lives": 4, "bounty": 5},
    # seed 3047; 35 expansions, 8-decision plan
    {"path": [[0, 4], [0, 3], [1, 3], [2, 3], [3, 3], [3, 4], [4, 4], [4, 5], [5, 5], [6, 5], [7, 5], [7, 4], [8, 4], [9, 4], [10, 4], [11, 4]],
     "slots": [[2, 2], [4, 6], [5, 6], [7, 6], [8, 3], [9, 3]], "waves": [[5, 11, 8, 10], [6, 16, 8, 10], [8, 25, 8, 10], [11, 39, 8, 14], [12, 66, 10, 10]],
     "gold": 140, "lives": 4, "bounty": 4},
    # seed 3048; 71 expansions, 6-decision plan
    {"path": [[0, 5], [1, 5], [2, 5], [3, 5], [4, 5], [4, 4], [5, 4], [5, 5], [6, 5], [7, 5], [8, 5], [9, 5], [10, 5], [10, 6], [11, 6]],
     "slots": [[0, 4], [2, 6], [3, 4], [4, 3], [6, 6], [8, 4], [11, 5], [11, 7]], "waves": [[5, 13, 11, 18], [8, 22, 11, 10], [9, 33, 12, 14], [12, 46, 13, 10]],
     "gold": 140, "lives": 3, "bounty": 4},
    # seed 3049; 6 expansions, 5-decision plan
    {"path": [[0, 5], [0, 4], [1, 4], [2, 4], [2, 3], [3, 3], [3, 2], [4, 2], [5, 2], [5, 3], [6, 3], [6, 2], [7, 2], [8, 2], [8, 1], [9, 1], [9, 0], [10, 0], [11, 0]],
     "slots": [[0, 6], [1, 3], [2, 2], [2, 5], [3, 1], [3, 4], [7, 1], [8, 3]], "waves": [[4, 15, 12, 18], [8, 22, 13, 18], [9, 31, 15, 18]],
     "gold": 100, "lives": 3, "bounty": 4},
    # seed 3050; 7 expansions, 5-decision plan
    {"path": [[0, 5], [1, 5], [1, 4], [2, 4], [3, 4], [4, 4], [4, 3], [5, 3], [5, 2], [6, 2], [7, 2], [8, 2], [8, 1], [9, 1], [10, 1], [11, 1]],
     "slots": [[5, 1], [6, 1], [7, 3], [8, 3], [10, 0], [10, 2]], "waves": [[6, 15, 9, 14], [6, 22, 11, 14], [9, 31, 14, 10]],
     "gold": 100, "lives": 4, "bounty": 4},
    # seed 3051; 94 expansions, 8-decision plan
    {"path": [[0, 2], [0, 3], [0, 4], [1, 4], [1, 3], [2, 3], [3, 3], [3, 4], [4, 4], [5, 4], [5, 5], [6, 5], [7, 5], [7, 4], [8, 4], [9, 4], [10, 4], [10, 3], [11, 3]],
     "slots": [[0, 1], [2, 2], [2, 4], [3, 2], [7, 3], [9, 3]], "waves": [[5, 12, 8, 18], [8, 18, 8, 14], [8, 31, 11, 10], [12, 45, 13, 18], [12, 77, 13, 14]],
     "gold": 140, "lives": 4, "bounty": 4},
    # seed 3052; 9 expansions, 8-decision plan
    {"path": [[0, 2], [0, 1], [1, 1], [1, 2], [2, 2], [2, 3], [2, 4], [3, 4], [4, 4], [4, 5], [5, 5], [5, 4], [6, 4], [7, 4], [8, 4], [9, 4], [10, 4], [10, 3], [11, 3]],
     "slots": [[1, 3], [3, 3], [5, 3], [7, 3], [8, 3], [10, 5], [11, 4]], "waves": [[5, 12, 10, 14], [6, 21, 13, 18], [8, 31, 16, 18], [10, 45, 17, 18], [14, 77, 17, 14]],
     "gold": 120, "lives": 2, "bounty": 5},
    # seed 3053; 15 expansions, 7-decision plan
    {"path": [[0, 3], [1, 3], [1, 4], [2, 4], [3, 4], [4, 4], [5, 4], [5, 3], [6, 3], [7, 3], [8, 3], [8, 2], [9, 2], [10, 2], [10, 3], [11, 3]],
     "slots": [[0, 4], [1, 2], [5, 2], [5, 5], [6, 4], [7, 2]], "waves": [[5, 14, 12, 10], [7, 24, 15, 18], [10, 35, 18, 10], [11, 58, 18, 18]],
     "gold": 140, "lives": 2, "bounty": 4},
    # seed 3054; 14 expansions, 7-decision plan
    {"path": [[0, 2], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [6, 2], [7, 2], [7, 3], [8, 3], [9, 3], [10, 3], [10, 2], [10, 1], [11, 1]],
     "slots": [[0, 0], [0, 3], [3, 2], [4, 0], [6, 0], [9, 2]], "waves": [[5, 11, 12, 14], [6, 17, 15, 14], [10, 23, 15, 14], [10, 38, 18, 10]],
     "gold": 100, "lives": 3, "bounty": 4},
    # seed 3055; 43 expansions, 6-decision plan
    {"path": [[0, 2], [0, 1], [1, 1], [1, 0], [2, 0], [3, 0], [3, 1], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2], [8, 1], [8, 0], [9, 0], [10, 0], [11, 0]],
     "slots": [[1, 2], [2, 1], [3, 3], [4, 0], [5, 3], [8, 3]], "waves": [[5, 15, 11, 18], [7, 24, 14, 14], [8, 40, 15, 18], [10, 56, 18, 10]],
     "gold": 140, "lives": 4, "bounty": 4},
    # seed 3056; 207 expansions, 8-decision plan
    {"path": [[0, 2], [1, 2], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1], [8, 1], [8, 0], [9, 0], [10, 0], [11, 0]],
     "slots": [[0, 3], [2, 0], [3, 2], [5, 0], [5, 2], [6, 0], [6, 2], [10, 1]], "waves": [[6, 13, 12, 10], [7, 20, 12, 14], [10, 29, 13, 18], [12, 51, 13, 10], [14, 75, 15, 14]],
     "gold": 140, "lives": 4, "bounty": 4},
    # seed 3057; 58 expansions, 4-decision plan
    {"path": [[0, 2], [1, 2], [2, 2], [3, 2], [3, 3], [4, 3], [5, 3], [6, 3], [7, 3], [8, 3], [9, 3], [9, 2], [10, 2], [10, 3], [11, 3]],
     "slots": [[0, 1], [1, 3], [2, 1], [3, 1], [5, 2], [5, 4], [10, 1]], "waves": [[6, 16, 11, 18], [7, 28, 12, 10], [8, 47, 15, 10]],
     "gold": 140, "lives": 3, "bounty": 5},
    # seed 3058; 10 expansions, 9-decision plan
    {"path": [[0, 2], [1, 2], [2, 2], [3, 2], [4, 2], [4, 3], [5, 3], [6, 3], [7, 3], [7, 2], [8, 2], [8, 1], [9, 1], [10, 1], [11, 1]],
     "slots": [[0, 1], [0, 3], [2, 1], [5, 4], [7, 4], [10, 2]], "waves": [[5, 16, 12, 10], [8, 24, 15, 10], [9, 34, 15, 10], [10, 49, 15, 14], [12, 86, 18, 14]],
     "gold": 140, "lives": 2, "bounty": 6},
    # seed 3059; 7 expansions, 5-decision plan
    {"path": [[0, 3], [1, 3], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [5, 3], [6, 3], [7, 3], [7, 4], [8, 4], [8, 5], [9, 5], [10, 5], [11, 5]],
     "slots": [[0, 2], [1, 1], [3, 1], [9, 4], [10, 6], [11, 4]], "waves": [[5, 16, 8, 10], [8, 25, 9, 14], [9, 40, 10, 14]],
     "gold": 100, "lives": 3, "bounty": 6},
    # seed 3060; 60 expansions, 8-decision plan
    {"path": [[0, 2], [1, 2], [1, 3], [2, 3], [2, 2], [3, 2], [4, 2], [5, 2], [5, 3], [5, 4], [5, 5], [6, 5], [7, 5], [8, 5], [8, 4], [9, 4], [9, 3], [9, 2], [10, 2], [11, 2]],
     "slots": [[1, 4], [2, 1], [6, 4], [9, 5], [10, 1], [10, 4]], "waves": [[6, 16, 10, 18], [8, 26, 11, 14], [9, 41, 11, 10], [12, 58, 12, 10], [14, 103, 12, 18]],
     "gold": 140, "lives": 4, "bounty": 5},
    # seed 3061; 80 expansions, 8-decision plan
    {"path": [[0, 4], [1, 4], [1, 3], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2], [7, 1], [8, 1], [8, 2], [9, 2], [9, 3], [10, 3], [11, 3]],
     "slots": [[1, 5], [2, 3], [3, 3], [4, 1], [4, 3], [7, 0], [11, 2]], "waves": [[6, 14, 12, 14], [6, 21, 14, 14], [10, 32, 17, 18], [12, 52, 19, 14], [12, 89, 20, 18]],
     "gold": 140, "lives": 2, "bounty": 4},
    # seed 3062; 10 expansions, 6-decision plan
    {"path": [[0, 3], [0, 2], [1, 2], [2, 2], [3, 2], [3, 1], [4, 1], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0]],
     "slots": [[1, 1], [3, 3], [5, 1], [7, 1], [8, 1], [10, 1]], "waves": [[4, 14, 11, 14], [6, 22, 11, 14], [8, 32, 12, 18], [12, 45, 14, 18]],
     "gold": 100, "lives": 3, "bounty": 4},
    # seed 3063; 41 expansions, 8-decision plan
    {"path": [[0, 2], [0, 3], [1, 3], [2, 3], [2, 4], [3, 4], [4, 4], [5, 4], [6, 4], [7, 4], [8, 4], [8, 5], [9, 5], [9, 6], [10, 6], [11, 6]],
     "slots": [[0, 4], [1, 2], [2, 5], [5, 5], [8, 6], [9, 7]], "waves": [[6, 12, 9, 18], [8, 21, 11, 10], [9, 34, 11, 10], [12, 57, 14, 14], [14, 86, 15, 14]],
     "gold": 140, "lives": 3, "bounty": 4},
    # seed 3064; 9 expansions, 5-decision plan
    {"path": [[0, 4], [1, 4], [2, 4], [2, 5], [3, 5], [4, 5], [4, 6], [4, 7], [5, 7], [5, 6], [5, 5], [5, 4], [6, 4], [7, 4], [7, 5], [8, 5], [8, 6], [9, 6], [10, 6], [11, 6]],
     "slots": [[2, 3], [2, 6], [7, 3], [9, 5], [9, 7], [10, 7]], "waves": [[4, 12, 11, 10], [7, 21, 14, 18], [10, 36, 17, 18]],
     "gold": 100, "lives": 4, "bounty": 6},
    # seed 3065; 22 expansions, 7-decision plan
    {"path": [[0, 2], [1, 2], [1, 3], [2, 3], [3, 3], [4, 3], [5, 3], [6, 3], [6, 2], [7, 2], [7, 1], [8, 1], [9, 1], [10, 1], [11, 1]],
     "slots": [[1, 1], [2, 2], [2, 4], [3, 4], [8, 2], [9, 2], [11, 2]], "waves": [[6, 16, 9, 14], [8, 22, 12, 18], [9, 31, 14, 10], [10, 49, 16, 18]],
     "gold": 120, "lives": 3, "bounty": 4},
    # seed 3066; 26 expansions, 10-decision plan
    {"path": [[0, 5], [1, 5], [1, 6], [2, 6], [3, 6], [3, 7], [4, 7], [5, 7], [6, 7], [7, 7], [7, 6], [8, 6], [8, 5], [8, 4], [9, 4], [9, 3], [9, 2], [10, 2], [11, 2]],
     "slots": [[0, 4], [1, 4], [2, 5], [2, 7], [8, 3], [10, 3], [11, 1]], "waves": [[4, 16, 11, 14], [6, 22, 13, 10], [9, 33, 15, 14], [11, 59, 16, 18], [12, 82, 17, 14]],
     "gold": 120, "lives": 3, "bounty": 6},
    # seed 3067; 221 expansions, 6-decision plan
    {"path": [[0, 3], [1, 3], [2, 3], [3, 3], [3, 2], [4, 2], [5, 2], [5, 3], [5, 4], [6, 4], [7, 4], [8, 4], [9, 4], [9, 5], [10, 5], [10, 6], [10, 7], [11, 7]],
     "slots": [[0, 4], [1, 4], [3, 1], [4, 3], [5, 5], [7, 3], [9, 3], [10, 4]], "waves": [[6, 12, 8, 10], [6, 20, 9, 14], [8, 34, 12, 14], [12, 58, 12, 10]],
     "gold": 140, "lives": 2, "bounty": 4},
    # seed 3068; 50 expansions, 6-decision plan
    {"path": [[0, 4], [0, 5], [0, 6], [0, 7], [1, 7], [2, 7], [3, 7], [4, 7], [5, 7], [5, 6], [6, 6], [6, 7], [7, 7], [8, 7], [9, 7], [10, 7], [10, 6], [11, 6]],
     "slots": [[0, 3], [1, 4], [1, 5], [2, 6], [5, 5], [6, 5], [8, 6]], "waves": [[5, 15, 8, 14], [8, 22, 11, 14], [8, 37, 14, 18], [10, 63, 14, 14]],
     "gold": 140, "lives": 3, "bounty": 4},
    # seed 3069; 8 expansions, 7-decision plan
    {"path": [[0, 4], [1, 4], [1, 5], [2, 5], [3, 5], [4, 5], [4, 6], [5, 6], [6, 6], [6, 7], [7, 7], [8, 7], [9, 7], [10, 7], [11, 7]],
     "slots": [[2, 6], [4, 4], [4, 7], [5, 7], [7, 6], [8, 6], [9, 6]], "waves": [[6, 13, 8, 18], [8, 21, 8, 18], [8, 36, 8, 14], [11, 60, 8, 14]],
     "gold": 120, "lives": 4, "bounty": 5},
    # seed 3070; 8 expansions, 6-decision plan
    {"path": [[0, 2], [0, 1], [1, 1], [2, 1], [3, 1], [3, 0], [4, 0], [5, 0], [5, 1], [6, 1], [6, 0], [7, 0], [8, 0], [8, 1], [9, 1], [9, 2], [9, 3], [9, 4], [10, 4], [10, 5], [10, 6], [11, 6]],
     "slots": [[0, 0], [0, 3], [4, 1], [8, 3], [9, 6], [11, 4], [11, 5]], "waves": [[6, 10, 10, 10], [7, 17, 13, 10], [10, 24, 15, 10], [12, 37, 17, 18]],
     "gold": 100, "lives": 3, "bounty": 5},
    # seed 3071; 8 expansions, 7-decision plan
    {"path": [[0, 3], [0, 2], [1, 2], [2, 2], [2, 1], [3, 1], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2], [7, 3], [8, 3], [9, 3], [10, 3], [10, 4], [10, 5], [11, 5]],
     "slots": [[0, 4], [1, 1], [1, 3], [2, 0], [3, 0], [8, 4]], "waves": [[5, 13, 9, 10], [7, 22, 9, 10], [10, 38, 12, 10], [10, 59, 12, 10]],
     "gold": 140, "lives": 2, "bounty": 6},
    # seed 3072; 10 expansions, 5-decision plan
    {"path": [[0, 4], [1, 4], [2, 4], [3, 4], [3, 3], [3, 2], [4, 2], [4, 1], [5, 1], [5, 2], [6, 2], [7, 2], [8, 2], [9, 2], [10, 2], [11, 2]],
     "slots": [[0, 3], [3, 1], [5, 3], [8, 1], [10, 3], [11, 3]], "waves": [[4, 15, 8, 18], [7, 22, 8, 10], [9, 35, 9, 10]],
     "gold": 100, "lives": 2, "bounty": 4},
    # seed 3073; 7 expansions, 6-decision plan
    {"path": [[0, 3], [1, 3], [2, 3], [2, 4], [3, 4], [4, 4], [4, 5], [5, 5], [5, 6], [6, 6], [7, 6], [8, 6], [9, 6], [10, 6], [10, 7], [11, 7]],
     "slots": [[0, 2], [1, 2], [2, 5], [7, 7], [8, 7], [10, 5]], "waves": [[6, 16, 10, 10], [8, 27, 11, 14], [8, 43, 11, 10]],
     "gold": 120, "lives": 2, "bounty": 5},
    # seed 3074; 111 expansions, 9-decision plan
    {"path": [[0, 3], [1, 3], [1, 4], [2, 4], [2, 3], [3, 3], [4, 3], [5, 3], [6, 3], [6, 2], [7, 2], [8, 2], [8, 1], [9, 1], [9, 2], [10, 2], [10, 3], [11, 3]],
     "slots": [[1, 5], [3, 2], [3, 4], [4, 4], [6, 4], [10, 4], [11, 2], [11, 4]], "waves": [[4, 15, 12, 10], [8, 23, 12, 10], [10, 32, 14, 18], [12, 46, 15, 14], [14, 73, 15, 14]],
     "gold": 100, "lives": 3, "bounty": 5},
    # seed 3075; 9 expansions, 8-decision plan
    {"path": [[0, 2], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [10, 1], [11, 1]],
     "slots": [[1, 2], [2, 2], [5, 2], [6, 1], [10, 2], [11, 0]], "waves": [[5, 11, 9, 14], [6, 17, 10, 14], [9, 26, 11, 14], [10, 39, 14, 18], [12, 61, 14, 10]],
     "gold": 100, "lives": 4, "bounty": 6},
    # seed 3076; 341 expansions, 7-decision plan
    {"path": [[0, 4], [1, 4], [2, 4], [3, 4], [3, 3], [4, 3], [5, 3], [5, 4], [6, 4], [7, 4], [7, 3], [7, 2], [8, 2], [9, 2], [10, 2], [11, 2]],
     "slots": [[3, 5], [4, 4], [6, 2], [6, 3], [8, 3], [9, 1], [10, 3], [11, 1]], "waves": [[4, 15, 8, 14], [7, 26, 9, 10], [10, 41, 12, 14], [10, 72, 13, 18], [12, 108, 16, 14]],
     "gold": 140, "lives": 3, "bounty": 5},
    # seed 3077; 10 expansions, 9-decision plan
    {"path": [[0, 4], [0, 3], [1, 3], [2, 3], [2, 2], [3, 2], [4, 2], [4, 1], [4, 0], [5, 0], [6, 0], [6, 1], [7, 1], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0]],
     "slots": [[0, 5], [2, 1], [3, 1], [5, 1], [6, 2], [8, 1]], "waves": [[5, 11, 9, 10], [7, 19, 10, 14], [10, 26, 10, 10], [12, 38, 12, 14], [14, 63, 15, 10]],
     "gold": 120, "lives": 2, "bounty": 6},
    # seed 3078; 221 expansions, 7-decision plan
    {"path": [[0, 2], [1, 2], [2, 2], [3, 2], [4, 2], [4, 1], [5, 1], [6, 1], [7, 1], [8, 1], [9, 1], [10, 1], [11, 1]],
     "slots": [[0, 1], [3, 3], [8, 0], [8, 2], [9, 2], [10, 2], [11, 0], [11, 2]], "waves": [[6, 16, 11, 18], [7, 27, 13, 18], [10, 46, 14, 10], [11, 69, 15, 10]],
     "gold": 140, "lives": 4, "bounty": 6},
    # seed 3079; 443 expansions, 8-decision plan
    {"path": [[0, 2], [0, 3], [1, 3], [2, 3], [3, 3], [3, 4], [4, 4], [5, 4], [6, 4], [6, 5], [7, 5], [7, 6], [7, 7], [8, 7], [8, 6], [9, 6], [9, 7], [10, 7], [11, 7]],
     "slots": [[1, 4], [3, 2], [4, 3], [6, 3], [6, 6], [6, 7], [8, 5]], "waves": [[6, 16, 9, 10], [8, 24, 11, 18], [9, 38, 14, 18], [11, 54, 17, 18], [12, 76, 17, 10]],
     "gold": 140, "lives": 4, "bounty": 4},
    # seed 3080; 75 expansions, 9-decision plan
    {"path": [[0, 5], [0, 6], [1, 6], [2, 6], [2, 7], [3, 7], [4, 7], [5, 7], [6, 7], [7, 7], [8, 7], [9, 7], [10, 7], [11, 7]],
     "slots": [[0, 4], [1, 5], [3, 6], [4, 6], [8, 6], [9, 6], [10, 6], [11, 6]], "waves": [[5, 15, 8, 18], [8, 21, 9, 14], [8, 32, 11, 18], [11, 57, 12, 14], [13, 90, 15, 18]],
     "gold": 120, "lives": 4, "bounty": 5},
    # seed 3081; 20 expansions, 7-decision plan
    {"path": [[0, 5], [0, 4], [1, 4], [1, 3], [2, 3], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [6, 1], [7, 1], [7, 2], [8, 2], [9, 2], [10, 2], [11, 2]],
     "slots": [[0, 6], [2, 4], [3, 1], [4, 3], [6, 0], [10, 1], [11, 3]], "waves": [[6, 16, 10, 10], [7, 27, 11, 10], [10, 46, 11, 18], [11, 72, 14, 18]],
     "gold": 120, "lives": 2, "bounty": 5},
    # seed 3082; 16 expansions, 7-decision plan
    {"path": [[0, 5], [1, 5], [1, 6], [2, 6], [3, 6], [3, 5], [4, 5], [4, 6], [5, 6], [5, 5], [6, 5], [7, 5], [8, 5], [9, 5], [9, 6], [10, 6], [10, 5], [10, 4], [11, 4]],
     "slots": [[2, 5], [3, 4], [5, 4], [5, 7], [6, 6], [8, 6], [10, 7], [11, 5]], "waves": [[4, 14, 11, 10], [7, 20, 12, 18], [10, 28, 15, 18], [11, 42, 18, 10]],
     "gold": 120, "lives": 3, "bounty": 4},
    # seed 3083; 17 expansions, 9-decision plan
    {"path": [[0, 3], [0, 2], [1, 2], [2, 2], [3, 2], [3, 3], [4, 3], [5, 3], [5, 2], [5, 1], [6, 1], [6, 0], [7, 0], [7, 1], [7, 2], [8, 2], [9, 2], [10, 2], [11, 2]],
     "slots": [[0, 4], [1, 3], [2, 3], [3, 1], [4, 1], [4, 2], [5, 4], [9, 1]], "waves": [[5, 14, 9, 18], [6, 24, 10, 18], [9, 33, 10, 10], [10, 51, 10, 14], [14, 71, 13, 18]],
     "gold": 120, "lives": 4, "bounty": 5},
    # seed 3084; 13 expansions, 7-decision plan
    {"path": [[0, 3], [0, 4], [1, 4], [2, 4], [3, 4], [3, 3], [3, 2], [4, 2], [4, 3], [5, 3], [5, 2], [6, 2], [7, 2], [7, 3], [8, 3], [9, 3], [10, 3], [11, 3]],
     "slots": [[0, 2], [0, 5], [3, 5], [4, 1], [4, 4], [6, 3], [9, 4]], "waves": [[4, 12, 11, 14], [6, 18, 12, 10], [10, 27, 14, 10], [10, 42, 15, 10]],
     "gold": 100, "lives": 4, "bounty": 5},
    # seed 3085; 10 expansions, 5-decision plan
    {"path": [[0, 4], [1, 4], [1, 5], [2, 5], [3, 5], [3, 6], [4, 6], [5, 6], [6, 6], [7, 6], [8, 6], [8, 5], [8, 4], [9, 4], [9, 5], [10, 5], [11, 5]],
     "slots": [[1, 6], [4, 5], [4, 7], [6, 5], [7, 7], [9, 3], [9, 6], [11, 4]], "waves": [[4, 13, 12, 10], [7, 20, 14, 18], [8, 30, 17, 10]],
     "gold": 100, "lives": 2, "bounty": 4},
    # seed 3086; 26 expansions, 7-decision plan
    {"path": [[0, 5], [0, 4], [1, 4], [2, 4], [3, 4], [4, 4], [5, 4], [5, 3], [6, 3], [6, 2], [6, 1], [7, 1], [7, 0], [8, 0], [9, 0], [10, 0], [11, 0]],
     "slots": [[0, 6], [2, 3], [2, 5], [3, 3], [5, 2], [9, 1], [11, 1]], "waves": [[6, 15, 10, 10], [7, 24, 13, 18], [10, 33, 14, 10], [12, 51, 16, 14]],
     "gold": 140, "lives": 2, "bounty": 4},
    # seed 3087; 7 expansions, 6-decision plan
    {"path": [[0, 2], [1, 2], [2, 2], [3, 2], [3, 3], [4, 3], [5, 3], [6, 3], [7, 3], [7, 4], [8, 4], [8, 3], [8, 2], [8, 1], [9, 1], [10, 1], [11, 1]],
     "slots": [[0, 3], [3, 1], [5, 4], [6, 4], [7, 1], [7, 5], [9, 0], [10, 2]], "waves": [[6, 16, 10, 14], [6, 28, 11, 18], [8, 45, 11, 10]],
     "gold": 120, "lives": 3, "bounty": 5},
    # seed 3088; 93 expansions, 6-decision plan
    {"path": [[0, 3], [1, 3], [2, 3], [3, 3], [3, 4], [4, 4], [5, 4], [6, 4], [7, 4], [7, 3], [7, 2], [8, 2], [9, 2], [9, 3], [10, 3], [10, 2], [11, 2]],
     "slots": [[0, 2], [2, 2], [5, 3], [5, 5], [6, 2], [7, 1]], "waves": [[6, 12, 10, 10], [7, 21, 11, 14], [8, 35, 12, 10], [11, 60, 12, 10]],
     "gold": 140, "lives": 4, "bounty": 5},
    # seed 3089; 39 expansions, 7-decision plan
    {"path": [[0, 4], [0, 5], [1, 5], [1, 6], [2, 6], [3, 6], [4, 6], [5, 6], [6, 6], [6, 7], [7, 7], [7, 6], [8, 6], [9, 6], [10, 6], [10, 5], [11, 5]],
     "slots": [[0, 3], [5, 5], [5, 7], [6, 5], [10, 7], [11, 6]], "waves": [[4, 14, 9, 10], [7, 20, 12, 14], [9, 29, 15, 10], [11, 41, 17, 10]],
     "gold": 120, "lives": 3, "bounty": 5},
    # seed 3090; 73 expansions, 7-decision plan
    {"path": [[0, 5], [1, 5], [2, 5], [2, 4], [3, 4], [4, 4], [4, 3], [5, 3], [6, 3], [7, 3], [8, 3], [8, 2], [8, 1], [8, 0], [9, 0], [10, 0], [11, 0]],
     "slots": [[1, 4], [4, 2], [6, 2], [7, 0], [8, 4], [9, 2], [9, 3], [11, 1]], "waves": [[4, 16, 11, 18], [6, 22, 12, 10], [8, 34, 12, 18], [10, 48, 13, 10], [13, 70, 15, 18]],
     "gold": 140, "lives": 3, "bounty": 4},
    # seed 3091; 7 expansions, 5-decision plan
    {"path": [[0, 4], [0, 3], [1, 3], [2, 3], [3, 3], [4, 3], [4, 4], [5, 4], [5, 3], [6, 3], [6, 2], [7, 2], [8, 2], [9, 2], [9, 3], [9, 4], [10, 4], [11, 4]],
     "slots": [[0, 2], [0, 5], [1, 4], [5, 2], [5, 5], [7, 3], [9, 1], [10, 2]], "waves": [[4, 13, 8, 14], [7, 20, 9, 18], [8, 33, 10, 10]],
     "gold": 100, "lives": 3, "bounty": 4},
    # seed 3092; 7 expansions, 5-decision plan
    {"path": [[0, 5], [0, 4], [0, 3], [0, 2], [0, 1], [1, 1], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [6, 1], [7, 1], [7, 2], [8, 2], [9, 2], [10, 2], [11, 2]],
     "slots": [[1, 4], [4, 1], [6, 2], [7, 0], [8, 3], [9, 3]], "waves": [[6, 14, 10, 10], [8, 23, 11, 10], [8, 32, 14, 10]],
     "gold": 120, "lives": 3, "bounty": 4},
    # seed 3093; 16 expansions, 9-decision plan
    {"path": [[0, 5], [1, 5], [1, 4], [1, 3], [2, 3], [3, 3], [3, 2], [4, 2], [5, 2], [6, 2], [6, 3], [7, 3], [7, 2], [8, 2], [8, 1], [9, 1], [10, 1], [11, 1]],
     "slots": [[0, 4], [2, 4], [3, 4], [4, 3], [5, 3], [6, 4], [7, 4], [8, 0]], "waves": [[6, 12, 8, 10], [6, 19, 8, 18], [10, 31, 11, 18], [11, 54, 13, 18], [12, 95, 14, 18]],
     "gold": 120, "lives": 3, "bounty": 4},
    # seed 3094; 65 expansions, 8-decision plan
    {"path": [[0, 2], [1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [4, 4], [4, 3], [5, 3], [5, 4], [5, 5], [6, 5], [6, 6], [7, 6], [7, 5], [8, 5], [9, 5], [10, 5], [10, 4], [11, 4]],
     "slots": [[0, 1], [2, 2], [3, 3], [6, 3], [10, 3], [11, 5]], "waves": [[5, 11, 12, 10], [7, 19, 15, 14], [8, 30, 18, 14], [12, 44, 19, 10], [12, 68, 19, 10]],
     "gold": 140, "lives": 4, "bounty": 6},
    # seed 3095; 14 expansions, 6-decision plan
    {"path": [[0, 4], [1, 4], [2, 4], [3, 4], [3, 3], [4, 3], [5, 3], [5, 4], [6, 4], [7, 4], [8, 4], [9, 4], [9, 3], [10, 3], [10, 2], [11, 2]],
     "slots": [[0, 5], [1, 3], [7, 5], [8, 3], [8, 5], [9, 2], [10, 4]], "waves": [[4, 11, 11, 14], [8, 19, 11, 10], [8, 28, 13, 10], [12, 48, 16, 18]],
     "gold": 100, "lives": 4, "bounty": 4},
    # seed 3096; 104 expansions, 6-decision plan
    {"path": [[0, 5], [1, 5], [1, 4], [1, 3], [1, 2], [2, 2], [2, 1], [3, 1], [3, 0], [4, 0], [4, 1], [5, 1], [6, 1], [7, 1], [8, 1], [9, 1], [10, 1], [11, 1]],
     "slots": [[0, 2], [5, 0], [5, 2], [6, 0], [7, 2], [8, 0], [8, 2]], "waves": [[4, 13, 8, 14], [7, 22, 9, 10], [10, 38, 9, 18], [11, 64, 9, 10]],
     "gold": 140, "lives": 3, "bounty": 4},
    # seed 3097; 433 expansions, 9-decision plan
    {"path": [[0, 2], [1, 2], [2, 2], [2, 1], [3, 1], [4, 1], [5, 1], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0], [10, 0], [10, 1], [10, 2], [11, 2]],
     "slots": [[1, 3], [5, 2], [6, 1], [7, 1], [8, 1], [9, 1], [10, 3]], "waves": [[5, 16, 9, 10], [7, 26, 9, 14], [10, 40, 11, 18], [10, 62, 11, 10], [14, 92, 13, 10]],
     "gold": 140, "lives": 2, "bounty": 6},
    # seed 3098; 32 expansions, 6-decision plan
    {"path": [[0, 5], [0, 6], [0, 7], [1, 7], [2, 7], [3, 7], [4, 7], [5, 7], [6, 7], [6, 6], [6, 5], [7, 5], [8, 5], [8, 4], [9, 4], [10, 4], [11, 4]],
     "slots": [[0, 4], [1, 5], [5, 5], [7, 7], [9, 3], [10, 3], [10, 5]], "waves": [[5, 16, 9, 10], [8, 27, 11, 18], [9, 47, 11, 10]],
     "gold": 140, "lives": 3, "bounty": 4},
    # seed 3099; 52 expansions, 10-decision plan
    {"path": [[0, 5], [1, 5], [2, 5], [3, 5], [3, 4], [4, 4], [5, 4], [6, 4], [7, 4], [7, 5], [8, 5], [8, 4], [9, 4], [10, 4], [10, 5], [11, 5]],
     "slots": [[0, 6], [2, 6], [5, 3], [6, 5], [7, 6], [9, 5], [10, 3]], "waves": [[6, 16, 12, 10], [8, 26, 14, 18], [8, 38, 16, 18], [11, 57, 18, 18], [13, 87, 20, 14]],
     "gold": 140, "lives": 2, "bounty": 5},
)
