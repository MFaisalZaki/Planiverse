"""An early-game factory on factory-sim: walk to the ore, place burner drills and stone furnaces
on it, fuel them, and bring the iron plates the goal asks for back before the deadline.

factory-sim (https://github.com/divagr18/factory-sim, MIT) is a C simulator of a small slice of
Factorio's early game, checked decision by decision and tick by tick against traces recorded
from the game itself (Factorio 2.0.60, through its FactorioRL harness). It has none of the
game in it: every rule and number was measured on the running game and written down, and no
code, data, art or sound of Factorio's is copied. Factorio is a game by Wube Software Ltd and
"Factorio" is Wube's trademark; neither factory-sim nor this environment is affiliated with or
endorsed by Wube. The simulator has to be built from its source (`scripts/build_factory_sim.sh`)
and nothing of it is included here.

The game is the simulator's. A character stands near a patch of iron ore with a few burner
mining drills, stone furnaces and some coal. A decision is thirty ticks (half a second of the
game): walk, place a drill on the ore or a furnace where a drill drops its ore, hand a machine
coal, mine ore by hand, take the plates out of a furnace, or wait. A drill burns coal at 150 kW
and lifts one ore every four seconds; a furnace burns it at 90 kW and smelts a plate in 3.2
seconds; a piece of coal is four megajoules; the hand reaches ten tiles for machines and 2.7 for
ore. How much coal to give which machine, how many lines to build with what is carried, where
to stand so that everything is in reach, and how long to wait before the plates are worth
collecting is only known by running the ticks, which is why the environment is here.

## Determinism and state

The simulator is deterministic and its whole state is one C struct with no pointers in it, so
the environment snapshots it as bytes, restores it to expand a state, and replays a state's
decisions from the start when it has no snapshot of it. A state is the contents that decide
its future (tick, position, inventory, every machine's fuel, energy and progress, the ore
left and the piles on the ground), not the decisions that led to it, so two orders of the same
decisions that leave the same factory are one state.

## Instances and generation

`generate_instance(seed, ...)` draws a scene with factory-sim's own scene generator (a patch of
ore, walls in some families, a starting position, all drawn the way FactorioRL draws them),
draws what the character carries and a horizon, and measures a few scripted lines on it: the
nearest place to build one line, or two, and each way of splitting the coal between drills and
furnaces. The target is the most plates any of them brings back by the horizon, that plan is
left in `witness`, and a scene none of them can work is thrown back. The bundled patches are
such draws, embedded as plain data with the seed each came from. The method is
generate-and-test with the target set off a reference policy (search-based procedural content
generation: Togelius et al. 2011, https://doi.org/10.1109/TCIAIG.2011.2148116).
"""
import math
import random
import re
import struct
from collections import OrderedDict, deque, namedtuple

from planiverse.environments.base import Environment
from planiverse.environments.generation import draw_until, rng

DIRECTIONS = ("north", "east", "south", "west")
#: Unit vectors per direction; y grows southwards, as in the game.
VECTORS = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}
#: Ticks a decision runs, and how many of them a walk lasts: a `move` covers about 4.5 tiles
#: and a `step` about one, at the character's 0.15 tiles a tick.
DECISION = 30
STRIDES = {"move": 30, "step": 7}
TILE = 256
TILES_PER_TICK = 38 / TILE
#: Coal a hand gives a machine at once (one piece runs a drill for 27 seconds and a furnace for
#: 44), and the waits, in decisions.
AMOUNTS = (1, 2, 5, 10)
WAITS = (1, 10, 40)
#: Ore one hand-mining action digs before the decision ends, about two seconds a piece, and
#: the decisions it may take.
MINE_COUNT, MINE_LIMIT = 5, 40
#: The distances the game measures: a machine may be placed or reached within ten tiles of the
#: character, and ore mined by hand within 2.7.
BUILD_DISTANCE, REACH, ORE_REACH = 10.0, 10.0, 2.7
#: How far from the origin the character may walk, in tiles. The benchmark map the simulator was
#: measured on has no water within it, so its terrain is not needed.
ARENA = 30
#: Ore under each tile of a patch, and what a furnace's ore slot and result slot hold.
ORE_AMOUNT, FURNACE_ORE, FURNACE_PLATES = 10000, 54, 100
#: The item names the simulator uses, by the short names the actions use.
ITEMS = {"drill": "burner-mining-drill", "furnace": "stone-furnace", "coal": "coal",
         "ore": "iron-ore", "plate": "iron-plate"}
SHORT = {long: short for short, long in ITEMS.items()}
#: factory-sim's scene families, by the task its generator draws them for.
SCENE_FAMILIES = (("construct_smelting_line", "open_patch"), ("construct_smelting_line", "offset_patch"),
                  ("construct_smelting_line", "obstructed_patch"), ("construct_smelting_line", "varied_patch"),
                  ("construct_smelting_line", "cluttered_patch"), ("build_line", "square_patch"),
                  ("build_line", "offset_patch"), ("build_line", "narrow_patch"),
                  ("build_line", "varied_patch"), ("build_line", "cluttered_patch"))
#: What the generator draws when not told: the drills and furnaces the character carries (two
#: of each half the time, since two lines are the richer problem), the coal, and the horizon
#: in ticks (40 seconds to two minutes of the game).
CARRIED = ((1, 1), (1, 2), (2, 1), (2, 2), (2, 2), (2, 2))
COALS = (6, 8, 12, 16, 24, 40)
HORIZONS = (2400, 3600, 4800, 6000, 7200)
SNAPSHOTS = 64


def _fsim():
    import fsim
    import fsim.scenes  # noqa: F401  (a submodule; the package does not import it)
    return fsim


class FactoryAction:
    """One decision: `move(d)` or `step(d)`, `place(drill, x, y, d)`, `place(furnace, x, y)`,
    `give(x, y, coal, n)`, `give(x, y, ore)`, `take(x, y)`, `mine(tx, ty)` or `wait(n)`.

    Positions of machines are their centres, which sit on tile corners; a mined tile is named
    by its corner; `wait(n)` waits `n` decisions.
    """

    PATTERN = re.compile(r"^(\w+)\((.*)\)$")

    def __init__(self, verb, direction=None, item=None, x=None, y=None, count=None):
        self.verb, self.direction, self.item = verb, direction, item
        self.x, self.y, self.count = x, y, count
        self.name = self.__name__()

    def __name__(self):
        verb = self.verb
        if verb in STRIDES:
            return f"{verb}({self.direction})"
        if verb == "place":
            facing = f", {self.direction}" if self.item == "drill" else ""
            return f"place({self.item}, {self.x}, {self.y}{facing})"
        if verb == "give":
            count = f", {self.count}" if self.item == "coal" else ""
            return f"give({self.x}, {self.y}, {self.item}{count})"
        if verb in ("take", "mine"):
            return f"{verb}({self.x}, {self.y})"
        if verb == "wait":
            return f"wait({self.count})"
        raise ValueError(f"unknown verb: {verb!r}")

    @classmethod
    def parse(cls, text):
        match = cls.PATTERN.match(str(text).strip())
        if not match:
            raise ValueError(f"not a factory action: {text!r}")
        verb, arguments = match.group(1), [a.strip() for a in match.group(2).split(",") if a.strip()]
        if verb in STRIDES:
            return cls(verb, direction=arguments[0])
        if verb == "place":
            item, x, y = arguments[0], int(arguments[1]), int(arguments[2])
            return cls(verb, item=item, x=x, y=y, direction=arguments[3] if item == "drill" else None)
        if verb == "give":
            x, y, item = int(arguments[0]), int(arguments[1]), arguments[2]
            return cls(verb, item=item, x=x, y=y, count=int(arguments[3]) if item == "coal" else None)
        if verb in ("take", "mine"):
            return cls(verb, x=int(arguments[0]), y=int(arguments[1]))
        if verb == "wait":
            return cls(verb, count=int(arguments[0]))
        raise ValueError(f"unknown verb: {verb!r}")

    def cost(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, FactoryAction) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    __repr__ = __str__


MOVES = tuple(FactoryAction(stride, direction=d) for stride in STRIDES for d in DIRECTIONS)
WAIT_ACTIONS = tuple(FactoryAction("wait", count=n) for n in WAITS)

#: A machine as the state shows it: its centre, what it faces (drills), the engine's status, and
#: the coal, ore and plates in its slots. `dynamics` is the rest of what decides its future.
Machine = namedtuple("Machine", "kind x y facing status fuel ore plates dynamics")


class FactoryState:
    """The factory as it stands: the tick, the character, what it holds, the machines, the ore
    left under the patch and the piles on the ground, with the decisions that led here kept for
    replay. Identity is the contents; `path` and `depth` are bookkeeping."""

    def __init__(self, path, tick, position, inventory, machines, piles, ore, produced,
                 target=0, horizon=0, depth=0):
        self.path = tuple(path)
        self.tick, self.position = tick, position
        self.inventory = dict(inventory)
        self.machines, self.piles, self.ore = tuple(machines), tuple(piles), tuple(ore)
        self.produced = produced
        self.plates = self.inventory.get("plate", 0)
        self.target, self.horizon, self.depth = target, horizon, depth
        self.key = (tick, position, tuple(sorted(self.inventory.items())), self.machines,
                    self.piles, self.ore)
        self.__hash = hash(self.key)
        literals = [f"at({position[0] // TILE}, {position[1] // TILE})"]
        literals += [f"holds({item}, {count})" for item, count in sorted(self.inventory.items())]
        for m in self.machines:
            literals.append(f"drill({m.x}, {m.y}, {m.facing})" if m.kind == "drill"
                            else f"furnace({m.x}, {m.y})")
            literals.append(f"status({m.kind}, {m.x}, {m.y}, {m.status})")
            literals.append(f"fuel({m.kind}, {m.x}, {m.y}, {m.fuel})")
            if m.kind == "furnace":
                literals.append(f"ore_in({m.x}, {m.y}, {m.ore})")
                literals.append(f"plates_in({m.x}, {m.y}, {m.plates})")
        literals += [f"made({produced})", f"tick({tick})"]
        self.literals = frozenset(literals)

    def __eq__(self, other):
        return isinstance(other, FactoryState) and self.key == other.key

    def __hash__(self):
        return self.__hash

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        x, y = self.position[0] / TILE, self.position[1] / TILE
        held = ", ".join(f"{item} {count}" for item, count in sorted(self.inventory.items())) or "nothing"
        parts = [f"tick {self.tick}: at ({x:.1f}, {y:.1f}); holds {held}"]
        for m in self.machines:
            text = f"{m.kind} ({m.x}, {m.y})" + (f" {m.facing}" if m.kind == "drill" else "")
            text += f" {m.status}, fuel {m.fuel}"
            if m.kind == "furnace":
                text += f", ore {m.ore}, plates {m.plates}"
            parts.append(text)
        parts.append(f"made {self.produced}, brought back {self.plates} of {self.target}")
        return "; ".join(parts)

    def __repr__(self):
        return f"<FactoryState(tick={self.tick}, plates={self.plates}, machines={len(self.machines)})>"


# ----------------------------------------------------------------------------- scenes

def scene_instance(family, scene):
    """The instance fields of a factory-sim scene: its ore rectangle, walls and start."""
    xs = sorted({int(r["position"][0]) for r in scene["resources"]})
    ys = sorted({int(r["position"][1]) for r in scene["resources"]})
    if len(scene["resources"]) != len(xs) * len(ys):
        raise ValueError("the patch is not a rectangle")
    walls = sorted([int(e["position"][0]), int(e["position"][1])] for e in scene["entities"])
    start = [float(v) for v in scene["character"]["position"]]
    return {"family": family, "ore": [xs[0], ys[0], xs[-1], ys[-1]], "walls": walls, "start": start}


def blueprint_of(instance):
    """The scene payload factory-sim resets to, from an instance."""
    x0, y0, x1, y1 = instance["ore"]
    resources = [{"name": ITEMS["ore"], "position": [float(x), float(y)], "amount": ORE_AMOUNT}
                 for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
    walls = [{"name": "stone-wall", "position": [float(x), float(y)], "direction": "north",
              "force": "neutral"} for x, y in instance["walls"]]
    inventory = {ITEMS["drill"]: instance["drills"], ITEMS["furnace"]: instance["furnaces"],
                 ITEMS["coal"]: instance["coal"]}
    return {"entities": walls, "resources": resources,
            "character": {"position": list(instance["start"]), "inventory": inventory}}


def footprint(x, y):
    """The four tiles of a two-by-two machine centred on the corner (x, y)."""
    return ((x - 1, y - 1), (x, y - 1), (x - 1, y), (x, y))


def distance(position, x, y):
    """Tiles from a position in 1/256 tiles to the point (x, y)."""
    return math.hypot(position[0] / TILE - x, position[1] / TILE - y)


#: A drill and the furnace its ore drops into: centres and the drill's facing.
Line = namedtuple("Line", "drill facing furnace")


class FactoryEnv(Environment):
    """Bring the plates the goal asks for back before the horizon."""

    def __init__(self):
        super().__init__("factory")
        self.instance = None
        self.index = None
        self.state = None
        self.state_history = []
        self.witness = None
        self.witness_expansions = None
        self.sim = None
        self.blueprint = None
        self.tiles = ()
        self.walls = frozenset()
        self.anchors = ()
        self.__initial__ = None
        self.__snapshots__ = OrderedDict()
        self.__loaded__ = None

    # ------------------------------------------------------------------ instances

    def set_index(self, index):
        if not 0 <= index < len(PATCHES):
            raise IndexError(f"Invalid index: {index}. There are {len(PATCHES)} patches, so the "
                             f"index must be 0-{len(PATCHES) - 1}.")
        self.set_instance(PATCHES[index])
        self.index = index

    def set_instance(self, instance):
        """Select a patch: `family`, `ore` (the rectangle's corners, inclusive), `walls`,
        `start`, `drills`, `furnaces`, `coal`, `target` and `horizon`."""
        for key in ("ore", "walls", "start", "drills", "furnaces", "coal", "target", "horizon"):
            if key not in instance:
                raise ValueError(f"a patch needs `{key}`")
        x0, y0, x1, y1 = (int(v) for v in instance["ore"])
        self.instance = {"family": str(instance.get("family", "")), "ore": [x0, y0, x1, y1],
                         "walls": [[int(x), int(y)] for x, y in instance["walls"]],
                         "start": [float(v) for v in instance["start"]],
                         "drills": int(instance["drills"]), "furnaces": int(instance["furnaces"]),
                         "coal": int(instance["coal"]), "target": int(instance["target"]),
                         "horizon": int(instance["horizon"])}
        self.index = None
        self.witness = self.witness_expansions = None
        self.blueprint = blueprint_of(self.instance)
        self.tiles = tuple((x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1))
        ore = set(self.tiles)
        self.walls = frozenset(map(tuple, self.instance["walls"]))
        self.anchors = tuple((x, y) for x in range(x0 + 1, x1 + 1) for y in range(y0 + 1, y1 + 1)
                             if set(footprint(x, y)) <= ore and not set(footprint(x, y)) & self.walls)
        self.__initial__ = None
        self.__snapshots__ = OrderedDict()
        self.__loaded__ = None

    def generate_instance(self, seed=None, drills=None, furnaces=None, coal=None, horizon=None,
                          min_plates=5, attempts=40):
        """Draw a patch, select it, and return it as the dict `set_instance` takes.

        A scene comes from factory-sim's own generator, one of its ten families at a seed of the
        draw's, so the patch, its walls and the start are shaped as FactorioRL shapes them; what
        the character carries and the horizon are drawn from `CARRIED`, `COALS` and `HORIZONS`
        unless given.
        The scripted lines are then measured (`reference_plans`) and the target is the most
        plates any of them brings back by the horizon, with that plan left in `witness` and how
        many were measured in `witness_expansions`. A scene where no line brings back
        `min_plates` is thrown back.
        """
        random_, _ = rng(seed)
        fsim = _fsim()
        found = {}

        def draw(attempt):
            task, family = random_.choice(SCENE_FAMILIES)
            scene = fsim.scenes.GENERATORS[task](family, random.Random(random_.randrange(2 ** 31)))
            instance = scene_instance(family, scene)
            carried = random_.choice(CARRIED)
            instance.update(drills=drills or carried[0], furnaces=furnaces or carried[1],
                            coal=coal or random_.choice(COALS),
                            horizon=horizon or random_.choice(HORIZONS), target=0)
            return instance

        def accept(instance):
            self.set_instance(instance)
            plans = self.reference_plans()
            if not plans or plans[0][0] < min_plates:
                return False
            instance["target"] = plans[0][0]
            found["plan"], found["measured"] = plans[0][1], len(plans)
            return True

        instance = draw_until(draw, accept, attempts, "factory patch")
        self.set_instance(instance)
        self.witness, self.witness_expansions = found["plan"], found["measured"]
        return instance

    # ------------------------------------------------------------------ the simulator

    def _sim(self):
        if self.sim is None:
            self.sim = _fsim().Sim(water=[])
        return self.sim

    def _snapshot(self):
        return bytes(_fsim().ffi.buffer(self.sim.env))

    def _restore(self, snapshot):
        _fsim().ffi.memmove(self.sim.env, snapshot, len(snapshot))

    def _remember(self, path, snapshot):
        self.__snapshots__[path] = snapshot
        self.__snapshots__.move_to_end(path)
        while len(self.__snapshots__) > SNAPSHOTS:
            self.__snapshots__.popitem(last=False)

    def _load(self, path):
        """Bring the simulator to the state at the end of `path`: from a snapshot when there is
        one, else by replaying the decisions past the longest snapshotted prefix."""
        if self.__loaded__ == path:
            return
        done = 0
        for k in range(len(path), 0, -1):
            snapshot = self.__snapshots__.get(path[:k])
            if snapshot is not None:
                self._restore(snapshot)
                done = k
                break
        else:
            self._restore(self.__initial__)
        for action in path[done:]:
            if not self._apply(action):
                raise RuntimeError(f"a replayed decision was refused: {action}")
        self.__loaded__ = path

    def _handle_at(self, x, y):
        """The simulator's handle for the machine centred on (x, y), or -1."""
        env, lib = self.sim.env, _fsim().lib
        for k in range(env.seen_count):
            e = env.entities[env.seen[k].entity]
            if e.pos.x == x * TILE and e.pos.y == y * TILE and e.kind in (lib.K_DRILL, lib.K_FURNACE):
                return env.seen[k].handle
        return -1

    def _tile_handle(self, tx, ty):
        """The simulator's handle for the ore tile (tx, ty), or -1."""
        env = self.sim.env
        for k in range(env.tile_count):
            r = env.resources[env.tiles[k].resource]
            if r.tx == tx and r.ty == ty:
                return env.tiles[k].handle
        return -1

    def _apply(self, action):
        """Run one decision on the simulator; False if it refused the action."""
        sim, lib = self.sim, _fsim().lib
        verb = action.verb
        if verb in STRIDES:
            sim.step(f"{verb}_{action.direction}", ticks=DECISION)
        elif verb == "place":
            sim.step("place_at", {"item": ITEMS[action.item], "position": [action.x, action.y],
                                  "direction": action.direction or "north"}, ticks=DECISION)
        elif verb == "give":
            handle = self._handle_at(action.x, action.y)
            if handle < 0:
                return False
            count = action.count if action.item == "coal" else FURNACE_ORE
            sim.step("give_to", {"to": f"h{handle}", "item": ITEMS[action.item], "count": count},
                     ticks=DECISION)
        elif verb == "take":
            handle = self._handle_at(action.x, action.y)
            if handle < 0:
                return False
            sim.step("take_from", {"from": f"h{handle}", "item": ITEMS["plate"],
                                   "count": FURNACE_PLATES}, ticks=DECISION)
        elif verb == "mine":
            handle = self._tile_handle(action.x, action.y)
            if handle < 0:
                return False
            sim.step("mine_at", {"handle": f"h{handle}", "count": MINE_COUNT}, ticks=DECISION)
            if sim.env.act.status == lib.R_REJECTED:
                return False
            for _ in range(MINE_LIMIT):
                env = sim.env
                if env.slot_mine < 0 or env.inflight[env.slot_mine].terminal:
                    break
                sim.step("wait", ticks=DECISION)
            return True
        elif verb == "wait":
            sim.step("wait", ticks=DECISION * action.count)
        else:
            return False
        return sim.env.act.status != lib.R_REJECTED

    def _read_state(self, path, depth):
        fsim = _fsim()
        env, lib, ffi = self.sim.env, fsim.lib, fsim.ffi
        slots = struct.unpack_from(f"{2 * lib.FSIM_MAIN_SLOTS}i", ffi.buffer(env.main))
        inventory = {}
        for k in range(0, len(slots), 2):
            if slots[k + 1] > 0:
                item = SHORT.get(fsim.ITEM_NAMES[slots[k]], fsim.ITEM_NAMES[slots[k]])
                inventory[item] = inventory.get(item, 0) + slots[k + 1]
        amounts = struct.unpack_from(f"{5 * env.resource_count}i", ffi.buffer(env.resources))[4::5]
        machines, piles = [], []
        for i in range(env.entity_count):
            e = env.entities[i]
            if not e.alive or e.kind == lib.K_WALL:
                continue
            if e.kind == lib.K_PILE:
                piles.append((e.pos.x, e.pos.y, e.pile.item, e.pile.count))
                continue
            kind = "drill" if e.kind == lib.K_DRILL else "furnace"
            dynamics = (e.energy, e.remaining, e.burning, e.fuel.item, e.crafting,
                        e.products_finished, e.seconds, e.progress, e.held, e.linked_unit,
                        e.mine_cursor, e.mine_count, e.unit)
            machines.append(Machine(kind, e.pos.x // TILE, e.pos.y // TILE,
                                    DIRECTIONS[e.direction // 4], fsim.STATUS_NAME[e.status],
                                    e.fuel.count, e.source.count, e.result.count, dynamics))
        return FactoryState(path, int(env.tick), (env.char_pos.x, env.char_pos.y), inventory,
                            machines, piles, amounts, int(env.produced[lib.IT_IRON_PLATE]),
                            target=self.instance["target"], horizon=self.instance["horizon"],
                            depth=depth)

    # ------------------------------------------------------------------- contract

    def reset(self):
        if self.instance is None:
            self.set_index(0)
        sim = self._sim()
        sim.reset(self.blueprint)
        self.__initial__ = self._snapshot()
        self.__snapshots__ = OrderedDict()
        self.__loaded__ = ()
        self.state = self._read_state((), 0)
        self.state_history = [self.state]
        return self.state, {"patch": self.index, "family": self.instance["family"],
                            "target": self.instance["target"], "horizon": self.instance["horizon"],
                            "generated": self.index is None}

    def is_goal(self, state):
        return state.plates >= self.instance["target"] and state.tick <= self.instance["horizon"]

    def is_terminal(self, state):
        return state.tick >= self.instance["horizon"] and not self.is_goal(state)

    def get_actions(self, state=None):
        """The decisions open in a state, before the simulator has its say on reach and
        collisions: walks that stay in the arena, drills on free ore and furnaces at a drill's
        drop within build distance, coal and ore for machines within reach, plates from a
        furnace that has some, ore tiles within the hand's reach, and the waits."""
        state = state or self.state
        held = state.inventory
        position = state.position
        taken = set(self.walls)
        for m in state.machines:
            taken.update(footprint(m.x, m.y))
        actions = []
        for action in MOVES:
            vx, vy = VECTORS[action.direction]
            reach = STRIDES[action.verb] * TILES_PER_TICK
            end = (position[0] / TILE + vx * reach, position[1] / TILE + vy * reach)
            if abs(end[0]) <= ARENA and abs(end[1]) <= ARENA:
                actions.append(action)
        if held.get("drill", 0) > 0:
            for x, y in self.anchors:
                if not set(footprint(x, y)) & taken and distance(position, x, y) <= BUILD_DISTANCE:
                    actions += [FactoryAction("place", item="drill", x=x, y=y, direction=d)
                                for d in DIRECTIONS]
        if held.get("furnace", 0) > 0:
            for m in state.machines:
                if m.kind != "drill":
                    continue
                vx, vy = VECTORS[m.facing]
                fx, fy = m.x + 2 * vx, m.y + 2 * vy
                tiles = footprint(fx, fy)
                if (not set(tiles) & taken and distance(position, fx, fy) <= BUILD_DISTANCE
                        and all(abs(tx) <= ARENA and abs(ty) <= ARENA for tx, ty in tiles)):
                    actions.append(FactoryAction("place", item="furnace", x=fx, y=fy))
        near = [m for m in state.machines if distance(position, m.x, m.y) <= REACH + 1]
        coal = held.get("coal", 0)
        for m in near:
            actions += [FactoryAction("give", item="coal", x=m.x, y=m.y, count=n)
                        for n in AMOUNTS if n <= coal and n <= 50 - m.fuel]
        if held.get("ore", 0) > 0:
            actions += [FactoryAction("give", item="ore", x=m.x, y=m.y)
                        for m in near if m.kind == "furnace" and m.ore < FURNACE_ORE]
        actions += [FactoryAction("take", x=m.x, y=m.y) for m in near
                    if m.kind == "furnace" and m.plates > 0]
        for (tx, ty), amount in zip(self.tiles, state.ore):
            if amount >= MINE_COUNT and distance(position, tx + 0.5, ty + 0.5) <= ORE_REACH + 0.5:
                actions.append(FactoryAction("mine", x=tx, y=ty))
        actions += list(WAIT_ACTIONS)
        return actions

    def successors(self, state):
        if self.is_goal(state) or self.is_terminal(state):
            return []
        self._load(state.path)
        parent = self._snapshot()
        self._remember(state.path, parent)
        children = []
        for action in self.get_actions(state):
            if self._apply(action):
                child = self._read_state(state.path + (action,), state.depth + 1)
                if child != state:
                    children.append((action, child))
            self._restore(parent)
        self.__loaded__ = state.path
        return children

    def __advance__(self, state, action):
        if self.is_goal(state) or self.is_terminal(state):
            return state
        if not isinstance(action, FactoryAction):
            action = FactoryAction.parse(action)
        if action not in self.get_actions(state):
            return state
        self._load(state.path)
        parent = self._snapshot()
        self._remember(state.path, parent)
        if not self._apply(action):
            self._restore(parent)
            return state
        child = self._read_state(state.path + (action,), state.depth + 1)
        self.__loaded__ = child.path
        if child == state:
            return state
        return child

    def simulate(self, plan):
        state, _ = self.reset()
        trace = [state]
        for action in plan:
            trace.append(self.__advance__(trace[-1], action))
        return trace

    def step(self, action):
        if self.state is None:
            raise ValueError("call reset() first")
        before = self.state.plates
        self.state = self.__advance__(self.state, action)
        self.state_history.append(self.state)
        return self.state, self.state.plates - before

    def render(self):
        lines = [f"step {k}: {state}" for k, state in enumerate(self.state_history)]
        print("\n".join(lines))
        return lines

    def close(self):
        self.sim = None
        self.__initial__ = None
        self.__snapshots__ = OrderedDict()
        self.__loaded__ = None

    # ------------------------------------------------------------- the scripted lines

    def lines(self):
        """Every line the patch admits, nearest the start first: a drill on four ore tiles and
        the furnace its ore drops into, on tiles that are free, in the arena and off the walls."""
        start = self.instance["start"]
        found = []
        for ax, ay in self.anchors:
            for facing in DIRECTIONS:
                vx, vy = VECTORS[facing]
                fx, fy = ax + 2 * vx, ay + 2 * vy
                tiles = footprint(fx, fy)
                if set(tiles) & self.walls or set(tiles) & set(footprint(ax, ay)):
                    continue
                if any(abs(tx) > ARENA or abs(ty) > ARENA for tx, ty in tiles):
                    continue
                found.append(Line((ax, ay), facing, (fx, fy)))
        found.sort(key=lambda line: (math.hypot(line.drill[0] - start[0], line.drill[1] - start[1]),
                                     line.drill, DIRECTIONS.index(line.facing)))
        return found

    def _walk(self, state, machines, limit=400):
        """A walk from `state` to a spot within build distance of every machine centre and off
        their tiles, by breadth-first search over the walking decisions; None if none is found."""
        tiles = set()
        for x, y in machines:
            tiles.update(footprint(x, y))

        def arrived(position):
            tile = (position[0] // TILE, position[1] // TILE)
            return tile not in tiles and all(distance(position, x, y) <= BUILD_DISTANCE - 0.5
                                             for x, y in machines)

        if arrived(state.position):
            return []
        self._load(state.path)
        origin = self._snapshot()
        frontier = deque([(state.position, ())])
        seen = {state.position}
        expansions = 0
        walk = None
        while frontier and expansions < limit and walk is None:
            position, moves = frontier.popleft()
            expansions += 1
            for action in MOVES:
                vx, vy = VECTORS[action.direction]
                reach = STRIDES[action.verb] * TILES_PER_TICK
                if abs(position[0] / TILE + vx * reach) > ARENA or abs(position[1] / TILE + vy * reach) > ARENA:
                    continue
                self._restore(origin)
                for earlier in moves:
                    self._apply(earlier)
                self._apply(action)
                there = (self.sim.env.char_pos.x, self.sim.env.char_pos.y)
                if there in seen:
                    continue
                seen.add(there)
                if arrived(there):
                    walk = list(moves) + [action]
                    break
                frontier.append((there, moves + (action,)))
        self._restore(origin)
        self.__loaded__ = state.path
        return walk

    def reference_plan(self, lines, split):
        """Walk to the lines, build them, fuel them as `split` says (coal for each drill and
        furnace), wait for the horizon and take the plates: `(plates brought back, plan)`, or
        None when the simulator refuses a step of it."""
        horizon = self.instance["horizon"]
        state, _ = self.reset()
        machines = [line.drill for line in lines] + [line.furnace for line in lines]
        walk = self._walk(state, machines)
        if walk is None:
            return None
        plan = list(walk)
        for line in lines:
            plan.append(FactoryAction("place", item="drill", x=line.drill[0], y=line.drill[1],
                                      direction=line.facing))
            plan.append(FactoryAction("place", item="furnace", x=line.furnace[0], y=line.furnace[1]))
        for line, (for_drill, for_furnace) in zip(lines, split):
            plan.append(FactoryAction("give", item="coal", x=line.drill[0], y=line.drill[1], count=for_drill))
            plan.append(FactoryAction("give", item="coal", x=line.furnace[0], y=line.furnace[1],
                                      count=for_furnace))
        for action in plan:
            after = self.__advance__(state, action)
            if after is state:
                return None
            state = after
        left = horizon - state.tick - DECISION * len(lines)
        waits = []
        for wait in sorted(WAITS, reverse=True)[:-1]:     # the shortest wait is left for slack
            while left >= DECISION * wait:
                waits.append(FactoryAction("wait", count=wait))
                left -= DECISION * wait
        takes = [FactoryAction("take", x=line.furnace[0], y=line.furnace[1]) for line in lines]
        for action in waits + takes:
            after = self.__advance__(state, action)
            if after is state:
                return None
            state = after
        if state.tick > horizon:
            return None
        return state.plates, plan + waits + takes

    def reference_plans(self, layouts=2):
        """The scripted lines measured on the selected patch, best first: one line or two at
        the nearest `layouts` places each, with every split of the coal among the machines in
        the amounts a hand gives. Each entry is `(plates brought back, plan)`."""
        instance = self.instance
        singles = self.lines()
        sets = [[line] for line in singles[:layouts]]
        if instance["drills"] >= 2 and instance["furnaces"] >= 2:
            pairs = []
            for i, first in enumerate(singles):
                for second in singles[i + 1:]:
                    tiles = [set(footprint(*first.drill)), set(footprint(*first.furnace)),
                             set(footprint(*second.drill)), set(footprint(*second.furnace))]
                    if all(not a & b for k, a in enumerate(tiles) for b in tiles[k + 1:]):
                        pairs.append([first, second])
                if len(pairs) >= layouts:
                    break
            sets += pairs[:layouts]
        coal = instance["coal"]
        results = []
        target = instance["target"]
        instance["target"] = math.inf         # nothing is a goal while the lines are measured
        try:
            for lines in sets:
                splits = [[(d, f)] * len(lines) for d in AMOUNTS for f in AMOUNTS
                          if len(lines) * (d + f) <= coal]
                for split in splits:
                    outcome = self.reference_plan(lines, split)
                    if outcome is not None:
                        results.append(outcome)
        finally:
            instance["target"] = target
        results.sort(key=lambda outcome: (-outcome[0], len(outcome[1])))
        return results


#: The bundled patches: `generate_instance(seed)` for the seed beside each, embedded as the plain
#: data `set_instance` takes, with the plan each was accepted on in
#: `tests/data/factory_solutions.json`.
PATCHES = (
    # seed 9000; square_patch, 2 lines, 18-step plan
    {"family": "square_patch", "ore": [-3, -3, 3, 3], "walls": [], "start": [5.4, -11.6], "drills": 2, "furnaces": 2, "coal": 6, "target": 26, "horizon": 7200},
    # seed 9001; open_patch, 1 line, 13-step plan
    {"family": "open_patch", "ore": [-3, -3, 3, 3], "walls": [], "start": [3.5, -11.2], "drills": 2, "furnaces": 1, "coal": 6, "target": 13, "horizon": 7200},
    # seed 9002; cluttered_patch, 2 lines, 17-step plan
    {"family": "cluttered_patch", "ore": [-4, -5, 2, 1], "walls": [[-5, -4], [-5, -3], [-5, -2], [-5, -1], [-3, 2], [-2, 2], [-1, 2], [0, 2], [5, -3], [5, -2], [5, -1], [5, 0]], "start": [-8.6, 5.1], "drills": 2, "furnaces": 2, "coal": 8, "target": 26, "horizon": 6000},
    # seed 9003; cluttered_patch, 1 line, 11-step plan
    {"family": "cluttered_patch", "ore": [-8, -9, -2, -3], "walls": [[-10, -5], [-10, -4], [-10, -3], [-10, -2], [-9, -3], [-9, -2], [-5, -2], [-4, -2], [-3, -2]], "start": [-12.2, -0.7], "drills": 1, "furnaces": 1, "coal": 24, "target": 18, "horizon": 4800},
    # seed 9004; narrow_patch, 2 lines, 16-step plan
    {"family": "narrow_patch", "ore": [-5, -1, 4, 0], "walls": [[6, -1], [6, 0], [6, 1]], "start": [6.8, 8.6], "drills": 2, "furnaces": 2, "coal": 6, "target": 26, "horizon": 4800},
    # seed 9005; square_patch, 1 line, 9-step plan
    {"family": "square_patch", "ore": [-3, -3, 3, 3], "walls": [], "start": [-3.3, -8.8], "drills": 1, "furnaces": 1, "coal": 12, "target": 8, "horizon": 2400},
    # seed 9006; offset_patch, 1 line, 9-step plan
    {"family": "offset_patch", "ore": [-12, -12, -6, -6], "walls": [], "start": [2.3, -10.0], "drills": 2, "furnaces": 1, "coal": 8, "target": 8, "horizon": 2400},
    # seed 9007; square_patch, 1 line, 10-step plan
    {"family": "square_patch", "ore": [-3, -3, 3, 3], "walls": [], "start": [10.5, 6.5], "drills": 1, "furnaces": 2, "coal": 40, "target": 13, "horizon": 3600},
    # seed 9008; cluttered_patch, 2 lines, 16-step plan
    {"family": "cluttered_patch", "ore": [-5, 4, 1, 10], "walls": [[-7, 3], [-6, 3], [-5, 3], [-3, 1], [-2, 1]], "start": [3.0, 16.6], "drills": 2, "furnaces": 2, "coal": 16, "target": 36, "horizon": 4800},
    # seed 9009; cluttered_patch, 2 lines, 15-step plan
    {"family": "cluttered_patch", "ore": [-12, -8, -6, -2], "walls": [[-10, -9], [-9, -9], [-8, -9], [-7, -9]], "start": [1.6, -10.6], "drills": 2, "furnaces": 2, "coal": 40, "target": 26, "horizon": 3600},
    # seed 9010; offset_patch, 2 lines, 15-step plan
    {"family": "offset_patch", "ore": [6, 6, 12, 12], "walls": [], "start": [2.6, -0.6], "drills": 2, "furnaces": 2, "coal": 40, "target": 24, "horizon": 3600},
    # seed 9011; varied_patch, 1 line, 12-step plan
    {"family": "varied_patch", "ore": [-15, 8, -7, 14], "walls": [], "start": [-20.2, 4.5], "drills": 2, "furnaces": 1, "coal": 12, "target": 23, "horizon": 6000},
    # seed 9012; square_patch, 2 lines, 15-step plan
    {"family": "square_patch", "ore": [-3, -3, 3, 3], "walls": [], "start": [7.8, -3.1], "drills": 2, "furnaces": 2, "coal": 12, "target": 26, "horizon": 3600},
    # seed 9013; square_patch, 2 lines, 14-step plan
    {"family": "square_patch", "ore": [-3, -3, 3, 3], "walls": [], "start": [9.8, 5.0], "drills": 2, "furnaces": 2, "coal": 16, "target": 16, "horizon": 2400},
    # seed 9014; square_patch, 2 lines, 15-step plan
    {"family": "square_patch", "ore": [-3, -3, 3, 3], "walls": [], "start": [8.2, 2.6], "drills": 2, "furnaces": 2, "coal": 12, "target": 26, "horizon": 3600},
    # seed 9015; open_patch, 1 line, 13-step plan
    {"family": "open_patch", "ore": [-3, -3, 3, 3], "walls": [], "start": [9.2, 7.4], "drills": 1, "furnaces": 1, "coal": 12, "target": 28, "horizon": 7200},
    # seed 9016; offset_patch, 1 line, 13-step plan
    {"family": "offset_patch", "ore": [5, -11, 11, -5], "walls": [], "start": [-0.5, -1.6], "drills": 2, "furnaces": 2, "coal": 12, "target": 28, "horizon": 7200},
    # seed 9017; varied_patch, 1 line, 11-step plan
    {"family": "varied_patch", "ore": [9, -2, 11, 0], "walls": [], "start": [17.0, -9.5], "drills": 2, "furnaces": 1, "coal": 40, "target": 13, "horizon": 3600},
    # seed 9018; cluttered_patch, 2 lines, 17-step plan
    {"family": "cluttered_patch", "ore": [-9, -7, -3, -1], "walls": [[-7, 1], [-6, 1], [-5, 1], [-1, -2], [-1, -1], [-1, 0]], "start": [5.8, -8.2], "drills": 2, "furnaces": 2, "coal": 40, "target": 44, "horizon": 6000},
    # seed 9019; offset_patch, 1 line, 13-step plan
    {"family": "offset_patch", "ore": [5, -11, 11, -5], "walls": [], "start": [14.6, 0.9], "drills": 1, "furnaces": 2, "coal": 16, "target": 28, "horizon": 7200},
    # seed 9020; varied_patch, 1 line, 12-step plan
    {"family": "varied_patch", "ore": [-7, 7, -3, 13], "walls": [], "start": [-6.2, 20.0], "drills": 1, "furnaces": 1, "coal": 40, "target": 23, "horizon": 6000},
    # seed 9021; offset_patch, 2 lines, 16-step plan
    {"family": "offset_patch", "ore": [6, -12, 12, -6], "walls": [], "start": [-1.1, -11.7], "drills": 2, "furnaces": 2, "coal": 12, "target": 26, "horizon": 4800},
    # seed 9022; obstructed_patch, 1 line, 9-step plan
    {"family": "obstructed_patch", "ore": [-1, -5, 1, 5], "walls": [[5, -2], [5, -1], [5, 0], [5, 1], [5, 2]], "start": [6.6, -8.4], "drills": 2, "furnaces": 1, "coal": 8, "target": 8, "horizon": 2400},
    # seed 9023; varied_patch, 2 lines, 18-step plan
    {"family": "varied_patch", "ore": [-6, 3, -2, 5], "walls": [], "start": [6.9, 6.4], "drills": 2, "furnaces": 2, "coal": 24, "target": 54, "horizon": 7200},
    # seed 9024; open_patch, 1 line, 11-step plan
    {"family": "open_patch", "ore": [-3, -3, 3, 3], "walls": [], "start": [-5.9, 9.6], "drills": 1, "furnaces": 1, "coal": 24, "target": 18, "horizon": 4800},
    # seed 9025; varied_patch, 1 line, 11-step plan
    {"family": "varied_patch", "ore": [-12, -5, -10, 3], "walls": [], "start": [-7.0, 10.7], "drills": 1, "furnaces": 2, "coal": 8, "target": 18, "horizon": 4800},
    # seed 9026; cluttered_patch, 1 line, 10-step plan
    {"family": "cluttered_patch", "ore": [-6, 5, 0, 11], "walls": [[1, 2], [1, 3], [2, 11], [2, 12], [2, 13], [2, 14], [3, 3], [3, 4]], "start": [9.1, 7.7], "drills": 2, "furnaces": 1, "coal": 16, "target": 13, "horizon": 3600},
    # seed 9027; square_patch, 1 line, 9-step plan
    {"family": "square_patch", "ore": [-3, -3, 3, 3], "walls": [], "start": [-1.8, -8.0], "drills": 1, "furnaces": 1, "coal": 40, "target": 8, "horizon": 2400},
    # seed 9028; offset_patch, 1 line, 13-step plan
    {"family": "offset_patch", "ore": [5, -11, 11, -5], "walls": [], "start": [7.8, -16.9], "drills": 1, "furnaces": 1, "coal": 24, "target": 28, "horizon": 7200},
    # seed 9029; offset_patch, 1 line, 13-step plan
    {"family": "offset_patch", "ore": [5, 5, 11, 11], "walls": [], "start": [3.7, 14.8], "drills": 2, "furnaces": 1, "coal": 12, "target": 28, "horizon": 7200},
    # seed 9030; cluttered_patch, 1 line, 10-step plan
    {"family": "cluttered_patch", "ore": [-7, 5, -1, 11], "walls": [[-7, 2], [-6, 2], [-5, 2], [-4, 2], [-1, 13], [0, 13], [1, 13]], "start": [8.9, 7.7], "drills": 2, "furnaces": 1, "coal": 16, "target": 8, "horizon": 2400},
    # seed 9031; square_patch, 1 line, 12-step plan
    {"family": "square_patch", "ore": [-3, -3, 3, 3], "walls": [], "start": [6.6, 4.5], "drills": 1, "furnaces": 2, "coal": 6, "target": 13, "horizon": 6000},
    # seed 9032; narrow_patch, 2 lines, 14-step plan
    {"family": "narrow_patch", "ore": [-5, -1, 4, 0], "walls": [[6, -1], [6, 0], [6, 1]], "start": [8.9, 3.1], "drills": 2, "furnaces": 2, "coal": 16, "target": 16, "horizon": 2400},
    # seed 9033; square_patch, 2 lines, 15-step plan
    {"family": "square_patch", "ore": [-3, -3, 3, 3], "walls": [], "start": [5.3, 11.3], "drills": 2, "furnaces": 2, "coal": 12, "target": 24, "horizon": 3600},
    # seed 9034; obstructed_patch, 2 lines, 14-step plan
    {"family": "obstructed_patch", "ore": [-1, -5, 1, 5], "walls": [[5, -2], [5, -1], [5, 0], [5, 1], [5, 2]], "start": [10.7, -0.7], "drills": 2, "furnaces": 2, "coal": 16, "target": 14, "horizon": 2400},
    # seed 9035; cluttered_patch, 2 lines, 14-step plan
    {"family": "cluttered_patch", "ore": [-6, -10, 0, -4], "walls": [[-7, -11], [-7, -10], [-2, -13], [-1, -13], [0, -13], [1, -13]], "start": [-0.7, 3.6], "drills": 2, "furnaces": 2, "coal": 12, "target": 14, "horizon": 2400},
    # seed 9036; varied_patch, 2 lines, 17-step plan
    {"family": "varied_patch", "ore": [-11, 1, -3, 9], "walls": [], "start": [-15.5, 7.1], "drills": 2, "furnaces": 2, "coal": 8, "target": 26, "horizon": 6000},
    # seed 9037; cluttered_patch, 2 lines, 17-step plan
    {"family": "cluttered_patch", "ore": [-5, 6, 1, 12], "walls": [[-7, 15], [-6, 15], [4, 12], [4, 13], [4, 14]], "start": [-4.5, 17.2], "drills": 2, "furnaces": 2, "coal": 6, "target": 26, "horizon": 6000},
    # seed 9038; open_patch, 1 line, 9-step plan
    {"family": "open_patch", "ore": [-3, -3, 3, 3], "walls": [], "start": [-8.6, 4.0], "drills": 1, "furnaces": 2, "coal": 24, "target": 8, "horizon": 2400},
    # seed 9039; obstructed_patch, 1 line, 14-step plan
    {"family": "obstructed_patch", "ore": [-1, -5, 1, 5], "walls": [[5, -2], [5, -1], [5, 0], [5, 1], [5, 2]], "start": [-8.5, -8.8], "drills": 2, "furnaces": 1, "coal": 12, "target": 28, "horizon": 7200},
    # seed 9040; square_patch, 2 lines, 15-step plan
    {"family": "square_patch", "ore": [-3, -3, 3, 3], "walls": [], "start": [7.2, -8.4], "drills": 2, "furnaces": 2, "coal": 16, "target": 26, "horizon": 3600},
    # seed 9041; varied_patch, 2 lines, 17-step plan
    {"family": "varied_patch", "ore": [3, -12, 7, -8], "walls": [], "start": [16.9, -6.7], "drills": 2, "furnaces": 2, "coal": 16, "target": 44, "horizon": 6000},
    # seed 9042; cluttered_patch, 1 line, 12-step plan
    {"family": "cluttered_patch", "ore": [-1, -2, 5, 4], "walls": [[-3, 6], [-2, 6], [1, 5], [2, 5], [3, 5], [6, -3], [6, -2], [6, -1], [6, 0]], "start": [9.5, -5.4], "drills": 2, "furnaces": 1, "coal": 24, "target": 23, "horizon": 6000},
    # seed 9043; cluttered_patch, 2 lines, 15-step plan
    {"family": "cluttered_patch", "ore": [0, -10, 6, -4], "walls": [[4, -2], [5, -2]], "start": [2.3, -17.2], "drills": 2, "furnaces": 2, "coal": 12, "target": 26, "horizon": 3600},
    # seed 9044; varied_patch, 2 lines, 15-step plan
    {"family": "varied_patch", "ore": [-4, -12, 2, -6], "walls": [], "start": [-6.3, 0.9], "drills": 2, "furnaces": 2, "coal": 24, "target": 24, "horizon": 3600},
    # seed 9045; offset_patch, 2 lines, 14-step plan
    {"family": "offset_patch", "ore": [5, 5, 11, 11], "walls": [], "start": [-1.1, 4.0], "drills": 2, "furnaces": 2, "coal": 16, "target": 16, "horizon": 2400},
    # seed 9046; varied_patch, 2 lines, 17-step plan
    {"family": "varied_patch", "ore": [-6, 4, 2, 12], "walls": [], "start": [7.4, 11.1], "drills": 2, "furnaces": 2, "coal": 12, "target": 26, "horizon": 6000},
    # seed 9047; narrow_patch, 1 line, 12-step plan
    {"family": "narrow_patch", "ore": [-5, -1, 4, 0], "walls": [[6, -1], [6, 0], [6, 1]], "start": [10.1, 1.1], "drills": 1, "furnaces": 1, "coal": 24, "target": 23, "horizon": 6000},
    # seed 9048; cluttered_patch, 1 line, 10-step plan
    {"family": "cluttered_patch", "ore": [-4, 6, 2, 12], "walls": [[-5, 4], [-4, 4], [-3, 4], [-2, 4], [3, 7], [3, 8], [3, 9], [3, 11], [3, 12], [3, 13], [3, 14]], "start": [-4.3, -2.9], "drills": 1, "furnaces": 2, "coal": 8, "target": 8, "horizon": 2400},
    # seed 9049; square_patch, 1 line, 13-step plan
    {"family": "square_patch", "ore": [-3, -3, 3, 3], "walls": [], "start": [-6.5, -6.2], "drills": 2, "furnaces": 1, "coal": 6, "target": 13, "horizon": 7200},
    # seed 9050; narrow_patch, 2 lines, 17-step plan
    {"family": "narrow_patch", "ore": [-5, -1, 4, 0], "walls": [[-6, -1], [-6, 0], [-6, 1]], "start": [-0.9, -10.8], "drills": 2, "furnaces": 2, "coal": 24, "target": 44, "horizon": 6000},
    # seed 9051; obstructed_patch, 2 lines, 14-step plan
    {"family": "obstructed_patch", "ore": [-1, -5, 1, 5], "walls": [[5, -2], [5, -1], [5, 0], [5, 1], [5, 2]], "start": [10.5, 0.1], "drills": 2, "furnaces": 2, "coal": 16, "target": 14, "horizon": 2400},
    # seed 9052; open_patch, 1 line, 9-step plan
    {"family": "open_patch", "ore": [-3, -3, 3, 3], "walls": [], "start": [-3.5, -9.7], "drills": 2, "furnaces": 1, "coal": 24, "target": 8, "horizon": 2400},
    # seed 9053; square_patch, 1 line, 13-step plan
    {"family": "square_patch", "ore": [-3, -3, 3, 3], "walls": [], "start": [-0.0, 12.7], "drills": 1, "furnaces": 2, "coal": 16, "target": 23, "horizon": 6000},
    # seed 9054; offset_patch, 1 line, 12-step plan
    {"family": "offset_patch", "ore": [5, 5, 11, 11], "walls": [], "start": [11.4, -3.6], "drills": 1, "furnaces": 1, "coal": 8, "target": 18, "horizon": 4800},
    # seed 9055; obstructed_patch, 1 line, 14-step plan
    {"family": "obstructed_patch", "ore": [-1, -5, 1, 5], "walls": [[5, -2], [5, -1], [5, 0], [5, 1], [5, 2]], "start": [-10.6, 3.3], "drills": 1, "furnaces": 2, "coal": 40, "target": 28, "horizon": 7200},
    # seed 9056; varied_patch, 1 line, 12-step plan
    {"family": "varied_patch", "ore": [-9, 0, -1, 8], "walls": [], "start": [-15.1, 9.5], "drills": 2, "furnaces": 1, "coal": 8, "target": 23, "horizon": 6000},
    # seed 9057; offset_patch, 2 lines, 18-step plan
    {"family": "offset_patch", "ore": [6, -12, 12, -6], "walls": [], "start": [-3.6, -10.3], "drills": 2, "furnaces": 2, "coal": 24, "target": 54, "horizon": 7200},
    # seed 9058; varied_patch, 2 lines, 17-step plan
    {"family": "varied_patch", "ore": [0, 5, 6, 9], "walls": [], "start": [0.1, -2.2], "drills": 2, "furnaces": 2, "coal": 40, "target": 46, "horizon": 6000},
    # seed 9059; varied_patch, 1 line, 11-step plan
    {"family": "varied_patch", "ore": [-8, 2, 0, 6], "walls": [], "start": [-10.1, 13.5], "drills": 2, "furnaces": 1, "coal": 6, "target": 13, "horizon": 4800},
    # seed 9060; narrow_patch, 1 line, 10-step plan
    {"family": "narrow_patch", "ore": [-5, -1, 4, 0], "walls": [[-6, -1], [-6, 0], [-6, 1]], "start": [-12.2, -0.1], "drills": 1, "furnaces": 2, "coal": 6, "target": 13, "horizon": 3600},
    # seed 9061; varied_patch, 2 lines, 15-step plan
    {"family": "varied_patch", "ore": [-2, -8, 2, -6], "walls": [], "start": [-9.9, -9.8], "drills": 2, "furnaces": 2, "coal": 8, "target": 24, "horizon": 3600},
    # seed 9062; cluttered_patch, 2 lines, 14-step plan
    {"family": "cluttered_patch", "ore": [-11, 4, -5, 10], "walls": [[-14, 1], [-14, 2], [-12, 3], [-12, 11], [-11, 3], [-11, 11], [-10, 11], [-9, 11]], "start": [4.0, 5.1], "drills": 2, "furnaces": 2, "coal": 12, "target": 16, "horizon": 2400},
    # seed 9063; square_patch, 1 line, 12-step plan
    {"family": "square_patch", "ore": [-3, -3, 3, 3], "walls": [], "start": [7.3, 4.6], "drills": 1, "furnaces": 1, "coal": 12, "target": 23, "horizon": 6000},
    # seed 9064; narrow_patch, 1 line, 12-step plan
    {"family": "narrow_patch", "ore": [-5, -1, 4, 0], "walls": [[6, -1], [6, 0], [6, 1]], "start": [3.2, 6.6], "drills": 1, "furnaces": 2, "coal": 24, "target": 23, "horizon": 6000},
    # seed 9065; cluttered_patch, 1 line, 12-step plan
    {"family": "cluttered_patch", "ore": [0, -6, 6, 0], "walls": [[8, 0], [8, 1]], "start": [-9.2, -2.7], "drills": 2, "furnaces": 1, "coal": 12, "target": 18, "horizon": 4800},
    # seed 9066; offset_patch, 1 line, 13-step plan
    {"family": "offset_patch", "ore": [5, -11, 11, -5], "walls": [], "start": [0.6, -13.5], "drills": 1, "furnaces": 1, "coal": 40, "target": 28, "horizon": 7200},
    # seed 9067; varied_patch, 1 line, 11-step plan
    {"family": "varied_patch", "ore": [-5, -4, -1, 0], "walls": [], "start": [5.0, -10.5], "drills": 1, "furnaces": 2, "coal": 6, "target": 13, "horizon": 3600},
    # seed 9068; open_patch, 1 line, 11-step plan
    {"family": "open_patch", "ore": [-3, -3, 3, 3], "walls": [], "start": [-9.8, 0.7], "drills": 1, "furnaces": 2, "coal": 8, "target": 18, "horizon": 4800},
    # seed 9069; offset_patch, 2 lines, 14-step plan
    {"family": "offset_patch", "ore": [-12, 6, -6, 12], "walls": [], "start": [-4.4, -2.2], "drills": 2, "furnaces": 2, "coal": 8, "target": 14, "horizon": 2400},
    # seed 9070; offset_patch, 2 lines, 14-step plan
    {"family": "offset_patch", "ore": [-11, 5, -5, 11], "walls": [], "start": [-12.4, 19.4], "drills": 2, "furnaces": 2, "coal": 6, "target": 14, "horizon": 2400},
    # seed 9071; varied_patch, 2 lines, 17-step plan
    {"family": "varied_patch", "ore": [6, 1, 12, 9], "walls": [], "start": [-3.4, 6.3], "drills": 2, "furnaces": 2, "coal": 12, "target": 26, "horizon": 6000},
    # seed 9072; obstructed_patch, 2 lines, 18-step plan
    {"family": "obstructed_patch", "ore": [-1, -5, 1, 5], "walls": [[5, -2], [5, -1], [5, 0], [5, 1], [5, 2]], "start": [12.3, 1.3], "drills": 2, "furnaces": 2, "coal": 6, "target": 26, "horizon": 7200},
    # seed 9073; square_patch, 2 lines, 16-step plan
    {"family": "square_patch", "ore": [-3, -3, 3, 3], "walls": [], "start": [7.7, 10.5], "drills": 2, "furnaces": 2, "coal": 8, "target": 26, "horizon": 4800},
    # seed 9074; varied_patch, 1 line, 9-step plan
    {"family": "varied_patch", "ore": [5, 10, 13, 12], "walls": [], "start": [14.4, 2.1], "drills": 2, "furnaces": 1, "coal": 16, "target": 8, "horizon": 2400},
    # seed 9075; varied_patch, 2 lines, 16-step plan
    {"family": "varied_patch", "ore": [7, -6, 11, -2], "walls": [], "start": [8.0, -12.7], "drills": 2, "furnaces": 2, "coal": 40, "target": 36, "horizon": 4800},
    # seed 9076; cluttered_patch, 1 line, 11-step plan
    {"family": "cluttered_patch", "ore": [-11, 3, -5, 9], "walls": [[-11, 0], [-10, 0], [-10, 10], [-9, 10], [-8, 10], [-7, 10]], "start": [-2.4, -4.5], "drills": 1, "furnaces": 2, "coal": 40, "target": 18, "horizon": 4800},
    # seed 9077; offset_patch, 1 line, 13-step plan
    {"family": "offset_patch", "ore": [-12, 6, -6, 12], "walls": [], "start": [-2.0, -1.1], "drills": 1, "furnaces": 2, "coal": 6, "target": 13, "horizon": 7200},
    # seed 9078; cluttered_patch, 1 line, 13-step plan
    {"family": "cluttered_patch", "ore": [-2, 4, 4, 10], "walls": [[-5, 10], [-5, 11]], "start": [-2.9, 18.6], "drills": 2, "furnaces": 1, "coal": 8, "target": 27, "horizon": 7200},
    # seed 9079; obstructed_patch, 2 lines, 18-step plan
    {"family": "obstructed_patch", "ore": [-1, -5, 1, 5], "walls": [[5, -2], [5, -1], [5, 0], [5, 1], [5, 2]], "start": [3.6, 9.9], "drills": 2, "furnaces": 2, "coal": 16, "target": 54, "horizon": 7200},
    # seed 9080; narrow_patch, 1 line, 13-step plan
    {"family": "narrow_patch", "ore": [-5, -1, 4, 0], "walls": [[6, -1], [6, 0], [6, 1]], "start": [6.7, 3.0], "drills": 1, "furnaces": 2, "coal": 24, "target": 28, "horizon": 7200},
    # seed 9081; narrow_patch, 1 line, 10-step plan
    {"family": "narrow_patch", "ore": [-1, -5, 0, 4], "walls": [[-6, -1], [-6, 0], [-6, 1]], "start": [-3.1, 11.3], "drills": 1, "furnaces": 1, "coal": 6, "target": 13, "horizon": 3600},
    # seed 9082; varied_patch, 1 line, 11-step plan
    {"family": "varied_patch", "ore": [-7, -5, 1, -3], "walls": [], "start": [-6.1, 4.4], "drills": 1, "furnaces": 1, "coal": 8, "target": 18, "horizon": 4800},
    # seed 9083; square_patch, 1 line, 9-step plan
    {"family": "square_patch", "ore": [-3, -3, 3, 3], "walls": [], "start": [0.2, -8.9], "drills": 2, "furnaces": 1, "coal": 40, "target": 8, "horizon": 2400},
    # seed 9084; varied_patch, 1 line, 12-step plan
    {"family": "varied_patch", "ore": [-3, -13, 3, -5], "walls": [], "start": [6.1, -0.6], "drills": 1, "furnaces": 2, "coal": 16, "target": 23, "horizon": 6000},
    # seed 9085; narrow_patch, 2 lines, 18-step plan
    {"family": "narrow_patch", "ore": [-5, -1, 4, 0], "walls": [[6, -1], [6, 0], [6, 1]], "start": [6.5, -10.3], "drills": 2, "furnaces": 2, "coal": 24, "target": 54, "horizon": 7200},
    # seed 9086; obstructed_patch, 2 lines, 17-step plan
    {"family": "obstructed_patch", "ore": [-1, -5, 1, 5], "walls": [[5, -2], [5, -1], [5, 0], [5, 1], [5, 2]], "start": [1.5, 9.1], "drills": 2, "furnaces": 2, "coal": 40, "target": 46, "horizon": 6000},
    # seed 9087; cluttered_patch, 2 lines, 16-step plan
    {"family": "cluttered_patch", "ore": [-10, -3, -4, 3], "walls": [[-13, 4], [-12, 4], [-11, -5], [-11, 4], [-10, -5], [-9, -5]], "start": [-10.9, 9.3], "drills": 2, "furnaces": 2, "coal": 8, "target": 26, "horizon": 4800},
    # seed 9088; cluttered_patch, 2 lines, 18-step plan
    {"family": "cluttered_patch", "ore": [-8, 5, -2, 11], "walls": [[-7, 3], [-6, 3], [-5, 3], [0, 3], [0, 4], [0, 5], [0, 6]], "start": [-13.9, 12.6], "drills": 2, "furnaces": 2, "coal": 24, "target": 54, "horizon": 7200},
    # seed 9089; offset_patch, 1 line, 9-step plan
    {"family": "offset_patch", "ore": [5, 5, 11, 11], "walls": [], "start": [19.3, 7.6], "drills": 1, "furnaces": 1, "coal": 24, "target": 8, "horizon": 2400},
    # seed 9090; cluttered_patch, 1 line, 10-step plan
    {"family": "cluttered_patch", "ore": [5, -11, 11, -5], "walls": [[3, -3], [4, -13], [4, -3], [5, -13], [5, -3], [6, -3]], "start": [16.6, -3.3], "drills": 2, "furnaces": 1, "coal": 8, "target": 13, "horizon": 3600},
    # seed 9091; cluttered_patch, 2 lines, 17-step plan
    {"family": "cluttered_patch", "ore": [-1, -6, 5, 0], "walls": [[2, 2], [3, 2], [4, -9], [5, -9], [6, -9], [7, -9]], "start": [5.6, 6.9], "drills": 2, "furnaces": 2, "coal": 8, "target": 26, "horizon": 6000},
    # seed 9092; offset_patch, 1 line, 10-step plan
    {"family": "offset_patch", "ore": [-12, 6, -6, 12], "walls": [], "start": [-10.6, -1.5], "drills": 1, "furnaces": 2, "coal": 40, "target": 13, "horizon": 3600},
    # seed 9093; varied_patch, 1 line, 13-step plan
    {"family": "varied_patch", "ore": [8, -5, 12, 3], "walls": [], "start": [2.8, -9.6], "drills": 2, "furnaces": 2, "coal": 12, "target": 28, "horizon": 7200},
    # seed 9094; cluttered_patch, 2 lines, 15-step plan
    {"family": "cluttered_patch", "ore": [4, 3, 10, 9], "walls": [[3, 4], [3, 5]], "start": [10.7, -3.8], "drills": 2, "furnaces": 2, "coal": 8, "target": 26, "horizon": 3600},
    # seed 9095; offset_patch, 1 line, 10-step plan
    {"family": "offset_patch", "ore": [5, -11, 11, -5], "walls": [], "start": [13.5, 1.8], "drills": 1, "furnaces": 1, "coal": 24, "target": 13, "horizon": 3600},
    # seed 9096; offset_patch, 1 line, 9-step plan
    {"family": "offset_patch", "ore": [6, 6, 12, 12], "walls": [], "start": [9.1, 20.8], "drills": 2, "furnaces": 1, "coal": 6, "target": 8, "horizon": 2400},
    # seed 9097; cluttered_patch, 1 line, 11-step plan
    {"family": "cluttered_patch", "ore": [2, -6, 8, 0], "walls": [[0, -9], [0, -8], [0, -7], [0, -6], [0, -5], [1, -9], [8, 1], [9, 1], [10, 1]], "start": [4.8, 8.6], "drills": 2, "furnaces": 1, "coal": 16, "target": 18, "horizon": 4800},
    # seed 9098; obstructed_patch, 1 line, 10-step plan
    {"family": "obstructed_patch", "ore": [-1, -5, 1, 5], "walls": [[5, -2], [5, -1], [5, 0], [5, 1], [5, 2]], "start": [-8.0, -7.6], "drills": 1, "furnaces": 1, "coal": 8, "target": 13, "horizon": 3600},
    # seed 9099; open_patch, 2 lines, 14-step plan
    {"family": "open_patch", "ore": [-3, -3, 3, 3], "walls": [], "start": [-2.1, -11.5], "drills": 2, "furnaces": 2, "coal": 24, "target": 14, "horizon": 2400},
)
