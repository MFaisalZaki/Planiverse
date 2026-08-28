"""A side-scrolling platformer in pure Python: no ROM, no emulator, no dependencies.

This is the dependency-free counterpart to [`super_mario_land`](../gameboy/super_mario_land.py), and it
is a weaker claim than the other two pairs in this library. `puzznic` and `flipull` are twins
of their cartridges: their rules were derived from the real hardware and reproduce it, exactly
in one case and partly in the other. **This is not a twin of Super Mario Land.** Nothing here
was read off the cartridge. It is a platformer in the same genre, with physics chosen to be
simple, integral and stated, built so that there is a dependency-free environment of that
*shape* — run, jump, gaps, patrolling enemies, a flag at the end — to plan in.

Modelling a physics platformer faithfully is a much larger job than modelling a turn-based
puzzle, and a half-reverse-engineered one would be worse than none: it would look like a
prediction of the cartridge without being one. So this does not try, and says so.

## The physics, stated

Positions are in units, `TILE` units to a tile, the way the Game Boy itself counts. Mario is
one tile square. Each tick, in this order:

1. **Horizontal.** `left`/`right` accelerate towards a target speed — `WALK`, or `RUN` while
   `b` is held — by `ACCEL` a tick, and `FRICTION` slows you when nothing is held. Speed
   carries: what you are travelling at when you leave a ledge is what you cross the gap at,
   so where you start running from decides whether you make it.
2. **Jump.** Pressing `a` while standing sets `vy = JUMP_SPEED`. Holding it keeps the rise;
   releasing it while still rising cuts the climb to `JUMP_CUT`, which is what makes a short
   hop and a full jump different actions rather than the same one.
3. **Gravity.** `vy += GRAVITY`, capped at `MAX_FALL`.
4. **Collision.** Resolved on one axis at a time — horizontal first, then vertical — so a
   corner never lets you through.

Falling below the level kills. So does touching a hazard, and so does touching an enemy from
the side. Landing on an enemy from above kills the *enemy* and bounces you.

## What this has that the emulator environment does not

`is_terminal` is **exact about death**. `SuperMarioEnv` cannot tell you whether Mario died on
contact: it has a proximity test over the object array, and whether contact is fatal depends on
a power-up byte the memory map never confirmed, so it deliberately reports only the music track
changing. Here death is defined, so a planner prunes the moment it happens rather than playing
on into a position that no longer exists.

It is *not* a test for whether the level is still winnable, and nothing here claims to be.
Mario can be alive in a pit he cannot jump out of, and no environment in this library detects
that — deciding it in general means solving the level. Dead states are pruned; stuck ones cost
the planner its budget, the same as they would on the cartridge.

There is no timer, and no score. The cartridge has both; this does not model them.
"""
from planiverse.environments.base import Environment

#: Units to a tile — the Game Boy's own granularity, and enough for jump arcs to have shape.
TILE = 8

#: Horizontal speeds, in units per tick. `b` selects the faster one.
WALK, RUN = 2, 4

#: How fast you reach that speed, and how fast you lose it with nothing held. Momentum is
#: what makes position matter: reaching `RUN` from a standstill takes four ticks and ten
#: units of runway, and the speed you are carrying when you leave a ledge is the speed you
#: cross the gap at. Without it a run-up is free, every gap is either always or never
#: jumpable, and the levels collapse into "hold run and jump" — which is exactly what the
#: first draft of this file did.
ACCEL, FRICTION = 1, 1

#: Vertical. A jump from `JUMP_SPEED` rises for five ticks and 3.75 tiles, and the whole arc
#: lasts about eleven — far enough at `RUN` to clear a five-tile gap, and not at `WALK`. That
#: difference is the entire reason a run button exists.
GRAVITY, MAX_FALL, JUMP_SPEED, JUMP_CUT, BOUNCE = 2, 12, -12, -4, -8

#: `#` solid, ` ` air, `^` hazard, `E` an enemy's starting tile, `M` Mario's, `G` the flag.
SOLID, AIR, HAZARD, ENEMY, START, GOAL = "#", " ", "^", "E", "M", "G"

#: Buttons and how long to hold them. The vocabulary mirrors `super_mario_land`'s
#: `button,ticks` actions, minus `down`: there is no ducking in this model.
#:
#: The short hold has to be genuinely short or the jump cut never fires. `vy` climbs past
#: `JUMP_CUT` three ticks into a jump, so releasing any later than that changes nothing — a
#: first draft used `(4, 8, 12)` and all three `a` actions produced the same arc, which made
#: two thirds of the action set dead weight and a hop impossible.
BUTTON_SETS = ("a+right", "a+left", "b+right", "b+left", "a+b+right", "a+b+left")
HOLD_TICKS = (2, 6, 12)
SHORT_BUTTONS = ("nop", "right", "left")
SHORT_TICKS = 4

#: Enemies step one unit per tick and turn at a wall or a ledge.
ENEMY_SPEED = 1


def _actions():
    held = [f"{buttons},{ticks}" for ticks in HOLD_TICKS for buttons in BUTTON_SETS]
    return tuple(held + [f"{buttons},{SHORT_TICKS}" for buttons in SHORT_BUTTONS])


#: The full action vocabulary: 21 held-button combinations.
ACTION_NAMES = _actions()


def parse_level(text):
    """An ASCII level into `(tiles, mario_start, enemy_starts, goal)`.

    The `M`, `E` and `G` markers are stripped out of the tile map — they say where things
    begin, not what the terrain is — so the tiles that remain are only terrain.
    """
    rows = text.split("\n")
    width = max(len(row) for row in rows)
    tiles = [list(row.ljust(width)) for row in rows]
    start, goal, enemies = None, None, []
    for y, row in enumerate(tiles):
        for x, cell in enumerate(row):
            if cell == START:
                start, tiles[y][x] = (x * TILE, y * TILE), AIR
            elif cell == ENEMY:
                enemies.append((x * TILE, y * TILE))
                tiles[y][x] = AIR
            elif cell == GOAL:
                goal = (x, y)
    if start is None:
        raise ValueError("the level has no 'M' marking where Mario starts")
    if goal is None:
        raise ValueError("the level has no 'G' marking the flag")
    return tuple(tuple(row) for row in tiles), start, tuple(enemies), goal


def tile_at(tiles, x, y):
    """The terrain at a unit position. Outside the map is air, except below it."""
    tx, ty = x // TILE, y // TILE
    if ty < 0 or not 0 <= tx < len(tiles[0]):
        return AIR
    if ty >= len(tiles):
        return AIR
    return tiles[ty][tx]


def covers(tiles, x, y, kinds):
    """Does the one-tile box at `(x, y)` overlap any tile of one of `kinds`?"""
    for corner_x in (x, x + TILE - 1):
        for corner_y in (y, y + TILE - 1):
            if tile_at(tiles, corner_x, corner_y) in kinds:
                return True
    return False


def blocked(tiles, x, y):
    return covers(tiles, x, y, (SOLID,))


def _move_enemy(tiles, x, y, direction):
    """One enemy step. Turns at a wall, and at the edge of what it is standing on."""
    nx = x + direction * ENEMY_SPEED
    ahead = nx + (TILE - 1 if direction > 0 else 0)
    turn = (blocked(tiles, nx, y)
            or tile_at(tiles, ahead, y + TILE) != SOLID
            or not 0 <= nx <= (len(tiles[0]) - 1) * TILE)
    if turn:
        return x, y, -direction
    return nx, y, direction


def _settle(tiles, x, y):
    """Drop Mario to the floor under his starting tile.

    Without this the opening state is airborne, so `on_ground` is false, the first action
    cannot be a jump, and every level begins with a wasted step. Only Mario moves — the
    enemies have not had a tick yet.
    """
    while not blocked(tiles, x, y + 1) and y // TILE <= len(tiles):
        y += 1
    return x, y


class PlatformerAction:
    """A set of buttons held for a number of ticks, spelled `"a+right,8"`."""

    #: What each button costs, so that a plan can be scored by effort rather than by length.
    COSTS = {"a": 2, "b": 2, "left": 1, "right": 1, "nop": 0}

    def __init__(self, name):
        if name not in ACTION_NAMES:
            raise ValueError(f"unknown action: {name!r}")
        self.name = name
        buttons, ticks = name.split(",")
        self.buttons = frozenset(buttons.split("+"))
        self.ticks = int(ticks)

    def cost(self):
        return sum(self.COSTS[button] for button in self.buttons) * self.ticks

    def __eq__(self, other):
        return isinstance(other, PlatformerAction) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return (self.cost(), self.name) < (other.cost(), other.name)

    def __str__(self):
        return self.name

    __repr__ = __str__


class PlatformerState:
    """Mario's position and velocity, and where the living enemies are.

    Enemy positions are carried in the state rather than derived from a tick counter. It
    would be tidier to keep a clock and compute them, but then two identical configurations
    reached at different times would compare unequal and search would never close anything —
    the state space would be infinite for no reason.
    """

    def __init__(self, tiles, x, y, vx, vy, on_ground, enemies, goal, dead=False, depth=0):
        self.tiles = tiles
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.on_ground = on_ground
        self.enemies = tuple(sorted(enemies))
        self.goal = goal
        self.dead = dead
        self.depth = depth

        self.tile_x, self.tile_y = x // TILE, y // TILE
        # Two sets of atoms, coarse and exact. The coarse ones are what a heuristic reads and
        # what novelty is worth measuring over; the exact ones are there because `literals`
        # has to be a faithful projection of the state. Enemies move one unit a tick, so with
        # tile-granularity atoms alone two genuinely different positions can share a literal
        # set — and then an action that really did change the world looks like a self-loop.
        literals = [f"at(mario, {self.tile_x}, {self.tile_y})",
                    f"progress({self.tile_x})",
                    f"speed({vx})",
                    f"falling({1 if vy > 0 else 0})",
                    f"grounded({1 if on_ground else 0})",
                    f"pos(mario, {x}, {y})"]
        literals += [f"at(enemy, {ex // TILE}, {ey // TILE})" for ex, ey, _ in self.enemies]
        literals += [f"pos(enemy, {ex}, {ey}, {heading})" for ex, ey, heading in self.enemies]
        literals.append(f"enemies({len(self.enemies)})")
        if dead:
            literals.append("dead()")
        self.literals = frozenset(literals)

    def key(self):
        return (self.x, self.y, self.vx, self.vy, self.on_ground, self.enemies, self.dead)

    def __eq__(self, other):
        return isinstance(other, PlatformerState) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        rows = [list(row) for row in self.tiles]
        gx, gy = self.goal
        if 0 <= gy < len(rows) and 0 <= gx < len(rows[0]):
            rows[gy][gx] = GOAL
        for ex, ey, _ in self.enemies:
            ty, tx = ey // TILE, ex // TILE
            if 0 <= ty < len(rows) and 0 <= tx < len(rows[0]):
                rows[ty][tx] = ENEMY
        if 0 <= self.tile_y < len(rows) and 0 <= self.tile_x < len(rows[0]):
            rows[self.tile_y][self.tile_x] = "x" if self.dead else START
        status = "dead" if self.dead else ("grounded" if self.on_ground else "airborne")
        return "\n".join("".join(row) for row in rows) + \
            f"\nmario: ({self.tile_x}, {self.tile_y})  {status}  " \
            f"vx {self.vx} vy {self.vy}  enemies: {len(self.enemies)}"

    def __repr__(self):
        return (f"<PlatformerState(x={self.tile_x}, y={self.tile_y}, vx={self.vx}, "
                f"vy={self.vy}, enemies={len(self.enemies)}, dead={self.dead})>")


class PlatformerGame(Environment):
    """A run-and-jump level as a planning problem. Needs nothing installed."""

    def __init__(self, levels=None):
        """`levels` overrides the shipped set with your own ASCII levels.

        Handy for a one-off problem, and it is what the tests use to pin the physics down on
        boards built for the purpose rather than on whichever shipped level happens to have
        the right shape.
        """
        super().__init__("platformer")
        self.levels = tuple(levels) if levels is not None else LEVELS
        self.index = 0
        self.state = None
        self.state_history = []

    # ------------------------------------------------------------------- the contract

    def fix_index(self, index):
        if not 0 <= index < len(self.levels):
            raise IndexError(
                f"Invalid index: {index}. There are {len(self.levels)} levels, so the index "
                f"must be 0-{len(self.levels) - 1}.")
        self.index = index

    def reset(self):
        tiles, start, enemies, goal = parse_level(self.levels[self.index])
        placed = tuple((x, y, 1) for x, y in enemies)
        x, y = _settle(tiles, *start)
        self.state = PlatformerState(tiles, x, y, 0, 0, True, placed, goal)
        self.state_history = [self.state]
        return self.state, {"level": self.index,
                            "width": len(tiles[0]),
                            "enemies": len(placed),
                            "goal": goal}

    def is_goal(self, state):
        """Mario's box overlaps the flag tile.

        Overlap rather than an exact tile match: he moves up to `MAX_FALL` units in a tick,
        so a test on tile coordinates alone would let a fast fall skip straight past the flag
        and count as a miss.
        """
        gx, gy = state.goal
        return not state.dead and abs(state.x - gx * TILE) < TILE \
            and abs(state.y - gy * TILE) < TILE

    def is_terminal(self, state):
        """Mario died — fell out of the level, touched a hazard, or was hit by an enemy.

        Exact about death, unlike `SuperMarioEnv.is_terminal`, which cannot tell a fatal touch
        from a survivable one and so reports only the death music. It is not a test for
        whether the level is still winnable: Mario can be alive in a pit he cannot leave, and
        deciding that in general means solving the level.
        """
        return state.dead

    def successors(self, state):
        if state.dead or self.is_goal(state):
            return []
        successors = []
        for name in ACTION_NAMES:
            action = PlatformerAction(name)
            successor = self.__advance__(state, action)
            if successor == state:
                continue
            successors.append((action, successor))
        return successors

    def simulate(self, plan):
        state, _ = self.reset()
        trace = [state]
        for action in plan:
            trace.append(self.__advance__(trace[-1], action))
        return trace

    def step(self, action):
        if self.state is None:
            raise ValueError("Game not initialized. Call reset() first.")
        before = self.state.tile_x
        self.state = self.__advance__(self.state, action)
        self.state_history.append(self.state)
        return self.state, self.state.tile_x - before

    def get_actions(self):
        return [PlatformerAction(name) for name in ACTION_NAMES]

    def render(self):
        rendered = [str(state) for state in self.state_history]
        for step, text in enumerate(rendered):
            print(f"Step: {step}")
            print(text)
            print("--------------")
        return rendered

    # --------------------------------------------------------------------- the engine

    def __advance__(self, state, action):
        if state.dead or self.is_goal(state):
            return state
        if isinstance(action, str):
            action = PlatformerAction(action)

        tiles = state.tiles
        x, y, vx, vy = state.x, state.y, state.vx, state.vy
        on_ground = state.on_ground
        enemies = list(state.enemies)
        jumping = "a" in action.buttons
        speed = RUN if "b" in action.buttons else WALK
        heading = 1 if "right" in action.buttons else -1 if "left" in action.buttons else 0
        target = speed * heading

        for _ in range(action.ticks):
            # 1. Horizontal. Accelerate towards the target, or coast down to a stop, then
            #    move — resolved on its own axis so a corner never lets you through.
            if heading:
                vx = min(vx + ACCEL, target) if target > vx else max(vx - ACCEL, target)
            elif vx:
                vx = max(0, vx - FRICTION) if vx > 0 else min(0, vx + FRICTION)
            if vx:
                stepped = x + vx
                if blocked(tiles, stepped, y) or \
                        not 0 <= stepped <= (len(tiles[0]) - 1) * TILE:
                    vx = 0                     # a wall takes your speed away
                else:
                    x = stepped

            # 2. Jump, and the cut that makes a hop different from a full jump.
            if jumping and on_ground:
                vy, on_ground = JUMP_SPEED, False
            elif not jumping and vy < JUMP_CUT:
                vy = JUMP_CUT

            # 3. Gravity.
            vy = min(MAX_FALL, vy + GRAVITY)

            # 4. Vertical, resolved after the horizontal move. `falling` is recorded before
            #    the collision because landing on the floor zeroes `vy`, and a stomp is
            #    judged on having been on the way down — a fall fast enough to reach the
            #    floor and an enemy in the same tick would otherwise read as standing still.
            was_y, stepped, falling = y, y + vy, vy > 0
            if blocked(tiles, x, stepped):
                # Walk it to the contact point rather than snapping, so a fast fall lands on
                # the floor it passed through instead of inside it.
                direction = 1 if vy > 0 else -1
                while not blocked(tiles, x, y + direction) and y != stepped:
                    y += direction
                on_ground, vy = vy > 0, 0
            else:
                y, on_ground = stepped, False

            # Contact is checked twice a tick: once after Mario moves and once after the
            # enemies do. Checking only one of the two lets a head-on pair swap places
            # without ever overlapping, and walk through each other.
            for stage in (0, 1):
                if stage:
                    enemies = [_move_enemy(tiles, ex, ey, facing)
                               for ex, ey, facing in enemies]
                resolved = self.__resolve__(tiles, x, y, was_y, vy, falling, enemies)
                if resolved is None:
                    return PlatformerState(tiles, x, y, vx, vy, on_ground, enemies,
                                           state.goal, dead=True, depth=state.depth + 1)
                x, y, vy, stomped, enemies = resolved
                if stomped:
                    on_ground = False

            if y // TILE > len(tiles):
                return PlatformerState(tiles, x, y, vx, vy, False, enemies, state.goal,
                                       dead=True, depth=state.depth + 1)
            if covers(tiles, x, y, (HAZARD,)):
                return PlatformerState(tiles, x, y, vx, vy, on_ground, enemies, state.goal,
                                       dead=True, depth=state.depth + 1)

            reached = PlatformerState(tiles, x, y, vx, vy, on_ground, enemies, state.goal,
                                      depth=state.depth + 1)
            if self.is_goal(reached):
                return reached

        return PlatformerState(tiles, x, y, vx, vy, on_ground, enemies, state.goal,
                               depth=state.depth + 1)

    def __resolve__(self, tiles, x, y, was_y, vy, falling, enemies):
        """Enemy contact. Returns `(x, y, vy, stomped, enemies)`, or `None` if Mario died.

        Landing on one from above kills it and bounces; anything else kills Mario, which is
        the classic rule and the reason a jump is an attack as well as a way across.

        "From above" is judged on where Mario was *before* the vertical move, not where he
        ended up. Testing the landing position alone makes a fast fall unstompable: at
        `MAX_FALL` he crosses the whole of an enemy's upper half within a single tick, so he
        is never observed above it and dies on something he plainly landed on.
        """
        enemies = list(enemies)
        for index, (ex, ey, _) in enumerate(enemies):
            if abs(x - ex) >= TILE or abs(y - ey) >= TILE:
                continue
            if falling and was_y + TILE <= ey + TILE // 2:
                return x, y, BOUNCE, True, enemies[:index] + enemies[index + 1:]
            return None
        return x, y, vy, False, enemies


#: Levels, in rising order of difficulty. Each was checked by search before being shipped —
#: see `tests/test_platformer.py`, which re-derives a route through every one of them.
LEVELS = (
    """\


           ####


M     E E E       G
############  ######
############  ######""",
    """\


           E
         ####
   ###   ####
M  ###  E########           E       G
#################  ##########    #####
#################  ##########    #####""",
    """\


        ###
          #####
         ######
M   ##^########                  ###   E  G
###############  #######   #################
###############  #######   #################""",
    """\





M   E                      E    E   G
######^#  #^##     ###########  ######
########  ####     ###########  ######""",
    """\



           ####          ##
                                   E
M          E                    E #####   G
######     ###   #####  ###   #########  ###
######     ###   #####  ###   #########  ###""",
    """\


       ####
         ####          ^#
            ####       ##
M      ####E           ##   E             G
#############     ############   ###########
#############     ############   ###########""",
    """\


      #####
     ###  ##
                    ####
M  ####^#           #### E      EE        G
###########     ##########     ###   #######
###########     ##########     ###   #######""",
    """\


      ###
                 #####
      ###     E
M   E ### E####               G
#################    ####   ####
#################    ####   ####""",
)

#: BFWS(w=2) expansions when each level was accepted, in the same order — the ramp,
#: recorded rather than asserted. `tests/test_platformer.py` re-derives a route
#: through every one of them, so a level that stops being finishable fails the suite.
MEASURED_EXPANSIONS = (4, 7, 7, 66, 385, 468, 482, 1302)
