"""Flipull in pure Python: no ROM, no emulator, no dependencies.

The rules are implemented directly, the way [`puzznic`](puzznic.py) is, so this is a
dependency-free benchmark rather than a prediction of the original game.

## The rules, stated

The player stands to the right of a wall of blocks holding one of them, and moves up and down
the rows or throws. A throw sends the held block leftward along the player's row:

1. It **destroys** each block of its own type it meets, and keeps going.
2. The first block of a **different** type takes the thrown block's place, and comes back into
   the player's hand: a swap.
3. If the *very first* block it meets is a different type, **nothing happens at all**: the
   block flies out and comes back, and the position is unchanged.
4. Every destroyed cell **collapses its column**: everything above it falls one row.

A stage is cleared when few enough blocks are left.

Rule 3 is the one that makes this a puzzle rather than a shuffling exercise. Without it every
throw would be legal and the board would be a permutation group; with it, most rows are
refused most of the time and choosing which row to stand on is the whole game.

## How faithful is this to the cartridge?

Partly, and the honest answer is worth more than a claim. The rules above were derived by
driving Flipull and predicting what it would do, and they reproduce it **exactly**
(field and hand, cell for cell) for throws taken level with the wall in the positions
checked. Over a longer automated comparison they agreed on about half of the level throws and
four in five of the throws from above the wall, so something more is going on that has not
been pinned down: the staircase, or a bounce, or a fall the model does not have.

So this is a Flipull-*like* environment with a stated rule set, not a clone. What it is good
for is a well-defined, dependency-free planning problem; what it is not good for is
predicting the original game.

## What a stated rule set buys

Because the rules are known here, `is_terminal` is **exact**: a position is a dead end when no
throw from any row would connect. An emulator cannot compute that (it does not know what a
throw hits) and can only tell you the clock ran out. Dead-end detection is most of what makes
a puzzle searchable, so this is not a small difference.

## Where the stages came from

The 32 stages replicate the cartridge's own stage table: stage for stage, the same board
size and the same CLEAR target as Flipull. The arrangements are generated rather
than copied, for two reasons. First, the cartridge has no canonical arrangements to copy:
it draws each stage's block layout from an RNG seeded by boot timing, and its ROM stores
only the block total and the CLEAR target per stage. Second, arrangements the cartridge
happens to draw are mostly unreachable to their targets under this module's stated rules
(26 of 32 in one deterministic draw, proved by exhausting their state spaces), which is a
measure of how much the unpinned throw mechanics matter. So each board here was produced
randomly, explored exhaustively, and kept only when the fewest blocks it can be reduced
to is exactly the cartridge's target. `tests/test_flipull.py` re-derives a solution for
every one of them, so a stage whose goal drifts out of reach fails the suite rather than
quietly wasting a planner's budget.

## Generating stages

`generate_instance(seed, ...)` draws stages the same way the bundled ones were made: a
random wall of blocks of the requested size, explored exhaustively, and kept only when the
fewest blocks it can be reduced to is low enough to be worth playing for. That count becomes
the stage's CLEAR target unless the caller names one, so a generated stage is always
clearable and never clearable by accident.
"""
from planiverse.environments.base import Environment
from planiverse.environments.generation import draw_until, from_profile, rng

#: `1`-`4` are block types, `#` is wall, and a space is empty. There is no staircase: the
#: cartridge has a fixed diagonal one at the left of some stages, and since it is not clear
#: what a thrown block does when it meets it, this module leaves it out rather than guess.
WALL, EMPTY = "#", " "
BLOCK_TYPES = ("1", "2", "3", "4")

#: `(stage, clear_target)`, matching the cartridge's own 32-entry stage table: the same
#: board size (25, 30 or 36 blocks) and the same CLEAR target (9 down to 6) as each stage
#: of Flipull. The arrangements are this module's own, because the cartridge has
#: none to copy: it draws each stage's arrangement from an RNG seeded by boot timing, so
#: there is no canonical layout per stage, only a contract. Each board here was generated
#: randomly and explored exhaustively, and kept only when the fewest blocks it can be
#: reduced to under this module's rules is *exactly* the cartridge's target, so a stage is
#: only cleared by playing it out rather than by chipping away at it.
#:
#: The player starts on the bottom row of the wall, as on the cartridge: the position where
#: `down` does nothing.
STAGES = (
    ("#######\n#     #\n#42442#\n#44311#\n#34431#\n#24133#\n#31211#\n#######", 9),
    ("#######\n#     #\n#24111#\n#34422#\n#21334#\n#11141#\n#23421#\n#######", 9),
    ("#######\n#     #\n#34441#\n#24231#\n#43413#\n#42442#\n#44413#\n#######", 8),
    ("#######\n#     #\n#31131#\n#42221#\n#14132#\n#23243#\n#22333#\n#33231#\n#######", 8),
    ("#######\n#     #\n#21142#\n#14243#\n#14122#\n#44434#\n#42222#\n#12211#\n#######", 8),
    ("#######\n#     #\n#43313#\n#43444#\n#23331#\n#44133#\n#11223#\n#22121#\n#######", 7),
    ("#######\n#     #\n#41331#\n#41233#\n#13341#\n#42314#\n#42433#\n#41122#\n#######", 7),
    ("########\n#      #\n#423324#\n#441431#\n#122341#\n#232321#\n#434441#\n#142312#\n########", 7),
    ("########\n#      #\n#144144#\n#232242#\n#443124#\n#321242#\n#141314#\n#231434#\n########", 8),
    ("########\n#      #\n#114411#\n#331224#\n#241444#\n#231442#\n#442321#\n#343142#\n########", 8),
    ("########\n#      #\n#121241#\n#334131#\n#422343#\n#114244#\n#412144#\n#321111#\n########", 8),
    ("#######\n#     #\n#23222#\n#12324#\n#42442#\n#12141#\n#31214#\n#44412#\n#######", 7),
    ("#######\n#     #\n#21324#\n#42214#\n#32323#\n#43342#\n#21214#\n#31311#\n#######", 7),
    ("########\n#      #\n#442434#\n#133234#\n#413421#\n#132114#\n#243224#\n#414444#\n########", 7),
    ("########\n#      #\n#131113#\n#433111#\n#331112#\n#444344#\n#132211#\n#133443#\n########", 7),
    ("#######\n#     #\n#22244#\n#21323#\n#41142#\n#21331#\n#14424#\n#12133#\n#######", 7),
    ("#######\n#     #\n#23214#\n#22441#\n#12424#\n#41222#\n#34144#\n#42113#\n#######", 7),
    ("#######\n#     #\n#43132#\n#11324#\n#11433#\n#41334#\n#24432#\n#31233#\n#######", 7),
    ("#######\n#     #\n#12334#\n#32431#\n#31244#\n#42232#\n#14221#\n#33124#\n#######", 7),
    ("#######\n#     #\n#43234#\n#44431#\n#32322#\n#43312#\n#42422#\n#34334#\n#######", 7),
    ("#######\n#     #\n#21121#\n#21142#\n#14443#\n#24441#\n#44324#\n#11234#\n#######", 7),
    ("#######\n#     #\n#42342#\n#12242#\n#34424#\n#43113#\n#23344#\n#13424#\n#######", 6),
    ("#######\n#     #\n#23214#\n#44123#\n#24432#\n#43122#\n#42421#\n#24141#\n#######", 6),
    ("########\n#      #\n#342424#\n#434333#\n#432324#\n#423341#\n#121221#\n#112131#\n########", 6),
    ("########\n#      #\n#343232#\n#341111#\n#414222#\n#232222#\n#423322#\n#443113#\n########", 6),
    ("########\n#      #\n#131133#\n#243414#\n#214313#\n#342112#\n#314211#\n#231433#\n########", 6),
    ("########\n#      #\n#443443#\n#413113#\n#341411#\n#433413#\n#411111#\n#442234#\n########", 6),
    ("########\n#      #\n#144412#\n#432431#\n#143323#\n#212141#\n#111234#\n#311123#\n########", 6),
    ("########\n#      #\n#412114#\n#123122#\n#213124#\n#222442#\n#243141#\n#232134#\n########", 6),
    ("#######\n#     #\n#22121#\n#44314#\n#12433#\n#23323#\n#21314#\n#######", 6),
    ("#######\n#     #\n#13113#\n#24124#\n#14213#\n#42444#\n#42433#\n#######", 6),
    ("#######\n#     #\n#21422#\n#31342#\n#42134#\n#11441#\n#21123#\n#######", 6),
)


#: Stages the generator drew, kept after the cartridge's so that `set_index` offers them
#: too: `(stage, clear_target)` pairs like `STAGES`, each `generate_instance(seed)` for the
#: seed in its comment, with the size and target of one of the cartridge's stages.
GENERATED_STAGES = (
    # seed 2000, defaults; 8703 positions explored
    ("########\n#      #\n#344234#\n#414223#\n#424234#\n#144243#\n#242242#\n#422214#\n########", 7),
    # seed 2001, defaults; 4005 positions explored
    ("########\n#      #\n#313424#\n#222112#\n#224242#\n#442211#\n#433132#\n#441114#\n########", 7),
    # seed 2002, defaults; 11375 positions explored
    ("########\n#      #\n#114214#\n#122244#\n#134214#\n#311131#\n#422442#\n#142112#\n########", 7),
    # seed 2003, defaults; 364 positions explored
    ("#######\n#     #\n#31242#\n#22244#\n#21331#\n#43422#\n#11322#\n#13143#\n#######", 7),
    # seed 2004, defaults; 1953 positions explored
    ("#######\n#     #\n#21121#\n#42413#\n#14111#\n#23442#\n#14114#\n#14214#\n#######", 6),
    # seed 2005, defaults; 2541 positions explored
    ("########\n#      #\n#343113#\n#324223#\n#131114#\n#412444#\n#123344#\n#311112#\n########", 6),
    # seed 2006, defaults; 6083 positions explored
    ("########\n#      #\n#423433#\n#134341#\n#211324#\n#414432#\n#213334#\n#121333#\n########", 7),
    # seed 2007, defaults; 1443 positions explored
    ("########\n#      #\n#213243#\n#222324#\n#412322#\n#121344#\n#144444#\n#434444#\n########", 6),
    # seed 2008, defaults; 557 positions explored
    ("#######\n#     #\n#24411#\n#11341#\n#13342#\n#41413#\n#11312#\n#33414#\n#######", 7),
    # seed 2009, defaults; 1047 positions explored
    ("#######\n#     #\n#44231#\n#31121#\n#43442#\n#44333#\n#44434#\n#31131#\n#######", 7),
    # seed 2010, defaults; 266 positions explored
    ("#######\n#     #\n#42334#\n#43314#\n#11332#\n#23223#\n#32223#\n#######", 6),
    # seed 2011, defaults; 2724 positions explored
    ("#######\n#     #\n#33131#\n#32114#\n#24322#\n#21313#\n#13333#\n#31121#\n#######", 7),
    # seed 2012, defaults; 1081 positions explored
    ("#######\n#     #\n#23321#\n#12211#\n#13221#\n#14113#\n#44224#\n#42333#\n#######", 7),
    # seed 2013, defaults; 223 positions explored
    ("#######\n#     #\n#31131#\n#22411#\n#41313#\n#23113#\n#33134#\n#######", 6),
    # seed 2014, defaults; 859 positions explored
    ("########\n#      #\n#234344#\n#244441#\n#121442#\n#133334#\n#224344#\n#312444#\n########", 8),
    # seed 2015, defaults; 3006 positions explored
    ("#######\n#     #\n#31321#\n#33234#\n#13342#\n#43332#\n#33112#\n#44311#\n#######", 7),
)

#: Everything `set_index` selects between: the cartridge's stages, then the generated ones.
INSTANCES = STAGES + GENERATED_STAGES


def profile(stage):
    """The contract of a stage, as the options `generate_instance` takes to draw one like
    it: the wall's width and height in blocks, and its CLEAR target."""
    text, target = stage
    grid = parse_stage(text)
    rows = block_rows(grid)
    width = sum(1 for cell in grid[rows[0]] if cell in BLOCK_TYPES)
    return {"width": width, "height": len(rows), "clear_target": target}


def parse_stage(text):
    """An ASCII stage into a grid of single characters, padded to a rectangle."""
    rows = text.split("\n")
    width = max(len(row) for row in rows)
    return [list(row.ljust(width)) for row in rows]


def generate_stage(random_, width=5, height=5, types=4):
    """A random wall of `height` rows of `width` blocks, in the stage alphabet.

    The same shape as the bundled stages: a ring of walls, one empty row above the wall for
    the player to stand on, and every cell of the wall itself a block of one of `types`
    types. Whether it is worth playing is a separate question, answered by exploring it; see
    `fewest_blocks_reachable`.
    """
    kinds = BLOCK_TYPES[:types]
    rows = [WALL * (width + 2), WALL + EMPTY * width + WALL]
    rows += [WALL + "".join(random_.choice(kinds) for _ in range(width)) + WALL
             for _ in range(height)]
    rows.append(WALL * (width + 2))
    return "\n".join(rows)


def fewest_blocks_reachable(text, limit=200_000):
    """Explore every position reachable from a stage's opening, and report the fewest blocks
    any of them has left. Returns `(fewest, exhausted, plan, explored)`: `plan` reaches a
    position with that many, and `explored` is how many positions were seen doing it.

    Exhaustive rather than goal-directed, because the question is not "can the target be
    reached" but "what is the lowest target this board could honestly be given". `exhausted`
    is False when `limit` positions were seen first, in which case `fewest` is only a bound.
    """
    game = FlipullGame()
    game.set_instance((text, 0))          # a target of 0 means no position counts as cleared
    start, _ = game.reset()
    parents, frontier, best = {start: None}, [start], start
    while frontier:
        if len(parents) >= limit:
            return best.blocks_remaining, False, _plan_to(parents, best), len(parents)
        state = frontier.pop()
        for action, successor in game.successors(state):
            if successor in parents:
                continue
            parents[successor] = (state, action)
            if successor.blocks_remaining < best.blocks_remaining:
                best = successor
            frontier.append(successor)
    return best.blocks_remaining, True, _plan_to(parents, best), len(parents)


def _plan_to(parents, state):
    """Walk the parent links back from `state` to the opening position."""
    plan = []
    while parents[state] is not None:
        state, action = parents[state]
        plan.append(action)
    return plan[::-1]


def block_rows(grid):
    """Row indices that hold at least one block."""
    return [row for row, cells in enumerate(grid) if any(c in BLOCK_TYPES for c in cells)]


def playable_rows(grid):
    """Rows the player may stand on: everything inside the border."""
    return list(range(1, len(grid) - 1))


def count_blocks(grid):
    return sum(1 for row in grid for cell in row if cell in BLOCK_TYPES)


def throw(grid, row, held):
    """Apply a throw from `row` holding `held`. Returns `(grid, held)` or `None` for a no-op.

    The whole rule set, in one place, so that it can be read and argued with.
    """
    if row is None or not 0 <= row < len(grid) or held not in BLOCK_TYPES:
        return None

    grid = [cells[:] for cells in grid]
    destroyed, killed, new_held = [], 0, held

    for col in range(len(grid[row]) - 1, -1, -1):
        cell = grid[row][col]
        if cell not in BLOCK_TYPES:
            continue                       # empty or wall: the block flies past
        if cell == held:
            destroyed.append(col)
            killed += 1
            continue
        if killed == 0:
            return None                    # a different type first: the throw is refused
        grid[row][col] = held              # swap ours in and take theirs
        new_held = cell
        break

    if killed == 0:
        return None

    for col in destroyed:
        collapse(grid, row, col)

    return grid, new_held


def collapse(grid, row, col):
    """Drop the blocks stacked above `(row, col)` by one, in place.

    Only *blocks* fall. The run stops at the first cell above that is not one (empty air or
    the border), so a block with a gap under it stays where it is and the walls stay where
    they are. Getting this wrong is quiet and ugly: an earlier version shifted whatever was
    above, which walked the top border down into the play area one throw at a time, and the
    board still looked plausible while it happened.
    """
    top = row
    while top - 1 >= 0 and grid[top - 1][col] in BLOCK_TYPES:
        grid[top][col] = grid[top - 1][col]
        top -= 1
    grid[top][col] = EMPTY



#: The contract of each cartridge stage: what `generate_instance` draws a stage's size and
#: target from when the caller leaves them unset.
PROFILES = tuple(profile(stage) for stage in STAGES)

class FlipullAction:
    """`up`, `down`, or `throw`."""

    def __init__(self, name):
        if name not in ("up", "down", "throw"):
            raise ValueError(f"unknown action: {name!r}")
        self.name = name

    def cost(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, FlipullAction) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class FlipullState:
    """A board, a row the player stands on, and a block in hand."""

    def __init__(self, grid, row, held, clear_target, depth=0):
        self.grid = tuple(tuple(cells) for cells in grid)
        self.row = row
        self.held = held
        self.clear_target = clear_target
        self.depth = depth
        self.blocks_remaining = count_blocks(self.grid)

        literals = [f"at(block-{cell}, {r}, {c})"
                    for r, cells in enumerate(self.grid)
                    for c, cell in enumerate(cells) if cell in BLOCK_TYPES]
        literals.append(f"at(player, {row})")
        literals.append(f"holding(block-{held})")
        literals.append(f"remaining({self.blocks_remaining})")
        self.literals = frozenset(literals)

    def any_throw_connects(self):
        """Is there a row this player could stand on and throw from?

        What makes `is_terminal` exact here. The cartridge cannot answer this.
        """
        return any(throw([list(cells) for cells in self.grid], row, self.held) is not None
                   for row in playable_rows(self.grid))

    def __eq__(self, other):
        return (isinstance(other, FlipullState) and self.grid == other.grid
                and self.row == other.row and self.held == other.held)

    def __hash__(self):
        return hash((self.grid, self.row, self.held))

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        lines = []
        for index, cells in enumerate(self.grid):
            marker = "<" if index == self.row else " "
            lines.append("".join(cells) + marker)
        lines.append(f"held: {self.held}   blocks: {self.blocks_remaining}"
                     f"/{self.clear_target}")
        return "\n".join(lines)

    def __repr__(self):
        return (f"<FlipullState(row={self.row}, held={self.held}, "
                f"blocks={self.blocks_remaining}/{self.clear_target})>")


class FlipullGame(Environment):
    """Flipull, implemented rather than emulated. Needs nothing installed."""

    def __init__(self):
        super().__init__("flipull")
        self.index = 0
        #: The `(stage, clear_target)` pair `reset` builds: a bundled one after `set_index`,
        #: or whatever `set_instance` was given.
        self.instance = STAGES[0]
        #: The plan that reached the fewest blocks when `generate_instance` explored the
        #: current stage, and how many positions that exploration saw; None for a bundled or
        #: hand-made one.
        self.witness = None
        self.witness_expansions = None
        self.state = None
        self.state_history = []

    def set_index(self, index):
        if not 0 <= index < len(INSTANCES):
            raise IndexError(
                f"Invalid index: {index}. There are {len(INSTANCES)} stages, so the index "
                f"must be 0-{len(INSTANCES) - 1}.")
        self.index = index
        self.instance = INSTANCES[index]
        self.witness = self.witness_expansions = None

    def set_instance(self, instance):
        """Select a `(stage_text, clear_target)` pair, in the shape of the entries of `STAGES`."""
        text, target = instance
        if not block_rows(parse_stage(text)):
            raise ValueError("a stage needs at least one block")
        self.instance = (str(text), int(target))
        self.index = None
        self.witness = self.witness_expansions = None

    def generate_instance(self, seed=None, width=None, height=None, types=4,
                          clear_target=None, max_target_fraction=None, search_limit=200_000,
                          attempts=100):
        """Draw a fresh stage, select it, and return it as a `[stage_text, clear_target]` pair.

        Each draw is a random `width` by `height` wall of `types` block types, explored
        exhaustively (up to `search_limit` positions). With nothing set, a draw takes the
        size and the CLEAR target of one of the cartridge's 32 stages at random and is kept
        exactly when the fewest blocks it can be reduced to is that target, which is how the
        bundled stages were made. With a `width` and `height` of the caller's own and no
        target, the fewest blocks reachable becomes the target, provided that is no more
        than `max_target_fraction` (0.4) of the wall; with a `clear_target` given, a draw is
        kept when it can be reduced that far.
        """
        random_, _ = rng(seed)
        found = {}

        def draw(attempt):
            if width is None and height is None:
                options = from_profile(random_, PROFILES, clear_target=clear_target)
                found["exact"] = clear_target is None
            else:
                options = {"width": width or 5, "height": height or 5,
                           "clear_target": clear_target}
                found["exact"] = False
            found["options"] = options
            return generate_stage(random_, options["width"], options["height"], types)

        def accept(text):
            fewest, exhausted, plan, explored = fewest_blocks_reachable(text, search_limit)
            if not exhausted:
                return False              # undecided within the budget: not this one
            options = found["options"]
            target = options["clear_target"]
            if target is None:
                blocks = options["width"] * options["height"]
                if fewest > (0.4 if max_target_fraction is None else max_target_fraction) * blocks:
                    return False
                target = fewest
            elif fewest != target if found["exact"] else fewest > target:
                return False
            found["target"] = int(target)
            found["plan"], found["explored"] = plan, explored
            return True

        text = draw_until(draw, accept, attempts, "Flipull stage")
        instance = [text, found["target"]]
        self.set_instance(instance)
        self.witness, self.witness_expansions = found["plan"], found["explored"]
        return instance

    def reset(self):
        text, target = self.instance
        grid = parse_stage(text)
        rows = block_rows(grid)
        row = rows[-1] if rows else len(grid) - 2
        # The opening hand is the type of the block the player would meet first from the
        # bottom row, so the first throw always connects and the stage opens with a move.
        held = next((grid[row][col] for col in range(len(grid[row]) - 1, -1, -1)
                     if grid[row][col] in BLOCK_TYPES), BLOCK_TYPES[0])
        self.state = FlipullState(grid, row, held, target)
        self.state_history = [self.state]
        return self.state, {"stage": self.index,
                            "generated": self.index is None,
                            "blocks": self.state.blocks_remaining,
                            "clear_target": target,
                            "rows": len(playable_rows(grid))}

    def is_goal(self, state):
        return state.blocks_remaining <= state.clear_target

    def is_terminal(self, state):
        """No throw from any row would connect, so the board can never change again.

        Exact, because the rules are known here. The Game Boy sibling cannot compute this
        (it does not know what a throw hits) and can only report that the clock ran out.
        """
        return not self.is_goal(state) and not state.any_throw_connects()

    def successors(self, state):
        successors = []
        if self.is_goal(state) or self.is_terminal(state):
            return successors
        for action in ("up", "down", "throw"):
            successor = self.__advance__(state, FlipullAction(action))
            if successor == state:
                continue
            successors.append((FlipullAction(action), successor))
        return successors

    def __advance__(self, state, action):
        if self.is_goal(state) or self.is_terminal(state):
            return state
        name = action.name if isinstance(action, FlipullAction) else str(action)

        if name in ("up", "down"):
            row = state.row + (-1 if name == "up" else 1)
            if row not in playable_rows(state.grid):
                return state                     # into the ceiling or the floor
            return FlipullState(state.grid, row, state.held, state.clear_target,
                                state.depth + 1)

        outcome = throw([list(cells) for cells in state.grid], state.row, state.held)
        if outcome is None:
            return state                         # the throw was refused
        grid, held = outcome
        return FlipullState(grid, state.row, held, state.clear_target, state.depth + 1)

    def simulate(self, plan):
        state, _ = self.reset()
        trace = [state]
        for action in plan:
            trace.append(self.__advance__(trace[-1], action))
        return trace

    def step(self, action):
        if self.state is None:
            raise ValueError("Game not initialized. Call reset() first.")
        before = self.state.blocks_remaining
        self.state = self.__advance__(self.state, action)
        self.state_history.append(self.state)
        return self.state, before - self.state.blocks_remaining

    def get_actions(self):
        return [FlipullAction(name) for name in ("up", "down", "throw")]

    def render(self):
        rendered = [str(state) for state in self.state_history]
        for step, text in enumerate(rendered):
            print(f"Step: {step}")
            print(text)
            print("--------------")
        return rendered
