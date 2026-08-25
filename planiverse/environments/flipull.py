"""Flipull in pure Python: no ROM, no emulator, no dependencies.

The sibling [`flipull_gb`](flipull_gb.py) drives the real cartridge. This one implements the
rules directly, the way [`puzznic`](puzznic.py) stands beside `puzznic_gb`. Use this one for a
dependency-free benchmark; use that one for the cartridge's actual behaviour.

## The rules, stated

The player stands to the right of a wall of blocks holding one of them, and moves up and down
the rows or throws. A throw sends the held block leftward along the player's row:

1. It **destroys** each block of its own type it meets, and keeps going.
2. The first block of a **different** type takes the thrown block's place, and comes back into
   the player's hand — a swap.
3. If the *very first* block it meets is a different type, **nothing happens at all**: the
   block flies out and comes back, and the position is unchanged.
4. Every destroyed cell **collapses its column**: everything above it falls one row.

A stage is cleared when few enough blocks are left.

Rule 3 is the one that makes this a puzzle rather than a shuffling exercise. Without it every
throw would be legal and the board would be a permutation group; with it, most rows are
refused most of the time and choosing which row to stand on is the whole game.

## How faithful is this to the cartridge?

Partly, and the honest answer is worth more than a claim. The rules above were derived by
driving `Flipull (USA)` and predicting what it would do, and they reproduce it **exactly** —
field and hand, cell for cell — for throws taken level with the wall in the positions
checked. Over a longer automated comparison they agreed on about half of the level throws and
four in five of the throws from above the wall, so something further is going on that has not
been pinned down: the staircase, or a bounce, or a fall the model does not have.

So this is a Flipull-*like* environment with a stated rule set, not a clone — the same
posture as the synthetic test cartridge in `tests/fake_flipull_rom.py`. What it is good for is
a well-defined, dependency-free planning problem; what it is not good for is predicting the
cartridge, which is what `flipull_gb` is there to do.

## What the Python twin can do that the cartridge cannot

Because the rules are known here, `is_terminal` is **exact**: a position is a dead end when no
throw from any row would connect. The Game Boy environment cannot compute that — it does not
know what a throw hits — and can only tell you the clock ran out. Dead-end detection is most
of what makes a puzzle searchable, so this is not a small difference.

## Where the stages came from

They were generated and measured, not drawn. Hand-drawn boards kept turning out to have
unreachable targets — symmetric patterns in particular create parity traps where the last few
blocks can never be matched — so each stage here was produced randomly, explored exhaustively,
and kept only once its target was known to be reachable. `tests/test_flipull.py` re-derives a
solution for every one of them, so a stage whose goal drifts out of reach fails the suite
rather than quietly wasting a planner's budget.
"""
from planiverse.environments.base import Environment

#: `1`-`4` are block types, `#` is wall, and a space is empty. There is no staircase: the
#: cartridge has a fixed diagonal one at the left of some stages, and since it is not clear
#: what a thrown block does when it meets it, this twin leaves it out rather than guess.
WALL, EMPTY = "#", " "
BLOCK_TYPES = ("1", "2", "3", "4")

#: `(stage, clear_target)`, in rising order of difficulty: the shortest optimal solution runs
#: 4 moves and the longest 52, over reachable state spaces from a dozen states to ~150,000.
#: Every target is the fewest blocks the board can actually be reduced to, so a stage is only
#: cleared by playing it out rather than by chipping away at it.
#:
#: The player starts on the bottom row of the wall, as on the cartridge — the position where
#: `down` does nothing, and the reason the Game Boy environment's sprite probe had to stop
#: demanding movement in both directions.
STAGES = (
    ("#####\n#   #\n#111#\n#211#\n#####", 1),
    ("#####\n#   #\n#222#\n#222#\n#211#\n#####", 1),
    ("######\n#    #\n#2122#\n#2121#\n#1121#\n######", 1),
    ("#######\n#     #\n#21222#\n#22211#\n#21211#\n#######", 1),
    ("######\n#    #\n#1221#\n#1221#\n#1 12#\n#2  1#\n#1 21#\n######", 1),
    ("########\n#      #\n#1112 2#\n#221122#\n#112112#\n#11121 #\n#21  11#\n########", 1),
    ("#######\n#     #\n#21222#\n#22121#\n#21111#\n#22221#\n#22112#\n#22111#\n#######", 1),
    ("#########\n#       #\n#2222111#\n#1111112#\n#2121221#\n#1111212#\n#2112112#\n#########", 1),
    ("#########\n#       #\n#2122122#\n#1222112#\n#1222221#\n#2112221#\n#2111122#\n#########", 1),
    ("########\n#      #\n#3 3233#\n#133321#\n#22 121#\n#232 12#\n#3 1312#\n#121333#\n########", 2),
)


def parse_stage(text):
    """An ASCII stage into a grid of single characters, padded to a rectangle."""
    rows = text.split("\n")
    width = max(len(row) for row in rows)
    return [list(row.ljust(width)) for row in rows]


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

    Only *blocks* fall. The run stops at the first cell above that is not one — empty air or
    the border — so a block with a gap under it stays where it is and the walls stay where
    they are. Getting this wrong is quiet and ugly: an earlier version shifted whatever was
    above, which walked the top border down into the play area one throw at a time, and the
    board still looked plausible while it happened.
    """
    top = row
    while top - 1 >= 0 and grid[top - 1][col] in BLOCK_TYPES:
        grid[top][col] = grid[top - 1][col]
        top -= 1
    grid[top][col] = EMPTY


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

    def can_throw(self):
        """Would a throw from here connect? Known exactly, because the rules are known."""
        return throw([list(cells) for cells in self.grid], self.row, self.held) is not None

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
        self.state = None
        self.state_history = []

    def fix_index(self, index):
        if not 0 <= index < len(STAGES):
            raise IndexError(
                f"Invalid index: {index}. There are {len(STAGES)} stages, so the index must "
                f"be 0-{len(STAGES) - 1}.")
        self.index = index

    def reset(self):
        text, target = STAGES[self.index]
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
                            "blocks": self.state.blocks_remaining,
                            "clear_target": target,
                            "rows": len(playable_rows(grid))}

    def is_goal(self, state):
        return state.blocks_remaining <= state.clear_target

    def is_terminal(self, state):
        """No throw from any row would connect, so the board can never change again.

        Exact, because the rules are known here. The Game Boy sibling cannot compute this —
        it does not know what a throw hits — and can only report that the clock ran out.
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
