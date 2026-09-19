# Puzznic

Taito's 1989 block-matching puzzle, re-implemented in Python. A cursor drags coloured blocks
sideways across a walled grid; blocks fall, touching blocks of one colour vanish, and the level is
won when the grid is empty. It is the smallest environment in the library, which is why the
README's examples use it, and it needs no emulator, no ROM and no dependency beyond the standard
library.

- **Class:** `PuzznicGame`
- **Import:** `from planiverse.environments.games.puzznic import PuzznicGame`
- **Source:** [`planiverse/environments/games/puzznic.py`](../../planiverse/environments/games/puzznic.py)
- **Instances:** 100 levels, indices `0` to `99`: the first 100 of the original game's 128 rounds
- **Generator:** `generate_instance(seed, width=None, height=None, colours=None, ...)`; see [Generating levels](#generating-levels)
- **Dependencies:** none

## Context

Puzznic is a single-screen puzzle. The screen is a grid of walls and empty cells with a few
coloured blocks resting on the walls or on each other, and the player steers a cursor over it.
The cursor can grab the block under it and drag it one cell to the left or right, provided the
cell is empty. Whatever the player does, gravity then acts: every block with empty space below it
falls, and when two or more blocks of the same colour come to touch they vanish, which lets the
blocks above them fall in turn and may form another match. The round is won when every block has
gone. Matching is pairwise, so a colour left with exactly one block on the board can never be
cleared, and the round is lost from that moment however many moves remain (i.e., it is a dead end
that no later move can undo). The skill of the game is in the order of the clears, since a block
that is needed as a landing for another must not be matched away too early.

We re-implemented the game rather than run the cartridge, so plans are cheap to expand (a
successor costs microseconds) and the rules are ours to read. The cost is fidelity. We know of two
differences from the cartridge. First, the cursor starts on the nearest empty cell rather than on
a block, because a level string marks the cursor's cell as empty and cannot also put a block
there, so a plan found here does not transfer to the cartridge move for move. Second, the
re-implementation rescans the whole board after every step and clears every adjacent same-colour
pair, whereas the cartridge leaves some such pairs untouched. The scoring is likewise
reconstructed rather than copied from the arcade formula (see [Planning problem](#planning-problem)).

## Actions

The action set holds six cursor moves, each costing 1:

| Action | Cost | Effect |
|---|---|---|
| `left`, `right`, `up`, `down` | 1 | move the cursor one cell |
| `left-hold`, `right-hold` | 1 | move the cursor one cell, dragging the block under it |

A `-hold` with no block under the cursor is refused, and the cursor does not move either; when
the cursor is on a block, the block slides only if the destination cell is empty. There is no
`up-hold` or `down-hold`, because blocks slide sideways and cannot be lifted. A move into a wall
changes nothing, and `successors` drops any successor equal to its parent, so such a move is not
offered.

Applying an action runs a pipeline of three stages. First, the cursor moves and, for a hold, so
does its block. Second, the board settles and clears until it is stable: gravity drops every
block with empty space below it, then every block orthogonally adjacent to a same-colour block
is cleared. A clear lets the blocks above fall and may form a new match, so the two steps repeat
until the grid stops changing. Third, each clear appends its points to the state's score.
The whole cascade is one action from the planner's side, which is the first half of how a game
played frame by frame becomes a planning problem; the second half is the goal test below.

## Planning problem

A `PuzznicState` holds the grid (a 2-D list of `Wall`, `EmptySpace`, `Box` and `Cursor`
elements), the cursor, a `score` list with one entry per clear, and the `cleared_boxes` so far.
Two states are the same state when their grids and cursors are the same; the score and the
cleared history do not distinguish two otherwise identical positions during expansion. Note that a
planner keying its visited set on `literals` sees a finer distinction than `__eq__` gives, because
the literals carry the cleared history. They are:

| Literal | When |
|---|---|
| `at(cursor, x, y)` | always |
| `at(box-L, x, y)` | per block of colour `L` on the grid |
| `cleared(box-L, x, y)` | per block cleared so far, at the position it was cleared from |
| `all-boxes-matched(box-L)` | colour `L` has been fully cleared off the grid |
| `goal-reached`, `terminal-state` | in a goal state and in a dead-end state |
| `score(N)` | in a goal or terminal state |

With `blocks(s, L)` the number of blocks of colour `L` on the grid of `s`, the goal and the dead
end are

```
goal(s)     ≡ ∀L. blocks(s, L) = 0
terminal(s) ≡ ∃L. blocks(s, L) = 1
```

Both are absorbing (i.e., no action leads out of them): the successor function returns the state
unchanged rather than expanding it, so a level holding a single block of some colour is born
terminal and ignores every action, which is worth knowing when hand-writing a level.

The game as played is a real-time loop in which the board keeps settling while the player thinks,
and a round is won or lost by conditions the game checks every frame. We turn it into an
initial-to-goal problem in two moves. First, the loop between two decisions (the fall, the clears
and the cascade) is folded into the action, as described above, so that a state is always a
settled board and the planner never sees a block in mid-air. Second, the game's win and loss
conditions become the goal test and the dead-end test on that settled board. The win is a
condition on the grid alone. The loss, which the cartridge only reports when the player gives
up, is decided at once from the block counts, which prunes every position from which no clear is
possible. What the reduction leaves out is the score. It is kept on the state and reported at the
goal, but it is not the objective, since every action costs 1 and a plan is judged by its length.
The scoring is our reconstruction: 10 points per cleared block, the award multiplied by 1.5 times
the number of distinct colours when a single clear removes more than one, and 50 bonus points per
colour with more than two blocks in the clear.

The progress measure the width planners take (`planiverse.benchmark.measures.puzznic`) is the
number of blocks left on the grid.

## An example

Instance `0` is the cartridge's first round, a six-by-seven grid with two blocks each of colours
1 and 2 and the cursor starting top right:

```
######
#12c #
###  #
#    #
#2  1#
##21##
######
```

Iterated BFWS, run as the benchmark runs it (i.e., with the measure above and a width bound of
1000), solved it in 122 expansions and a tenth of a second. Its twelve-move plan is

```
left, right-hold, down, down, down, left-hold, left, up, up, up, right-hold, right-hold
```

which drags the top-row block of colour 2 off its ledge so that it falls onto the 2 on the floor
and clears. The cursor then walks back up for the block of colour 1 and drags it twice to the
right, where it falls onto the 1 beside the far wall and clears the board.

![BFWS solving puzznic instance 0](../renders/puzznic.gif)

The render is the state's own text, typeset, one frame per state. `render_trace` falls back to
`str(state)` for an environment with no screen to photograph, and for a grid this small the text
is the clearest picture of it. The same plan is also a [contact sheet](../renders/puzznic.png),
every frame on one image, captioned with the step number, the action that produced it, and a
note on the goal state. Both were generated by solving the instance and handing the trace to
`render_trace`:

```python
from planiverse.environments.games.puzznic import PuzznicGame
from planiverse.benchmark import measures
from planiverse.planners.width import IteratedBFWS

env = PuzznicGame()
env.set_index(0)
state, info = env.reset()
print(state)                       # the grid above
env.get_actions()                  # ['left', 'right', 'up', 'down', 'left-hold', 'right-hold']

result = IteratedBFWS(max_width=1000, progress=measures.puzznic).solve(env)
trace = env.simulate(result.plan)

env.render_trace(trace, "puzznic.gif")                                   # animated
env.render_trace(trace, "puzznic.png", actions=result.plan, env=env)     # contact sheet
```

Stateful play, as opposed to expansion, goes through `step`, which returns the state and the
points the move scored; `env.render()` prints the de-duplicated history of `step` calls and
returns it as a list of strings. Note that consecutive identical states are collapsed there, so
the printed step numbers do not line up with plan indices. See
[docs/rendering.md](../rendering.md) for the other output formats.

## Levels

`set_index(i)` selects level `i` from `PuzznicGame.levelsstr`, a list of 100 ASCII level strings
embedded in the module. Indices run from `0` to `99` and match the cartridge's first 100 rounds.
Level strings use this alphabet:

| Char | Meaning |
|---|---|
| `#` | Wall |
| (space) | Empty cell |
| `1` to `9` | A block; the digit is its colour |
| `0` | Empty cell, treated as a space |
| `c` | Cursor start position; the cell itself is empty |

Levels are rectangular and walled on all four sides. The outer ring is unreachable: bounds
checking is `0 <= pos < shape - 1` on both axes, so the cursor stays strictly inside.

## Generating levels

`generate_instance` draws a fresh level, selects it the way `set_index` selects a bundled one,
and returns it as a level string in the alphabet above, so it can be saved and handed back to
`set_instance` later:

```python
env = PuzznicGame()
level = env.generate_instance(seed=7)     # or make("puzznic", seed=7)
state, info = env.reset()                 # info["generated"] is True
env.witness                               # the plan the draw was accepted on
```

| Option | Default | What it does |
|---|---|---|
| `width`, `height` | a bundled one's | interior cells inside the ring of walls |
| `colours` | a bundled one's | block colours, `1` upward |
| `walls` | a bundled one's | share of the interior filled with wall cells |
| `blocks_per_colour` | a bundled one's | how many blocks a colour gets, one choice per colour |
| `solvable` | `True` | search each draw and keep only one with a plan |
| `min_plan_length` | 4 | reject a draw whose shortest plan is shorter |
| `search_limit` | 20000 | expansions the check may spend per draw |
| `attempts` | 200 | draws before giving up with `GenerationError` |

Blocks are dropped in at rest, so the board starts settled; no two blocks of a colour touch, so
nothing clears itself; and no colour has a single block, so no draw is born terminal. The check is
a breadth-first search over positions, so `min_plan_length` is a lower bound on the shortest plan
and a fair difficulty knob. A draw the search cannot decide within `search_limit` is rejected,
which biases the generator towards levels a small search can solve. Larger boards mostly reject,
so give them a larger limit and some patience, or pass `solvable=False` for an unchecked draw. The
same seed and options always give the same level.

Left unset, the layout options come from the profile of a cartridge round drawn at random
(`PROFILES`, one per round): its interior size, its number of colours, how many blocks each
colour has and how much of it is wall. A generated level is therefore the shape of a real one
rather than a shape of the generator's own. Only rounds of at most `max_blocks` blocks (10) are
drawn from, because a breadth-first check cannot decide the shapes of the larger rounds within
its budget; `max_blocks=None` draws from all 100, and any option given explicitly wins over the
profile. The method is generate-and-test, which the procedural content generation literature
calls search-based PCG (Togelius, Yannakakis, Stanley and Browne, 2011,
https://doi.org/10.1109/TCIAIG.2011.2148116; Shaker, Togelius and Nelson, *Procedural Content
Generation in Games*, 2016, https://pcgbook.com/); the test here is breadth-first search over
positions.

## Files

| Path | What |
|---|---|
| [`puzznic.py`](../../planiverse/environments/games/puzznic.py) | `Element`/`Box`/`Cursor`/`Wall`/`EmptySpace`, `PuzznicState`, `Level`, `PuzznicGame` |
| [`tests/test_puzznic.py`](../../tests/test_puzznic.py) | Tests |
