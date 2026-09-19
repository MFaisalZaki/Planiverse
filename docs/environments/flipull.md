# Flipull

A puzzle in the spirit of Taito's *Flipull* (the arcade *Plotting*), re-implemented in Python
with a rule set we state below rather than one reverse-engineered from the cartridge. A wall of
coloured blocks stands to the player's left. The player rides up and down the rows holding one
block, and throws it leftward along whichever row they are standing on. A stage is cleared once
few enough blocks are left, which the cartridge calls the CLEAR target, and the environment
needs no emulator, no ROM and no dependency beyond the standard library.

- **Class:** `FlipullGame`
- **Import:** `from planiverse.environments.games.flipull import FlipullGame`
- **Source:** [`planiverse/environments/games/flipull.py`](../../planiverse/environments/games/flipull.py)
- **Instances:** 100 stages, indices `0` to `99`: 32 to the cartridge's stage table, then 68 the generator drew
- **Generator:** `generate_instance(seed, width=None, height=None, types=4, clear_target=None, ...)`; see [Generating stages](#generating-stages)
- **Dependencies:** none

## Context

Flipull is a single-screen puzzle. The screen holds a wall of blocks of four types, stacked in
rows inside a ring of walls with one empty row above it. The player stands at the right-hand edge
holding a block of one of the types. Each turn the player climbs or descends a row, or
throws the held block leftward along the row they stand on, where it destroys, swaps or bounces
according to the rules below. The stage is cleared once the blocks left are down to the CLEAR
target. What the player decides is which row to throw from, since the block a row reaches and
the block in hand together fix whether a throw is legal at all.

The rules are five. First, the thrown block destroys every block of its own type it meets, and
keeps going. Second, the first block of a different type takes the thrown block's place and
comes back into the player's hand (i.e., a swap). Third, if the first block it meets at all is
of a different type, nothing happens: the throw is refused and the position is unchanged.
Fourth, a wall turns the block downward: it slides down the face of the wall it met, and the
first three rules apply to what it meets on the way down. The floor bounces it back into the
hand, and whatever it destroyed on the way stays destroyed. So a throw from above the wall of
blocks reaches the top of the far column, and a throw that empties its row carries on down the
far side. Fifth, every destroyed cell collapses its column: the run of blocks stacked directly
above it falls one row, and the run stops at the first gap, so a block with air under it stays
put.

Rule 3 is what makes this a puzzle rather than a shuffling exercise. From a row only its
rightmost block is reachable, and from above the wall only the top of the far column. A throw
is therefore legal only where that block matches what the player is holding. What the player
is holding was decided by the previous throw, and a board where neither matches the hand
anywhere can never change again.

We re-implemented the game rather than run the cartridge, so a successor costs microseconds
and, because the rules are known, a dead end can be recognised exactly (see
[Planning problem](#planning-problem)). The cost is fidelity, so we say where each rule stands
with respect to the cartridge. Rules 1 to 3 and 5 we derived by driving a real Flipull cartridge
and predicting what it would do next. They reproduce field and hand exactly for throws taken
level with the wall in the positions we checked. Rule 4 is the original's, documented for the
arcade *Plotting* the cartridge ports ([the arcade
FAQ](https://gamefaqs.gamespot.com/arcade/584111-plotting/faqs/41573),
[Wikipedia](https://en.wikipedia.org/wiki/Plotting_(video_game))). There, a block that reaches
the back wall slides straight down it, as it does off the ceiling, and one that reaches the
floor bounces back to the player. It is what an earlier automated comparison against the
cartridge was missing. The throws it disagreed on were those whose row was empty or emptied,
which this module used to refuse or stop and the cartridge carries down the far side.

Two simplifications remain. First, the cartridge's stages carry a staircase of fixed bricks at
the left and, later, pipes. A `#` inside the board deflects a block here the way the back wall
does, which is what the staircase does, but the bundled stages carry none and pipes are not
modelled. Second, there is no clock, so the only failure is a genuine dead end and a plan's
length is bounded by the search budget rather than by a timer.

## Actions

The action set holds three actions, the same three the cartridge has, each costing 1:

| Action | Cost | Effect |
|---|---|---|
| `up` | 1 | move up one row |
| `down` | 1 | move down one row |
| `throw` | 1 | throw the held block along the current row |

`up` is refused at the top row and `down` at the bottom row. A throw is refused unless the first
block it meets, along the row or down the far wall, is of the held type (i.e., unless rule 1 or
rule 2 applies). A throw that meets no block at all is refused too. `successors` never offers
an action that would leave the position unchanged, so a refused throw does not appear at all and
the branching factor is between 1 and 3. A planner is never handed a self-loop to close, and a
goal state or a dead end has no successors at all.

Applying a throw runs the whole flight and its aftermath as one action. The block flies leftward
along the row and is turned down the face of the first wall it meets. It stops at the first
block of another type, which it swaps with, or at the floor, which bounces it back. Every cell it
destroyed then collapses its column, in the order the cells were met, so a run down one column
drops the stack above it by the length of the run. The board the planner sees is always the
settled one.

## Planning problem

A `FlipullState` holds the grid (a tuple of tuples of single characters in the stage alphabet),
the row the player is on, the block in hand, the clear target and a depth counter. Two states
are the same state when their grid, row and hand are the same. Depth is bookkeeping, because two
identical boards reached by different routes must be one state or search never closes anything.
The literals are a `frozenset` of strings:

```
at(block-2, 3, 4)      a block of type 2 at row 3, column 4
at(player, 3)          the player is standing on row 3
holding(block-1)       the block in hand
remaining(7)           blocks left on the board
```

With `T` the stage's CLEAR target and `remaining(s)` the blocks left on the grid, let
`connects(s, r)` say whether a throw from row `r` would first meet a block of the held type.
The goal and the dead end are

```
goal(s)     ≡ remaining(s) ≤ T
terminal(s) ≡ remaining(s) > T ∧ ¬∃r. connects(s, r)
```

`state.any_throw_connects()` decides the second outright, by trying the throw from every row
the player may stand on, so dead-end detection here is exact. Both are absorbing (i.e., no
action leads out of them): `successors` returns nothing for either, and `simulate` carries the
state through unchanged.

The game as played is a real-time one: the thrown block crosses the screen, the columns fall
while the player watches. A stage is won by a counter and lost only when the clock runs out.
We turn it into an initial-to-goal problem in two moves. First, the loop between two decisions
(the flight, the swap or bounce, and the collapse) is folded into the action, as described
above. A state is then always a settled wall, and the planner never sees a block in flight.
Second, the game's win and loss conditions become the goal test and the dead-end test on that
settled wall. The win is the count against the target. The loss, which the cartridge can only
report as the clock running out, is decided at once from whether any throw would connect. That
prunes a doomed branch the moment a planner enters one, which is most of what makes a puzzle
searchable. An emulator-backed environment could not make the second move, since it does not
know what a throw hits. What the reduction gives up is the timer: a plan is judged by its
length, since every action costs 1, and nothing here says whether the cartridge's clock would
have allowed it.

The progress measure the width planners take (`planiverse.benchmark.measures.flipull`) is the
number of blocks left above the clear target.

## An example

Instance `0` is the cartridge's first stage: a wall of 25 blocks, five rows of five in four
types, with a CLEAR target of 9. There are six rows the player may stand on, the empty one
above the wall included. The player starts on the bottom row of the wall, as on the cartridge.
The hand holds a 4, the type of the block a throw from there would meet first, so the stage
opens with a legal throw:

```
####### 
#     # 
#41321# 
#12423# 
#33431# 
#24231# 
#42334#<
####### 
held: 4   blocks: 25/9
```

Iterated BFWS, run as the benchmark runs it (i.e., with the measure above and a width bound of
1000), solved it in 163 expansions and under a tenth of a second. Its thirty-five-move plan is

```
throw, up, up, throw, up, up, up, throw, down, down, down, down, throw, up, up, up, throw,
down, down, down, throw, throw, down, throw, up, up, up, throw, down, down, throw, up, up,
throw, throw
```

which is twelve throws and twenty-three row changes. Each throw clears one or two blocks and,
through the swap, decides what the hand holds next, so the row changes between throws walk the
player to whichever row's reachable block matches the new hand. The first throw, from the bottom
row, destroys the 4 at the wall's edge and takes the 3 behind it into the hand. The last two,
thrown from one row, clear three blocks between them and leave 9, which is the target.

![BFWS solving flipull instance 0](../renders/flipull.gif)

The render is the state's own text, typeset, one frame per state: `render_trace` falls back to
`str(state)` for an environment with no screen to photograph. The text is the board in the stage
alphabet, with a marker on the player's row and the hand and the count beneath. The same
plan is also a [contact sheet](../renders/flipull.png), every frame on one image, captioned with
the step number, the action that produced it, and a note on the goal state. Both were generated
by solving the instance and handing the trace to `render_trace`:

```python
from planiverse.environments.games.flipull import FlipullGame
from planiverse.benchmark import measures
from planiverse.planners.width import IteratedBFWS

env = FlipullGame()
env.set_index(0)
state, info = env.reset()
print(state)                       # the board above
print(info)                        # {'stage': 0, 'generated': False, 'blocks': 25,
                                   #  'clear_target': 9, 'rows': 6}

result = IteratedBFWS(max_width=1000, progress=measures.flipull).solve(env)
trace = env.simulate(result.plan)

env.render_trace(trace, "flipull.gif")                                   # animated
env.render_trace(trace, "flipull.png", actions=result.plan, env=env)     # contact sheet
```

Stateful play, as opposed to expansion, goes through `step`, which returns the state and how
many blocks the action cleared; `env.render()` prints the state history and returns it as a list
of strings. `successors(state)` returns the action and state pairs for the moves that change the
position, and `successor.blocks_remaining` is the count each one leaves. See
[docs/rendering.md](../rendering.md) for the other output formats.

## Stages

`set_index(i)` selects stage `i`. Indices `0` to `31` match the cartridge's stage table, giving
the same board size (25, 30 or 36 blocks) and the same CLEAR target (9 down to 6), stage for
stage. `STAGES` is a literal tuple of `(ascii, clear_target)` pairs in the module, so the indices
are stable.

We generated the arrangements rather than copying them, because the cartridge stores none: it
draws each stage's layout from an RNG seeded by boot timing. Each board is `generate_instance`
at the seed beside it, kept when the cartridge's target can be reached under the rules above.
That is what the cartridge's own stages are: a random layout against a fixed target.

Stage strings use this alphabet:

| Char | Meaning |
|---|---|
| `#` | Wall |
| (space) | Empty cell |
| `1` to `4` | A block; the digit is its type |

Indices `32` to `99` are stages the generator drew (`GENERATED_STAGES`), each to the size and
CLEAR target of one of the cartridge's stages, with the seed beside it in the module. The plan
each of the hundred stages was accepted on is in `tests/data/flipull_solutions.json`.
`tests/test_solutions.py` replays every one of them, so a stage whose goal drifts out of reach
fails the suite rather than quietly wasting a planner's budget.

## Generating stages

`generate_instance` draws a stage the way the bundled ones were made, selects it, and returns
it as a `[stage_text, clear_target]` pair that `set_instance` accepts back:

```python
env = FlipullGame()
text, target = env.generate_instance(seed=7)     # or make("flipull", seed=7)
state, info = env.reset()                        # info["generated"] is True
env.witness                                      # the plan the draw was accepted on
```

| Option | Default | What it does |
|---|---|---|
| `width`, `height` | a bundled one's | the wall of blocks, in blocks |
| `types` | 4 | block types, `1` upward |
| `clear_target` | a bundled one's | a CLEAR target to demand; unset, the draw's own is used |
| `max_target_fraction` | 0.4 | with a size given and no target, reject a draw whose fewest reachable blocks exceed this share of the wall |
| `search_limit` | 200000 | positions a draw's search may visit |
| `attempts` | 100 | draws before giving up with `GenerationError` |

With a target, a bundled stage's or the caller's, a draw is searched breadth-first for a
position that meets it and kept when one is found within `search_limit` positions; the plan is
left in `env.witness`. With a size of the caller's own and no target, the draw is explored
exhaustively instead (`fewest_blocks_reachable`). The fewest blocks it can be worn down to
becomes its target, so the stage is clearable by construction. A draw whose state space outgrows
`search_limit` is rejected as undecided, which biases the generator towards stages a small
search can decide, and a caller who wants larger ones has to raise the limit.

Left unset, a draw takes the size and the CLEAR target of one of the cartridge's 32 stages at
random (`PROFILES`) and is kept when that target can be reached. That is how the bundled
stages were made, and what the cartridge's own are: a random layout against a fixed target. A
size of the caller's own goes with the fewest-blocks rule above instead. The method is
generate-and-test, which the procedural content generation literature calls search-based PCG
(Togelius, Yannakakis, Stanley and Browne, 2011, https://doi.org/10.1109/TCIAIG.2011.2148116;
Shaker, Togelius and Nelson, *Procedural Content Generation in Games*, 2016,
https://pcgbook.com/). The test is breadth-first search to the target, or an exhaustive
exploration of the positions for a size of the caller's own. The same seed and options always
give the same stage.

## Rendering

`str(state)` draws the board in the stage alphabet, with `<` marking the player's row and the
hand and the count on a last line, and `env.render()` prints the state history. See
[docs/rendering.md](../rendering.md) for the other output formats.

## Files

| Path | What |
|---|---|
| [`flipull.py`](../../planiverse/environments/games/flipull.py) | `FlipullGame`, `FlipullState`, `FlipullAction`, `STAGES`, `GENERATED_STAGES` |
| [`tests/test_flipull.py`](../../tests/test_flipull.py) | Tests |
| [`tests/data/flipull_solutions.json`](../../tests/data/flipull_solutions.json) | The plan each stage was accepted on |
