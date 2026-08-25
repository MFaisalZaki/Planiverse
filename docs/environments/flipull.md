# Flipull

A pure-Python puzzle environment in the spirit of Taito's *Flipull* (arcade *Plotting*), with a
rule set that is written down rather than reverse-engineered. No emulator, no ROM, no
dependencies beyond the standard library.

- **Class:** `FlipullGame`
- **Import:** `from planiverse.environments.flipull import FlipullGame`
- **Source:** [`planiverse/environments/flipull.py`](../../planiverse/environments/flipull.py)
- **Instances:** 10 stages, indices `0`–`9`
- **Dependencies:** none
- **Sibling:** [`FlipullGBEnv`](flipull-gb.md) plays the real Game Boy cartridge in an emulator

## The game

A wall of coloured blocks stands to the player's left. The player rides up and down the rows
holding one block, and throws it leftward along whichever row they are standing on.

1. The thrown block **destroys** every block of its own type it meets, and keeps going.
2. The first block of a **different** type takes the thrown block's place, and comes back into
   the player's hand — a swap.
3. If the *very first* block it meets is a different type, **nothing happens at all**. The
   throw is refused and the position is unchanged.
4. Every destroyed cell **collapses its column**: the run of blocks stacked directly above it
   falls one row. The run stops at the first gap, so a block with air under it stays put.

A stage is cleared when few enough blocks are left.

Rule 3 is the one that makes this a puzzle rather than a shuffling exercise. Only the
*rightmost* block of a row is reachable, so a throw is legal only where that block matches what
the player is holding — and what the player is holding is decided by the previous throw. Choosing
which row to stand on is the whole game, and choosing wrong strands you: a board where no row's
rightmost block matches your hand can never change again.

## Quickstart

```python
from planiverse.environments.flipull import FlipullGame

env = FlipullGame()
env.fix_index(0)
state, info = env.reset()

print(state)
print(info)     # {'stage': 0, 'blocks': ..., 'clear_target': ..., 'rows': ...}

for action, successor in env.successors(state):
    print(action, successor.blocks_remaining)
```

Stateful play goes through `step`, which returns how many blocks the action cleared:

```python
state, cleared = env.step('throw')
env.render()          # prints the state history
```

## Actions

Three, and they are the same three the cartridge has:

| Action | Effect |
|---|---|
| `up` | Move up one row. Refused at the top row. |
| `down` | Move down one row. Refused at the bottom row. |
| `throw` | Throw the held block along the current row. Refused unless rule 1 or 2 applies. |

`successors` never offers an action that would leave the position unchanged, so a refused throw
simply does not appear. That means the branching factor is genuinely 1–3 and a planner is never
handed a self-loop to close.

## Stages

`fix_index(i)` selects stage `i`. Indices are `0`–`9` and stable — `STAGES` is a literal tuple of
`(ascii, clear_target)` pairs in the module.

Stage strings use this alphabet:

| Char | Meaning |
|---|---|
| `#` | Wall |
| (space) | Empty cell |
| `1`–`4` | A block; the digit is its type |

The stages were **generated and then measured**, not drawn by hand. Hand-drawn boards kept
turning out to have unreachable targets — symmetric patterns in particular create parity traps
where the last few blocks can never be matched — so each shipped stage was produced randomly,
explored exhaustively, and kept only if its target is provably reachable. The tests re-derive a
solution for every stage, so a stage whose goal drifts out of reach fails the suite.

## State representation

`FlipullState` holds the grid (a tuple of tuples of single characters), the row the player is on,
the block in hand, the clear target, and a depth counter. Equality and hashing are over
`(grid, row, held)` only — depth is bookkeeping, and two identical boards reached by different
routes must be the same state or search never closes anything.

`literals` is a `frozenset` of strings:

```
at(block-2, 3, 4)      a block of type 2 at row 3, column 4
at(player, 3)          the player is standing on row 3
holding(block-1)       the block in hand
remaining(7)           blocks left on the board
```

## Goals and dead ends

`is_goal` is `blocks_remaining <= clear_target`.

`is_terminal` is **exact**, and this is the one place where the Python twin is strictly better
than the cartridge one. Because the rules are known here, a position is a dead end exactly when
no throw from any row would connect — `state.any_throw_connects()` decides it outright.
[`FlipullGBEnv`](flipull-gb.md) cannot compute that: it does not know what a throw hits, so all
it can report is that the clock ran out. Dead-end detection is most of what makes a puzzle
searchable, so this is not a small difference — a planner on the Python environment prunes a
doomed branch the moment it enters one, and on the cartridge it does not.

## How faithful is this to the cartridge?

Partly, and the honest answer is worth more than the claim. The rules above were derived by
driving a real `Flipull (USA)` cartridge and predicting what it would do next. They reproduce it
**exactly** — field and hand, cell for cell — for throws taken level with the wall in the
positions checked. Over a longer automated comparison they agreed on about half of the level
throws and four in five of the throws from above the wall, so something further is going on that
has not been pinned down: the staircase, a bounce, or a fall the model does not have.

So this is a Flipull-*like* environment with a stated rule set, not a clone — the same posture as
the synthetic test cartridge in [`tests/fake_flipull_rom.py`](../../tests/fake_flipull_rom.py).
What it is good for is a well-defined, dependency-free planning problem. What it is *not* good
for is predicting the cartridge, which is what [`FlipullGBEnv`](flipull-gb.md) is there to do.

Two deliberate simplifications follow from that:

- **No staircase.** Some cartridge stages have a fixed diagonal structure at the left. Since it
  is not established what a thrown block does when it meets one, this twin leaves it out rather
  than model a guess.
- **No clock.** The cartridge fails you on time. Here the only failure is a genuine dead end, so
  a plan's length is bounded by the search budget rather than by a timer.

## Planning with it

The state space is small — hundreds to a few thousand reachable states per stage — but
dead-end-rich, which is the interesting combination: blind search wanders into positions it can
never leave, and the useful signal is how many blocks are gone.

```python
from planiverse.environments.flipull import FlipullGame
from planiverse.planners.width import BFWSSearch, Budget

env = FlipullGame()
env.fix_index(3)
result = BFWSSearch(width=2, progress=lambda s: s.blocks_remaining).solve(
    env, Budget(max_expansions=200000, max_seconds=60))
print(result.status, len(result.plan))
```

`blocks_remaining` is the natural progress measure and the reason BFWS does well here. Plain
`IWSearch(width=1)` generally does not: novelty over single atoms cannot tell a board that is one
throw from stuck apart from one that is not.
