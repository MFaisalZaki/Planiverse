# Puzznic

A from-scratch Python re-implementation of the 1989 Taito block-matching puzzle game, used as a
planning benchmark. 50 hand-written levels, no emulator, no dependencies beyond the standard library.

- **Class:** `PuzznicGame`
- **Import:** `from planiverse.environments.gameboy_py.puzznic import PuzznicGame`
- **Source:** [`planiverse/environments/gameboy_py/puzznic.py`](../../planiverse/environments/gameboy_py/puzznic.py)
- **Instances:** 50 levels, indices `0`–`49`
- **Dependencies:** none
- **Sibling:** [`PuzznicGBEnv`](puzznic-gb.md) plays the real Game Boy cartridge in an emulator

This is the reference implementation of the Planiverse interface: it is the only environment that
implements every method in the contract (`step`, `validate`, `get_actions`, and `render` included).
If you want to read one environment to understand the library, read this one.

## The game

The grid holds coloured blocks. You steer a cursor, grab a block, and drag it sideways. Blocks fall
under gravity, and when two or more of the same colour touch, they vanish. Clear every block to win.
The catch is that you can strand yourself: matching is pairwise, so leaving a colour with exactly one
block on the board makes the level unwinnable — a genuine dead end, which is what makes this
interesting for planning rather than reflexes.

## Quickstart

```python
from planiverse.environments.gameboy_py.puzznic import PuzznicGame

env = PuzznicGame()
env.fix_index(0)
state, info = env.reset()

print(state)
# ######
# #12c #
# ###  #
# #    #
# #2  1#
# ##21##
# ######

env.get_actions()
# ['left', 'right', 'up', 'down', 'left-hold', 'right-hold']

for action, successor in env.successors(state):
    print(action, sum(successor.score))
```

Stateful play, as opposed to expansion, goes through `step`:

```python
state, score = env.step('left')
env.render()          # prints the de-duplicated state history
```

## Levels

`fix_index(i)` selects level `i` from `PuzznicGame.levelsstr`, a list of 50 ASCII level strings
embedded in the module. Indices are `0`–`49` and stable — they're a literal list, not a directory
listing.

Level strings use this alphabet:

| Char | Meaning |
|---|---|
| `#` | Wall |
| (space) | Empty cell |
| `1`–`9` | A block; the digit is its colour/letter |
| `0` | Empty cell (treated as space) |
| `c` | Cursor start position (the cell itself is empty) |

Levels are rectangular and walled on all four sides. The outer ring is unreachable: bounds checking
is `0 <= pos < shape - 1` on both axes, so the cursor stays strictly inside.

## State representation

`PuzznicState` holds the grid (a 2-D list of `Element` objects — `Wall`, `EmptySpace`, `Box`,
`Cursor`), the cursor, a `score` list, and the `cleared_boxes` accumulated so far.

Literals are exact and fine-grained — there is no abstraction here, unlike the epidemic or urban
environments:

| Literal | When |
|---|---|
| `at(cursor, x, y)` | always |
| `at(box-L, x, y)` | per block `L` on the grid |
| `cleared(box-L, x, y)` | per block cleared so far, at the position it was cleared from |
| `all-boxes-matched(box-L)` | colour `L` has been fully cleared off the grid |
| `goal-reached` | in a goal state |
| `terminal-state` | in a dead-end state |
| `score(N)` | in a goal or terminal state |

State equality compares grid and cursor directly (`__eq__`), *not* literals — so score and cleared
history do not distinguish two otherwise-identical positions during expansion. Planners that key
their visited set on `literals` (as `TreeSearchPlanner` does) will see a finer distinction than
`__eq__` gives, since literals carry `cleared(...)` history.

## Actions

Six actions, all cursor moves:

| Action | Effect |
|---|---|
| `left` / `right` / `up` / `down` | Move the cursor one cell |
| `left-hold` / `right-hold` | Move the cursor one cell, dragging the block under it |

A `-hold` with no block under the cursor is rejected outright — the cursor does not move either. When
the cursor *is* on a block, the block slides only if the destination is `EmptySpace`. There is no
`up-hold`/`down-hold`: you cannot lift a block, only slide it.

## Transition

`_compute_successor_state_` runs a pipeline per action:

1. **Move** — `apply_action` moves the cursor and, for a hold, the block.
2. **Settle and clear, until stable** — gravity drops every block that has empty space below it, then
   `_check_and_remove_matches_` clears every block orthogonally adjacent to a same-colour block. A
   clear lets the blocks above fall, which can form a new match, so the two repeat until the grid
   stops changing.
3. **Score** — each clear appends its points to `state.score`.

Goal and terminal states are absorbing: `_compute_successor_state_` returns a copy of the state
unchanged rather than expanding them. `successors` additionally drops any successor equal to its
parent, so a cursor move into a wall is not offered as an action.

Note that a level holding a single block of some colour is *born* terminal, and therefore ignores
every action — worth knowing when hand-writing a level to test against.

## Goal and terminal

- **Goal** (`is_goal`) — no `Box` remains on the grid.
- **Terminal** (`is_terminal`) — some colour has exactly **one** block left. It can never be matched,
  so the level is lost. This is one of only three environments in the repo that computes a real
  terminal condition — [`PuzznicGBEnv`](puzznic-gb.md) applies the same rule to the cartridge.

## Scoring

`state.score` is a *list* with one entry per clear (not per action); use `sum(state.score)` for the
total. Per the module's own comment the scoring rules are an assumed reconstruction, not the original
arcade formula:

- 10 points per cleared block;
- if a single clear removes more than one distinct colour, the whole award is multiplied by
  `1.5 × (number of distinct colours)`;
- +50 bonus per colour that had more than two blocks in the clear.

`_compute_score_` takes the set of blocks a match actually cleared. It used to diff the grid before
and after the step instead — and since blocks are identified by colour *and* position, a block that
merely fell appeared on both sides of the diff and scored as though it had been cleared twice.

## Known quirks

- **`_check_and_remove_matches_` asserts `len(to_remove) != 1`.** Reaching a state where exactly one
  block is marked for removal raises `AssertionError` rather than returning a state. Matches are
  found symmetrically, so this shouldn't trigger — but it is an assert on a hot path, not a guard.
- **`apply_action` returns `False` on rejection and `None` on success**, and callers ignore the
  return value. Rejected actions are detected by the successor comparing equal to the parent.
- **`PuzznicState.__init__` has a mutable default argument** (`cleared_boxes=[]`). It is copied
  (`cleared_boxes[:]`) before use, so it is not corrupted — but it is a trap for future edits.
- **`render()` de-duplicates consecutive identical states** before printing, so the printed step
  numbers do not line up with plan indices.
- **The scoring formula is a guess.** The multiplier keys off the number of distinct colours in a
  clear, while the comment describes a multiplier for *consecutive* cascading matches. Cascades now
  score as separate entries in `state.score`, so implementing the described rule is straightforward
  if the original arcade behaviour matters.

## Files

| Path | What |
|---|---|
| [`puzznic.py`](../../planiverse/environments/gameboy_py/puzznic.py) | Everything: elements, state, level parser, game |

Key classes: `Element`/`Box`/`Cursor`/`Wall`/`EmptySpace` (grid cells), `PuzznicState` (state),
`Level` (parses a level string), `PuzznicGame` (the environment).
