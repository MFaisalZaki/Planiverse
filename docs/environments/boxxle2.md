# Boxxle II

Sokoban, implemented rather than emulated: no ROM, no emulator, no dependencies. The sibling
[`boxxle2_gb`](boxxle2-gb.md) plays the same 120 levels on the real Game Boy cartridge.

- **Class:** `Boxxle2Game`
- **Import:** `from planiverse.environments.gameboy_py.boxxle2 import Boxxle2Game, Boxxle2Action`
- **Source:** [`planiverse/environments/gameboy_py/boxxle2.py`](../../planiverse/environments/gameboy_py/boxxle2.py)
- **Dependencies:** none

## The game

A warehouse keeper walks a grid one cell at a time:

1. A step into a **wall** is refused, and the position does not change.
2. A step into a **box** pushes it one cell in the same direction, but only if the cell
   *behind* the box is empty floor. A box cannot be pulled, and no chain of boxes can be pushed
   at once.
3. Any other step just moves the keeper.

The level is solved when every box stands on a goal. Every level ships with exactly as many
goals as boxes, so "every box home" and "every goal filled" are the same sentence.

There is no undo and no restart. The cartridge has both; leaving them out is what makes this a
search problem, because a wrong push is then permanent.

## Quickstart

```python
from planiverse.environments.gameboy_py.boxxle2 import Boxxle2Game

env = Boxxle2Game()
env.fix_index(0)
state, info = env.reset()

print(state)
#  #######
#  # @ooo#
#  #   ####
# ###$    #
# #   #$# #
# # $ #   #
# #   #####
# #####

print(info)
# {'level_index': 0, 'level': '1-01', 'size': (9, 8), 'boxes': 3, 'goals': 3}

for action, successor in env.successors(state):
    print(action, successor.boxes_home)
# left 0
# down 0
# right 0
```

`#` wall, `$` box, `o` goal, `*` box on a goal, `@` keeper, `+` keeper on a goal, space floor;
the same alphabet [`boxxle2_gb`](boxxle2-gb.md) prints, so a board from either module reads
the same.

## Actions

Four, one per direction, each costing 1:

```python
env.get_actions()          # [left, up, down, right]
```

`successors` returns only the ones that change something, so a step into a wall or against an
immovable box never appears.

## Levels

All 120, twelve stages of ten, at the same indices the cartridge uses: `fix_index(7)` here and
on `boxxle2_gb` are the same board.

```python
env.fix_index(37)
env.reset()[1]["level"]    # '4-08'
```

They were **decoded out of the ROM**, not transcribed: `boxxle2_gb.read_levels` implements the
cartridge's two-stage level decompressor, and the 120 boards it produced are what
`LEVELS` holds. The decoder is verified two independent ways (every record's computed size
matches its pointer-table delta, and every level has as many goals as boxes), and the boards
were then loaded on the cartridge and compared cell for cell, with no mismatches.

That matters because the failure mode of transcribed level data is silent. A wall in the wrong
cell still parses, still renders and still looks like a puzzle; it is just a different one, or
an unsolvable one. Two such slips were found in the hand-typed Puzznic levels. There are none
here to find, and `tests/test_boxxle2.py` re-decodes the ROM and compares when one is
available, so a level cannot drift away from the cartridge unnoticed.

## State representation

```python
state.boxes        # frozenset of (row, col)
state.player       # (row, col)
state.boxes_home   # how many boxes are on a goal
state.solved
state.level        # the static half: walls, goals, dead squares, shape
```

The walls, the goals and the precomputed dead squares live on the `Level`, not on the state:
they never change, and copying them into every node would make each state far bigger than the
position it describes. Two states are equal when their boxes and keeper match.

### Literals

```python
{'at(player, 1, 3)', 'boxes-home(0)',
 'at(box, 3, 3)', 'at(box, 4, 5)', 'at(box, 5, 2)',
 'goal(cell, 1, 4)', 'goal(cell, 1, 5)', 'goal(cell, 1, 6)'}
```

Identical to the Game Boy environment's, which is what lets a plan and a state be compared
across the two. `goal-reached` and `terminal-state` are added when they apply. Walls are not
literals: they are true in every state of a level, so as atoms they are pure noise for a
novelty measure.

## Goals and dead ends

```python
env.is_goal(state)        # every box home
env.is_terminal(state)    # a box on a square it can never be pushed off again
```

`dead_squares` marks every non-goal cell with a wall on one of its vertical sides *and* one of
its horizontal sides. A box there is stuck for good, so the level is lost. The test is
**sound** (it never calls a solvable position dead) and deliberately **not complete**: frozen
pairs of boxes, and rows that hug a wall with no goal along them, are dead too and are not
claimed. A wrong dead end prunes a solvable branch, which is much worse than letting a doomed
one run.

Unusually, the Game Boy sibling computes exactly the same test from exactly the same
information. This cartridge keeps its walls in work RAM in plain form, so for once the pure
Python twin has no analytical advantage over the emulator.

## How faithful is this to the cartridge?

Exactly, as far as anything has been able to show.

All 120 levels were driven with twenty-five random moves each, on the cartridge through
[`boxxle2_gb`](boxxle2-gb.md) and here, and the two boards were compared after every one of the
3,000 moves: **0 divergences**. That covers walls refusing a step, boxes refusing a push, boxes
blocked by boxes, and boards in both of the cartridge's cell-size modes.

The reason it comes out this clean, and the Flipull and Puzznic twins do not, is not that this
one was written more carefully. It is that Boxxle II has almost nothing to reproduce: the
cartridge decompresses each level into three plain byte planes and applies the three rules
above to them. No gravity, no animation that changes state, no timer, no score, no randomness.

The one thing the cartridge has that this does not is the level counter: clearing a level there
loads the next one over the top of it. Here a solved position is simply terminal.

## Where the solutions came from

`tests/data/boxxle2_solutions.json` maps a level index to a plan that solves it, and
`tests/test_solutions.py` replays every one of them. Two provenances, and they are worth
keeping apart:

**Most are human.** ASchultz's Boxxle II FAQ on GameFAQs writes each level's solution as a list
of box pushes (`0 2D 4U 3R, 2 1R 4U 2R, ...`, "box 0 down two, up four, right three") against
a map that labels the boxes. Those were parsed, aligned against the ROM's own board, and turned
back into keeper moves by walking the keeper into place before each push. The FAQ has typos in
its labels (the author flags one himself), so the labels are treated as a preference rather
than a fact and the push *shapes*, which carry the information, drive a small backtracking
search. **Nothing is taken on trust:** a reconstructed plan is kept only after it has been
replayed here and seen to solve the level, and the plans in
`tests/data/boxxle2_gb_solutions.json` were replayed on the cartridge too.

**The rest are searched.** Levels whose FAQ entry is prose rather than notation were handed to
a push-based A\* solver, and kept on the same terms.

Coverage is partial and pinned by a test, the same way Puzznic's is: some of the later levels
have 30 to 59 boxes and are not solved by anything here within a sane budget. A level *losing*
its solution has to fail loudly, because that means a level changed.

## Planning with it

```python
from planiverse.planners.width.bfws import BFWSSearch
from planiverse.planners.width.result import Budget

env = Boxxle2Game()
env.fix_index(0)
env.reset()

result = BFWSSearch(
    width=1,
    progress=lambda s: len(s.boxes) - s.boxes_home,
    heuristic=lambda s: sum(min(abs(b[0] - g[0]) + abs(b[1] - g[1]) for g in s.level.goals)
                            for b in s.boxes),
).solve(env, budget=Budget(max_expansions=200000, max_seconds=30))
```

Expect the first few levels to fall in well under a second and the rest not to fall at all.
Sokoban's difficulty is in the *order* of the pushes, not in the choice at any one step, so a
branching factor of three or four hides a state space that grows about as fast as anything in
this repository. That is what makes it a useful benchmark rather than a solved one.

## Files

| | |
|---|---|
| [`planiverse/environments/gameboy_py/boxxle2.py`](../../planiverse/environments/gameboy_py/boxxle2.py) | the environment and its 120 levels |
| [`planiverse/environments/gameboy/boxxle2_gb.py`](../../planiverse/environments/gameboy/boxxle2_gb.py) | the cartridge twin, and the level decoder |
| [`tests/test_boxxle2.py`](../../tests/test_boxxle2.py) | tests |
| [`tests/data/boxxle2_solutions.json`](../../tests/data/boxxle2_solutions.json) | a known plan per solved level |
