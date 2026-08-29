# Amazing Tater (pure Python)

Amazing Tater's rules, implemented rather than emulated. No ROM, no emulator, no dependencies
beyond the standard library: the whole environment is one module and the 105 rooms it ships
are in it.

The sibling [`amazing_tater_gb.py`](amazing-tater-gb.md) plays the real cartridge inside PyBoy.
Use that one when you want the cartridge's own transition function; use this one for a
dependency-free benchmark that expands states about four orders of magnitude faster.

- **Class:** `AmazingTaterGame`
- **Import:** `from planiverse.environments.gameboy_py.amazing_tater import AmazingTaterGame`
- **Source:** [`planiverse/environments/gameboy_py/amazing_tater.py`](../../planiverse/environments/gameboy_py/amazing_tater.py)
- **Dependencies:** none

## Quickstart

```python
from planiverse.environments.gameboy_py.amazing_tater import AmazingTaterGame

game = AmazingTaterGame()
game.fix_index(0)                  # choose the room *before* reset
state, info = game.reset()

print(state)
#       #####
#  ### #.....# ###
# #...##o+.+.##...#
# #.E...+++o....1.#
# #...##.o+..##...#
#  ### #.....# ###
#       #####

print(info)
# {'level_index': 0, 'level': 'A-01', 'size': (15, 5), 'taters': 1,
#  'blocks': 0, 'turnstiles': 3, 'pits': 0}

for action, successor in game.successors(state):
    print(action, action.cost())
```

`1` is the tater, `E` is the exit flag it has to reach, `o` is a turnstile pivot and `+` its
arms. `$` would be a block and `O` a pit; this room has neither.

## The game

A tater (a potato with legs) walks the four directions one cell at a time and has to reach
the exit flag. Some rooms hold more than one tater; `switch` (SELECT on the console) hands the
controls to the next one, and the room is not finished until every one of them has reached the
flag.

1. **Walls** and the area outside the room refuse a step.
2. A **pit** refuses a step. Pits are crossed by filling them, not by walking over them.
3. A step into a **block** shoves the whole block one cell, and it moves only if every square
   it would land on is clear. Blocks come in every shape the cartridge felt like drawing
   (1×1, 1×5, 4×3, L-shapes), and a shape moves as one piece. Two different blocks may sit
   flush against each other and still be two blocks.
4. A block square that comes to rest over a pit has **settled into it**. You cannot shove that
   square: a push has to be aimed at a square of the block that is standing on floor. The rest
   of the same block can still be pushed from a square that is.
5. When *every* square of a block sits over a pit, the block **dissolves**: the pits it covered
   become floor, permanently, and the block is gone. That is the only way to cross a pit, and
   it cannot be undone.
6. A step into a **turnstile arm** turns the whole turnstile 90 degrees, in whichever direction
   carries that arm the way you pushed. Pushing an arm along its own axis does nothing, and
   neither does pushing one that is hanging over a pit. The turn needs room: every square an
   arm lands on must be clear, and so must the diagonal each arm sweeps through on its way
   there. Arms may swing over pits; taters may not walk on them.
7. Where the pusher ends up depends on whether they are shut into a compartment. If another arm
   swings into the square you pushed, you were standing between two arms and the turnstile
   carries you round with it: your position rotates about the pivot, exactly like a revolving
   door. If nothing swings in behind it, the square you pushed is now empty and you simply step
   into it.
8. The **pivot** at the centre of a turnstile is solid and never moves.

Nothing is reversible. There is no undo here, which is the point: a block shoved into the wrong
pit is gone, and so is the room.

## Actions and costs

| Action | Cost |
|---|---|
| `up`, `right`, `down`, `left` | 1 |
| `switch` | 0 |

`switch` is free because it moves nobody. Charging for it would measure the cheapest plan for a
two-tater room partly in how often you swapped, which is not a property of the puzzle.

`successors` drops any action the game refuses, so a returned action always changes the
position. In a one-tater room `switch` is never offered.

## States

`AmazingTaterState` carries everything a move can change: where the taters are, which of them
have got home, which one has the controls, where the blocks are, how each turnstile is turned,
and which pits a dissolved block has filled. The level carries the walls, the flag and the set
of squares that were ever pits.

Equality is on the position and who holds the controls; depth and history are deliberately not
part of it, so a position reached two ways compares equal and search closes.

`state.literals` is a frozenset of strings: `at(tater1, 8, 14)`, `at(block, 3, 7)`,
`turnstile(7, 7, 6)`, `pit(9, 11)`, `taters-home(1)`, `goal-reached`.

`str(state)` prints the friendly board above. `board(level, state)` prints the exact one, in
the alphabet the levels are stored in, and that is the one the tests compare; see below.

## Goals and dead ends

`is_goal` is every tater home. `is_terminal` is a position with no move left in it and no tater
home: sound, and no more than that. Amazing Tater has dead ends this does not catch (a block
settled into the one pit that had to be crossed somewhere else is lost, and so is the room),
but recognising them needs reachability under moving turnstiles, and a wrong `is_terminal`
prunes a solvable branch, which is the worse failure of the two.

## The rooms

105 of them: 41 behind the cartridge's PUZZLE MODE (`A-01` to `A-41`) and 64 behind BEGINNER
and ACTION MODE (`C-01` to `C-64`). `fix_index(n)` here and on `amazing_tater_gb` select the
same room.

The cartridge has a third set (the 96 rooms behind PRACTICE MODE), and it is deliberately
absent. That mode is a timed climb through ten floors: its board buffer holds the corridors of
the neighbouring floors as well as the room, and the tater starts outside the room. It is a
different game, not a different level.

They range from a 15×5 room with three turnstiles and nothing else to an 18×16 with four taters,
a dozen blocks and forty pits. Difficulty is not uniform in index order; breadth-first search
solves `A-01` in a few thousand states and does not finish `A-15` in four hundred thousand.

### Where they came from

All 105 were dumped off `Amazing Tater (U).gb` by
`amazing_tater_gb.AmazingTaterGBEnv.levels`, which boots the cartridge to each room and reads
the board the game itself composes in work RAM. Nothing was transcribed by hand, and
`tests/test_amazing_tater.py` re-dumps the cartridge and compares when a ROM is available, so a
room cannot drift away from the cartridge unnoticed.

### The alphabet

One character per cell, and one character per *cell code the cartridge uses*, so a stored room
and a board dumped out of the emulator are the same string:

| Glyph | Meaning |
|---|---|
| `' '` | outside the room |
| `#` | wall |
| `.` | floor |
| `O` | an open pit |
| `E` | the exit flag |
| `1`–`4` | the taters |
| `@` | a turnstile pivot |
| `^ > v <` | a turnstile arm, pointing the way it sticks out from its pivot |
| `U R D L` | the same four arms, hanging over a pit |
| `a`–`p` | a block square on floor, one letter per set of neighbours it is joined to |
| `ABCFGHIJKMNPQSTV` | the same sixteen, for a square settled into a pit |

Blocks are letters rather than a single `$` because the cartridge records, for every block
square, which of its neighbours belong to the same block, and two *different* blocks are flush
against each other in half of these rooms. A `$` for all of them would quietly weld them into
one piece that the cartridge would never move as one, which is a bug this had before it was a
paragraph.

Arms carry a direction so that an arm names its own pivot. Thirty-six arms across these
rooms are orthogonally adjacent to two pivots, and adjacency alone cannot say which one
they belong to.

`str(state)` collapses all of that for reading: `$` for every block square, `&` for a settled
one, `+` and `*` for arms, `o` for a pivot.

## Where this differs from the cartridge

Nowhere that has been found, and the search was not casual. The rules above were established by
walking this module and the cartridge forward in lockstep (the same random press, then a
cell-by-cell comparison of the whole board) for three runs of two hundred presses in every one
of the 105 rooms. Sixty-three thousand transitions, no disagreements.

Four of the eight rules are worded the way they are *because* that comparison rejected a
simpler guess:

| Rule | What the simpler guess was | What the cartridge did |
|---|---|---|
| 3, "still be two blocks" | Group block squares by adjacency | Moved half of a touching pair, in `A-04` |
| 4, a settled square cannot be shoved | Shove a block from any of its squares | Refused, in `A-36` |
| 6, the swept diagonal | Check only the squares the arms land on | Refused a turn whose corner was another turnstile's arm, in `A-01` |
| 6, an arm over a pit | Shove any arm | Refused, in `C-43` |
| 7, the compartment | Always step into the square you pushed | Carried the pusher a quarter-turn round the pivot, in `A-05` |

What is missing is everything outside the room: the cartridge's move counter, its timer, the
pause menu behind A with its RETRY and QUIT, and the level counter that loads the next room over
the top of a cleared one. Here a solved room is simply terminal.

## Solving

`solve(index)` is a breadth-first search for a shortest plan, and is how the stored solutions in
the tests were found:

```python
from planiverse.environments.gameboy_py.amazing_tater import solve
plan = solve(0)
print(len(plan), plan[:6])
# 38 ['left', 'left', 'left', 'left', 'up', 'up']
```

It searches the full position (every block, every turnstile angle, every tater), so it exhausts
its default four-hundred-thousand-state budget on most of the later rooms. That is the point of
shipping them.
