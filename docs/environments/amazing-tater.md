# Amazing Tater

This environment implements Amazing Tater's rules in Python rather than emulating the cartridge.
It needs no ROM, no emulator, and no dependencies beyond the standard library: the environment is
one module, and the 105 rooms it ships are in it.

A tater (i.e., a potato with legs) walks the four directions one cell at a time and has to reach
the exit flag. Some rooms hold more than one tater, and `switch` hands the controls to the next
one, so the room is not finished until every tater has reached the flag. Nothing here is
reversible: a block shoved into the wrong pit is gone, and so is the room.

- **Class:** `AmazingTaterGame`
- **Import:** `from planiverse.environments.gameboy_py.amazing_tater import AmazingTaterGame`
- **Source:** [`planiverse/environments/gameboy_py/amazing_tater.py`](../../planiverse/environments/gameboy_py/amazing_tater.py)
- **Instances:** 105 rooms, indices `0`–`104`
- **Dependencies:** none
- **Sibling:** [`AmazingTaterGBEnv`](amazing-tater-gb.md) plays the real Game Boy cartridge in an emulator, about four orders of magnitude slower per expansion

## The rules

1. **Walls** and the area outside the room refuse a step.
2. A **pit** refuses a step. Pits are crossed by filling them, not by walking over them.
3. A step into a **block** shoves the whole block one cell, and it moves only if every square it
   would land on is clear. Blocks come in several shapes (1×1, 1×5, 4×3, L-shapes), and a shape
   moves as one piece. Two different blocks may sit flush against each other and still be two
   blocks.
4. A block square that comes to rest over a pit has **settled into it** and cannot be shoved. A
   push has to be aimed at a square of the block that is standing on floor; the rest of the same
   block can still be pushed from a square that is.
5. When every square of a block sits over a pit, the block **dissolves**: the pits it covered
   become floor, permanently, and the block is gone. That is the only way to cross a pit.
6. A step into a **turnstile arm** turns the whole turnstile 90 degrees, in whichever direction
   carries that arm the way you pushed. Pushing an arm along its own axis does nothing, and
   neither does pushing one that is hanging over a pit. The turn needs room: every square an arm
   lands on must be clear, and so must the diagonal each arm sweeps through on its way there. Arms
   may swing over pits; taters may not walk on them.
7. Where the pusher ends up depends on whether they are shut into a compartment. If another arm
   swings into the square you pushed, the turnstile carries you round with it and your position
   rotates about the pivot. If nothing swings in behind it, you step into the square you pushed.
8. The **pivot** at the centre of a turnstile is solid and never moves.

We do not model anything outside the room itself: the cartridge's move counter, its timer, the
pause menu behind A with its RETRY and QUIT, and the level counter that loads the next room over
the top of a cleared one are all absent. Here a solved room is simply terminal.

## Quickstart

```python
from planiverse.environments.gameboy_py.amazing_tater import AmazingTaterGame

game = AmazingTaterGame()
game.fix_index(0)                  # choose the room before reset
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

`solve(index)` runs a breadth-first search for a shortest plan over the full position, meaning
every block, every turnstile angle and every tater:

```python
from planiverse.environments.gameboy_py.amazing_tater import solve
plan = solve(0)
print(len(plan), plan[:6])
# 38 ['left', 'left', 'left', 'left', 'up', 'up']
```

Its default budget is four hundred thousand states, which most of the later rooms exhaust.

## Rooms

The environment ships 105 rooms: 41 behind the cartridge's PUZZLE MODE (`A-01` to `A-41`) and 64
behind BEGINNER and ACTION MODE (`C-01` to `C-64`). `fix_index(n)` here and on
[`amazing_tater_gb`](amazing-tater-gb.md) select the same room. We left out the 96 rooms behind
PRACTICE MODE. That mode is a timed climb through ten floors, its board buffer holds the corridors
of the neighbouring floors as well as the room, and the tater starts outside the room, which makes
it a different game rather than a different level.

The rooms range from a 15×5 with three turnstiles and nothing else to an 18×16 with four taters, a
dozen blocks and forty pits. Note that difficulty is not uniform in index order.

We dumped all 105 off `Amazing Tater (U).gb` with `AmazingTaterGBEnv.levels`, which boots the
cartridge to each room and reads the board the game composes in work RAM, so nothing was
transcribed by hand.

### The alphabet

The alphabet uses one character per cell, and one character per cell code the cartridge uses, so a
stored room and a board dumped out of the emulator are the same string:

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

Blocks are letters rather than a single glyph because the cartridge records, for every block
square, which of its neighbours belong to the same block, and two different blocks sit flush
against each other in half of these rooms. A single glyph for all of them would weld two blocks
into one piece the cartridge would never move as one. Arms carry a direction so that an arm names
its own pivot, which is needed because thirty-six arms across these rooms are orthogonally
adjacent to two pivots, and adjacency alone cannot say which one they belong to.

## State

`AmazingTaterState` carries everything a move can change: where the taters are, which of them have
got home, which one has the controls, where the blocks are, how each turnstile is turned, and
which pits a dissolved block has filled. The level carries the walls, the flag and the set of
squares that were ever pits.

Equality is on the position and who holds the controls; depth and history are not part of it, so a
position reached two ways compares equal and search closes.

`state.literals` is a frozenset of strings: `at(tater1, 8, 14)`, `at(block, 3, 7)`, `turnstile(7,
7, 6)`, `pit(9, 11)`, `taters-home(1)`, `goal-reached`.

## Actions

| Action | Cost |
|---|---|
| `up`, `right`, `down`, `left` | 1 |
| `switch` | 0 |

`switch` is free because it moves nobody. Charging for it would measure the cheapest plan for a
two-tater room partly in how often the controls were swapped, which is not a property of the
puzzle. `successors` drops any action the game refuses, so a returned action always changes the
position, and in a one-tater room `switch` is never offered.

## Goal and terminal

- **Goal** (`is_goal`): every tater home.
- **Terminal** (`is_terminal`): a position with no move left in it and no tater home.

`is_terminal` is sound and no more than that. Amazing Tater has dead ends this does not catch, for
instance a block settled into the one pit that had to be crossed somewhere else, and recognising
them needs reachability under moving turnstiles. We accept the missed dead ends because a wrong
`is_terminal` prunes a solvable branch, which is the worse failure of the two.

## Rendering

`str(state)` prints the friendly board, with `$` for every block square, `&` for a settled one,
`+` and `*` for arms, and `o` for a pivot. `board(level, state)` prints the exact board in the
alphabet the levels are stored in, which is what the tests compare.

`render_trace` typesets `str(state)` into a contact sheet, PDF, GIF or directory of PNGs:

```python
from planiverse.planners.width import IteratedBFWS
from planiverse.benchmark import measures

result = IteratedBFWS(max_width=1000, progress=measures.amazing_tater).solve(game)
trace = game.simulate(result.plan)

game.render_trace(trace, "amazing_tater.gif")                                 # animated
game.render_trace(trace, "amazing_tater.png", actions=result.plan, env=game)  # contact sheet
```

See [docs/rendering.md](../rendering.md) for the other output formats.

## Files

| Path | What |
|---|---|
| [`amazing_tater.py`](../../planiverse/environments/gameboy_py/amazing_tater.py) | `AmazingTaterGame`, `AmazingTaterState`, `solve`, `board` |
| [`tests/test_amazing_tater.py`](../../tests/test_amazing_tater.py) | Tests |
