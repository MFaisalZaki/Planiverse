# Adventures of Lolo (pure Python)

Adventures of Lolo's puzzle, implemented rather than emulated, over the cartridge's own 163 rooms.
No ROM, no emulator, no dependencies.

The sibling [`lolo_gb`](lolo-gb.md) drives the real Game Boy cartridge. Use this one for a
dependency-free benchmark; use that one when you need the cartridge's actual behaviour, which for
137 of the 163 rooms you do, and the reason is [below](#where-this-differs-from-the-cartridge).

- **Class:** `LoloGame`
- **Import:** `from planiverse.environments.gameboy_py.lolo import LoloGame, LoloAction`
- **Source:** [`planiverse/environments/gameboy_py/lolo.py`](../../planiverse/environments/gameboy_py/lolo.py)
- **Dependencies:** none

## Quickstart

```python
from planiverse.environments.gameboy_py.lolo import LoloGame

game = LoloGame()
game.fix_index(38)                    # int 1-1
state, info = game.reset()

print(info)
# {'room_index': 38, 'room': 'int 1-1', 'hearts': 6, 'shots': 0, 'door': (1, 1),
#  'start': (6, 1), 'exact': True, 'unmodelled_enemies': ()}

print(state)
# ##.#####
# #D..H.H#
# #......#
# #......#
# #...H.H#
# #......#
# #@..H.H#
# ########

for action, successor in game.successors(state):
    print(action, successor.lolo, successor.hearts_left)
```

Or by name:

```python
from planiverse.environments import make
game = make("lolo", index=38)
```

## The rules, stated

Every rule here was measured on the cartridge (the probes are in the
[memory map](lolo-gb-memory-map.md) §5) rather than taken from a manual.

Lolo walks the four directions, one cell at a time, on an 8 × 8 board.

1. A step into a **rock** `#`, a **tree** `T`, a **river** `~` or the edge is refused.
2. A step into a **one-way pass** `v` `<` `^` `>` is refused if it goes against the arrow, and
   allowed from the other three sides.
3. A step into an **Emerald Framer** `O` or an **egg** `e` pushes it one cell the same way, if the
   cell behind is empty walkable ground. Nothing can be pulled and no chain of two can be pushed.
4. A step onto a **heart framer** collects it. `h` (the cartridge stores two heart codes and this
   is the second) also gives Lolo **two magic shots**; plain `H` gives none.
5. A step into an **enemy** is refused. Enemies never move on their own.
6. **`shoot`** fires one cell in the direction Lolo last *tried* to move, whether or not that move
   succeeded. An enemy there becomes an egg, which can then be pushed; an egg there is blasted out
   of the room. Each shot costs one.
7. A **Medusa** `M` kills Lolo when he stands anywhere in its row or column, unless a tree, an
   Emerald Framer, a heart framer, an enemy or an egg stands between them. Rocks, rivers, bridges,
   deserts, flower beds, one-way passes and break tiles do **not** block a Medusa.
8. The **door** `D` opens once every heart framer is collected. Standing on the open door clears
   the room.

Rule 7's exception list is the one that surprises people: a rock does not stop a Medusa, a tree
does. That was measured one candidate at a time and it is not a bug in either implementation.

## Where this differs from the cartridge

In one place that matters, and it is worth stating plainly.

**Six of the eight enemies are frozen here.** Snakey and Medusa never move on the cartridge either,
so rooms holding only those two are modelled exactly. The other six (Leeper, Rocky, Alma, Gol,
Skull and Don Medusa) do move there, in lock-step with Lolo, and this module leaves them where
they started. For a room that contains one, a plan found here is a plan against a *strictly easier*
puzzle and may well die on the cartridge. That is the wrong direction for an approximation to err
in, so it is flagged rather than smoothed over:

```python
from planiverse.environments.gameboy_py.lolo import EXACT_ROOMS
len(EXACT_ROOMS)      # 26

state, info = game.reset()
info["exact"]                  # True when the model is faithful for this room
info["unmodelled_enemies"]     # ('K', 'R') — the kinds that would have moved
```

**Rafts are refused.** On the cartridge an egg shoved into a river floats, and Lolo can step onto
it and ride across; that is how `int 1-3` is cleared. Five of the six river codes accept one, and
two of those then *carry* the raft, one cell every few frames. Modelling a moving raft means
modelling time, which nothing else here needs, so this module refuses the push. Rooms needing a
raft cannot be cleared here; `lolo_gb` clears them.

**The hammer is not modelled.** A few rooms start with one in the cartridge's PWR meter (`int 1-5`
does, and cannot be cleared without it), and what it breaks was not established.

Two smaller divergences, both in the safe direction. Medusa's shot is instant here where the
cartridge gives one move of grace, but that move cannot be used to escape, so no plan is lost. And
an Emerald Framer is not pushed onto a heart framer, a door or a marker, which was never tested and
is refused rather than guessed.

## Evidence

Breadth-first search over this module found a plan for 32 of the 163 rooms. Every one of those
plans was then replayed on the real cartridge:

| | plans found | cleared the room on the cartridge |
|---|---|---|
| rooms in `EXACT_ROOMS` | 7 | **7** |
| rooms whose enemies this module freezes | 25 | 3 |

That is the claim above, measured: where the model is faithful it is faithful, and where it is an
approximation the approximation is the easy direction: the twenty-two failures are Lolo walking
into an enemy that was not standing still. `tests/test_solutions.py` pins both halves, and replays
the ten confirmed plans on the cartridge when a ROM is available.

The raft rule came out of the same exercise: the one plan that failed for a reason other than a
moving enemy pushed an egg into a river and rode it, which the cartridge does allow, but with a
current this module does not model. Rafts are refused now rather than modelled wrongly.

## Rooms

163, at indices 0–162, at the same indices `lolo_gb` uses: `fix_index(38)` is the same room in
both. They were decoded out of the ROM by `lolo_gb.read_rooms`, not transcribed, and
`tests/test_lolo.py` re-decodes and compares when a ROM is available.

| Indices | `label` | What |
|---|---|---|
| 0–37 | `tutorial 1a` … `tutorial 19b` | 19 puzzles, each stored twice |
| 38–107 | `int 1-1` … `int 5-14` | 5 intermediate floors of 14 |
| 108–157 | `adv 1-1` … `adv 10-5` | 10 advanced floors of 5 |
| 158–162 | `pro 1` … `pro 5` | the Pro rooms |

Print any of them:

```console
$ python -m planiverse.environments.gameboy_py.lolo --room 0
---   0 tutorial 1a  2 hearts
  |##D#####|
  |#......#|
  |#....TTT|
  |#....SH#|
  |#....###|
  |#......#|
  |#.h..@.#|
  |########|
```

## Magic shots

The meter starts empty, like the cartridge on a cold boot, because it belongs to the player rather
than to the room. A few rooms need a shot they cannot earn in-room:

```python
game = LoloGame(magic_shots=2)
```

`lolo_gb.LoloGBEnv` takes the same argument and means the same thing by it.

## Actions

`left`, `up`, `down`, `right`, `shoot`. Cost 1 each.

A refused *move* is still a successor when it leaves Lolo facing somewhere new, because the facing
is what decides where the next shot goes. Walking into a rock to turn is a real move in this game.

## Goal and terminal

**Goal:** every heart framer collected and Lolo standing on the door.

**Terminal:** Lolo in a Medusa's clear line. Absorbing; there is nothing left to plan for.
