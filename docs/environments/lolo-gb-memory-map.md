# Adventures of Lolo (Game Boy) memory map

Reference for reading live game state out of `Adventures of Lolo (U) [S][!].gb`: the room table,
the cell codes, the work RAM layout and the rules. 262,144 bytes, HAL Laboratory and Nintendo,
internal title `LOLO2`, MD5 `8f6b6ef366a787852f664d945c86eb72`.

This is what [`lolo_gb.py`](../../planiverse/environments/gameboy/lolo_gb.py) reads and what
[`lolo.py`](../../planiverse/environments/gameboy_py/lolo.py) re-implements.

---

## 1. Cartridge

| Header field | Offset | Value | Meaning |
|---|---|---|---|
| Entry point | `$0100` | `00 C3 50 01` | `NOP` / `JP $0150` |
| Title | `$0134` | `LOLO2` | the European/US release of the Game Boy Lolo |
| CGB flag | `$0143` | `$00` | DMG only |
| SGB flag | `$0146` | `$03` | Super Game Boy features present |
| Cartridge type | `$0147` | `$01` | MBC1 |
| ROM size | `$0148` | `$03` | 256 KiB, 16 banks |
| RAM size | `$0149` | `$00` | none; no battery save, so all state is in work RAM |

`RST $30` (→ `$0586`) is the bank switcher: it masks `IE`, shadows the bank number in `$C5FC` and
writes `$2100`. Bank 0 is the controller, bank 13 is the room table, banks 5 and 10 are compressed
graphics, bank 15 is the sound driver entered from the timer interrupt.

---

## 2. The room table

`LoadRoom` at `$11E5`:

```
$11E5:  PUSH AF/BC/DE/HL
        LD A,$0D / RST $30          ; bank 13
$11EC:  LD A,($C3A6)                ; the room number  <-- the hookable instruction
        HL = A << 6 ; HL += $4000   ; room * 64
        DE = $C3BF ; C = $40
$1201:  copy 64 bytes
$1207:  LD A,$01 / RST $30          ; back to bank 1
        POP HL/DE/BC/AF / RET
```

So the table is flat and uncompressed: bank 13, `$4000 + N*64`, which is file offset `13*$4000 +
N*64`. Each room is 8 × 8 cells, one byte per cell, row major, stride 8, copied verbatim into
`$C3BF`, and 16 KiB divided by 64 gives 256 slots.

`$11EC` is where `lolo_gb` hooks. Writing `$C3A6` from outside on a frame boundary loses to the
route into a room, which sets the number and calls the loader inside a single frame.

### How many slots are rooms

163 of them are rooms. The first 163 slots use only codes from the vocabulary in §3, and each
holds exactly one Lolo (`$00`) and exactly one door (`$96`), whereas slot 163 onwards fails all
three tests. `lolo_gb.verify_room_table` runs that check, and it returns empty for `range(163)`.

### How they are grouped

`$26AA` divides the room number into a floor and a level within that floor:

```
$26AA:  LD A,($C3A6)
$26AD:  SUB $26                     ; room - 38
        LD B,$FF
loop:   INC B / SUB $0E / JR NC,loop
        ADD A,$0E
        LD ($C3A5),A                ; level = (room - 38) mod 14
        LD A,B / LD ($C3A4),A       ; floor = (room - 38) div 14
```

That fixes 38 as the end of the tutorial and 14 as the levels per floor. The rest follows from the
163 live slots and matches the published description of the European release (144 puzzles):

| Slots | Count | What |
|---|---|---|
| 0–37 | 38 | the tutorial, "First steps in Eden": 19 puzzles, each stored twice, a demonstration the game plays for itself then the same room to try. The two halves of a pair differ by a few cells |
| 38–107 | 70 | 5 intermediate floors × 14 rooms |
| 108–157 | 50 | 10 advanced floors × 5 rooms |
| 158–162 | 5 | the Pro rooms. `$179D` loads them: `LD A,$9E` (= 158) into `$C3A6`, then a loop of five. |

19 + 70 + 50 + 5 = 144 puzzles in 163 slots.

---

## 3. Cell codes

Established by patching a synthetic room over the board buffer at `$1207`, letting the game draw
it, and reading the result off the screen. The names are the cartridge's own: the object list at
`$2CA9` is plain ASCII and reads

> `SNAKEY LEEPER GOL ROCKY ALMA SKULL MEDUSA DON MEDUSA` …
> `EMERALD FRAMERS TREES ROCKS DESERTS ENEMY HOLES RIVERS BREAK TILE FLOWER BEDS AND JEWEL BOXES` …
> `BRIDGE ONE-WAY PASS HAMMER`

### Terrain, `$80`–`$9F`

| Code | Name | Glyph | Behaviour |
|---|---|---|---|
| `$80` | tree | `T` | blocks Lolo; **blocks a Medusa's line** |
| `$81` | rock | `#` | blocks Lolo; does *not* block a Medusa's line |
| `$82`–`$87` | river | `~` | blocks Lolo. An egg pushed in floats and can be ridden; see §6 |
| `$88` | floor | `.` | |
| `$89`, `$8A` | bridge | `=` | walkable, no effect |
| `$8B` `$8C` `$8D` `$8E` | one-way pass ↓ ← ↑ → | `v` `<` `^` `>` | refuses entry from the arrow's own reverse only; the other three sides pass. Does not restrict a *push* |
| `$8F` | Emerald Framer | `O` | pushed one cell when walked into; blocks a Medusa's line |
| `$90` | heart framer | `H` | collected on contact |
| `$91` | heart framer, **magic** | `h` | collected on contact **and gives two magic shots** |
| `$92` | desert | `,` | walkable; halves Lolo's speed |
| `$93`, `$94`, `$9F` | break tile | `x` | walkable; changes tile as it is crossed |
| `$95` | flower bed | `*` | walkable, no effect |
| `$96` | door | `D` | walkable at any time; opens when the last heart is collected |
| `$97`–`$9C` | marker | `o` | walkable, no observed effect. At most one per room in 42 rooms; likely the enemy hole or the jewel box, not established |

The `$90`/`$91` split is the one that matters most and is easy to miss: they draw the *same* tile
and are indistinguishable on screen. Collecting `$91` sets `$C4AD` to 2; collecting `$90` leaves
it at 0. A room with no `$91` in it gives Lolo no magic shot at all.

### Actors, `$00`–`$23`

Each character occupies four consecutive codes, one per facing.

| Codes | Character | Moves when Lolo moves? |
|---|---|---|
| `$00`–`$03` | Lolo | exactly one per room; the sprite starts on it |
| `$04`–`$07` | Leeper | yes; walks toward Lolo, then stops dead |
| `$08`–`$0B` | Rocky | yes; charges, and shoves Lolo across the board |
| `$0C`–`$0F` | Alma | yes; chases, then leaves the board |
| `$10`–`$13` | Gol | no, while hearts remain |
| `$14`–`$17` | Skull | no, while hearts remain |
| `$18`–`$1B` | Snakey | never |
| `$1C`–`$1F` | Medusa | never; kills at range along its row and column, see §5 |
| `$20`–`$23` | Don Medusa | yes; patrols a path of its own |

The four that move only ever move on a frame in which Lolo moved. Left alone the board is
completely still, which is what makes a settle predicate work at all.

---

## 4. Work RAM

| Address | Name | Notes |
|---|---|---|
| `$C000`–`$C09F` | shadow OAM | the OAM DMA source. **Every actor lives here and nowhere else**: Lolo in slots 0–1, then one pair of 8×16 sprites per enemy or egg |
| `$C3A4` | `Floor` | `(room - 38) div 14` |
| `$C3A5` | `LevelInFloor` | `(room - 38) mod 14` |
| `$C3A6` | `RoomNumber` | index into the bank 13 table |
| `$C3A9` | `HeartsLeft` | heart framers still to collect; the number the status bar shows. The only byte in work RAM that reads 2, then 1, then 0 as two hearts are taken |
| `$C3BE` | `Scene` | which screen is running. `$17` while a graded room is being played; `$14` for a tutorial demonstration room, which plays itself |
| `$C3BF`–`$C3FE` | `BoardBuffer` | the room as loaded, 8 × 8 stride 8. Never written again; see §7 |
| `$C400`+`i` | stationary-enemy type | `$FF` for an empty slot, 20 slots |
| `$C414`+`i` | stationary-enemy cell | `(2*row << 4) | (2*col)`, i.e. tile coordinates |
| `$C4AD` | `MagicShots` | shots in hand. `0` on a fresh boot, `+2` per magic heart framer, `-1` per shot |
| `$C5FC` | `CurrentRomBank` | shadow of `$2100` |

`$C400`/`$C414` is a struct-of-arrays with a pointer table at `$C48F` (`$C400 $C414 $C428 $C43C
$C464 $C478`, twenty slots each). It holds only the enemies that never move; Lolo's own position
is not in it.

---

## 5. Rules

### Movement

Lolo walks on a half-cell grid: a d-pad press moves him 8 pixels, half a cell, per 16 frames of
hold.

| Hold (frames) | Cells moved |
|---|---|
| 1–14 | ½ |
| 16–28 | 1 |
| 32–46 | 1½ |
| 60 | 2 |

`lolo_gb.PRESS_TICKS` is 20, in the middle of the one-cell band. `measure_hold_window` measures
the band off the cartridge and reports `(16, 28)`.

### Facing

Lolo faces the direction of his last move action, whether or not it succeeded. Walking right into
a rock still turns him right, and the shot that follows goes right.

### Pushing

A step into an Emerald Framer or an egg pushes it one cell the same way. We measured the
destinations:

| Destination | Framer | Egg |
|---|---|---|
| floor, desert, flower bed, one-way pass (any direction) | pushed | pushed |
| rock, tree, another Framer, an enemy | refused | refused |
| heart framer, door | refused | refused |
| river | refused | floats; see §6 |

A one-way pass refuses a *walk* against its arrow but not a *push*.

### The magic shot

Button A fires one cell in Lolo's facing direction:

- an **enemy** there becomes an **egg**, which can then be pushed;
- an **egg** there is blasted out of the room, freeing the cell for good;
- anything else, and nothing happens.

Each shot costs one from `$C4AD`. A magic heart framer (`$91`) is worth exactly two: after
collecting one, two shots land and a third does nothing.

### Medusa

A Medusa kills Lolo when he stands anywhere in its row or column. The shot takes one action to
arrive, and moving out of the line does not save him, so for planning purposes entering the line
is death.

We measured what blocks the line one cell at a time, by putting each candidate between Lolo and
the Medusa:

| Between them | Lolo |
|---|---|
| nothing, rock, river, bridge, desert, flower bed, one-way pass, break tile | **dies** |
| tree, Emerald Framer, heart framer, an enemy | survives |

A rock does not block a Medusa; a tree does.

An **egg** was not in that sweep, but it shields: five of the plans in `tests/data/lolo_solutions.json`
depend on one standing between Lolo and a Medusa, and all five cleared their room on the cartridge.
A **marker** (`$97`-`$9C`) was not in the sweep either and is still untested; see §9.

### Clearing a room

The door tile changes on the frame the last heart framer is taken, from *n* to *n*+2 in every
environment we measured, and stepping onto the open door ends the room. `lolo_gb.is_goal` is
exactly that: `hearts_left == 0` and Lolo standing on the door cell.

---

## 6. Rafts

An egg pushed into a river floats rather than being refused, and Lolo can step onto it and ride
across. We measured this on all six river codes, with a magic heart, a Snakey and a one-cell-wide
river:

| Code | Push into it | Lolo steps on |
|---|---|---|
| `$82` | refused | n/a |
| `$83` | accepted | rides, and drifts **up** |
| `$84`, `$86`, `$87` | accepted | rides, and stays put |
| `$85` | accepted | rides, and drifts **down** |

That drift is a current, meaning the raft keeps moving after Lolo boards it. It is how `int 1-3`
is cleared, and it is why tutorial 14a, whose river is `$83`, cannot be crossed by stepping
straight on. `lolo.py` refuses to push an egg into a river rather than model a moving raft, while
`lolo_gb` does whatever the cartridge does.

---

## 7. Where the live position is

`$C3BF` is written once by `LoadRoom` and never again. Collecting a heart, pushing a Framer and
opening the door all leave it byte-for-byte as it was loaded.

What the game redraws is the BG tilemap, so that is where `lolo_gb` reads the live position: the
playfield is BG columns 1–16 and rows 1–16, one 2 × 2 tile block per cell, and cell `(r, c)`
starts at tile `(1 + 2c, 1 + 2r)`. Actors are not there at all; they are sprites in `$C000`.

Tile numbers are per-environment: the cartridge ships eight terrain themes and the same cell code
draws different tiles in each, so `lolo_gb.learn_tiles` measures the four numbers it needs (heart,
Framer, closed door, open door) from the freshly loaded room, while the board buffer and the
tilemap still agree, and decodes every later frame against that.

---

## 8. Booting

The route from power-on to a graded room runs in six steps:

1. 400 frames of title animation.
2. START, for the NEW GAME / CONTINUE wheel.
3. A, for NEW GAME, then the King's introduction.
4. 27 taps of A reach the screen offering `Push A: ENTRY / Push B: INTERMEDIATE`.
5. B, not A. A starts the tutorial, where the first room of every pair is a demonstration the game
   plays for itself, a board that answers every action with a position nobody asked for.
6. A through the orchestra cutscene until `$C3BE` reads `$17` and a board is up.

`LoloGBEnv.reset` retries the neighbouring counts at step 4, because being one screen out lands in
the tutorial, and `is_playing` is what catches it.

---

## 9. Not resolved

| Item | Status |
|---|---|
| **How the six mobile enemies move** | Nothing here records it. §3 says *that* Leeper, Rocky, Alma and Don Medusa move when Lolo moves, and sketches each in a clause -- "walks toward Lolo, then stops dead", "charges, and shoves Lolo across the board", "chases, then leaves the board", "patrols a path of its own" -- but no probe fixes a trigger, a range, a tie-break between two equally close routes, or what ends the walk. Gol and Skull have the row below. Until somebody probes them, `lolo.py` cannot move them without guessing, and leaves all six standing |
| Whether a **marker** blocks a Medusa's line | Not tested. It matters: `tutorial 13a` is cleared on the cartridge by walking column 0 under a marker at `(3, 0)` that would otherwise be a Medusa's clear line, and treating the marker as a shield is the one change that makes `lolo.py` clear that room too. It would also say what `$97`-`$9C` are |
| The hammer | int 1-5 starts with one in the status bar's PWR meter and cannot be cleared without it. What button uses it, and what it breaks, is not established |
| `$97`–`$9C` | walkable and inert in the environment tested. "ENEMY HOLES" and "JEWEL BOXES" are both in the cartridge's own object list and unaccounted for |
| Gol and Skull waking | both are dormant while hearts remain; whether they move once the last one is taken was not tested |
| Whether `$C4AD` carries between rooms | `lolo_gb` boots each room cold, so it starts at 0. On a real playthrough the meter is the player's, not the room's |
| The six river codes | which is which shape, and what sets a current's direction |
