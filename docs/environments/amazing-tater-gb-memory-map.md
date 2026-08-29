# Amazing Tater (U) Game Boy Memory Map

Reverse-engineering reference for reading live game state: the composed board, which tater has
the controls, and which of them are still out. This is the source for every address the
[`AmazingTaterGBEnv`](amazing-tater-gb.md) environment reads, and for the 105 rooms stored in
the [pure-Python twin](amazing-tater.md).

| | |
|---|---|
| File | `Amazing Tater (U).gb` |
| Size | 65,536 bytes (64 KiB) |
| MD5 | `53b746bff74c50cd3ebcf41161c66cf3` |
| Title (`$0134`) | `AMAZING-TATER` |
| Cartridge type (`$0147`) | `$01` — MBC1 |
| ROM size (`$0148`) | `$01` — 64 KiB, 4 banks |
| Cartridge RAM (`$0149`) | `$00` — **none** |
| Destination (`$014A`) | `$01` — non-Japanese |
| Old licensee (`$014B`) | `$EB` — Atlus, 1991 |
| CGB / SGB flags | `$00` / `$00` — DMG only |
| Header checksum | `$45` — valid |
| Global checksum | `$A7D0` — valid |
| Entry point | `$0100`: `NOP` / `JP $0150` |

Because the cartridge has no external RAM, **all game state lives in the 8 KiB of work RAM at
`$C000–$DFFF`** plus HRAM. The stack lives in WRAM at `$DCFF`, not in HRAM, which is why HRAM
here holds almost nothing but the OAM DMA trampoline.

Amazing Tater is the sequel to Kwirk (*Puzzle Boy II* in Japan) and plays by the same rules:
walk a tater to the exit flag past pushable blocks, pits that must be filled, and turnstiles
that swing when you shove an arm.

---

## 1. Quick reference

The five things worth memorising:

| What | Where |
|---|---|
| The composed board | `$C2F2`, 20 columns × 18 rows, stride 20, one byte per cell |
| Room width + 2, height + 2 | `$C2BD`, `$C2BE` |
| Taters still out | `$C2AD` — **a bitmask**, not a count; `0` means the room is solved |
| Tater holding the controls | `$C2AE` — the character's number, 0–3 |
| The loader | `$08C0`, entered with `HL` = level index × 2 |

Cell codes are in §4. A single hook on `$08C0` is enough to choose any level in a set.

---

## 2. Bank layout

MBC1 keeps bank 0 fixed at `$0000–$3FFF` and switches banks 1–3 into `$4000–$7FFF`. Bank
selection is always

```
LD A,n / LD ($C2D3),A / LD ($2000),A
```

with `$C2D3` holding a shadow copy of the current bank. There are 19 such sites, all in bank 0.

| Bank | Role |
|---|---|
| 0 | The entire engine — 14,873 of 16,384 bytes trace as code |
| 1 | A dispatch stub at `$4000`, the sound driver at `$6026–$62A5`, and data |
| 2 | Data only, apart from one 47-byte stub at `$4000` |
| 3 | **All level data**, plus its pointer tables |

Four interrupts are in use: VBlank (`$0593`), LCD STAT (`$1904`, a raster split driven by
`LYC`), Serial (`$06EC`; this game has link-cable two-player), and the timer and joypad
vectors are bare `RETI`. `RST $30` is a four-slot request queue at `$DF80`.

---

## 3. WRAM layout

213 distinct WRAM addresses are referenced by traced code. The clusters that matter:

| Range | Size | Contents |
|---|---|---|
| `$C000–$C09F` | 160 | OAM DMA source buffer (40 sprites × 4 bytes) |
| `$C100–$C14C` | 77 | Main game variables |
| `$C142–$C2A9` | 360 | Screen tilemap staging, 20×18, copied to VRAM by `$0D38` |
| `$C2AC–$C2F1` | 70 | Engine state, including everything in §5 |
| `$C2F2–$C459` | 360 | **The composed board**, 20×18 (see §4) |
| `$D35C–$D3B7` | 92 | Level working set — the second hot cluster |
| `$DCFF` | — | Stack top |
| `$DF80–$DF84` | 5 | `RST $30` request queue: a count and four slots |

---

## 4. The composed board (`$C2F2`)

360 bytes, **20 columns × 18 rows with a stride of 20**, not the hardware tilemap's 32. One
byte per cell, and that byte is the whole game state for that cell: terrain, whatever object
is standing on it, and enough shape information to say what that object is joined to.

`LoadLevel` centres the room in the buffer with a one-cell border, writing `W + 2` to `$C2BD`
and `H + 2` to `$C2BE`, and the horizontal offset `(20 − (W + 2)) / 2` to `$C2BF`.

### Cell codes

| Code | Meaning |
|---|---|
| `$00` | Floor |
| `$40`–`$4F` | A block square standing on floor. The low nibble is a mask of which neighbours are part of the same block: **1 right, 2 down, 4 left, 8 up** |
| `$50`–`$5F` | The same, for a square that has settled into a pit |
| `$80`–`$83` | A turnstile arm. The low two bits are the direction from its pivot: 0 up, 1 right, 2 down, 3 left |
| `$90`–`$93` | The same four arms, hanging over a pit |
| `$A0`–`$AE` | A turnstile pivot. `code − $A0` indexes the shape table at `$0BF6` (see below) |
| `$C0`–`$C3` | A tater, by character |
| `$D0` | The exit flag |
| `$E0` | An open pit |
| `$F0`–`$FE` | Wall. Fifteen graphics; the distinction is cosmetic |
| `$FF` | Outside the room |

The block mask is what makes a block a block. Two *different* blocks sit flush against each
other in half the rooms on this cartridge, and the mask is the only thing that says so: a
square joined to nothing is `$40`, and one joined on all four sides is `$4F`. Every square's
mask agrees with its neighbours' in all 201 rooms, which is the cheapest evidence there is
that this reading is right.

### The turnstile shape table (`$0BF6`)

Fifteen bytes:

```
80 41 22 13 CE 6D 34 95 E6 77 B8 D9 FA AB 5C
```

The **high nibble** of entry `i` is the arm mask of the pivot whose code is `$A0 + i`, with
bit 8 up, 4 right, 2 down and 1 left. The fifteen high nibbles are exactly the fifteen
non-empty four-bit masks (four single arms, four L-shapes, four T-shapes, two straight
turnstiles and the four-armed one), which is what identifies the table. Every pivot on the
cartridge has precisely the arms its entry claims, and every arm's direction code points back
at a pivot.

### Reading a board

```python
buffer = pyboy.memory[0xC2F2:0xC2F2 + 360]
width  = pyboy.memory[0xC2BD] - 2
height = pyboy.memory[0xC2BE] - 2
cell   = buffer[row * 20 + col]
```

`amazing_tater_gb.decode_board` does this and turns each code into the glyph the twin's levels
are written in.

---

## 5. Game state variables

| Address | Name | Notes |
|---|---|---|
| `$C131` | `GameMode` | 0 PUZZLE, 1 and 2 PRACTICE, 3 BEGINNER and ACTION. Selects the level set |
| `$C2AC` | `Overlay` | `$00` while the room is being played; non-zero under the A-button pause menu |
| `$C2AD` | `TatersOnBoard` | **A bitmask**: bit 0 is the first character, bit 3 the fourth. Zero once every tater has reached the flag |
| `$C2AE` | `ActiveTater` | Which character SELECT has the controls on, 0–3 |
| `$C2BD` | `BoardWidthPlus2` | |
| `$C2BE` | `BoardHeightPlus2` | |
| `$C2BF` / `$C2C0` | board offsets | Where the bordered box sits in the 20×18 buffer |
| `$C2D3` | `CurrentRomBank` | Shadow of the `$2000` write |
| `$C2F2` | `BoardMap` | §4 |
| `$D37D` | `MenuCursor` | `$80` while the title screen is up, a small number wherever a menu cursor exists |
| `$D385` / `$D386` | stage / level-in-stage | What `$085F` turns into a level index |

**`$C2AD` is a mask, not a count**, and that took a four-tater room to notice. Three of the
six rooms with more than one tater read 15 there and a room holding the first and fourth
characters reads 9, which is nonsense as a count and exactly right as a mask.

It is also the only sound way to ask whether a room is solved. A tater in mid-step is *taken
off* the board map and drawn as a sprite until it arrives, so for a dozen frames after every
single press the board shows no tater at all: indistinguishable, from the board alone, from a
room that has just been finished.

---

## 6. Level data (bank 3)

### The set descriptor table (`$0C05`)

Nine consecutive words, three per level set: the bases of that set's record-pointer, plane-A
and plane-B arrays. Each set stores its three parallel word arrays of N entries back to back,
immediately followed by the records.

| Set | Records | Plane A | Plane B | Levels | Reached by |
|---|---|---|---|---|---|
| A | `$5000` | `$5052` | `$50A4` | 41 | PUZZLE MODE |
| B | `$5CE2` | `$5DA2` | `$5E62` | 96 | PRACTICE MODE |
| C | `$686F` | `$68EF` | `$696F` | 64 | BEGINNER MODE and ACTION MODE |

The level count is not stored: it is `(plane_a_base − records_base) / 2`, which is how
`amazing_tater_gb.level_counts` derives 41, 96 and 64 from the cartridge rather than from a
constant.

### `LoadLevel` (`$08C0`)

Entered with `HL` holding **twice the level index**, worked out by its caller at `$085F` from
`$D385` (stage) and `$D386` (level within stage) and a per-set stride from the table at
`$0C17` (10, 32, 20). The arithmetic does not cover a set evenly (set A's 41st room is the
one left over after four stages of ten, and set C has a `+2` fudge for its third stage), which
is exactly why this environment writes `HL` on the way in rather than reversing it.

For modes 0 and 3 the loader reads the room's width and height from the record. For modes 1
and 2 it does not: it keeps the 10 × 8 default it wrote on entry, because PRACTICE MODE's rooms
are all 8 × 6.

### Record format

```
+0   W                 board width  in cells
+1   H                 board height in cells
+2   [extra byte]      present on some records only
     plane A           ceil(W/8) bytes per row, H rows, MSB first
     plane B           the same again
     tail              variable length
```

Both planes are **row-aligned** bitmaps: every row restarts on a byte boundary. A continuous
bitstream matches only 15 of set A's 41 records; row alignment matches all of them, and
`planeB_ptr − planeA_ptr == ceil(W/8) × H` holds for all 41 set A and all 64 set C records.

The header length is not constant (set C's records 0–19 have a two-byte header and records 20
onward have three), which is why the plane addresses are stored in their own arrays rather than
derived from the record base.

**The record tail is not decoded here, and does not need to be.** The board this environment
reads is the one the cartridge itself composes at `$C2F2`, which is downstream of every part of
the record; the twin's 105 rooms were dumped from there rather than from the ROM. What the
planes *are* was never satisfactorily pinned down either: plane A is set for walls, pits, the
flag, the taters, every turnstile pivot and exactly those block squares whose mask has neither
a left nor an up bit (the top-left square of each block), which reads like an "object starts
here" layer rather than terrain, and plane B does not separate cleanly at all.

---

## 7. Driving the cartridge

### The front end

Title screen → `SELECT MODE` (BEGINNER, PUZZLE, PRACTICE, ACTION) → then, depending on the
mode chosen, either a tutor with several screens of advice or an `ENTER PASSWORD` grid. START
advances all of them.

Only `SELECT MODE` needs the d-pad, and telling it apart from the other screens is what
`$D37D` is for: it holds `$80` while the title is on screen and a small number wherever a
cursor exists. The password grid reuses the same byte as its own row cursor, so a d-pad press
aimed at the mode menu but landing there walks the alphabet instead, which is why `boot` stops
touching the d-pad the moment the mode is chosen.

### Timing, all measured rather than assumed

| What | Frames | How it was found |
|---|---|---|
| A d-pad press that moves exactly one cell | hold 1–9 | Sweeping the hold length and counting cells moved |
| Auto-repeat: one press, two cells | hold 10+ | The same sweep |
| A move's board writes | up to 16 | Watching `$C2F2` frame by frame |
| The longest still spell *inside* an unfinished move | 8 | The same, over random play — it is a block dissolving into pits |
| Before the room takes its first press | ~60 | Probing a snapshot at increasing offsets |
| After SELECT, before the next press is taken | 33 | Probing two presses at increasing gaps |
| After a direction, before the next press is taken | 5 | The same |

The SELECT lockout is the nastiest of these, because nothing on the board says it happened:
handing the controls over changes no cell, so an ordinary settle predicate is satisfied
immediately and the next press lands inside the lockout and is dropped. Two switches in a row
became one, the plan and the game ended up with different taters under the controls, and the
boards only visibly disagreed several moves later.

### Choosing a level

Hook `$08C0` and write `HL`:

```python
def force(context):
    pyboy, index = context
    pyboy.register_file.HL = index[0] * 2

pyboy.hook_register(0, 0x08C0, force, (pyboy, [level_index]))
```

The game mode still comes from the menu, so the cartridge stays in a state it put itself in.
Writing `$C131` in the same hook does select the other set and does load the right board, but
`$C2AD` then comes back wrong for every multi-tater room, so it is not done.

---

## 8. Verifying in an emulator

```python
from pyboy import PyBoy
pyboy = PyBoy("Amazing Tater (U).gb", sound_emulated=False, window="null")
# ... boot to a room ...
assert pyboy.memory[0xC2BD] - 2 == 15                 # A-01 is 15 x 5
assert bin(pyboy.memory[0xC2AD]).count("1") == 1      # one tater
board = pyboy.memory[0xC2F2:0xC2F2 + 360]
assert board[8 * 20 + 3] == 0xD0                      # the flag
```

`tests/test_amazing_tater_gb.py` does this for real, and `tests/test_amazing_tater.py` replays
stored solutions on both the cartridge and the twin.

---

## 9. Confidence

| Claim | How it was established |
|---|---|
| Header, bank roles, `$0C05`, `$0BF6` | Static analysis, then read back out of the ROM in the tests |
| Level counts 41 / 96 / 64 | Derived from the pointer arrays themselves, and each set was loaded end to end |
| Board buffer, dimensions, offsets | Read live for all 201 rooms and checked against the rendered screen |
| Every cell code | Matched against the rendered screen, and against the shape table for the pivots |
| Block masks and arm directions | Checked for mutual consistency across all 201 rooms |
| `$C2AD` is a mask | Six multi-tater rooms; the mask matches the characters on the board in each |
| `$C2AE` is the active character | Watched across SELECT presses in two-, three- and four-tater rooms |
| The rules themselves | The twin and the cartridge were walked forward in lockstep, the same random press at a time, across all 105 rooms — see [amazing-tater.md](amazing-tater.md) |
| The record tail, plane semantics | **Not established.** Neither is needed; see §6 |
