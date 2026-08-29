# Boxxle II (USA, Europe) Game Boy Memory Map

Reverse-engineering reference for reading live game state: the board, the keeper's position,
and which of the 120 levels is loaded. This is the source for every address the
[`Boxxle2GBEnv`](boxxle2-gb.md) environment reads.

| | |
|---|---|
| File | `Boxxle II (USA, Europe).gb` |
| Size | 32,768 bytes (32 KiB) |
| MD5 | `308abd707a48ee9d69c287d818469fd6` |
| SHA-1 | `36315dab12915d2d2fad7a37fcb5ce6809118c8a` |
| Title (`$0134`) | `BOXXLE2` |
| Cartridge type (`$0147`) | `$00` — **ROM ONLY**, no mapper |
| ROM size (`$0148`) | `$00` — 32 KiB, 2 banks, no banking |
| Cartridge RAM (`$0149`) | `$00` — **none** |
| Destination (`$014A`) | `$01` — non-Japanese |
| Old licensee (`$014B`) | `$CE` — FCI / Pony Canyon |
| CGB / SGB flags | `$00` / `$00` — DMG only |
| Header checksum | `$14` — valid |
| Global checksum | `$AED5` — valid |
| Entry point | `$0100`: `NOP` / `JP $0150` |

Because the cartridge type is ROM ONLY, **CPU `$0000–$7FFF` maps 1:1 onto file offsets**
(every ROM address below is also a file offset), and because there is no cartridge RAM, all
game state lives in the 8 KiB of work RAM at `$C000–$DFFF` plus HRAM. Progress is carried by
a four-character passkey, not by a battery save.

## Where this came from, and what is verified

Two independent passes, and they are worth keeping apart because they carry different weight.

**Static.** A purpose-built LR35902 disassembler and recursive-descent tracer reached 13,588
bytes (41.5%) of the ROM as code in 158 subroutines, resolved every indirect dispatch, and
accounted for the remainder as data. That pass produced the level format, the compression
schemes and the WRAM labels.

**Live, on PyBoy.** Everything the environment actually depends on was then re-checked by
running the cartridge, because a disassembly can be read wrongly and a running console cannot.
Three of the checks are worth naming, and one of them changed a documented fact:

* All 120 boards were loaded through the level-select hook and compared, cell for cell,
  against the same 120 records decoded straight out of the ROM image. **0 mismatches.**
* Random walks (all 120 levels, twenty-five moves each) were replayed on the cartridge and
  on the pure-Python twin and compared after every one of the 3,000 moves. **0 divergences.**
* **`$C34E` does not hold the values the dispatch table suggests.** The disassembly reads
  state 4 as gameplay and 0 as the title screen; the running game uses 4 for the title screen
  and 0 for gameplay. The environment uses the measured numbers, listed in §5.

---

## 1. Quick reference

The five things worth memorising:

| What | Where |
|---|---|
| Goal squares | `$C922 + 20*row + col`, one byte per cell, `$00`/`$01` |
| Boxes | `$CA8A + 20*row + col` |
| Walls | `$CBF2 + 20*row + col` |
| Keeper | `$C110:$C10F`, a 16-bit linear offset into the same 20-stride grid |
| Board size | `$C121` = width, `$C120` = height |

The stride is **20**, not the 32 of the hardware tilemap. A level is solved exactly when every
cell set in the box plane is also set in the goal plane.

---

## 2. ROM map

| Range | Size | Contents |
|---|---|---|
| `$0000–$014F` | 336 | vectors and header. Only VBlank (`$0040`→`$02DA`) and Timer (`$0050`→`$0324`) are enabled; `IE` is `$05` once running |
| `$0150–$37F5` | 14,502 | all game code |
| `$37F6–$455D` | ~3.4 K | records, text and tilemap fragments |
| `$455E–$47FE` | 673 | pointer tables and their records |
| `$47FF–$480E` | 16 | SFX track pointer table |
| `$480F–$481E` | 16 | BGM track pointer table |
| `$481F–$4E17` | ~1.5 K | music and instrument data |
| **`$4E18–$4F07`** | 240 | **level pointer table, 120 words** |
| `$4F08–$4F22` | 27 | the attract-mode level, used when `$C350 ≠ 0` |
| **`$4F23–$61D8`** | 4,790 | **the 120 level records** |
| `$61D9–$7B49` | 6,513 | 8 compressed graphics blobs |
| `$7B4A–$7FFF` | 1,206 | music tables and sequences |

Code landmarks the environment cares about:

| Address | Name | Role |
|---|---|---|
| `$0224` | `MainLoop` | top of frame |
| `$04E4` | `CellXY_To_Offset` | `HL = Y*20 + X`, run once when a level starts |
| `$0B62` | `ProbeMoveTarget` | the Sokoban push test |
| `$0D88` | `StartLevel` | |
| **`$0F53`** | **`LoadLevelHeader`** | reads `$C162` at `$0F5D` and `$C352` at `$0F6E` — the hook point |
| `$100B` | `CentreBoardScroll` | |
| `$1089` | `RecordMove` | undo log writer |
| `$26C3` | `DecompressLevel` | level stage 1 |
| `$27F6` | `UnpackBoardRow` | level stage 2 |
| `$2787` | `DecompressToVram` | graphics |

---

## 3. The level format

**(verified two independent ways, 0 failures on all 120 records)**

### 3.1 Pointer table

120 little-endian words at `$4E18`. `LoadLevelHeader` indexes it at `$0F5A` as

```
entry = $4E18 + StageNumber*20 + LevelInStage*2
```

so the layout is 12 stages × 10 levels, stage in `$C162` (0–11) and level in `$C352` (0–9).
The environment's `fix_index(i)` writes `i // 10` and `i % 10` into those two bytes.

### 3.2 Record layout

| Offset | Field |
|---|---|
| `+0` | `W` — board width in cells |
| `+1` | `H` — board height in cells |
| `+2` | keeper start column, **1-based** |
| `+3` | keeper start row, **1-based** |
| `+4` | flag bitmap, `ceil(3*W*H/64)` bytes, MSB first |
| … | literal bytes, one per set flag bit |

**Stage 1** (`$26C3`) expands the bitmap and literals into `$CD5A`, producing
`ceil(3*W*H/8)` bytes: a set bit takes the next literal, a clear bit emits `$00`. The output
length is implied by `W` and `H` rather than stored, which is why a decoder has to compute the
record size to find the next record.

**Stage 2** (`$27F6`) reads that byte stream as one continuous MSB-first bitstream (`W` bits
per row, `H` rows, three consecutive planes), expanding each bit into a whole byte in a
20-byte-stride buffer:

| Plane | Destination | Meaning |
|---|---|---|
| 0 | `$C922` | goal squares |
| 1 | `$CA8A` | boxes |
| 2 | `$CBF2` | walls |

Both checks on this format:

1. Computed record sizes match the pointer-table deltas for **all 120 levels**, and the last
   record ends exactly at `$61D8`, immediately before the first graphics blob.
2. `popcount(plane 0) == popcount(plane 1)` for **every one of the 120 levels**: 1,447 goals
   and 1,447 boxes in total, which is the invariant Sokoban requires and strong evidence the
   plane assignment is the right way round.

`boxxle2_gb.read_levels` is this decoder in Python, and
`boxxle2_gb.verify_level_table` is check 1 as a function you can run on a dump in hand.

Board dimensions run from 6×5 to 16×16 and box counts from 3 to 59.

### 3.3 Cell size and centring

`LoadLevelHeader` picks a zoom from the dimensions:

```
if W < 12 and H < 11:  CellSize ($C0FE) = $10   ; 16px cells, 2x2 tiles
else:                  CellSize ($C0FE) = $08   ; 8px cells, 1x1 tile
```

and `CentreBoardScroll` at `$100B` centres the board by writing negative scroll shadows. None
of this touches the plane buffers, so the environment reads the same grid either way, which
was checked on 14×12 and 16×16 boards as well as small ones.

---

## 4. Board addressing

The three plane buffers are 360 bytes each (`$168`), laid out as **20 columns × 18 rows with a
stride of 20**. `GetPlaneBase` at `$0C05` selects one with `HL = $C922 + PlaneSelect * $168`.

A cell is addressed by a 16-bit linear offset held in `$C110:$C10F` (high:low). The conversion
happens once per level at `$04E4`:

```
HL = 0
repeat startY times: HL += 20
HL += startX
$C10F = L ; $C110 = H          (also copied to $C111/$C112 and $C113/$C114)
```

Moves are applied to that offset by `ProbeMoveTarget` at `$0B62`:

| `$C116` | Direction | Step |
|---|---|---|
| 1 | left | −1 |
| 2 | up | −20 |
| 3 | down | +20 |
| 4 | right | +1 |

The offset and the plane buffers are updated **in the same frame the pad is read**. What takes
sixteen frames afterwards is the sprite, not the state; see §6.

---

## 5. WRAM map

| Range | Contents |
|---|---|
| `$C000–$C09F` | **Shadow OAM** — 40 sprites × 4 bytes, the OAM DMA source |
| `$C0A0–$C0FF` | rendering scratch, sprite staging, decompressor counters |
| `$C100–$C19C` | main gameplay variables |
| `$C334–$C359` | mode, timing and bitstream state |
| `$C35B–$C41A` | sound channel state, two 96-byte slots |
| `$C41B–$C450` | sound control, undo log |
| `$C922–$CEC1` | board plane buffers and the level decompression scratch |

### 5.1 The addresses the environment reads

| Address | Name | Purpose |
|---|---|---|
| `$C000` | `ShadowOAM` | 160 bytes; the keeper's slide is written here and nowhere else |
| `$C0FE` | `CellSize` | `$10` or `$08` |
| `$C10F`/`$C110` | `PlayerOffset` | linear board offset, low/high |
| `$C116` | `MoveDirection` | 1 left, 2 up, 3 down, 4 right |
| `$C120` | `BoardHeight` | |
| `$C121` | `BoardWidth` | |
| `$C162` | `StageNumber` | 0–11 |
| `$C34E` | `GameState` | main-loop dispatch selector — see below |
| `$C350` | `DemoLevelFlag` | non-zero forces the attract-mode board at `$4F08` |
| `$C351` | `EditModeFlag` | selects the RAM-resident board used by CREATE |
| `$C352` | `LevelInStage` | 0–9 |
| `$C440` | `MoveHistory` | undo log, 2 bits per move |
| `$C922` | `BoardGoal` | 360 bytes |
| `$CA8A` | `BoardBox` | 360 bytes |
| `$CBF2` | `BoardWall` | 360 bytes |

### 5.2 `$C34E`: measured, not inferred

These are the values the running cartridge takes, watched frame by frame from power-on. They
are **not** the numbering the `MainLoop` dispatch table suggests, and where the two disagree
these are the ones that hold:

| Value | Screen |
|---|---|
| `$04` | title — "PUSH START KEY" |
| `$10` | MUSIC: BGM A / B / C |
| `$20` | MENU: PLAY / PASSKEY / CREATE |
| `$06` | the story cutscene before stage 1 |
| `$00` | **playing** (and also the first ~60 frames after power-on) |
| `$40` | the pause overlay START opens mid-level |
| `$50` | the level-cleared sequence |

Because `$00` means both "playing" and "still booting", the environment's `level_is_loaded`
requires the state *and* a plausible board size *and* a keeper on the board *and* at least one
box before it believes a level is up.

---

## 6. Driving the game

Measured on the cartridge; none of it is in the ROM's data.

### 6.1 One press, one cell

Holding a direction for 1–19 frames moves the keeper exactly one cell. At **20 frames the
d-pad repeats** and one press becomes two moves, which in a Sokoban is not a longer plan but
a box shoved somewhere nobody asked for. `boxxle2_gb.measure_hold_window` re-derives this
bound from whatever dump is in hand rather than trusting the number here, and picks the middle
of the window.

### 6.2 What "settled" means

The plane buffers and `$C10F`/`$C110` are correct one frame after the press. The **keeper is
not**: it slides one pixel per frame for sixteen frames, and until it arrives every further
press is ignored. That slide is visible only in the shadow OAM at `$C000`: a byte-by-byte
diff of `$C0A0–$C460` across the animation finds nothing that is not also changing when the
game is idle. So the settle predicate is the three planes, the keeper offset **and** the
160-byte sprite buffer, all holding still together for three frames.

Watching only the planes drops roughly every second press. That failure looks exactly like a
planner's action having no effect, which is the reason this is written down here.

Two further things are needed before the settle is trustworthy. The slide **pauses for a frame
or two partway through**, and a hold long enough to trip auto-repeat looks perfectly settled in
the gap between the first move and the second, so the settle is also held open until frame 22,
one past where the repeat fires. And after a save-state is restored, the machine needs **two
idle frames** before it will see a button edge at all: press on the very next frame and
`ReadJoypad` misses it, on some states but not others.

### 6.3 Clearing a level destroys the board

About 320 frames after the last box goes home the cartridge switches `$C34E` to `$50` and runs
its congratulation-and-replay sequence, which **rewrites the plane buffers with something that
is not a Sokoban position**. A board snapshotted 30 frames after the winning push decodes as
garbage: boxes and goals scattered over the walls. So the environment stops settling
the instant every box is home, and treats a solved state as absorbing.

### 6.4 Getting to a level

The front end is three screens deep: title (`$04`) → music (`$10`) → menu (`$20`), all three
advanced by START, then a story cutscene (`$06`) that runs about 970 frames and **cannot be
skipped**. Power-on to a playable board is roughly 1,370 frames, about 0.05 s of emulation.

START during play opens the pause overlay, so a boot routine that taps START on a timer will
eventually pause the game it just started. The environment presses START only while `$C34E`
names one of the three menu screens.

Level selection is done by hooking `LoadLevelHeader` at `$0F53` and writing `$C162`, `$C352`
and `$C350` from the hook body. Writing them from outside on a frame boundary does not work:
the menu resets both counters and calls the loader within the same frame, and the loader wins.

### 6.5 The passkey

`PASSKEY` on the menu takes four characters from a 35-glyph alphabet
(`BDGHJKLMNPQRTVWXYZ!?0123456789` plus five card-suit symbols) and jumps to the level it
encodes. The encoding has **not** been located in the ROM, so the environment does not use
this route; the `$0F53` hook reaches all 120 levels without it. Third-party walkthroughs
publish the table if you want to type one by hand.

---

## 7. Hardware registers

Only these are ever touched: `P1` (`$FF00`, polled; the joypad interrupt is unused), `TAC`
(`$FF07`, `$07`, giving a 64 Hz audio tick), `NR30`/`NR50`/`NR51`/`NR52`, `LCDC` (`$FF40`,
`$C3` when on), `STAT`, `SCY`/`SCX` from the shadows at `$C127`/`$C125`, `LY`, `DMA` (`$FF46`,
only from the HRAM trampoline at `$FF80`), `BGP`/`OBP0` = `$E4`, `OBP1` = `$7F`, `WY`/`WX`,
and `IE`.

### Joypad encoding

`ReadJoypad` at `$25A8` selects the direction row, `SWAP`s it into the high nibble, then ORs
in the button row:

| Bit | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |
|---|---|---|---|---|---|---|---|---|
| Button | Down | Up | Left | Right | Start | Select | B | A |

Bits are active-high after the `CPL`, and `JoypadPressed` is `(old XOR new) AND new`, an edge,
which is why a held button does not move the keeper twice until auto-repeat fires.

---

## 8. What is not resolved

| Item | Status |
|---|---|
| Passkey encoding | not located; §6.5 |
| Sound engine internals (`$29E9`, the 96-byte channel struct) | entry points and slot layout mapped, per-field semantics not decoded |
| `$3804–$455D` | ~3.4 KiB of records and tilemap fragments, partially indexed |
| CREATE mode (`$C351`) | the RAM board pointers at `$1005` are identified; the editor UI is not traced |
| Cell → tile lookup | the environment reads the planes, so it never needed one |

None of these is on the path the environment takes.
