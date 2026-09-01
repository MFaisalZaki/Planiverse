# Flipull (USA) Game Boy memory map

Reference for reading live game state out of `Flipull (USA).gb`: the block field, block types and
the on-screen counters. This is the source for every address the [`FlipullGBEnv`](flipull-gb.md)
environment reads.

| | |
|---|---|
| File | `Flipull (USA).gb` |
| Size | 32,768 bytes (32 KiB) |
| MD5 | `4fcc13db8144687e6b28200387aed25c` |
| Title (`$0134`) | `FLIPULL` |
| Cartridge type (`$0147`) | `$00`, ROM ONLY, no mapper |
| ROM size (`$0148`) | `$00`, 32 KiB, 2 banks |
| Cartridge RAM (`$0149`) | `$00`, none |
| Destination (`$014A`) | `$01`, non-Japan |
| Header checksum | `$0E`, valid |
| Global checksum | `$8AB0`, valid |
| Entry point | `$0100`: `NOP` / `JP $0150` |

No mapper means the entire ROM is flat at `$0000–$7FFF` with no bank switching, which is the
simplest possible cartridge. There is no cartridge RAM, so all state lives in WRAM `$C000–$DFFF`
and HRAM `$FF80–$FFFE`, and Flipull leans on HRAM unusually heavily, keeping nearly every counter
there rather than in WRAM.

---

## 1. Quick reference

| What | Where |
|---|---|
| Block field | `$C840 + 32*row + col`, 14 rows |
| Blocks remaining | `$FFC9` (ones) + `$FFCA` (tens), decimal digits |
| Timer | `$FFCE` min, `$FFCC` sec-tens, `$FFCB` sec-ones |
| Clear target | `$FFCF` |
| Stage number | `$FFC6` (ones) + `$FFC7` (tens), decimal digits |
| Previously held block | `$FFD4` |

A cell holds a playable block when its value is `$83`–`$86`.

---

## 2. The block field (`$C840`)

```
cell_address = $C840 + 32*row + col        ; row stride $20
```

The field spans 14 rows, `$C840` through `$C9E0`, starting at:

| row | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| addr | `C840` | `C860` | `C880` | `C8A0` | `C8C0` | `C8E0` | `C900` |

| row | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|
| addr | `C920` | `C940` | `C960` | `C980` | `C9A0` | `C9C0` | `C9E0` |

Row 0 is the ceiling and row 13 the floor, and both read `$80` across 16 columns. Column 0 is the
left wall (`$80`) on every row, so the usable field is columns 1 to 15, of which stage 1 occupies
columns 1 to 5.

Although the row stride is 32 bytes, only the first 16 of each row carry meaning, since the upper
half of every row read `$00` throughout observation.

### Cell values

| value | meaning |
|---|---|
| `$00` | outside the field / unused |
| `$80` | border: ceiling, floor, left wall |
| `$83`–`$86` | playable block, four distinct types |
| `$87` | fixed staircase wall, the diagonal in the upper left of Stage 1 |

`$87` should not be confused with a playable block. It is structural, forming the stepped
diagonal, and is excluded from the block count, so count only `$83`–`$86`.

Stage 1 opens with five rows of five blocks in rows 8–12, columns 1–5:

```
row08 C940   80 84 85 85 86 85
row09 C960   80 84 85 83 83 84
row10 C980   80 86 85 84 85 86
row11 C9A0   80 84 84 83 84 83
row12 C9C0   80 83 86 86 86 83
```

That gives 25 cells in `$83`–`$86`, matching the on-screen `BLOCK 25`. After one throw the count
read 24 and the grid held 24.

### Column collapse

When a block is destroyed its column falls, as observed on column 5:

```
before (rows 8-12):  85 84 86 83 83
after:               00 85 84 86 83
```

The column shifted down one cell and the bottom block was removed, so gravity is per-column and
immediate.

---

## 3. Counters (HRAM)

Flipull stores counters as separate decimal digits, ones first, rather than as binary or packed
BCD. Searching for `25` or `$19` finds nothing, because the value lives as `05` and `02` in
adjacent bytes.

| Address | Field |
|---|---|
| `$FFC9` | Blocks remaining, ones digit |
| `$FFCA` | Blocks remaining, tens digit |
| `$FFC0` | Stage's initial block count, ones digit |
| `$FFC1` | Stage's initial block count, tens digit |
| `$FFCB` | Timer, seconds ones digit |
| `$FFCC` | Timer, seconds tens digit |
| `$FFCE` | Timer, minutes |
| `$FFCD` | Sub-second tick counter |
| `$FFCF` | Clear target, the `CLEAR` number |
| `$FFC6` | Stage number, ones digit |
| `$FFC7` | Stage number, tens digit |

`$FFC0`/`$FFC1` should not be confused with the live count: they hold the stage's starting total
and do not move as blocks are cleared. Read `$FFC9`/`$FFCA` for the live value:

```python
blocks_left = hram[0xFFCA] * 10 + hram[0xFFC9]
seconds     = hram[0xFFCE] * 60 + hram[0xFFCC] * 10 + hram[0xFFCB]
stage       = hram[0xFFC7] * 10 + hram[0xFFC6]
```

The clock starts at `3:00`: the loader writes `$FFCE = 3` and zeroes the seconds.

---

## 4. The thrown block

| Address | Field |
|---|---|
| `$FFD4` | The block **previously** in hand, the one just thrown. It lags the hand by one throw and reads `$00` until the first throw of a stage |
| `$FFD2`, `$FFD3` | A count of completed throws: `0,0 → 1,1 → 2,2 → 3,3`. It stays `0` for the whole flight and rises only when the block lands, and does not move at all for a throw that changes nothing |
| `$FFDF` | A free-running counter. It falls by 17 a frame, wrapping through zero, whether or not anything is in flight |
| `$FFDE` | Unidentified; drifted `56`→`58` |
| `$FFAF` | Free-running counter / RNG; advances about 50 a frame regardless of input |

There is no address holding the block currently in hand. That value is the hand sprite's tile,
which carries the same `$83`–`$86` encoding as the field, and [the environment
page](flipull-gb.md#calibration) describes how it is read.

---

## 5. The stage table

`0:2D55` is the loader. It turns the two stage digits into `10*tens + ones - 1`, indexes a table
of pointers at `$3A0E`, and copies the three bytes each entry points at into the HUD counters:

```
2D64  ldh a,($FFC6)     ; ones
2D66  add a,b           ;   + 10 * tens
2D67  sub $01           ; zero-based
2D69  rlca              ; two bytes per pointer
2D6A  ld hl,$3A0E       ; the table
...
2D7B  ldh ($FFCA),a     ; blocks, tens digit
2D80  ldh ($FFC9),a     ; blocks, ones digit
2D85  ldh ($FFCF),a     ; the CLEAR target
```

Each descriptor is `[clear target, blocks ones, blocks tens]`. The table holds 32 entries, and a
second, shorter table follows at `$3A4E` whose first pointer runs back to stage 1's descriptor, so
reading past the end of the first table silently builds stage 1 again.

`0:1673` advances the stage number, carrying the ones digit into the tens at ten:

```
1673  ld hl,$FFC6
1676  inc (hl)          ; stage++
1677  ld a,(hl)
1678  cp $0A            ; ...and at ten,
167A  jr nz,$1681
167C  ld a,$00
167E  ld (hl+),a        ; zero the ones, step to $FFC7
167F  jr $1676          ; and carry into the tens
```

---

## 6. Other regions

| Range | Contents |
|---|---|
| `$CA00–$CA23` | 36 bytes of block-type values (`$83`–`$86`), shifting during play. Unidentified; most likely the queue of upcoming blocks |
| `$CA30–$CA35` | Small counter block; `$CA32` held `09` matching `CLEAR`. Purpose unresolved |
| `$C000–$C00A` | Frequently-written bytes; `$C002` tracks vertical input (`89`/`8F`) rather than position |
| `$CFC0–$CFFF` | Mixed working area, changes continuously |

---

## 7. Reading state

```python
GRID, STRIDE, ROWS, COLS = 0xC840, 0x20, 14, 16

def cell(row, col):
    return GRID + row * STRIDE + col

blocks = [(r, c, wram[cell(r, c)])
          for r in range(ROWS) for c in range(COLS)
          if 0x83 <= wram[cell(r, c)] <= 0x86]

remaining = hram[0xFFCA] * 10 + hram[0xFFC9]
assert len(blocks) == remaining

walls = [(r, c) for r in range(ROWS) for c in range(COLS)
         if wram[cell(r, c)] == 0x87]  # fixed staircase, not clearable
```

---

## 8. Not resolved

| Item | Status |
|---|---|
| The score | Not located. It advances 0 → 100, but no candidate byte was isolated |
| The field's address calculator | Not found in code. The `$C840`/stride-32 geometry comes from the dump's structure rather than from the routine that computes it |
| `$CA00–$CA23` | Read as the upcoming-block queue from its contents alone |
| Field width beyond column 5 | Inferred from the ceiling and floor rows reading `$80` across 16 columns; no stage has been seen using the full width |
| What a throw hits | Not established. See [the environment page](flipull-gb.md#what-a-throw-hits-is-not-modelled) |
