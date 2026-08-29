# Flipull (USA) Game Boy Memory Map

Reverse-engineering reference for reading live game state: the block field,
block types, and the on-screen counters. This is the source for every address the
[`FlipullGBEnv`](flipull-gb.md) environment reads.

> **Section 4 is wrong, and this document is kept as written anyway.** It was produced by
> recording RAM in a purpose-written emulator across one stage and a handful of throws.
> Driving the environment against the same dump afterwards showed that `$FFD2`/`$FFD3` are a
> completed-throw *count* rather than in-flight flags (they stay `0` for the whole flight),
> that `$FFD4` holds the *previously* held block, and that `$FFDF` is a free-running counter
> rather than the in-flight X. The field geometry, the cell encoding, the digit-per-byte
> counters and the column collapse in sections 2 and 3 all hold exactly as written.
> `$FFC6` is not wrong either, only incomplete: the stage number is two decimal digits and
> `$FFC7` holds the tens, which is what makes all 32 stages selectable.
> [What the cartridge corrected](flipull-gb.md#what-the-cartridge-corrected) and
> [Stages](flipull-gb.md#stages) have the details and what each byte really does.

| | |
|---|---|
| File | `Flipull (USA).gb` |
| Size | 32,768 bytes (32 KiB) |
| MD5 | `4fcc13db8144687e6b28200387aed25c` |
| Title (`$0134`) | `FLIPULL` |
| Cartridge type (`$0147`) | `$00` — **ROM ONLY, no mapper** |
| ROM size (`$0148`) | `$00` — 32 KiB, 2 banks |
| Cartridge RAM (`$0149`) | `$00` — none |
| Destination (`$014A`) | `$01` — non-Japan |
| Header checksum | `$0E` — valid |
| Global checksum | `$8AB0` — valid |
| Entry point | `$0100`: `NOP` / `JP $0150` |

No mapper means the **entire ROM is flat at `$0000–$7FFF`** with no bank
switching, the simplest possible cartridge. No cartridge RAM, so all state
lives in WRAM `$C000–$DFFF` and HRAM `$FF80–$FFFE`. Flipull leans unusually
heavily on HRAM: nearly every counter is there rather than in WRAM.

---

## 1. Quick reference

| What | Where |
|---|---|
| Block field | `$C840 + 32*row + col`, 14 rows |
| Blocks remaining | `$FFC9` (ones) + `$FFCA` (tens), decimal digits |
| Timer | `$FFCE` min, `$FFCC` sec-tens, `$FFCB` sec-ones |
| Clear target | `$FFCF` |
| Held / in-flight block type | `$FFD4` |

A cell holds a playable block when its value is **`$83`–`$86`**.

---

## 2. The block field (`$C840`)

```
cell_address = $C840 + 32*row + col        ; row stride $20
```

14 rows, `$C840` through `$C9E0`. Row starts:

| row | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| addr | `C840` | `C860` | `C880` | `C8A0` | `C8C0` | `C8E0` | `C900` |

| row | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|
| addr | `C920` | `C940` | `C960` | `C980` | `C9A0` | `C9C0` | `C9E0` |

Row 0 is the ceiling and row 13 the floor; both read `$80` across 16 columns.
Column 0 is the left wall (`$80`) on every row. So the usable field is columns
1–15; Stage 1 occupies columns 1–5.

Although the row stride is 32 bytes, only the first 16 of each row carry
meaning: the upper half of every row was `$00` throughout observation.

### Cell values

| value | meaning |
|---|---|
| `$00` | outside the field / unused |
| `$80` | border — ceiling, floor, left wall |
| `$83`–`$86` | **playable block**, four distinct types |
| `$87` | fixed staircase wall (the diagonal in the upper left of Stage 1) |

`$87` is structural, not playable: it forms the stepped diagonal and is excluded
from the block count. Count only `$83`–`$86`.

### Verified against the HUD

Stage 1 opens with five rows of five blocks in rows 8–12, columns 1–5:

```
row08 C940   80 84 85 85 86 85
row09 C960   80 84 85 83 83 84
row10 C980   80 86 85 84 85 86
row11 C9A0   80 84 84 83 84 83
row12 C9C0   80 83 86 86 86 83
```

That is exactly 25 cells in `$83`–`$86`, matching the on-screen `BLOCK 25`.
After one throw the count read 24 and the grid held 24.

### Column collapse

When a block is destroyed its column falls. Observed on column 5:

```
before (rows 8-12):  85 84 86 83 83
after:               00 85 84 86 83
```

The column shifted down one cell and the bottom block was removed. Gravity is
per-column and immediate.

---

## 3. Counters (HRAM)

Flipull stores counters as **separate decimal digits, ones-first**, not as
binary or packed BCD. Searching for `25` or `$19` finds nothing; the value lives
as `05` and `02` in adjacent bytes.

| Address | Field | Confidence |
|---|---|---|
| `$FFC9` | **Blocks remaining — ones digit** | **Verified** (`05`→`04` as HUD went 25→24) |
| `$FFCA` | Blocks remaining — tens digit | Good (`02`, consistent) |
| `$FFC0` | Stage's *initial* block count — ones digit | Moderate |
| `$FFC1` | Stage's initial block count — tens digit | Moderate |
| `$FFCB` | **Timer — seconds, ones digit** | **Verified** (`09`→`02` as HUD went 2:59→2:52) |
| `$FFCC` | Timer — seconds, tens digit | Good (`05`) |
| `$FFCE` | Timer — minutes | Good (`02`) |
| `$FFCD` | Sub-second tick counter | Moderate (`$28`, free-running) |
| `$FFCF` | Clear target (`CLEAR 09`) | Moderate — never changed |
| `$FFC6` | Stage number | **Unverified** (`01` in Stage 1) — **resolved:** the ones digit; `$FFC7` is the tens |

`$FFC0`/`$FFC1` held `05`,`02` at stage start and **stayed** at `05`,`02` after
the count dropped to 24, which is why they read as the stage's starting total
rather than the live count. Read `$FFC9`/`$FFCA` for the live value:

```python
blocks_left = hram[0xFFCA] * 10 + hram[0xFFC9]
seconds     = hram[0xFFCE] * 60 + hram[0xFFCC] * 10 + hram[0xFFCB]
```

---

## 4. The thrown block

| Address | Field | Confidence |
|---|---|---|
| `$FFD4` | Held / in-flight block type — read `$83`, a valid block value | Moderate — **superseded:** it is the *previously* held block |
| `$FFD2`, `$FFD3` | Throw state flags — both `00`→`01` on release | Moderate — **superseded:** a count of completed throws |
| `$FFDF` | In-flight block X position — decreased steadily (`64 5E 57 4E 45 3F 39 32 29 20 1A 14 0C 03`) as the block travelled left, then reset | Moderate — **superseded:** a free-running counter, falling by 17 a frame whether or not anything is in flight |
| `$FFDE` | In-flight Y position — drifted `56`→`58` | Low |
| `$FFAF` | Free-running counter / RNG — advanced ~50/frame regardless of input | Moderate |

---

## 5. Other regions

| Range | Contents |
|---|---|
| `$CA00–$CA23` | 36 bytes of block-type values (`$83`–`$86`). Contents shift during play. Most likely the queue of upcoming blocks, **not confirmed**. |
| `$CA30–$CA35` | Small counter block; `$CA30` changed `$23`→`$00` across the run, `$CA32` held `09` matching `CLEAR`. Purpose unresolved. |
| `$C000–$C00A` | Small set of frequently-written bytes; `$C002` tracked vertical input (`89`/`8F`) — likely player sprite state |
| `$CFC0–$CFFF` | Mixed working area, changes continuously |

---

## 6. Reading state

```python
GRID, STRIDE, ROWS, COLS = 0xC840, 0x20, 14, 16

def cell(row, col):
    return GRID + row * STRIDE + col

blocks = [(r, c, wram[cell(r, c)])
          for r in range(ROWS) for c in range(COLS)
          if 0x83 <= wram[cell(r, c)] <= 0x86]

remaining = hram[0xFFCA] * 10 + hram[0xFFC9]
assert len(blocks) == remaining        # held true in observation

walls = [(r, c) for r in range(ROWS) for c in range(COLS)
         if wram[cell(r, c)] == 0x87]  # fixed staircase, not clearable
```

---

## 7. Confidence and open questions

### Method

Booted in a purpose-written SM83 emulator, driven to Stage 1, then WRAM and
HRAM recorded once per frame across a scripted sequence of throws and vertical
moves. Findings were anchored against known on-screen values (`BLOCK 25`,
`CLEAR 09`, `TIME 2:59`, `STAGE 1`), which makes the counter identifications
unusually well-grounded. Screen renders confirmed emulator accuracy.

Static tracing reached 6,675 instructions (far better than Super Mario Land's
2,011, thanks to the absence of banking), but **the field's address calculator
was not located in code.** The `$C840`/stride-32 geometry is derived from the
memory dump's structure, which is unambiguous (14 evenly spaced rows with
consistent wall patterns at both ends), but it is empirical, not code-confirmed
the way Puzznic's `$29CE` calculator was.

### Verified

- Field base `$C840`, stride `$20`, 14 rows, column 0 as left wall
- Cell encoding: `$80` border, `$87` fixed staircase, `$83`–`$86` playable
- Occupied count matched `BLOCK` on the HUD both before (25) and after (24)
- Per-column collapse on destruction
- `$FFC9` blocks-remaining ones digit; `$FFCB` timer seconds ones digit

### Unverified

- Only Stage 1 was played, and only one block was destroyed. Multi-block
  chains, which are the core of Flipull's scoring, were never exercised.
- The **score** was never located. It advanced 0 → 100, but no candidate byte
  was isolated.
- `$FFC6` as stage number: matched `01` but never seen changing. **Since settled:** it is
  the ones digit of a two-digit stage number whose tens live in `$FFC7`, and the loader at
  `0:2D55` indexes a 32-entry table at `$3A0E` with `10*tens + ones - 1`.
- `$CA00–$CA23` as the upcoming-block queue is a guess from its contents.
- Field width beyond column 5 is inferred from the ceiling/floor rows showing
  `$80` across 16 columns; no stage was observed using the full width.

### Verifying in an emulator

In SameBoy, `x $FFC9` should decrement each time a block is destroyed, and
`x $FFCB` should tick down once per second. Dump `$C940` and watch a column
collapse as you throw. Note that SameBoy parses `x $C840/16` as arithmetic
rather than address-plus-length; use `x $C840` and check `help examine` for
your build's syntax.