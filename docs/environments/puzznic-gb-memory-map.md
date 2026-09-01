# Puzznic (J) Game Boy memory map

Reference for reading live game state out of `Puzznic (J).gb`: block positions, block types, and
how many blocks remain. This is the source for every address the [`PuzznicGBEnv`](puzznic-gb.md)
environment reads.

| | |
|---|---|
| File | `Puzznic (J).gb` |
| Size | 65,536 bytes (64 KiB) |
| MD5 | `9a777d82cd7a8913ba1aed2cc854fa50` |
| Title (`$0134`) | `PUZZNIC` |
| Cartridge type (`$0147`) | `$01`, MBC1 |
| ROM size (`$0148`) | `$01`, 64 KiB, 4 banks |
| Cartridge RAM (`$0149`) | `$00`, none |
| Destination (`$014A`) | `$00`, Japan |
| CGB / SGB flags | `$00` / `$00`, DMG only |
| Header checksum | `$F2`, valid |
| Global checksum | `$03B7`, valid |
| Entry point | `$0100`: `NOP` / `JP $0150` |

The cartridge has no external RAM, so all game state lives in the 8 KiB of work RAM at
`$C000–$DFFF` plus HRAM. Nothing of interest is at `$A000–$BFFF`.

---

## 1. Quick reference

The three addresses that matter most are:

| What | Where |
|---|---|
| Blocks remaining | `$D019` (one byte) |
| Playfield grid | `$DF00 + 20*row + 2*col`, 12 rows × 10 cols |
| Cursor | `$D012` = column, `$D013` = row |

A cell holds a block when its value is `$08`–`$0F`, and the block's type is `value - 7`.

---

## 2. WRAM layout

| Range | Size | Contents |
|---|---|---|
| `$C000–$C09F` | 160 | OAM DMA source buffer (40 sprites × 4 bytes). Copied to OAM by the HRAM routine at `$FF84`. |
| `$C100–$C125` | 38 | Init-filled buffer (written by `$2F6A`) |
| `$C180–$C1EF` | 112 | Init-filled buffer (written by `$2F9E`) |
| `$C700–$C7FF` | 256 | Task 1 stack, seeded at `$C7FD/$C7FE` |
| `$C800–$C8FF` | 256 | Task 2 stack, seeded at `$C8FD/$C8FE` |
| `$C900–$C9FF` | 256 | Task 3 stack, seeded at `$C9FD/$C9FE` |
| `$CAFF` | 1 | Main stack top; grows downward |
| `$D000–$D030` | ~48 | Core game state variables (see §3) |
| `$D200–$D221` | ~34 | Secondary state |
| `$D700–$D758` | 90 | 6-entry array of 15-byte structs (fields at +0, +3, +9). Purpose not identified; likely audio channels. |
| `$D740` | 1 | Sync/vblank spin flag, polled about 2000 times a frame |
| `$D800–$D967` | 360 | Tilemap shadow, 20×18, for the status panel (`ROUND` / `SCORE`) |
| `$DC00–$DC9F` | 160 | OAM staging buffer |
| `$DCA0–$DCC7` | 40 | Buffer written by `1:5BDB` |
| `$DCD0–$DCD7` | 8 | Scratch copy of the block record currently being processed: the 6-byte record, then a 2-byte pointer back to its live entry |
| `$DCE0–$DCE3` | 4 | Stage-loader scratch (see §7) |
| `$DD00–$DDFF` | 256 | **Block record array page** (see §5). Cleared wholesale by `1:59C9`. |
| `$DF00–$DFEF` | 240 | **Playfield grid** (see §4) |
| `$DFF0–$DFF7` | 8 | State written during load (`$0430`, `$0513`) |

At boot, `$0150` clears `$C000–$EFFF`, sets `SP = $CAFF`, selects ROM bank 1 via `LD ($2100),A`,
and seeds three cooperative task stacks with entry points `$4D21`, `$5002` and `$2319` (SP low
bytes held in HRAM `$FF8D`–`$FF8F`).

---

## 3. Game state variables

| Address | Meaning |
|---|---|
| `$D003` | Stage index. Used at `$045A` (`SLA A`, then added to a 16-bit pointer table) to select the stage layout |
| `$D012` | Cursor column |
| `$D013` | Cursor row |
| `$D018` | Total blocks loaded this stage. Set to 0 at `$0452`, incremented once per block at `$050D`, never decremented. Also the iteration bound over the record array at `1:5018` |
| `$D019` | Blocks remaining. A copy of `$D018` at load, decremented by `1:541F` each time a block is removed |
| `$D01F` | Offset applied when randomising block types at load |
| `$D221` | PRNG seed for block-type randomisation |

`$D018` and `$D019` read the same value at stage start, and clearing a pair drops `$D019` by 2
while leaving `$D018` unchanged.

---

## 4. The playfield grid (`$DF00`)

The geometry is fixed and identical for every stage: 12 rows by 10 columns, 2 bytes per cell,
occupying `$DF00–$DFEF` (240 bytes).

```
cell_address = $DF00 + 20*row + 2*col
```

Row-start addresses (stride 20 = `$14`, so they do not align to a 16-byte hex view):

| row | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| addr | `DF00` | `DF14` | `DF28` | `DF3C` | `DF50` | `DF64` | `DF78` | `DF8C` | `DFA0` | `DFB4` | `DFC8` | `DFDC` |

Those offsets come from the row-offset table at ROM `$29E8`, which reads `00 14 28 3C 50 64 78 8C
A0 B4 C8 DC`: twelve entries, 0 to 220 in steps of 20.

### Cell format

| byte | meaning |
|---|---|
| +0 | Cell type code |
| +1 | Record slot index into `$DD00`; meaningful only when +0 ≥ `$08` |

### Cell type codes

| value | meaning |
|---|---|
| `$00` | Empty; the only value that permits movement |
| `$01` | Transient: block clearing or in motion. Written by `1:5417`, zeroed by `1:54B7`. |
| `$02` | Solid ledge or platform; blocks rest on it |
| `$03` | Outside the playfield |
| `$06` | Wall |
| `$08`–`$0F` | **Block.** Type = `value - 7` (so types 1–8). |

The block test is `value >= $08`, taken from the loader's per-cell handler at `$04B8`, which opens
with `CP $08` / `RET C`.

Movement treats only `$00` as free. The check at `1:506E` does `AND A` / `JP NZ` on the cell
value, so `$02` and `$03` obstruct exactly as `$06` does.

### Level shape

There are no per-level width or height variables, because the dimensions are immediates in the
loader (see §7). Every stage fills all 120 cells, and a small stage is the same array with more
`$03` around the edges, so derive the effective bounding box instead:

```python
occupied = [(r, c) for r in range(12) for c in range(10)
            if wram[0xDF00 + 20*r + 2*c] != 0x03]
row_range = (min(r for r, _ in occupied), max(r for r, _ in occupied))
col_range = (min(c for _, c in occupied), max(c for _, c in occupied))
```

---

## 5. Block record array (`$DD00`)

The array holds 6-byte records, at `record_n = $DD00 + n*6`.

| offset | meaning |
|---|---|
| +0 | Block type. `$00` marks an empty/dead slot. |
| +1 | State / flags |
| +2 | Row |
| +3 | Column |
| +4 | Constant `$62` (rendering) |
| +5 | Render offset = `(40*row + 5*col + 103) & $FF` |

Fields +4 and +5 feed the drawing path (routine `$2985`, tables `$29B6`/`$29C2`, ROM base
`$6194`). They should not be confused with the logic grid, so ignore them when reading state.

### Slots are not compacted

When blocks are cleared their records are zeroed in place, and surviving blocks keep their
original slot numbers, which is what keeps the grid's +1 back-references valid. After clearing
slots 0 and 1:

```
DD00  00 00 00 00 00 00   <- cleared
DD06  00 00 00 00 00 00   <- cleared
DD0C  03 00 08 06 62 C5   <- survivors keep slots 2..5
DD12  04 00 08 03 62 B6
DD18  03 00 09 05 62 E8
DD1E  04 00 09 04 62 E3
```

Do not walk the list until the first `$00` type byte: after any clear that terminates immediately
and reports zero blocks. Iterate `$D018` slots and skip records whose type is `$00`, or scan the
grid.

The grid and the record array cross-reference each other, so the board can be read from either
end.

---

## 6. Reading state

```python
def cell(row, col):
    return 0xDF00 + 20 * row + 2 * col

# how many are left, one byte
remaining = wram[0xD019]

# where they are, with type and slot
blocks = [
    (r, c, wram[cell(r, c)] - 7, wram[cell(r, c) + 1])
    for r in range(12) for c in range(10)
    if wram[cell(r, c)] >= 0x08
]

# the same thing from the record side
total = wram[0xD018]
live = []
for slot in range(total):
    rec = wram[0xDD00 + slot * 6 : 0xDD00 + slot * 6 + 6]
    if rec[0] != 0x00:
        live.append({'slot': slot, 'type': rec[0],
                     'row': rec[2], 'col': rec[3]})

cursor = (wram[0xD013], wram[0xD012])   # (row, col)
```

`len(blocks)` and `len(live)` should both equal `remaining`. If they disagree, one of the three
has drifted; `$D019` and the grid scan are the authoritative pair.

### Block types are randomised per playthrough

The stage data in ROM records only *that* a block occupies a cell. The actual type is drawn at
load time by the PRNG at `$04C9` (table at bank 1 `$7975`, seed `$D221`, offset from `$D01F`),
finishing with `AND $07` / `ADD $08`.

Note that expected type values per stage cannot be hard-coded. The invariant that does hold is
that two blocks which look identical on screen carry identical type bytes.

---

## 7. Stage loading

The loader is entered around `$0430`. It switches to ROM bank 2, points `DE` at `$4000`, skips two
bytes, reads a 16-bit pointer, then indexes a pointer table by `$D003` (stage index × 2) to reach
the stage's data.

The fill loop at `$0466`:

- Starts at `$DF12` (row 0, column 9) and fills **right to left**
- Unpacks **two cells per ROM byte**: high nibble first (`AND $F0` / `SWAP A`), then low nibble
  (`AND $0F`). Each nibble is the raw cell type code from §4.
- Calls `$04B8` per cell, which returns immediately for terrain (`CP $08` / `RET C`) and otherwise
  allocates a block record
- Advances to the next row with `L += 40` at `$049F`

### Where the dimensions are hard-coded

Columns are 10. `$0470` sets `DCE1 = $09`, and the inner loop decrements it twice per byte until
it wraps past zero (9→8, 7→6, 5→4, 3→2, 1→0, FF), giving five iterations of two cells.

Rows are 12. `$04A6` increments `DCE0` and compares against the immediate `$0C`:

```
LD A,(DCE0h)
INC A
LD (DCE0h),A
CP 0Ch
JP NZ,0470h
```

`DCE0` and `DCE1` are loader scratch only, and should not be confused with live geometry:
mid-stage they sit at their terminal values `$0C` and `$FF`.

---

## 8. Key ROM routines

| Address | Role |
|---|---|
| `0:0150` | Boot / init: clear WRAM, set up stacks and tasks |
| `0:0430` | Stage loader entry |
| `0:0466` | Grid fill loop (nibble unpack) |
| `0:04A6` | Row counter; the `$0C` row-count immediate |
| `0:04B8` | Per-cell handler; `CP $08` block threshold |
| `0:04C9` | Block-type PRNG |
| `0:0508` | Block record allocator; writes slot index, increments `$D018` |
| `0:2985` | Canvas/tile address calc for drawing (tables `$29B6`, `$29C2`) |
| `0:29CE` | **Grid cell address calc**; `H`=row, `L`=col, giving `HL`=cell addr and `A`=cell value |
| `0:29E8` | Row-offset table (12 entries) |
| `0:3D5A` | Seeds cursor `$D012` from a record's column field |
| `1:5018` | Iterate block list, bounded by `$D018`, stride 6 from `$DD01` |
| `1:50B0` | Write a block into a cell (`AND $0F` / `ADD $07`) |
| `1:5417` | Write cell = `$01` (clearing) |
| `1:541F` | **Decrement `$D019`** |
| `1:54B7` | Write cell = `$00` (cleared) |
| `1:59C9` | Clear the whole `$DD00` page |

The cell-address calculator, annotated:

```
29CE  CB 25     SLA L          ; L = col*2
29D0  7D        LD A,L
29D1  E0 BD     LDH ($FFBD),A
29D3  7C        LD A,H         ; A = row
29D4  21 E8 29  LD HL,$29E8    ; row-offset table
29D7  85        ADD A,L
29D8  6F        LD L,A
29DC  7E        LD A,(HL)      ; = 0,20,40,...,220
29DD  26 DF     LD H,$DF       ; grid page
29DF  6F        LD L,A
29E0  F0 BD     LDH A,($FFBD)  ; col*2
29E2  85        ADD A,L
29E3  6F        LD L,A         ; HL = $DF00 + 20*row + 2*col
29E4  7E        LD A,(HL)      ; cell value
29E5  FE 06     CP $06
29E7  C9        RET
```

---

## 9. Not resolved

| Item | Status |
|---|---|
| Stage transitions | `$D018`/`$D019` should be re-initialised on entering the next round, but that has not been watched |
| The score | Lives in the status-panel tilemap shadow at `$D800`; the digit layout there is not worked out |
| The timer | Puzznic takes a life when the per-stage time limit runs out. The timer address is not identified |
| `$D700` | A 6-entry array of 15-byte structs, unidentified; likely audio channels |
| HRAM `$FFAF` | Low nibble is a mode selector; when it equals 5 the loader suppresses blocks (`$04C5`). Purpose unknown |
| Cell value `$07` | Unreachable from the loader, since types generate `$08`–`$0F`. Whether it ever appears at runtime is unknown |
