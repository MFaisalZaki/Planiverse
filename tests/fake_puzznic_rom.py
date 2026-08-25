"""A synthetic Game Boy ROM that reproduces Puzznic's documented memory layout.

Puzznic is copyrighted and cannot ship with the repo, which would leave everything the
environment does against a real cartridge — booting, hooking the stage loader, decoding
the grid, waiting for a move to settle, spotting a cleared stage — untested. This module
builds a small homebrew cartridge that puts the *same facts at the same addresses*:

    $D003  stage index, read by the stage loader at $0430
    $D012  cursor column          $D013  cursor row
    $D018  blocks loaded          $D019  blocks remaining
    $DD00  6-byte block records    $DF00  12x10 grid, 2 bytes per cell

It is emphatically **not** a Puzznic clone. It has no gravity, no timer, no score and no
cascades: a push that leaves two same-typed blocks orthogonally adjacent clears both
after a three-frame `$01` transient, and that is the whole rule set. What it exercises is
the *interface* between the environment and a Game Boy, which is the part that can
otherwise only be checked by hand with a cartridge nobody can commit.

    from fake_puzznic_rom import build_rom, write_rom
    path = write_rom(tmp_path / "fake-puzznic.gb")

Controls match the environment's action set: the d-pad moves the cursor, and A plus
left/right pushes the block under the cursor one cell sideways.
"""
import atexit
import os
import shutil
import tempfile

from sm83 import Assembler

# --- the addresses the environment reads, straight out of the memory map -----------
STAGE_INDEX = 0xD003
CURSOR_COL = 0xD012
CURSOR_ROW = 0xD013
TOTAL_BLOCKS = 0xD018
BLOCKS_REMAINING = 0xD019
RECORDS = 0xDD00
GRID = 0xDF00
STAGE_LOADER = 0x0430          # the address the environment hooks to force a stage

# --- scratch this ROM uses for itself; deliberately clear of everything above ------
PAD = 0xC010                   # buttons held this frame
PREV_PAD = 0xC011              # buttons held last frame
NEW_PAD = 0xC012               # buttons pressed *this* frame
TMP = 0xC013
SLOT = 0xC014
DIR = 0xC015
ROW_COUNT = 0xC016
COL_COUNT = 0xC017
CLEARED = 0xC018
HOLD = 0xC019                  # frames the current direction has been held
TRIG = 0xC01A                  # NEW_PAD, plus auto-repeat re-fires
MARKS = 0xC100                 # 240 bytes, indexed exactly like the grid

# Holding a direction moves the cursor once, then again after REPEAT_DELAY frames, then
# every REPEAT_RATE frames — the behaviour that puts an upper bound on how long an action
# may hold a button before it moves the cursor twice.
REPEAT_DELAY = 16
REPEAT_RATE = 6

ROWS, COLS = 12, 10
CELL_BYTES = 2

# Cell type codes (memory map §4).
OUTSIDE, WALL, EMPTY, LEDGE = 0x03, 0x06, 0x00, 0x02

# The 48-byte logo a real cartridge carries at $0104.
NINTENDO_LOGO = bytes.fromhex(
    "CEED6666CC0D000B03730083000C000D0008111F8889000E"
    "DCCC6EE6DDDDD999BBBB67636E0EECCCDDDC999FBBB9333E"
)

SYMBOLS = {
    "STAGE_INDEX": STAGE_INDEX, "CURSOR_COL": CURSOR_COL, "CURSOR_ROW": CURSOR_ROW,
    "TOTAL_BLOCKS": TOTAL_BLOCKS, "BLOCKS_REMAINING": BLOCKS_REMAINING,
    "RECORDS": RECORDS, "GRID": GRID, "MARKS": MARKS,
    "PAD": PAD, "PREV_PAD": PREV_PAD, "NEW_PAD": NEW_PAD, "TMP": TMP, "SLOT": SLOT,
    "DIR": DIR, "ROW_COUNT": ROW_COUNT, "COL_COUNT": COL_COUNT, "CLEARED": CLEARED,
    "HOLD": HOLD, "TRIG": TRIG,
    "REPEAT_DELAY": REPEAT_DELAY, "REPEAT_RATE": REPEAT_RATE,
    "REPEAT_RELOAD": REPEAT_DELAY - REPEAT_RATE,
    "LCDC": 0x40, "LY": 0x44, "JOYP": 0x00,
}


# --------------------------------------------------------------------------- stages

def _blank():
    return [[OUTSIDE] * COLS for _ in range(ROWS)]


def _box(grid, top, left, bottom, right):
    """A walled room with an empty floor plan inside it."""
    for row in range(top, bottom + 1):
        for col in range(left, right + 1):
            edge = row in (top, bottom) or col in (left, right)
            grid[row][col] = WALL if edge else EMPTY
    return grid


def stage_layouts():
    """The four stages this cartridge can load, selected by `$D003 & 3`.

    Stage 0 is the interesting one: two blocks of the same type two cells apart, so
    `a+right` twice matches them and clears the stage.
    """
    # 0 — solvable in two pushes.
    zero = _box(_blank(), 3, 2, 6, 7)
    zero[5][3] = 0x08
    zero[5][6] = 0x08

    # 1 — a different shape and two pairs, so it cannot be confused with stage 0.
    one = _box(_blank(), 2, 1, 8, 8)
    one[7][2] = 0x08
    one[7][6] = 0x08
    one[4][3] = 0x09
    one[4][5] = 0x09
    one[6][4] = LEDGE

    # 2 — a single block: no pair, so the stage is a dead end the moment it loads.
    two = _box(_blank(), 4, 3, 7, 6)
    two[6][4] = 0x0A

    # 3 — three types, one of them a triple, plus ledges.
    three = _box(_blank(), 1, 1, 10, 8)
    three[9][2] = 0x08
    three[9][5] = 0x08
    three[5][3] = 0x0B
    three[5][6] = 0x0B
    three[3][2] = 0x0C
    three[3][4] = 0x0C
    three[3][6] = 0x0C
    three[6][4] = LEDGE
    three[6][5] = LEDGE
    return [zero, one, two, three]


def _flatten(grid):
    return [cell for row in grid for cell in row]


# ------------------------------------------------------------------------- the code

PROGRAM = """
; ---------------------------------------------------------------- boot
boot:
    di
    ld sp, $CFFF
    ld hl, $C000                ; the real game clears WRAM at $0150; so do we
    ld bc, $2000
.clear:
    xor a
    ld (hl+), a
    dec bc
    ld a, b
    or c
    jr nz, .clear
    ld a, $91                   ; LCD on, so PyBoy keeps completing frames
    ldh (LCDC), a

; The title screen. It rewrites the stage index every frame and then loads a stage the
; instant START is pressed, which is what makes pinning $D003 from outside unreliable
; and hooking the loader the thing that actually works.
menu:
    xor a
    ld (STAGE_INDEX), a
    call read_pad
    ld a, (NEW_PAD)
    bit 7, a                    ; START
    jr nz, .start
    call wait_frame
    jr menu
.start:
    call load_stage
    jp main

; ---------------------------------------------------------------- main loop
main:
    call wait_frame
    call read_pad
    ld a, (TRIG)
    and a
    jr z, main
    ld a, (PAD)
    bit 4, a                    ; A held turns a direction into a push
    jp z, move_cursor
    ld a, (TRIG)
    bit 1, a
    jp nz, push_left
    bit 0, a
    jp nz, push_right
    jr main

move_cursor:
    ld a, (TRIG)
    bit 0, a
    jr nz, .right
    bit 1, a
    jr nz, .left
    bit 2, a
    jr nz, .up
    bit 3, a
    jr nz, .down
    jp main
.right:
    ld a, (CURSOR_COL)
    cp 9
    jp z, main
    inc a
    ld (CURSOR_COL), a
    jp main
.left:
    ld a, (CURSOR_COL)
    and a
    jp z, main
    dec a
    ld (CURSOR_COL), a
    jp main
.up:
    ld a, (CURSOR_ROW)
    and a
    jp z, main
    dec a
    ld (CURSOR_ROW), a
    jp main
.down:
    ld a, (CURSOR_ROW)
    cp 11
    jp z, main
    inc a
    ld (CURSOR_ROW), a
    jp main

; ---------------------------------------------------------------- pushing a block
push_left:
    ld a, (CURSOR_COL)
    and a
    jp z, main
    xor a
    ld (DIR), a
    jr push
push_right:
    ld a, (CURSOR_COL)
    cp 9
    jp z, main
    ld a, 1
    ld (DIR), a

push:
    call cursor_cell            ; hl = the cell under the cursor
    ld a, (hl)
    cp $08
    jp c, main                  ; nothing to push
    ld (TMP), a                 ; block type
    inc hl
    ld a, (hl)
    ld (SLOT), a                ; record slot
    dec hl
    ld a, (DIR)
    and a
    jr z, .left
    inc hl
    inc hl
    jr .target
.left:
    dec hl
    dec hl
.target:
    ld a, (hl)
    and a
    jp nz, main                 ; the destination is not empty
    ld a, (TMP)
    ld (hl+), a
    ld a, (SLOT)
    ld (hl), a
    dec hl
    ld a, (DIR)                 ; back to the cell we came from, and empty it
    and a
    jr z, .back_right
    dec hl
    dec hl
    jr .empty
.back_right:
    inc hl
    inc hl
.empty:
    xor a
    ld (hl+), a
    ld (hl), a
    ld a, (DIR)                 ; the cursor rides along with the block
    and a
    ld a, (CURSOR_COL)
    jr z, .step_left
    inc a
    jr .store
.step_left:
    dec a
.store:
    ld (CURSOR_COL), a
    call record_addr            ; keep the record's column field in step
    inc hl
    inc hl
    inc hl
    ld a, (CURSOR_COL)
    ld (hl), a
    call clear_matches
    jp main

; ---------------------------------------------------------------- matching
; Two passes over the grid: mark every block that touches a same-typed neighbour, then
; clear what was marked. Marking first keeps the scan from seeing its own edits.
clear_matches:
    ld hl, MARKS                ; 240 bytes, same indexing as the grid
    ld bc, 240
.wipe:
    xor a
    ld (hl+), a
    dec bc
    ld a, b
    or c
    jr nz, .wipe

    ld bc, 0
    xor a
    ld (ROW_COUNT), a
.row:
    xor a
    ld (COL_COUNT), a
.col:
    ld hl, GRID
    add hl, bc
    ld a, (hl)
    cp $08
    jr c, .next
    ld (TMP), a
    ld a, (COL_COUNT)           ; neighbour to the right
    cp 9
    jr z, .down
    inc hl
    inc hl
    ld a, (hl)
    ld e, a
    ld a, (TMP)
    cp e
    jr nz, .down
    ld hl, MARKS
    add hl, bc
    ld a, 1
    ld (hl+), a
    inc hl
    ld (hl), a
.down:
    ld a, (ROW_COUNT)           ; neighbour below
    cp 11
    jr z, .next
    ld hl, GRID
    add hl, bc
    push bc
    ld bc, 20
    add hl, bc
    pop bc
    ld a, (hl)
    ld e, a
    ld a, (TMP)
    cp e
    jr nz, .next
    ld hl, MARKS
    add hl, bc
    ld a, 1
    ld (hl), a
    push bc
    ld bc, 20
    add hl, bc
    pop bc
    ld a, 1
    ld (hl), a
.next:
    inc bc
    inc bc
    ld a, (COL_COUNT)
    inc a
    ld (COL_COUNT), a
    cp 10
    jr nz, .col
    ld a, (ROW_COUNT)
    inc a
    ld (ROW_COUNT), a
    cp 12
    jr nz, .row

; Marked cells go to $01 ("clearing") for a few frames before they go away, so a reader
; that snapshots mid-animation can tell it has to wait.
    ld bc, 0
    xor a
    ld (CLEARED), a
.mark:
    ld hl, MARKS
    add hl, bc
    ld a, (hl)
    and a
    jr z, .mark_next
    ld hl, GRID
    add hl, bc
    ld a, $01
    ld (hl), a
    ld a, (CLEARED)
    inc a
    ld (CLEARED), a
.mark_next:
    inc bc
    inc bc
    ld a, c
    cp 240
    jr nz, .mark
    ld a, (CLEARED)
    and a
    ret z

    ld b, 3
    call wait_frames

    ld bc, 0
.sweep:
    ld hl, MARKS
    add hl, bc
    ld a, (hl)
    and a
    jr z, .sweep_next
    ld hl, GRID
    add hl, bc
    xor a
    ld (hl+), a
    ld a, (hl)
    ld (SLOT), a
    xor a
    ld (hl), a
    push bc
    call record_addr            ; records are zeroed in place, slots are not compacted
    xor a
    ld (hl), a
    pop bc
    ld a, (BLOCKS_REMAINING)
    dec a
    ld (BLOCKS_REMAINING), a
.sweep_next:
    inc bc
    inc bc
    ld a, c
    cp 240
    jr nz, .sweep
    ret

; ---------------------------------------------------------------- helpers
; hl = GRID + 20*row + 2*col, for row in b and col in c.
cell_addr:
    ld h, 0
    ld l, b
    add hl, hl
    add hl, hl
    ld d, h
    ld e, l
    add hl, hl
    add hl, hl
    add hl, de                  ; row*20
    ld d, 0
    ld e, c
    add hl, de
    add hl, de                  ; + col*2
    ld de, GRID
    add hl, de
    ret

cursor_cell:
    ld a, (CURSOR_ROW)
    ld b, a
    ld a, (CURSOR_COL)
    ld c, a
    jp cell_addr

; hl = RECORDS + 6*(SLOT)
record_addr:
    ld a, (SLOT)
    add a, a
    ld e, a
    add a, a
    add a, e
    ld e, a
    ld d, 0
    ld hl, RECORDS
    add hl, de
    ret

wait_frames:
    call wait_frame
    dec b
    jr nz, wait_frames
    ret

wait_frame:
    ldh a, (LY)
    cp $90
    jr nz, wait_frame
.leave:
    ldh a, (LY)
    cp $90
    jr z, .leave
    ret

; PAD: bit 0-3 right/left/up/down, bit 4-7 A/B/select/start. NEW_PAD is the rising edge.
read_pad:
    ld a, (PAD)
    ld (PREV_PAD), a
    ld a, $20
    ldh (JOYP), a
    ldh a, (JOYP)
    ldh a, (JOYP)
    cpl
    and $0F
    ld b, a
    ld a, $10
    ldh (JOYP), a
    ldh a, (JOYP)
    ldh a, (JOYP)
    ldh a, (JOYP)
    ldh a, (JOYP)
    cpl
    and $0F
    swap a
    or b
    ld (PAD), a
    ld a, $30
    ldh (JOYP), a
    ld a, (PREV_PAD)
    cpl
    ld b, a
    ld a, (PAD)
    and b
    ld (NEW_PAD), a

; Auto-repeat. A direction held unchanged re-fires REPEAT_DELAY frames in, then every
; REPEAT_RATE frames after that, which is what stops an action from simply holding a
; button for as long as it likes.
    ld (TRIG), a                ; the rising edge always triggers
    ld a, (PAD)
    and $0F                     ; directions only; A and START do not repeat
    jr z, .idle
    ld b, a
    ld a, (PREV_PAD)
    and $0F
    cp b
    jr nz, .idle                ; a different set is held now, so start counting again
    ld a, (HOLD)
    inc a
    ld (HOLD), a
    cp REPEAT_DELAY
    ret nz
    ld a, REPEAT_RELOAD
    ld (HOLD), a
    ld a, (TRIG)
    or b
    ld (TRIG), a
    ret
.idle:
    xor a
    ld (HOLD), a
    ret
"""

# The stage loader sits at $0430, exactly where the real cartridge keeps it, so the
# environment's hook address is the thing under test rather than a stand-in.
LOADER = """
load_stage:
    ld hl, RECORDS              ; the real loader clears the whole record page first
    ld b, 0
.wipe:
    xor a
    ld (hl+), a
    dec b
    jr nz, .wipe

    ld a, (STAGE_INDEX)
    and 3
    add a, a
    ld l, a
    ld h, 0
    ld de, stage_table
    add hl, de
    ld a, (hl+)
    ld h, (hl)
    ld l, a
    ld d, h
    ld e, l                     ; de = this stage's 120 cell codes

    xor a
    ld (TOTAL_BLOCKS), a
    ld (ROW_COUNT), a
    ld (CURSOR_COL), a
    ld (CURSOR_ROW), a
    ld hl, GRID
.row:
    xor a
    ld (COL_COUNT), a
.col:
    ld a, (de)
    inc de
    ld (TMP), a
    ld (hl+), a
    cp $08
    jr c, .terrain

    ld a, (TOTAL_BLOCKS)        ; the cell's second byte back-references the record
    ld (hl), a
    ld (SLOT), a
    push hl
    push de
    call record_addr
    ld a, (TMP)
    ld (hl+), a                 ; +0 type
    xor a
    ld (hl+), a                 ; +1 state
    ld a, (ROW_COUNT)
    ld (hl+), a                 ; +2 row
    ld a, (COL_COUNT)
    ld (hl+), a                 ; +3 column
    ld a, $62
    ld (hl+), a                 ; +4 the constant the renderer wants
    xor a
    ld (hl), a                  ; +5 render offset
    pop de
    pop hl
    ld a, (TOTAL_BLOCKS)
    inc a
    ld (TOTAL_BLOCKS), a
    jr .next
.terrain:
    xor a
    ld (hl), a
.next:
    inc hl
    ld a, (COL_COUNT)
    inc a
    ld (COL_COUNT), a
    cp 10
    jr nz, .col
    ld a, (ROW_COUNT)
    inc a
    ld (ROW_COUNT), a
    cp 12
    jr nz, .row

    ld a, (TOTAL_BLOCKS)        ; remaining starts equal to the total
    ld (BLOCKS_REMAINING), a
    ld a, 5                     ; park the cursor somewhere inside the playfield
    ld (CURSOR_COL), a
    ld (CURSOR_ROW), a
    ret
"""


def build_rom(title=b"PUZZNICFAKE"):
    """Assemble the cartridge and return its 32 KiB image."""
    asm = Assembler(SYMBOLS)

    asm.org(0x0100)
    asm.asm("nop\njp boot")
    asm.org(0x0104)
    asm.db(NINTENDO_LOGO)
    asm.org(0x0134)
    asm.db(title[:16].ljust(16, b"\0"))
    asm.org(0x0147)
    asm.db([0x00, 0x00, 0x00])            # ROM only, 32 KiB, no cartridge RAM
    asm.org(0x014A)
    asm.db([0x00, 0x00, 0x00])            # Japan, licensee, version

    asm.org(STAGE_LOADER)
    asm.asm(LOADER)
    if asm.pc > 0x0700:
        raise AssertionError("the loader ran into the main program")

    asm.org(0x0700)
    asm.asm(PROGRAM)

    layouts = stage_layouts()
    asm.org(0x2000)
    asm.label("stage_table")
    for index in range(len(layouts)):
        asm.asm(f"dw stage_{index}")
    for index, layout in enumerate(layouts):
        asm.label(f"stage_{index}")
        asm.db(_flatten(layout))

    rom = bytearray(asm.link())
    _stamp_checksums(rom)
    return bytes(rom)


def _stamp_checksums(rom):
    header = 0
    for byte in rom[0x0134:0x014D]:
        header = (header - byte - 1) & 0xFF
    rom[0x014D] = header
    rom[0x014E] = rom[0x014F] = 0
    total = sum(rom) & 0xFFFF
    rom[0x014E] = total >> 8
    rom[0x014F] = total & 0xFF


def write_rom(path):
    """Build the cartridge and write it to `path`, returning the path as a string."""
    path = str(path)
    with open(path, "wb") as handle:
        handle.write(build_rom())
    return path


_CARTRIDGE = None


def synthetic_rom():
    """The cartridge, built once per process into a temp file that is cleaned up at exit."""
    global _CARTRIDGE
    if _CARTRIDGE is None:
        directory = tempfile.mkdtemp(prefix="planiverse-fake-puzznic-")
        atexit.register(shutil.rmtree, directory, ignore_errors=True)
        _CARTRIDGE = write_rom(os.path.join(directory, "fake-puzznic.gb"))
    return _CARTRIDGE


if __name__ == "__main__":
    import sys
    print(write_rom(sys.argv[1] if len(sys.argv) > 1 else "fake-puzznic.gb"))
