"""A synthetic Game Boy ROM that reproduces Flipull's documented memory layout.

Flipull is copyrighted and cannot ship with the repo, which would leave everything the
environment does against a real cartridge untested. This builds a small homebrew cartridge
that puts the *same facts at the same addresses*:

    $C840  the 14-row block field, stride $20
    $FFC9/$FFCA  blocks remaining, as separate decimal digits, ones first
    $FFC0/$FFC1  the stage's starting total      $FFCF  the CLEAR target
    $FFCB/$FFCC/$FFCE  the timer                 $FFD4  the held block
    $FFD2/$FFD3  throw flags                     $C000  the player's sprite

It is **not** a Flipull clone. A throw destroys at most one block — the real game chains
through every block of its own type — and there is no scoring and no upcoming-block queue.
What it exercises is the *interface* between the environment and a Game Boy: booting,
decoding the field, reading digit-per-byte counters, waiting for a throw to finish, watching
a column collapse, and finding the player's sprite.

The stage it loads is the one the memory map records verbatim, so `decode_blocks` on it
returns the 25 blocks the map saw as `BLOCK 25`.
"""
import atexit
import os
import shutil
import tempfile

from sm83 import Assembler

FIELD = 0xC840
ROW_STRIDE = 0x20
FIELD_ROWS = 14
FIELD_COLS = 16

CELL_OUTSIDE, CELL_BORDER, CELL_STAIRCASE = 0x00, 0x80, 0x87
BLOCK_MIN, BLOCK_MAX = 0x83, 0x86

# HRAM, as low bytes for `ldh`.
H_INITIAL_ONES, H_INITIAL_TENS = 0xC0, 0xC1
H_STAGE = 0xC6
H_BLOCKS_ONES, H_BLOCKS_TENS = 0xC9, 0xCA
H_SEC_ONES, H_SEC_TENS, H_SUBSECOND, H_MINUTES = 0xCB, 0xCC, 0xCD, 0xCE
H_CLEAR_TARGET = 0xCF
H_THROW_A, H_THROW_B, H_HELD = 0xD2, 0xD3, 0xD4
H_INFLIGHT_Y, H_INFLIGHT_X = 0xDE, 0xDF

# WRAM scratch. Kept well clear of $C000-$C09F: that is the OAM DMA buffer, and a variable
# parked in it is read back as sprite data — which is exactly what happened the first time,
# and what made `probe_player_sprite` refuse to name a player sprite.
OAM = 0xC000                   # the player is sprite 0
PLAYER_ROW = 0xC700
PAD, PREV_PAD, NEW_PAD, TRIG, HOLD = 0xC701, 0xC702, 0xC703, 0xC704, 0xC705
TMP, TMP2, COL = 0xC706, 0xC707, 0xC708

REPEAT_DELAY = 16              # frames a held direction waits before it repeats
REPEAT_RATE = 6
INTRO_FRAMES = 45              # the stage is readable this long before it will listen
THROW_FRAMES = 12              # how long a thrown block spends crossing the field
TOP_ROW = 1                    # rows 1..12 are playable; 0 is the ceiling, 13 the floor

CLEAR_TARGET = 9
START_MINUTES, START_SECONDS = 2, 59

NINTENDO_LOGO = bytes.fromhex(
    "CEED6666CC0D000B03730083000C000D0008111F8889000E"
    "DCCC6EE6DDDDD999BBBB67636E0EECCCDDDC999FBBB9333E"
)

SYMBOLS = {
    "FIELD": FIELD, "OAM": OAM, "PLAYER_ROW": PLAYER_ROW,
    "PAD": PAD, "PREV_PAD": PREV_PAD, "NEW_PAD": NEW_PAD, "TRIG": TRIG, "HOLD": HOLD,
    "TMP": TMP, "TMP2": TMP2, "COL": COL,
    "H_INITIAL_ONES": H_INITIAL_ONES, "H_INITIAL_TENS": H_INITIAL_TENS, "H_STAGE": H_STAGE,
    "H_BLOCKS_ONES": H_BLOCKS_ONES, "H_BLOCKS_TENS": H_BLOCKS_TENS,
    "H_SEC_ONES": H_SEC_ONES, "H_SEC_TENS": H_SEC_TENS, "H_SUBSECOND": H_SUBSECOND,
    "H_MINUTES": H_MINUTES, "H_CLEAR_TARGET": H_CLEAR_TARGET,
    "H_THROW_A": H_THROW_A, "H_THROW_B": H_THROW_B, "H_HELD": H_HELD,
    "H_INFLIGHT_X": H_INFLIGHT_X, "H_INFLIGHT_Y": H_INFLIGHT_Y,
    "REPEAT_DELAY": REPEAT_DELAY, "REPEAT_RATE": REPEAT_RATE,
    "REPEAT_RELOAD": REPEAT_DELAY - REPEAT_RATE,
    "INTRO_FRAMES": INTRO_FRAMES, "THROW_FRAMES": THROW_FRAMES,
    "TOP_ROW": TOP_ROW, "CLEAR_TARGET": CLEAR_TARGET,
    "START_MINUTES": START_MINUTES, "START_SECONDS_TENS": START_SECONDS // 10,
    "START_SECONDS_ONES": START_SECONDS % 10,
    "LCDC": 0x40, "LY": 0x44, "JOYP": 0x00,
}


def stage_one():
    """The stage the memory map recorded, byte for byte.

    Rows 8-12, columns 1-5 are its own dump; the borders and the staircase are the shape it
    describes around them.
    """
    field = [[CELL_OUTSIDE] * FIELD_COLS for _ in range(FIELD_ROWS)]
    for col in range(FIELD_COLS):
        field[0][col] = CELL_BORDER               # ceiling
        field[FIELD_ROWS - 1][col] = CELL_BORDER  # floor
    for row in range(FIELD_ROWS):
        field[row][0] = CELL_BORDER               # left wall
    for row, cells in ((8,  [0x84, 0x85, 0x85, 0x86, 0x85]),
                       (9,  [0x84, 0x85, 0x83, 0x83, 0x84]),
                       (10, [0x86, 0x85, 0x84, 0x85, 0x86]),
                       (11, [0x84, 0x84, 0x83, 0x84, 0x83]),
                       (12, [0x83, 0x86, 0x86, 0x86, 0x83])):
        field[row][1:1 + len(cells)] = cells
    # The stepped diagonal the map describes in the upper left.
    for step, row in enumerate(range(4, 8)):
        field[row][1 + step] = CELL_STAIRCASE
    return field


def block_count(field):
    return sum(BLOCK_MIN <= cell <= BLOCK_MAX for row in field for cell in row)


PROGRAM = """
boot:
    di
    ld sp, $CFFF
    ld hl, $C000
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

menu:                           ; a title screen that waits for START
    call read_pad
    ld a, (NEW_PAD)
    bit 7, a
    jr nz, .start
    call wait_frame
    jr menu
.start:
    call load_stage
    ld b, INTRO_FRAMES          ; the stage announces itself before it will listen
    call wait_frames
    jp main

; ---------------------------------------------------------------- stage setup
load_stage:
    ld de, stage_data           ; 14 rows of 16 bytes, into a field of stride $20
    ld hl, FIELD
    ld c, 14
.row:
    ld b, 16
.cell:
    ld a, (de)
    inc de
    ld (hl+), a
    dec b
    jr nz, .cell
    push bc                     ; step hl on to the next row: $20 - $10
    ld bc, 16
    add hl, bc
    pop bc
    dec c
    jr nz, .row

    ld a, BLOCK_TOTAL_ONES      ; counters are decimal digits, ones first
    ldh (H_BLOCKS_ONES), a
    ldh (H_INITIAL_ONES), a
    ld a, BLOCK_TOTAL_TENS
    ldh (H_BLOCKS_TENS), a
    ldh (H_INITIAL_TENS), a
    ld a, CLEAR_TARGET
    ldh (H_CLEAR_TARGET), a
    ld a, 1
    ldh (H_STAGE), a
    ld a, START_MINUTES
    ldh (H_MINUTES), a
    ld a, START_SECONDS_TENS
    ldh (H_SEC_TENS), a
    ld a, START_SECONDS_ONES
    ldh (H_SEC_ONES), a
    xor a
    ldh (H_SUBSECOND), a
    ldh (H_THROW_A), a
    ldh (H_THROW_B), a
    ld a, $83                   ; the block in hand to start with
    ldh (H_HELD), a
    ld a, 8                     ; the player starts level with the top row of blocks
    ld (PLAYER_ROW), a
    call draw_player
    ret

; The player is sprite 0 in the OAM buffer: y = 40 + row*8, x = 140.
draw_player:
    ld a, (PLAYER_ROW)
    add a, a
    add a, a
    add a, a
    add a, 40
    ld hl, OAM
    ld (hl+), a
    ld a, 140
    ld (hl+), a
    ld a, $01
    ld (hl+), a
    xor a
    ld (hl), a
    ret

; ---------------------------------------------------------------- main loop
main:
    call wait_frame
    call tick_timer
    call read_pad
    ld a, (TRIG)
    and a
    jr z, main
    bit 2, a                    ; up
    jr nz, .up
    bit 3, a                    ; down
    jr nz, .down
    bit 4, a                    ; A throws
    jp nz, throw
    jr main
.up:
    ld a, (PLAYER_ROW)
    cp TOP_ROW
    jr z, main
    dec a
    ld (PLAYER_ROW), a
    call draw_player
    jr main
.down:
    ld a, (PLAYER_ROW)
    cp 12
    jr z, main
    inc a
    ld (PLAYER_ROW), a
    call draw_player
    jr main

; ---------------------------------------------------------------- the throw
; Scan the player's row from the right. The first playable block either matches what is in
; hand -- destroy it, and its column falls -- or does not, and the two swap.
throw:
    ld a, 1
    ldh (H_THROW_A), a
    ldh (H_THROW_B), a
    ld a, 100
    ldh (H_INFLIGHT_X), a
    ld b, THROW_FRAMES          ; the block spends this long crossing the field
.fly:
    ldh a, (H_INFLIGHT_X)
    sub 7
    ldh (H_INFLIGHT_X), a
    push bc
    call wait_frame
    pop bc
    dec b
    jr nz, .fly

    ld a, (PLAYER_ROW)
    call row_base               ; hl = FIELD + row*$20
    ld bc, 15
    add hl, bc
    ld a, 15
    ld (COL), a
.scan:
    ld a, (hl)
    cp $83
    jr c, .next
    cp $87
    jr nc, .next
    ld (TMP), a                 ; a playable block: same type as the one in hand?
    ldh a, (H_HELD)
    ld b, a
    ld a, (TMP)
    cp b
    jr z, .destroy
    ldh a, (H_HELD)             ; different: swap it into the field and take that one
    ld (hl), a
    ld a, (TMP)
    ldh (H_HELD), a
    jr .done
.destroy:
    ld a, (PLAYER_ROW)
    ld e, a
    call collapse
    call decrement_blocks
    jr .done
.next:
    dec hl
    ld a, (COL)
    dec a
    ld (COL), a
    jr nz, .scan
.done:
    xor a
    ldh (H_THROW_A), a
    ldh (H_THROW_B), a
    jp main

; hl = the destroyed cell, e = its row. Everything above it falls one row.
collapse:
    ld a, e
    cp 2
    jr c, .top
    push hl
    ld bc, $FFE0
    add hl, bc
    ld a, (hl)
    pop hl
    ld (hl), a
    ld bc, $FFE0
    add hl, bc
    dec e
    jr collapse
.top:
    xor a
    ld (hl), a
    ret

; Counters are decimal digits, ones first, so this borrows rather than subtracts.
decrement_blocks:
    ldh a, (H_BLOCKS_ONES)
    and a
    jr nz, .ones
    ld a, 9
    ldh (H_BLOCKS_ONES), a
    ldh a, (H_BLOCKS_TENS)
    and a
    ret z
    dec a
    ldh (H_BLOCKS_TENS), a
    ret
.ones:
    dec a
    ldh (H_BLOCKS_ONES), a
    ret

; ---------------------------------------------------------------- helpers
; hl = FIELD + a*$20
row_base:
    ld l, a
    ld h, 0
    add hl, hl
    add hl, hl
    add hl, hl
    add hl, hl
    add hl, hl
    ld bc, FIELD
    add hl, bc
    ret

tick_timer:
    ldh a, (H_SUBSECOND)
    inc a
    ldh (H_SUBSECOND), a
    cp 60
    ret nz
    xor a
    ldh (H_SUBSECOND), a
    ldh a, (H_SEC_ONES)
    and a
    jr nz, .ones
    ld a, 9
    ldh (H_SEC_ONES), a
    ldh a, (H_SEC_TENS)
    and a
    jr nz, .tens
    ld a, 5
    ldh (H_SEC_TENS), a
    ldh a, (H_MINUTES)
    and a
    ret z
    dec a
    ldh (H_MINUTES), a
    ret
.tens:
    dec a
    ldh (H_SEC_TENS), a
    ret
.ones:
    dec a
    ldh (H_SEC_ONES), a
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

; PAD: bit 0-3 right/left/up/down, 4-7 A/B/select/start. TRIG adds auto-repeat.
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
    ld (TRIG), a
    ld a, (PAD)
    and $0C                     ; up and down repeat; A does not
    jr z, .idle
    ld b, a
    ld a, (PREV_PAD)
    and $0C
    cp b
    jr nz, .idle
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


def build_rom(title=b"FLIPULLFAKE"):
    """Assemble the cartridge and return its 32 KiB image."""
    field = stage_one()
    total = block_count(field)
    symbols = dict(SYMBOLS, BLOCK_TOTAL_ONES=total % 10, BLOCK_TOTAL_TENS=total // 10)
    asm = Assembler(symbols)

    asm.org(0x0100)
    asm.asm("nop\njp boot")
    asm.org(0x0104)
    asm.db(NINTENDO_LOGO)
    asm.org(0x0134)
    asm.db(title[:16].ljust(16, b"\0"))
    asm.org(0x0147)
    asm.db([0x00, 0x00, 0x00])            # ROM only, 32 KiB, no cartridge RAM — like Flipull
    asm.org(0x014A)
    asm.db([0x01, 0x00, 0x00])            # non-Japan

    asm.org(0x0400)
    asm.asm(PROGRAM)

    asm.org(0x2000)
    asm.label("stage_data")
    asm.db([cell for row in field for cell in row])

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
    path = str(path)
    with open(path, "wb") as handle:
        handle.write(build_rom())
    return path


_CARTRIDGE = None


def synthetic_rom():
    """The cartridge, built once per process into a temp file cleaned up at exit."""
    global _CARTRIDGE
    if _CARTRIDGE is None:
        directory = tempfile.mkdtemp(prefix="planiverse-fake-flipull-")
        atexit.register(shutil.rmtree, directory, ignore_errors=True)
        _CARTRIDGE = write_rom(os.path.join(directory, "fake-flipull.gb"))
    return _CARTRIDGE


if __name__ == "__main__":
    import sys
    print(write_rom(sys.argv[1] if len(sys.argv) > 1 else "fake-flipull.gb"))
