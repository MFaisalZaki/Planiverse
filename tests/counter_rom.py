"""A synthetic Game Boy cartridge for testing the generic Game Boy environment.

Nothing commercial: an original program, assembled by `sm83.py`, that keeps a counter in
work RAM and draws it on the background map every frame. Right adds one, left takes one
away (never below zero), and A puts it back to zero. Every press is counted too. That is
enough to give the environment something to plan over, with a cartridge title no game
wrapper claims, so what is tested is the generic path: buttons in, tiles and watched memory
out, save states in between.

    COUNTER  $C000   what right and left move
    PRESSES  $C001   how many presses the cartridge has seen

Both are drawn at the top-left of the background map, so `game_area()` shows them too.
"""
import atexit
import os
import shutil
import tempfile

from sm83 import Assembler

COUNTER = 0xC000
PRESSES = 0xC001
PAD, PREV_PAD, NEW_PAD = 0xC010, 0xC011, 0xC012
TITLE = b"PLANIVERSE-CNT"

NINTENDO_LOGO = bytes.fromhex(
    "CEED6666CC0D000B03730083000C000D0008111F8889000E"
    "DCCC6EE6DDDDD999BBBB67636E0EECCCDDDC999FBBB9333E"
)

SYMBOLS = {
    "COUNTER": COUNTER, "PRESSES": PRESSES,
    "PAD": PAD, "PREV_PAD": PREV_PAD, "NEW_PAD": NEW_PAD,
    "LCDC": 0x40, "LY": 0x44, "JOYP": 0x00,
    "TILEMAP": 0x9800,
}

PROGRAM = """
boot:
    di
    ld sp, $CFFF
    ld hl, $C000                ; clear work RAM
    ld bc, $1000
.clear:
    xor a
    ld (hl+), a
    dec bc
    ld a, b
    or c
    jr nz, .clear
    ld a, $91                   ; LCD on, so frames complete
    ldh (LCDC), a

main:
    call wait_frame             ; returns inside VBlank, when VRAM may be written
    ld a, (COUNTER)
    ld (TILEMAP), a             ; the counter is the tile at the top-left corner
    ld a, (PRESSES)
    ld (TILEMAP+1), a           ; and the press count is the one beside it
    call read_pad
    ld a, (NEW_PAD)
    and a
    jr z, main
    ld b, a
    ld a, (PRESSES)
    inc a
    ld (PRESSES), a
    bit 0, b                    ; right
    jr z, .not_right
    ld a, (COUNTER)
    cp 250
    jr z, main
    inc a
    ld (COUNTER), a
    jr main
.not_right:
    bit 1, b                    ; left
    jr z, .not_left
    ld a, (COUNTER)
    and a
    jr z, main
    dec a
    ld (COUNTER), a
    jr main
.not_left:
    bit 4, b                    ; A
    jr z, main
    xor a
    ld (COUNTER), a
    jr main

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
    ret
"""


def build_rom():
    """Assemble the cartridge and return its 32 KiB image."""
    asm = Assembler(SYMBOLS)
    asm.org(0x0100)
    asm.asm("nop\njp boot")
    asm.org(0x0104)
    asm.db(NINTENDO_LOGO)
    asm.org(0x0134)
    asm.db(TITLE[:16].ljust(16, b"\0"))
    asm.org(0x0147)
    asm.db([0x00, 0x00, 0x00])            # ROM only, 32 KiB, no cartridge RAM
    asm.org(0x014A)
    asm.db([0x00, 0x00, 0x00])            # Japan, licensee, version
    asm.org(0x0150)
    asm.asm(PROGRAM)
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


def counter_rom():
    """The cartridge, built once per process into a temp file that is cleaned up at exit."""
    global _CARTRIDGE
    if _CARTRIDGE is None:
        directory = tempfile.mkdtemp(prefix="planiverse-counter-rom-")
        atexit.register(shutil.rmtree, directory, ignore_errors=True)
        _CARTRIDGE = write_rom(os.path.join(directory, "counter.gb"))
    return _CARTRIDGE


if __name__ == "__main__":
    import sys
    print(write_rom(sys.argv[1] if len(sys.argv) > 1 else "counter.gb"))
