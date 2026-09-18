# Provenance

Where the Game Boy material in this repository came from, and how it was produced. It exists
so that "where did this come from" has one answer, for reviewers and for the licensing
review recorded in [THIRD-PARTY-NOTICES.md](../THIRD-PARTY-NOTICES.md).

Nothing here restates the licence. It records method.

## 1. Method

Every address this project reads was found by running the cartridge under emulation and
watching its memory, never by reading a published disassembly or a leaked source listing.
The procedure was the same for each title: boot the game in [PyBoy](https://github.com/Baekalfen/PyBoy),
drive it with scripted input, record work RAM frame by frame, and match what changed against
what was on the screen at the time.

The two maps that state their method in as many words are the model for the rest:

- Flipull's was "derived behaviourally (recording WRAM and HRAM every frame against known
  on-screen values)", which is why
  [`flipull_gb.py`](../planiverse/environments/gameboy/flipull_gb.py) grades its constants by
  confidence rather than asserting them flatly.
- Super Mario Land's was derived the same way, "recording work RAM every frame while driving
  scripted input".

Amazing Tater's cell codes were matched against the rendered screen and, for the block
shapes, against the cartridge's own 15-entry shape table at `$0BF6`.

Two things follow from deriving a map this way, and both are recorded per title rather than
smoothed over.

**Confidence varies field by field.** Each memory map ends with a section saying what the
probing did not settle: [Lolo §9](environments/lolo-gb-memory-map.md),
[Puzznic §9](environments/puzznic-gb-memory-map.md),
[Flipull §8](environments/flipull-gb-memory-map.md) and
[Amazing Tater §8](environments/amazing-tater-gb-memory-map.md). Flipull's records that what
a throw actually hits was never established; Lolo's records that the six mobile enemies run
on the frame clock once woken and that no probe yet fixes their trigger or range.

**The environments re-measure at run time rather than trusting the map.** The four puzzle
cartridges each carry a `calibrate` that measures the d-pad hold window off the cartridge in
front of it, and Flipull and Puzznic add `probe_sprites`, `probe_initial_hand`,
`probe_throw_button` and `probe_push_scheme` for the facts that differ between dumps. A
cartridge that behaves unexpectedly is handled rather than silently mis-driven.

## 2. The reimplementations

The pure-Python environments in [`gameboy_py/`](../planiverse/environments/gameboy_py/) were
written from the behaviour observed above. They are not ports, translations or adaptations of
the original programs: no original code was disassembled into them, and none of them shares
the cartridge's structure, only its observable rules.

Each module states where it stops agreeing with the cartridge, in its own docstring, and does
so in the terms a reader would need to catch it out.
[`lolo.py`](../planiverse/environments/gameboy_py/lolo.py)'s "Where this differs from the
cartridge" is the model the others follow: six of the eight enemies are frozen, `EXACT_ROOMS`
names the 26 of 163 rooms the model is therefore faithful for, and the divergence is shown to
run in both directions rather than being presented as a relaxation — in tutorial 4a a frozen
Rocky blocks the only approach to a required heart framer, so BFWS proves the room unsolvable
in the twin while clearing it on the cartridge in 56 moves.

[`flipull.py`](../planiverse/environments/gameboy_py/flipull.py) is the most explicit: it
calls itself "a Flipull-*like* environment with a stated rule set, not a clone", because over
an automated comparison it agreed with the cartridge on about half of the level throws and
four in five of the throws from above the wall.

## 3. Level data

The layouts shipped as benchmark instances are the part of this repository derived most
directly from the original titles, and they differ title by title:

| Environment | Shipped | Derived from the cartridge? | How |
|---|---|---|---|
| Lolo | 163 rooms | Yes | Decoded out of the ROM's room table (bank 13, flat and uncompressed) by `lolo_gb.read_rooms`. Nothing transcribed by hand |
| Amazing Tater | 105 rooms | Yes | Dumped from the board the cartridge composes in work RAM, through `AmazingTaterGBEnv.levels`. Nothing transcribed by hand. The 41 PUZZLE rooms and 64 BEGINNER/ACTION rooms; the 96 PRACTICE rooms are deliberately absent |
| Puzznic | 128 rounds | Yes | The first 50 transcribed by hand, the rest read out of the grid at `$DF00` by booting each round |
| Flipull | 32 stages | **No — contract only** | Only the board size and CLEAR target match the cartridge's 32-entry stage table. The arrangements are this project's own: the cartridge draws each stage from an RNG seeded by boot timing, so there is no canonical layout to copy |
| Super Mario Land | 12 levels | **No** | Original levels. Only the count matches the cartridge's four worlds of three |

So three environments ship layouts taken from the original titles and two do not. The
cartridge-backed environments in [`gameboy/`](../planiverse/environments/gameboy/) ship no
level data at all: they read it out of the user's own ROM at run time.

For the three that do, the basis relied on is section 29A of the Copyright, Designs and
Patents Act 1988 — non-commercial research — with the sources acknowledged in
[THIRD-PARTY-NOTICES.md](../THIRD-PARTY-NOTICES.md).

## 4. What is not here, and what is

This repository ships **no ROM image, no fragment of one from which a ROM could be
reconstructed, and no audio**. The emulator-backed environments require the user to supply
their own legally obtained cartridge image, and each verifies it by MD5 before use. The only
original graphics present are the `_gb` screen captures carved out of the licence in
[THIRD-PARTY-NOTICES.md](../THIRD-PARTY-NOTICES.md).

Short extracts of cartridge code **are** present, however, and are listed here rather than
left for a reviewer to find:

| Location | What it is |
|---|---|
| [`puzznic-gb-memory-map.md` §8](environments/puzznic-gb-memory-map.md) | 16 lines of the grid cell-address routine at `$29CE`, annotated, reproducing the original opcode bytes alongside the mnemonics |
| [`lolo-gb-memory-map.md` §2](environments/lolo-gb-memory-map.md) | Instruction sequences from `LoadRoom` at `$11E5`, from `$26AA` and from `$179D`, as mnemonics without opcode bytes |
| [`amazing-tater-gb-memory-map.md` §2](environments/amazing-tater-gb-memory-map.md) | A three-instruction bank-switch sequence, as mnemonics |
| Cartridge header rows in four memory maps | The entry point at `$0100`, `NOP` / `JP $0150`; Lolo's map also gives its four bytes |

They are there because each one documents a hook point the environment depends on — the
instruction `lolo_gb` registers its room hook against, the routine whose geometry gives
Puzznic's grid stride — and they are extracts for explanation, not a disassembly of the
programs.

TODO(author): this conflicts with the sentence in `THIRD-PARTY-NOTICES.md` that reads "This
repository ships no ROM image, no fragment of one, no disassembled or decompiled original
code, and no audio." That sentence is not accurate as written while the extracts above are in
the tree. The notice is a draft for legal review, so the wording is left as drafted and the
conflict is flagged here: legal should decide whether to narrow the sentence, bring the
extracts under the same criticism-and-review carve-out as the screen captures, or remove the
Puzznic listing, which is the only one reproducing original bytes.

## 5. Open question

The copyright status of the derived layouts in section 3 is under review by the author's
institution. This document will be updated when that review reports.

TODO(author): record the outcome here, including whether section 29A is the right basis for
the three environments that ship derived layouts, and whether the Flipull and Super Mario
Land instances — which are this project's own work throughout — need any acknowledgement
beyond the trade mark notice.
