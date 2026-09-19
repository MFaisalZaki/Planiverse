# Provenance

Where the game material in this repository came from, and how it was produced. It exists so
that "where did this come from" has one answer, for reviewers and for the licensing review
recorded in [THIRD-PARTY-NOTICES.md](../THIRD-PARTY-NOTICES.md).

Nothing here restates the licence. It records method.

## 1. Method

Every rule the four game environments implement was established by running the original title
under emulation, driving it with scripted input, recording its memory frame by frame and
matching what changed against what was on the screen at the time; never by reading a published
disassembly or a leaked source listing. The emulator-backed environments and the memory maps
that recorded those observations have been withdrawn from this repository. What remains is
what was written from them: the reimplementations, and the level data described below.

Deriving rules that way leaves confidence uneven, and each module says where it stops agreeing
with the original rather than smoothing it over:

- [`lolo.py`](../planiverse/environments/games/lolo.py)'s "Where this differs from the
  cartridge" is the model the others follow: six of the eight enemies are frozen, `EXACT_ROOMS`
  names the 21 of the 100 shipped rooms the model is therefore faithful for, and the divergence is shown
  to run in both directions rather than being presented as a relaxation.
- [`flipull.py`](../planiverse/environments/games/flipull.py) states its rules outright and
  says where each came from: four were derived by driving the cartridge, and the one an
  automated comparison showed to be missing, that a wall turns a thrown block downward, is
  the original's documented rule, cited in the module.
- [`puzznic.py`](../planiverse/environments/games/puzznic.py) records two known gaps, in
  where the cursor starts and in how eagerly matches are cleared.
- [`amazing_tater.py`](../planiverse/environments/games/amazing_tater.py) records none
  found, after a lockstep comparison across all 105 rooms.

## 2. The reimplementations

The pure-Python environments in [`games/`](../planiverse/environments/games/) were
written from the behaviour observed above. They are not ports, translations or adaptations of
the original programs: no original code was disassembled into them, and none of them shares
the original's structure, only its observable rules.

## 3. Level data

The layouts shipped as bundled instances are the part of this repository derived most directly
from the original titles, and they differ title by title:

| Environment | Shipped | Derived from the original? | How |
|---|---|---|---|
| Lolo | 100 rooms, the cartridge's first 100 of 163 | Yes | Decoded out of the cartridge's room table. Nothing transcribed by hand |
| Amazing Tater | 100 rooms, of the cartridge's 105 | Yes | Dumped from the board the running game composes in work RAM. Nothing transcribed by hand. The 41 PUZZLE rooms and 64 BEGINNER/ACTION rooms; the 96 PRACTICE rooms are deliberately absent |
| Puzznic | 100 rounds, the cartridge's first 100 of 128 | Yes | The first 50 transcribed by hand, the rest read out of the running game's grid memory |
| Flipull | 32 stages | **No: contract only** | Only the board size and CLEAR target match the original's 32-entry stage table. The arrangements are this project's own: the original draws each stage from an RNG seeded by boot timing, so there is no canonical layout to copy |

So three environments ship layouts taken from the original titles and one does not. Every
environment also generates instances of its own (`generate_instance`), and those are this
project's work throughout: a generated level, room, stage or season derives from nothing but
the seed.

For the three that ship derived layouts, the basis relied on is section 29A of the Copyright,
Designs and Patents Act 1988 (non-commercial research), with the sources acknowledged in
[THIRD-PARTY-NOTICES.md](../THIRD-PARTY-NOTICES.md).

## 4. What is not here

This repository ships **no ROM image, no fragment of one from which a ROM could be
reconstructed, no disassembled or decompiled original code, no original graphics and no
audio**. The renders under [`docs/renders/`](renders/) are drawn from this repository's own
reimplementations, with two exceptions the directory's README names: the Game Boy render
shows the test suite's own cartridge, and the Stable-Retro render shows *Airstriker*, the game
Stable-Retro ships and redistributes. The two emulator environments
([`emulated/`](../planiverse/environments/emulated/)) run a cartridge the user supplies and
supply none; the cartridge their tests run on is an original program written for this
repository ([`tests/counter_rom.py`](../tests/counter_rom.py)), and the knowledge of any
commercial game's memory they rely on lives in PyBoy's wrappers and Stable-Retro's
integrations, not here.

## 5. Open question

The copyright status of the derived layouts in section 3 is under review by the author's
institution. When that review reports, this document will record whether section 29A is the
right basis for the three environments that ship derived layouts, and whether the Flipull
instances, which are this project's own work throughout, need any
acknowledgement beyond the trade mark notice.
