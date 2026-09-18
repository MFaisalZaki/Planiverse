# Third-party material

The GPL-3.0 licence in [LICENSE](LICENSE) covers the source code, documentation,
memory maps and benchmark definitions authored for this repository. It does not
cover, and cannot grant any rights in, the third-party material described below.

This repository ships no ROM image, no fragment of one, no disassembled or
decompiled original code, and no audio. The emulator-backed environments require
the user to supply their own legally obtained cartridge image.

## Screen captures

The following files are frames captured from commercial Game Boy titles running
under emulation, and contain those titles' original graphics:

- `docs/renders/lolo_gb.png`, `docs/renders/lolo_gb.gif`
- `docs/renders/puzznic_gb.png`, `docs/renders/puzznic_gb.gif`
- `docs/renders/flipull_gb.png`, `docs/renders/flipull_gb.gif`
- `docs/renders/amazing_tater_gb.png`, `docs/renders/amazing_tater_gb.gif`

They are reproduced solely to illustrate and evaluate the accuracy of the
reimplementations in this repository, as criticism, review and quotation under
sections 30(1) and 30(1ZA) of the Copyright, Designs and Patents Act 1988.

**These files are not licensed under the GPL-3.0.** They remain the copyright of
their respective owners, and no right to redistribute or modify them is granted
by this repository's licence. Downstream users who redistribute this repository
should satisfy themselves of their own position, or omit these files.

## Copyright and trade marks

- *Adventures of Lolo* — © HAL Laboratory, Inc. / Nintendo Co., Ltd.
- *Puzznic* — © Taito Corporation
- *Flipull* / *Plotting* — © Taito Corporation
- *Amazing Tater* — © Atlus Co., Ltd.
- *Super Mario Land* — © Nintendo Co., Ltd.

All trade marks are the property of their respective owners and are used here
descriptively, to identify the titles studied. This project is unofficial and is
not affiliated with, endorsed by, or sponsored by any of the above.

## Game mechanics and level data

The pure-Python environments in `planiverse/environments/gameboy_py/` were written
independently from behaviour observed on the cartridge, as documented in the
corresponding memory maps under `docs/environments/`. They are not ports,
translations or adaptations of the original programs.

The level and room layouts used as benchmark instances are derived from the
original titles. They are included for non-commercial research use under
section 29A of the Copyright, Designs and Patents Act 1988, with the sources
acknowledged above. See [docs/provenance.md](docs/provenance.md).
