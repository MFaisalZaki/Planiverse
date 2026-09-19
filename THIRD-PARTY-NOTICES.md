# Third-party material

The GPL-3.0 licence in [LICENSE](LICENSE) covers the source code, documentation and
benchmark definitions authored for this repository. It does not cover, and cannot grant
any rights in, the third-party material described below.

This repository ships no ROM image, no fragment of one, no disassembled or decompiled
original code, no original graphics and no audio. The four game environments are
reimplementations in Python; nothing in this repository runs the original programs. The two
emulator environments run whatever cartridge the user supplies, and supply none.

## Emulators

Two environments drive an emulator rather than a Python reimplementation:
`planiverse/environments/emulated/game_boy.py` runs a cartridge under
[PyBoy](https://github.com/Baekalfen/PyBoy) (LGPL-3.0), and
`planiverse/environments/emulated/stable_retro.py` runs a console under
[Stable-Retro](https://github.com/Farama-Foundation/stable-retro) (MIT, a fork of OpenAI's
Gym Retro). Both are dependencies, installed from PyPI, and neither is included here.

The Game Boy environment takes the cartridge from the user (`PLANIVERSE_GB_ROM`, or the
`rom=` argument) and ships none. The cartridge its tests run on, `tests/counter_rom.py`, is
an original program written for this repository and assembled by `tests/sm83.py`; it
contains no code, data or graphics from any published title. PyBoy's game wrappers, which
read a game's memory on the environment's behalf, are PyBoy's work; this repository holds no
memory map of any commercial title.

Stable-Retro ships one game with its own package, *Airstriker* (© Electrokinesis, distributed
by Stable-Retro on its author's terms), which is the Stable-Retro environment's default and
the only game its tests run. Every other Stable-Retro integration needs a ROM the user imports
into Stable-Retro; none is here.

## The flood adaptation model

`planiverse/environments/flood_transport/` follows the MAAT environment of
[floods_transport_rl](https://github.com/MLSM-at-DTU/floods_transport_rl) (MLSM-at-DTU, MIT):
Costa, Petersen, Vandervoort, Drews, Morrissey and Pereira, *Climate Adaptation with
Reinforcement Learning: Experiments with Flooding and Transportation in Copenhagen*, 2024.
The damage curves, road values and measure costs it tabulates come from van Ginkel, Dottori,
Alfieri, Feyen and Koks, *Flood risk assessment of the European road network*, Natural Hazards
and Earth System Sciences 21 (2021), and the rainfall projections from the Danish Klimaatlas,
both as MAAT tabulates them. None of MAAT's data is included: its cities, road networks and
flood maps stay with it, and the cities here are drawn from a seed.

## Copyright and trade marks

- *Adventures of Lolo*: © HAL Laboratory, Inc. / Nintendo Co., Ltd.
- *Puzznic*: © Taito Corporation
- *Flipull* / *Plotting*: © Taito Corporation
- *Amazing Tater*: © Atlus Co., Ltd.
- *Super Mario Land*: © Nintendo Co., Ltd.

All trade marks are the property of their respective owners and are used here
descriptively, to identify the titles studied. This project is unofficial and is
not affiliated with, endorsed by, or sponsored by any of the above.

## Game mechanics and level data

The environments in `planiverse/environments/games/` were written independently
from behaviour observed while the original titles were played. They are not ports,
translations or adaptations of the original programs.

The level and room layouts used as benchmark instances in three of them (the first 100 of
Puzznic's 128 rounds, the first 100 of Adventures of Lolo's 163 rooms and 100 of Amazing
Tater's 105 rooms) are derived from the original titles. They are included for
non-commercial research use under section 29A of the Copyright, Designs and Patents Act
1988, with the sources acknowledged above.
Flipull's stages are this project's own work throughout, as are all generated instances. See [docs/provenance.md](docs/provenance.md).
