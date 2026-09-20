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
the only game its tests run; `docs/renders/retro.gif` and `retro.png` show its screen, as
drawn by the emulator from a trace. Every other Stable-Retro integration needs a ROM the user
imports into Stable-Retro; none is here.

## Physics engines

`planiverse/environments/slingshot/` simulates its world with
[pymunk](https://www.pymunk.org/) (MIT), the Python binding of
[Chipmunk2D](https://chipmunk-physics.net/) (MIT). Both are dependencies installed from PyPI
and neither is included here. The game the environment plays is the genre's own; its
structures, materials, rules of breaking and levels are this repository's work.

`planiverse/environments/billiards/` plays on [pooltool](https://github.com/ekiefl/pooltool)
(Apache-2.0; Kiefl, JOSS 2024), an event-based billiards simulator, installed from PyPI as
`pooltool-billiards` and not included here. The environment places the balls itself and
keeps only the physics; its tables and rules are this repository's work.

## The Micropolis engine

`planiverse/environments/micropolis/` drives MicropolisCore, the C++ simulation engine of
[Micropolis](https://github.com/SimHacker/micropolis), the GPL-3.0 release of the original
SimCity's source by Electronic Arts (2008). It is not included here: `scripts/build_micropolis.sh`
fetches it, builds its SWIG binding for Python 3 and installs it, and the environment imports it
as `micropolisengine`. The engine's licence carries additional terms under GPL section 7 that
any conveyance must reproduce: no trademark or publicity rights are granted, and in particular
no right in the trademark SimCity or any other Electronic Arts trademark; a modification may
not be distributed under the trademark SimCity or claim affiliation with Electronic Arts;
modified versions must be marked as such; and the program is provided as is, with the
disclaimer the source carries. This repository conveys no part of the program and no
modification of it; the build script applies three one-line patches on the user's machine (two
Python 2 names in the SWIG callback, and one access specifier so that the engine's random seed
can be set from Python), and the environment's cities and rules are this repository's work.

Micropolis is a registered trademark of Micropolis Corporation (Micropolis GmbH) and is licensed
here as a courtesy of the owner (https://micropolis.com/), under the *"Micropolis" Public Name
License* that accompanies the source, which asks for this attribution wherever the name is used.

## The factory simulator

`planiverse/environments/factory/` drives [factory-sim](https://github.com/divagr18/factory-sim)
(MIT License, Copyright (c) 2026 factory-sim contributors), a C simulator of a small slice of
Factorio's early game with a Python binding. It is not included here: `scripts/build_factory_sim.sh`
fetches it at a pinned commit, compiles it, and installs it with its LICENSE and NOTICE files
beside the package, and the environment imports it as `fsim`. factory-sim's own notice is explicit about
what it holds: every rule and number in it was measured on the running game and written down,
and no code, data file, prototype definition, art, sound or font of Factorio's was copied or is
included. This repository adds nothing of Factorio's either; the environment's decisions, rules,
patches and reference lines are this repository's work, and its scenes are drawn with
factory-sim's own scene generator.

Factorio is a game by Wube Software Ltd and "Factorio" is a trademark of Wube Software Ltd,
used here descriptively to say what the simulator was measured against. Neither factory-sim nor
this repository is affiliated with, sponsored by or endorsed by Wube Software Ltd.

## The operational simulators

Four environments drive simulators installed from PyPI, none of them included here:

- `planiverse/environments/epidemic/` runs [Covasim](https://github.com/institutefordiseasemodeling/covasim)
  (MIT), the Institute for Disease Modeling's agent-based model of COVID-19.
- `planiverse/environments/traffic/` runs [SUMO](https://eclipse.dev/sumo/), the Eclipse
  Foundation's traffic simulator, under the Eclipse Public License 2.0 with the GNU GPL version 2
  or later as a secondary licence, through the `eclipse-sumo` binaries and the `libsumo` binding;
  the grid is drawn by its `netgenerate` on the user's machine.
- `planiverse/environments/airspace/` runs [BlueSky](https://github.com/TUDelft-CNS-ATM/bluesky)
  (TU Delft, MIT), which flies aircraft on the [OpenAP](https://github.com/junzis/openap)
  performance model (LGPL-3.0) over BlueSky's navigation data package (GPL-3.0);
  `scripts/install_bluesky.sh` installs the three, since the simulator's own dependency list names
  a package that no longer builds.
- `planiverse/environments/reservoir/` runs [pywr](https://github.com/pywr/pywr) (University of
  Manchester, GPL-3.0-or-later), the water resource system simulator, through a network this
  repository builds in code.

Each is used as a library through its published interface; the instances, decisions, constraints
and targets are this repository's work.

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
- *Pipe Mania* / *Pipe Dream*: © The Assembly Line; published as *Pipe Dream* by Lucasfilm
  Games, and on the Game Boy by Bullet-Proof Software
- *Super Mario Land*: © Nintendo Co., Ltd.
- *Factorio*: © Wube Software Ltd.

All trade marks are the property of their respective owners and are used here
descriptively, to identify the titles studied. This project is unofficial and is
not affiliated with, endorsed by, or sponsored by any of the above.

## Game mechanics and level data

The environments in `planiverse/environments/games/`, and the Pipe Dream-like in
`planiverse/environments/pipe_dream/`, were written independently from behaviour observed
while the original titles were played. They are not ports, translations or adaptations of
the original programs.

The level and room layouts used as benchmark instances in three of them (the first 100 of
Puzznic's 128 rounds, the first 100 of Adventures of Lolo's 163 rooms and 100 of Amazing
Tater's 105 rooms) are derived from the original titles. They are included for
non-commercial research use under section 29A of the Copyright, Designs and Patents Act
1988, with the sources acknowledged above.
Flipull's stages and every Pipe Dream level are this project's own work throughout, as are all
generated instances. See [docs/provenance.md](docs/provenance.md).
