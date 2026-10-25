# Renders

Planner traces on the first instance of each environment, one pair per environment: an
animation and a sheet with the whole plan on one image (the network attack has the sheet
alone). Each file is named after its environment. A state
is drawn as its text picture, black on white: a board, a field or map, or a schematic of the
system, with one status line; the two emulators are drawn as their consoles' own screens. See [docs/rendering.md](../rendering.md).

Every file in this directory was drawn by this repository's renderer from a trace of one of
its environments. Two contain a console's screen: `game_boy` shows the cartridge the test
suite assembles for itself (an original program, with PyBoy's own boot logo on screen) and
`retro` shows *Airstriker*, the game Stable-Retro ships with its package and redistributes on
its author's terms (see [THIRD-PARTY-NOTICES.md](../../THIRD-PARTY-NOTICES.md)). Everything
else is this repository's own work and covered by the project licence.
