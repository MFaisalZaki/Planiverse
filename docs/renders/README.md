# Renders

Planner traces on the first instance of each environment, one pair per environment: an
animation and a sheet with the whole plan on one image. A board is drawn as its own text;
an environment whose state is a handful of readings (the city, the factory, the crop, the
grid, the water network, the flood city, the tower defence) is drawn as a chart of those
readings over the plan, and its files are named `<environment>_chart`; the two emulators are
drawn as their consoles' own screens. See [docs/rendering.md](../rendering.md).

Every file in this directory was drawn by this repository's renderer from a trace of one of
its environments. Two contain a console's screen: `game_boy` shows the cartridge the test
suite assembles for itself (an original program, with PyBoy's own boot logo on screen) and
`retro` shows *Airstriker*, the game Stable-Retro ships with its package and redistributes on
its author's terms (see [THIRD-PARTY-NOTICES.md](../../THIRD-PARTY-NOTICES.md)). Everything
else is this repository's own work and covered by the project licence.
