# Renders

Planner traces on the first instance of each environment, one pair per environment: an
animation and a sheet with the whole plan on one image. A board is drawn as its own text;
an environment whose state is a handful of readings (the city, the factory, the crop, the
grid, the water network, the flood city, the tower defence) is drawn as a chart of those
readings over the plan, and its files are named `<environment>_chart`. See
[docs/rendering.md](../rendering.md).

Every file in this directory was drawn by this repository's renderer from a trace of one of
its environments; none contains imagery from any third party, and all are covered by the
project licence.
