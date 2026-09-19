# Traffic signals

Signals on a SUMO grid. Three by three signalised junctions, a stub road off each edge of the
map, a seeded morning of trips from one stub to another, and a decision every thirty seconds:
switch one junction, a row, a column, all nine, or none. A switch is three seconds of amber and
then the other green. The goal is every trip through before the horizon with the total travel
time under the target. Car-following, lane changing, queues spilling back through the grid and
the time each trip takes are SUMO's, and the effect of a switch is only known by running the
next thirty seconds, which keeps the environment out of PDDL.

[SUMO](https://eclipse.dev/sumo/) (Eclipse Public License 2.0 with GPL-2.0-or-later as a
secondary licence; Lopez et al. 2018, https://doi.org/10.1109/ITSC.2018.8569938) is the Eclipse
Foundation's microscopic traffic simulator. It is a dependency installed from PyPI as
`eclipse-sumo` (the binaries) and `libsumo` (the in-process binding); nothing of it is included
here. The grid is drawn by its `netgenerate` on first use and the trips by the environment from
the instance's seed.

- **Import:** `from planiverse.environments.traffic.environment import TrafficEnv`
- **Source:** [`planiverse/environments/traffic/environment.py`](../../planiverse/environments/traffic/environment.py)
- **Instances:** 100 mornings, indices `0` to `99`, all drawn by the generator at recorded seeds
- **Generator:** `generate_instance(seed, vehicles=None, spacing=None, horizon=None, ...)`; see [Generating mornings](#generating-mornings)
- **Dependencies:** `eclipse-sumo`, `libsumo`

## How a trajectory problem becomes a goal

| The trajectory asks for | The environment does |
|---|---|
| everyone through by the horizon | the goal; the horizon passing with vehicles still on the road is a dead end |
| little travel time overall | vehicle-seconds accumulate in the state and are bounded at the goal; everyone through over the bound is a dead end, since nothing can then improve |
| the horizon itself | the time is in the state |

The target is the least travel any fixed cycle achieves with a tenth in hand, so the planner is
asked to come within a tenth of a timer by switching where the queues are. [NOTES.md](../../NOTES.md) sets the
reduction out in full.

## Quickstart

```python
from planiverse.environments.traffic.environment import TrafficEnv, TrafficAction

env = TrafficEnv()
env.set_index(0)
state, info = env.reset()          # info: morning, vehicles, horizon, target, generated
print(state)                       # t=0s: 0 of 120 arrived, 0 on the road, 120 to come; ...

state, arrived = env.step("switch(B1)")
for action, child in env.successors(state):
    print(action, child.running, child.travel)
env.close()
```

## The rules

1. The grid is SUMO's `netgenerate` three-by-three grid with 150 m blocks and 120 m stubs, every
   junction a signal with a north-south and an east-west green. The trips are the instance's:
   one departure every `spacing` seconds from a seeded entry stub to a seeded exit stub on
   another side.
2. A decision is thirty seconds. `switch(j)` puts junction `j` on amber for three seconds and
   then on the other green; `switch(row1)`, `switch(col2)` and `switch(all)` do it to a row, a
   column or all nine at once; `hold` changes nothing.
3. The goal is every vehicle arrived before the horizon with the vehicle-seconds spent on the
   road under the target. The horizon passing with vehicles left is a dead end, and so is
   everyone arrived over the target.

## State

`TrafficState` holds the decisions so far and the grid after them: the time, the vehicles
arrived, on the road and still to depart, the vehicle-seconds spent, and the green each
junction shows. Identity is the path: SUMO replays exactly from its seed and the same commands,
and every expansion replays the morning from the first vehicle, which takes a tenth of a second
for two hundred vehicles.

```
green(B1, A)
decision(4)
arrived(31)
running(24)
pending(65)
travel(2500)
```

## Actions

Seventeen: `switch` of each of the nine junctions, of each of three rows and three columns, of
all, and `hold`, each costing 1.

## Mornings

The hundred mornings are the generator's own draws, embedded in the module as the plain data
`set_instance` takes (`seed`, `vehicles`, `spacing`, `horizon`, `target`) with the seed each came
from beside it, and the plan each was accepted on is in `tests/data/traffic_solutions.json`.
Mornings have 120 to 240 vehicles, one every one to two seconds, and horizons of ten to fifteen
minutes.

## Generating mornings

```python
env = TrafficEnv()
morning = env.generate_instance(seed=7, vehicles=160)
print(env.witness, env.witness_expansions)     # the cycle it was accepted on, and the cycles measured
```

| Option | Default | What it does |
|---|---|---|
| `vehicles` | 120, 160, 200 or 240 | the morning's trips |
| `spacing` | 1.0, 1.5 or 2.0 s | between departures |
| `horizon` | 600, 750 or 900 s | the deadline |
| `attempts` | 40 | draws before giving up with `GenerationError` |

A draw is measured by fixed cycles that switch every junction every one, two or three
decisions, and by holding the signals as they start. The target is the least travel any of them
clears the grid with before the horizon, with a tenth in hand; the draw is thrown back when none
does or when holding is already best, and the best cycle is the witness.

## Files

| File | Contents |
|---|---|
| [`environment.py`](../../planiverse/environments/traffic/environment.py) | `TrafficAction`, `TrafficState`, the grid (`network`, `trips`), `TrafficEnv`, `MORNINGS` |
| [`tests/test_operational_four.py`](../../tests/test_operational_four.py) | Tests |
| [`tests/data/traffic_solutions.json`](../../tests/data/traffic_solutions.json) | The plan each morning was accepted on |
