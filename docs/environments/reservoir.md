# Reservoir

Reservoir operations on pywr: two reservoirs in series, the upper fed by the hills and emptied
through a turbine at the release the operator sets, the lower fed by the turbine and the
valley. The lower one supplies a river that must keep flowing, a city that must be supplied,
and a farm whose seasonal demand may be rationed. A decision a month for a year: the release, and
whether the farm gets its full demand or half. The goal is the year complete with the farm's
shortfall under the target and both reservoirs above their reserves at the end.

- **Import:** `from planiverse.environments.reservoir.environment import ReservoirEnv`
- **Source:** [`planiverse/environments/reservoir/environment.py`](../../planiverse/environments/reservoir/environment.py)
- **Instances:** 100 years, indices `0` to `99`, all drawn by the generator at recorded seeds
- **Generator:** `generate_instance(seed, ...)`; see [Generating years](#generating-years)
- **Dependency:** `pywr`

## Context

The game here is the one a reservoir operator plays. Water comes off the hills into the upper
reservoir through the winter and dries up through the summer, when the farm wants most of it.
The city wants the same amount every month and the river a minimum flow. Each month we set two
things: how much the turbine lets down from the upper reservoir to the lower, and whether the
farm gets its full demand or is rationed to half. Which demand gets what each month is pywr's
allocation by priority; when to hold water back and when to let it through is the plan. What
makes it hard is that the decisions are coupled across the year. Water released in January is
in the lower reservoir, where the city and the farm draw on it, and no longer in the upper one
for July. Water held back keeps the upper reservoir above its reserve but leaves the lower one
to run short, and a single month that shorts the city or the river ends the year.

[pywr](https://github.com/pywr/pywr) (GPL-3.0-or-later; Tomlinson, Arnott and Harou 2020,
https://doi.org/10.1016/j.envsoft.2020.104635) is the water resource system simulator from the
University of Manchester used across the UK water industry. It models a network of storages,
demands and links, and allocates each step's flows by a linear programme over the nodes'
priorities (i.e., each node carries a cost, and the flows chosen minimise the total). It is a dependency installed from PyPI; nothing of it is included here. Of the four
operational environments added together it is the one a numeric planner could model most
nearly, since a mass balance with priorities is a small linear system. It is here for what it
is, a real water-industry simulator with a year's worth of coupled decisions. What the simulator buys is
an allocation the water industry recognises; what it costs is only a few milliseconds a month,
which makes this the cheapest of the four to expand.

The rules are four. First, the network is built in code from the instance: the hills feed the
upper reservoir, the turbine and the valley the lower, and the river, city, farm and sea come
off it. pywr allocates each month by cost: the river first, then the
city, then the farm, with the reservoirs holding what is not taken and spilling to the sea only
when full. Second, a month is one decision: `release(r, share)`, the turbine's flow `r` from 2
to 10 and the farm's share `full` or `half`. Rationing counts against the farm's full demand as
shortfall. Third, a month in which the river gets less than its minimum or the city less than
ninety per cent of its demand is a dead end. Fourth, the goal is the twelfth month done with the
farm's shortfall under the target and both reservoirs at or above their reserves. The year
ending otherwise is a dead end.

## Actions

| Action | Cost | Release | Farm's share | Effect |
|---|---|---|---|---|
| `release(r, full)`, `r` in 2, 4, 6, 8, 10 | 1 | `r` a month through the turbine | the full demand | a month passes with the farm supplied as far as the water allows |
| `release(r, half)`, `r` in 2, 4, 6, 8, 10 | 1 | `r` a month through the turbine | half the demand | a month passes with the farm rationed, the rest of its demand counted as shortfall |

Ten actions in all, and `get_actions` offers all ten in every state; `ReservoirAction` refuses
any other release or share with a `ValueError`, and a goal or dead-end state has no successors
at all. Applying an action sets the turbine's flow and the farm's share on the model, steps pywr
one month, and reads what the city, the farm, the river and the turbine got. It then adds to
three running shortfalls: the farm's against its full demand, the city's against ninety per
cent of its demand, and the river's against its minimum. Note that the environment keeps the
model for the settings it last applied and steps it on when the next settings extend them;
otherwise it replays the year from January, which takes a few milliseconds. The model is built
once per instance, as twelve steps of pywr's clock standing for the months.

## Planning problem

A `ReservoirState` holds the month, the two volumes, the three shortfalls so far and the last
month's deliveries (the city's, the farm's, the river's and the turbine's), with the settings
kept for replay. Two states are the same state when their month, volumes and shortfalls are the
same, to two decimals. pywr is deterministic and memoryless beyond its storages, so two release
histories that leave the same water are one state. This is the one respect in which the
environment differs from the other three, whose states are paths. The literals name the month
and the farm's shortfall in whole units, and the volumes in bands of five, with `city_short`
and `river_short` added once either has been shorted. The bands are what the width-based
planners' novelty tests see:

```
month(4)
upper(60)
lower(25)
farm_short(3)
```

With `F` the farm shortfall the target allows and `R_u` and `R_l` the reserves the two
reservoirs must hold at the end of the year, the goal and the dead end are

```
goal(s)     ≡ month(s) = 12 ∧ city_short(s) = 0 ∧ river_short(s) = 0 ∧ farm_short(s) ≤ F
              ∧ upper(s) ≥ R_u ∧ lower(s) ≥ R_l
terminal(s) ≡ city_short(s) > 0 ∨ river_short(s) > 0 ∨ (month(s) = 12 ∧ ¬goal(s))
```

Note that the problem as an operator would state it is not an initial-to-goal problem but a
trajectory problem (i.e., one whose requirements hold over the whole run rather than at its
end). The river and the city must be met every month, the farm should be rationed little over
the year, and the reserves must be in hand at its end. We turn it into the goal above with the
reduction the epidemic, traffic and airspace environments share, which
[NOTES.md](../../NOTES.md) sets out in full, in three moves. First, a constraint that must hold
throughout becomes a dead end: the first month that shorts the river or the city is terminal,
so no plan through it exists. A check over the whole trajectory thus becomes a check at each
state. Second, a quantity that accumulates becomes part of the state and is bounded at the
goal. The farm's shortfall is carried on the state, added to every month it is rationed, and
compared with the target when the year is done. Third, the horizon becomes time in the state:
the month is a state variable, so "by the end of the year" is a condition on a state. The
reserves are a condition on that final state too. The cost of the reduction is that the goal
depends on a target, and a target has to come from somewhere. Here it is the least shortfall
any fixed policy manages inside the constraints (see [Generating years](#generating-years)), and
a draw that a steady release all year already solves is thrown back. Every instance is thus
solvable by construction, and the planner is asked to do at least as well as the best rule by
varying the release month by month.

The progress measure the width planners take (`planiverse.benchmark.measures.reservoir`) is the
months still to run plus the farm's shortfall over the target.

## An example

Instance `0` is a year with reservoirs of 80 and 40 units, starting at 40 and 29 and due to
hold 28 and 19 at the end. The city wants 5.4 a month, the river 1.0, and the farm 48.0 over the
year, from 0.9 in November to 8.7 in June. The hills bring 70.4 over the year, from 9.5 in
January down to 1.7 in August, and the valley 22.1. The target allows a farm shortfall of
18.15. Iterated BFWS, run as the benchmark runs it (i.e., with the measure above and a width
bound of 1000), solved it in 151 expansions and 2.3 seconds. Its twelve-month plan is

```
release(2, full), release(2, full), release(2, full), release(2, full),
release(10, half), release(10, half), release(8, half), release(10, full),
release(8, full), release(8, half), release(10, full), release(8, half)
```

which holds the release at 2 through the winter, when the hills are high and the farm wants
little, so the upper reservoir fills from 40 to 64 while the lower drains to 15. It then opens
the turbine to 8 or 10 for the rest of the year, rationing the farm through May, June and July
and again in October and December. The year ends with the upper at 30 and the lower at 20, two
and one above their reserves, and a shortfall of 13.2 against the 18.15 allowed. The seasonal
policy the year was accepted on (4 through the winter, 10 with the farm on half through the
summer) ends at 18.15, which is where the target comes from. Note that no steady release works
here: 2 and 4 short the city by the
summer, and 6, 8 and 10 leave one reservoir under its reserve.

![BFWS solving reservoir instance 0](../renders/reservoir_chart.gif)

The render draws the readings of each state over the plan rather than the state's text, since a
state here is a handful of numbers. The panels are the two volumes, the month's release and
deliveries, and the farm's shortfall against the target, month by month. A frame of the GIF is
the chart up to its state, so the animation grows a step at a time. The same plan is also a
[sheet](../renders/reservoir_chart.png), the whole plan on one figure. There the action that
produced each state runs along the bottom, the target is a dashed line, and the goal is marked
where the plan ends. Both were generated by solving the instance and handing the trace to
`render_trace`:

```python
from planiverse.environments.reservoir.environment import ReservoirEnv
from planiverse.benchmark import measures
from planiverse.planners.width import IteratedBFWS

env = ReservoirEnv()
env.set_index(0)
state, info = env.reset()          # info: year, target, reserves, generated
print(state)                       # month 0: upper 40, lower 29; last month released 0, ...

result = IteratedBFWS(max_width=1000, progress=measures.reservoir).solve(env)
trace = env.simulate(result.plan)

env.render_trace(trace, "reservoir_chart.gif")                                   # animated
env.render_trace(trace, "reservoir_chart.png", actions=result.plan, env=env)     # the sheet
```

Stateful play, as opposed to expansion, goes through `step`, which takes an action or its name
(e.g., `"release(6, full)"`). It returns the state and the change in the farm's shortfall, as
before minus after, so zero for a month that rations nothing. `successors(state)` returns the
action and state pairs for all ten actions. The readings each environment charts, and the
panels they go on, are in
[`planiverse/rendering/readings.py`](../../planiverse/rendering/readings.py); `charts=False`
asks `render_trace` for the state's text instead, and [docs/rendering.md](../rendering.md)
covers the other output formats.

## Years

The hundred years are the generator's own draws, embedded in the module as the plain data
`set_instance` takes (`inflow_upper`, `inflow_lower`, `city`, `farm`, `river`, `capacity`,
`initial`, `reserves`, `target`) with the seed each came from beside it. The plan each was
accepted on is in `tests/data/reservoir_solutions.json`.

## Generating years

`generate_instance` draws a year, selects it, and returns it as the dict `set_instance` takes
back:

```python
env = ReservoirEnv()
year = env.generate_instance(seed=7)
print(env.witness, env.witness_expansions)     # the policy it was accepted on, and the policies measured
```

| Option | Default | What it does |
|---|---|---|
| `attempts` | 60 | draws before giving up with `GenerationError` |

Inflows are seasonal shapes with seeded noise; the demands, capacities, initial volumes and
reserves are drawn round them. The fixed policies are then run: each release all year with the
farm on full, each with the farm on half through the summer, and seasonal patterns, high in
summer and low in winter. The target is the least shortfall any of them ends
the year with inside the constraints. A draw is thrown back when none does, or when a steady
release all year does as well as any, since then nothing needs deciding. The same seed always
gives the same year.

## Files

| File | Contents |
|---|---|
| [`environment.py`](../../planiverse/environments/reservoir/environment.py) | `ReservoirAction`, `ReservoirState`, `reference_plans`, `ReservoirEnv`, `YEARS` |
| [`tests/test_operational_four.py`](../../tests/test_operational_four.py) | Tests |
| [`tests/data/reservoir_solutions.json`](../../tests/data/reservoir_solutions.json) | The policy each year was accepted on |
