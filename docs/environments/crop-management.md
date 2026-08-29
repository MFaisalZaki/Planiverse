# Crop management (irrigation scheduling)

A potato crop goes in the ground in April and comes out in October. Every ten days through
the season the farmer decides whether to irrigate and by how much. The yield is not the sum
of those decisions; it is the **integral** of a crop growth model driven by the actual
weather. Water applied on day 20 changes the leaf area, which changes how much light the
canopy intercepts for the next eighty days, which changes the tuber weight at harvest.

- **Class:** `CropEnv`
- **Import:** `from planiverse.environments.crop_management.environment import CropEnv`
- **Source:** [`environment.py`](../../planiverse/environments/crop_management/environment.py)
- **Dependencies:** `pcse`. The weather ships inside it and the crop parameters are cached
  locally by PCSE itself, so a season runs offline.

## Why this is not a PDDL domain

**An action's effect is invisible when it is taken.** Water applied at the first decision
point changes nothing measurable by the second: the crop has not had time to respond. There
is no add-list to write, because the effect is a change to the trajectory of a differential
equation that only becomes a yield eighty days later.

**And the same action is worth anything from nothing to most of the crop.** One 4 cm
application (the identical action) measured on 1986, changing only which day it lands on:

| Day after sowing | 10 | 20 | 30 | 40 | 50 | 60 | 70 | **80** | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| Yield gain (kg/ha) | +0 | +0 | +214 | +871 | +871 | +1530 | +1791 | **+2544** | +2371 | +1915 |

Early on it does nothing at all: the soil is already at capacity and the crop is tiny, so
the water drains straight through. Its value climbs to a peak around day 80 and then falls
away again: **non-monotone in timing alone**. No delete-list carries that; the effect of the
action is a function of a growth stage which is itself the integral of every decision before
it.

## Quickstart

```python
from planiverse.environments.crop_management.environment import CropEnv, CropAction

env = CropEnv()
env.fix_index(10)                     # the 1986 season
state, info = env.reset()
info["rainfed"], info["reference"], info["target"]
# (6758.3, 9456.3, 9267.2)

plan = env.reference_plan()           # the naive fixed-calendar schedule
env.validate(plan)                    # True — 9456 kg/ha on 8 cm of water
env.close()
```

## Seasons

`fix_index(i)` picks a growing season: the same field, a different year's weather. Twenty-two
of them, 1976–1999. 1990 and 1991 are absent because the bundled weather has gaps.

Both numbers per season are measurements (the rainfed yield, and the yield under the
reference schedule), recorded so a scenario that stops reproducing fails loudly rather than
quietly changing difficulty.

| Year | Rainfed | Reference | Gain | |
|---|---|---|---|---|
| 1976 | 4241 | 5678 | +1436 | drought year |
| 1980 | 13363 | 13363 | **+0** | wet enough on its own |
| 1983 | 4969 | 6749 | +1781 | |
| 1985 | 14957 | 14989 | +31 | wet |
| 1986 | 6758 | 9456 | **+2698** | irrigation worth most |
| 1999 | 10662 | 12811 | +2149 | |

**The spread is the point.** In 1980 the reference schedule gains nothing at all and the
right plan applies no water. In 1986 the same schedule is worth 2698 kg/ha and a plan that
waits loses most of the crop. On the day the first decision is taken those two seasons look
identical: whether irrigation helps depends on weather that has not happened yet. Knowing
when *not* to act is as much a part of this problem as knowing when to.

### Every season has a witness

The target is 98% of what the reference schedule achieves, so **the reference schedule is a
solution by construction** and no instance ships whose goal nobody has reached. It is also
a baseline worth beating: a fixed calendar of 2 cm on days 20, 40, 60 and 80, ignoring the
weather entirely, which in a wet year spends the whole budget for nothing.

## Actions, goal, and shape

Ten decision points, ten days apart, from day 10 to day 100. At each: wait, or apply 1, 2 or
4 cm, subject to an 8 cm budget for the season. **Cost is the water**, so waiting is free,
which is what makes doing nothing a real decision rather than a filler action.

- **Goal**: harvested, at or above the target yield, within the water budget.
- **Terminal**: the season ended short of the target, or the budget is already overspent.

**No part-grown state is ever a goal.** The yield only exists at harvest, so the whole season
must be planned before the outcome is visible. That is a finite-horizon problem of fixed
depth 10 and branching up to 4 (about a million schedules) where the objective is only
observable at the leaves. That makes it a different shape from the other two simulator
environments here: the [water network](water-distribution.md) has cheap goal tests at every
node, and the [power grid](power-grid.md) is wide and shallow. This one is narrow and deep,
and blind until the end.

## Determinism

Verified rather than assumed: the same schedule in the same year produces the same yield to
the last digit, so the schedule tuple is the whole state. `__eq__` and `__hash__` are
on it, results memoise on it, and `simulate` replaying from scratch is an independent check
on `successors`.

States hold the schedule rather than a model snapshot because a PCSE model carries its whole
output history; replaying is cheap (a full season is under half a second) and exactly
reproducible.

## Attribution

Built on [PCSE](https://github.com/ajwdewit/pcse), Wageningen University's Python
implementation of the WOFOST crop model, with the CABO weather files it ships (Netherlands,
1976–1999).

## Files

| Path | What |
|---|---|
| [`environment.py`](../../planiverse/environments/crop_management/environment.py) | `CropEnv`, `CropState`, `CropAction` |
| [`tests/test_crop_management.py`](../../tests/test_crop_management.py) | Tests, including the timing-response result above |
