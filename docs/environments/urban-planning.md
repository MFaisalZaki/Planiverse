# Urban planning

Rezone a city. Parcels of land are nodes in a spatial graph, each with a use — residential, office,
green space, commercial, facilities — and an action converts a slice of one use into another. Over a
horizon of rezoning decisions, the planner shapes the city's sustainability and land-use diversity.

Based on *"AI Agent as Urban Planner: Steering Stakeholder Dynamics in Urban Planning via
Consensus-based Multi-Agent Reinforcement Learning"*.

- **Class:** `UrbanPlanningEnv`
- **Import:** `from planiverse.problems.real_world_problems.urban_planning.environment import UrbanPlanningEnv`
- **Source:** [`environment.py`](../../planiverse/problems/real_world_problems/urban_planning/environment.py)
- **Instances:** 2 cities, indices `0`–`1`
- **Install:** `pip install ".[urban]"`
- **Dependencies:** `pandas`, `networkx`, `numpy`

This is the least finished of the six environments, and the one whose action model involved the most
guesswork to repair — see [Fixed](#fixed) for what changed and why it may not match what the original
author intended.

## Quickstart

```python
from planiverse.problems.real_world_problems.urban_planning.environment import UrbanPlanningEnv

env = UrbanPlanningEnv(horizon=100)      # horizon = number of rezoning steps
env.fix_index(0)                          # Kendall Square
state, info = env.reset()

print(env.urban_name, env.graph.number_of_nodes())
print(state.sustainability_score, state.diversity_score)
print(sorted(state.literals))
# ['c_0', 'depth_0', 'f_15', 'g_54', 'n_60', 'o_53', 'r_567']

for action, successor in env.successors(state):
    print(action, successor.sustainability_score, successor.diversity_score)
```

## Cities

`fix_index(i)` loads a city from [`cities/`](../../planiverse/problems/real_world_problems/urban_planning/cities):

| Index | City | Parcels | Initial land use |
|---|---|---|---|
| 0 | Kendall Square | 749 | 567 residential, 60 empty, 54 green, 53 office, 15 facilities, **0 commercial** |
| 1 | St Andrews | 260 | 223 green, 19 residential, 15 commercial, 3 facilities, **0 empty, 0 office** |

The two are shaped very differently — Kendall Square is a dense residential grid with room to
develop, St Andrews is mostly green space — and both are missing land-use classes entirely, which
interacts badly with the action model below.

### Data format

Each city directory holds two CSVs, in the original dataset's format:

**`node_info.csv`** — one row per parcel:

| Column | Meaning |
|---|---|
| `node_id` | Parcel identifier |
| `landuse_type` | Numeric land-use code (see below) |
| `area_sqm` | Parcel area |
| `longitude`, `latitude` | Position (projected coordinates for Kendall, WGS84 for St Andrews) |

**`node_pairs_knn4.csv`** — the 4-nearest-neighbour spatial graph:

| Column | Meaning |
|---|---|
| `node`, `node_adj` | The two parcels |
| `distance_m` | Distance in metres |

Edges are skipped when either endpoint is absent from `node_info.csv`. Files **must** be named
`node_info.csv` and `node_pairs_knn4.csv`. (`Kendall_Square_data/node_info_cluster13.csv` is an
unused alternative clustering.)

Land-use codes:

| Code | Type | Symbol |
|---|---|---|
| `-1` | Empty / undeveloped (e.g. water) | `n` |
| `0` | Residential | `r` |
| `1` | Office | `o` |
| `2` | Commercial | `c` |
| `3` | Facilities | `f` |
| `4` | Green space | `g` |

## State representation

`UrbanEnvState` wraps a `networkx.Graph` (parcels as nodes carrying `type`, edges as spatial
adjacency) plus a `depth`.

Literals are **aggregate counts, not per-parcel facts**:

```python
sorted(state.literals)
# ['c_0', 'depth_0', 'f_15', 'g_54', 'n_60', 'o_53', 'r_567']
```

`r_567` means "567 residential parcels". The per-parcel alternative
(`land_42_is_r`) is in the source, commented out, with the reason: this abstraction *"speed[s] up the
search for iw by breaking symmetry"*. Which particular parcel got rezoned doesn't matter to the
objective — only how many of each type there are — so collapsing to counts turns a combinatorial
explosion of equivalent states into one. The cost is that spatial structure is invisible to the
planner's state comparison: two cities with identical counts but opposite layouts are the same state,
even though the graph (and any spatially-aware objective) differs.

`__eq__` compares literals, which include `depth_N`.

Each state also computes two scores (see [Objectives](#objectives)).

## Actions

Six actions, each rezoning a 5% slice (`CHANGE_RATIO`) of one land-use class and dealing the selected
parcels round-robin across the target types:

| Action | Converts | Into |
|---|---|---|
| `ConvertEmptyAction` | Empty land | residential / office / green / commercial / facilities |
| `ConvertGreenSpaceAction` | Green space | facilities / commercial |
| `ConvertOfficesAction` | Offices | commercial |
| `ConvertCommercialAction` | Commercial | facilities |
| `ConvertFacilitiesAction` | Facilities | green space / commercial |
| `RemoveResidentialAction` | Residential | empty |

The selected parcels are the **first** `ceil(count × 0.05)` of that type in graph insertion order (CSV
order) — not sampled, not chosen by any criterion. Expansion is fully deterministic. Rounding up means
a class with fewer than 20 parcels is still rezonable, one parcel at a time.

`successors` instantiates all six action classes each call and returns `(action, successor_state)`
pairs, **skipping any action whose land-use class is absent** from the city. Actions stringify as
`action_<node>_<newtype>__...`.

## Objectives

Neither score is used by `is_goal` — they exist for a planner's heuristic or cost function to
optimise. This is where the actual problem lives:

**Sustainability** — the share of parcels that are green, commercial, or facilities:

```python
(#green + #commercial + #facilities) / (parcels that aren't empty)
```

Note the formula in the source comment says `(#g + #c) / total`, while the code also counts
facilities. Rounded to 1 decimal place.

**Diversity** — normalised Shannon entropy over the five non-empty land-use counts, `0` (monoculture)
to `1` (even mix). Rounded to 1 decimal place.

The rounding is aggressive: both scores step in units of 0.1, so a heuristic reading them gets a very
coarse signal and many actions won't move them at all.

## Goal and terminal

- **Goal** (`is_goal`) — `state.depth >= horizon`. As with the epidemic environment, the goal is
  running out the clock; plan *quality* must come from the scores above.
- **Terminal** (`is_terminal`) — always `False`.

## Effective actions per city

With `ceil(count × 0.05)` over the initial land use, and no-op actions filtered out:

| Action | Kendall Square | St Andrews |
|---|---|---|
| `ConvertEmpty` | 3 parcels → 1 residential, 1 office, 1 green | *not offered* (no empty land) |
| `ConvertGreenSpace` | 3 → 2 facilities, 1 commercial | 12 → 6 facilities, 6 commercial |
| `ConvertOffices` | 3 → commercial | *not offered* (no offices) |
| `ConvertCommercial` | *not offered* (no commercial) | 1 → facilities |
| `ConvertFacilities` | 1 → green space | 1 → green space |
| `RemoveResidential` | 29 → empty | 1 → empty |

So the branching factor is 5 for Kendall Square and 4 for St Andrews.

## Known quirks

- **`UrbanPlanningEnv.__init__` doesn't call `super().__init__()`**, so `self.name` is unset;
  `self.statsitics` (sic) is the land-use summary built by `reset`.
- **`fix_index` must precede `reset()`** — `reset` reads `self.node_info` / `self.node_pairs`.
- **The goal ignores the scores.** Reaching the horizon is the only goal test; sustainability and
  diversity are there for a planner's heuristic to optimise, not for the environment to check.
- **Score rounding is coarse.** Both scores round to 1 decimal, so they step in units of 0.1 and many
  single actions won't move them at all.

## Fixed

These were real bugs, verified against the bundled data. They changed how the environment behaves, so
earlier results will not reproduce.

- **`ConvertEmptyAction` converted everything to facilities.** It built its conversion list as *every
  land × every type*, and `apply` walked that list assigning each in turn, so the last assignment
  won. Its stated intent — "split all of the empty spaces evenly between r, o, g, c, f" — never
  happened. It now deals parcels round-robin across the five uses.
- **The 5%-of-5% slice was almost always empty.** `ConvertGreenSpaceAction` and
  `ConvertFacilitiesAction` took 5% of a class and then split *that* slice with another 5% cut, which
  rounded to zero on these city sizes — so green space only ever became commercial, never facilities.
  The slice is now a single cut, split evenly across the target types.
- **Classes under 20 parcels were frozen forever.** `int(15 × 0.05)` truncates to 0, so Kendall's 15
  facilities and St Andrews' 19 residential parcels could never be rezoned, and St Andrews had
  exactly **one** effective action. Selection now rounds up.
- **No-op actions were offered as successors.** The filter was `if successor_state == state`, but
  literals include `depth_N` and every successor sits at `depth + 1`, so a successor could never
  compare equal to its parent and the filter was dead code. `successors` now drops actions that
  rezoned nothing (`action.converted_nodes` is empty).

  Note the fix keeps `depth_N` in the literals rather than removing it: the goal test is
  `depth >= horizon`, so a planner keying its visited set on literals needs depth there — without it,
  a state at depth 3 would prune a state with the same land-use counts at depth 99 and the horizon
  could become unreachable.
- **Actions were stateful and not reusable.** `converted_nodes` was created in `__call__`, so a fresh
  action passed to `apply`/`str` raised `AttributeError` — which is exactly what `simulate` does —
  and repeated `apply` calls accumulated into it while leaving `actionstr` stale. Both are now set up
  in `__init__` and recomputed per `apply`.

### A judgment call worth reviewing

The **split proportions** were ambiguous in the original. Dead comments variously described "80% to
be g and 20% to be commercial" and "split evenly", while the code did neither. Each action now splits
its slice **evenly** across its target types, which is what the class comments say most often. If the
paper this is based on specifies particular proportions, that belongs in `__split_evenly__`'s
callers.

## Files

| Path | What |
|---|---|
| [`environment.py`](../../planiverse/problems/real_world_problems/urban_planning/environment.py) | `UrbanPlanningEnv`, `UrbanEnvState`, the action classes, `LandUseType` |
| [`cities/Kendall_Square_data/`](../../planiverse/problems/real_world_problems/urban_planning/cities/Kendall_Square_data) | Kendall Square parcels and adjacency |
| [`cities/st_andrews_data/`](../../planiverse/problems/real_world_problems/urban_planning/cities/st_andrews_data) | St Andrews parcels and adjacency |
