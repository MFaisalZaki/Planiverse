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
- **Dependencies:** `pandas`, `networkx`, `numpy` — ⚠️ **`pandas` and `networkx` are not declared in
  `pyproject.toml`**; install them yourself (`pip install pandas networkx`).

This environment is the least finished of the six. The action model has known bugs
(see [Known quirks](#known-quirks)) that make several actions no-ops on the bundled cities. Read that
section before using it as a benchmark.

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

Six actions, each converting a fixed 5% slice of one land-use class:

| Action | Intent |
|---|---|
| `ConvertEmptyAction` | Develop empty land, split evenly across all five uses |
| `ConvertGreenSpaceAction` | Green space → facilities / commercial |
| `ConvertOfficesAction` | Offices → commercial |
| `ConvertCommercialAction` | Commercial → facilities |
| `ConvertFacilitiesAction` | Facilities → green space / commercial |
| `RemoveResidentialAction` | Residential → empty |

The selected parcels are the **first** `int(count × 0.05)` of that type in graph insertion order (CSV
order) — not sampled, not chosen by any criterion. Expansion is fully deterministic.

`successors` instantiates all six action classes each call and returns
`(action, successor_state)` pairs. Actions stringify as `action_<node>_<newtype>__...`.

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

## Known quirks

These are real bugs, verified against the bundled data. Treat this environment as a work in progress.

- **`ConvertEmptyAction` converts everything to facilities.** It builds its conversion list as *every
  land × every type* and `apply` walks that list assigning each in turn — so the last assignment
  wins, and all selected parcels end up `FACILITIES`. The intended "split empty land evenly between
  r/o/g/c/f" does not happen.
- **The 5%-of-5% slice is almost always empty.** `ConvertGreenSpaceAction` and
  `ConvertFacilitiesAction` take 5% of a type, then split *that* slice with another 5% cut. On these
  city sizes the inner cut rounds to zero, so their first branch never fires: green space converts
  entirely to commercial, never facilities. `ConvertOfficesAction`/`ConvertCommercialAction` also
  slice with `[int(len × 0.05):]`, which drops nothing but is clearly not what was meant.
- **Several actions are silent no-ops.** With `int(count × 0.05)` and the counts above, at the
  initial state:

  | Action | Kendall Square | St Andrews |
  |---|---|---|
  | `ConvertEmpty` | 3 parcels → facilities | **0 — no-op** |
  | `ConvertGreenSpace` | 2 → commercial | 11 → commercial |
  | `ConvertOffices` | 2 → commercial | **0 — no-op** |
  | `ConvertCommercial` | **0 — no-op** | **0 — no-op** (15 × 0.05 = 0) |
  | `ConvertFacilities` | **0 — no-op** (15 × 0.05 = 0) | **0 — no-op** |
  | `RemoveResidential` | 28 → empty | **0 — no-op** (19 × 0.05 = 0) |

  Any type with fewer than 20 parcels can never be converted. St Andrews has exactly **one**
  effective action at reset.
- **No-op actions are still returned as successors.** `successors` filters with
  `if successor_state == state: continue`, but literals include `depth_N` and every successor sits at
  `depth + 1` — so a successor can *never* compare equal to its parent and the filter is dead code.
  The branching factor is a constant 6, four or five of which may only advance the clock. This is the
  one environment where the self-loop filter doesn't work.
- **Actions are stateful and not reusable.** `converted_nodes` is initialised in `__call__`, not
  `__init__`, and `apply` appends to it. So a freshly constructed action passed straight to
  `apply`/`str` raises `AttributeError`, and `simulate` — which calls `action.apply(state)` directly
  — accumulates into `converted_nodes` across calls and leaves `actionstr` stale. Only use actions
  handed to you by `successors`, once.
- **`UrbanPlanningEnv.__init__` doesn't call `super().__init__()`**, so `self.name` is unset;
  `self.statsitics` (sic) is the land-use summary built by `reset`.
- **`fix_index` must precede `reset()`** — `reset` reads `self.node_info` / `self.node_pairs`.

## Files

| Path | What |
|---|---|
| [`environment.py`](../../planiverse/problems/real_world_problems/urban_planning/environment.py) | `UrbanPlanningEnv`, `UrbanEnvState`, the action classes, `LandUseType` |
| [`cities/Kendall_Square_data/`](../../planiverse/problems/real_world_problems/urban_planning/cities/Kendall_Square_data) | Kendall Square parcels and adjacency |
| [`cities/st_andrews_data/`](../../planiverse/problems/real_world_problems/urban_planning/cities/st_andrews_data) | St Andrews parcels and adjacency |
