# Epidemic control

Pick a public-health policy — how hard to push vaccination, masks, school and workplace closures —
and live with what the epidemic does over the following year. The environment wraps
[EpiPolicy](https://github.com/huda-lab/RL-Epidemic-Benchmark), a compartmental epidemic simulator,
which is vendored under `epidemic_control/epipolicy/`.

- **Class:** `EpiEnv`
- **Import:** `from planiverse.problems.real_world_problems.epidemic_control.environment import EpiEnv`
- **Source:** [`environment.py`](../../planiverse/problems/real_world_problems/epidemic_control/environment.py)
- **Instances:** 7 scenarios, indices `0`–`6`
- **Dependencies:** `numba`, `dill`, `numpy`, `gym` (EpiPolicy JIT-compiles its inner loop)

This is the most heavily *abstracted* environment in the repo. A compartmental model is continuous
and never revisits a state, so search over it doesn't terminate without help. Almost every design
decision below — the 7-day step, the 3-level discretisation, the fuzzy state equality — exists to
carve a searchable graph out of an ODE. Read the [caveats](#state-abstraction-read-this) before
trusting any result.

## Quickstart

```python
from planiverse.problems.real_world_problems.epidemic_control.environment import EpiEnv

env = EpiEnv(delay_vaccination_time=30, horizon=364)   # days
env.fix_index(0)              # COVID_A
state, info = env.reset()

print(state)
# EpiState(depth=0, D=0, E=..., Hcri=..., ..., S=..., V=...)

for action, successor in env.successors(state):
    print(action, successor.depth)
# Vaccination = 0.0 ^ Masks = 0.5   7
# ...

trace = env.simulate([action for action, _ in env.successors(state)][:1])
```

**Constructor arguments**

| Argument | Meaning |
|---|---|
| `delay_vaccination_time` | Day before which vaccination is unavailable (models rollout lag) |
| `horizon` | Day count that defines the goal — search ends when `depth >= horizon` |

## Scenarios

`fix_index(i)` loads the `i`-th `.json` in [`jsons/`](../../planiverse/problems/real_world_problems/epidemic_control/jsons),
**sorted by filename**:

| Index | Scenario | Compartments | Locales | Interventions |
|---|---|---|---|---|
| 0 | `COVID_A` | S, E, Ipre, Iasym, Imild, Hsev, Hcri, D, R, Qs, Qe, Qpre, Qasym, Hmild, V | 4 | Vaccination, Masks |
| 1 | `COVID_B` | as COVID_A | 4 | + School closure, Workplace closure |
| 2 | `COVID_C` | as COVID_A | 4 | + Mass screening & contact tracing, Border closures |
| 3 | `SIRV_A` | S, I, R, V1, V2 | 1 | Vaccination, Masks |
| 4 | `SIRV_B` | S, I, R, V1, V2 | 1 | + School closure, Workplace closure |
| 5 | `SIR_A` | S, I, R | 1 | Vaccination, Masks |
| 6 | `SIR_B` | S, I, R | 1 | + School closure, Workplace closure |

Difficulty rises with both the model (SIR → SIRV → COVID) and the intervention count, which drives
the branching factor. `warmup_scenario.json.ignore` is skipped — the filter matches `.json` only.

The multi-locale COVID scenarios simulate `UnitedProvinces` and its three sub-regions (`Beaches`,
`Hills`, `Pastures`) with mobility between them. Interventions are applied with locale regex `*`,
i.e. everywhere at once.

## Actions

Each intervention exposes one **control parameter** — `compliance`, `degree`, or `percentage`
depending on the intervention — plus cost parameters. The action space is built in `__reset__`:

1. **Discretise.** Each intervention's control parameter is split into `itv_split = 3` levels via
   `np.linspace(min_value, max_value, 3)` — typically 0.0 / 0.5 / 1.0.
2. **Combine.** Take the Cartesian product across interventions, so an action sets *every* lever at
   once. Drop the first (all-minimum) combination.
3. **Filter.** Keep only combinations where at least half the interventions are set to zero
   (`count(0) >= len(basic_interventions)/2`). This is a deliberate sparsity bias — it keeps the
   branching factor down and reflects that real policy rarely pulls every lever simultaneously.
4. **Mask vaccination.** Before `delay_vaccination_time`, `__disable_vaccination__` strips the
   Vaccination lever out of the action entirely.

An action is a list of `EpiAction` objects (plus the scenario's `EpiCost` objects, appended
automatically). `successors` returns them wrapped in `EpiAppliedInterventions`, whose `.action`
property is the flat list `simulate` expects:

```python
str(action)
# 'Vaccination = 0.5 ^ Masks = 1.0'
```

`successors` deduplicates by action string — necessary because once vaccination is masked out, many
distinct combinations collapse onto the same effective action — and skips all-zero actions and
successors that compare equal to the parent.

## Transition

Actions run for a **`PERIOD = 7` day** block: policy is set weekly, not daily, which shrinks the
horizon from 364 decisions to 52. Interventions with a zero control value are dropped before
execution (`cpv_list[0] > 0`) purely to save simulation time.

```python
next_state, _ = self.epi.get_next_state(state.state, _execute_action)
for i in range(1, PERIOD+1):
    next_state, _ = self.epi.get_next_state(next_state, _execute_action)
return EpiState(next_state, state.depth + PERIOD, self.epi.static)
```

Note the loop runs `PERIOD` times *after* the first call — **8 daily steps, while `depth` advances by
7**. Simulated time therefore runs ~14% ahead of `depth`, and `depth` is what the goal test uses.
This is a bug, not a modelling choice.

## State abstraction (read this)

`EpiState` wraps EpiPolicy's state, a `depth` (days elapsed), and the static model description.
Three things make it unlike the other environments:

**Literals are a single string.** Rather than one literal per fact, `__update__` joins every
compartment's population, the depth, and the state's hash into *one* literal:

```
's(9959) ^ i(41) ^ r(0) ^ depth(0) ^ hash ^ 1234567890'
```

Since the hash is derived from the literals and depth, this makes each literal effectively unique.
Width-based planners that consume literals as atoms have nothing useful to work with here.

**Equality is approximate, and loosens over time.** `__eq__` vectorises only the `I` and `R`
compartments, then compares by L1 distance against a threshold that grows with depth:

| Depth | Threshold |
|---|---|
| ≤ 20 | 50 |
| > 20 | 60 |
| > 30 | 65 |
| > 60 | 75 |
| > 90 | 85 |
| > 120 | 95 |
| > 150 | 105 |
| > 180 | 115 |

So two states are "the same" if their infected and recovered counts are within ~50 people early on,
widening later as trajectories naturally spread. This is what makes the state space finite enough to
search — and it is lossy by construction. Two states that differ in *susceptible* population, or in
per-locale distribution, are indistinguishable. Equality is also **not transitive** (a ≈ b and b ≈ c
does not give a ≈ c), so `successors`' `if successor_state == state: continue` prunes on a relation
that isn't an equivalence. Tune the thresholds for your scenario's population size rather than
assuming they transfer.

**`__hash__` and `__eq__` disagree.** `__hash__` uses `(literals, depth)`, which is exact;
`__eq__` is the fuzzy L1 test. Two "equal" states will hash differently, so `EpiState` in a `set` or
`dict` behaves as if equality were exact.

`__repr__` prints the compartment breakdown, which is the most useful view when debugging:

```
EpiState(depth=7, D=0, E=12, ..., S=9959, V=0)
```

## Goal and terminal

- **Goal** (`is_goal`) — `state.depth >= horizon`. The goal is *surviving the horizon*, not
  eradicating the disease. Every path of the right length is a "solution"; quality lives in the cost
  of the interventions applied, not in the goal test.
- **Terminal** (`is_terminal`) — always `False`.

Both methods have dead code after the `return` (an infected-count goal and an all-infected terminal
check) left from earlier iterations.

## Known quirks

- **8 simulated days per 7-day step** — see [Transition](#transition).
- **The filter reads `cpv_list[0]`, not the control parameter.** Both the sparsity filter and the
  zero-skip in `successors`/`__perform_action__` index `cpv_list[0]` — the *first* control parameter
  — while `EpiAction` carefully locates the real one and stores it at `control_parameter_index`. In
  all 7 bundled scenarios the control parameter happens to be listed first (`degree` for
  Vaccination, `compliance` for Masks, `percentage` for the closures), so the two agree and the code
  works. A new scenario that lists, say, `cost_per_day` first would silently filter on a cost value
  instead of the policy level. Use `control_parameter_index` if you touch this code.
- **Dead code in `__reset__`.** The intervention-rewriting block (`updated_interventions`,
  `updated_optimze_interventions`) is computed then discarded — the lines that would apply it to the
  session are commented out. `session['schedules']` *is* cleared, which matters: any schedule baked
  into the scenario JSON is dropped so the planner has full control.
- **`reset()` re-parses and re-JITs the scenario** through `construct_epidemic`. The first call is
  slow (numba compilation); repeated calls stay slow.
- **Branching factor explodes with interventions.** COVID_C has 6 interventions × 3 levels = 729
  combinations before filtering. Combined with 8 ODE steps per expansion, expansion is the
  bottleneck — start with SIR_A.
- **`EpiEnv.__init__` never sets `self.scenario`.** It is created by `fix_index`, so calling
  `reset()` first raises `AttributeError`.

## Files

| Path | What |
|---|---|
| [`environment.py`](../../planiverse/problems/real_world_problems/epidemic_control/environment.py) | `EpiEnv`, `EpiState`, `EpiAction`, `EpiCost`, `EpiAppliedInterventions` |
| [`jsons/`](../../planiverse/problems/real_world_problems/epidemic_control/jsons) | The 7 scenario definitions |
| `epipolicy/` | Vendored EpiPolicy simulator (`core/`, `obj/`, `parser/`, `matrix/`, `optimizer/`, `utility/`) |

Inside `epipolicy/`, the entry points this environment uses are `core.epidemic.construct_epidemic`
(builds the model from a session dict) and its `get_next_state` / `reset` methods. The `optimizer/`
subpackage (PPO, SAC) is EpiPolicy's own RL tooling and is unused by Planiverse. `deprecated/`
subdirectories throughout are upstream leftovers.
