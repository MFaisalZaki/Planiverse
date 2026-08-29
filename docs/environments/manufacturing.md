# Manufacturing

A capacity-planning problem: buy machine configurations from a market, run them, and meet a
production demand before the clock runs out. Each configuration trades off purchase cost, running
cost, throughput, and setup time, so the plan is a bet about which machines to buy and when to start
paying for them. Adapted from [mfgrl](https://github.com/torayeff/mfgrl).

- **Class:** `MfgEnv`
- **Import:** `from planiverse.environments.manufacturing.mfenv import MfgEnv, ConfigurationAction, ActionType`
- **Source:** [`mfenv.py`](../../planiverse/environments/manufacturing/mfenv.py)
- **Instances:** 7 data files, indices `0`–`6`
- **Dependencies:** `numpy` only

## Quickstart

```python
from planiverse.environments.manufacturing.mfenv import MfgEnv

env = MfgEnv()
env.fix_index(0)
state, info = env.reset()

print(env.DEMAND, env.DEMAND_TIME, env.NUM_CFGS, env.BUFFER_SIZE)
# 1000 100 5 10

for action, successor in env.successors(state):
    print(action, successor._state["demand"], successor._state["demand_time"])
# buy_cfg_0 1000 100
# buy_cfg_1 1000 100
# ...

sorted(state.literals)[:3]
# ['bought(cfg0 false)', 'bought(cfg1 false)', 'bought(cfg2 false)']
```

Only buy actions are available initially: nothing can produce until something is bought.

## Instances

`fix_index(i)` selects the `i`-th data file, **sorted by filename**:

| Index | File | Demand | Demand time | Configs | Buffer |
|---|---|---|---|---|---|
| 0 | `data.json` | 1000 | 100 | 5 | 10 |
| 1 | `data1.json` | 2000 | 150 | 5 | 10 |
| 2 | `data2.json` | 2000 | 80 | 5 | 10 |
| 3 | `data3.json` | 1000 | 100 | 5 | 10 |
| 4 | `data4.json` | 1000 | 48 | 5 | 10 |
| 5 | `data5.json` | 400 | 24 | 5 | 10 |
| 6 | `data6.json` | 24 | 100 | 5 | 10 |

The instances span the difficulty range: `data6.json` (demand 24 in 100 days) is nearly free, while
`data5.json` (400 units in 24 days) and `data2.json` (2000 in 80) are tight.

The index used to come from a bare `os.listdir`, whose order is filesystem-dependent, so "index 0"
named a different instance on different machines and results were not reproducible. If you have
results recorded against the old numbering, they were collected against whatever order that machine
returned; on the machine these docs were written on, index 0 was `data1.json`.

### Data format

```json
{
  "buffer_size": 10,
  "demand": 1000,
  "demand_time": 100,
  "configurations": {
    "cfg_name": {
      "incurring_cost": 100.0,
      "recurring_cost": 5.0,
      "production_rate": 1.5,
      "setup_time": 8.0
    }
  }
}
```

| Field | Meaning |
|---|---|
| `buffer_size` | Machine slots available |
| `demand` | Units to produce |
| `demand_time` | Days available |
| `incurring_cost` | One-off purchase cost |
| `recurring_cost` | Cost per day of running |
| `production_rate` | Units per day once ready |
| `setup_time` | Days until the machine is ready |

`_setup_data` asserts feasibility on load and refuses infeasible instances:

```python
(DEMAND_TIME - SETUP_TIMES[best]) * PRODN_RATES[best] * BUFFER_SIZE > DEMAND
```

i.e. even filling the buffer with the highest-throughput configuration must beat the demand. The
same load step computes `PENALTY_K` (worst-case cost bound), inherited from the RL formulation and
**unused** here.

## State representation

`MfgState` holds a per-configuration dict plus the global `demand` / `demand_time` counters:

| Field | Meaning |
|---|---|
| `bought` | Has this configuration been purchased |
| `incurred_costs` | Purchase cost paid |
| `recurring_costs` | Per-day cost, once bought |
| `production_rates` | Units/day, once bought |
| `setup_times` | Days to ready, once bought |
| `cfgs_status` | Readiness in `[0, 1]`; production only counts at 1 |
| `produced_counts` | Units made by this configuration |
| `market_*` | The market's advertised values (constant, pre-purchase) |

Literals are stringified state variables, lowercased, with `.` escaped to `_`: `bought(cfg0 false)`,
`production_rates(cfg1 1_5)`, `demand(1000)`, `demand_time(100)`. `__eq__` compares literals, so the
literal set *is* the state identity here.

The `.` used to be stripped rather than escaped, which was lossy: a production rate of `1.5` and one
of `15` both rendered as `15`, silently merging two distinct states into one.

## Actions

`ConfigurationAction(cfg_id, action_type, batch_size=-1)` over `ActionType`:

| Action type | Meaning | Offered by `successors`? |
|---|---|---|
| `BUY_CFG` | Purchase configuration `cfg_id` | ✅ for every unbought config |
| `BATCH_PRODUCTION` | Run `cfg_id` for `batch_size` days | ✅ sizes 10, 20, 50, 100 |
| `CONTINUE_PRODUCTION` | Run `cfg_id` for one day | ❌ implemented, commented out |
| `FINISH_PRODUCTION_CFG` | Run `cfg_id` until goal/terminal | ❌ implemented, commented out |
| `FINISH_PRODUCTION_ALL` | Run every config until goal/terminal | ❌ constructed, never appended |

The live action set is then: **buy any unbought configuration**, or **run a bought configuration for
10/20/50/100 days**. `apply_action` still handles all five types, so `simulate` accepts plans using
the unoffered ones; they are disabled in expansion, not removed.

The macro-actions are the point. Single-day stepping makes a 150-day horizon unsearchable; batching
in 10–100 day chunks turns it into a handful of decisions, at the cost of not being able to stop
production mid-batch.

## Transition

**Buy** (`buy_cfg`) copies market values into the live fields, sets `bought`, and initialises
`cfgs_status = 1 / market_setup_times`. It does **not** advance time.

**Produce** (`continue_production`) runs one day:

```python
produced_counts += cfgs_status.astype(int) * production_rates   # only counts when status == 1
cfgs_status = clip(cfgs_status + ceil(cfgs_status) * (1/setup_times + 1e-9), 0, 1)
demand      = DEMAND - sum(produced_counts of this cfg)
demand_time -= 1
```

`.astype(int)` floors readiness, so a machine contributes nothing until fully set up; `ceil` gates
progress so only bought machines advance. `batch_production` loops this `batch_size` times, breaking
early on goal or terminal.

## Goal and terminal

- **Goal** (`is_goal`): `demand_time <= 0`. As the source comment says: *"Let's consider the goal
  state as the state where the demand_time is out."* **The goal is the clock expiring, not the demand
  being met.** Any plan that burns 150 days is a "solution"; whether it produced anything is not
  checked. Quality has to come from your cost function.
- **Terminal** (`is_terminal`): always `False`.

## Known quirks

- **`MfgEnv.__init__` does not call `super().__init__()`**, so `self.name` (from `Environment`)
  is never set. Nothing dispatches on it (the contract check is structural), but code that
  reads `env.name` gets the class attribute default instead of an instance name.
- **`fix_index` must precede `reset()`**: `reset` reads `self.NUM_CFGS` etc., which `_setup_data`
  creates.
- **Rewards are computed and thrown away.** `buy_cfg`/`continue_production` compute a local `reward`
  variable that is never returned, a leftover from the RL original. Cost accounting is the planner's
  job; read it off the state's cost fields.
- **`buffer_size` is loaded but unmodelled.** The multi-machine buffer of the RL original is not
  implemented: each configuration is bought once, not stocked *n* times. It survives only in the
  feasibility assert and `PENALTY_K`.
- **The goal ignores demand.** Running the clock out is a "solution" even if nothing was produced;
  see [Goal and terminal](#goal-and-terminal).

## Fixed

Recorded because they changed how the environment behaves, so results collected before them will not
reproduce:

- **`demand` now aggregates across configurations.** It was computed from the produced count of
  whichever configuration ran last (`DEMAND - sum(produced_counts[cfg_id])`), so output from every
  other machine was invisible to it. Use `total_produced(state._state)` for the shop-floor total.
- **`demand_time` advances one day per day.** `finish_production_all` decremented it once per
  configuration inside its loop, so a single day of running five machines burned five days of clock.
- **Indices are sorted**; see [Instances](#instances).
- **Literals no longer collide**; see [State representation](#state-representation).

## Files

| Path | What |
|---|---|
| [`mfenv.py`](../../planiverse/environments/manufacturing/mfenv.py) | `MfgEnv`, `MfgState`, `ConfigurationAction`, `ActionType` |
| [`data/`](../../planiverse/environments/manufacturing/data) | The 7 instance files |
