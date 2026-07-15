# Manufacturing

A capacity-planning problem: buy machine configurations from a market, run them, and meet a
production demand before the clock runs out. Each configuration trades off purchase cost, running
cost, throughput, and setup time — so the plan is a bet about which machines to buy and when to start
paying for them. Adapted from [mfgrl](https://github.com/torayeff/mfgrl).

- **Class:** `MfgEnv`
- **Import:** `from planiverse.problems.real_world_problems.manufacturing_environment.mfenv import MfgEnv, ConfigurationAction, ActionType`
- **Source:** [`mfenv.py`](../../planiverse/problems/real_world_problems/manufacturing_environment/mfenv.py)
- **Instances:** 7 data files, indices `0`–`6`
- **Dependencies:** `numpy` only

## Quickstart

```python
from planiverse.problems.real_world_problems.manufacturing_environment.mfenv import MfgEnv

env = MfgEnv()
env.fix_index(0)
state, info = env.reset()

print(env.DEMAND, env.DEMAND_TIME, env.NUM_CFGS, env.BUFFER_SIZE)
# 2000 150 5 10

for action, successor in env.successors(state):
    print(action, successor._state["demand"], successor._state["demand_time"])
# buy_cfg_0 2000 150
# buy_cfg_1 2000 150
# ...

sorted(state.literals)[:3]
# ['bought(cfg0 false)', 'bought(cfg1 false)', 'bought(cfg2 false)']
```

Only buy actions are available initially — nothing can produce until something is bought.

## Instances

⚠️ **Indices are filesystem-ordered, not sorted.** `_load_setup_datafiles` builds `data_index` from a
bare `os.listdir(data_dir)`, so the index→file mapping depends on directory order and is **not
guaranteed stable across machines**. On the machine where these docs were written:

| Index | File | Demand | Demand time | Configs | Buffer |
|---|---|---|---|---|---|
| 0 | `data1.json` | 2000 | 150 | 5 | 10 |
| 1 | `data6.json` | 24 | 100 | 5 | 10 |
| 2 | `data.json` | 1000 | 100 | 5 | 10 |
| 3 | `data4.json` | 1000 | 48 | 5 | 10 |
| 4 | `data5.json` | 400 | 24 | 5 | 10 |
| 5 | `data2.json` | 2000 | 80 | 5 | 10 |
| 6 | `data3.json` | 1000 | 100 | 5 | 10 |

Always confirm with `print(env.data_index)` before quoting an index in results. Sorting that
`os.listdir` would make indices reproducible and is worth doing.

The instances span the difficulty range: `data6.json` (demand 24 in 100 days) is nearly free, while
`data5.json` (400 units in 24 days) and `data2.json` (2000 in 80) are tight.

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

i.e. even filling the buffer with the highest-throughput configuration must beat the demand. It also
computes `PENALTY_K` (worst-case cost bound), inherited from the RL formulation and **unused** here.

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

Literals are stringified state variables with dots stripped and lowercased — `bought(cfg0 false)`,
`production_rates(cfg1 15)`, `demand(1000)`, `demand_time(100)`. `__eq__` compares literals, so the
literal set *is* the state identity here.

Note the punctuation stripping is lossy: `production_rate` 1.5 and 15 both render as `15`. Distinct
states can therefore collapse into one. It hasn't bitten the bundled data, but it is a sharp edge.

## Actions

`ConfigurationAction(cfg_id, action_type, batch_size=-1)` over `ActionType`:

| Action type | Meaning | Offered by `successors`? |
|---|---|---|
| `BUY_CFG` | Purchase configuration `cfg_id` | ✅ for every unbought config |
| `BATCH_PRODUCTION` | Run `cfg_id` for `batch_size` days | ✅ sizes 10, 20, 50, 100 |
| `CONTINUE_PRODUCTION` | Run `cfg_id` for one day | ❌ implemented, commented out |
| `FINISH_PRODUCTION_CFG` | Run `cfg_id` until goal/terminal | ❌ implemented, commented out |
| `FINISH_PRODUCTION_ALL` | Run every config until goal/terminal | ❌ constructed, never appended |

So the live action set is: **buy any unbought configuration**, or **run a bought configuration for
10/20/50/100 days**. `apply_action` still handles all five types, so `simulate` accepts plans using
the unoffered ones — they're disabled in expansion, not removed.

The macro-actions are the point. Single-day stepping makes a 150-day horizon unsearchable; batching
in 10–100 day chunks turns it into a handful of decisions, at the cost of not being able to stop
production mid-batch.

## Transition

**Buy** (`buy_cfg`) — copies market values into the live fields, sets `bought`, and initialises
`cfgs_status = 1 / market_setup_times`. It does **not** advance time.

**Produce** (`continue_production`) — one day:

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

- **Goal** (`is_goal`) — `demand_time <= 0`. As the source comment says: *"Let's consider the goal
  state as the state where the demand_time is out."* **The goal is the clock expiring, not the demand
  being met.** Any plan that burns 150 days is a "solution"; whether it produced anything is not
  checked. Quality has to come from your cost function.
- **Terminal** (`is_terminal`) — always `False`.

## Known quirks

- **`demand` doesn't aggregate across configurations.** `continue_production` computes
  `demand = DEMAND - sum(produced_counts[cfg_id])` — the units made by *that one* configuration.
  Producing on cfg 0 then cfg 1 leaves `demand` reflecting only cfg 1's output; `finish_production_all`
  loops over configs and overwrites `demand` each time, so it ends up showing the last config's
  contribution. Any goal test or heuristic reading `demand` is reading a per-config number, not
  total production. Sum `produced_counts` across configs yourself.
- **`demand_time` decrements per configuration too.** `finish_production_all` decrements
  `demand_time` once per config inside its loop, so a single "day" of running 5 configs burns 5 days
  of clock.
- **Indices are `os.listdir`-ordered** — see [Instances](#instances).
- **`MfgEnv.__init__` doesn't call `super().__init__()`**, so `self.name` (from `RealWorldProblem`)
  is never set. `Simulator`'s `isinstance` dispatch still works.
- **`fix_index` must precede `reset()`** — `reset` reads `self.NUM_CFGS` etc., which `_setup_data`
  creates.
- **Rewards are computed and thrown away.** `buy_cfg`/`continue_production` compute a local `reward`
  variable that is never returned — a leftover from the RL original. Cost accounting is the planner's
  job; read it off the state's cost fields.
- **`buffer_size` is loaded but unmodelled.** The multi-machine buffer of the RL original is not
  implemented — each configuration is bought once, not stocked *n* times. It survives only in the
  feasibility assert and `PENALTY_K`.

## Files

| Path | What |
|---|---|
| [`mfenv.py`](../../planiverse/problems/real_world_problems/manufacturing_environment/mfenv.py) | `MfgEnv`, `MfgState`, `ConfigurationAction`, `ActionType` |
| [`data/`](../../planiverse/problems/real_world_problems/manufacturing_environment/data) | The 7 instance files |
