# Benchmarking

`planiverse-bench` is the tool paper's evaluation protocol as code. It runs the five planner
configurations the paper compares, once each on every instance of every environment, under the
paper's limits, and turns the results into the paper's tables and figures.

- **Package:** [`planiverse/benchmark/`](../planiverse/benchmark/): the harness, and the progress
  measures SIW and BFWS take per environment.
- **Command:** `planiverse-bench`, installed with the library. `python -m planiverse.benchmark`
  is the same thing.

## Running it

```bash
./setup_benchmark.sh --partition <p> --qos <q>    # builds .venv, installs, runs generate
bash sandbox/submit.sh                            # one sbatch per planner
planiverse-bench report --sandbox-dir sandbox
```

Without a cluster, `bash sandbox/run_local.sh 8` runs the same commands eight at a time. The
harness applies the same limits either way, so the outcomes are comparable; wall-clock times from
a loaded laptop are not comparable with a cluster's.

### `generate`

`planiverse-bench generate [--sandbox-dir sandbox] [--partition P] [--qos Q] [--account A] [--parallel N]`

Builds each registered environment and walks `set_index` upwards until it refuses, which is how
many instances it has. Then it writes:

- `sandbox/tasks.json`, the instance count per environment. `report` reads it to know which runs
  to expect, so a job that never ran is `MISSING` rather than silently absent.
- `sandbox/cmds/<planner>.txt`, one `solve` command per instance. Line *n* is array element *n*,
  so a failed element can be re-run by hand from its line.
- `sandbox/slurm/<planner>.sbatch`, a job array that reads that file by `$SLURM_ARRAY_TASK_ID`,
  throttled to `--parallel` elements at a time (default 50), and given 35 minutes and 9 GB so the
  harness records its own `TIMEOUT` or `MEMOUT` before SLURM steps in.
- `sandbox/submit.sh` and `sandbox/run_local.sh`.

The commands call the interpreter that ran `generate` by absolute path, so the jobs need no
activation and cannot pick up a different install. An environment that cannot be built here (a
missing dependency, or a Game Boy environment with no cartridge) is skipped, and `generate` says
so. Arrays longer than the site's `MaxArraySize` (commonly 1001) are rejected at submission; the
suite's 938 instances fit.

### Cartridges

The Game Boy environments run the original games, which are copyrighted and cannot ship. Export
the path to each before `generate`, and the jobs carry it:

```bash
export PLANIVERSE_PUZZNIC_ROM=...
export PLANIVERSE_FLIPULL_ROM=...
export PLANIVERSE_LOLO_ROM=...
export PLANIVERSE_AMAZING_TATER_ROM=...
export PLANIVERSE_SUPER_MARIO_LAND_ROM=...
```

### `solve`

`planiverse-bench solve [--sandbox-dir sandbox] <planner> <environment>@<index>`

What one array element runs: one planner on one instance, under the limits, written to
`sandbox/results/<planner>/<environment>__<index>.json` whatever happened, with exit code zero
either way. The failure is the result, and a non-zero exit would make SLURM file it among the
infrastructure errors.

## The protocol

| Limit | Value |
|---|---|
| Wall clock | 30 minutes: a search budget checked between expansions, and a hard alarm 2% above it for the expansion that overruns |
| Memory | 8 GB, as an address-space limit, so an overrun is a `MemoryError` the run records |
| Expansions | 500,000 |
| Cores | one per run |
| Solved | only if the returned plan, replayed through `simulate`, reaches a goal |

| Planner | Class | Parameters |
|---|---|---|
| `bfws` | `IteratedBFWS` | `max_width=1000` |
| `iw` | `IteratedWidth` | `max_width=1000, strict=False` |
| `siw` | `SIWSearch` | `width=1, max_width=1000, strict=False` |
| `mcts` | `MCTSPlanner` | `iterations=2000, seed=0`; the exploration constant √2, 30-step rollouts, max backup and the 0.001 length penalty are the class defaults |
| `fsx` | `FSXPlanner` | `horizon=6, walkers=8, seed=0`; the distinct-state count, zero temperature and 200 committed steps are the class defaults |

SIW and BFWS take a `progress(state)` callback in place of the unachieved-goal count a classical
planner would use. [`measures.py`](../planiverse/benchmark/measures.py) supplies one per
environment, lower is better; they are search guides, not admissible heuristics.

## Statuses

Every run ends in exactly one:

| Status | Meaning |
|---|---|
| `SOLVED` | a plan that replays to a goal |
| `INVALID` | a plan that does not: a planner bug, reported as one |
| `UNSOLVED` | the search stopped on its own without a plan |
| `TIMEOUT` | the wall-clock limit ran out first |
| `NODEOUT` | the expansion limit ran out first |
| `MEMOUT` | the memory limit was hit |
| `ERROR` | it raised |
| `UNSUPPORTED` | the environment could not be built |
| `MISSING` | no result file; assigned by `report` |

`UNSOLVED` says the planner stopped looking, not that there is no plan: only BFWS is complete. A
search that reports `out_of_budget` without reaching either limit (an iterated search whose
per-width allowances ran out, or FSX at its step cap or a dead end) is filed as `NODEOUT`.

## `report`

`planiverse-bench report [--sandbox-dir sandbox]` writes into `sandbox/report/`:

- `coverage.tex`: instances solved per environment and planner, in the paper's families and order
  (its Table 2).
- `statuses.tex`: how every run ended, one row per planner, one column per status that occurred,
  and the median solve time (its Table 3). A `MISSING` run is counted as unsolved there, as the
  paper does; `facts.txt` still lists it.
- `cactus.pdf`: each planner's sorted solve times, with its time-outs and memory-outs charged the
  full limit and appended.
- `overlap_bfws_iw_siw.pdf`: one bar per environment, split by which of the three width planners
  solved each instance, ordered by the share all three solved.
- `runtime_bfws_iw_siw.pdf`: BFWS's time per instance against IW (filled, left axis) and SIW
  (hollow, right axis), with failures on the limit.
- `facts.txt`: the numbers the paper's prose quotes: coverage, what each planner solved outside
  BFWS's set, medians, the per-instance speed ratios, errors, IW's widths, plan lengths.

The sandbox behind the paper is `sandbox.zip` on the
[v0.0.1 release](https://github.com/MFaisalZaki/Planiverse/releases). Unzip it beside the
repository and `report` regenerates every number in the paper from it.
