# Benchmarking

`planiverse-bench` is the tool paper's evaluation protocol as code. It runs the five planner
configurations the paper compares on every instance of every environment, under the paper's
limits, with five seeds for the two stochastic planners, and turns the results into the paper's
tables and figures.

- **Package:** [`planiverse/benchmark/`](../planiverse/benchmark/): the harness, and the progress
  measures SIW and BFWS take per environment.
- **Command:** `planiverse-bench`, installed with the library. `python -m planiverse.benchmark`
  is the same thing.

## Running it

```bash
tools/setup_benchmark.sh --partition <p> --qos <q>    # builds .venv, installs, runs generate
bash sandbox/submit.sh                                # one sbatch per job array
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
- `sandbox/cmds/<group>.txt`, one `solve` command per instance, where a group is a planner
  (`bfws`) or one seed of a seeded planner (`mcts-s3`). Line *n* is array element *n*, so a
  failed element can be re-run by hand from its line.
- `sandbox/slurm/<group>.sbatch`, a job array that reads that file by `$SLURM_ARRAY_TASK_ID`,
  throttled to `--parallel` elements at a time (default 50), and given 35 minutes and 9 GB so the
  harness records its own `TIMEOUT` or `MEMOUT` before SLURM steps in. One array per group keeps
  every array one instance set long, under a site's `MaxArraySize`, and finishes seed 0 first.
- `sandbox/submit.sh` and `sandbox/run_local.sh`.

The suite's 938 instances make 13 arrays: three for the width planners and five each for MCTS
and FSX, 12,194 runs. The commands call the interpreter that ran `generate` by absolute path, so
the jobs need no activation and cannot pick up a different install. An environment that cannot be
built here (a missing dependency, or a Game Boy environment with no cartridge) is skipped, and
`generate` says so.

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

`planiverse-bench solve [--sandbox-dir sandbox] <planner> <environment>@<index> [--seed N]`

What one array element runs: one planner on one instance, under the limits, written to
`sandbox/results/<planner>/<environment>__<index>.json`, or `..._<index>__s<seed>.json` for a
seeded planner, whatever happened, with exit code zero either way. The failure is the result, and
a non-zero exit would make SLURM file it among the infrastructure errors. The generated commands
carry `--seed`; a seeded planner run by hand without one gets the first seed.

## The protocol

| Limit | Value |
|---|---|
| Wall clock | 30 minutes: a search budget checked between expansions, and a hard alarm 2% above it for the expansion that overruns |
| Memory | 8 GB, as an address-space limit, so an overrun is a `MemoryError` the run records |
| Expansions | 500,000 |
| Cores | one per run |
| Seeds | 0 to 4 for MCTS and FSX, each (instance, seed) a full run under the limits above; the width planners are deterministic and run once |
| Solved | only if the returned plan, replayed through `simulate`, reaches a goal |

| Planner | Class | Parameters |
|---|---|---|
| `bfws` | `IteratedBFWS` | `max_width=1000` |
| `iw` | `IteratedWidth` | `max_width=1000, strict=False` |
| `siw` | `SIWSearch` | `width=1, max_width=1000, strict=False` |
| `mcts` | `MCTSPlanner` | `iterations=2000`, the run's seed; the exploration constant √2, 30-step rollouts, max backup and the 0.001 length penalty are the class defaults |
| `fsx` | `FSXPlanner` | `horizon=6, walkers=8`, the run's seed; the distinct-state count, zero temperature and 200 committed steps are the class defaults |

SIW and BFWS take a `progress(state)` callback in place of the unachieved-goal count a classical
planner would use. [`measures.py`](../planiverse/benchmark/measures.py) supplies one per
environment, lower is better; they are search guides, not admissible heuristics. The environments
are deterministic, so for MCTS and FSX the seed is the only source of variance.

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

`planiverse-bench report [--sandbox-dir sandbox]` writes into `sandbox/report/`. A seeded planner
is summarised over its seeds: coverage is the mean per seed with the standard deviation in
brackets, never the best seed; solve times are pooled; and the claims about what it solved that
another planner did not use the union over seeds, which is the strongest form of a negative.

- `coverage.tex`: instances solved per environment and planner, in the paper's families and order
  (its Table 2); `65.2 (1.9)` for a seeded planner.
- `statuses.tex`: how every run ended, one row per planner, one column per status that occurred,
  and the median solve time (its Table 3). A seeded planner's counts are means per seed, so its
  row still sums to the instance count. A `MISSING` run is counted as unsolved there, as the paper
  does; `facts.txt` still lists it.
- `cactus.pdf`: each planner's sorted solve times, with its time-outs and memory-outs charged the
  full limit and appended; for a seeded planner the runs are pooled and the count divided by the
  number of seeds, which is the mean curve.
- `overlap_bfws_iw_siw.pdf`: one bar per environment, split by which of the three width planners
  solved each instance, ordered by the share all three solved.
- `runtime_bfws_iw_siw.pdf`: BFWS's time per instance against IW (filled, left axis) and SIW
  (hollow, right axis), with failures on the limit.
- `facts.txt`: the numbers the paper's prose quotes: coverage per seed and in some or every
  seed, what each planner solved outside BFWS's set, medians, the per-instance speed ratios,
  errors and missing runs, IW's widths, plan lengths.

The sandbox behind the paper is `sandbox.zip` on the
[release page](https://github.com/MFaisalZaki/Planiverse/releases). Unzip it beside the
repository and `report` regenerates every number in the paper from it.
