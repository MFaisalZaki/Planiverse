# Benchmarking

`planiverse-bench` is the library's evaluation protocol as code. It runs the four reference
planners the library has, twenty-five configurations, on every instance of every
environment, under fixed limits, and turns the results into tables and figures.

- **Package:** [`planiverse/benchmark/`](../planiverse/benchmark/): the benchmark, and the progress
  measures the heuristic-guided planners take per environment.
- **Command:** `planiverse-bench`, installed with the library. `python -m planiverse.benchmark`
  is the same thing.

## Running it

```bash
tools/setup_benchmark.sh --partition <p> --qos <q>    # builds .venv, installs, runs generate
bash sandbox/submit.sh                                # one sbatch per job array
planiverse-bench report --sandbox-dir sandbox
```

Without a cluster, `bash sandbox/run_local.sh 8` runs the same commands eight at a time. The
benchmark applies the same limits either way, so the outcomes are comparable; wall-clock times from
a loaded laptop are not comparable with a cluster's.

### `generate`

`planiverse-bench generate [--sandbox-dir sandbox] [--partition P] [--qos Q] [--account A] [--parallel N]`

Builds each registered environment and walks `set_index` upwards until it refuses, which is how
many instances it has. Then it writes:

- `sandbox/tasks.json`, the instance count per environment. `report` reads it to know which runs
  to expect, so a job that never ran is `MISSING` rather than silently absent.
- `sandbox/cmds/<group>.txt`, one `solve` command per instance, where a group is a planner
  (`bfws`). Line *n* is array element *n*, so a failed element can be re-run by hand from its
  line. SLURM caps an array at the site's
  `MaxArraySize`, commonly 1001 elements and often less, so a group over more than 1,000
  instances is cut into parts of 1,000, `<group>-p0.txt`, `<group>-p1.txt`, …, each with its
  own array; with the suite's 2,000 instances every group is two parts.
- `sandbox/slurm/<group>.sbatch` (or `<group>-p<k>.sbatch`), a job array that reads that file by
  `$SLURM_ARRAY_TASK_ID`, throttled to `--parallel` elements at a time (default 50), and given
  35 minutes and 9 GB so the benchmark records its own `TIMEOUT` or `MEMOUT` before SLURM steps
  in.
- `sandbox/submit.sh` and `sandbox/run_local.sh`.

The suite's 2,000 instances, a hundred in each of the twenty environments the report covers,
make one group per planner: 25 groups, 50,000 runs in 50 arrays of 1,000. The commands
call the interpreter that ran `generate` by absolute path, so the jobs need no activation and
cannot pick up a different install. An environment that cannot be built here (a missing
dependency) is skipped, and
`generate` says so.

The benchmark runs the bundled instances. Each environment can also generate its own (see
[Generating instances](../README.md#generating-instances)); a generated benchmark is a matter of
writing the instances to a file and looping `set_instance` over them, which `generate` does not
do for you.

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
| Runs | one per (planner, instance): every planner and every environment is deterministic |
| Solved | only if the returned plan, replayed through `simulate`, reaches a goal |

The configurations, one per planner and per documented variant of one, are in
[`planners.py`](../planiverse/benchmark/planners.py) under the tags the
[catalogue](planners/catalogue.md) lists; for instance `bfws` is `BFWS(width=1)`, `iw` is
`IW(max_width=1000, strict=False)` and `siw` is `SIW(width=1, max_width=1000, strict=False)`.
Anything a configuration does not name is the class's own default.

SIW and BFWS take a `progress(state)` callback in place of the unachieved-goal count a classical
planner would use, and so does every other planner that takes a heuristic.
[`measures.py`](../planiverse/benchmark/measures.py) supplies one per environment, lower is
better; they are search guides, not admissible heuristics, and nothing in the benchmark is a
reward.

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

`UNSOLVED` says the planner stopped looking, not that there is no plan: most planners here
are incomplete. For an online planner it means its walk ended at a dead end or the step cap.
A search that reports `out_of_budget` without reaching either limit (an iterated search whose
per-width allowances ran out) is filed as `NODEOUT`.

## `report`

`planiverse-bench report [--sandbox-dir sandbox]` writes into `sandbox/report/`, over every
planner on every instance `tasks.json` lists.

- `coverage.tex`: instances solved per environment and planner, by family.
- `statuses.tex`: how every run ended, one row per planner, one column per status that occurred,
  and the median solve time. A `MISSING` run is counted as unsolved there; `facts.txt` still
  lists it.
- `cactus.pdf`: each planner's sorted solve times, with its time-outs and memory-outs charged the
  full limit and appended.
- `overlap_bfws_iw_siw.pdf`: one bar per environment, split by which of the three width planners
  solved each instance, ordered by the share all three solved.
- `runtime_bfws_iw_siw.pdf`: BFWS's time per instance against IW (filled, left axis) and SIW
  (hollow, right axis), with failures on the limit.
- `facts.txt`: the numbers a write-up would quote: coverage, what each planner solved outside
  BFWS's set, medians, the per-instance speed ratios with their sign tests, errors and
  missing runs, IW's widths, plan lengths, and each planner's coverage per family; then the
  other aggregations (the mean fraction solved over
  environments and the IPC quality score), each planner's plan lengths against BFWS's, every
  planner's statuses per environment, the cost of an expansion per environment, and the
  difficulty profile (open instances, BFWS's plan lengths, successors per expansion, IW's
  largest width).

## The planners

Every planner in the library ([docs/planners/catalogue.md](planners/catalogue.md)) is
registered in `planiverse/benchmark/planners.py` under its own tag: twenty-five
configurations covering every planner class the library exports and the documented variants
of each (A* and weighted A* beside greedy best-first). The three that need more than one
number from an environment get it from the progress measure: FESS searches the feature space
it spans (`fess`), BFWS over boundary-extension features extends its range (`bee`), and
multi-queue alternation pairs it with quantified novelty over it (`multi`). `report`
tabulates every planner on every instance; the overlap and runtime figures are over BFWS, IW
and SIW. `solve` takes any tag, so one run can be made by hand:

```bash
python -m planiverse.benchmark solve --sandbox-dir sandbox ehc puzznic@0
```
