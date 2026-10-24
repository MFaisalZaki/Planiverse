# Benchmarking

`planiverse-bench` is the library's evaluation protocol as code. It runs the four reference
planner configurations (BFWS, IW, SIW and FSX) and the forty-one others, every planner the
library has, on every instance of every environment, under fixed limits, with five
seeds for every planner that takes one. `--reference` keeps `generate` to the four.

The harness is [pyPMTEvalToolkit](https://github.com/pyPMT/pyPMTEvalToolkit)'s: its sandbox
layout, its command files and SLURM job arrays, its local runner, and its `analyze` and
`report` stages. What Planiverse adds is `solve`, the one stage the toolkit cannot do for a
simulator: the toolkit's own parses PDDL and runs a Unified Planning engine, and here there is a
simulator and a planner object instead. `solve` writes the toolkit's result format, so the
toolkit reads the runs as its own.

- **Package:** [`planiverse/benchmark/`](../planiverse/benchmark/): `generate`, `solve` and
  `report`, the planner configurations, and the progress measures the heuristic-guided planners
  take per environment.
- **Command:** `planiverse-bench`, installed with the library. `python -m planiverse.benchmark`
  is the same thing, and the toolkit's `pypmtevalcli analyze` and `pypmtevalcli report` work on
  the same sandbox.

## Running it

```bash
tools/setup_benchmark.sh --partition <p> --qos <q>    # builds .venv, installs, runs generate
bash sandbox/slurm/submit_all.sh                      # one sbatch per job array
planiverse-bench report --sandbox-dir sandbox
```

Without a cluster, `bash sandbox/run_local.sh 8` runs the same commands eight at a time. The
benchmark applies the same limits either way.

### `generate`

`planiverse-bench generate [--sandbox-dir sandbox] [--partition P] [--qos Q] [--account A] [--parallel N] [--reference]`

Builds each registered environment and walks `set_index` upwards until it refuses, which is how
many instances it has. An instance is a toolkit task: suite `planiverse`, domain the environment,
instance its index, track `classical` (see [Statuses](#statuses)). Then, through the toolkit's
writers, it writes:

- `sandbox/tasks.json`, every task and, per planner group, the runs expected of it. The toolkit's
  `analyze` reads it, so a job that never ran is `MISSING` rather than silently absent.
- `sandbox/cmds/<group>.txt`, one `solve` command per instance, where a group is a planner
  (`bfws`) or one seed of a seeded planner (`fsx-s3`). Line *n* is array element *n* of the
  group's arrays, so a failed element can be re-run by hand from its line.
- `sandbox/slurm/pypmteval-<group>.sbatch`, a job array that reads that file by
  `$SLURM_ARRAY_TASK_ID`, throttled to `--parallel` elements at a time (default 50), and given
  35 minutes and 9 GB so the benchmark records its own `TIMEOUT` or `MEMOUT` before SLURM steps
  in. SLURM caps an array at the site's `MaxArraySize`, commonly 1001 elements and often less,
  so a group over more than 1,000 instances becomes several arrays,
  `pypmteval-<group>.1.sbatch` onward, each over its own slice of the command file.
- `sandbox/slurm/submit_all.sh` and `sandbox/run_local.sh`.

Every environment but the two emulators has a hundred instances; the Game Boy has as many as the
cartridge's wrapper offers, and Stable-Retro one per save state of its integration. There is one
group per unseeded planner and five per seeded one: 25 unseeded and 20 seeded planners, so 125
groups, each over every instance. `--reference` writes the four reference planners alone, 8
groups (three for the deterministic width planners and five for FSX). The commands call the
interpreter that ran `generate` by absolute path, so the jobs need no activation and cannot pick
up a different install. An environment that cannot be built here (a missing dependency) is
skipped, and `generate` says so.

The benchmark runs the bundled instances. Each environment can also generate its own (see
[Generating instances](../README.md#generating-instances)), and `generate` does not include
those.

### `solve`

`planiverse-bench solve [--sandbox-dir sandbox] <planner> <environment>@<index> [--seed N]`

What one array element runs: one planner on one instance, under the limits, written to
`sandbox/results/<group>/planiverse_<environment>_<index>.json` whatever happened, with exit code
zero either way. The failure is the result, and a non-zero exit would make SLURM file it among
the infrastructure errors. The file is the toolkit's result format: the task, the planner (its
group tag, its class's tag as `engine`, the seed, the parameters), the limits, the run, the
status, the timings, the plan, and `metrics` with the plan length, whether the plan replayed,
the peak memory, and the search's own counts (expansions, generated, widths tried, rollouts,
episodes). A `.running` marker beside it while the run is under way lets the toolkit tell a job
SLURM killed (`KILLED`) from one that never started (`MISSING`). The generated commands carry
`--seed`; a seeded planner run by hand without one gets the first seed.

## The protocol

| Limit | Value |
|---|---|
| Wall clock | 30 minutes: a search budget checked between expansions, and a hard alarm 2% above it for the expansion that overruns |
| Memory | 8 GB, as an address-space limit, so an overrun is a `MemoryError` the run records |
| Expansions | 500,000 |
| Cores | one per run |
| Seeds | 0 to 4 for FSX and every other planner that takes one, each (instance, seed) a full run under the limits above; BFWS, IW, SIW and the other deterministic planners run once |
| Solved | only if the returned plan, replayed through `simulate`, reaches a goal |

| Planner | Class | Parameters |
|---|---|---|
| `bfws` | `BFWS` | `width=1` |
| `iw` | `IW` | `max_width=1000, strict=False` |
| `siw` | `SIW` | `width=1, max_width=1000, strict=False` |
| `fsx` | `FSXPlanner` | `horizon=6, walkers=8`, the run's seed; the distinct-state count, zero temperature and 200 committed steps are the class defaults |

SIW and BFWS take a `progress(state)` callback in place of the unachieved-goal count a classical
planner would use, and so does every other planner that takes a heuristic.
[`measures.py`](../planiverse/benchmark/measures.py) supplies one per environment, lower is
better; they are search guides, not admissible heuristics, and nothing in the benchmark is a
reward. The environments are deterministic, so for the seeded planners the seed is the only
source of variance.

## Statuses

Every run ends in exactly one of the toolkit's statuses:

| Status | Meaning |
|---|---|
| `SOLVED` | a plan came back; it counts as solved only if `metrics.validated` is true, i.e., it replays through `simulate` to a goal. One that does not is a planner bug, and the toolkit lists it apart from coverage |
| `EXHAUSTED` | the search stopped without a plan: on its own (`metrics.stopped` is `on its own`), or at the expansion limit (`at the expansion cap`) |
| `TIMEOUT` | the wall-clock limit ran out first |
| `MEMOUT` | the memory limit was hit |
| `ERROR` | it raised; the traceback is in the result |
| `UNSUPPORTED` | the environment could not be built |
| `KILLED` | SLURM killed the job before it wrote; assigned by the toolkit's `analyze` |
| `MISSING` | no result file; assigned by the toolkit's `analyze` |

A search that stops on its own says the planner stopped looking, not that there is no plan: of
the reference planners only BFWS is complete. For an online planner it means its walk ended at a
dead end or the step cap. A search that reports `out_of_budget` without reaching either limit
(an iterated search whose per-width allowances ran out) is filed at the expansion cap.

Every environment is filed under the toolkit's `classical` track: each is a classical planning
problem in the toolkit's sense, deterministic, single-agent and with a goal, and the toolkit's
tables lay out the classical, numeric and temporal tracks. The environments themselves are the
domains, so the per-domain tables give them one by one.

## `report`

`planiverse-bench report [--sandbox-dir sandbox] [--no-plots]` runs the toolkit's `analyze` and
`report` over the sandbox, the same as `pypmtevalcli analyze --sandbox-dir sandbox --per-domain`
followed by `pypmtevalcli report --sandbox-dir sandbox`:

- `analysis/results.csv`, one row per expected run, and `analysis/summary.txt` and
  `summary.json`: coverage per planner, the status breakdown, runtimes over solved runs, a
  head-to-head over the instances every planner attempted, and per-domain coverage.
- `report/results.txt`, with `coverage.tex`, `per-domain-coverage.tex` and `outcomes.tex` as
  booktabs LaTeX: coverage, outcomes, the IPC quality and time scores, and per-domain coverage.
- `report/plots/`: survival (cactus) plots over time and memory, coverage and outcome bars, and
  runtime and plan-length scatters per pair of planners, as PDF and PNG.

A seeded planner's seeds are separate groups (`fsx-s0` to `fsx-s4`), so the tables list them
side by side rather than as a mean over seeds. The toolkit's line figures draw the first six
planners and its scatters the first six pairs.

## The other planners

Every planner in the library ([docs/planners/catalogue.md](planners/catalogue.md)) is
registered in `planiverse/benchmark/planners.py` under its own tag, the four reference
configurations and forty-one others covering every planner class the library exports and
the documented variants of each (A* and weighted A* beside greedy best-first, local exploration by walks beside by
search, iterated local search beside annealing, KPIECE beside EST and SST). The four that
need more than one number from an environment get it from the progress measure: FESS
searches the feature space it spans (`fess`), KPIECE projects onto it (`kpiece`), BFWS over
boundary-extension features extends its range (`bee`), and multi-queue alternation pairs it
with FSX's option count (`multi`). `generate` writes their jobs beside the reference planners',
and `report` tabulates every group `tasks.json` expects, so a sandbox generated under
`--reference` reports the four reference planners and nothing else. `solve` takes any tag, so
one run can be made by hand:

```bash
python -m planiverse.benchmark solve --sandbox-dir sandbox goexp puzznic@0 --seed 0
```
