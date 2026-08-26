# Benchmarking

`planiverse-bench` runs every planner in the library over every environment, on a SLURM cluster
or on one machine, and turns the results into tables and plots. It is modelled on
[pyPMTEvalToolkit](https://github.com/pyPMT/pyPMTEvalToolkit): an experiment is a directory of
JSON, a sandbox is a directory of results, and the stages between them run independently.

- **Package:** [`planiverse/benchmark/`](../planiverse/benchmark/)
- **Command:** `planiverse-bench` (installed with the library)
- **Dependencies:** none beyond the library. Plots need matplotlib, which is already a
  dependency; without it the tables are still written.

## The five minute version

```bash
planiverse-bench init      --exp-dir experiment
planiverse-bench discover  --exp-dir experiment --sandbox-dir sandbox
planiverse-bench generate  --exp-dir experiment --sandbox-dir sandbox
bash sandbox/slurm/submit_all.sh          # or: bash sandbox/run_local.sh 8
planiverse-bench analyze   --sandbox-dir sandbox
planiverse-bench report    --sandbox-dir sandbox
```

`init` writes a default experiment; `discover` resolves which `(environment, index)` pairs
exist here; `generate` writes the commands and the SLURM job arrays; `analyze` collects the
result files into tables and a CSV; `report` adds LaTeX and plots. `solve` is the one stage you
do not normally type — it is what each array element runs.

## Why the stages are separate

Each writes files the next one reads, so no stage needs another to be running. That is what
lets a benchmark be prepared on a laptop, run on a cluster, and analysed somewhere else again —
and it is what makes a run reproducible, because the sandbox records which experiment produced
it and a result can always be traced back to the limits it was obtained under.

## The experiment directory

```
experiment/
├── exp-details.json          limits, task selection, SLURM directives
└── planners/
    ├── bfws-2.json
    ├── iw-1.json
    └── ...
```

`exp-details.json`:

```json
{
    "name": "planiverse-bench",
    "limits": {
        "time": "30m",
        "memory": "8GB",
        "max-expansions": 100000,
        "validate-plans": true
    },
    "tasks": {
        "tags": [],
        "include-environments": [],
        "exclude-environments": [],
        "max-instances-per-environment": 10,
        "selection": "even",
        "include-rom-environments": false,
        "selected-tasks": []
    },
    "slurm": {
        "cpus-per-task": 1,
        "partition": null,
        "account": null,
        "max-parallel-jobs": 50,
        "max-array-size": 1000,
        "time-headroom": "00:05:00",
        "memory-headroom": "1GB",
        "extra-directives": [],
        "setup-commands": []
    }
}
```

Durations take either spelling — `"30m"` or `"00:30:00"` — because SLURM writes one and people
write the other. Sizes take `"8GB"`. Unknown keys are ignored, so a config written for a later
version still loads.

A planner file:

```json
{
    "tag": "bfws-2",
    "planner": "bfws",
    "params": { "width": 2 },
    "tags": [],
    "exclude-environments": [],
    "enabled": true
}
```

`tag` names the planner everywhere afterwards — result filenames, sbatch job name, every table
— so it must be filesystem-safe and stable. `planner` is a name from the catalogue
(`planiverse-bench planners` lists them). A misspelled parameter is **refused**, not ignored: a
benchmark that quietly drops `"widht": 2` reports a width-1 result under a width-2 name and
nothing downstream can tell.

`tags` and `exclude-environments` narrow one planner's task list, so a single experiment can run
a cheap planner over everything and an expensive one over a subset.

## The sandbox

```
sandbox/
├── tasks.json                    the resolved task list and planner-task pairs
├── cmds/<planner>.txt            one command per line
├── slurm/
│   ├── <name>-<planner>.sbatch   a job array per planner
│   └── submit_all.sh
├── run_local.sh                  the no-cluster path
├── logs/<planner>/               SLURM stdout and stderr
├── results/<planner>/<task>.json one file per run
├── analysis/                     results.csv, summary.json
└── report/                       results.txt, coverage.tex, report.json, plots
```

## SLURM

One **job array per planner**, not one job per run. A benchmark is thousands of short runs, and
a scheduler handling them as thousands of jobs spends longer scheduling than computing. Line
*n* of `cmds/<planner>.txt` is array index *n*, and nothing re-derives that ordering — which is
what lets a failed element be re-run by hand.

```bash
#SBATCH --job-name=planiverse-bench-bfws-2
#SBATCH --array=0-15%50
#SBATCH --time=00:35:00
#SBATCH --mem=9216M
...
COMMANDS=/abs/path/sandbox/cmds/bfws-2.txt
INDEX=$(( ${SLURM_ARRAY_TASK_ID:-0} + OFFSET ))
COMMAND=$(sed -n "$(( INDEX + 1 ))p" "$COMMANDS")
eval "$COMMAND"
```

Three things a generated array has to get right, none of them obvious the first time:

- **Array size.** `MaxArraySize` is a site limit, commonly 1001, and an array over it is
  rejected at submission with a message that does not name the cause. Long lists are split
  across several `sbatch` files with an offset baked in. Check yours with
  `scontrol show config | grep MaxArraySize` and set `max-array-size` to match.
- **Throttling.** `--array=0-999%50` runs fifty at a time. Without the `%`, a benchmark
  submitted on a shared cluster takes every free node on the partition.
- **Headroom.** SLURM's `--time` and `--mem` are set *above* the harness's own limits. The
  harness wants to notice its own timeout and write a `TIMEOUT` result; if SLURM kills it at
  the same instant, the row is missing instead — and a missing row reads as an infrastructure
  problem rather than a slow planner.

`setup-commands` are prepended to every job body, for `module load` and `conda activate`.
`--per-task-scripts` writes one file per run instead of arrays, for sites that disable them.
`${SLURM_ARRAY_TASK_ID:-0}` means a script run directly executes its first element, which is
how you debug one without a scheduler.

`--entry-point` changes how the jobs invoke the CLI, for a checkout that is not pip-installed:

```bash
planiverse-bench generate --exp-dir experiment --sandbox-dir sandbox \
    --entry-point "python -m planiverse.benchmark.cli"
```

## Statuses

Every run ends in one, because a failure has to be *recorded* rather than raised — a planner
that dies on instance 12 must not take the other 400 runs with it.

| Status | Meaning |
|---|---|
| `SOLVED` | a plan, and one that replays to a goal |
| `INVALID` | a plan that does **not** replay to a goal. A planner bug, reported as one |
| `UNSOLVED` | the planner stopped of its own accord without a plan |
| `TIMEOUT` | the wall-clock limit ran out first |
| `NODEOUT` | the expansion limit ran out first |
| `MEMOUT` | the memory limit was hit |
| `ERROR` | it raised; the traceback is in the result file |
| `UNSUPPORTED` | the environment could not be built here |
| `MISSING` | no result file. Assigned at analysis time, never written by a run |

`TIMEOUT` and `NODEOUT` are separate because they say different things about the same planner:
one is too slow per node, the other is looking in the wrong place.

`UNSOLVED` deliberately does **not** mean "unsolvable". Only BFWS among these planners is
complete, so for everything else it means no more than "this planner stopped looking". The
tables mark incomplete planners with `*`.

`MISSING` is the one worth watching. A benchmark's most common failure is a job that never ran
— cancelled, evicted, submitted to a partition that does not exist — and an analysis that reads
only the files that are there gives a planner that crashed on half the set excellent coverage
over the half it survived. The expected set comes from `tasks.json`, so anything without a file
is counted as a failure and called out in the report.

### Limits, and how they are enforced

- **Time.** The planner's own `Budget(max_seconds=...)`, plus a `SIGALRM` on top. The budget is
  only checked between expansions, which is enough right up until one expansion is itself slow
  — the power grid environment spends 8 to 19 seconds inside one.
- **Memory.** `RLIMIT_AS`, so an overrun raises `MemoryError` and can be recorded. Without it
  the OOM killer ends the process and the run leaves no trace.
- **Nodes.** `max-expansions`. Against a simulator this is usually the binding limit, because
  wall-clock is just however many expansions you allowed times the cost of one.

## Progress measures

`SIWSearch` and `BFWSSearch` take a `progress(state)` callback standing in for the
unachieved-goal count that classical width-based planners lean on. Against a simulator that
count does not exist — `is_goal` is a black-box predicate — so the measure is supplied per
environment, in [`measures.py`](../planiverse/benchmark/measures.py). Lower is better.

They live in the benchmark rather than in the environments for two reasons. They are a property
of how you choose to *search* an environment, not of the environment: two people can disagree
about the right measure for the water network without either being wrong about what the water
network is. And they are visible — `planiverse-bench environments` prints which environments
have one, so a weak result on an environment without a measure is legible as such instead of
looking like a weak planner.

Three environments have none, and the reason is recorded next to each: `epidemic` (the goal is
a threshold on a trajectory, and every monotone stand-in preferred doing nothing),
`manufacturing` (the objective is a cost over a whole schedule, not a distance) and
`urban_planning` (multi-objective by construction). Without a measure BFWS becomes
breadth-first search ordered by novelty alone and SIW becomes a single IW call. That is a real
result, not a broken one, but it is a different experiment — and the reports mark the rows with
`†`.

## Game Boy environments

They need a cartridge, which is copyrighted and cannot ship here. Point an environment variable
at one and set `"include-rom-environments": true`:

```bash
export PLANIVERSE_PUZZNIC_ROM="/path/to/Puzznic (J).gb"
export PLANIVERSE_FLIPULL_ROM="/path/to/Flipull (USA).gb"
export PLANIVERSE_SML_ROM="/path/to/Super Mario Land.gb"
```

Set them in `setup-commands` too, or the cluster jobs will not see them.

## Reading the report

`report/results.txt` has coverage, an outcome breakdown, solved-per-environment, head to head,
and IPC scores.

**Head to head** is there because coverage totals hide the interesting part: two planners can
each solve 40 of 60 and have only 20 in common, which is a completely different situation from
solving the same 40.

**Runtime** is summarised over solved runs only. Averaging in the timeouts would reward a
planner for failing quickly.

**IPC scores** follow the competition rules — quality is `best_length / this_length`, agile is
`1 / (1 + log10(t/t*))` — and both are relative to the planners in the same table. Adding a
planner changes everyone's numbers, so a score means nothing on its own.

**The plots** are the conventional pair: a survival ("cactus") plot of tasks solved against
time, and a runtime scatter between two planners with failures drawn on the border rather than
dropped. Coverage alone cannot distinguish a planner that solves 40 tasks quickly from one that
solves the same 40 just inside the limit.

## A worked example

7 planners over 16 tasks, 20-second limit, run locally:

```
Coverage
planner       solved  of   coverage  total time  median   plan len
bfws-2        12      16   75%       74.7s       1.40s    18.8
bfws-1        11      16   69%       51.0s       0.82s    20.5
mcts *        7       16   44%       122.5s      20.05s   5.1
iw-1 *        6       16   38%       30.6s       2.55s    19.8
iw-2 *        6       16   38%       8.4s        0.17s    4.0
siw-1 *       6       16   38%       17.4s       0.54s    20.2
fsx *         2       16   12%       40.0s       20.02s   2.0
```

Worth noticing: `iw-2` and `siw-1` solve the same number as `iw-1` but `iw-2`'s plans are a
quarter the length and it is fifteen times faster overall, and `mcts` finds much shorter plans
than anything else while spending the entire limit doing it. None of that is visible from a
coverage column alone, which is the argument for the rest of the tables.

## Python API

Every stage is importable, if you would rather script it than shell out:

```python
from planiverse.benchmark import (
    ExperimentConfig, Limits, PlannerSpec, TaskSelection, discover, pair_up, solve,
)

experiment = ExperimentConfig(
    limits=Limits(time="60s", max_expansions=5000),
    tasks=TaskSelection(selected_tasks=("flipull@0",)),
    planners=(PlannerSpec(tag="bfws-2", planner="bfws", params={"width": 2}),),
)
record = solve(experiment.planners[0], "flipull@0", experiment.limits)
print(record["status"], record["plan_length"])
```
