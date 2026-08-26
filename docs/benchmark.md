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
./setup_benchmark.sh                      # asks about limits and your Game Boy cartridges
bash sandbox/slurm/submit_all.sh          # or: bash sandbox/run_local.sh 8
planiverse-bench analyze   --sandbox-dir sandbox
planiverse-bench report    --sandbox-dir sandbox
```

`setup_benchmark.sh` is the front door. It runs the first three stages and asks the one thing
nothing else can work out — where your cartridges are. `--yes` takes every default and asks
nothing. The stages underneath are ordinary commands if you would rather drive them:

```bash
planiverse-bench init      --exp-dir experiment --rom-puzznic /path/to/Puzznic.gb
planiverse-bench discover  --exp-dir experiment --sandbox-dir sandbox
planiverse-bench generate  --exp-dir experiment --sandbox-dir sandbox
```

`init` writes a default experiment; `discover` resolves which `(environment, index)` pairs
exist here; `generate` writes the commands and the SLURM job arrays; `analyze` collects the
result files into tables and a CSV; `report` adds LaTeX and plots. `solve` is the one stage you
do not normally type — it is what each array element runs.

## Installing, and where

By default `setup_benchmark.sh` installs nothing. It uses whatever `planiverse` is already
importable and **stops with instructions if there is none**, rather than picking a command that
will fail three stages later on an `ImportError`.

`--venv DIR` makes it do the work: create the virtualenv if it is not there, `pip install` the
repository into it, activate it for the rest of the run, and — the part that matters on a
cluster — write `source DIR/bin/activate` into every generated job.

```bash
./setup_benchmark.sh --venv ~/shared/planiverse-venv --rom-puzznic ~/roms/Puzznic.gb
```

The directory has to be on a filesystem the **compute nodes** can see. A virtualenv under
`/tmp` on the submitting host does not exist on the node that runs the job, and every array
element would fail identically. The script cannot tell which of your paths is shared, so it
does not guess — it uses the one you name. `--python BIN` chooses the interpreter to build it
with, for a site where `python3` is older than the library needs.

Running it again with the same `--venv` reuses the environment and reinstalls, which is what
you want after editing the library.

If you would rather manage this yourself, `pip install -e .` and skip the flag; and
`--setup-command` on `init` puts arbitrary lines at the top of every job (`module load
python/3.11`, `conda activate ...`), repeatable and in order.

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
    ├── bfws-1.json
    ├── bfws-2.json
    ├── iw.json
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
        "max-instances-per-environment": 0,
        "selection": "even",
        "include-rom-environments": true,
        "selected-tasks": []
    },
    "roms": {
        "puzznic_gb": "/path/to/Puzznic (J).gb"
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

`max-instances-per-environment` is `0`, meaning **every instance of every environment**. Set
a number for a quick look, but be clear about what it costs: a benchmark that samples a tenth
of each environment is reporting on a sample it chose, and `selection: "even"` decides which
tenth. `include-rom-environments` is on, so a Game Boy environment is benchmarked next to its
pure-Python twin — which is most of the point of having both. Without a cartridge those
environments are skipped with a reason rather than failing.

`roms` records where the cartridges are. It lives in the experiment rather than in environment
variables so the experiment is self-contained: a variable exported in the shell that ran
`generate` is not there on the compute node, and the whole array would come back
`UNSUPPORTED`. A variable is still honoured as a fallback, and a recorded path that does not
exist falls back to one — a path written on the machine that made the config is a promise
about a different filesystem until it is checked.

`tag` names the planner everywhere afterwards — result filenames, sbatch job name, every table
— so it must be filesystem-safe and stable. `planner` is a name from the catalogue
(`planiverse-bench planners` lists them). A misspelled parameter is **refused**, not ignored: a
benchmark that quietly drops `"widht": 2` reports a width-1 result under a width-2 name and
nothing downstream can tell.

`tags` and `exclude-environments` narrow one planner's task list, so a single experiment can run
a cheap planner over everything and an expensive one over a subset.

### Why `iw` and `siw` have no width, and `bfws` does

Because IW does not have one. IW(k) is a family, and which member you need is a property of the
problem, not a setting — so the default experiment runs `iterated_width`, which tries IW(1),
IW(2), … up to `max_width`, set to 1000.

That bound is a bound, not a plan. The loop stops as soon as any of three things happens: a
width solves the problem; the budget runs out (the usual outcome above width 2, since IW(k)
enumerates every k-tuple of every state's atoms); or **a width exhausts the reachable space
without discarding anything for novelty**, at which point no larger width can reach further and
there is no plan. In practice it stops at 1 or 2 and the 1000 is never approached.

Widths above 2 need `strict: false`. `NoveltyTable` refuses them otherwise, on the grounds that
C(n, k) tuples per state is usually more expensive than the simulator it is meant to be saving
— which is true, and is exactly why the bound has to be discovered rather than assumed.

**`siw` iterates for the same reason**, because each of its legs *is* an IW search. Novelty is
a filter there too, so if no state within IW(k)'s pruned reach improves progress, the leg fails
and the whole search fails — even though a wider leg would have found one. `SIWSearch` takes a
`max_width`, and a leg tries increasing widths until it makes progress, runs out of budget, or
finds that a width covered everything reachable from its start without pruning anything, which
means no wider leg could do better either. The widths share one budget, so a stubborn leg
cannot spend more than a leg that succeeded immediately. `statistics.widths_tried` then reports
the widths the legs actually needed — a problem whose hardest leg needed IW(2) is a different
problem from one every leg solved at IW(1), and pinning the width hid that.

**`bfws` does not iterate, and that is deliberate.** BFWS uses novelty as a *sort key* rather
than as a filter: nothing is ever discarded, so BFWS(k) is complete at every width. A BFWS(1)
that fails ran out of budget — it did not run out of width — and restarting at a wider one
would re-expand everything while splitting the same budget across the attempts, which is
strictly worse. What the width changes is the search *order*, so `bfws-1` and `bfws-2` are two
different strategies worth comparing side by side rather than a weaker and a stronger one.
That is why they are two entries and `iw` and `siw` are one each.

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

`setup-commands` are prepended to every job body, for `module load` and `conda activate`. Set
them with `--setup-command` on `init`, repeatable and in order; `setup_benchmark.sh --venv`
fills in the activation for you.
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

`UNSOLVED` deliberately does **not** mean "unsolvable". It means the planner stopped of its
own accord without a plan, and whether that is a proof depends on the planner:

- **BFWS** is complete — it uses novelty as a sort key rather than as a filter, so nothing is
  ever discarded. Its `UNSOLVED` rows are proofs.
- **Iterated Width** is complete *on the runs where it says so*. It reports `exhausted` when
  one of its widths covered the reachable space without discarding anything for novelty; that
  is a proof. Stopped by the budget instead, it proves nothing. So each result carries its own
  `complete` flag rather than inheriting one from the planner.
- **IW at a fixed width, SIW, FSX and MCTS** prove nothing either way.

The coverage table stars a planner when at least one of its `UNSOLVED` rows is not a proof, and
`results.csv` has the per-run `complete` and `search_status` columns behind it.

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

They need a cartridge, which is copyrighted and cannot ship here — so the path can only come
from you. `./setup_benchmark.sh` asks for all three, checks each file exists, and records the
paths in the experiment. That is the recommended route, and it is why the script exists.

There is a flag per cartridge, and it works on both `setup_benchmark.sh` and
`planiverse-bench init`:

| Flag | Environment | Variable |
|---|---|---|
| `--rom-puzznic PATH` | `puzznic_gb` | `PLANIVERSE_PUZZNIC_ROM` |
| `--rom-flipull PATH` | `flipull_gb` | `PLANIVERSE_FLIPULL_ROM` |
| `--rom-sml PATH`, `--rom-mario PATH` | `super_mario_land` | `PLANIVERSE_SML_ROM` |

```bash
./setup_benchmark.sh --rom-puzznic ~/roms/"Puzznic (J).gb" \
                     --rom-flipull ~/roms/"Flipull (USA).gb" \
                     --rom-mario   ~/roms/"Super Mario Land.gb"
```

Give one on the command line and the script does not ask about it; leave it out and it does.
The same flags work on `init` directly:

```bash
planiverse-bench init --exp-dir experiment --force \
    --rom-puzznic "/path/to/Puzznic (J).gb" \
    --rom-flipull "/path/to/Flipull (USA).gb" \
    --rom-mario   "/path/to/Super Mario Land.gb"
```

The flags are generated from the registry, so a new Game Boy environment gets one by existing,
and each is the same name as its environment variable in a different spelling — which is where
`--rom-sml` comes from. `--rom-mario` is an alias, because nobody types "sml" first. A path
that does not exist is refused there and then: a typo caught while typing costs a second, and
one caught after submitting four thousand jobs costs rather more.

`--rom puzznic_gb=PATH` keys the same thing by environment name, which is easier in a loop
where the environment is itself a variable.

Environment variables still work as a fallback, but they are the weaker option for a cluster
run: a variable exported in your shell is not there on the compute node unless you also put it
in `setup-commands`. The recorded path travels with the experiment.

Each of these has a pure-Python counterpart — `puzznic`, `flipull`, `platformer` — so
supplying a cartridge is what lets you compare an emulated environment against an implemented
one under the same planners and the same limits. Skip one and it is reported as skipped, with
the reason, rather than quietly dropped.

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

7 planners over 18 tasks — two instances from each environment, a Puzznic cartridge supplied —
at a 20-second limit, run locally.

This was recorded before SIW was changed to iterate its width, so the `siw-1` and `siw-2` rows
are the pinned configurations; the default set now has a single iterated `siw` in their place.
Everything else is current.

```
Coverage
planner       solved  of   coverage  total time  median   plan len
bfws-1        15      18   83%       76.7s       4.49s    16.3
bfws-2        15      18   83%       75.2s       3.58s    16.3
iw            14      18   78%       98.4s       4.93s    12.2
mcts          8       18   44%       145.3s      20.03s   4.6
siw-1 *       7       18   39%       33.8s       0.56s    17.6
siw-2 *       7       18   39%       13.6s       0.06s    6.3
fsx           2       18   11%       26.8s       13.42s   2.0

Solved per environment
environment       bfws-1  bfws-2  fsx   iw    mcts  siw-1  siw-2
crop_management   2/2     2/2     0/2   2/2   2/2   1/2    1/2
flipull           2/2     2/2     0/2   1/2   1/2   1/2    1/2
manufacturing †   2/2     2/2     2/2   2/2   2/2   2/2    2/2
platformer        2/2     2/2     0/2   2/2   1/2   0/2    0/2
power_grid        2/2     2/2     0/2   2/2   1/2   1/2    0/2
puzznic           1/2     1/2     0/2   1/2   0/2   0/2    1/2
puzznic_gb        1/2     1/2     0/2   1/2   0/2   0/2    1/2
urban_planning †  1/2     1/2     0/2   1/2   0/2   1/2    0/2
water_network     2/2     2/2     0/2   2/2   1/2   1/2    1/2
```

Worth noticing. `iw` — Iterated Width, not a width someone picked — lands within one task of
BFWS, and finds shorter plans than either BFWS configuration while doing it; an earlier run
with IW pinned at 1 and 2 had each managing well under half of what BFWS did. `mcts` finds much
shorter plans than anything else and spends the entire limit doing it. `siw-2` is an order of
magnitude faster than `siw-1` at identical coverage. And `puzznic` and `puzznic_gb` come out
identical planner for planner, which is the cartridge and its pure-Python twin agreeing — the
comparison you can only make by supplying a ROM.

None of that is visible from a coverage column alone, which is the argument for the rest of the
tables.

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
