# Benchmarking

`planiverse-bench` runs every planner in the library over every environment, on a SLURM cluster or
on one machine, and turns the results into tables and plots. We modelled it on
[pyPMTEvalToolkit](https://github.com/pyPMT/pyPMTEvalToolkit): an experiment is a directory of
JSON, a sandbox is a directory of results, and the stages between them run independently.

- **Package:** [`planiverse/benchmark/`](../planiverse/benchmark/)
- **Command:** `planiverse-bench`, installed with the library
- **Dependencies:** none beyond the library. Plots need matplotlib, which is already a dependency;
  without it the tables are still written.

## The five minute version

```bash
./setup_benchmark.sh                      # asks about limits, environments and cartridges
bash sandbox/slurm/submit_all.sh          # or: bash sandbox/run_local.sh 8
planiverse-bench analyze   --sandbox-dir sandbox
planiverse-bench report    --sandbox-dir sandbox
```

`setup_benchmark.sh` is the front door. It runs the first three stages and asks the two things
nothing else can work out: which environments to run, where Enter keeps all of them, and where
your cartridges are. `--yes` takes every default and asks nothing, and `--environments` answers
the question from the command line. The stages underneath are ordinary commands, if you would
rather drive them yourself:

```bash
planiverse-bench init      --exp-dir experiment --rom-puzznic /path/to/Puzznic.gb
planiverse-bench discover  --exp-dir experiment --sandbox-dir sandbox
planiverse-bench generate  --exp-dir experiment --sandbox-dir sandbox
```

`init` writes a default experiment; `discover` resolves which `(environment, index)` pairs exist
here; `generate` writes the commands and the SLURM job arrays; `analyze` collects the result files
into tables and a CSV; `report` adds LaTeX and plots. `solve` is what each array element runs.

Each stage writes files the next one reads, so no stage needs another to be running. That is what
lets a benchmark be prepared on a laptop, run on a cluster, and analysed somewhere else again. It
is also what makes a run reproducible: the sandbox records which experiment produced it, so a
result can always be traced back to the limits it was obtained under.

## Installing

`setup_benchmark.sh` builds a virtualenv and installs the library into it before it does anything
else:

```
== creating virtualenv at /path/to/repo/.venv
== installing planiverse from /path/to/repo
```

The venv defaults to `.venv` beside the script, is reused if it is already there, and gets an
editable install (`pip install -e`), so editing the library and re-running the benchmark does not
need a reinstall.

The generated jobs call the venv's own console script by absolute path:

```bash
/path/to/repo/.venv/bin/planiverse-bench solve --exp-dir ... --task puzznic@0
```

That is deliberate, and it is the part that makes this work on a cluster. An activation depends on
the shell it happened in, and a job runs in a shell that never saw yours, so it either fails or,
worse, silently finds some other `planiverse-bench` on `PATH` and benchmarks a different version
of the library. An absolute path can do neither. The venv is activated in the jobs as well, so
anything they run after the CLI gets the same interpreter, and `run_local.sh` activates it too, so
a local run and a cluster run use the same Python rather than differing by whatever was on `PATH`.

| Flag | |
|---|---|
| `--venv DIR` | put the virtualenv somewhere else |
| `--no-venv` | do not build one; use whatever `planiverse` is already importable |
| `--python BIN` | interpreter to build it with (default `python3`) |

On a cluster, `--venv` has to name a filesystem the compute nodes can see. A virtualenv under
`/tmp` on the login node does not exist on the node that runs the job, and every array element
fails identically. The script cannot tell which of your paths is shared, so the default sits next
to the repository and moving it is your call.

`--no-venv` uses the current environment and stops with instructions if the library is not
importable, rather than picking an entry point that fails three stages later. `--setup-command` on
`init` adds arbitrary lines to the top of every job, for example `module load python/3.11`,
repeatable and in order.

## The experiment directory

```
experiment/
├── exp-details.json          limits, task selection, SLURM directives
└── planners/
    ├── bfws.json
    ├── iw.json
    ├── siw.json
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

Durations take either spelling, `"30m"` or `"00:30:00"`. Sizes take `"8GB"`. Unknown keys are
ignored, so a config written for a later version still loads.

A planner file:

```json
{
    "tag": "bfws",
    "planner": "iterated_bfws",
    "params": { "max_width": 1000 },
    "tags": [],
    "exclude-environments": [],
    "enabled": true
}
```

`max-instances-per-environment` is `0`, meaning every instance of every environment. Setting a
number gives a quick look, at a cost worth being clear about: a benchmark that samples a tenth of
each environment is reporting on a sample it chose, and `selection: "even"` decides which tenth.
`include-rom-environments` is on, so a Game Boy environment is benchmarked next to its Python
twin, which is most of the point of having both, and without a cartridge those environments are
skipped with a reason rather than failing.

`include-environments` is empty, meaning all of them. The answer to `setup_benchmark.sh`'s
question arrives here through `init --environments puzznic,super_mario_land`, as comma-separated
registry names, refused at `init` if one is misspelled, because a typo that silently selects
nothing costs an empty experiment. Everything not selected is reported as skipped, with the
reason, rather than silently dropped, and the setup script stops asking for cartridges whose
environments are not in the selection.

`roms` records where the cartridges are. It lives in the experiment rather than in environment
variables so that the experiment is self-contained: a variable exported in the shell that ran
`generate` is not there on the compute node, and the whole array would come back `UNSUPPORTED`. A
variable is still honoured as a fallback, and a recorded path that does not exist falls back to
one, because a path written on the machine that made the config is a promise about a different
filesystem until it is checked.

`tag` names the planner everywhere afterwards, in result filenames, the sbatch job name and every
table, so it must be filesystem-safe and stable. `planner` is a name from the catalogue, and
`planiverse-bench planners` lists them. A misspelled parameter is refused rather than ignored,
because a benchmark that quietly drops `"widht": 2` reports a width-1 result under a width-2 name
and nothing downstream can tell.

`tags` and `exclude-environments` narrow one planner's task list, so a single experiment can run a
cheap planner over everything and an expensive one over a subset.

### Why no default planner has a pinned width

No default planner has a pinned width, because IW does not have one. IW(k) is a family, and which
member you need is a property of the problem rather than a setting. The default experiment runs
`iterated_width` for that reason, which tries IW(1), IW(2), … up to `max_width`, set to 1000.
Every width planner in the default spread is the iterated version of itself, for reasons that
differ per planner, while the sampling planners `fsx` and `mcts` have no width to iterate and run
as themselves.

That bound is a bound rather than a plan. The loop stops as soon as any of three things happens:
(1) a width solves the problem; (2) the budget runs out, which is the usual outcome above width 2
since IW(k) enumerates every k-tuple of every state's atoms; or (3) a width exhausts the reachable
space without discarding anything for novelty, at which point no larger width can reach further
and there is no plan. In practice it stops at 1 or 2 and the 1000 is never approached.

Widths above 2 need `strict: false`. `NoveltyTable` refuses them otherwise, on the grounds that
C(n, k) tuples per state is usually more expensive than the simulator it is meant to be saving,
which is true, and is exactly why the bound has to be discovered rather than assumed.

**`siw` iterates for the same reason**, because each of its legs is an IW search. Novelty is a
filter there too, so if no state within IW(k)'s pruned reach improves progress, the leg fails and
the whole search fails, even though a wider leg would have found one. `SIWSearch` takes a
`max_width`, and a leg tries increasing widths until it makes progress, runs out of budget, or
finds that a width covered everything reachable from its start without pruning anything. The
widths share one budget, so a stubborn leg cannot spend more than a leg that succeeded
immediately. `statistics.widths_tried` reports the widths the legs needed.

**`bfws` iterates for a different reason.** Plain BFWS uses novelty as a sort key rather than as a
filter, so nothing is ever discarded, no width can make it miss anything, and iterating it would
be pointless. Measured over a few tasks at a 3000-expansion budget:

```
task              w status           exp  pruned_novelty  plan
puzznic@1         1 solved            89               0    10
puzznic@1         2 solved            89               0    10
puzznic@1         4 solved            89               0    10
puzznic@30        1 solved          2042               0    65
puzznic@30        2 out_of_budget   3000               0     -
puzznic@30        4 out_of_budget   2788               0     -
flipull@9         1 solved          2395               0    61
flipull@9         2 out_of_budget   3000               0     -

IW(1) puzznic@1: exhausted, pruned_novelty=47
```

Three things follow. `pruned_novelty` is zero at every width, since plain BFWS discards nothing,
where IW(1) threw away 47 states on the same task. There is no "the width was too small" outcome
to escalate from, because BFWS(1) ends solved, `exhausted` (it covered the reachable space, which
proves there is no plan), or `out_of_budget`, and all three are stop conditions. And iterating
unpruned BFWS under one budget would spend it at width 1 and never reach width 2, making it
exactly BFWS(1) with extra machinery.

The default runs `iterated_bfws`, whose rounds are the pruned variant, k-BFWS: IW's novelty filter
with BFWS's ordering inside it. A pruned round has IW's bounded frontier, so it is cheap, and it
can run out of width, so escalating it means something. The rounds run at widths 1 and 2, and
whatever budget they leave goes to one unpruned round, which is complete. The bound is the same
1000 as the others', but `strict` is left on: here escalation competes with the final complete
round for the same budget, and pruned rounds above width 2 would spend it enumerating tuples.

Wider is not stronger. On `puzznic@30` and `flipull@9`, BFWS(1) solves where BFWS(2) and above run
out of budget: a higher width discriminates more finely and so changes the search *order*, and on
those tasks the coarser order is the better one. Width in BFWS is a dial rather than a ladder,
which is why the final complete round runs at width 1, the cheapest member of a family whose
members are all complete.

For IW and SIW novelty is a filter, so a width too low genuinely loses states and climbing is the
only way to find out what the problem needs. That is why `iw` and `siw` climb with `strict: false`
while `bfws` keeps the strict guard and stops climbing at 2.

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

We generate one job array per planner rather than one job per run. A benchmark is thousands of
short runs, and a scheduler handling them as thousands of jobs spends longer scheduling than
computing. Line *n* of `cmds/<planner>.txt` is array index *n*, and nothing re-derives that
ordering, which is what lets a failed element be re-run by hand.

```bash
#SBATCH --job-name=planiverse-bench-bfws
#SBATCH --array=0-15%50
#SBATCH --time=00:35:00
#SBATCH --mem=9216M
...
COMMANDS=/abs/path/sandbox/cmds/bfws.txt
INDEX=$(( ${SLURM_ARRAY_TASK_ID:-0} + OFFSET ))
COMMAND=$(sed -n "$(( INDEX + 1 ))p" "$COMMANDS")
eval "$COMMAND"
```

A generated array has to get three things right, none of them obvious the first time:

- **Array size.** `MaxArraySize` is a site limit, commonly 1001, and an array over it is rejected
  at submission with a message that does not name the cause. Long lists are split across several
  `sbatch` files with an offset baked in. Check yours with `scontrol show config | grep
  MaxArraySize` and set `max-array-size` to match.
- **Throttling.** `--array=0-999%50` runs fifty at a time. Without the `%`, a benchmark submitted
  on a shared cluster takes every free node on the partition.
- **Headroom.** SLURM's `--time` and `--mem` are set above the harness's own limits, so the
  harness notices its own timeout and writes a `TIMEOUT` result rather than being killed at the
  same instant and leaving a missing row.

`setup-commands` are prepended to every job body and to `run_local.sh`, so a local run and a
cluster run use the same interpreter. `setup_benchmark.sh` puts its virtualenv activation there,
and your own go in with `--setup-command` on `init`, repeatable and in order. `--per-task-scripts`
writes one file per run instead of arrays, for sites that disable them.
`${SLURM_ARRAY_TASK_ID:-0}` means a script run directly executes its first element, which is how
to debug one without a scheduler.

`--entry-point` changes how the jobs invoke the CLI, for a checkout that is not pip-installed:

```bash
planiverse-bench generate --exp-dir experiment --sandbox-dir sandbox \
    --entry-point "python -m planiverse.benchmark.cli"
```

## Statuses

Every run ends in one, because a failure has to be recorded rather than raised: a planner that
dies on instance 12 must not take the other 400 runs with it.

| Status | Meaning |
|---|---|
| `SOLVED` | a plan, and one that replays to a goal |
| `INVALID` | a plan that does not replay to a goal. A planner bug, reported as one |
| `UNSOLVED` | the planner stopped of its own accord without a plan |
| `TIMEOUT` | the wall-clock limit ran out first |
| `NODEOUT` | the expansion limit ran out first |
| `MEMOUT` | the memory limit was hit |
| `ERROR` | it raised; the traceback is in the result file |
| `UNSUPPORTED` | the environment could not be built here |
| `MISSING` | no result file. Assigned at analysis time, never written by a run |

`TIMEOUT` and `NODEOUT` are separate because they say different things about the same planner: one
is too slow per node, the other is looking in the wrong place.

`UNSOLVED` should not be confused with "unsolvable". It means the planner stopped of its own
accord without a plan, and whether that counts as a proof depends on the planner:

- **BFWS** is complete: it uses novelty as a sort key rather than as a filter, so nothing is ever
  discarded. Its `UNSOLVED` rows are proofs.
- **Iterated Width** is complete on the runs where it says so. It reports `exhausted` when one of
  its widths covered the reachable space without discarding anything for novelty; stopped by the
  budget instead, it proves nothing. Each result carries its own `complete` flag.
- **IW at a fixed width, SIW, FSX and MCTS** prove nothing either way.

The coverage table stars a planner when at least one of its `UNSOLVED` rows is not a proof, and
`results.csv` has the per-run `complete` and `search_status` columns behind it.

`MISSING` is the one worth watching, because a benchmark's most common failure is a job that never
ran, whether cancelled, evicted, or submitted to a partition that does not exist. An analysis that
reads only the files that are there would give a planner which crashed on half the set excellent
coverage over the half it survived. The expected set comes from `tasks.json`, so anything without
a file is counted as a failure and called out in the report.

### Limits, and how they are enforced

- **Time.** The planner's own `Budget(max_seconds=...)`, plus a `SIGALRM` on top. The budget is
  only checked between expansions, which is enough right up until one expansion is itself slow:
  the power grid environment spends 8 to 19 seconds inside one.
- **Memory.** `RLIMIT_AS`, so an overrun raises `MemoryError` and can be recorded. Without it the
  OOM killer ends the process and the run leaves no trace.
- **Nodes.** `max-expansions`. Against a simulator this is usually the binding limit, because
  wall-clock time is however many expansions were allowed times the cost of one.

## Progress measures

`SIWSearch` and `BFWSSearch` take a `progress(state)` callback standing in for the unachieved-goal
count that classical width-based planners lean on. Against a simulator that count does not exist,
because `is_goal` is a black-box predicate, so the measure is supplied per environment in
[`measures.py`](../planiverse/benchmark/measures.py). Lower is better.

They live in the benchmark rather than in the environments for two reasons. First, they are a
property of how you choose to *search* an environment rather than of the environment itself, so
two people can disagree about the right measure for the water network without either being wrong
about what the water network is. Second, they are visible: `planiverse-bench environments` prints
which environments have one, so a weak result on an environment without a measure can be read as
such rather than looking like a weak planner.

Every environment currently has one, so `WITHOUT_MEASURE` is empty, and we keep the path because
the next environment added may not. Without a measure BFWS becomes breadth-first search ordered by
novelty alone and SIW becomes a single IW call, which is a real result rather than a broken one
but a different experiment, so the reports mark those rows with `†`.

## Game Boy environments

These environments need a cartridge, which is copyrighted and cannot ship here, so the path can
only come from you. `./setup_benchmark.sh` asks for all of them, checks each file exists, and
records the paths in the experiment.

There is a flag per cartridge, and it works on both `setup_benchmark.sh` and `planiverse-bench
init`:

| Flag | Environment | Variable |
|---|---|---|
| `--rom-puzznic PATH` | `puzznic_gb` | `PLANIVERSE_PUZZNIC_ROM` |
| `--rom-flipull PATH` | `flipull_gb` | `PLANIVERSE_FLIPULL_ROM` |
| `--rom-lolo PATH` | `lolo_gb` | `PLANIVERSE_LOLO_ROM` |
| `--rom-super-mario-land PATH`, `--rom-mario PATH` | `super_mario_land_gb` | `PLANIVERSE_SUPER_MARIO_LAND_ROM` |

```bash
./setup_benchmark.sh --rom-puzznic ~/roms/"Puzznic (J).gb" \
                     --rom-flipull ~/roms/"Flipull (USA).gb" \
                     --rom-lolo    ~/roms/"Adventures of Lolo (U) [S][!].gb" \
                     --rom-mario   ~/roms/"Super Mario Land.gb"
```

Give one on the command line and the script does not ask about it. The same flags work on `init`:

```bash
planiverse-bench init --exp-dir experiment --force \
    --rom-puzznic "/path/to/Puzznic (J).gb" \
    --rom-flipull "/path/to/Flipull (USA).gb" \
    --rom-lolo    "/path/to/Adventures of Lolo (U) [S][!].gb" \
    --rom-mario   "/path/to/Super Mario Land.gb"
```

The flags are generated from the registry, so a new Game Boy environment gets one by existing.
Each is its environment variable's name in a different spelling, which is where
`--rom-super-mario-land` comes from, while `--rom-mario` is a shorter alias and `--rom-sml` is
kept for the old abbreviation. A path that does not exist is refused there and then, because a
typo caught while typing costs a second and one caught after submitting four thousand jobs costs
rather more.

`--rom puzznic_gb=PATH` keys the same thing by environment name, which is easier in a loop where
the environment is itself a variable.

Environment variables still work as a fallback, but a variable exported in your shell is not there
on the compute node unless you also put it in `setup-commands`. The recorded path travels with the
experiment.

Each of these has a Python counterpart (`puzznic`, `flipull`, `super_mario_land`), so supplying a
cartridge is what lets you compare an emulated environment against an implemented one under the
same planners and the same limits. Skip one and it is reported as skipped, with the reason, rather
than quietly dropped.

## Reading the report

`report/results.txt` has coverage, an outcome breakdown, solved-per-environment, head to head, and
IPC scores.

**Head to head** is there because coverage totals hide the interesting part: two planners can each
solve 40 of 60 and have only 20 in common, which is a completely different situation from solving
the same 40.

**Runtime** is summarised over solved runs only. Averaging in the timeouts would reward a planner
for failing quickly.

**IPC scores** follow the competition rules: quality is `best_length / this_length`, and agile is
`1 / (1 + log10(t/t*))`. Both are relative to the planners in the same table, so adding a planner
changes everyone's numbers and a score means nothing on its own.

**The plots** are the conventional pair: a survival, or cactus, plot of tasks solved against time,
and a runtime scatter between two planners with failures drawn on the border rather than dropped.
Coverage alone cannot distinguish a planner that solves 40 tasks quickly from one that solves the
same 40 just inside the limit.

## Python API

Every stage is importable, for scripting rather than shelling out:

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
