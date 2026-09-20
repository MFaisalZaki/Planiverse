# Benchmarking

`planiverse-bench` is the tool paper's evaluation protocol as code. It runs the five planner
configurations the paper compares, and the two rollout planners added since, on every instance
of every environment, under the paper's limits, with five seeds for every planner that takes
one, and turns the results into the paper's tables and figures.

- **Package:** [`planiverse/benchmark/`](../planiverse/benchmark/): the benchmark, and the progress
  measures SIW, BFWS, Rollout IW and π-IW take per environment.
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
  (`bfws`) or one seed of a seeded planner (`mcts-s3`). Line *n* is array element *n*, so a
  failed element can be re-run by hand from its line.
- `sandbox/slurm/<group>.sbatch`, a job array that reads that file by `$SLURM_ARRAY_TASK_ID`,
  throttled to `--parallel` elements at a time (default 50), and given 35 minutes and 9 GB so the
  benchmark records its own `TIMEOUT` or `MEMOUT` before SLURM steps in. One array per group keeps
  every array one instance set long, under a site's `MaxArraySize`, and finishes seed 0 first.
- `sandbox/submit.sh` and `sandbox/run_local.sh`.

The suite's 2,000 instances, a hundred in each of the twenty environments the report covers
(the paper's eight and the twelve added since, eleven of them simulation-driven and one a
re-implemented game), make 23 arrays: three for the deterministic width planners and five each
for MCTS, FSX, Rollout IW and π-IW, 46,000 runs. The commands call the interpreter that ran
`generate` by absolute path, so the jobs need no activation and cannot pick up a different
install. An environment that cannot be built here (a missing dependency) is skipped, and
`generate` says so.

The benchmark runs the bundled instances. Each environment can also generate its own (see
[Generating instances](../README.md#generating-instances)); a generated benchmark is a matter of
writing the instances to a file and looping `set_instance` over them, which `generate` does not
do for you.

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
| Seeds | 0 to 4 for MCTS, FSX, Rollout IW and π-IW, each (instance, seed) a full run under the limits above; BFWS, IW and SIW are deterministic and run once |
| Solved | only if the returned plan, replayed through `simulate`, reaches a goal |

| Planner | Class | Parameters |
|---|---|---|
| `bfws` | `IteratedBFWS` | `max_width=1000` |
| `iw` | `IteratedWidth` | `max_width=1000, strict=False` |
| `siw` | `SIWSearch` | `width=1, max_width=1000, strict=False` |
| `mcts` | `MCTSPlanner` | `iterations=2000`, the run's seed; the exploration constant √2, 30-step rollouts, max backup and the 0.001 length penalty are the class defaults |
| `fsx` | `FSXPlanner` | `horizon=6, walkers=8`, the run's seed; the distinct-state count, zero temperature and 200 committed steps are the class defaults |
| `riw` | `RolloutIW` | `width=1, expansions_per_step=1000`, the run's seed; the 0.99 discount, 200-step episodes, one episode, subtree reuse and dead-end avoidance are the class defaults |
| `piiw` | `PiIW` | `width=1, expansions_per_step=100`, the run's seed; the network (2048 hashed inputs, 64 hidden units), τ = 0.5, one Adam step per decision on a batch of 32 from a replay of 10,000, episodes until the budget ends, and the environment's literals as the novelty atoms are the class defaults |

SIW and BFWS take a `progress(state)` callback in place of the unachieved-goal count a classical
planner would use, and Rollout IW and π-IW take the same callback in place of the score their
papers' Atari games provide: a transition's reward is the drop in it.
[`measures.py`](../planiverse/benchmark/measures.py) supplies one per environment, lower is
better; they are search guides, not admissible heuristics. The environments are deterministic, so
for the seeded planners the seed is the only source of variance. Rollout IW gets the larger
per-decision budget and π-IW the smaller, as in their papers: π-IW's claim is that a learned
policy makes a small lookahead go a long way, and it keeps learning across episodes for as long
as the budget lasts. See [rollout-width.md](planners/rollout-width.md).

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

`UNSOLVED` says the planner stopped looking, not that there is no plan: only BFWS is complete. For
Rollout IW it means its one episode ended at a dead end or the step cap, or its lookahead solved
the root without a goal beneath it. A search that reports `out_of_budget` without reaching either
limit (an iterated search whose per-width allowances ran out, or FSX at its step cap or a dead
end) is filed as `NODEOUT`.

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
  seed, what each planner solved outside BFWS's set, medians, the per-instance speed ratios
  with their sign tests, errors and missing runs, IW's widths, plan lengths, and each seeded
  planner's coverage per family; then the protocol's other aggregations (the mean fraction
  solved over environments and the IPC quality score), each planner's plan lengths against
  BFWS's, what the rollout planners reached by the width IW needed, π-IW against Rollout IW on
  the runs both solved and its episodes per solved run, every planner's statuses per
  environment, the cost of an expansion per environment, and the
  difficulty profile (open instances, BFWS's plan lengths, successors per expansion, IW's
  largest width) that the paper's open-challenges section tabulates.

The sandbox behind the paper is `paper-results.zip` on the
[release page](https://github.com/MFaisalZaki/Planiverse/releases). Unzip it beside the
repository and `report` regenerates every number in the paper from it. That sandbox also holds
results for the five emulator-backed environments the paper compared, which have since been
withdrawn from this repository; `report` still tabulates whatever `tasks.json` lists, but those
runs can no longer be repeated from here.

## Bringing the paper up to date

The paper compares five planners and the code runs seven. Rollout IW and π-IW have no results
in the released sandbox, so `report` lists their runs as `MISSING` until their arrays are run.
We have run the pipeline end to end on a pilot (two instances, both planners, one seed each,
then `report`), which checked that `generate` writes the ten arrays, that `solve` records
`rollouts` and `episodes`, and that `report` emits the two coverage columns, the two status
rows and the two cactus curves. What remains is the compute, and then the prose, which is
drafted below so that the runs are the only thing between the code and the paper.

**The runs.** `generate` writes `riw-s0` to `riw-s4` and `piiw-s0` to `piiw-s4` beside the
existing arrays: ten arrays over the instances in the tree, 20,000 runs for the 2,000 the report's
environments have. Nothing already run needs repeating, since the protocol, the limits and the
other planners' parameters are unchanged. The paper's cartridge rows cannot be extended, since
those environments are no longer here, and the tables have to say so. Afterwards, re-release
`paper-results.zip` with the new result files in it.

**The planner section.** Two paragraphs, to follow the paragraph on BFWS, with the
adaptations stated the way the paper states SIW's and BFWS's:

> Bandres, Bonet and Geffner (2018) proposed Rollout IW, an online form of IW(k) that keeps
> the novelty filter and replaces the breadth-first order with rollouts (i.e., paths grown
> from the root one node at a time, each stopping at the first node that is not novel, is a
> dead end, or has nothing left beneath it). Novelty is measured against depth: a tuple of
> atoms is novel at a node when it has never been seen that shallow, which lets rollouts
> arrive in any order without the first deep visit pre-empting every shallower one. After a
> budget of 1,000 expansions, the action whose subtree backed up the best discounted return
> is committed to, its subtree is kept, and the novelty table is reset; the reset is what
> renews exploration, and it is why the online form solves instances that a single width-1
> lookahead cannot. We adapt it to simulators in three ways. First, there is no score, so the
> reward of a transition is the drop in the environment's progress measure, the same measure
> SIW and BFWS take, and a goal ends the search as soon as a rollout reaches one. Second, a
> dead end backs up a return of minus infinity, because `is_terminal` means that no goal is
> reachable from there, whereas the Atari reading would merely stop scoring at the step into
> it. Third, an expansion generates every child of a node, since the contract is
> `successors()`, and the rollout picks one; a sibling reached by a later rollout has its
> novelty assessed then, as a new node. The planner runs a single episode, so an episode that
> ends at a dead end or at the step cap is an unsolved run.

> Junyent, Jonsson and Gómez (2019) proposed π-IW, which keeps Rollout IW's lookahead and
> replaces the uniform choice of the child a rollout follows with a sample from a policy
> trained on the planner itself: after every committed action, the returns the lookahead
> backed up to the root's children become a target distribution, a softmax of the returns at
> temperature τ, and the network is pushed toward it by cross-entropy. We give it a budget of
> 100 expansions per decision, a tenth of Rollout IW's, since its claim is that a learned
> policy makes a small lookahead go a long way, and we let it learn across episodes for as
> long as the run's budget lasts, so that an episode which ends without a goal is not wasted.
> The network is a one-hidden-layer model over hashed literals (2,048 inputs and 64 hidden
> units) rather than a convolutional one over pixels, trained by one Adam step per decision on
> a batch of 32 drawn from a replay of the last 10,000 targets, with τ = 0.5. Returns are
> scaled to [0, 1] among the root's children before the softmax, so that one temperature
> serves every environment, and a dead end's minus infinity enters the softmax as a
> probability of zero, so the policy is taught not to go there. The environment's literals are
> the atoms novelty is measured over; the paper's binarised hidden layer is available as an
> option and is not used here. This is enough for the mechanism the paper describes and is
> not a claim to match its Atari numbers.

**The protocol paragraph.** "Five planner configurations" becomes seven, and "five seeds for
the two stochastic planners" becomes five seeds for the four that take one, with the note that
for Rollout IW and π-IW the seed drives the rollouts and, for π-IW, the network's
initialisation and its training batches. The parameter table gains the two rows in
[The protocol](#the-protocol) above. The claims that set the width family against the sampling
planners have to be reworded, since Rollout IW is a width planner that samples.

**Tables, figures and numbers.** Table 2 (coverage) gains the columns `RIW` and `PIIW`, as
means over seeds with the standard deviation in brackets, which `coverage.tex` already emits;
the bold total may move. Table 3 (statuses) gains two rows and their median solve times,
which `statuses.tex` already emits; Rollout IW's single episode makes a dead end or the step
cap an `UNSOLVED`, whereas π-IW keeps starting episodes and, short of a plan, ends only at a
limit. The cactus plot gains two curves; the overlap and runtime figures stay as they are,
since they compare the three deterministic width planners and a seeded planner has no single
time per instance to put on them. The numbers the prose quotes are all in `facts.txt`:
coverage per seed, what each rollout planner solved in some seed that BFWS never did and the
reverse, the medians, and the per-family coverage line the report writes for every seeded
planner.

**The discussion.** One result is worth a paragraph whichever way it falls: whether resetting
the novelty table per decision lets Rollout IW(1) reach instances that Iterated Width needed
width 3 or 4 for, and whether π-IW's policy shortens its plans and its expansions against
Rollout IW's on the environments where the progress measure is dense.

One item is unrelated to the planners and still pending: the caption of the overlap figure
describes four groups and the figure has five, and the bar order the report writes is not the
caption's.
