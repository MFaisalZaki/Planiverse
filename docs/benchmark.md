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

The suite's 938 instances make 23 arrays: three for the deterministic width planners and five
each for MCTS, FSX, Rollout IW and π-IW, 21,574 runs. The commands call the interpreter that ran `generate` by absolute path, so
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
  seed, what each planner solved outside BFWS's set, medians, the per-instance speed ratios,
  errors and missing runs, IW's widths, plan lengths, and each seeded planner's coverage per
  family.

The sandbox behind the paper is `sandbox.zip` on the
[release page](https://github.com/MFaisalZaki/Planiverse/releases). Unzip it beside the
repository and `report` regenerates every number in the paper from it.

## Bringing the paper up to date

The paper compares five planners. The code now runs seven, and the two rollout planners have no
results in the released sandbox, so the report shows them as `MISSING` until their arrays are run.
What has to happen, in order:

1. **Run the ten new arrays.** `generate` writes `riw-s0` to `riw-s4` and `piiw-s0` to `piiw-s4`
   beside the existing thirteen; with every cartridge present that is 9,380 runs on top of the
   paper's 12,194. Nothing already run needs repeating: the protocol, the limits and the other
   planners' parameters are unchanged. Re-release `sandbox.zip` afterwards.
2. **The planner section.** Two new paragraphs, with the adaptations stated the way the paper
   states SIW's and BFWS's:
   - Rollout IW (Bandres, Bonet and Geffner, AAAI 2018): novelty measured against the depth an
     atom was first seen at, rollouts instead of a breadth-first frontier, a per-decision budget
     of 1,000 expansions, subtree reuse, the novelty table reset at every committed action. The
     reward is the drop in the environment's progress measure, since there is no score; a goal
     ends the search as soon as a rollout finds one; and a dead end backs up minus infinity,
     because `is_terminal` means no goal is reachable, where the paper's Atari reading would score
     the step into it.
   - π-IW (Junyent, Jonsson and Gómez, ICAPS 2019): the same lookahead with a budget of 100
     expansions per decision, rollouts sampled from a policy network trained online by
     cross-entropy toward softmax(R/τ) over the root's returns, one Adam step per decision from a
     replay buffer, and learning carried across episodes for as long as the budget lasts. The
     network is a one-hidden-layer model over hashed literals rather than a convolutional one over
     pixels; returns are scaled to [0, 1] among the root's children before the softmax so one
     temperature serves every environment; the environment's literals are the novelty atoms,
     with the paper's binarised hidden layer available as an option.
   - The two citations in the bibliography.
3. **The protocol paragraph.** "Five planner configurations" becomes seven; "five seeds for the
   two stochastic planners" becomes five seeds for the four that take one, with the note that
   for Rollout IW and π-IW the seed drives the rollouts and, for π-IW, the network's
   initialisation and its training batches. The parameter table gains the two rows in
   [The protocol](#the-protocol) above.
4. **Table 2 (coverage)** gains columns `RIW` and `PIIW`, as mean over seeds with the standard
   deviation in brackets; `coverage.tex` already emits them. The bold total may move.
5. **Table 3 (statuses)** gains two rows and the median solve time for each; `statuses.tex`
   already emits them. Both planners end an episode at a dead end or the step cap, so expect
   `UNSOLVED` rather than `TIMEOUT` to dominate their rows on the puzzles.
6. **The cactus plot** gains two curves. The overlap and runtime figures stay as they are: they
   compare the three deterministic width planners, and a seeded planner has no single time per
   instance to put on them.
7. **The numbers in the prose**, all read from `facts.txt`: coverage per seed, what each rollout
   planner solved in some seed that BFWS never did and the reverse, the medians, and the per-family
   coverage line the report now writes for every seeded planner. The claims comparing the width
   family against the sampling planners have to be reworded, since Rollout IW is a width planner
   that samples.
8. **The discussion.** The one result worth a paragraph is whichever way it falls: whether
   resetting the novelty table per decision lets Rollout IW(1) reach instances Iterated Width
   needed width 3 or 4 for, and whether π-IW's policy shortens its plans and its expansions
   against Rollout IW's on the environments where the progress measure is dense.

Unrelated to the planners but still pending: the caption of the overlap figure describes four
groups and the figure has five, and the bar order the report writes is not the caption's.
