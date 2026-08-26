# Planiverse

A Python library for **planning with simulators**.

Classical planners need a declarative model of the world. Many interesting problems don't have one —
they have a *simulator*: an epidemic model, a network attack emulator, a Game Boy running Super Mario
Land. Planiverse wraps those simulators behind one small, uniform interface so that a search-based
planner can expand states, test goals, and validate plans without knowing what is underneath.

Every environment answers the same four questions:

- What is the initial state? (`reset`)
- What can happen next? (`successors`)
- Am I done, and did I win? (`is_goal` / `is_terminal`)
- What does this plan actually do? (`simulate`)

## Environment catalog

| Environment | `make()` name | Instances | Tags | Docs |
|---|---|---|---|---|
| Water distribution | `water_network` | 9 contamination scenarios | infrastructure | [docs](docs/environments/water-distribution.md) |
| Power grid | `power_grid` | 9 contingencies | infrastructure | [docs](docs/environments/power-grid.md) |
| Crop management | `crop_management` | 22 growing seasons | agriculture | [docs](docs/environments/crop-management.md) |
| Epidemic control | `epidemic` | 7 scenarios (SIR / SIRV / COVID) | health, policy | [docs](docs/environments/epidemic-control.md) |
| Network attack | `network_attack` | 18 NASim benchmarks | security | [docs](docs/environments/network-attack.md) |
| Manufacturing | `manufacturing` | 7 demand/capacity instances | operations | [docs](docs/environments/manufacturing.md) |
| Urban planning | `urban_planning` | 2 cities (Kendall Square, St Andrews) | policy | [docs](docs/environments/urban-planning.md) |
| Puzznic | `puzznic` | 50 levels | game | [docs](docs/environments/puzznic.md) |
| Puzznic (Game Boy) | `puzznic_gb` | 128 rounds (needs a ROM you supply) | game, emulator | [docs](docs/environments/puzznic-gb.md) |
| Flipull | `flipull` | 10 stages | game | [docs](docs/environments/flipull.md) |
| Flipull (Game Boy) | `flipull_gb` | 32 stages (needs a ROM you supply) | game, emulator | [docs](docs/environments/flipull-gb.md) |
| Platformer | `platformer` | 8 levels | game, platformer | [docs](docs/environments/platformer.md) |
| Super Mario Land | `super_mario_land` | 12 levels (needs a ROM you supply) | game, emulator | [docs](docs/environments/super-mario-land.md) |

```python
from planiverse.environments import list_environments, make

[spec.name for spec in list_environments(tag="infrastructure")]
# ['power_grid', 'water_network']

env = make("water_network", index=8)
state, info = env.reset()
```

Every environment lives in one flat `planiverse.environments` package behind one base class.
What used to be two package trees — `real_world_problems` and `retro_games` — is now a `tags`
field on a registry entry, because that split recorded where an environment came from rather
than what a planner could do with it. See [Architecture](#architecture).

PDDL domains are also supported, through a [PDDLGym](https://github.com/tomsilver/pddlgym) wrapper —
see [The Simulator facade](#the-simulator-facade).

## Installation

Requires Python **≥ 3.11, < 3.14** — numba and scipy have no 3.14 wheels yet, and building them from
source needs a system OpenBLAS.

```bash
git clone https://github.com/MFaisalZaki/Planiverse.git
cd Planiverse
pip install -e ".[dev]"      # --extras dev adds pytest
```

Or, with poetry:

```bash
poetry env use python3.12
poetry install --extras dev
```

One install gets you every environment, on every supported Python.

`tests/test_packaging.py` walks the import graph from each environment's entry point and fails if
anything it reaches is undeclared. That is how `gym` — imported by the simulator facade and the
epidemic environment, but declared nowhere, working only because pddlgym happened to pull it in —
stopped being able to go missing.

### PDDLGym is vendored

`pddlgym 0.0.7` — the last release — requires `pillow <10`, and Pillow published no wheels
for Python 3.13. Building Pillow 9.5.0 from source there fails inside its own `setup.py`,
which reads the version by `exec`-ing a file and pulling `__version__` back out of
`locals()` — something [PEP 667](https://peps.python.org/pep-0667/) stopped working in 3.13:

```
File "<string>", line 26, in get_version
KeyError: '__version__'
```

That took the whole install down on 3.13, PDDL support or not, and pip has no way to override
another package's requirements (`--constraint` can only narrow a range, never widen one). So
PDDLGym is vendored under
[`planiverse/simulator/wrappers/pddlgym/`](planiverse/simulator/wrappers/pddlgym/), the same
way EpiPolicy is, with the one-line fix its own `TODO` asked for — `Image.ANTIALIAS` (removed
in Pillow 10) becomes `Image.Resampling.LANCZOS`. The pin is then unnecessary, and every
supported Python gets the full wrapper.

[`VENDORING.md`](planiverse/simulator/wrappers/pddlgym/VENDORING.md) records the source
version, every edit, and how to re-sync. `matplotlib`, `imageio` and `scikit-image` are
declared on its behalf.

### ROMs

Only the Game Boy environments need anything extra. The water, power grid and crop
environments ship their benchmark data inside their dependencies, so they run offline with
nothing to supply.

The three Game Boy environments additionally need a ROM each — `SuperMarioLand.gb`,
`Puzznic (J).gb` and `Flipull (USA).gb` — which are **not** and cannot be distributed with this repo.
See their docs: [Super Mario Land](docs/environments/super-mario-land.md),
[Puzznic (Game Boy)](docs/environments/puzznic-gb.md), [Flipull (Game Boy)](docs/environments/flipull-gb.md).

## Tests

```bash
pytest                  # the whole suite
pytest -m "not slow"    # skip the slow epidemic/search tests
```

Tests for an environment whose dependencies are missing skip rather than fail, so the suite is
runnable from a partial install. The tests that need a copyrighted ROM are opt-in — point the
matching environment variable at one to run them:

```bash
PLANIVERSE_SML_ROM=/path/to/SuperMarioLand.gb pytest tests/test_super_mario.py
PLANIVERSE_PUZZNIC_ROM="/path/to/Puzznic (J).gb" pytest tests/test_puzznic_gb.py
PLANIVERSE_FLIPULL_ROM="/path/to/Flipull (USA).gb" pytest tests/test_flipull_gb.py
```

The two Taito environments are still covered without one:
[`tests/fake_puzznic_rom.py`](tests/fake_puzznic_rom.py) and
[`tests/fake_flipull_rom.py`](tests/fake_flipull_rom.py) assemble synthetic cartridges that reproduce
each game's memory layout, so booting, stage selection, field decoding, calibration and settling are
all tested against a real emulator.

[`tests/test_interface.py`](tests/test_interface.py) checks the contract below uniformly across every
environment; the other modules cover per-environment behaviour.

## Quickstart

Puzznic is the dependency-free environment, so it's the fastest way to see the interface:

```python
from planiverse.environments.puzznic import PuzznicGame

env = PuzznicGame()
env.fix_index(0)              # choose the instance *before* reset
state, info = env.reset()

print(state)
# ######
# #12c #
# ###  #
# #    #
# #2  1#
# ##21##
# ######

for action, successor in env.successors(state):
    print(action, env.is_goal(successor), env.is_terminal(successor))
# left False False
# right False False
# up False False
# down False False
```

Plans are lists of actions, and `simulate` replays one into a state trace:

```python
trace = env.simulate(['left', 'down', 'right'])
print(sum(trace[-1].score))     # score is a list of per-step awards
```

## Core concepts

### The environment interface

An environment is a plain Python class — there is no registry and no metaclass. It subclasses
`RealWorldProblem` or `RetroGame` (both are little more than markers, used by `Simulator` to
recognise the object) and implements as much of this contract as it needs:

| Method | Returns | Notes |
|---|---|---|
| `reset()` | `(state, info)` | Builds the initial state. Call `fix_index` first. |
| `fix_index(index)` | — | Selects which scenario/level/instance to load. |
| `successors(state)` | `[(action, next_state), ...]` | The expansion step. Self-loops are filtered out. |
| `is_goal(state)` | `bool` | |
| `is_terminal(state)` | `bool` | Dead end — no goal reachable from here. |
| `simulate(plan)` | `[state, ...]` | Replays a plan from the initial state. |
| `step(action)` | `(state, reward)` | Optional; stateful stepping. |
| `validate(plan)` | `bool` | Optional. |
| `get_actions()` | `[action, ...]` | Optional. |

Not every environment implements every method. What is actually there today:

| | `reset` | `fix_index` | `successors` | `is_goal` | `is_terminal` | `simulate` | `step` | `validate` | `get_actions` |
|---|---|---|---|---|---|---|---|---|---|
| `PuzznicGame` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `PuzznicGBEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `FlipullGame` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `FlipullGBEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `PlatformerGame` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `WaterNetworkEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `PowerGridEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `CropEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `SuperMarioEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `EnvNASim` | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | — | — | — |
| `EpiEnv` | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | — | — | — |
| `MfgEnv` | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | — | — | — |
| `UrbanPlanningEnv` | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | — | — | — |
| `PDDLGymEnv` | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | — | ✅ | — |

⚠️ `is_terminal` returns a hard-coded `False` in these environments: they have no dead ends, or
detecting them is left to the planner. The two Puzznics, `FlipullGame` and Super Mario Land are the
ones that really compute a positional dead end — and `FlipullGame`'s is *exact*, because the rules
are known in Python and it can ask outright whether any throw would connect. `FlipullGBEnv` computes
one too, but only the clock running out: the emulator does not know what a throw hits.
The three simulator-backed environments compute real ones too: a water network whose service has
collapsed, a blacked-out grid, and a growing season whose water budget is spent.

`Environment.capabilities()` reports this per class, so the table above is checked against the code
rather than trusted — `tests/test_interface.py` asserts the two agree.

Note that `validate` is now provided by the base class for everything, derived from `simulate` and
`is_goal`, so no environment writes it out. `step` and `get_actions` have base defaults too, but
theirs only explain their own absence, which is why "does the class override it" is the wrong test
and `capabilities()` asks whether the method would actually do something.

### States and `literals`

Every state object carries a `literals` attribute: a `frozenset` of facts about the state. Native
environments spell them as predicate-like strings; the PDDLGym wrapper passes pddlgym `Literal`
objects straight through, so only "frozenset" is common to both.

```python
sorted(state.literals)[:4]
# ['at(box-1, 1, 1)', 'at(box-1, 4, 4)', 'at(box-1, 5, 3)', 'at(box-2, 1, 2)']
```

This is the bridge back to symbolic planning. Planners use `literals` as the visited-set key, and
width-based methods (IW and friends) use them as the atoms whose novelty they measure. It is also
where each environment makes its central modelling decision: what counts as *the same state*. The
choice differs sharply per environment — Puzznic's literals are exact, while the epidemic
environment's literals (and `__eq__`) deliberately blur nearby states together so that search over a
continuous compartmental model terminates. Each environment's doc has a "State representation"
section spelling out what it chose and what that costs you.

States also commonly expose `depth`, and define `__eq__` (and sometimes `__hash__` and `__lt__`, the
latter so they can be tie-broken inside a priority queue).

### The `fix_index` pattern

Environments are constructed empty and loaded by integer index:

```python
env = EnvNASim()
env.fix_index(3)        # 'small' benchmark
state, _ = env.reset()  # fix_index must come first — reset asserts on it
```

The index is a stable handle for "instance *n* of this environment", which is what a benchmark runner
wants. The mapping from index to instance is listed in each environment's doc.

### The Simulator facade

`Simulator` is a thin adapter that lets a planner accept either a native Planiverse environment or a
PDDLGym environment without caring which:

```python
import pddlgym
from planiverse.simulator.simulator import Simulator

sim = Simulator(pddlgym.make("PDDLEnvBlocks-v0"))   # wrapped in PDDLGymEnv
sim.simulator.fix_index(0)                          # pick one of the domain's problem files
state, info = sim.reset()
for action, successor in sim.successors(state):
    ...
```

It dispatches on type: a gym `OrderEnforcing` object is wrapped in
[`PDDLGymEnv`](planiverse/simulator/wrappers/pddlgymenv.py), while a `RetroGame` or
`RealWorldProblem` is used as-is. Anything else raises an assertion.

A PDDL domain ships several problem files, and PDDLGym picks one *at random* on every reset. The
wrapper pins problem `0` on construction so that `reset()` is repeatable like every other
environment's; `fix_index(i)` selects a different one.

Passing a native environment through `Simulator` adds nothing but delegation — and because it
forwards `step`/`validate`/`get_actions` that most environments don't implement, calling those
through the facade will raise `AttributeError`/`NotImplementedError` rather than fail gracefully. Use
it when you need PDDL and simulator environments behind one interface; use the environment directly
otherwise.

## Rendering a trace

```python
from planiverse.rendering import render_trace

trace = env.simulate(plan)
render_trace(trace, "plan.png", actions=plan, env=env)
render_trace(trace, "plan.pdf", actions=plan, env=env, per_page=6)
```

Text states are typeset; Game Boy states become real console screenshots when you pass
`gamerom=`. Goals and dead ends are marked. See [docs/rendering.md](docs/rendering.md).

## Planners

| Family | Where | What it needs from an environment |
|---|---|---|
| Width-based — IW(k), Iterated Width, SIW, BFWS | [`planiverse/planners/width/`](planiverse/planners/width/) | `successors` and `literals`; a `progress` callback helps |
| MCTS / UCT | [`planiverse/planners/mcts.py`](planiverse/planners/mcts.py) | `successors`; a `reward` callback helps a lot |
| Future State Maximization | [`planiverse/planners/fsx.py`](planiverse/planners/fsx.py) | `successors`, and **nothing else** — no goal, no heuristic |
| Tree search / A* | [`planiverse/planners/super_mario_planner_gb.py`](planiverse/planners/super_mario_planner_gb.py) | a heuristic and a cost function |

```python
from planiverse.planners.width import IWSearch, BFWSSearch, Budget

env.fix_index(0)
result = IWSearch(width=2).solve(env, Budget(max_expansions=5000, max_seconds=60))
if result:
    env.validate(result.plan)
```

The width-based family is documented in [docs/planners/width-based.md](docs/planners/width-based.md),
including what has to change when the task is a simulator: there is no goal conjunction to
count, so SIW and BFWS take a `progress` callback instead; expansions are expensive, so every
search takes a budget and reports what it spent; and dead ends are real, which turns out to
be worth a solved instance to SIW.

MCTS and Future State Maximization are in
[docs/planners/sampling-based.md](docs/planners/sampling-based.md). FSX is the odd one: it is
given no goal and no heuristic at all and picks whichever action leaves the most futures
open, which makes `option_count` a goal-free measure of how close a state is to being stuck
— useful as a heuristic for the other planners precisely when heuristics are hardest to
write.

## Benchmarking

`planiverse-bench` runs every planner over every environment, on a SLURM cluster or on one
machine, and turns the results into tables and plots. It follows
[pyPMTEvalToolkit](https://github.com/pyPMT/pyPMTEvalToolkit): an experiment is a directory of
JSON, a sandbox is a directory of results, and the stages between them run independently — so a
benchmark can be prepared on a laptop, run on a cluster, and analysed somewhere else again.

```bash
./setup_benchmark.sh                      # asks about limits and your Game Boy cartridges
bash sandbox/slurm/submit_all.sh          # or: bash sandbox/run_local.sh 8
planiverse-bench analyze   --sandbox-dir sandbox
planiverse-bench report    --sandbox-dir sandbox
```

`setup_benchmark.sh` runs `init`, `discover` and `generate`, and asks the one thing nothing
else can work out: where your Puzznic, Flipull and Super Mario Land cartridges are. They are
copyrighted and cannot ship here, so supplying them is what lets an emulated environment be
compared against its pure-Python twin under the same planners and limits; skip one and it is
reported as skipped rather than quietly dropped. `--yes` takes every default and asks nothing.

`generate` writes one **job array per planner** — a benchmark is thousands of short runs, and a
scheduler handling them as thousands of jobs spends longer scheduling than computing. Arrays are
split at the site's `MaxArraySize`, throttled with `%N` so a shared partition survives, and given
time and memory headroom above the harness's own limits so that a timeout is recorded as a
`TIMEOUT` row rather than vanishing as a killed job.

Every run ends in a status — `SOLVED`, `INVALID`, `UNSOLVED`, `TIMEOUT`, `NODEOUT`, `MEMOUT`,
`ERROR`, `UNSUPPORTED`, `MISSING` — because a failure has to be recorded rather than raised. The
expected set of runs comes from `tasks.json`, so a job that never ran is counted as `MISSING`
rather than quietly improving a planner's coverage.

By default the benchmark covers **every instance of every environment it can run**, cartridge
ones included. `iw` runs as Iterated Width up to a bound of 1000 rather than at a width someone
picked: IW(k) is a family, and which member a problem needs is a property of the problem. The
loop stops when a width solves it, the budget runs out, or a width covers the reachable space
without pruning anything for novelty — which is also a proof that there is no plan.

A real run — 7 planners over 18 tasks, a Puzznic cartridge supplied, 20-second limit:

```
planner       solved  of   coverage  total time  median   plan len
bfws-1        15      18   83%       76.7s       4.49s    16.3
bfws-2        15      18   83%       75.2s       3.58s    16.3
iw            14      18   78%       98.4s       4.93s    12.2
mcts          8       18   44%       145.3s      20.03s   4.6
siw-1 *       7       18   39%       33.8s       0.56s    17.6
siw-2 *       7       18   39%       13.6s       0.06s    6.3
fsx           2       18   11%       26.8s       13.42s   2.0

* at least one UNSOLVED row from this planner is not a proof that there is no plan.
```

`puzznic` and `puzznic_gb` come out identical planner for planner in the per-environment table
— the cartridge and its pure-Python twin agreeing, which is the comparison supplying a ROM
buys you.

The full documentation is in [docs/benchmark.md](docs/benchmark.md), including the progress
measures SIW and BFWS need per environment, the three environments that have none and why, and
how to point the harness at a Game Boy cartridge.

## Writing a planner

Environments are planner-agnostic. [`planiverse/planners/super_mario_planner_gb.py`](planiverse/planners/super_mario_planner_gb.py)
contains a small best-first `TreeSearchPlanner` that works against any environment implementing the
contract:

```python
class TreeSearchPlanner:
    def search(self, state, env, hfn, costfn):
        queue = PriorityQueue()
        visited = set()
        queue.push(([state], [], []), 0)
        while not queue.isEmpty():
            state_trace, action_trace, ltl_trace = queue.pop()
            state = state_trace[0]
            if env.is_goal(state):  return action_trace
            if state.literals in visited: continue
            visited.add(state.literals)
            for action, successor_state in env.successors(state):
                ...
```

The pieces you supply are a `Heuristic` and a `CostFunction`, both callables over states and traces.
`SuperMarioPlanner` in the same file is a worked example (a re-implementation of Robin Baumgarten's
A* Mario agent), and is discussed in the [Super Mario Land docs](docs/environments/super-mario-land.md#planner).

Note that `PriorityQueue` pushes `(priority, item)` tuples, so ties compare the items themselves —
which is why state and action classes define `__lt__`.

## Architecture

One flat package, one base class, and a registry.

```
planiverse/environments/
├── base.py          # Environment — the six-method contract, and nothing else
├── registry.py      # EnvironmentSpec per environment: instances, tags, deps, state identity
└── <one module or subpackage per environment>
```

**Why it changed.** There used to be two base classes in two package trees,
`RealWorldProblem` and `RetroGame`. That split recorded an environment's *provenance*, not
its *capabilities*, so nothing could usefully dispatch on it — the `Simulator` facade ended
up asking `isinstance(env, RetroGame) or isinstance(env, RealWorldProblem)`, two branches
doing identical work. Meanwhile the distinctions a planner actually cares about were written
down nowhere.

So the taxonomy became data. `EnvironmentSpec` carries what you might select on:

| Field | What it tells a planner |
|---|---|
| `deterministic` | whether expanding a state twice gives the same children |
| `state_identity` | `value`, `path` or `snapshot` — **how branching is possible at all** |
| `requires` | third-party modules, so listing the catalogue imports none of them |
| `needs_rom` | needs a copyrighted file the user supplies |
| `tags` | `game`, `infrastructure`, `continuous-dynamics`, … — the old split, as one field among several |

`state_identity` is the one worth understanding. A `value` state carries its own contents and
expanding is pure. A `snapshot` state carries a serialised simulator image — a Game Boy
save-state. A `path` state *is* the decision sequence, replayed on demand, which is sound
only because the simulator is deterministic. Most simulators are step-only and cannot be
rewound; that is the property that decides whether something can be a Planiverse environment
at all, and it now has a name.

`Simulator` dispatches structurally (`implements_contract`), so an environment brought from
outside works without inheriting from anything.

**Old import paths still work**, via shims that raise `DeprecationWarning`:

```python
from planiverse.problems.retro_games.puzznic import PuzznicGame   # works, warns
from planiverse.environments.puzznic import PuzznicGame           # the new home
```

## Adding an environment

1. Subclass `Environment` (`planiverse/environments/base.py`) and implement the six methods.
2. Define a state class exposing `literals`, `__eq__`, and — if search will hash it — `__hash__`.
   Decide deliberately how coarse `literals` should be; that decision is your state space.
3. Implement `reset`, `fix_index`, `successors`, `is_goal`, `is_terminal`, and `simulate`.
4. Filter self-loops out of `successors` (`if successor_state == state: continue`) — every bundled
   environment does this, and planners rely on it. Check that it can actually fire: if `literals`
   include a step counter, no successor ever equals its parent and the filter is dead code (this is
   what happens in [urban planning](docs/environments/urban-planning.md#known-quirks)).
5. Add an `EnvironmentSpec` to `planiverse/environments/registry.py` — that is what puts it
   in the catalogue and in `make()`.
6. Add a doc under `docs/environments/` and a row to the catalog above.

## Repository layout

```
planiverse/
├── environments/
│   ├── base.py                         # Environment — the one base class
│   ├── registry.py                     # EnvironmentSpec, list_environments(), make()
│   ├── puzznic.py                      # PuzznicGame
│   ├── puzznic_gb.py                   # PuzznicGBEnv (PyBoy)
│   ├── flipull.py                      # FlipullGame
│   ├── flipull_gb.py                   # FlipullGBEnv (PyBoy)
│   ├── platformer.py                   # PlatformerGame
│   ├── super_mario_land.py             # SuperMarioEnv (PyBoy)
│   ├── epidemic_control/               # EpiEnv + vendored EpiPolicy + jsons/
│   ├── network_attack/                 # EnvNASim (wraps NASim)
│   ├── manufacturing/                  # MfgEnv + data/
│   ├── urban_planning/                 # UrbanPlanningEnv + cities/
│   ├── water_network/                  # WaterNetworkEnv (WNTR/EPANET)
│   ├── power_grid/                     # PowerGridEnv (Grid2Op)
│   └── crop_management/                # CropEnv (PCSE/WOFOST)
├── problems/                           # deprecated shims for the old import paths
├── planners/
│   ├── width/                          # IW, Iterated Width, SIW, BFWS
│   ├── fsx.py                          # FSXPlanner (future state maximisation)
│   ├── mcts.py                         # MCTSPlanner (UCT)
│   └── super_mario_planner_gb.py       # TreeSearchPlanner, SuperMarioPlanner
├── rendering/                          # traces to PNG and PDF
├── benchmark/                          # planiverse-bench: run the planners, generate SLURM jobs
│   ├── cli.py                          # init / discover / generate / solve / analyze / report
│   ├── config.py                       # exp-details.json and planners/*.json
│   ├── catalogue.py                    # which planners exist and how to build them
│   ├── measures.py                     # per-environment progress measures
│   ├── discovery.py                    # resolving (environment, index) task lists
│   ├── runner.py                       # one run, under limits, with a status
│   ├── slurm.py                        # job arrays, submit_all.sh, run_local.sh
│   ├── analysis.py                     # coverage, head to head, IPC scores, CSV
│   └── report.py                       # text and LaTeX tables, cactus and scatter plots
└── simulator/
    ├── simulator.py                    # Simulator facade
    └── wrappers/
        ├── base.py                     # SimulatorBase interface
        ├── pddlgymenv.py               # PDDLGym adapter
        └── pddlgym/                    # vendored PDDLGym 0.0.7 — see its VENDORING.md
docs/environments/                      # per-environment documentation
docs/benchmark.md                       # the benchmark harness
setup_benchmark.sh                      # interactive benchmark setup; asks for the cartridges
tests/
├── sm83.py                             # minimal SM83 assembler, for the test cartridges
├── fake_puzznic_rom.py                 # synthetic Game Boy ROM with Puzznic's memory layout
└── fake_flipull_rom.py                 # synthetic Game Boy ROM with Flipull's memory layout
```

There was a `dev/` scratch directory; it is gone. It held two files. `dev.py` was stale — it
imported names that no longer exist (`SuperMario`, `super_mario_bros_grid`,
`super_mario_planner_tile`) and could not run — and it was the only thing that ever imported
`pcg_benchmark`. `earthmodel.py` was a vendored copy of the c:GLOBAL gym environment
(© Felix Strnad), kept for a port that never happened; it never implemented the Planiverse
interface. Both are recoverable from git history if the port is ever picked up.

`chex`, `flax`, `jaxmarl` and `dill` were declared as required dependencies but are imported nowhere
in the library, so they are gone too; nothing outside `epipolicy/**/deprecated/` referenced them. The
`gym` they were sitting next to went the other way — it was imported and never declared.

## Attribution

Planiverse adapts several upstream simulators. Each is credited in its own doc; the sources are:

| Environment | Upstream |
|---|---|
| Epidemic control | [EpiPolicy](https://github.com/huda-lab/RL-Epidemic-Benchmark) (vendored under `epidemic_control/epipolicy/`) |
| Network attack | [NASim](https://github.com/MFaisalZaki/NetworkAttackSimulator) (fork), [PenGym](https://github.com/cyb3rlab/PenGym) |
| Manufacturing | [mfgrl](https://github.com/torayeff/mfgrl) |
| Urban planning | *AI Agent as Urban Planner: Steering Stakeholder Dynamics in Urban Planning via Consensus-based Multi-Agent Reinforcement Learning* |
| Super Mario Land, Puzznic (GB), Flipull (GB) | [PyBoy](https://github.com/Baekalfen/PyBoy) |
| Water distribution | [WNTR](https://github.com/USEPA/WNTR) (US EPA's EPANET wrapper) |
| Power grid | [Grid2Op](https://github.com/Grid2Op/grid2op) (RTE) |
| Crop management | [PCSE / WOFOST](https://github.com/ajwdewit/pcse) (Wageningen University) |

The flood/transport environment ([floods_transport_rl](https://github.com/MLSM-at-DTU/floods_transport_rl))
is referenced as a planned addition but is not yet in the tree.

## Status

- [x] README and per-environment docs
- [x] Super Mario Land via PyBoy, with world/level selection wired into `reset`
- [x] NASim network attack
- [x] Test suite (`poetry run pytest`)
- [x] Water distribution, power grid and crop management — three simulator-backed
      environments whose transitions are solves, not add/delete lists
- [x] One flat `planiverse.environments` package with a registry, replacing the
      `real_world_problems` / `retro_games` split
- [ ] Flood application
- [ ] Optional dependency groups, so one environment doesn't pull in all of them
- [ ] Replace `pddlgym` (unmaintained, pins `pillow <10`, caps the PDDL wrapper at Python 3.12)
- [ ] `is_terminal` dead-end detection for the four environments that hard-code `False`
- [ ] Confirm Super Mario Land's level-complete address (`0xDFE8`) and enemy tile IDs
- [ ] `SuperMarioPlanner.search` returns `None` and has no replanning loop
- [x] Run `FlipullGBEnv` against a real `Flipull (USA).gb`, and correct what the memory map had
      wrong about the throw
- [x] Flipull stage selection: all 32, via the loader's own stage digits
- [x] A pure-Python Flipull twin (`FlipullGame`), with generated-and-verified stages and exact
      dead-end detection
- [ ] Work out what a Flipull throw actually hits — every row connects, so it is not simply the
      first block in the player's row. Until it is settled, `FlipullGame` is a Flipull-*like*
      environment with a stated rule set rather than a clone of the cartridge
- [x] A dependency-free platformer (`PlatformerGame`) to stand where a pure-Python Super
      Mario Land would. It is a genre counterpart with stated physics, **not** a twin of the
      cartridge — nothing in it was read off that ROM, and the docs lead with that
- [ ] An actual pure-Python Super Mario Land. Deliberately not attempted: reverse-engineering
      a physics platformer is a far larger job than a turn-based puzzle, and a half-modelled
      one would look like a prediction of the cartridge without being one
- [ ] Flipull's second stage table at `$3A4E`, reached through the RNG — a bonus course, unexplored

## License

GPL-3.0. See [LICENSE](LICENSE).
