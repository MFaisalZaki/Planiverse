# Planiverse

A Python library for **planning with simulators**.

Classical planners need a declarative model of the world. Many interesting problems do not have one;
they have a *simulator* instead: a water distribution network, a network attack emulator, a Game Boy running
Super Mario Land. Planiverse wraps those simulators behind one small, uniform interface so that a search-based
planner can expand states, test goals, and validate plans without knowing what is underneath.

Every environment answers the same four questions:

- What is the initial state? (`reset`)
- What can happen next? (`successors`)
- Am I done, and did I win? (`is_goal` / `is_terminal`)
- What does this plan actually do? (`simulate`)

## Environment catalogue

| Environment | `make()` name | Instances | Tags | Docs |
|---|---|---|---|---|
| Water distribution | `water_network` | 9 contamination scenarios | operational, infrastructure | [docs](docs/environments/water-distribution.md) |
| Power grid | `power_grid` | 9 contingencies | operational, infrastructure | [docs](docs/environments/power-grid.md) |
| Crop management | `crop_management` | 22 growing seasons | operational, agriculture | [docs](docs/environments/crop-management.md) |
| Network attack | `network_attack` | 18 NASim benchmarks | security | [docs](docs/environments/network-attack.md) |
| Puzznic | `puzznic` | 128 levels | game | [docs](docs/environments/puzznic.md) |
| Puzznic (Game Boy) | `puzznic_gb` | 128 rounds (needs a ROM you supply) | game, emulator | [docs](docs/environments/puzznic-gb.md) |
| Flipull | `flipull` | 32 stages | game | [docs](docs/environments/flipull.md) |
| Flipull (Game Boy) | `flipull_gb` | 32 stages (needs a ROM you supply) | game, emulator | [docs](docs/environments/flipull-gb.md) |
| Adventures of Lolo | `lolo` | 163 rooms | game | [docs](docs/environments/lolo.md) |
| Adventures of Lolo (Game Boy) | `lolo_gb` | 163 rooms (needs a ROM you supply) | game, emulator | [docs](docs/environments/lolo-gb.md) |
| Amazing Tater | `amazing_tater` | 105 rooms | game | [docs](docs/environments/amazing-tater.md) |
| Amazing Tater (Game Boy) | `amazing_tater_gb` | 105 rooms (needs a ROM you supply) | game, emulator | [docs](docs/environments/amazing-tater-gb.md) |
| Super Mario Land | `super_mario_land` | 12 levels | game, platformer | [docs](docs/environments/super-mario-land.md) |
| Super Mario Land (Game Boy) | `super_mario_land_gb` | 12 levels (needs a ROM you supply) | game, emulator | [docs](docs/environments/super-mario-land-gb.md) |

```python
from planiverse.environments import list_environments, make

[spec.name for spec in list_environments(tag="operational")]
# ['crop_management', 'power_grid', 'water_network']

env = make("water_network", index=8)
state, info = env.reset()
```

Every environment lives in one flat `planiverse.environments` package behind one base class.
What used to be two package trees (`real_world_problems` and `retro_games`) is now a `tags`
field on a registry entry, because that split recorded where an environment came from rather
than what a planner could do with it. The catalogue falls into three families: `game`,
`operational` (an agent running a system it is responsible for, whether that is a power
grid or a production line) and `security`, where the agent probes a network rather than
operates it. See [Architecture](#architecture).

## Installation

Requires Python **≥ 3.11, < 3.14**: numba and scipy have no 3.14 wheels yet, and building them from
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
anything it reaches is undeclared; a dependency that only works because another package happens
to pull it in cannot go missing silently.

### ROMs

Only the Game Boy environments need anything extra. The water, power grid and crop
environments ship their benchmark data inside their dependencies, so they run offline with
nothing to supply.

The five Game Boy environments need a ROM each on top of that: `SuperMarioLand.gb`,
`Puzznic (J).gb`, `Flipull (USA).gb`,
`Adventures of Lolo (U) [S][!].gb` and `Amazing Tater (U).gb`. These are **not** and cannot
be distributed with this repo. See their docs: [Super Mario Land](docs/environments/super-mario-land.md),
[Puzznic (Game Boy)](docs/environments/puzznic-gb.md), [Flipull (Game Boy)](docs/environments/flipull-gb.md),
[Adventures of Lolo (Game Boy)](docs/environments/lolo-gb.md),
[Amazing Tater (Game Boy)](docs/environments/amazing-tater-gb.md).

## Tests

```bash
pytest                  # the whole suite
pytest -m "not slow"    # skip the slow search tests
```

Tests for an environment whose dependencies are missing skip rather than fail, so the suite is
runnable from a partial install. The tests that need a copyrighted ROM are opt-in; point the
matching environment variable at one to run them:

```bash
PLANIVERSE_SUPER_MARIO_LAND_ROM=/path/to/SuperMarioLand.gb pytest tests/test_super_mario_land_gb.py
PLANIVERSE_PUZZNIC_ROM="/path/to/Puzznic (J).gb" pytest tests/test_puzznic_gb.py
PLANIVERSE_FLIPULL_ROM="/path/to/Flipull (USA).gb" pytest tests/test_flipull_gb.py
PLANIVERSE_LOLO_ROM="/path/to/Adventures of Lolo (U) [S][!].gb" pytest tests/test_lolo_gb.py
PLANIVERSE_AMAZING_TATER_ROM="/path/to/Amazing Tater (U).gb" pytest tests/test_amazing_tater_gb.py
```

The two Taito environments are still covered without one:
[`tests/fake_puzznic_rom.py`](tests/fake_puzznic_rom.py) and
[`tests/fake_flipull_rom.py`](tests/fake_flipull_rom.py) assemble synthetic cartridges that reproduce
each game's memory layout, so booting, stage selection, field decoding, calibration and settling are
all tested against a real emulator. Amazing Tater takes the other route and needs no
cartridge at all: its level decoder and board decoder are pure functions of bytes, and
[`tests/test_amazing_tater_gb.py`](tests/test_amazing_tater_gb.py) exercises them against synthetic
ROM images and synthetic work RAM.

[`tests/test_interface.py`](tests/test_interface.py) checks the contract below uniformly across every
environment; the other modules cover per-environment behaviour.

## Quickstart

Puzznic is the dependency-free environment, so it is the fastest way to see the interface:

```python
from planiverse.environments.gameboy_py.puzznic import PuzznicGame

env = PuzznicGame()
env.set_index(0)              # choose the instance *before* reset
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

An environment is a plain Python class. It subclasses `Environment` (or just satisfies the
contract, since dispatch is structural) and implements as much of this contract as it needs
(registration is an explicit `EnvironmentSpec` entry, not a metaclass):

| Method | Returns | Notes |
|---|---|---|
| `reset()` | `(state, info)` | Builds the initial state. Call `set_index` first. |
| `set_index(index)` | — | Selects which scenario/level/instance to load. |
| `successors(state)` | `[(action, next_state), ...]` | The expansion step. Self-loops are filtered out. |
| `is_goal(state)` | `bool` | |
| `is_terminal(state)` | `bool` | Dead end: no goal reachable from here. |
| `simulate(plan)` | `[state, ...]` | Replays a plan from the initial state. |
| `step(action)` | `(state, reward)` | Optional; stateful stepping. |
| `validate(plan)` | `bool` | Optional. |
| `get_actions()` | `[action, ...]` | Optional. |

Not every environment implements every method. What is actually there today:

| | `reset` | `set_index` | `successors` | `is_goal` | `is_terminal` | `simulate` | `step` | `validate` | `get_actions` |
|---|---|---|---|---|---|---|---|---|---|
| `PuzznicGame` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `PuzznicGBEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `FlipullGame` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `FlipullGBEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `LoloGame` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `LoloGBEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `SuperMarioLandGame` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `WaterNetworkEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `PowerGridEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `CropEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `SuperMarioLandGBEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `EnvNASim` | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | — | — | — |

⚠️ `is_terminal` returns a hard-coded `False` in these environments: they have no dead ends, or
detecting them is left to the planner. The two Puzznics, `FlipullGame` and Super
Mario Land are the ones that really compute a positional dead end; `FlipullGame`'s is *exact*,
because the rules are known in Python and it can ask outright whether any throw would connect.
`FlipullGBEnv` computes one too, but only the clock running out: the emulator does not know what a
throw hits.
The three simulator-backed environments compute real ones too: a water network whose service has
collapsed, a blacked-out grid, and a growing season whose water budget is spent.

`Environment.capabilities()` reports this per class, so the table above is checked against the code
rather than trusted: `tests/test_interface.py` asserts the two agree.

Note that `validate` is now provided by the base class for everything, derived from `simulate` and
`is_goal`, so no environment writes it out. `step` and `get_actions` have base defaults too, but
theirs only explain their own absence. That is why "does the class override it" is the wrong test,
and `capabilities()` asks whether the method would actually do something.

### States and `literals`

Every state object carries a `literals` attribute: a `frozenset` of predicate-like strings,
facts about the state.

```python
sorted(state.literals)[:4]
# ['at(box-1, 1, 1)', 'at(box-1, 4, 4)', 'at(box-1, 5, 3)', 'at(box-2, 1, 2)']
```

This is the bridge back to symbolic planning. Planners use `literals` as the visited-set key, and
width-based methods (IW and friends) use them as the atoms whose novelty they measure. It is also
where each environment makes its central modelling decision: what counts as *the same state*. The
choice differs sharply per environment: Puzznic's literals are exact, while the water network's
literals deliberately bucket a continuous contamination level so that search over a
continuous model terminates. Each environment's doc has a "State representation"
section spelling out what it chose and what that costs you.

States also commonly expose `depth`, and define `__eq__` (and sometimes `__hash__` and `__lt__`, the
latter so they can be tie-broken inside a priority queue).

### The `set_index` pattern

Environments are constructed empty and loaded by integer index:

```python
env = EnvNASim()
env.set_index(3)        # 'small' benchmark
state, _ = env.reset()  # set_index must come first — reset asserts on it
```

The index is a stable handle for "instance *n* of this environment", which is what a benchmark runner
wants. The mapping from index to instance is listed in each environment's doc.

### Bringing your own environment

Planners call environments directly; there is no wrapper to construct. An environment brought
from outside the library counts as long as it answers the six contract methods:
`implements_contract` checks structurally, so no subclassing is required. There used to be a
`Simulator` facade between planners and environments; once its PDDLGym dispatch was removed it
delegated every call one-to-one, so it went the way of the two-base-class split.

## Rendering a trace

```python
trace = env.simulate(plan)
env.render_trace(trace, "plan.gif")        # an animated GIF, one frame per state
env.render_trace(trace, "plan-frames/")    # a directory of independent PNGs
```

Rendering a trace is one image per state, nothing more: a real console screenshot when the
state can produce one (`env.render_trace` passes a cartridge-backed environment's own ROM
automatically), the state's own text typeset otherwise. The free-standing
`planiverse.rendering.render_trace` takes `gamerom=` explicitly.

On a Game Boy environment, `env.render()` does the same for the positions `step` has played
through, returning the console's own frames rather than the text board:

```python
frames = env.render()          # PIL images, one per de-duplicated step
env.render("play.gif")         # or write them, in any format render_trace spells
```

See [docs/rendering.md](docs/rendering.md).

## Planners

| Family | Where | What it needs from an environment |
|---|---|---|
| Width-based: IW(k), Iterated Width, SIW, BFWS | [`planiverse/planners/width/`](planiverse/planners/width/) | `successors` and `literals`; a `progress` callback helps |
| Rollout IW, and π-IW with a policy it learns as it plans | [`planiverse/planners/width/rollout.py`](planiverse/planners/width/rollout.py), [`policy.py`](planiverse/planners/width/policy.py) | `successors` and `literals`; a `progress` callback stands in for the score |
| MCTS / UCT | [`planiverse/planners/mcts.py`](planiverse/planners/mcts.py) | `successors`; a `reward` callback helps a lot |
| Future State Maximization | [`planiverse/planners/fsx.py`](planiverse/planners/fsx.py) | `successors`, and **nothing else**: no goal, no heuristic |
| Tree search / A* | [`planiverse/planners/super_mario_planner_gb.py`](planiverse/planners/super_mario_planner_gb.py) | a heuristic and a cost function |

```python
from planiverse.planners.width import IWSearch, BFWSSearch, Budget

env.set_index(0)
result = IWSearch(width=2).solve(env, Budget(max_expansions=5000, max_seconds=60))
if result:
    env.validate(result.plan)
```

The width-based family is documented in [docs/planners/width-based.md](docs/planners/width-based.md),
including three things that change when the task is a simulator: (1) there is no goal conjunction
to count, so SIW and BFWS take a `progress` callback instead; (2) expansions are expensive, so
every search takes a budget and reports what it spent; and (3) dead ends are real, and detecting
them is most of what makes a simulator task searchable.

Rollout IW (Bandres, Bonet and Geffner, 2018) and π-IW (Junyent, Jonsson and Gómez, 2019) are
in [docs/planners/rollout-width.md](docs/planners/rollout-width.md): the novelty filter kept,
the breadth-first order replaced by rollouts that commit to an action every few hundred
expansions, and in π-IW a small policy network, trained on the planner's own lookaheads, that
steers the rollouts and can supply the atoms novelty is measured over.

MCTS and Future State Maximization are in
[docs/planners/sampling-based.md](docs/planners/sampling-based.md). FSX is the odd one: it is
given no goal and no heuristic at all and picks whichever action leaves the most futures
open. That makes `option_count` a goal-free measure of how close a state is to being stuck,
useful as a heuristic for the other planners precisely when heuristics are hardest to
write.

## Benchmarking

`planiverse-bench` is the tool paper's evaluation protocol as code: the five planner
configurations the paper compares plus Rollout IW and π-IW, on every instance of every
environment, under a 30-minute wall-clock limit, an 8 GB address-space cap and a
500,000-expansion bound, with five seeds for each of the four planners that take one, on a
SLURM cluster or on one machine. There is no configuration
file, because the protocol is the point.

```bash
tools/setup_benchmark.sh --partition <p> --qos <q>   # venv, install, then `generate`
bash sandbox/submit.sh                               # or: bash sandbox/run_local.sh 8
planiverse-bench report --sandbox-dir sandbox
```

`generate` asks each registered environment how many instances it has and writes one command
per (planner, instance, seed) under `sandbox/cmds/`, and one SLURM job array per planner, or
per seed of a seeded planner, under `sandbox/slurm/`. The Game Boy environments need their
cartridges, which are copyrighted and cannot ship here: pass them to the setup script as
`--rom-puzznic`, `--rom-flipull`, `--rom-lolo`, `--rom-amazing-tater` and
`--rom-super-mario-land`, or export `PLANIVERSE_PUZZNIC_ROM`, `PLANIVERSE_FLIPULL_ROM`,
`PLANIVERSE_LOLO_ROM`, `PLANIVERSE_AMAZING_TATER_ROM` and `PLANIVERSE_SUPER_MARIO_LAND_ROM`
before generating. A flag overrides the variable. An environment without one is skipped and
says so.

Every run ends in exactly one status, written to `sandbox/results/<planner>/<env>__<i>.json`
(`..._<i>__s<seed>.json` for a seeded planner) whatever happened: `SOLVED` (the plan replays to a
goal), `INVALID` (it does not), `UNSOLVED` (the search stopped on its own), `TIMEOUT`,
`NODEOUT`, `MEMOUT`, `ERROR`, `UNSUPPORTED` (the environment could not be built), and
`MISSING`, which `report` assigns to a run that left no file, so a job that never ran cannot
pass for coverage.

`report` writes the paper's two tables (`coverage.tex`, `statuses.tex`), its three figures
(`cactus.pdf`, `overlap_bfws_iw_siw.pdf`, `runtime_bfws_iw_siw.pdf`) and `facts.txt`, the
numbers its prose quotes, into `sandbox/report/`. A seeded planner is reported as its mean over
seeds with the standard deviation, never its best seed. The sandbox behind the paper is attached
to the [release page](https://github.com/MFaisalZaki/Planiverse/releases); unzip it beside the
repository and `report` regenerates every number from it. [docs/benchmark.md](docs/benchmark.md)
has the details, and its last section lists what the paper has to change to take in Rollout IW
and π-IW: two coverage columns, two status rows, two cactus curves, the planner descriptions,
and the numbers its prose quotes.

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

Note that `PriorityQueue` pushes `(priority, item)` tuples, so ties compare the items themselves,
which is why state and action classes define `__lt__`.

## Architecture

One flat package, one base class, and a registry.

```
planiverse/environments/
├── base.py          # Environment — the six-method contract, and nothing else
├── registry.py      # EnvironmentSpec per environment: instances, tags, deps, state identity
├── gameboy/         # the PyBoy-backed environments, grouped around their shared gb.py tail
├── gameboy_py/      # their dependency-free counterparts, so each pair is easy to find
└── <one module or subpackage per environment>
```

**Why it changed.** There used to be two base classes in two package trees,
`RealWorldProblem` and `RetroGame`. That split recorded an environment's *origin*, not
its *capabilities*, so nothing could usefully dispatch on it: the `Simulator` facade ended
up asking `isinstance(env, RetroGame) or isinstance(env, RealWorldProblem)`, two branches
doing identical work. Meanwhile the distinctions a planner actually cares about were written
down nowhere.

So the taxonomy became data. `EnvironmentSpec` carries what you might select on:

| Field | What it tells a planner |
|---|---|
| `deterministic` | whether expanding a state twice gives the same children |
| `state_identity` | `value`, `path` or `snapshot`: **how branching is possible at all** |
| `requires` | third-party modules, so listing the catalogue imports none of them |
| `needs_rom` | needs a copyrighted file the user supplies |
| `tags` | the family (`game`, `operational`, `security`) plus finer ones like `continuous-dynamics` |

`state_identity` is the one worth understanding. A `value` state carries its own contents and
expanding is pure. A `snapshot` state carries a serialised simulator image, a Game Boy
save-state. A `path` state *is* the decision sequence, replayed on demand, which is sound
only because the simulator is deterministic. Most simulators are step-only and cannot be
rewound; that is the property that decides whether something can be a Planiverse environment
at all, and it now has a name.

The contract check is structural (`implements_contract`), so an environment brought from
outside works without inheriting from anything.

## Adding an environment

1. Subclass `Environment` (`planiverse/environments/base.py`) and implement the six methods.
2. Define a state class exposing `literals`, `__eq__`, and, if search will hash it, `__hash__`.
   Decide deliberately how coarse `literals` should be; that decision is your state space.
3. Implement `reset`, `set_index`, `successors`, `is_goal`, `is_terminal`, and `simulate`.
4. Filter self-loops out of `successors` (`if successor_state == state: continue`); every bundled
   environment does this, and planners rely on it. Check that it can actually fire: if `literals`
   include a step counter, no successor ever equals its parent and the filter is dead code.
5. Add an `EnvironmentSpec` to `planiverse/environments/registry.py`; that is what puts it
   in the catalogue and in `make()`.
6. Add a doc under `docs/environments/` and a row to the catalogue above.

## Repository layout

```
planiverse/
├── environments/
│   ├── base.py                         # Environment — the one base class
│   ├── registry.py                     # EnvironmentSpec, list_environments(), make()
│   ├── gameboy/                        # the PyBoy-backed environments
│   │   ├── gb.py                       # GBEnv, GBState, GBAction — the shared tail
│   │   ├── puzznic_gb.py               # PuzznicGBEnv
│   │   ├── flipull_gb.py               # FlipullGBEnv
│   │   ├── lolo_gb.py                  # LoloGBEnv
│   │   ├── amazing_tater_gb.py         # AmazingTaterGBEnv
│   │   └── super_mario_land_gb.py      # SuperMarioLandGBEnv
│   ├── gameboy_py/                     # their dependency-free counterparts, no ROM needed
│   │   ├── puzznic.py                  # PuzznicGame     — twin of puzznic_gb
│   │   ├── flipull.py                  # FlipullGame     — twin of flipull_gb
│   │   ├── lolo.py                     # LoloGame        — twin of lolo_gb
│   │   ├── amazing_tater.py            # AmazingTaterGame — twin of amazing_tater_gb
│   │   └── super_mario_land.py         # SuperMarioLandGame — same genre as super_mario_land_gb,
│   │                                   #                  cartridge-fitted physics, not a twin
│   ├── network_attack/                 # EnvNASim (wraps NASim)
│   ├── water_network/                  # WaterNetworkEnv (WNTR/EPANET)
│   ├── power_grid/                     # PowerGridEnv (Grid2Op)
│   └── crop_management/                # CropEnv (PCSE/WOFOST)
├── planners/
│   ├── width/                          # IW, Iterated Width, SIW, BFWS, Rollout IW, π-IW
│   ├── fsx.py                          # FSXPlanner (future state maximisation)
│   ├── mcts.py                         # MCTSPlanner (UCT)
│   └── super_mario_planner_gb.py       # TreeSearchPlanner, SuperMarioPlanner
├── rendering/                          # traces to GIF or PNG frames (env.render_trace delegates here)
└── benchmark/                          # planiverse-bench: the paper's evaluation protocol
    ├── __init__.py                     # generate / solve / report, and the protocol's constants
    └── measures.py                     # per-environment progress measures for SIW and BFWS
docs/environments/                      # per-environment documentation
docs/benchmark.md                       # the benchmark: protocol, statuses, report
tools/setup_benchmark.sh                # builds the venv, installs, runs generate
tests/
├── sm83.py                             # minimal SM83 assembler, for the test cartridges
├── fake_puzznic_rom.py                 # synthetic Game Boy ROM with Puzznic's memory layout
└── fake_flipull_rom.py                 # synthetic Game Boy ROM with Flipull's memory layout
```

There was a `dev/` scratch directory; it is gone. It held two files. `dev.py` was stale; it
imported names that no longer exist (`SuperMario`, `super_mario_bros_grid`,
`super_mario_planner_tile`) and could not run, and it was the only thing that ever imported
`pcg_benchmark`. `earthmodel.py` was a vendored copy of the c:GLOBAL gym environment
(© Felix Strnad), kept for a port that never happened; it never implemented the Planiverse
interface. Both are recoverable from git history if the port is ever picked up.

`chex`, `flax`, `jaxmarl` and `dill` were declared as required dependencies but are imported nowhere
in the library, so they are gone too; nothing outside `epipolicy/**/deprecated/` referenced them.

## Attribution

Planiverse adapts several upstream simulators. Each is credited in its own doc; the sources are:

| Environment | Upstream |
|---|---|
| Network attack | [NASim](https://github.com/MFaisalZaki/NetworkAttackSimulator) (fork, MIT), [PenGym](https://github.com/cyb3rlab/PenGym) |
| Super Mario Land, Puzznic (GB), Flipull (GB), Adventures of Lolo (GB) | [PyBoy](https://github.com/Baekalfen/PyBoy) |
| Water distribution | [WNTR](https://github.com/USEPA/WNTR) (US EPA's EPANET wrapper) |
| Power grid | [Grid2Op](https://github.com/Grid2Op/grid2op) (RTE) |
| Crop management | [PCSE / WOFOST](https://github.com/ajwdewit/pcse) (Wageningen University) |

Two environments were removed over licensing: epidemic control vendored
[EpiPolicy](https://github.com/huda-lab/RL-Epidemic-Benchmark), and urban planning shipped the
city datasets of [a consensus-MARL paper's repository](https://github.com/mao1207/Steering-Stakeholder-Dynamics-in-Urban-Planning-via-Consensus-based-MARL).
Neither upstream publishes a licence, so neither the simulator nor the data can be
redistributed here. Both remain in git history should their upstreams ever license them.

## Status

What is in the tree:

- Fourteen environments: three simulator-backed operational ones (water distribution, power
  grid, crop management), the NASim network attack, and five Game Boy games, each as a cartridge
  environment and as a dependency-free Python counterpart. Four of the counterparts are twins of
  their cartridge; the Super Mario Land one shares the genre and the measured physics, not the
  levels.
- Nine planners: IW(k), Iterated Width, SIW, BFWS and Iterated BFWS; Rollout IW and π-IW, the
  latter with a policy it learns from its own lookaheads; MCTS; and Future State Maximization.
- `planiverse-bench`, the paper's protocol as code: seven planner configurations, five seeds for
  the four that take one, and a report that regenerates the paper's tables, figures and quoted
  numbers from the results.
- A test suite that skips what it cannot build, with synthetic cartridges for the two Taito
  games so their emulator code is tested without a ROM.

Open:

- [ ] Benchmark runs for Rollout IW and π-IW, and the paper edits that go with them; see
      [Bringing the paper up to date](docs/benchmark.md#bringing-the-paper-up-to-date).
- [ ] The flood/transport environment
      ([floods_transport_rl](https://github.com/MLSM-at-DTU/floods_transport_rl)), referenced as
      a planned addition and not yet in the tree.
- [ ] Optional dependency groups, so one environment does not pull in all of them. Today there is
      one dependency list and a `dev` extra.
- [ ] `is_terminal` for the network attack, the one environment that still hard-codes `False`.
- [ ] Confirm Super Mario Land's level-complete address (`0xDFE8`, marked unverified in the
      code) and its enemy tile IDs.
- [ ] `SuperMarioPlanner.search` returns nothing and has no replanning loop.
- [ ] What a Flipull throw actually hits. Every row connects, so it is not simply the first block
      in the player's row, and until it is settled `FlipullGame` is a Flipull-*like* environment
      with a stated rule set rather than a clone of the cartridge.
- [ ] Flipull's second stage table at `$3A4E`, reached through the RNG: a bonus course,
      unexplored.
- [ ] A full pure-Python Super Mario Land twin. Deliberately not attempted: reverse-engineering
      a physics platformer move for move is a far larger job than a turn-based puzzle, and a
      half-modelled one would look like a prediction of the cartridge without being one.

Withdrawn: Boxxle II, both the cartridge environment and its twin. Both worked and agreed move
for move over 3,000 random moves. They were removed because Boxxle II is Sokoban, whose
transition is an add/delete list and whose PDDL encoding is one page long; an environment a
declarative model handles well is not evidence for a library about planning with simulators. The
code is in the history.

## Licence

GPL-3.0. See [LICENSE](LICENSE).
