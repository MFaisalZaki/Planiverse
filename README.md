# Planiverse

A Python library for **planning with simulators**.

Classical planners need a declarative model of the world. Many interesting problems do not have one;
they have a *simulator* instead: a water distribution network, a network attack emulator, a crop
growth model. Planiverse wraps those simulators behind one small, uniform interface so that a
search-based planner can expand states, test goals, and validate plans without knowing what is
underneath.

Every environment answers the same four questions:

- What is the initial state? (`reset`)
- What can happen next? (`successors`)
- Am I done, and did I win? (`is_goal` / `is_terminal`)
- What does this plan actually do? (`simulate`)

And every environment can make more of itself: `generate_instance(seed)` draws a fresh level,
room, network, contingency or season, so a benchmark is not limited to what ships.

## Environment catalogue

| Environment | `make()` name | Bundled instances | `generate_instance` draws | Tags | Docs |
|---|---|---|---|---|---|
| Water distribution | `water_network` | 9 contamination scenarios | the network, and the junction the contaminant enters at | operational, infrastructure | [docs](docs/environments/water-distribution.md) |
| Power grid | `power_grid` | 9 contingencies | the time series, its starting step, and the line that trips | operational, infrastructure | [docs](docs/environments/power-grid.md) |
| Crop management | `crop_management` | 22 growing seasons | the year's weather and the sowing date | operational, agriculture | [docs](docs/environments/crop-management.md) |
| Network attack | `network_attack` | 18 NASim benchmarks | network topology, hosts, services, OSs and exploits | security | [docs](docs/environments/network-attack.md) |
| Puzznic | `puzznic` | 128 levels | board size, wall layout, block colours and pairs | game | [docs](docs/environments/puzznic.md) |
| Flipull | `flipull` | 32 stages | wall size, block types, arrangement and clear target | game | [docs](docs/environments/flipull.md) |
| Adventures of Lolo | `lolo` | 163 rooms | terrain, hearts, Emerald Framers, Snakeys and Medusas | game | [docs](docs/environments/lolo.md) |
| Amazing Tater | `amazing_tater` | 105 rooms | room size, walls, blocks, pits, turnstiles and taters | game | [docs](docs/environments/amazing-tater.md) |
| Super Mario Land | `super_mario_land` | 12 levels | level length, gaps, platforms, hazards and enemies | game, platformer | [docs](docs/environments/super-mario-land.md) |
| Game Boy | `game_boy` | one per stage, level or room the cartridge's wrapper reaches | the stage, the timer seed, and an opening played from its first frame | game, emulator | [docs](docs/environments/game-boy.md) |
| Stable-Retro | `retro` | one per save state the integration ships (Airstriker: 1) | the save state, an opening played from it, and the goal | game, emulator | [docs](docs/environments/stable-retro.md) |

```python
from planiverse.environments import list_environments, make

[spec.name for spec in list_environments(tag="operational")]
# ['crop_management', 'power_grid', 'water_network']

env = make("water_network", index=8)       # the ninth bundled scenario
env = make("puzznic", seed=7)              # a freshly generated level
state, info = env.reset()
```

Every environment lives in one flat `planiverse.environments` package behind one base class.
What used to be two package trees (`real_world_problems` and `retro_games`) is now a `tags`
field on a registry entry, because that split recorded where an environment came from rather
than what a planner could do with it. The catalogue falls into three families: `game`,
`operational` (an agent running a system it is responsible for, whether that is a power
grid or a production line) and `security`, where the agent probes a network rather than
operates it. See [Architecture](#architecture).

The five games are commercially published Game Boy titles reimplemented in pure Python, from
rules established by observing the originals. Nothing here runs the original programs, and no
ROM is needed or accepted; see [Studied titles](#studied-titles).

The two emulator environments are generic. `game_boy` runs whatever cartridge you point it at
under [PyBoy](https://github.com/Baekalfen/PyBoy) and reads the game through PyBoy's game
wrappers, so the knowledge of where a game keeps its counters stays in PyBoy; `retro` runs any
[Stable-Retro](https://github.com/Farama-Foundation/stable-retro) integration from its save
states and the variables the integration names. Neither ships a ROM. Stable-Retro's own package
includes one freely redistributable game, Airstriker, which is `retro`'s default; `game_boy`
needs a cartridge from you (`PLANIVERSE_GB_ROM`), and its tests run on a cartridge they assemble
themselves.

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

One install gets you every environment, on every supported Python, and nothing has to be
supplied: the five games are self-contained, the water, power grid and crop environments ship
their benchmark data inside their dependencies, so they run offline, and Stable-Retro ships
Airstriker. The one exception is a Game Boy cartridge for `game_boy`, which is copyrighted and
comes from you.

`tests/test_packaging.py` walks the import graph from each environment's entry point and fails if
anything it reaches is undeclared; a dependency that only works because another package happens
to pull it in cannot go missing silently.

## Tests

```bash
pytest                  # the whole suite
pytest -m "not slow"    # skip the slow search tests
```

Tests for an environment whose dependencies are missing skip rather than fail, so the suite is
runnable from a partial install.

[`tests/test_interface.py`](tests/test_interface.py) checks the contract below uniformly across every
environment, [`tests/test_generators.py`](tests/test_generators.py) checks the instance generators
the same way, and the other modules cover per-environment behaviour. The Game Boy tests run on
an original cartridge assembled by [`tests/counter_rom.py`](tests/counter_rom.py), so no
commercial ROM is involved.

## Quickstart

Puzznic is the smallest environment, so it is the fastest way to see the interface:

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

## Generating instances

`set_index` picks one of the instances an environment ships with. `generate_instance` makes a
new one from a seed and selects it the same way:

```python
env = PuzznicGame()
level = env.generate_instance(seed=7, width=6, height=6, colours=4)
state, info = env.reset()             # info["generated"] is True
print(level)                          # a level string, in the same alphabet as the bundled ones
print(env.witness)                    # the plan the draw was accepted on
```

Three things hold everywhere:

- **The draw is a function of the seed.** The same seed and options give the same instance on
  any machine, because everything random comes from one `random.Random(seed)`.
- **The instance is plain data**, the same shape as the bundled ones: a level string, a
  `[stage, target]` pair, a tuple of rows, a dict of scenario fields. `json.dumps` it, and
  `set_instance` takes it back later, so a generated benchmark replays exactly.
- **A draw is checked before it is handed out, the way the bundled instances were.** A random
  board is usually unsolvable, and an unsolvable instance is not an instance: a planner cannot
  tell "no plan" from "not yet". The generators reuse the checks the shipped instances passed.
  Flipull's stages were drawn at random and explored exhaustively, with the fewest blocks
  reachable as the target; the generator does the same. Super Mario Land's levels were each
  searched with BFWS(w=2) and ranked by what that cost (`MEASURED_EXPANSIONS`); the generator
  runs the same search and records the same number. Amazing Tater's stored solutions came from
  its `solve`, and the generator checks a drawn room with that search. The simulator
  environments draw by the tests their bundled scenarios were chosen by (a source that
  contaminates enough, a trip that leaves the grid doomed) and then search the draw, so a
  generated scenario carries `solved_at` like a bundled one; the crop season's reference
  schedule is a solution by construction. Everywhere, the plan is left in `env.witness` and
  what the search spent in `env.witness_expansions`; `min_plan_length` and `min_expansions`
  turn those into difficulty knobs. The check is a bias, since it favours instances a small
  search can solve, and it is a knob too: `solvable=False` hands out the raw draw.

What each generator varies is in the catalogue above; the options are in each environment's
doc. The shared machinery is [`planiverse/environments/generation.py`](planiverse/environments/generation.py):
the seeded draw, the bounded breadth-first and BFWS searches, the retry loop, and the board
helpers the four grid games draw with. The benchmark runs the bundled instances; a generated
benchmark is a file of instances and a loop over `set_instance`.

## Core concepts

### The environment interface

An environment is a plain Python class. It subclasses `Environment` (or just satisfies the
contract, since dispatch is structural) and implements the eight contract methods, the first
eight rows below, plus whichever of the optional ones it can (registration is an explicit
`EnvironmentSpec` entry, not a metaclass):

| Method | Returns | Notes |
|---|---|---|
| `reset()` | `(state, info)` | Builds the initial state. Call `set_index` first. |
| `set_index(index)` | — | Selects which bundled scenario/level/instance to load. |
| `successors(state)` | `[(action, next_state), ...]` | The expansion step. Self-loops are filtered out. |
| `is_goal(state)` | `bool` | |
| `is_terminal(state)` | `bool` | Dead end: no goal reachable from here. |
| `simulate(plan)` | `[state, ...]` | Replays a plan from the initial state. |
| `generate_instance(seed, **options)` | the instance | Draws a fresh instance from a seed, selects it, and returns it as plain data. The same seed always gives the same instance. |
| `set_instance(instance)` | — | Selects an instance drawn earlier. |
| `step(action)` | `(state, reward)` | Optional; stateful stepping. |
| `validate(plan)` | `bool` | Optional. |
| `get_actions()` | `[action, ...]` | Optional. |

Not every environment implements every method. What is actually there today:

| | `reset` | `set_index` | `successors` | `is_goal` | `is_terminal` | `simulate` | `generate_instance` | `step` | `validate` | `get_actions` | `render` | `close` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `PuzznicGame` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| `FlipullGame` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| `LoloGame` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| `AmazingTaterGame` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| `SuperMarioLandGame` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| `WaterNetworkEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `PowerGridEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `CropEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `EnvNASim` | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ | — | ✅ | — | — | — |
| `GameBoyEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `RetroEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

⚠️ `is_terminal` returns a hard-coded `False` in this environment: it has no dead ends, or
detecting them is left to the planner. Puzznic, `FlipullGame` and Super Mario Land are the ones
that really compute a positional dead end; `FlipullGame`'s is *exact*, because the rules are
known in Python and it can ask outright whether any throw would connect. The three
simulator-backed environments compute real ones too: a water network whose service has
collapsed, a blacked-out grid, and a growing season whose water budget is spent. The two
emulator environments take theirs from the game: a wrapper's `game_over()`, a lost life, or
an integration's done condition.

`Environment.capabilities()` reports this per class, so the table above is checked against the code
rather than trusted: `tests/test_interface.py` asserts the two agree.

Note that `validate` is provided by the base class for everything, derived from `simulate` and
`is_goal`, so no environment writes it out. `step` and `get_actions` have base defaults too,
but theirs only explain their own absence, and so do the eight contract methods, so a
half-written environment fails where a method is called rather than when it is built. That is
why "does the class override it" is the wrong test, and `capabilities()` asks whether the
method would actually do something.

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
state, _ = env.reset()  # set_index must come first — reset raises on it
```

The index is a stable handle for "instance *n* of this environment", which is what a benchmark runner
wants. The mapping from index to instance is listed in each environment's doc.
`generate_instance(seed)` and `set_instance(instance)` select an instance the same way, and
`info["generated"]` on the reset says which kind is loaded.

### Bringing your own environment

Planners call environments directly; there is no wrapper to construct. An environment brought
from outside the library counts as long as it answers the eight contract methods, the
generator's two included: `implements_contract` checks structurally, so no subclassing is
required. There used to be a `Simulator` facade between planners and environments;
once its PDDLGym dispatch was removed it delegated every call one-to-one, so it went the way of
the two-base-class split.

## Rendering a trace

```python
trace = env.simulate(plan)
env.render_trace(trace, "plan.gif")        # an animated GIF, one frame per state
env.render_trace(trace, "plan-frames/")    # a directory of independent PNGs
```

Rendering a trace is one image per state, nothing more: the state's own text, typeset. See
[docs/rendering.md](docs/rendering.md).

## Planners

| Family | Where | What it needs from an environment |
|---|---|---|
| Width-based: IW(k), Iterated Width, SIW, BFWS | [`planiverse/planners/width/`](planiverse/planners/width/) | `successors` and `literals`; a `progress` callback helps |
| Rollout IW, and π-IW with a policy it learns as it plans | [`planiverse/planners/width/rollout.py`](planiverse/planners/width/rollout.py), [`policy.py`](planiverse/planners/width/policy.py) | `successors` and `literals`; a `progress` callback stands in for the score |
| MCTS / UCT | [`planiverse/planners/mcts.py`](planiverse/planners/mcts.py) | `successors`; a `reward` callback helps a lot |
| Future State Maximization | [`planiverse/planners/fsx.py`](planiverse/planners/fsx.py) | `successors`, and **nothing else**: no goal, no heuristic |
| Tree search / A* | [`planiverse/planners/tree_search.py`](planiverse/planners/tree_search.py) | a heuristic and a cost function |

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
configurations the paper compares plus Rollout IW and π-IW, on every bundled instance of every
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
per seed of a seeded planner, under `sandbox/slurm/`. An environment whose dependencies are
missing is skipped and says so.

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

Environments are planner-agnostic. [`planiverse/planners/tree_search.py`](planiverse/planners/tree_search.py)
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

Note that `PriorityQueue` pushes `(priority, item)` tuples, so ties compare the items themselves,
which is why state and action classes define `__lt__`.

## Architecture

One flat package, one base class, and a registry.

```
planiverse/environments/
├── base.py          # Environment — the eight-method contract, and nothing else
├── registry.py      # EnvironmentSpec per environment: instances, tags, deps, state identity
├── generation.py    # what the generators share: a seeded draw, a bounded search, a retry loop
├── gameboy_py/      # the five games, reimplemented in pure Python
├── emulated/        # one environment per emulator (PyBoy, Stable-Retro), for any game
└── <one subpackage per simulator-backed environment>
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
| `generates` | what `generate_instance` draws at random, in words |
| `tags` | the family (`game`, `operational`, `security`) plus finer ones like `continuous-dynamics` |

`state_identity` is the one worth understanding. A `value` state carries its own contents and
expanding is pure. A `path` state *is* the decision sequence, replayed on demand, which is
sound only because the simulator is deterministic. A `snapshot` state carries a serialised
emulator image (a save state) and is told apart from others by what the game wrapper or the
integration reads off it. Most simulators are step-only and cannot be rewound; that is the
property that decides whether something can be a Planiverse environment at all, and it now
has a name.

The contract check is structural (`implements_contract`), so an environment brought from
outside works without inheriting from anything.

## Adding an environment

1. Subclass `Environment` (`planiverse/environments/base.py`) and implement the eight methods.
2. Define a state class exposing `literals`, `__eq__`, and, if search will hash it, `__hash__`.
   Decide deliberately how coarse `literals` should be; that decision is your state space.
3. Implement `reset`, `set_index`, `successors`, `is_goal`, `is_terminal`, and `simulate`.
4. Filter self-loops out of `successors` (`if successor_state == state: continue`); every bundled
   environment does this, and planners rely on it. Check that it can actually fire: if `literals`
   include a step counter, no successor ever equals its parent and the filter is dead code.
5. Give it a generator, which is part of the contract: `set_instance` takes an instance as
   plain data and `generate_instance` draws one from a seed and selects it. Everything random
   comes from `random.Random(seed)`, so the same seed gives the same instance anywhere. Keep
   the instance in one attribute that `reset` builds from, so `set_index` and `set_instance`
   are two ways of filling the same slot. If a draw can be unsolvable, check it with
   `planiverse.environments.generation.solvable_draw` before handing it out.
6. Add an `EnvironmentSpec` to `planiverse/environments/registry.py`; that is what puts it
   in the catalogue and in `make()`.
7. Add a doc under `docs/environments/` and a row to the catalogue above.

## Repository layout

```
planiverse/
├── environments/
│   ├── base.py                         # Environment — the one base class
│   ├── registry.py                     # EnvironmentSpec, list_environments(), make()
│   ├── generation.py                   # rng, bounded_search, draw_until, solvable_draw
│   ├── gameboy_py/                     # the five games in pure Python, nothing to supply
│   │   ├── puzznic.py                  # PuzznicGame
│   │   ├── flipull.py                  # FlipullGame
│   │   ├── lolo.py                     # LoloGame
│   │   ├── amazing_tater.py            # AmazingTaterGame
│   │   └── super_mario_land.py         # SuperMarioLandGame — measured physics, original levels
│   ├── emulated/                       # any game, nothing shipped
│   │   ├── game_boy.py                 # GameBoyEnv — a cartridge under PyBoy, read through its wrappers
│   │   └── stable_retro.py             # RetroEnv — a Stable-Retro integration from its save states
│   ├── network_attack/                 # EnvNASim (wraps NASim)
│   ├── water_network/                  # WaterNetworkEnv (WNTR/EPANET)
│   ├── power_grid/                     # PowerGridEnv (Grid2Op)
│   └── crop_management/                # CropEnv (PCSE/WOFOST)
├── planners/
│   ├── width/                          # IW, Iterated Width, SIW, BFWS, Rollout IW, π-IW
│   ├── fsx.py                          # FSXPlanner (future state maximisation)
│   ├── mcts.py                         # MCTSPlanner (UCT)
│   └── tree_search.py                  # TreeSearchPlanner, Heuristic, CostFunction
├── rendering/                          # traces to GIF or PNG frames (env.render_trace delegates here)
└── benchmark/                          # planiverse-bench: the paper's evaluation protocol
    ├── __init__.py                     # generate / solve / report, and the protocol's constants
    └── measures.py                     # per-environment progress measures for SIW and BFWS
docs/environments/                      # per-environment documentation
docs/benchmark.md                       # the benchmark: protocol, statuses, report
docs/provenance.md                      # where the game rules and level data came from
tools/setup_benchmark.sh                # builds the venv, installs, runs generate
tests/
├── test_interface.py                   # the contract, across every environment
├── test_generators.py                  # the instance generators, across every environment
├── test_emulated.py                    # the two emulator environments
├── counter_rom.py, sm83.py             # the test cartridge, and the assembler that builds it
└── test_<environment>.py               # per-environment behaviour
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
| Water distribution | [WNTR](https://github.com/USEPA/WNTR) (US EPA's EPANET wrapper) |
| Power grid | [Grid2Op](https://github.com/Grid2Op/grid2op) (RTE) |
| Crop management | [PCSE / WOFOST](https://github.com/ajwdewit/pcse) (Wageningen University) |
| Game Boy | [PyBoy](https://github.com/Baekalfen/PyBoy) (LGPL-3.0), with the game wrappers of [MFaisalZaki/PyBoy](https://github.com/MFaisalZaki/PyBoy) |
| Stable-Retro | [Stable-Retro](https://github.com/Farama-Foundation/stable-retro) (Farama Foundation, MIT), which ships Airstriker |

Two environments were removed over licensing: epidemic control vendored
[EpiPolicy](https://github.com/huda-lab/RL-Epidemic-Benchmark), and urban planning shipped the
city datasets of [a consensus-MARL paper's repository](https://github.com/mao1207/Steering-Stakeholder-Dynamics-in-Urban-Planning-via-Consensus-based-MARL).
Neither upstream publishes a licence, so neither the simulator nor the data can be
redistributed here. Both remain in git history should their upstreams ever license them.

### Studied titles

The five game environments reimplement commercially published titles in Python. This
repository ships no ROM image, no original code and no original graphics, and runs none of
the original programs.

*Adventures of Lolo* (HAL Laboratory / Nintendo), *Puzznic* and *Flipull* (Taito),
*Amazing Tater* (Atlus) and *Super Mario Land* (Nintendo) are the copyright works
and trade marks of their respective owners, used here descriptively. This project
is unofficial and unaffiliated. See [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) and
[docs/provenance.md](docs/provenance.md).

## Status

What is in the tree:

- Eleven environments: three simulator-backed operational ones (water distribution, power
  grid, crop management), the NASim network attack, five games reimplemented in pure
  Python, and two generic emulator environments, one for any Game Boy cartridge under PyBoy
  and one for any Stable-Retro integration. Every one ships its bundled instances and
  generates more from a seed.
- Nine planners: IW(k), Iterated Width, SIW, BFWS and Iterated BFWS; Rollout IW and π-IW, the
  latter with a policy it learns from its own lookaheads; MCTS; and Future State Maximization.
- `planiverse-bench`, the paper's protocol as code: seven planner configurations, five seeds for
  the four that take one, and a report that regenerates the paper's tables, figures and quoted
  numbers from the results.
- A test suite that skips what it cannot build.

Open:

- [ ] Benchmark runs for Rollout IW and π-IW, and the paper edits that go with them; see
      [Bringing the paper up to date](docs/benchmark.md#bringing-the-paper-up-to-date).
- [ ] The flood/transport environment
      ([floods_transport_rl](https://github.com/MLSM-at-DTU/floods_transport_rl)), referenced as
      a planned addition and not yet in the tree.
- [ ] Optional dependency groups, so one environment does not pull in all of them. Today there is
      one dependency list and a `dev` extra.
- [ ] `is_terminal` for the network attack, the one environment that still hard-codes `False`.
- [ ] What a Flipull throw actually hits. Every row connects, so it is not simply the first block
      in the player's row, and until it is settled `FlipullGame` is a Flipull-*like* environment
      with a stated rule set rather than a clone of the original.
- [ ] A generated benchmark: `planiverse-bench` runs the bundled instances only.

Withdrawn: the five emulator-backed environments that drove the original cartridges through
PyBoy alongside their Python counterparts, with their memory maps, screen captures and
synthetic test cartridges. They are kept outside the repository, and what replaced them is
generic: one environment per emulator, reading a game through PyBoy's wrappers or Stable-Retro's
integration rather than through a memory map kept here. Withdrawn earlier: Boxxle II,
which worked, because Boxxle II is Sokoban, whose transition is an add/delete list and whose
PDDL encoding is one page long; an environment a declarative model handles well is not
evidence for a library about planning with simulators.

## Licence

GPL-3.0 for this repository's own code, documentation and benchmark
definitions. See [LICENSE](LICENSE).

The level layouts three of the games ship are derived from the original titles and are
third-party material that this project has no right to sublicense. See
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for the copyright holders and the basis on
which they are included.
