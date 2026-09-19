# Planiverse

A Python library for **planning with simulators**.

Classical planners need a declarative model of the world (i.e., a description of the actions
in terms of the facts they need and the facts they change). Many interesting problems do not
have one; they have a *simulator* instead: a water distribution network, a network attack
emulator, a crop growth model, a flooding city. Planiverse wraps such simulators behind one
small interface so that a search-based planner can expand states, test goals and validate
plans without knowing what is underneath.

Every environment answers the same four questions: what the initial state is (`reset`), what
can happen next (`successors`), whether a state is a goal or a dead end (`is_goal`,
`is_terminal`), and what a plan actually does (`simulate`). Every environment can also make
more of itself: `generate_instance(seed)` draws a fresh level, room, network, contingency,
season or city, so a benchmark is not limited to what ships.

## Environment catalogue

| Environment | `make()` name | Bundled instances | `generate_instance` draws | Tags | Docs |
|---|---|---|---|---|---|
| Water distribution | `water_network` | 100 contamination scenarios | the network, and the junction the contaminant enters at | operational, infrastructure | [docs](docs/environments/water-distribution.md) |
| Power grid | `power_grid` | 100 contingencies | the time series, its starting step, and the line that trips | operational, infrastructure | [docs](docs/environments/power-grid.md) |
| Crop management | `crop_management` | 100 growing seasons | the year's weather and the sowing date | operational, agriculture | [docs](docs/environments/crop-management.md) |
| Micropolis city | `micropolis` | 100 cities | the map, the layout, the horizon and the population target | operational, city | [docs](docs/environments/micropolis.md) |
| Factory | `factory` | 100 patches | the ore patch, its walls, the start, what is carried, the horizon and the plate target | game, operational, factory | [docs](docs/environments/factory.md) |
| Flood adaptation | `flood_transport` | 100 scenarios | the city, its storms, the horizon and the measures on offer | operational, infrastructure, climate | [docs](docs/environments/flood-transport.md) |
| Network attack | `network_attack` | 100 networks | network topology, hosts, services, OSs and exploits | security | [docs](docs/environments/network-attack.md) |
| Puzznic | `puzznic` | 100 levels | board size, wall layout, block colours and pairs | game | [docs](docs/environments/puzznic.md) |
| Flipull | `flipull` | 100 stages | wall size, block types, arrangement and clear target | game | [docs](docs/environments/flipull.md) |
| Adventures of Lolo | `lolo` | 100 rooms | terrain, hearts, Emerald Framers, Snakeys and Medusas | game | [docs](docs/environments/lolo.md) |
| Slingshot | `slingshot` | 100 levels | the structures, their materials, where the targets sit, and the shots | game, physics | [docs](docs/environments/slingshot.md) |
| Artillery | `artillery` | 100 fields | the terrain, where the targets dig in, the wind, and the shells | game, physics | [docs](docs/environments/artillery.md) |
| Tower defence | `tower_defence` | 100 maps | the path, the building slots, the waves, the gold and the lives | game | [docs](docs/environments/tower-defence.md) |
| Fluid | `fluid` | 100 caves | the cave, the spring, the basin, the drain, the water needed and the digs allowed | game | [docs](docs/environments/fluid.md) |
| Billiards | `billiards` | 100 tables | where the balls lie, how many there are, and the shots | game, physics | [docs](docs/environments/billiards.md) |
| Lemmings | `lemmings` | 100 levels | the platforms, gaps and walls, the entrance and exit, the crowd, the quota and the skills | game | [docs](docs/environments/lemmings.md) |
| Amazing Tater | `amazing_tater` | 100 rooms | room size, walls, blocks, pits, turnstiles and taters | game | [docs](docs/environments/amazing-tater.md) |
| Game Boy | `game_boy` | one per stage, level or room the cartridge's wrapper reaches | the stage, the timer seed, and an opening played from its first frame | game, emulator | [docs](docs/environments/game-boy.md) |
| Stable-Retro | `retro` | one per save state the integration ships (Airstriker: 1) | the save state, an opening played from it, and the goal | game, emulator | [docs](docs/environments/stable-retro.md) |

```python
from planiverse.environments import list_environments, make

[spec.name for spec in list_environments(tag="operational")]
# ['crop_management', 'flood_transport', 'power_grid', 'water_network']

env = make("water_network", index=8)       # the ninth bundled scenario
env = make("puzznic", seed=7)              # a freshly generated level
state, info = env.reset()
```

The catalogue falls into three families, which are tags on a registry entry rather than
package trees: `game`, `operational` (an agent running a system it is responsible for, such
as a power grid or a city's roads) and `security`, where the agent probes a network rather
than operates it. See [Architecture](#architecture).

The four games are commercially published Game Boy titles reimplemented in pure Python, from
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

Requires Python 3.11, 3.12 or 3.13: numba and scipy have no 3.14 wheels yet, and building them
from source needs a system OpenBLAS.

```bash
git clone https://github.com/MFaisalZaki/Planiverse.git
cd Planiverse
pip install -e ".[dev]"      # the dev extra adds pytest
```

Or, with poetry:

```bash
poetry env use python3.12
poetry install --extras dev
```

One install gets you every environment, on every supported Python, and nothing has to be
supplied: the four games and the flood environment are self-contained, the water, power grid
and crop environments ship their benchmark data inside their dependencies, so they run
offline, and Stable-Retro ships Airstriker. The one exception is a Game Boy cartridge for
`game_boy`, which is copyrighted and comes from you. `tests/test_packaging.py` walks the import
graph from each environment's entry point and fails if anything it reaches is undeclared, so a
dependency that only works because another package happens to pull it in cannot go missing
silently.

## Tests

```bash
pytest                  # the whole suite
pytest -m "not slow"    # skip the slow search tests
```

Tests for an environment whose dependencies are missing skip rather than fail, so the suite is
runnable from a partial install. [`tests/test_interface.py`](tests/test_interface.py) checks
the contract below uniformly across every environment and
[`tests/test_generators.py`](tests/test_generators.py) checks the instance generators the same
way; the other modules cover per-environment behaviour. The Game Boy tests run on an original
cartridge assembled by [`tests/counter_rom.py`](tests/counter_rom.py), so no commercial ROM is
involved.

## Quickstart

Puzznic is the smallest environment, so it is the fastest way to see the interface:

```python
from planiverse.environments.games.puzznic import PuzznicGame

env = PuzznicGame()
env.set_index(0)              # choose the instance before reset
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

Three things hold everywhere. First, the draw is a function of the seed: the same seed and
options give the same instance on any machine, because everything random comes from one
`random.Random(seed)`. Second, the instance is plain data of the same shape as the bundled
ones (a level string, a `[stage, target]` pair, a tuple of rows, a dict of scenario fields),
so `json.dumps` writes it down and `set_instance` takes it back, and a generated benchmark
replays exactly. Third, a draw is checked before it is handed out, the way the bundled
instances were, because a random board is usually unsolvable and an unsolvable instance is
not an instance: a planner cannot tell "no plan" from "not yet". The generators reuse the
checks the shipped instances passed. Flipull's stages were drawn at random and explored
exhaustively, with the fewest blocks reachable as the target, and the generator does the
same; Amazing Tater's stored solutions came from its
`solve`, which the generator checks a drawn room with; the simulator environments draw by the
tests their bundled scenarios were chosen by and then search the draw; the crop season's
reference schedule and the flood scenario's reference policy are solutions by construction.
Everywhere, the plan is left in `env.witness` and what the search spent in
`env.witness_expansions`, and `min_plan_length` turns the first into a difficulty knob.
The check is a bias, since it favours instances a small search can solve,
and it is a knob too: `solvable=False` hands out the raw draw.

What each generator varies is in the catalogue above, and the options are in each
environment's doc. A generator draws to the shape of the originals: a game takes the layout
options a caller leaves unset from the profile of one of its bundled instances (its size, its
counts of blocks or hearts, its share of scenery), and a simulator draws on the simulator's own
data (WNTR's networks, grid2op's time series, PCSE's weather, NASim's scenario generator).
Every environment ships a hundred instances. Where the originals fall short of that (Flipull
and the five simulator environments) the set is made up with instances the generator drew,
after the originals, with the seed each came from recorded with it and the plan it was
accepted on in the tests, so they can be re-derived. The techniques are the standard ones,
and each environment's doc gives the
references: generate-and-test with a solvability check is search-based procedural content
generation ([Togelius et al., 2011](https://doi.org/10.1109/TCIAIG.2011.2148116);
[Shaker et al., 2016](https://pcgbook.com/)), and the check is breadth-first search. The
shared machinery is
[`planiverse/environments/generation.py`](planiverse/environments/generation.py): the seeded
draw, the bounded breadth-first search, the retry loop, and the board helpers the
four grid games draw with.

## The environment interface

An environment is a plain Python class. It subclasses `Environment` (or just satisfies the
contract, since dispatch is structural) and implements the eight contract methods, the first
eight rows below, plus whichever of the optional ones it can. Registration is an explicit
`EnvironmentSpec` entry rather than a metaclass.

| Method | Returns | Notes |
|---|---|---|
| `reset()` | `(state, info)` | Builds the initial state. Call `set_index` first. |
| `set_index(index)` | | Selects which bundled scenario, level or instance to load. |
| `successors(state)` | `[(action, next_state), ...]` | The expansion step. Self-loops are filtered out. |
| `is_goal(state)` | `bool` | |
| `is_terminal(state)` | `bool` | Dead end: no goal reachable from here. |
| `simulate(plan)` | `[state, ...]` | Replays a plan from the initial state. |
| `generate_instance(seed, **options)` | the instance | Draws a fresh instance from a seed, selects it, and returns it as plain data. The same seed always gives the same instance. |
| `set_instance(instance)` | | Selects an instance drawn earlier. |
| `step(action)` | `(state, reward)` | Optional; stateful stepping. |
| `validate(plan)` | `bool` | Optional. |
| `get_actions()` | `[action, ...]` | Optional. |

What is actually there today:

| | `reset` | `set_index` | `successors` | `is_goal` | `is_terminal` | `simulate` | `generate_instance` | `step` | `validate` | `get_actions` | `render` | `close` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `PuzznicGame` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
| `FlipullGame` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
| `LoloGame` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
| `AmazingTaterGame` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
| `WaterNetworkEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `PowerGridEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `CropEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `FloodTransportEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
| `EnvNASim` | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ | | ✅ | | | |
| `GameBoyEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `RetroEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

⚠️ `is_terminal` returns a hard-coded `False` in the network attack: it has no dead ends, or
detecting them is left to the planner. Puzznic and Flipull compute a
positional dead end, and Flipull's is exact, because the rules are known in Python and it can
ask outright whether any throw would connect. The simulator environments compute real ones
too: a water network whose service has collapsed, a blacked-out grid, a growing season whose
water budget is spent, and a city whose adaptation has already cost more than the target. The
two emulator environments take theirs from the game: a wrapper's `game_over()`, a lost life,
or an integration's done condition.

`Environment.capabilities()` reports the optional methods per class, so the table above is
checked against the code rather than trusted: `tests/test_interface.py` asserts the two agree.
Note that `validate` is provided by the base class for everything, derived from `simulate` and
`is_goal`, so no environment writes it out; `step` and `get_actions` have base defaults too,
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

This is the bridge back to symbolic planning. Planners use `literals` as the visited-set key,
and width-based methods (IW and its relatives) use them as the atoms whose novelty they
measure. It is also where each environment makes its central modelling decision: what counts
as the same state. The choice differs sharply per environment. Puzznic's literals are exact,
whereas the water network's deliberately bucket a continuous contamination level so that
search over a continuous model terminates. Each environment's doc has a "State" section
spelling out what it chose and what that costs.

States also commonly expose `depth`, and define `__eq__` and `__hash__`, and often `__lt__`
so they can be tie-broken inside a priority queue.

### Selecting an instance

Environments are constructed empty and loaded by integer index:

```python
env = EnvNASim()
env.set_index(3)        # 'small' benchmark
state, _ = env.reset()  # set_index must come first; reset raises without it
```

The index is a stable handle for "instance *n* of this environment", which is what a benchmark
runner wants, and the mapping from index to instance is listed in each environment's doc.
`generate_instance(seed)` and `set_instance(instance)` select an instance the same way, and
`info["generated"]` on the reset says which kind is loaded.

### Bringing your own environment

Planners call environments directly; there is no wrapper to construct. An environment brought
from outside the library counts as long as it answers the eight contract methods, the
generator's two included: `implements_contract` checks structurally, so no subclassing is
required.

## Rendering a trace

```python
trace = env.simulate(plan)
env.render_trace(trace, "plan.gif")        # an animated GIF, one frame per state
env.render_trace(trace, "plan-frames/")    # a directory of independent PNGs
```

Rendering a trace is one image per state and nothing more: the state's own text, typeset. See
[docs/rendering.md](docs/rendering.md).

## Planners

| Family | Where | What it needs from an environment |
|---|---|---|
| Width-based: IW(k), Iterated Width, SIW, BFWS | [`planiverse/planners/width/`](planiverse/planners/width/) | `successors` and `literals`; a `progress` callback helps |
| Rollout IW, and π-IW with a policy it learns as it plans | [`planiverse/planners/width/rollout.py`](planiverse/planners/width/rollout.py), [`policy.py`](planiverse/planners/width/policy.py) | `successors` and `literals`; a `progress` callback stands in for the score |
| MCTS / UCT | [`planiverse/planners/mcts.py`](planiverse/planners/mcts.py) | `successors`; a `reward` callback helps a lot |
| Future State Maximization | [`planiverse/planners/fsx.py`](planiverse/planners/fsx.py) | `successors`, and nothing else: no goal, no heuristic |
| Tree search / A* | [`planiverse/planners/tree_search.py`](planiverse/planners/tree_search.py) | a heuristic and a cost function |

```python
from planiverse.planners.width import IWSearch, BFWSSearch, Budget

env.set_index(0)
result = IWSearch(width=2).solve(env, Budget(max_expansions=5000, max_seconds=60))
if result:
    env.validate(result.plan)
```

The width-based family is documented in [docs/planners/width-based.md](docs/planners/width-based.md),
including three things that change when the task is a simulator: (1) there is no goal
conjunction to count, so SIW and BFWS take a `progress` callback instead; (2) expansions are
expensive, so every search takes a budget and reports what it spent; and (3) dead ends are
real, and detecting them is most of what makes a simulator task searchable. Rollout IW and
π-IW are in [docs/planners/rollout-width.md](docs/planners/rollout-width.md): the novelty
filter kept, the breadth-first order replaced by rollouts that commit to an action every few
hundred expansions, and in π-IW a small policy network, trained on the planner's own
lookaheads, that steers the rollouts. MCTS and Future State Maximization are in
[docs/planners/sampling-based.md](docs/planners/sampling-based.md); FSX is the odd one, since
it is given no goal and no heuristic and picks whichever action leaves the most futures open,
which makes `option_count` a goal-free measure of how close a state is to being stuck.

## Benchmarking

`planiverse-bench` is the tool paper's evaluation protocol as code: the five planner
configurations the paper compares plus Rollout IW and π-IW, on every bundled instance of every
environment in the paper's tables, under a 30-minute wall-clock limit, an 8 GB address-space
cap and a 500,000-expansion bound, with five seeds for each of the four planners that take
one, on a SLURM cluster or on one machine. There is no configuration file, because the
protocol is the point.

```bash
tools/setup_benchmark.sh --partition <p> --qos <q>   # venv, install, then `generate`
bash sandbox/submit.sh                               # or: bash sandbox/run_local.sh 8
planiverse-bench report --sandbox-dir sandbox
```

`generate` asks each registered environment how many instances it has and writes one command
per planner, instance and seed, and one SLURM job array per planner or per seed of a seeded
planner. Every run ends in exactly one status, written to a result file whatever happened, and
`report` turns the results into the paper's tables, figures and quoted numbers. The sandbox
behind the paper is `paper-results.zip` on the
[release page](https://github.com/MFaisalZaki/Planiverse/releases); unzip it beside the
repository and `report` regenerates every number from it. [docs/benchmark.md](docs/benchmark.md)
has the protocol, the statuses and the report, and what the paper still has to take in.

## Writing a planner

Environments are planner-agnostic.
[`planiverse/planners/tree_search.py`](planiverse/planners/tree_search.py) contains a small
best-first `TreeSearchPlanner` that works against any environment implementing the contract:

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

The pieces you supply are a `Heuristic` and a `CostFunction`, both callables over states and
traces. Note that `PriorityQueue` pushes `(priority, item)` tuples, so ties compare the items
themselves, which is why state and action classes define `__lt__`.

## Architecture

One flat package, one base class, and a registry:

```
planiverse/environments/
├── base.py          # Environment: the eight-method contract, and nothing else
├── registry.py      # EnvironmentSpec per environment: instances, tags, deps, state identity
├── generation.py    # what the generators share: a seeded draw, a bounded search, a retry loop
├── games/           # the four games, reimplemented in pure Python
├── slingshot/       # a physics puzzle on pymunk
├── artillery/       # ballistics over a destructible field, pure Python
├── tower_defence/   # waves fought by simulation, pure Python
├── fluid/           # a cellular automaton of water, pure Python
├── billiards/       # pool on pooltool
├── lemmings/        # a crowd of walkers, pure Python
├── micropolis/      # a city on the Micropolis engine, built from source
├── factory/         # an early-game factory on factory-sim, built from source
├── emulated/        # one environment per emulator (PyBoy, Stable-Retro), for any game
└── <one subpackage per simulator-backed environment>
```

The taxonomy is data. `EnvironmentSpec` carries what a planner might select on:

| Field | What it tells a planner |
|---|---|
| `deterministic` | whether expanding a state twice gives the same children |
| `state_identity` | `value`, `path` or `snapshot`: how branching is possible at all |
| `requires` | third-party modules, so listing the catalogue imports none of them |
| `generates` | what `generate_instance` draws at random, in words |
| `tags` | the family (`game`, `operational`, `security`) plus finer ones like `continuous-dynamics` |

`state_identity` is the one worth understanding. A `value` state carries its own contents and
expanding is pure. A `path` state *is* the decision sequence, replayed on demand, which is
sound only because the simulator is deterministic. A `snapshot` state carries a serialised
emulator image (a save state) and is told apart from others by what the game wrapper or the
integration reads off it. Most simulators are step-only and cannot be rewound; that is the
property that decides whether something can be a Planiverse environment at all.

## Adding an environment

1. Subclass `Environment` (`planiverse/environments/base.py`) and implement the eight methods.
2. Define a state class exposing `literals`, `__eq__` and, if search will hash it, `__hash__`.
   Decide deliberately how coarse `literals` should be; that decision is your state space.
3. Filter self-loops out of `successors` (`if successor_state == state: continue`); every
   bundled environment does this, and planners rely on it. Check that it can actually fire: if
   `literals` include a step counter, no successor ever equals its parent and the filter is
   dead code.
4. Give it a generator, which is part of the contract: `set_instance` takes an instance as
   plain data and `generate_instance` draws one from a seed and selects it. Everything random
   comes from `random.Random(seed)`, so the same seed gives the same instance anywhere. Keep
   the instance in one attribute that `reset` builds from, so `set_index` and `set_instance`
   are two ways of filling the same slot. If a draw can be unsolvable, check it with
   `planiverse.environments.generation.solvable_draw` before handing it out.
5. Add an `EnvironmentSpec` to `planiverse/environments/registry.py`; that is what puts it in
   the catalogue and in `make()`.
6. Add a doc under `docs/environments/` and a row to the catalogue above.

## Repository layout

```
planiverse/
├── environments/
│   ├── base.py                         # Environment, the one base class
│   ├── registry.py                     # EnvironmentSpec, list_environments(), make()
│   ├── generation.py                   # rng, bounded_search, draw_until, solvable_draw
│   ├── games/                          # the four games in pure Python, nothing to supply
│   │   ├── puzznic.py                  # PuzznicGame
│   │   ├── flipull.py                  # FlipullGame
│   │   ├── lolo.py                     # LoloGame
│   │   └── amazing_tater.py            # AmazingTaterGame
│   ├── emulated/                       # any game, nothing shipped
│   │   ├── game_boy.py                 # GameBoyEnv: a cartridge under PyBoy, read through its wrappers
│   │   └── stable_retro.py             # RetroEnv: a Stable-Retro integration from its save states
│   ├── slingshot/                      # SlingshotEnv: a physics puzzle on pymunk
│   ├── artillery/                      # ArtilleryEnv: ballistics over a destructible field
│   ├── tower_defence/                  # TowerDefenceEnv: waves fought by simulation
│   ├── fluid/                          # FluidEnv: a cellular automaton of water
│   ├── billiards/                      # BilliardsEnv: pool on pooltool
│   ├── lemmings/                       # LemmingsEnv: a crowd of walkers steered with skills
│   ├── micropolis/                     # MicropolisEnv: a city on the Micropolis engine
│   ├── factory/                        # FactoryEnv: an early-game factory on factory-sim
│   ├── flood_transport/                # FloodTransportEnv (after MAAT), a city drawn from a seed
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
tools/export_maat_city.py               # exports a MAAT city as a flood instance
tests/
├── test_interface.py                   # the contract, across every environment
├── test_generators.py                  # the instance generators, across every environment
├── test_emulated.py                    # the two emulator environments
├── counter_rom.py, sm83.py             # the test cartridge, and the assembler that builds it
└── test_<environment>.py               # per-environment behaviour
```

## Attribution

Planiverse adapts several upstream simulators. Each is credited in its own doc; the sources are:

| Environment | Upstream |
|---|---|
| Network attack | [NASim](https://github.com/MFaisalZaki/NetworkAttackSimulator) (fork, MIT), [PenGym](https://github.com/cyb3rlab/PenGym) |
| Water distribution | [WNTR](https://github.com/USEPA/WNTR) (US EPA's EPANET wrapper) |
| Power grid | [Grid2Op](https://github.com/Grid2Op/grid2op) (RTE) |
| Crop management | [PCSE / WOFOST](https://github.com/ajwdewit/pcse) (Wageningen University) |
| Slingshot | [pymunk](https://www.pymunk.org/) (MIT), the Python binding of Chipmunk2D (MIT) |
| Billiards | [pooltool](https://github.com/ekiefl/pooltool) (Apache-2.0) |
| Micropolis city | [MicropolisCore](https://github.com/SimHacker/micropolis) (GPL-3.0 with Electronic Arts' additional terms), built from source by `scripts/build_micropolis.sh` |
| Factory | [factory-sim](https://github.com/divagr18/factory-sim) (MIT), a tick-exact C simulator of Factorio's early game measured on the game itself, built from source by `scripts/build_factory_sim.sh` |
| Flood adaptation | [floods_transport_rl](https://github.com/MLSM-at-DTU/floods_transport_rl) (DTU, MIT), the MAAT model, with a city of its own |
| Game Boy | [PyBoy](https://github.com/Baekalfen/PyBoy) (LGPL-3.0), with the game wrappers of [MFaisalZaki/PyBoy](https://github.com/MFaisalZaki/PyBoy) |
| Stable-Retro | [Stable-Retro](https://github.com/Farama-Foundation/stable-retro) (Farama Foundation, MIT), which ships Airstriker |

### Studied titles

The four game environments reimplement commercially published titles in Python. This
repository ships no ROM image, no original code and no original graphics, and runs none of
the original programs. *Adventures of Lolo* (HAL Laboratory / Nintendo), *Puzznic* and
*Flipull* (Taito), *Amazing Tater* (Atlus) and *Super Mario Land* (Nintendo) are the copyright
works and trade marks of their respective owners, used here descriptively. This project is
unofficial and unaffiliated. See [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) and
[docs/provenance.md](docs/provenance.md).

## Status

What is in the tree:

- Nineteen environments: five simulator-backed operational ones (water distribution, power
  grid, crop management, flood adaptation, a city on the Micropolis engine), the NASim network attack, four games reimplemented
  in pure Python, a slingshot physics puzzle on pymunk, billiards on pooltool, an artillery
  game, a tower defence, a cellular fluid puzzle, a Lemmings-like, an early-game factory on
  factory-sim, and two generic emulator environments, one for any Game Boy cartridge under
  PyBoy and one for any Stable-Retro integration. Every one ships its bundled instances and
  generates more from a seed.
- Nine planners: IW(k), Iterated Width, SIW, BFWS and Iterated BFWS; Rollout IW and π-IW, the
  latter with a policy it learns from its own lookaheads; MCTS; and Future State Maximization.
- `planiverse-bench`, the evaluation protocol as code: seven planner configurations, five seeds
  for the four that take one, and a report that builds the tables and figures from the results.
- A test suite that skips what it cannot build.

Open:

- [ ] Optional dependency groups, so one environment does not pull in all of them. Today there is
      one dependency list and a `dev` extra.
- [ ] A generated benchmark: `planiverse-bench` runs the bundled instances only.

## Licence

GPL-3.0 for this repository's own code, documentation and benchmark definitions. See
[LICENSE](LICENSE). The level layouts three of the games ship are derived from the original
titles and are third-party material that this project has no right to sublicense. See
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for the copyright holders and the basis on
which they are included.
