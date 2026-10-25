# Planiverse

[![CI](https://github.com/MFaisalZaki/Planiverse/actions/workflows/ci.yml/badge.svg)](https://github.com/MFaisalZaki/Planiverse/actions/workflows/ci.yml)

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
| Epidemic | `epidemic` | 100 outbreaks | the population, the seed infections, the beds, the disruption budget, the horizon and the death target | operational, health | [docs](docs/environments/epidemic.md) |
| Traffic signals | `traffic` | 100 mornings | the trips, their spacing, the horizon and the travel-time target | operational, infrastructure, traffic | [docs](docs/environments/traffic.md) |
| Airspace | `airspace` | 100 sectors | the aircraft, their entries, speeds and exits, the horizon and the exit-time target | operational, aviation | [docs](docs/environments/airspace.md) |
| Reservoir | `reservoir` | 100 years | the inflows, the demands, the capacities, the reserves and the shortfall target | operational, infrastructure, water | [docs](docs/environments/reservoir.md) |
| Inventory | `inventory` | 100 seasons | the demand distribution and the season's demand, the costs, the capacities, the lead times, the initial stock, the lost-sales allowance and the profit target | operational, supply-chain | [docs](docs/environments/inventory.md) |
| Microgrid | `microgrid` | 100 instances | the benchmark microgrid, the day and start hour, the horizon, the starting charge and the cost target | operational, infrastructure, energy | [docs](docs/environments/microgrid.md) |
| Railway | `railway` | 100 timetables | the cities, the rail network, the trains' stations and targets, the horizon and the train-steps target | operational, infrastructure, transport, scheduling | [docs](docs/environments/railway.md) |
| Flood adaptation | `flood_transport` | 100 scenarios | the city, its storms, the horizon and the measures on offer | operational, infrastructure, climate | [docs](docs/environments/flood-transport.md) |
| Network attack | `network_attack` | 100 networks | network topology, hosts, services, OSs and exploits | security | [docs](docs/environments/network-attack.md) |
| Puzznic | `puzznic` | 100 levels | board size, wall layout, block colours and pairs | game | [docs](docs/environments/puzznic.md) |
| Flipull | `flipull` | 100 stages | wall size, block types, arrangement and clear target | game | [docs](docs/environments/flipull.md) |
| Adventures of Lolo | `lolo` | 100 rooms | terrain, hearts, Emerald Framers, Snakeys and Medusas | game | [docs](docs/environments/lolo.md) |
| Slingshot | `slingshot` | 100 levels | the structures, their materials, where the targets sit, and the shots | game, physics | [docs](docs/environments/slingshot.md) |
| Tower defence | `tower_defence` | 100 maps | the path, the building slots, the waves, the gold and the lives | game | [docs](docs/environments/tower-defence.md) |
| Fluid | `fluid` | 100 caves | the cave, the spring, the basin, the drain, the water needed and the digs allowed | game | [docs](docs/environments/fluid.md) |
| Pipe Dream | `pipe_dream` | 100 levels | the board and its walls, the start, the queue of pieces, the distance, the countdown and the pace | game | [docs](docs/environments/pipe-dream.md) |
| Lemmings | `lemmings` | 100 levels | the platforms, gaps and walls, the entrance and exit, the crowd, the quota and the skills | game | [docs](docs/environments/lemmings.md) |
| Amazing Tater | `amazing_tater` | 100 rooms | room size, walls, blocks, pits, turnstiles and taters | game | [docs](docs/environments/amazing-tater.md) |
| Game Boy | `game_boy` | one per stage, level or room the cartridge's wrapper reaches | the stage, the timer seed, and an opening played from its first frame | game, emulator | [docs](docs/environments/game-boy.md) |
| Stable-Retro | `retro` | one per save state the integration ships (Airstriker: 1) | the save state, an opening played from it, and the goal | game, emulator | [docs](docs/environments/stable-retro.md) |

```python
from planiverse.environments import list_environments, make

[spec.name for spec in list_environments(tag="operational")]
# ['airspace', 'crop_management', 'epidemic', 'factory', 'flood_transport', 'inventory',
#  'microgrid', 'micropolis', 'power_grid', 'railway', 'reservoir', 'traffic', 'water_network']

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

Requires Python 3.11, 3.12 or 3.13. Linux, macOS and Windows are all supported, and the
[CI workflow](.github/workflows/ci.yml) installs and tests on each of them.

```bash
git clone https://github.com/MFaisalZaki/Planiverse.git
cd Planiverse
pip install -e ".[dev]"              # the dev extra adds pytest
python scripts/install_flatland.py   # the railway
python scripts/install_or_gym.py     # the inventory
```

Or, with poetry:

```bash
poetry env use python3.12
poetry install --extras dev
poetry run python scripts/install_flatland.py
poetry run python scripts/install_or_gym.py
```

One install gets you every environment, on every supported Python, and nothing has to be
supplied: the games and the flood, railway and inventory environments are self-contained, the
water, power grid, crop, epidemic, traffic, reservoir and microgrid environments ship their
benchmark data inside their dependencies, so they run offline, and Stable-Retro ships
Airstriker. The one exception is a Game Boy cartridge for `game_boy`, which is copyrighted and
comes from you. `tests/test_packaging.py` walks the import graph from each environment's entry
point and fails if anything it reaches is undeclared, so a dependency that only works because
another package happens to pull it in cannot go missing silently.

These dependencies are not declared the plain way, because their publishers do not cover
every platform or declare bounds the rest of the dependencies cannot meet. `tools/check_install.py` prints what the install in front of you covers and
what is missing.

- **Flatland** (railway) is installed by `scripts/install_flatland.py`. It declares `numpy<2`,
  which has no Python 3.13 wheels, and upper bounds on matplotlib, networkx, contourpy and
  rpds-py that the other dependencies exceed; the script installs it without its list and the
  list beside it with those bounds dropped.
- **OR-Gym** (inventory) is installed by `scripts/install_or_gym.py`. It declares `gym<=0.19`,
  whose metadata current pip refuses, so the script installs it without its list and the list
  beside it with Gym 0.26, which python-microgrid brings in. The inventory environment restores
  the two numpy names OR-Gym uses that numpy 2 removed.
- **pywr** (reservoirs) publishes wheels for Linux and Windows only, so the dependency list
  leaves it out on macOS. To have the reservoir environment there, build it from source:
  `brew install glpk`, then
  `CFLAGS="-I$(brew --prefix glpk)/include" LDFLAGS="-L$(brew --prefix glpk)/lib" PYWR_BUILD_LPSOLVE=false pip install pywr`.
- **Stable-Retro** publishes no Windows wheels and supports Windows through WSL2, so the
  dependency list leaves it out on Windows. The Game Boy environment under PyBoy is unaffected.
- **OpenSpiel** publishes wheels for Linux and macOS only, so the dependency list leaves it out
  on Windows, and with it `planiverse.spiel`.
- **libsumo**'s Windows wheel does not import on its own (a library of the same name earlier
  on the system's search path shadows one of SUMO's), so `planiverse.environments` loads the
  SUMO libraries by path first on Windows. Nothing to do; without it the import fails with
  "DLL load failed while importing _libsumo".

Two simulators are built or installed by the scripts under `scripts/` rather than from PyPI: the
Micropolis engine (`build_micropolis.sh`, needs `git`, a C++ compiler, the Python headers,
`perl` and `swig`) and BlueSky (`install_bluesky.sh`). They are shell scripts, so on Windows run
them from WSL2 or Git Bash. An environment whose dependency is missing is skipped by the tests
and recorded as `UNSUPPORTED` by the benchmark; everything else runs. The Micropolis engine
builds on macOS too, but its arithmetic there differs from Linux, where the bundled cities were
drawn; see [its doc](docs/environments/micropolis.md).

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
from planiverse.environments.games.puzznic.environment import PuzznicGame

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
`random.Random(seed)`. Second, the instance is plain data of the same shape as the bundled ones
(a level string, a `[stage, target]` pair, a tuple of rows, a dict of scenario fields), so
`json.dumps` writes it down and `set_instance` takes it back, and a generated benchmark replays
exactly. Third, a draw is checked before it is handed out, because a random board is usually
unsolvable and an unsolvable instance is not an instance: a planner cannot tell "no plan" from
"not yet". Most generators search the draw within a bounded budget; Flipull's, given a size and
no target, explores the draw exhaustively and takes the fewest blocks reachable as the target;
and the operational environments whose target is the best of a few reference policies (see
[docs/operational.md](docs/operational.md)) are solvable by construction. Everywhere, the plan
is left in `env.witness` and what the search spent in `env.witness_expansions`, and
`min_plan_length` turns the first into a difficulty knob. The check is a bias, since it favours
instances a small search can solve, and it is a knob too: `solvable=False` hands out the raw
draw.

What each generator varies is in the catalogue above, and the options are in each environment's
doc. A generator draws to the shape of the originals: a game takes the layout options a caller
leaves unset from the profile of one of its bundled instances (its size, its counts of blocks or
hearts, its share of scenery), and a simulator draws on the simulator's own data (WNTR's
networks, grid2op's time series, PCSE's weather, NASim's scenario generator). Every environment
but the two emulators ships a hundred instances. Where the originals fall short of that (Flipull
and the five simulator environments) the set is made up with instances the generator drew, after
the originals, with the seed each came from recorded with it and the plan it was accepted on in
the tests, so they can be re-derived. The techniques are the standard ones, and each
environment's doc gives the references: generate-and-test with a solvability check is
search-based procedural content generation
([Togelius et al., 2011](https://doi.org/10.1109/TCIAIG.2011.2148116);
[Shaker et al., 2016](https://pcgbook.com/)), and the check is breadth-first search by default.
The shared machinery is [`planiverse/environments/generation.py`](planiverse/environments/generation.py):
the seeded draw, the bounded breadth-first search, the retry loop, and the board helpers the
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
| `simulate(plan)` | `[state, ...]` | Replays a plan from the initial state. The base class's replays it through `__advance__(state, action)`, or through `successors`. |
| `generate_instance(seed, **options)` | the instance | Draws a fresh instance from a seed, selects it, and returns it as plain data. The same seed always gives the same instance. |
| `set_instance(instance)` | | Selects an instance drawn earlier. |
| `validate(plan)` | `bool` | Optional. |
| `get_actions()` | `[action, ...]` | Optional. |

What each environment provides:

| | `reset` | `set_index` | `successors` | `is_goal` | `is_terminal` | `simulate` | `generate_instance` | `validate` | `get_actions` | `close` |
|---|---|---|---|---|---|---|---|---|---|---|
| `PuzznicGame` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
| `FlipullGame` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
| `LoloGame` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
| `AmazingTaterGame` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
| `SlingshotEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
| `TowerDefenceEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
| `FluidEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
| `PipeDreamEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
| `LemmingsEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
| `MicropolisEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
| `EpidemicEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
| `TrafficEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `AirspaceEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ReservoirEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `FactoryEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `WaterNetworkEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `PowerGridEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `CropEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `FloodTransportEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
| `EnvNASim` | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | | |
| `GameBoyEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `RetroEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

⚠️ `is_terminal` returns a hard-coded `False` in the network attack, so detecting its dead
ends is left to the planner. Puzznic and Flipull compute a
positional dead end, and Flipull's is exact, because the rules are known in Python and it can
ask outright whether any throw would connect. The simulator environments compute real ones
too: a water network whose service has collapsed, a blacked-out grid, a growing season whose
water budget is spent, and a city whose adaptation has already cost more than the target. The
two emulator environments take theirs from the game: a wrapper's `game_over()`, a lost life,
or an integration's done condition.

`Environment.capabilities()` reports the optional methods per class, so the table above is
checked against the code rather than trusted: `tests/test_interface.py` asserts the two agree.
Note that `validate` and `simulate` are provided by the base class for everything, derived from
`reset`, `successors` and `is_goal`, so only Puzznic and the network attack write their own
`simulate`; `get_actions` has a base
default too, but it only explains its own absence, and so do the other contract methods, so a
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

### Through OpenSpiel

Every environment is also an [OpenSpiel](https://github.com/google-deepmind/open_spiel) game,
and every planner runs on any single-player OpenSpiel game, ours or OpenSpiel's own
(`planiverse/spiel.py`):

```python
import pyspiel
import planiverse.spiel as spiel

game = pyspiel.load_game("planiverse_lolo", {"index": 3})   # or {"seed": 7}
actions, result = spiel.solve(game, "bfws")                 # any benchmark tag, or a planner

state = game.new_initial_state()
for action in actions:
    state.apply_action(action)
state.returns()                                             # [1.0]: the goal

spiel.solve(pyspiel.load_game("deep_sea"), "iw")            # one of OpenSpiel's games
```

A `planiverse_*` game is single-player and deterministic, with reward 1 at a goal and 0
anywhere else; a goal and a dead end are both terminal, with no legal actions. Its states
carry `literals` and hand any other attribute through to the simulator's own state. On
OpenSpiel's games, the atoms the planners see are the nonzero entries of the observation
tensor, and chance is resolved by a generator seeded on the history, so expansion stays
deterministic. A goal there is a terminal state with a positive return unless `solve` is
given `goal=`. A state is observed the OpenSpiel way: `observation_string` is the simulator
state's own text, and where the state carries readings (the operational environments and the
tower defence), `observation_tensor` holds them, each named in the `dict` of the game's observer.
`render_trace` draws a trace of OpenSpiel states as it draws an environment's. OpenSpiel
publishes no Windows wheels, so a Windows install leaves it out.

## Rendering a trace

```python
trace = env.simulate(plan)
env.render_trace(trace, "plan.gif")        # an animated GIF, one frame per state
env.render_trace(trace, "plan-frames/")    # a directory of independent PNGs
```

Rendering a trace is one image per state: the state's text picture, typeset black on white. A
board has its walls joined into lines and its pieces as symbols, a field or map is laid out on a
grid of characters, and an operational system is drawn as a schematic of its parts (tanks,
networks, a radar scope, a town), with no figures under it. See
[docs/rendering.md](docs/rendering.md).

## Planners

| Family | Where | What it needs from an environment |
|---|---|---|
| Width-based: `IW`, `BFWS` (with BFWS(R) as `relevant="iw"`), quantified, count-based and approximate novelty | [`planiverse/planners/width/`](planiverse/planners/width/) | `successors` and `literals`; a `progress` callback helps |
| Heuristic search: greedy best-first, A* and weighted A*, restarting WA*, ε-greedy, type-based, diverse, local exploration by random walks, BULB, limited discrepancy search, FESS | [`planiverse/planners/heuristic/`](planiverse/planners/heuristic/) | a `progress` measure |
| Sampling: Go-Explore, KPIECE | [`planiverse/planners/sampling/`](planiverse/planners/sampling/) | `successors`; `progress` helps |
| Add-ons: dominated-action pruning, focused macro-actions | [`planiverse/planners/pruning.py`](planiverse/planners/pruning.py), [`planiverse/planners/macros.py`](planiverse/planners/macros.py) | `successors` and `literals` |

```python
from planiverse.planners.width import IW, BFWS, Budget

env.set_index(0)
result = IW(width=2).solve(env, Budget(max_expansions=5000, max_seconds=60))
if result:
    env.validate(result.plan)
```

The width-based family is documented in [docs/planners/width-based.md](docs/planners/width-based.md),
including three things that change when the task is a simulator: (1) there is no goal conjunction
to count, so BFWS takes a `progress` callback instead; (2) expansions are expensive, so
every search takes a budget and reports what it spent; and (3) dead ends are real. Go-Explore
and KPIECE, which explore from cells of the state space with random actions, are in
[docs/planners/sampling-based.md](docs/planners/sampling-based.md).

Every planner, with its reference, what it needs from an environment, its benchmark tag and
the choices its implementation makes, is in
[docs/planners/catalogue.md](docs/planners/catalogue.md).

This is a planning library, not a learning one, and that is a rule the tree follows: no
planner takes a reward, none learns before or while it plans, and none is a Monte Carlo tree
search. Everything runs on the black-box goal test and, at most, a `progress` heuristic. Every
planner here is in the benchmark: `planiverse-bench generate` runs them all, `--reference`
the two reference configurations.

## Benchmarking

`planiverse-bench` is the library's evaluation protocol as code: every planner in the library,
twenty configurations with BFWS and IW as the two reference ones, on every instance of every
environment, under a 30-minute wall-clock limit, an 8 GB address-space cap and a
500,000-expansion bound, with five seeds for every planner that takes one, on a SLURM cluster
or on one machine. `--reference` keeps it to the two. There is no configuration file, because
the protocol is the point.

```bash
tools/setup_benchmark.sh --partition <p> --qos <q>   # venv, install, then `generate`
bash sandbox/slurm/submit_all.sh                     # or: bash sandbox/run_local.sh 8
planiverse-bench report --sandbox-dir sandbox
```

The harness is [pyPMTEvalToolkit](https://github.com/pyPMT/pyPMTEvalToolkit)'s. `generate` asks
each registered environment how many instances it has and writes, through the toolkit, one
command per planner, instance and seed, and SLURM job arrays of at most 1,000 elements per
planner or per seed of a seeded planner. `solve`, the one stage that is Planiverse's own, runs a
planner on a simulator and writes the toolkit's result format whatever happened, and `report`
is the toolkit's `analyze` and `report`: a CSV, coverage, outcome and IPC-score tables in text
and LaTeX, and survival plots. [docs/benchmark.md](docs/benchmark.md) has the protocol, the
statuses and the report.

## Writing a planner

Environments are planner-agnostic. A planner is a class with `solve(env, budget=None,
state=None)` that returns a `SearchResult`. Breadth-first search, written against the
contract, is the whole pattern:

```python
from collections import deque

from planiverse.planners.common import finish
from planiverse.planners.width import Budget, SearchStatistics


class BreadthFirstSearch:
    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics()
        if state is None:
            state, _ = env.reset()
        frontier, closed = deque([(state, [], [state])]), {state.literals}
        while frontier:
            if budget.exhausted(statistics.expansions):
                return finish("out_of_budget", None, [], statistics, budget)
            node, plan, trace = frontier.popleft()
            statistics.expansions += 1
            for action, successor in env.successors(node):
                if successor.literals in closed:
                    continue
                closed.add(successor.literals)
                if env.is_goal(successor):
                    return finish("solved", plan + [action], trace + [successor],
                                  statistics, budget)
                if not env.is_terminal(successor):
                    frontier.append((successor, plan + [action], trace + [successor]))
        return finish("exhausted", None, [], statistics, budget)
```

Written to the contract, it runs on every environment here and, through
`planiverse.spiel.solve`, on any single-player OpenSpiel game. `finish` fills in the elapsed
time; the planners in [`planiverse/planners/`](planiverse/planners/) are built the same way.

## Architecture

One package, one base class, a registry, and two families of subpackages:

```
planiverse/environments/
├── base.py          # Environment: the eight-method contract
├── registry.py      # EnvironmentSpec per environment: instances, tags, deps, state identity
├── generation.py    # what the generators share: a seeded draw, a bounded search, a retry loop
├── games/           # every game, one subpackage each, its environment in environment.py
│   ├── puzznic/, flipull/, lolo/, amazing_tater/   # reimplemented in pure Python
│   ├── slingshot/       # a physics puzzle on pymunk
│   ├── tower_defence/   # waves fought by simulation, pure Python
│   ├── fluid/           # a cellular automaton of water, pure Python
│   ├── pipe_dream/      # a Pipe Dream-like, pure Python
│   ├── lemmings/        # a crowd of walkers, pure Python
│   ├── game_boy/        # any cartridge under PyBoy
│   └── retro/           # any Stable-Retro integration
└── operational/     # every simulator-backed environment, one subpackage each
    ├── micropolis/      # a city on the Micropolis engine, built from source
    ├── factory/         # an early-game factory on factory-sim, built from source
    ├── epidemic/        # an outbreak on Covasim
    ├── traffic/         # signals on a SUMO grid
    ├── airspace/        # crossing aircraft on BlueSky
    ├── reservoir/       # reservoir operations on pywr
    ├── inventory/       # a multi-echelon supply chain on or-gym
    ├── microgrid/       # microgrid operations on python-microgrid
    ├── railway/         # train dispatching on Flatland
    └── water_network/, power_grid/, crop_management/, flood_transport/, network_attack/
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
│   ├── games/                          # every game, its environment in <name>/environment.py
│   │   ├── puzznic/                    # PuzznicGame, in pure Python, nothing to supply
│   │   ├── flipull/                    # FlipullGame
│   │   ├── lolo/                       # LoloGame
│   │   ├── amazing_tater/              # AmazingTaterGame
│   │   ├── slingshot/                  # SlingshotEnv: a physics puzzle on pymunk
│   │   ├── tower_defence/              # TowerDefenceEnv: waves fought by simulation
│   │   ├── fluid/                      # FluidEnv: a cellular automaton of water
│   │   ├── pipe_dream/                 # PipeDreamEnv: pipe laid from a queue ahead of a flow
│   │   ├── lemmings/                   # LemmingsEnv: a crowd of walkers steered with skills
│   │   ├── game_boy/                   # GameBoyEnv: a cartridge under PyBoy, read through its wrappers
│   │   └── retro/                      # RetroEnv: a Stable-Retro integration from its save states
│   └── operational/                    # every simulator-backed environment
│       ├── micropolis/                 # MicropolisEnv: a city on the Micropolis engine
│       ├── factory/                    # FactoryEnv: an early-game factory on factory-sim
│       ├── epidemic/                   # EpidemicEnv: an outbreak on Covasim
│       ├── traffic/                    # TrafficEnv: signals on a SUMO grid
│       ├── airspace/                   # AirspaceEnv: crossing aircraft on BlueSky
│       ├── reservoir/                  # ReservoirEnv: reservoir operations on pywr
│       ├── inventory/                  # InventoryEnv: a multi-echelon supply chain on or-gym
│       ├── microgrid/                  # MicrogridEnv: microgrid operations on python-microgrid
│       ├── railway/                    # RailwayEnv: train dispatching on Flatland
│       ├── flood_transport/            # FloodTransportEnv (after MAAT), a city drawn from a seed
│       ├── network_attack/             # EnvNASim (wraps NASim)
│       ├── water_network/              # WaterNetworkEnv (WNTR/EPANET)
│       ├── power_grid/                 # PowerGridEnv (Grid2Op)
│       └── crop_management/            # CropEnv (PCSE/WOFOST)
├── planners/
│   ├── width/                          # IW, BFWS, and the other novelty planners
│   ├── heuristic/                      # best-first family, BULB, limited discrepancy search, FESS
│   ├── sampling/                       # Go-Explore, KPIECE
│   ├── common.py                       # SuccessorCache and the helpers the planners share
│   └── pruning.py, macros.py           # dominated-action pruning, focused macro-actions
├── spiel.py                            # every environment a pyspiel game; solve(game, planner)
├── rendering/                          # traces to GIF or PNG frames, state pictures typeset as text (env.render_trace delegates here)
└── benchmark/                          # planiverse-bench: the evaluation protocol
    ├── __init__.py                     # generate / solve / report, and the protocol's constants
    ├── planners.py                     # every planner, as a benchmark configuration under its tag
    └── measures.py                     # per-environment progress measures
docs/environments/                      # per-environment documentation
docs/planners/                          # the planner catalogue, the width-based and sampling docs
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
| Micropolis city | [MicropolisCore](https://github.com/SimHacker/micropolis) (GPL-3.0 with Electronic Arts' additional terms), built from source by `scripts/build_micropolis.sh` |
| Factory | [factory-sim](https://github.com/divagr18/factory-sim) (MIT), a tick-exact C simulator of Factorio's early game measured on the game itself, from PyPI as `factory-sim` |
| Epidemic | [Covasim](https://github.com/institutefordiseasemodeling/covasim) (Institute for Disease Modeling, MIT) |
| Traffic signals | [SUMO](https://eclipse.dev/sumo/) (Eclipse Foundation, EPL-2.0 or GPL-2.0-or-later), through `eclipse-sumo` and `libsumo` |
| Airspace | [BlueSky](https://github.com/TUDelft-CNS-ATM/bluesky) (TU Delft, MIT), with OpenAP (LGPL-3.0) and its navigation data (GPL-3.0), installed by `scripts/install_bluesky.sh` |
| Reservoir | [pywr](https://github.com/pywr/pywr) (University of Manchester, GPL-3.0-or-later) |
| Flood adaptation | [floods_transport_rl](https://github.com/MLSM-at-DTU/floods_transport_rl) (DTU, MIT), the MAAT model, with a city of its own |
| Game Boy | [PyBoy](https://github.com/Baekalfen/PyBoy) (LGPL-3.0), with the game wrappers of [MFaisalZaki/PyBoy](https://github.com/MFaisalZaki/PyBoy) |
| Stable-Retro | [Stable-Retro](https://github.com/Farama-Foundation/stable-retro) (Farama Foundation, MIT), which ships Airstriker |

### Studied titles

The four game environments reimplement commercially published titles in Python. This
repository ships no ROM image, no original code and no original graphics, and runs none of
the original programs. *Adventures of Lolo* (HAL Laboratory / Nintendo), *Puzznic* and
*Flipull* (Taito), *Amazing Tater* (Atlus) and *Super Mario Land* (Nintendo) are the copyright
works and trade marks of their respective owners, used here descriptively. Micropolis is a
registered trademark of Micropolis Corporation (Micropolis GmbH) and is licensed here as a
courtesy of the owner ([micropolis.com](https://micropolis.com/)); Factorio is a trademark of
Wube Software Ltd, named here to say what factory-sim was measured against. This project is
unofficial and unaffiliated. See [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) and
[docs/provenance.md](docs/provenance.md).

## Status

What is in the tree:

- Twenty-five environments: twelve simulator-backed operational ones (water distribution, power
  grid, crop management, flood adaptation, a city on the Micropolis engine, an epidemic on
  Covasim, traffic signals on SUMO, crossing aircraft on BlueSky, reservoirs on pywr, a railway
  on Flatland, a supply chain on OR-Gym and a microgrid on python-microgrid), the NASim network
  attack, four games reimplemented in pure Python, a slingshot physics puzzle on pymunk, a tower
  defence, a cellular fluid puzzle, a Pipe Dream-like, a Lemmings-like, an early-game
  factory on factory-sim, and two generic emulator environments, one for any Game Boy cartridge under
  PyBoy and one for any Stable-Retro integration. Every one ships its bundled instances and
  generates more from a seed.
- Twenty planner configurations: the width-based family, heuristic search, Go-Explore and
  KPIECE, and the two add-ons, catalogued in
  [docs/planners/catalogue.md](docs/planners/catalogue.md). No planner takes a reward or learns.
- `planiverse-bench`, the evaluation protocol as code: every planner in the library, five
  seeds for each that takes one, and a report that writes the tables, figures and quoted
  numbers from the results.
- A test suite that skips what it cannot build.

## Licence

GPL-3.0 for this repository's own code, documentation and benchmark definitions. See
[LICENSE](LICENSE). The level layouts three of the games ship are derived from the original
titles and are third-party material that this project has no right to sublicense. See
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for the copyright holders and the basis on
which they are included.
