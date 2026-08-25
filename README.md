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

| Environment | Class | Instances | Docs |
|---|---|---|---|
| Epidemic control | `EpiEnv` | 7 scenarios (SIR / SIRV / COVID) | [docs](docs/environments/epidemic-control.md) |
| Network attack | `EnvNASim` | 18 NASim benchmarks | [docs](docs/environments/network-attack.md) |
| Manufacturing | `MfgEnv` | 7 demand/capacity instances | [docs](docs/environments/manufacturing.md) |
| Urban planning | `UrbanPlanningEnv` | 2 cities (Kendall Square, St Andrews) | [docs](docs/environments/urban-planning.md) |
| Puzznic | `PuzznicGame` | 50 levels | [docs](docs/environments/puzznic.md) |
| Puzznic (Game Boy) | `PuzznicGBEnv` | 128 rounds (needs a ROM you supply) | [docs](docs/environments/puzznic-gb.md) |
| Super Mario Land | `SuperMarioEnv` | 12 levels (needs a ROM you supply) | [docs](docs/environments/super-mario-land.md) |
| Flipull (Game Boy) | `FlipullGBEnv` | 1 stage (needs a ROM you supply) | [docs](docs/environments/flipull-gb.md) |

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
from planiverse.problems.retro_games.puzznic import PuzznicGame

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
| `FlipullGBEnv` | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `SuperMarioEnv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — |
| `EnvNASim` | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | — | — | — |
| `EpiEnv` | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | — | — | — |
| `MfgEnv` | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | — | — | — |
| `UrbanPlanningEnv` | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | — | — | — |
| `PDDLGymEnv` | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | — | ✅ | — |

⚠️ `is_terminal` returns a hard-coded `False` in these environments: they have no dead ends, or
detecting them is left to the planner. The two Puzznics and Super Mario Land are the ones that really
compute a positional dead end; `FlipullGBEnv` computes one, but only the clock running out.

⚠️ `FlipullGBEnv.fix_index` accepts only `0`: no verified way to select a stage has been found, and
it asserts rather than quietly loading the wrong one. See
[its docs](docs/environments/flipull-gb.md#stages).

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

## Adding an environment

1. Subclass `RealWorldProblem` (`planiverse/problems/real_world_problems/base.py`) or `RetroGame`
   (`planiverse/problems/retro_games/base.py`).
2. Define a state class exposing `literals`, `__eq__`, and — if search will hash it — `__hash__`.
   Decide deliberately how coarse `literals` should be; that decision is your state space.
3. Implement `reset`, `fix_index`, `successors`, `is_goal`, `is_terminal`, and `simulate`.
4. Filter self-loops out of `successors` (`if successor_state == state: continue`) — every bundled
   environment does this, and planners rely on it. Check that it can actually fire: if `literals`
   include a step counter, no successor ever equals its parent and the filter is dead code (this is
   what happens in [urban planning](docs/environments/urban-planning.md#known-quirks)).
5. Add a doc under `docs/environments/` and a row to the catalog above.

## Repository layout

```
planiverse/
├── problems/
│   ├── real_world_problems/
│   │   ├── base.py                     # RealWorldProblem marker base
│   │   ├── epidemic_control/           # EpiEnv + vendored EpiPolicy + jsons/
│   │   ├── cyber_security_network_attack/  # EnvNASim (wraps NASim)
│   │   ├── manufacturing_environment/  # MfgEnv + data/
│   │   └── urban_planning/             # UrbanPlanningEnv + cities/
│   └── retro_games/
│       ├── base.py                     # RetroGame marker base
│       ├── puzznic.py                  # PuzznicGame
│       ├── puzznic_gb.py               # PuzznicGBEnv (PyBoy)
│       ├── flipull_gb.py               # FlipullGBEnv (PyBoy)
│       └── super_mario_bros_gb.py      # SuperMarioEnv (PyBoy)
├── planners/
│   └── super_mario_planner_gb.py       # TreeSearchPlanner, SuperMarioPlanner
└── simulator/
    ├── simulator.py                    # Simulator facade
    └── wrappers/
        ├── base.py                     # SimulatorBase interface
        ├── pddlgymenv.py               # PDDLGym adapter
        └── pddlgym/                    # vendored PDDLGym 0.0.7 — see its VENDORING.md
dev/                                    # scratch scripts — not part of the library
docs/environments/                      # per-environment documentation
tests/
├── sm83.py                             # minimal SM83 assembler, for the test cartridges
├── fake_puzznic_rom.py                 # synthetic Game Boy ROM with Puzznic's memory layout
└── fake_flipull_rom.py                 # synthetic Game Boy ROM with Flipull's memory layout
```

`dev/` is a scratch area and is **not** an entry point. `dev/dev.py` is stale: it imports names that
no longer exist (`SuperMario`, `super_mario_bros_grid`, `super_mario_planner_tile`) and will not run.
It is also the only thing that ever imported `pcg_benchmark`, which is no longer a declared
dependency — `pip install git+https://github.com/amidos2006/pcg_benchmark.git` if you revive it.
`dev/earthmodel.py` is a vendored copy of the c:GLOBAL gym environment (© Felix Strnad), kept for a
future port — it does not implement the Planiverse interface yet.

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

The flood/transport environment ([floods_transport_rl](https://github.com/MLSM-at-DTU/floods_transport_rl))
is referenced as a planned addition but is not yet in the tree.

## Status

- [x] README and per-environment docs
- [x] Super Mario Land via PyBoy, with world/level selection wired into `reset`
- [x] NASim network attack
- [x] Test suite (`poetry run pytest`)
- [ ] Flood application
- [ ] Optional dependency groups, so one environment doesn't pull in all of them
- [ ] Replace `pddlgym` (unmaintained, pins `pillow <10`, caps the PDDL wrapper at Python 3.12)
- [ ] `is_terminal` dead-end detection for the four environments that hard-code `False`
- [ ] Confirm Super Mario Land's level-complete address (`0xDFE8`) and enemy tile IDs
- [ ] `SuperMarioPlanner.search` returns `None` and has no replanning loop
- [x] Run `FlipullGBEnv` against a real `Flipull (USA).gb`, and correct what the memory map had
      wrong about the throw
- [ ] Find a stage-selection route for Flipull, so `fix_index` accepts more than `0`
- [ ] Work out what a Flipull throw actually hits — every row connects, so it is not simply the
      first block in the player's row

## License

GPL-3.0. See [LICENSE](LICENSE).
