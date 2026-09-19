# Fluid

Water pours from a spring into a cave of rock and earth, and the player digs channels, a cell at
a time and only so many, to bring enough of it to a basin before the spring runs dry, past a
drain that swallows whatever falls into it. Water is a cellular automaton in the falling-sand
family: each tick every water cell falls if it can, else slides diagonally, else moves
sideways, in a fixed scan order. That automaton is the transition function, and it is what
keeps the environment out of PDDL: a dig is one cell of earth removed, after which the water
runs for a period, and what reaches the basin depends on the whole cave and on everything dug
before. No add or delete list carries a flow.

The mechanic is the one the falling-sand games and the water-routing puzzles share. Nothing here
is taken from any published title, and the environment needs no dependency.

- **Import:** `from planiverse.environments.fluid.environment import FluidEnv`
- **Source:** [`planiverse/environments/fluid/environment.py`](../../planiverse/environments/fluid/environment.py)
- **Instances:** 100 caves, indices `0` to `99`, all drawn by the generator at recorded seeds
- **Generator:** `generate_instance(seed, need=None, digs=None, ...)`; see [Generating caves](#generating-caves)

## Quickstart

```python
from planiverse.environments.fluid.environment import FluidEnv, FluidAction, WAIT

env = FluidEnv()
env.set_index(0)
state, info = env.reset()          # info: cave, need, supply, digs, generated
print(state)                       # the cave: # rock, . earth, ~ water, S spring, T basin, X drain

state, taken = env.step(WAIT)      # let the spring run a period
for action, child in env.successors(state):
    print(action, child.filled, "in the basin")
```

## The rules

1. Each tick the spring gives one unit of water to the air below it, while it has any left, and
   then every water cell, scanned from the bottom row up, falls into air below it, else slides
   into air diagonally below, else moves into air beside it. The side tried first alternates
   with the tick, so a pool on a floor keeps shifting about, as the genre's water does; what
   matters to the puzzle is where it can fall next.
2. Water that moves onto the basin is taken and counted; water that moves onto the drain is
   lost.
3. An action is a dig of one earth cell that has water against it, or a wait; either way the
   water then runs for `PERIOD` (12) ticks. A dig spends one of the digs allowed.
4. The goal is the basin taking at least `need` units. A position is a dead end when the water
   left in the spring and afloat together cannot cover what the basin still needs, or when
   `MAX_TICKS` (240) have passed.

Digs are allowed only where the water is, so the player follows the flow rather than tunnelling
anywhere: that is what keeps the branching to a handful of cells and makes the order of digs the
puzzle.

## State

`FluidState` holds the cave as a tuple of row strings, the water the spring has left, the water
the basin has taken, the digs left and the tick. Equality and hashing are over all five, so a
position is the same position wherever it was reached from; `depth` is bookkeeping and `need` is
carried for the benchmark's measure.

`literals` names every earth, air and water cell and the counters:

```
cell(4, 2, water)
cell(5, 3, earth)
filled(3)
supply(9)
digs_left(5)
```

## Actions

`dig(x, y)` for each earth cell with water against it, and `wait`, each costing 1. A dig that the
budget or the water does not allow changes nothing and is not offered as a successor.
`get_actions()` lists the digs open in the current state and `FluidAction.parse` reads one back
from its name.

## Caves

The hundred caves are the generator's own draws, embedded in the module as the plain data
`set_instance` takes with the seed each came from beside it, and the plan each was accepted on
is in `tests/data/fluid_solutions.json`. A cave is sixteen by ten: earth inside a rock border
with a few pockets of air and veins of rock, the spring near the top left, the basin near the
bottom right with air over it, and the drain on the floor between them. The basin needs 8 to 12
units, the spring holds 3 to 6 more than that, and 6 to 9 digs are allowed.

## Generating caves

`generate_instance` draws a cave, selects it, and returns it as the dict `set_instance` takes
back:

```python
env = FluidEnv()
cave = env.generate_instance(seed=7, need=10)
print(env.witness, env.witness_expansions)     # the plan it was accepted on, and the search's cost
state, info = env.reset()                       # info["generated"] is True
```

| Option | Default | What it does |
|---|---|---|
| `need` | 8 to 12 at random | units the basin must take |
| `digs` | 6 to 9 at random | digs allowed |
| `search_limit` | 200 | expansions the acceptance search may spend per draw |
| `attempts` | 40 | draws before giving up with `GenerationError` |

A draw is kept only if waiting without digging fails and a best-first search over digs, guided
by what the basin still needs and then by how far the nearest water is from it, finds a plan
within `search_limit` expansions. The method is generate-and-test, which the procedural content
generation literature calls search-based PCG (Togelius, Yannakakis, Stanley and Browne, 2011,
https://doi.org/10.1109/TCIAIG.2011.2148116; Shaker, Togelius and Nelson, *Procedural Content
Generation in Games*, 2016, https://pcgbook.com/).

## Files

| File | Contents |
|---|---|
| [`environment.py`](../../planiverse/environments/fluid/environment.py) | `FluidAction`, `FluidState`, the automaton (`tick`, `run`), `diggable`, `draw_cave`, `FluidEnv`, `CAVES` |
| [`tests/test_fluid.py`](../../tests/test_fluid.py) | Tests |
| [`tests/data/fluid_solutions.json`](../../tests/data/fluid_solutions.json) | The plan each cave was accepted on |
