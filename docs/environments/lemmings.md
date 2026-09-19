# Lemmings

A crowd of walkers is let into a cave one at a time, walks without being told, and has to be
steered to the exit with a few skills handed out at the right moments, so that enough of them
are saved before the time runs out. Each tick every lemming acts on its own: a walker walks,
climbs a step of one, turns at a wall of two and falls off an edge; a faller dies if it falls
too far; a digger digs down through earth, a basher tunnels forward through it, a builder lays
a stair of bricks up and forward, and a blocker stands and turns the others back. Rock stops
every tool.

Between decisions the crowd runs for a fixed number of ticks, so what a skill is worth depends
on where everyone will be by then, on the terrain the earlier tools have already changed, and
on the order the lemmings were released in. That is the whole puzzle, and it is what keeps the
environment out of PDDL: a crowd walking a cave is a simulation, not an action model. The
mechanics are the genre's, at the resolution of a cell rather than a pixel, with the
public-domain [Lix](https://www.lixgame.com/) as the reference for what the skills do; the
levels are this environment's own, and it needs no dependency.

- **Import:** `from planiverse.environments.lemmings.environment import LemmingsEnv`
- **Source:** [`planiverse/environments/lemmings/environment.py`](../../planiverse/environments/lemmings/environment.py)
- **Instances:** 100 levels, indices `0` to `99`, all drawn by the generator at recorded seeds
- **Generator:** `generate_instance(seed, lemmings=None, quota=None, ...)`; see [Generating levels](#generating-levels)

## Quickstart

```python
from planiverse.environments.lemmings.environment import LemmingsEnv, LemmingsAction, WAIT

env = LemmingsEnv()
env.set_index(0)
state, info = env.reset()          # info: level, lemmings, quota, skills, generated

state, _ = env.step(WAIT)          # let the first lemmings in
print(state)                       # the cave: # rock, . earth, E entrance, X exit, > < walkers
for action, child in env.successors(state):
    print(action, child.saved, "saved")
```

## The rules

1. A lemming enters at the entrance every `RELEASE` (6) ticks until the level's crowd is out,
   facing right, and falls to the ground.
2. Each tick, a walker steps forward if the cell ahead is clear, climbs if it is one cell high,
   turns if it is two or more, and falls if there is nothing under it. A fall of more than
   `FATAL_FALL` (6) cells kills. A walker on the exit is saved.
3. A digger removes the earth under it, one cell a tick, until it meets rock or air. A basher
   removes the earth ahead of it two cells high and walks into the gap, until it meets rock or
   air. A builder lays a brick ahead of and above itself and steps up onto it, up to `BRICKS` (8)
   times or until something is in the way. A blocker stands where it is and turns any walker
   that reaches it. A tool used up turns its lemming back into a walker.
4. A decision is `assign(skill, lemming)` for a walker and a skill the level has left, or
   `wait`; either way the crowd then runs for `DECISION` (8) ticks.
5. The goal is `quota` lemmings saved. A position is a dead end when the saved, the ones
   still out and the ones still to enter together cannot reach the quota, or when `MAX_TICKS`
   (320) have passed.

## State

`LemmingsState` holds the terrain, every lemming out in the cave as `(x, y, direction, state,
counter)`, the counts released, saved and lost, the skills left and the tick. Equality and
hashing are over all of them, so a position is the same position wherever it was reached from;
`depth` is bookkeeping and `quota` is carried for the benchmark's measure.

`literals` names every earth cell, every lemming, the skills left and the counts:

```
earth(12, 7)
lemming(2, 9, 5, walker)
basher_left(1)
saved(3)
lost(1)
released(6)
```

## Actions

`assign(skill, k)` for each skill the level has left and each lemming `k` that is walking,
and `wait`, each costing 1. An assignment the skills or the lemming do not allow changes
nothing and is not offered as a successor. `get_actions()` lists the decisions open in a state
and `LemmingsAction.parse` reads one back from its name.

## Levels

The hundred levels are the generator's own draws, embedded in the module as the plain data
`set_instance` takes with the seed each came from beside it, and the plan each was accepted on
is in `tests/data/lemmings_solutions.json`. A level is thirty-two by sixteen: earth platforms at
different heights over earth and rock, gaps and walls between them, the entrance above the
first and the exit on the last, a crowd of five to eight with a quota one to three fewer, and
up to one blocker, two diggers, two bashers and two builders.

## Generating levels

`generate_instance` draws a level, selects it, and returns it as the dict `set_instance` takes
back:

```python
env = LemmingsEnv()
level = env.generate_instance(seed=7, lemmings=6, quota=4)
print(env.witness, env.witness_expansions)     # the plan it was accepted on, and the search's cost
state, info = env.reset()                       # info["generated"] is True
```

| Option | Default | What it does |
|---|---|---|
| `lemmings` | 5 to 8 at random | the crowd |
| `quota` | 1 to 3 fewer than the crowd | how many must be saved |
| `search_limit` | 400 | expansions the acceptance search may spend per draw |
| `attempts` | 40 | draws before giving up with `GenerationError` |

A draw is kept only if letting the crowd walk unaided saves fewer than the quota and a
best-first search over decisions, guided by the lemmings saved and then by how near the rest
are to the exit, finds a plan within `search_limit` expansions. The method is
generate-and-test, which the procedural content generation literature calls search-based PCG
(Togelius, Yannakakis, Stanley and Browne, 2011, https://doi.org/10.1109/TCIAIG.2011.2148116;
Shaker, Togelius and Nelson, *Procedural Content Generation in Games*, 2016,
https://pcgbook.com/).

## Files

| File | Contents |
|---|---|
| [`environment.py`](../../planiverse/environments/lemmings/environment.py) | `LemmingsAction`, `LemmingsState`, the crowd (`run_ticks`), `draw_level`, `LemmingsEnv`, `LEVELS` |
| [`tests/test_lemmings.py`](../../tests/test_lemmings.py) | Tests |
| [`tests/data/lemmings_solutions.json`](../../tests/data/lemmings_solutions.json) | The plan each level was accepted on |
