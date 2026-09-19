# Tower defence

Enemies march along a path from an entrance to an exit; towers beside the path shoot them; the
player decides between waves what to build with the gold the last wave paid, and wins by being
alive after the last wave. The rules are the genre's own, written as small as they can be: a
map is a path of cells with a few building slots beside it, two kinds of tower are on offer,
and a wave runs to its end once started.

What a placement is worth is only known by running the wave. Range, rate, damage, speed, hit
points and the timing of arrivals all interact, and the same tower in the same slot is worth a
lot against a slow thick wave and nothing against a fast thin one that is past it before it
reloads. That is what keeps the environment out of PDDL, and it is the same shape as the flood
environment: a decision now, a simulated period, and only then the bill. Nothing is taken from
any published title, and the environment needs no dependency.

- **Import:** `from planiverse.environments.tower_defence.environment import TowerDefenceEnv`
- **Source:** [`planiverse/environments/tower_defence/environment.py`](../../planiverse/environments/tower_defence/environment.py)
- **Instances:** 100 maps, indices `0` to `99`, all drawn by the generator at recorded seeds
- **Generator:** `generate_instance(seed, waves=None, slots=None, ...)`; see [Generating maps](#generating-maps)

## Quickstart

```python
from planiverse.environments.tower_defence.environment import TowerDefenceEnv, TowerAction, START

env = TowerDefenceEnv()
env.set_index(0)
state, info = env.reset()          # info: map, waves, slots, gold, lives, generated
print(state)                       # wave 0 fought, 3 lives, 120 gold; nothing built

state, _ = env.step(TowerAction("arrow", 2))
state, lives_lost = env.step(START)
```

## The rules

1. Between waves the player may build any number of towers, one per action, each on a free slot
   and for its cost in gold, and then start the next wave.
2. A wave is `(count, hp, speed, spacing)`: `count` enemies with `hp` hit points enter `spacing`
   ticks apart and walk the path at `speed` hundredths of a cell a tick.
3. Each tick every tower that has reloaded fires at the enemy furthest along the path within its
   range. An arrow tower costs 60, reaches 1.6 cells, hits for 4 and reloads in 2 ticks; a cannon
   costs 140, reaches 2.8 cells, hits for 24 and reloads in 6.
4. A kill pays the map's bounty. An enemy that reaches the exit costs a life.
5. The goal is to have fought every wave with a life left. No lives is a dead end.

## State

`TowerState` holds the towers as `(slot, kind)` pairs, the gold, the lives, and how many waves
have been fought. Equality and hashing are over the four, so a position is the same position
wherever it was reached from; `depth` is bookkeeping. The path, the slots and the waves belong
to the instance.

`literals` names the towers, the gold to the nearest twenty, the lives and the waves fought:

```
tower(arrow, 2)
gold(80)
lives(3)
waves_fought(1)
```

## Actions

`build(kind, slot)` for each kind and slot, and `start`. All cost 1. A build that the gold or the
slot does not allow changes nothing and is not offered as a successor. `get_actions()` lists the
whole alphabet for the selected map and `TowerAction.parse` reads one back from its name.

## Maps

The hundred maps are the generator's own draws, embedded in the module as the plain data
`set_instance` takes with the seed each came from beside it, and the plan each was accepted on is
in `tests/data/tower_defence_solutions.json`. A map is a path that wanders up and down across a
twelve by eight grid, six to eight building slots beside it, three to five waves whose hit points
grow by half or more each wave, a starting purse of 100 to 140 gold, two to four lives and a
bounty of 4 to 6 a kill. The purse buys one or two towers at the start; the rest has to be
earned, so the slots at the bends, where a tower sees the path twice, are the ones that matter.

## Generating maps

`generate_instance` draws a map, selects it, and returns it as the dict `set_instance` takes
back:

```python
env = TowerDefenceEnv()
game = env.generate_instance(seed=7, waves=4)
print(env.witness, env.witness_expansions)     # the plan it was accepted on, and the search's cost
state, info = env.reset()                       # info["generated"] is True
```

| Option | Default | What it does |
|---|---|---|
| `waves` | 3 to 5 at random | waves to fight |
| `slots` | 6 to 8 at random | building slots beside the path |
| `search_limit` | 600 | expansions the acceptance search may spend per draw |
| `attempts` | 40 | draws before giving up with `GenerationError` |

A draw is kept only if every thoughtless plan loses it (starting every wave building nothing,
and filling the slots in order with arrows, or with cannons, whenever the gold allows) and a
best-first search over decisions, guided by the waves left to fight and the lives lost, finds a
way to win within `search_limit` expansions. The method is generate-and-test, which the
procedural content generation literature calls search-based PCG (Togelius, Yannakakis, Stanley
and Browne, 2011, https://doi.org/10.1109/TCIAIG.2011.2148116; Shaker, Togelius and Nelson,
*Procedural Content Generation in Games*, 2016, https://pcgbook.com/).

## Files

| File | Contents |
|---|---|
| [`environment.py`](../../planiverse/environments/tower_defence/environment.py) | `TowerAction`, `TowerState`, the wave (`run_wave`), `draw_map`, `TowerDefenceEnv`, `MAPS` |
| [`tests/test_tower_defence.py`](../../tests/test_tower_defence.py) | Tests |
| [`tests/data/tower_defence_solutions.json`](../../tests/data/tower_defence_solutions.json) | The plan each map was accepted on |
