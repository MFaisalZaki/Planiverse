# Artillery

A gun stands on the left of a hilly field, targets are dug in behind the hills, a wind blows the
same way all game, and the player has a few shells to destroy every target. Each shot is an
angle and a power from a small set; the shell flies under gravity and the wind until it meets
the ground or leaves the field, and where it lands it blows a crater out of the terrain,
destroys any target within its radius, and drops a target whose ground was blown away, which
kills it if the fall is long. The genre is Scorched Earth's and Worms's, and the model here is
the textbook one.

What keeps it out of PDDL is that a shot's effect is a flight through a wind field onto a
surface that earlier shots have reshaped. A crater is not a fact to add, and a lob's landing
point moves with every hill it clears; the only way to know where a shell lands is to
integrate it. Nothing is taken from any published title. The ballistics, the crater and the
fields are this environment's own, and it needs no dependency.

- **Import:** `from planiverse.environments.artillery.environment import ArtilleryEnv`
- **Source:** [`planiverse/environments/artillery/environment.py`](../../planiverse/environments/artillery/environment.py)
- **Instances:** 100 fields, indices `0` to `99`, all drawn by the generator at recorded seeds
- **Generator:** `generate_instance(seed, targets=None, shells=None, ...)`; see [Generating fields](#generating-fields)

## Quickstart

```python
from planiverse.environments.artillery.environment import ArtilleryEnv, ArtilleryAction

env = ArtilleryEnv()
env.set_index(0)
state, info = env.reset()          # info: field, targets, shells, wind, generated
print(state)                       # the field drawn in text: # ground, T targets, G the gun

for action, child in env.successors(state):
    print(action, child.targets_left, "targets left")

state, destroyed = env.step(ArtilleryAction(45, 40))
```

## The rules

1. A shell leaves the gun at `power` units per second, `angle` degrees above the horizontal,
   and is integrated in steps of a twentieth of a second under gravity (20 units per second per
   second downward) and the field's wind (a constant horizontal acceleration). It lands where it
   first meets the ground on its way down, or is lost if it leaves the field.
2. Where it lands it blows a crater of radius 3: every column within that radius is lowered to
   the crater's floor.
3. A target within the radius of the impact is destroyed. A target whose ground was lowered drops
   to the new surface, and dies if the drop is 4 or more.
4. The goal is every target destroyed. A position with targets left and no shells left is a
   dead end.

The gun is never hit, since no shot of the alphabet lands on it, and the wind does not change
during a game.

## State

`ArtilleryState` holds the terrain as a tuple of eighty column heights, the targets as
`(column, height)` pairs and the shells left. Equality and hashing are over the three, so a
position is the same position wherever it was reached from; `depth` is bookkeeping. The wind
belongs to the instance rather than the state, since it never changes.

`literals` names every column's height, every target and the counts:

```
height(37, 12)          column 37 stands 12 high
target(52, 9)
shells_left(3)
targets_left(2)
```

## Actions

`fire(angle, power)` for `angle` in 15, 25, 35, 45, 55, 65 and 75 degrees and `power` in 30, 40
and 50 units per second: twenty-one shots, each costing 1, and a shot that changes nothing is
not offered as a successor. `get_actions()` lists them and `ArtilleryAction.parse` reads one
back from its name.

## Fields

The hundred fields are the generator's own draws, embedded in the module as the plain data
`set_instance` takes with the seed each came from beside it, and the plan each was accepted on
is in `tests/data/artillery_solutions.json`. A field's terrain is a sum of three hills of
different wavelengths over a random walk, clamped between 2 and 28 units; its two or three
targets sit in hollows, columns lower than the ridge between them and the gun, so that a flat
shot is blocked and the player has to lob or dig; its wind is drawn between minus 6 and 6; and
the shells are two more than the targets.

## Generating fields

`generate_instance` draws a field, selects it, and returns it as the dict `set_instance` takes
back:

```python
env = ArtilleryEnv()
field = env.generate_instance(seed=7, targets=3)
print(env.witness, env.witness_expansions)     # the plan it was accepted on, and the search's cost
state, info = env.reset()                       # info["generated"] is True
```

| Option | Default | What it does |
|---|---|---|
| `targets` | 2 or 3 at random | targets dug in behind the hills |
| `shells` | `targets + 2` | shells the player gets |
| `min_plan_length` | 2 | the shortest plan must be at least this long, so a field one shell clears is thrown back |
| `search_limit` | 300 | expansions the acceptance search may spend per draw |
| `attempts` | 60 | draws before giving up with `GenerationError` |

Each draw is searched breadth-first over shots and kept only when a plan is found within
`search_limit` expansions and is at least `min_plan_length` shells long. The method is
generate-and-test, which the procedural content generation literature calls search-based PCG
(Togelius, Yannakakis, Stanley and Browne, 2011, https://doi.org/10.1109/TCIAIG.2011.2148116;
Shaker, Togelius and Nelson, *Procedural Content Generation in Games*, 2016,
https://pcgbook.com/); the terrain is the usual sum of sines over a random walk.

## Files

| File | Contents |
|---|---|
| [`environment.py`](../../planiverse/environments/artillery/environment.py) | `ArtilleryAction`, `ArtilleryState`, the flight (`flight`, `fire`), `draw_field`, `ArtilleryEnv`, `FIELDS` |
| [`tests/test_artillery.py`](../../tests/test_artillery.py) | Tests |
| [`tests/data/artillery_solutions.json`](../../tests/data/artillery_solutions.json) | The plan each field was accepted on |
