# Platformer

A side-scrolling run-and-jump environment in pure Python, with physics that are chosen and
written down rather than reverse-engineered. No emulator, no ROM, no dependencies beyond the
standard library.

- **Class:** `PlatformerGame`
- **Import:** `from planiverse.environments.platformer import PlatformerGame`
- **Source:** [`planiverse/environments/platformer.py`](../../planiverse/environments/platformer.py)
- **Instances:** 8 levels, indices `0`–`7`, plus any you supply yourself
- **Dependencies:** none
- **Counterpart:** [`SuperMarioEnv`](super-mario-land.md) plays the real Game Boy cartridge

## What this is, and what it is not

This is the dependency-free counterpart to `super_mario_land`, and it makes a **weaker claim**
than the other two pairs in this library.

[`puzznic`](puzznic.md) and [`flipull`](flipull.md) are twins of their cartridges: their rules
were derived from the real hardware and reproduce it — exactly in Puzznic's case, partly in
Flipull's. **This is not a twin of Super Mario Land.** Nothing here was read off that
cartridge. It is a platformer of the same shape — run, jump, gaps, patrolling enemies, a flag
at the end — with physics picked to be simple, integral and stated.

Modelling a physics platformer faithfully is a far larger job than modelling a turn-based
puzzle, and a half-reverse-engineered one would be worse than none: it would look like a
prediction of the cartridge without being one. So this does not try, and says so here rather
than leaving you to find out.

## The physics

Positions are in units, `TILE` (8) units to a tile, the way the Game Boy itself counts. Mario
is one tile square. Each tick, in this order:

1. **Horizontal.** `left`/`right` accelerate towards a target speed — `WALK` (2), or `RUN` (4)
   while `b` is held — by `ACCEL` (1) a tick, and `FRICTION` slows you when nothing is held.
   Speed carries: what you are travelling at when you leave a ledge is what you cross the gap
   at. Running into a wall takes your speed away.
2. **Jump.** Pressing `a` while standing sets `vy = JUMP_SPEED` (−12). Holding it keeps the
   rise; releasing while still rising cuts the climb to `JUMP_CUT` (−4).
3. **Gravity.** `vy += GRAVITY` (2), capped at `MAX_FALL` (12).
4. **Collision.** Resolved one axis at a time — horizontal, then vertical — so a corner never
   lets you through.

The measured consequences, which is what you actually design levels against:

| | |
|---|---|
| Full jump | 3.75 tiles up |
| Short hop (`a,2` then release) | 2.25 tiles up |
| Widest gap clearable | 6 tiles |
| Runway needed to reach `RUN` | 2 tiles |
| Tallest step up | 3 tiles |

Momentum is why the levels are levels. Without it a run-up is free, every gap is either always
or never jumpable, and a level collapses into "hold run and jump" — which is exactly what the
first draft of this environment did, until momentum was added.

## Death, and what counts as a stomp

Falling out of the level kills. So does touching a hazard, and so does meeting an enemy from
the side. Landing on one from above kills the *enemy* and bounces you, which makes a jump an
attack as well as a way across.

"From above" is judged on where Mario was **before** the vertical move, not where he ended up.
At `MAX_FALL` he crosses the whole of an enemy's upper half inside a single tick, so a test on
the landing position alone would never see him above it and would kill him on something he
plainly landed on.

## Actions

The vocabulary mirrors `super_mario_land`'s `button,ticks` actions, minus `down` — there is no
ducking in this model. 21 actions: six button combinations held for 2, 6 or 12 ticks, plus
`nop`, `left` and `right` for 4.

```
a+right,2   a+left,2   b+right,2   b+left,2   a+b+right,2   a+b+left,2
a+right,6   ...                                             a+b+left,6
a+right,12  ...                                             a+b+left,12
nop,4       right,4    left,4
```

The short hold has to be genuinely short or the jump cut never fires: `vy` climbs past
`JUMP_CUT` three ticks into a jump, so releasing later than that changes nothing. A first draft
used holds of `(4, 8, 12)` and all three `a` actions produced an identical arc, which made two
thirds of the action set dead weight and a hop impossible.

`PlatformerAction.cost()` charges 2 per tick for `a` and `b` and 1 for a direction, so a plan
can be scored by effort rather than by length.

## Quickstart

```python
from planiverse.environments.platformer import PlatformerGame

env = PlatformerGame()
env.fix_index(3)
state, info = env.reset()

print(state)
print(info)     # {'level': 3, 'width': 38, 'enemies': 3, 'goal': (36, 5)}

state, gained = env.step("b+right,12")     # returns tiles of ground gained
```

## Levels

`fix_index(i)` selects level `i`. The shipped levels were **generated and then measured**, the
same way Flipull's stages were: random terrain, explored by a planner, and kept only if the
flag can actually be reached — and ranked by how much search that took, so the set is a ramp
rather than eight variations on one board.

`MEASURED_EXPANSIONS` records what each level cost BFWS(w=2) when it was accepted, in the same
order:

| Level | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Expansions | 4 | 7 | 7 | 66 | 385 | 468 | 482 | 1302 |

That is data, not a promise — change a physics constant and the numbers move. It is there so
the ramp is written down where it can be checked, and the tests re-derive a route through every
level, so one that stops being finishable fails the suite instead of quietly wasting a
planner's budget. Several candidates were dropped at exactly that gate: tightening the stomp
rule made three previously-solvable levels unreachable, and they were cut rather than shipped.

Level strings use this alphabet:

| Char | Meaning |
|---|---|
| `#` | Solid ground |
| (space) | Air |
| `^` | Hazard — fatal to touch |
| `E` | An enemy's starting tile |
| `M` | Mario's starting tile |
| `G` | The flag |

`M`, `E` and `G` say where things begin, not what the terrain is, so they are stripped out of
the tile map during parsing. Enemies patrol horizontally and turn at a wall or at the edge of
whatever they are standing on, so one placed in mid-air just oscillates on the spot — the
tests check that none of the shipped levels does that.

You can supply your own:

```python
env = PlatformerGame(levels=["""
        M    E        G
        ####   #########
"""])
env.fix_index(0)
```

## State representation

`PlatformerState` holds the terrain, Mario's position and velocity, whether he is on the
ground, and where the living enemies are. Equality and hashing are over everything except
depth.

Enemy positions are carried in the state rather than derived from a tick counter. Keeping a
clock and computing them would be tidier, but then two identical configurations reached at
different times would compare unequal, search would never close anything, and the state space
would be infinite for no reason.

`literals` is a `frozenset` of strings:

```
at(mario, 12, 5)       Mario's tile
progress(12)           his column, which is the natural progress measure
speed(4)               horizontal velocity
falling(1)  grounded(0)
at(enemy, 19, 5)       each living enemy
enemies(2)
dead()                 only in a dead state
```

## Goals and dead ends

`is_goal` is Mario's box overlapping the flag tile — an overlap rather than an exact tile
match, because he moves up to `MAX_FALL` units in a tick and a test on tile coordinates alone
would let a fast fall drop straight past the flag and count as a miss.

`is_terminal` is **exact about death**. [`SuperMarioEnv`](super-mario-land.md) cannot tell you
whether Mario died on contact: it has a proximity test over the object array, and whether
contact is fatal depends on a power-up byte the memory map never confirmed, so it deliberately
reports only the music track changing. Here death is defined, so a planner prunes the moment it
happens rather than playing on into a position that no longer exists.

It is **not** a test for whether the level is still winnable, and nothing here claims to be.
Mario can be alive in a pit he cannot jump out of, and no environment in this library detects
that — deciding it in general means solving the level. Dead states are pruned; stuck ones cost
the planner its budget, exactly as they would on the cartridge.

There is no timer and no score. The cartridge has both; this does not model them.

## Planning with it

```python
from planiverse.environments.platformer import PlatformerGame
from planiverse.planners.width import BFWSSearch, Budget

env = PlatformerGame()
env.fix_index(7)
_, info = env.reset()
result = BFWSSearch(width=2, progress=lambda s: info["width"] - s.tile_x).solve(
    env, Budget(max_expansions=200000, max_seconds=120))
print(result.status, len(result.plan))
```

`progress` on Mario's column is what makes this tractable, and it is worth being clear that it
is doing most of the work: a level whose only demand is "go right" is close to trivial for
BFWS. The shipped levels were selected for costing more than that — towers to climb, gaps that
need speed, enemies in the lane — but a run-and-jump level with a distance-to-goal heuristic is
an easier problem than Flipull, and the expansion counts show it.
