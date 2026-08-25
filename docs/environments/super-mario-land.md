# Super Mario Land

Planning directly against a Game Boy. This environment drives the real Super Mario Land ROM inside
the [PyBoy](https://github.com/Baekalfen/PyBoy) emulator, and reads game facts straight out of the
console's RAM. States are emulator save-states, so search can branch by rewinding the machine.

- **Class:** `SuperMarioEnv`
- **Import:** `from planiverse.problems.retro_games.super_mario_bros_gb import SuperMarioEnv, SuperMarioAction`
- **Source:** [`planiverse/problems/retro_games/super_mario_bros_gb.py`](../../planiverse/problems/retro_games/super_mario_bros_gb.py)
- **Dependencies:** `pyboy` + a `SuperMarioLand.gb` ROM you supply (`pillow` for screenshots)
- **Planner:** [`planiverse/planners/super_mario_planner_gb.py`](../../planiverse/planners/super_mario_planner_gb.py)

## The ROM

**Not included, and cannot be.** Super Mario Land is Nintendo's copyrighted work; the repo ships no
ROM and none will be added. Supply your own legally-obtained dump and pass its path:

```python
env = SuperMarioEnv("/path/to/SuperMarioLand.gb")
```

## Quickstart

```python
import os
from planiverse.problems.retro_games.super_mario_bros_gb import SuperMarioEnv, SuperMarioAction

env = SuperMarioEnv("SuperMarioLand.gb", render=False)   # render=True opens an SDL2 window
state, info = env.reset()

print(state)
# <SuperMarioState(depth=0, mario_position=Position(x=..., y=...), progress=..., enemies=0)>

for action, successor in env.successors(state):
    print(action, action.cost(), successor.level_progress)

# Replay a plan and dump screenshots
plan = [SuperMarioAction('right,3')] * 20
trace = env.simulate(plan)
for i, s in enumerate(trace):
    s.save("SuperMarioLand.gb", f"/tmp/frame_{i}.png")
```

`reset()` boots the ROM through PyBoy's game wrapper, starts the game at the selected level, and sets
lives to 0 so a death ends the run instead of restarting it:

```python
self.pyboy = create_pyboy(self.romfile, self.render)
self.game  = self.pyboy.game_wrapper
self.game.game_area_mapping(self.game.mapping_compressed, 0)
self.game.start_game(world_level=self.world_level)
self.game.set_lives_left(0)   # avoid replays
```

## Levels

`fix_index(i)` selects one of the 12 world/level pairs — Super Mario Land has 4 worlds of 3 levels,
both 1-indexed — and `reset()` starts the game there:

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| World, level | 1-1 | 1-2 | 1-3 | 2-1 | 2-2 | 2-3 | 3-1 | 3-2 | 3-3 | 4-1 | 4-2 | 4-3 |

Without `fix_index`, `world_level` stays `None` and the game boots at its default 1-1.

## State representation

`SuperMarioState` carries the **entire emulator save-state** (`gb_state`, the bytes from
`pyboy.save_state`) plus a set of facts scraped from RAM. The save-state is what makes branching
possible: applying an action loads the parent's bytes back into the emulator first, so siblings are
expanded from an identical machine.

## The memory map

Every address comes from a reverse-engineering pass over
`Super Mario Land (World) (Rev 1).gb` (MD5 `b259feb41811c7e4e1dc200167985c84`), and the
constructor hashes the file and warns when it is a different dump. Pass `verify_rom=False`
to silence it.

That map was derived **behaviourally** — recording all 8 KiB of work RAM once per frame
while driving scripted input, then correlating against the input phases — rather than from a
disassembly. Static tracing reached only ~2,000 instructions, because the game dispatches
through jump tables. So the confidence varies field by field, and the table says which is
which rather than presenting them all as equal.

### Mario — `$C200`

| Field | Address | Confidence |
|---|---|---|
| `mario_position.y` | `$C201` | **verified** |
| `mario_position.x` | `$C202` | **verified** |
| `animation_frame` | `$C203` | good |
| `mario_facing` | `$C205` — `$20` is left | **verified** |
| `jump_phase` | `$C207` | good |
| `on_ground` / `airborne` | `$C20A` — `$01` grounded | **verified** |
| `mario_speed` | `$C20C` — a magnitude | good |
| `mario_direction` | `$C20D` — `$00` still, `$10` right, `$20` left | **verified** |
| `moving` | `$C20F` | good |

**`$C201`/`$C202` are screen coordinates.** X stops at `$51` once Mario reaches the scroll
trigger and the camera takes over, so it says where he is on the display and never how far
through the level he is. Use `level_progress`.

**There is no velocity vector.** `$C20C` is a speed magnitude and `$C20D` a direction *code*;
the map found no vertical velocity byte at all — `jump_phase` and `on_ground` are what
describe vertical motion. These two used to be read as the x and y of a `velocity`
namedtuple, which made the `(supermario velocity X Y)` literal describe nothing: a "y
velocity" of 16 meant "travelling right".

### Objects — `$D100`

Ten slots of `$10` bytes; `$FF` in byte +0 means empty. A slot goes live when an object
scrolls into range and reverts when it leaves or dies, so `state.enemies` is what is on
screen, not everything in the level.

| Field | Offset | Confidence |
|---|---|---|
| status (`$FF` empty) | +0 | **verified** |
| `type` | +1 | moderate |
| `y` | +2 | **verified** |
| `x` | +3 | **verified** |
| `animation` | +4 | moderate |

**Caveat carried straight from the map:** only slot 0 was ever occupied during the recording,
by one ground-walking enemy in 1-1. The base, stride, empty marker and X/Y are solid;
everything from +4 on is a single sample and will not generalise to flying, shelled or boss
objects.

`state.touching_enemy` overlaps Mario's box with each object's — a proximity test, not a flag
the game sets, because the map found no damage byte.

### HUD and camera

| Field | Address | Confidence |
|---|---|---|
| `timeleft` | BCD at `$DA02`(high digit)/`$DA01`(low two) | **verified** |
| `lives_left` | `$DA15` | moderate — matched the screen, never seen change |
| `world` | `$DA16` | **unverified** — no level transition was observed |
| `camera_x` / `camera_y` | hardware `SCX` `$FF43` / `SCY` `$FF42` | **verified** |

The camera has **no WRAM mirror**: the map searched every byte for one and found nothing, so
the register is the only place to read it.

### Two things the map does not cover

- **`level_progress`** is PyBoy's own wrapper formula — `$C0AB` × 16 + scroll + Mario's X.
  The map looked for a 16-bit level X across all of WRAM and found none, so this is the only
  number that keeps rising across screens. Note it takes SCX at **scanline 16** rather than
  from `$FF43`: the HUD splits the screen, and the register holds whatever the split left
  behind, while scanline 16 is below it and so is the playfield's scroll.
- **`level_complete` (`$DFE8`) and `game_over` (`$C0A4`)** are inherited guesses. `$C000–$C09F`
  is the OAM shadow, so both sit just past it in territory the map does not cover, and
  neither was ever watched changing. Treat `is_goal` as unconfirmed.

Literals:

```
(supermario position X Y)
(supermario motion SPEED DIRECTION)
(supermario grounded 0|1)
(progress N)
(depth N)
(coins N)
(livesleft N)
(enemy TYPE X Y)        # one per live object slot
```

Two states are equal when their literals match **and** their `timeleft` differs by less than 5. The
time window is deliberate: without it, Mario standing in the same spot at different moments would be
distinct states and search would never close; with it, a state that merely burns a few frames in
place collapses into its parent and gets pruned by `successors`.

`state.save(gamerom, file)` writes a PNG screenshot of the state by booting a throwaway emulator,
loading the save-state, and upscaling the frame 4×.

## Actions

`SuperMarioAction` wraps a string of the form `"buttons,ticks"`, where buttons are `+`-joined and
ticks is how many frames to hold them. The 16 actions in `action_list`:

| Buttons | Ticks | Count |
|---|---|---|
| `a+left`, `a+right`, `b+left`, `b+right` | 5, 10, 15 | 12 |
| `nop`, `left`, `right`, `down` | 3 | 4 |

So `'a+right,10'` means "hold A and Right for 10 ticks" — a jump to the right. `a` jumps, `b` runs
(or fires), `nop` idles.

Applying an action loads the state's save-state, presses the buttons for their tick count, advances
`max(ticks) + 1` frames, and snapshots the result at `depth + 1`.

**Cost** (`action.cost()`) is `sum(cost of each button) × ticks`, using:

| Button | Cost |
|---|---|
| `a`, `b` | 2 |
| `left`, `right`, `down` | 1 |
| `nop` | 0 |

Jumping is charged double per frame, so a planner minimising cost prefers running to hopping.

## Goal and terminal

- **Goal** (`is_goal`) — `state.level_complete`, sampled from `0xDFE8` when the state was built. The
  source flags this address as a TODO; treat it as unconfirmed.
- **Terminal** (`is_terminal`) — `state.game_over` (the death music track was requested) or
  `state.collision`.

Both read flags captured on the passed `state`. They used to query the emulator's *current* memory,
which described whichever state was applied last rather than the state being asked about.

## Planner

`SuperMarioPlanner` (in [`super_mario_planner_gb.py`](../../planiverse/planners/super_mario_planner_gb.py))
re-implements Robin Baumgarten's A* Mario agent on top of the generic `TreeSearchPlanner`. It
overrides the environment's goal test with a sliding window and supplies its own heuristic and cost:

```python
def __is_goal__(self, state):
    return state.level_progress >= self.root.level_progress + 175

def __hueristic_fn__(self, state):
    distance_delta = state.level_progress - self.root.level_progress
    damage_penalty = state.mario_damage() * 100000
    if distance_delta <= 0: return 100000 + damage_penalty
    return 1.2 * (-1 * distance_delta + damage_penalty)

def __cost_fn__(self, state_trace, action_trace):
    combined_action = 2 if '+' in action_trace[-1].action else 1
    return 1.0*abs(state_trace[0].timeleft - state_trace[-1].timeleft) + state_trace[0].depth + combined_action
```

The idea is iterative replanning: plan until Mario advances 175 pixels, execute, replan from there.
Only the first half is implemented — `search()` computes a plan and then falls off the end of the
function, returning `None` instead of the plan, and the replanning loop is not written. It is a
starting point, not a working agent.

## Known quirks

- **Contact is not death.** `touching_enemy` overlaps boxes from the object array; whether contact
  kills depends on power-up state, and the map could not confirm that byte (`$C210` never changed —
  Mario stayed small for the whole recording). So `is_terminal` is the death music alone, and
  `mario_damage()` is offered as a heuristic penalty rather than a prune.
- **The object array is one sample deep.** See the caveat above: the fields past +3 came from a
  single walker in 1-1.
- **`successors` shares one emulator.** All expansion runs on `self.pyboy`; correctness relies on
  each action reloading its parent's save-state first. Don't parallelise expansion over one env.
- **`render=True` is for watching, not planning.** It opens an SDL2 window and slows expansion.
- **The goal address is unconfirmed** — see [Goal and terminal](#goal-and-terminal).

## Fixed

- **`$C20C`/`$C20D` were being read as a velocity vector.** They are a speed *magnitude* and a
  direction *code*; there is no vertical velocity byte in the map at all. The literal
  `(supermario velocity 6 16)` was reporting "y velocity 16" for Mario walking right on flat
  ground. It is now `(supermario motion 6 right)`, and `jump_phase`/`on_ground` cover vertical
  motion.
- **The planner's goal window could never be reached.** `__is_goal__` asked for
  `mario_position.x >= root.x + 175`, but that X is a screen coordinate which saturates at `$51`
  (81) while Mario starts near `$32` (50) — so it needed 225 from a byte that stops at 81, and the
  search could only ever run out. Both it and the heuristic now measure `level_progress`.
- **Enemies come from the object array at `$D100`** rather than counting sprites whose tile id is
  `145`, which recognised exactly one kind of thing. `state.enemies` carries slot, type and
  position for every live object.
- **The timer is decoded by the map's formula.** It agreed with the old `_bcd_to_dec` reading, but
  is now explicit about `$DA02` holding one digit and `$DA01` two.
- **`fix_index` now reaches `reset()`.** It validated the index and set `self.world_level`, but
  `reset()` never read it, so the game always booted at its default start and level selection did
  nothing. Its map was also `product(range(0,4), repeat=2)` — 16 pairs, 0-indexed — while Super Mario
  Land has 12 levels and PyBoy expects them 1-indexed.
- **`is_goal`/`is_terminal` read the state they are given** rather than live emulator memory — see
  [Goal and terminal](#goal-and-terminal).
- **`(depth N)` and `(coins N)` are separate literals.** A missing comma in the predicate list fused
  them into a single `(depth N)(coins N)` string. Harmless for equality, confusing for anything
  parsing literals.
- **`from tkinter import Image` removed.** It was dead, and raised `ImportError` on a Python build
  without tkinter.

## Files

| Path | What |
|---|---|
| [`super_mario_bros_gb.py`](../../planiverse/problems/retro_games/super_mario_bros_gb.py) | `SuperMarioEnv`, `SuperMarioState`, `SuperMarioAction` |
| [`super_mario_planner_gb.py`](../../planiverse/planners/super_mario_planner_gb.py) | `TreeSearchPlanner`, `SuperMarioPlanner` |
