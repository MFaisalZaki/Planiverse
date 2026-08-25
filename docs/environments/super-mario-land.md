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
# <SuperMarioState(depth=0, mario_position=Position(x=..., y=...), enemies_killed=0)>

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

Facts are read from these addresses ([RAM map reference](https://datacrystal.romhacking.net/wiki/Super_Mario_Land:RAM_map)):

| Field | Source |
|---|---|
| `mario_position` | `0xC202` (x), `0xC201` (y) |
| `mario_velocity` | `0xC20C` (x), `0xC20D` (y) |
| `timeleft` | BCD at `0xDA01`–`0xDA02` |
| `level_progress` | `0xC0AB` (level block) × 16 + scroll offset + mario x |
| `lives_left` | BCD at `0xDA15` |
| `coins` | summed from the background tilemap |
| `score` | summed from the background tilemap |
| `enemies_killed` | count of sprites with tile identifier `145` |
| `level_complete` | `0xDFE8 == 0x01` — sampled here, read by `is_goal` |
| `game_over` | `0xC0A4 == 0x39`, the death music track — read by `is_terminal` |

Literals:

```
(supermario position X Y)
(supermario velocity VX VY)
(progress N)
(depth N)
(coins N)
(livesleft N)
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
    return state.mario_position.x >= self.root.mario_position.x + 175

def __hueristic_fn__(self, state):
    distance_delta = state.mario_position.x - self.root.mario_position.x
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

- **`collision` is hard-coded to `False`.** The Mario-dead-jump-timer read (`0xC0AC`) is commented
  out, so `mario_damage()` always returns 0 and the heuristic's damage penalty never fires. Deaths
  are only caught via the music-track check.
- **Enemy detection is partial.** `enemies_killed` counts sprite tile identifier `145` only; the
  source notes that identifying the full set of enemy tile IDs needs more reverse engineering.
- **`successors` shares one emulator.** All expansion runs on `self.pyboy`; correctness relies on
  each action reloading its parent's save-state first. Don't parallelise expansion over one env.
- **`render=True` is for watching, not planning.** It opens an SDL2 window and slows expansion.
- **The goal address is unconfirmed** — see [Goal and terminal](#goal-and-terminal).

## Fixed

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
