# Super Mario Land (Game Boy)

This environment plays Super Mario Land on the cartridge. It drives the real ROM inside the
[PyBoy](https://github.com/Baekalfen/PyBoy) emulator and reads game facts out of the console's
RAM. States are emulator save-states (i.e., the whole machine serialised to bytes), so search can
branch by rewinding the machine.

- **Class:** `SuperMarioLandGBEnv`
- **Import:** `from planiverse.environments.gameboy.super_mario_land_gb import SuperMarioLandGBEnv, SuperMarioLandGBAction`
- **Source:** [`planiverse/environments/gameboy/super_mario_land_gb.py`](../../planiverse/environments/gameboy/super_mario_land_gb.py)
- **Instances:** 12 world/level pairs, indices `0`–`11`
- **Dependencies:** `pyboy` plus a `SuperMarioLand.gb` ROM you supply; `pillow` for screenshots
- **Planner:** [`planiverse/planners/super_mario_planner_gb.py`](../../planiverse/planners/super_mario_planner_gb.py)
- **Counterpart:** [`SuperMarioLandGame`](super-mario-land.md) is the dependency-free Python model with cartridge-fitted physics

No planner in the library has solved instance `0` inside a 30-minute, 100,000-expansion budget;
every run so far has ended in a timeout. The emulator settles roughly 5 expansions a second, so
half an hour buys about ten thousand of them, which is not many for a side-scroller whose goal is
off the right-hand edge of the level. The rendering itself works, and the [Python
counterpart](super-mario-land.md) does carry a solved instance.

## A solved instance

There is not one to show. No planner in the library has solved instance `0` of this
environment inside a 30-minute, 100,000-expansion budget: every run so far has ended in a
timeout. The emulator settles roughly **5 expansions a second**, so half an hour buys about
ten thousand of them, which is not many for a side-scroller whose goal is off the right-hand
edge of the level.

The rendering itself works , [`SuperMarioLandGBEnv`](super-mario-land-gb.md) states carry
`save(rom, path)` like every other cartridge environment, so any trace you do produce renders
as real console screenshots. Its pure-Python counterpart has a rendered plan:
[Super Mario Land](super-mario-land.md).

```python
import os
from planiverse.environments.gameboy.super_mario_land_gb import SuperMarioLandGBEnv

env = SuperMarioLandGBEnv(romfile=os.environ["PLANIVERSE_SUPER_MARIO_LAND_ROM"])
env.set_index(0)
env.reset()

# ... once you have a plan from somewhere:
env.render_trace(env.simulate(plan), "super_mario_land_gb.gif")
```

See [docs/rendering.md](../rendering.md) for the other output formats.

## The ROM

The repo ships no ROM, because Super Mario Land is Nintendo's copyrighted work, so you supply your
own legally obtained dump and pass its path:

```python
env = SuperMarioLandGBEnv("/path/to/SuperMarioLand.gb")
```

Every address comes from `Super Mario Land (World) (Rev 1).gb`, MD5
`b259feb41811c7e4e1dc200167985c84`. The constructor hashes the file and warns when it is a
different dump; pass `verify_rom=False` to silence it.

## Quickstart

```python
from planiverse.environments.gameboy.super_mario_land_gb import SuperMarioLandGBEnv, SuperMarioLandGBAction

env = SuperMarioLandGBEnv("SuperMarioLand.gb", render=False)   # render=True opens an SDL2 window
state, info = env.reset()

print(state)
# <SuperMarioLandGBState(depth=0, mario_position=Position(x=..., y=...), progress=..., enemies=0)>

for action, successor in env.successors(state):
    print(action, action.cost(), successor.level_progress)

plan = [SuperMarioLandGBAction('right,3')] * 20
trace = env.simulate(plan)
for i, s in enumerate(trace):
    s.save("SuperMarioLand.gb", f"/tmp/frame_{i}.png")
```

`reset()` boots the ROM through PyBoy's game wrapper, starts the game at the selected level, and
sets lives to 0 so a death ends the run instead of restarting it:

```python
self.pyboy = create_pyboy(self.romfile, self.render)
self.game  = self.pyboy.game_wrapper
self.game.game_area_mapping(self.game.mapping_compressed, 0)
self.game.start_game(world_level=self.world_level)
self.game.set_lives_left(0)
```

## Levels

`set_index(i)` selects one of the 12 world/level pairs. Super Mario Land has 4 worlds of 3 levels,
both 1-indexed on the console, while `set_index` takes a zero-based index.

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| World, level | 1-1 | 1-2 | 1-3 | 2-1 | 2-2 | 2-3 | 3-1 | 3-2 | 3-3 | 4-1 | 4-2 | 4-3 |

Without `set_index`, `world_level` stays `None` and the game boots at its default 1-1.

## State

`SuperMarioLandGBState` carries the entire emulator save-state (`gb_state`, the bytes from
`pyboy.save_state`) plus a set of facts scraped from RAM. The save-state is what makes branching
possible, since applying an action loads the parent's bytes back into the emulator first, so
siblings expand from an identical machine.

We derived the memory map behaviourally, by recording all 8 KiB of work RAM once per frame while
driving scripted input and then correlating against the input phases, rather than from a
disassembly. Static tracing reached only about 2,000 instructions, because the game dispatches
through jump tables, so the confidence varies field by field and the notes below say where it is
thin.

### Mario (`$C200`)

| Field | Address |
|---|---|
| `mario_position.y` | `$C201` |
| `mario_position.x` | `$C202` |
| `animation_frame` | `$C203` |
| `mario_facing` | `$C205`; `$20` is left |
| `jump_phase` | `$C207` |
| `on_ground` / `airborne` | `$C20A`; `$01` grounded |
| `mario_speed` | `$C20C`; a magnitude |
| `mario_direction` | `$C20D`; `$00` still, `$10` right, `$20` left |
| `moving` | `$C20F` |

`$C201`/`$C202` should not be confused with a position in the level: they are screen coordinates.
X stops at `$51` once Mario reaches the scroll trigger and the camera takes over, so it says where
he is on the display and never how far through the level he is. Use `level_progress` for that.

There is no velocity vector. `$C20C` is a speed magnitude and `$C20D` a direction *code*, and the
map found no vertical velocity byte at all, so `jump_phase` and `on_ground` are what describe
vertical motion.

### Objects (`$D100`)

Ten slots of `$10` bytes; `$FF` in byte +0 means empty. A slot goes live when an object scrolls
into range and reverts when it leaves or dies, so `state.enemies` is what is on screen rather than
everything in the level.

| Field | Offset |
|---|---|
| status (`$FF` empty) | +0 |
| `type` | +1 |
| `y` | +2 |
| `x` | +3 |
| `animation` | +4 |

Note that only slot 0 was ever occupied during the recording the map came from, by one
ground-walking enemy in 1-1. The base, stride, empty marker and X/Y are solid; the fields from +4
on rest on a single sample and will not generalise to flying, shelled or boss objects.

`state.touching_enemy` overlaps Mario's box with each object's. It is a proximity test rather than
a flag the game sets, because the map found no damage byte.

### HUD and camera

| Field | Address |
|---|---|
| `timeleft` | BCD at `$DA02` (high digit) / `$DA01` (low two) |
| `lives_left` | `$DA15` |
| `world` / `level` | `$FFB4`; high nibble world, low nibble level. The byte PyBoy's own wrapper reads |
| `camera_x` / `camera_y` | hardware `SCX` `$FF43` / `SCY` `$FF42` |

The camera has no WRAM mirror, so the register is the only place to read it.

`level_progress` is PyBoy's own wrapper formula: `$C0AB` × 16 + scroll + Mario's X. There is no
16-bit level X anywhere in WRAM, so this is the only number that keeps rising across screens. It
takes SCX at scanline 16 rather than from `$FF43`, because the HUD splits the screen and the
register holds whatever the split left behind.

`level_complete` (`$DFE8`) and `game_over` (`$C0A4`) are inherited guesses. Both sit just past the
OAM shadow at `$C000–$C09F`, in territory the memory map does not cover, and we have not watched
either change. Treat `is_goal` as unconfirmed.

### Literals

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

Two states are equal when their literals match *and* their `timeleft` differs by less than 5. The
time window is deliberate. Without it, Mario standing in the same spot at different moments would
be distinct states and search would never close; with it, a state that merely burns a few frames
in place collapses into its parent and gets pruned by `successors`.

## Actions

`SuperMarioLandGBAction` wraps a string of the form `"buttons,ticks"`, where buttons are
`+`-joined and ticks is how many frames to hold them. The 16 actions in `action_list`:

| Buttons | Ticks | Count |
|---|---|---|
| `a+left`, `a+right`, `b+left`, `b+right` | 5, 10, 15 | 12 |
| `nop`, `left`, `right`, `down` | 3 | 4 |

So `'a+right,10'` means "hold A and Right for 10 ticks", which is a jump to the right. `a` jumps,
`b` runs or fires, and `nop` idles.

Applying an action loads the state's save-state, presses the buttons for their tick count,
advances `max(ticks) + 1` frames, and snapshots the result at `depth + 1`.

Cost (`action.cost()`) is `sum(cost of each button) × ticks`:

| Button | Cost |
|---|---|
| `a`, `b` | 2 |
| `left`, `right`, `down` | 1 |
| `nop` | 0 |

Jumping is charged double per frame, so a planner minimising cost prefers running to hopping.

## Goal and terminal

- **Goal** (`is_goal`): `state.level_complete`, sampled from `$DFE8` when the state was built.
  Unconfirmed; see [HUD and camera](#hud-and-camera).
- **Terminal** (`is_terminal`): `state.game_over` (the death music track was requested) or
  `state.collision`.

Both read flags captured on the passed state rather than live memory.

Contact should not be confused with death. `touching_enemy` overlaps boxes from the object array,
and whether contact kills depends on power-up state, which the memory map could not confirm, since
`$C210` never changed and Mario stayed small for the whole recording. So `is_terminal` is the
death music alone, and `mario_damage()` is offered as a heuristic penalty rather than as a prune.

## Planner

`SuperMarioPlanner`, in
[`super_mario_planner_gb.py`](../../planiverse/planners/super_mario_planner_gb.py), re-implements
Robin Baumgarten's A* Mario agent on top of the generic `TreeSearchPlanner`. It overrides the
environment's goal test with a sliding window and supplies its own heuristic and cost:

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

The design is iterative replanning: plan until Mario advances 175 pixels, execute, then replan
from there. Only the first half is implemented. `search()` computes a plan and then falls off the
end of the function, returning `None` instead of the plan, and the replanning loop is not written,
so this is a starting point rather than a working agent.

## Rendering

`state.save(gamerom, file)` writes a PNG screenshot of the state by booting a throwaway emulator,
loading the save-state, and upscaling the frame four times. It needs Pillow.

See [docs/rendering.md](../rendering.md) for the other output formats.

## Notes and limits

- **The goal address is unconfirmed**; see [Goal and terminal](#goal-and-terminal).
- **Contact is not death**; see the same section.
- **The object array is one sample deep.** The fields past +3 came from a single walker in 1-1.
- **`successors` shares one emulator.** All expansion runs on `self.pyboy`, and correctness relies
  on each action reloading its parent's save-state first. Do not parallelise expansion over one
  environment.
- **`render=True` is for watching, not planning.** It opens an SDL2 window and slows expansion.

## Files

| Path | What |
|---|---|
| [`super_mario_land_gb.py`](../../planiverse/environments/gameboy/super_mario_land_gb.py) | `SuperMarioLandGBEnv`, `SuperMarioLandGBState`, `SuperMarioLandGBAction` |
| [`super_mario_planner_gb.py`](../../planiverse/planners/super_mario_planner_gb.py) | `TreeSearchPlanner`, `SuperMarioPlanner` |
| [`tests/test_super_mario_land_gb.py`](../../tests/test_super_mario_land_gb.py) | Tests |
