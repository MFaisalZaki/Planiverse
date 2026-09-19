# Game Boy (PyBoy)

One environment for any Game Boy cartridge. It runs the cartridge under
[PyBoy](https://github.com/Baekalfen/PyBoy), reads it through PyBoy's *game wrappers*, and
turns what the wrapper reads into a planning problem: a save state is a state, the wrapper's
measurements and the background map are its literals, and the wrapper's own verdict
(`stage_cleared()`, `level_solved()`, `room_solved()`) or a threshold on a measurement is
the goal. Nothing here knows where a particular game keeps its counters; that is the
wrapper's business, and it stays in PyBoy.

- **Class:** `GameBoyEnv`
- **Import:** `from planiverse.environments.emulated.game_boy import GameBoyEnv`
- **Source:** [`planiverse/environments/emulated/game_boy.py`](../../planiverse/environments/emulated/game_boy.py)
- **Instances:** one per stage, level or room the cartridge's wrapper can start at; see
  [Profiles](#profiles)
- **Generator:** `generate_instance(seed, warmup=(0, 30), index=None, goal=None, solvable=False, search_limit=2000, attempts=20)`;
  see [Generating instances](#generating-instances)
- **Dependencies:** `pyboy`, and a cartridge you supply

## Supplying a cartridge

Commercial cartridges are copyrighted, so none ships with this repository and none is
downloaded. The environment reads the path from `rom=` or from the `PLANIVERSE_GB_ROM`
variable, and raises `FileNotFoundError` naming the variable when it has neither:

```bash
export PLANIVERSE_GB_ROM=/path/to/puzznic.gb
```

```python
from planiverse.environments import make

env = make("game_boy", index=7)         # the wrapper's eighth stage
env = make("game_boy", seed=3)          # a generated one
state, info = env.reset()
info["wrapper"]                         # 'GameWrapperPuzznic'
```

Which wrapper a cartridge gets is PyBoy's decision, made from the cartridge title. PyPI's
PyBoy ships wrappers for Super Mario Land, Tetris, Kirby's Dream Land, Pokémon Pinball and
Pokémon Gen 1; [MFaisalZaki/PyBoy](https://github.com/MFaisalZaki/PyBoy) adds Puzznic,
Flipull, Amazing Tater and Adventures of Lolo (the four games this library also has in pure
Python) and Super Mario Bros. Deluxe:

```bash
pip install git+https://github.com/MFaisalZaki/PyBoy
```

A cartridge no wrapper claims still works, through PyBoy's generic wrapper: the state is
the background map plus whatever bytes of memory `watch=` names, and the goal is whatever
`goal=` says about them. That is the path the test suite exercises, on a cartridge it
assembles itself (`tests/counter_rom.py`, an original program that counts button presses).

## Profiles

`PROFILES` in the module says, per wrapper class, what the environment knows about the game.
A profile is data, not code: how to reach instance *i* (the `start_game` argument), which
wrapper attributes to read into every state, what the goal and the dead end are, which
buttons are worth offering, and how many frames a press lasts.

| Wrapper | Instances | `set_index(i)` starts | Fields read | Goal | Dead end | Actions |
|---|---|---|---|---|---|---|
| `GameWrapperPuzznic` | 128 | `stage=i` | stage, blocks remaining and total, cursor | `stage_cleared()` | `game_over()` | ←→↑↓, A+←, A+→ |
| `GameWrapperFlipull` | 32 | `stage=i+1` | stage, blocks remaining, clear target, time, row, held block | `stage_cleared()` | `game_over()` | ↑, ↓, throw |
| `GameWrapperAmazingTater` | 105 | `level=i` | level, taters left and home, active tater | `level_solved()` | `game_over()` | ←→↑↓, select |
| `GameWrapperAdventuresOfLolo` | 163 | `room=i` | room, hearts left, magic shots, position | `room_solved()` | `game_over()` (a lost life) | ←→↑↓, A |
| `GameWrapperSuperMarioLand` | 12 | `world_level=(w, l)` | progress, lives, coins, time, world | 250 columns of progress | `game_over()` or a lost life | →, A+→, ←, A+←, A, nop |
| `GameWrapperSuperMarioBrosDeluxe` | 1 | | as Super Mario Land | as Super Mario Land | as Super Mario Land | as Super Mario Land |
| `GameWrapperTetris` | 16 | `timer_div=16·i` | score, level, lines, next piece | one line cleared | `game_over()` | ←, →, ↓, A, B, nop |
| `GameWrapperKirbyDreamLand` | 1 | | score, health, lives | 100 points | `game_over()` or a lost life | →, A+→, ←, A+←, B, A, nop |
| any other | 1 | 120 frames of boot | every number the wrapper exposes, plus `watch=` | `survive` 20 actions | `game_over()` | every button, nop |

`hold` is 6, 5, 5, 20, 8, 8, 4 and 8 frames respectively; a wrapper that declares its own
`press_ticks` wins. The Game Boy's timer seeds the randomness of the games that have any, so
every bundled instance starts with the DIV register at zero, and Tetris, whose only variation
is the piece sequence, takes its instance number as the seed. The four wrappers from the fork
expose `settle()`, which runs the game to its next stable frame after a press; the others get
`settle=` frames, or one.

`actions=`, `hold=`, `goal=` and `terminal=` on the constructor override the profile, and
`profile=` replaces it, for a wrapper the module has no entry for.

## State representation

A `GBState` carries a save state (PyBoy's, zlib-compressed; about 200 KB uncompressed and
mostly zeros), the wrapper's `game_area()` as a tuple of rows of tile identifiers, and the
fields the profile named. Its literals are

- `tile(row, col, id)` for every cell of the game area,
- `field(name, value)` for every field (a tuple-valued field is flattened to `name.0`,
  `name.1`, ...),
- `steps(n)` only under a survival goal, since only then is the count part of the problem,
- `goal-reached` and `terminal-state`.

Two states with the same tiles and fields are the same state. That is the modelling
decision, and it is what lets search close: two histories that leave the board in the same
position are one node, whatever their save states say about the frame counter or the timer.
What it costs is exactness for games with randomness. Two equal states can have different
timer registers and so different futures, but the environment is deterministic given the
timer seed and the inputs, so what a planner sees is one fixed game, and the plan it
finds replays.

`state_identity` is `snapshot` in the registry, for this reason: the state is an emulator
image, told apart from others by what the wrapper reads off it.

## Goals and dead ends

A goal or dead-end spec is a dict with one of:

| Spec | Holds when |
|---|---|
| `{"method": "stage_cleared"}` | the wrapper method returns true |
| `{"attribute": "lines", "delta": 1}` | the field has risen by `delta` since the instance's initial state |
| `{"attribute": "coins", "at_least": 5}` | `at_least`, `at_most` or `equals` on the field |
| `{"memory": 0xC000, "at_least": 3}` | the same comparisons on one byte of memory |
| `{"attribute": "lives_left", "drop": true}` | (dead end) the field has fallen below its value at the initial state |
| `{"survive": 20}` | `survive` actions have been taken without a dead end |

A dead-end spec naming both a `method` and an `attribute` holds when either does, and the
attribute is read as a drop, which is how "game over, or a lost life" is written. Relative
specs (`delta`, `drop`) are measured from the instance's own initial state, after any
opening the generator played, not from the stage's first frame.

The progress measure the width planners take (`planiverse.benchmark.measures.emulated`)
is whatever the profile says is left: blocks remaining, hearts left, taters out, or the
negated progress or score; for a survival goal it is the negated step count.

## Actions

An action is a button name (`a`, `b`, `start`, `select`, `up`, `down`, `left`, `right`),
several joined with `+` (`a+right`), or `nop`. `throw` means the Flipull wrapper's
`throw_button`, and `switch` means select. Pressing holds the buttons for `hold` frames,
releases them, and settles. Actions can be given to `simulate` and `step` as strings.

`successors` tries every action in the vocabulary from the state's save state and drops the
ones that change nothing (a `nop` on a still menu, a wall bumped into). One expansion is one
save-state load and `hold` plus settle frames of emulation per action; PyBoy's compiled
build runs a few thousand frames a second, so an expansion costs a few milliseconds.

## Instances and generating instances

An instance is plain data:

```python
{"index": 7, "start": {"stage": 7}, "timer_div": 0, "warmup": [], "goal": {...}, "terminal": {...}}
```

`set_index(i)` builds the one the profile maps `i` to; `set_instance` takes a dict of this
shape; `generate_instance` draws one:

```python
instance = env.generate_instance(seed=3)             # any stage, a random timer seed, an opening
instance = env.generate_instance(seed=3, index=7)    # stage 7, but a fresh timer seed and opening
instance = env.generate_instance(seed=3, warmup=(10, 20), goal={"attribute": "coins", "delta": 3})
instance = env.generate_instance(seed=3, solvable=True, search_limit=500)
```

- **What varies.** The stage, level or room (unless `index` pins it); the timer seed, for
  the games whose randomness reads it (Tetris keeps its seed as the instance number); and an
  *opening*, a run of `warmup` (a range) actions from the vocabulary, played from the stage's
  first frame, so the problem starts somewhere in the level rather than at its door. A draw
  that is already won or lost after its opening is redrawn.
- **Determinism.** The emulator is deterministic given the cartridge, the timer seed and
  the inputs, and everything random in the draw comes from `random.Random(seed)`, so the
  same seed gives the same instance, and the same instance gives the same initial state
  byte for byte, on any machine with the same cartridge.
- **Checking.** With `solvable=True` the draw is searched, best-first under the profile's
  progress measure, for up to `search_limit` expansions, and only a draw the search solves is
  handed out; the plan is left in `env.witness` and its cost in `env.witness_expansions`.
  This is off by default, because an emulator expansion is milliseconds and a search of a
  real level is minutes.

The opening is random play from the game's own start, the no-op and random starts the Atari
evaluation protocol uses to vary a deterministic game (Mnih et al., 2015,
https://doi.org/10.1038/nature14236), and the instance is one of the game's own stages or save
states. The instance records the `seed` it came from as well, so a file of generated instances is
also a file of seeds.

## Rendering

`render()` returns the console's own frames, one PIL image per position `step` played
through; `render(target)` and `render_trace` write a trace as a GIF or a directory of
PNGs, captioned from the states' text (the tile grid and the fields).

## Caveats

- The profiles for the fork's wrappers (Puzznic, Flipull, Amazing Tater, Adventures of
  Lolo, Super Mario Bros. Deluxe) were written against those wrappers' interfaces and are
  not exercised by the test suite, which has no cartridge to run them on. The generic path,
  and everything the profiles share with it, is.
- Each stage needs a fresh emulator, because a wrapper's `start_game` runs once per boot.
  `reset` pays for a boot only when the instance's start changes, and returns to a cached
  save state otherwise.
- `game_area()` is the wrapper's view of the screen; for a scrolling game it moves with the
  camera, so a state's tiles are relative to the camera rather than to the level.
