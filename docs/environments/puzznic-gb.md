# Puzznic (Game Boy)

This environment plays Puzznic on the cartridge. It drives the Japanese Game Boy ROM inside
[PyBoy](https://github.com/Baekalfen/PyBoy) and reads the board out of the console's work RAM, so
the transition function is the game's own code rather than a reconstruction of its rules. States
are emulator save-states (i.e., the whole machine serialised to bytes), so search can branch by
rewinding the machine. What that buys in fidelity it pays for in speed, since one expansion is a
save-state load, a button press and a settle.

The sibling [`puzznic.py`](puzznic.md) re-implements the same game in Python over the same 128
rounds. Use that one for a dependency-free benchmark, and this one for the cartridge's own
behaviour.

- **Class:** `PuzznicGBEnv`
- **Import:** `from planiverse.environments.gameboy.puzznic_gb import PuzznicGBEnv, PuzznicGBAction`
- **Source:** [`planiverse/environments/gameboy/puzznic_gb.py`](../../planiverse/environments/gameboy/puzznic_gb.py)
- **Instances:** 128 rounds, indices `0`–`127`
- **Dependencies:** `pyboy` plus a `Puzznic (J).gb` ROM you supply; `pillow` for screenshots
- **Memory map:** [`puzznic-gb-memory-map.md`](puzznic-gb-memory-map.md)

## The ROM

The repo ships no ROM, because Puzznic is Taito's copyrighted work, so you supply your own legally
obtained dump and pass its path:

```python
env = PuzznicGBEnv("Puzznic (J).gb")
```

Every address below was read from one dump:

| | |
|---|---|
| File | `Puzznic (J).gb`, 65,536 bytes |
| MD5 | `9a777d82cd7a8913ba1aed2cc854fa50` |
| Cartridge | MBC1, 64 KiB, no cartridge RAM; all state is in work RAM |

The addresses are revision-specific, so the constructor hashes the file and raises a `UserWarning`
when it is not that dump. Pass `verify_rom=False` to silence it.

## Quickstart

```python
from planiverse.environments.gameboy.puzznic_gb import PuzznicGBEnv, PuzznicGBAction

env = PuzznicGBEnv("Puzznic (J).gb", render=False)   # render=True opens an SDL2 window
env.fix_index(0)                                     # choose the stage before reset
state, info = env.reset()

print(state)              # the playfield, trimmed to its bounding box:
# #######
# #..1..#
# #.c.2.#
# #1#2###
# #######

print(state.blocks_remaining, state.cursor)
for action, successor in env.successors(state):
    print(action, action.cost(), successor.blocks_remaining)
# left_for_15 1 6
# right_for_15 1 6

ticks = info["calibration"].press_ticks          # measured; do not hard-code it
trace = env.simulate([PuzznicGBAction(f"left,{ticks}"), PuzznicGBAction(f"a+right,{ticks}")])
print(env.is_goal(trace[-1]))
```

Note that the board above is illustrative. The stage layouts are in the ROM, but the block types
are drawn by a PRNG at load time, so what round 1 prints is not fixed.

Or from the command line:

```bash
python -m planiverse.environments.gameboy.puzznic_gb "Puzznic (J).gb" --stage 0
```

`reset()` boots the ROM, taps through the title screens until a stage is on the playfield, waits
for the board to accept input, calibrates, and snapshots.

## Rounds

`Puzznic (J)` has 128 rounds, and `fix_index(i)` selects one of them zero-based, so `fix_index(3)`
selects round 4. `reset()` reaches it the way a player would, by picking `PASSWORD` on the title
menu and typing the round's password.

```python
env.fix_index(49)
env.password_for(49)        # 'PASSWORD'
state, info = env.reset()
info["boot_route"], info["stage_index"]
# ('password', 49)
```

We read the passwords out of the cartridge rather than transcribing them from a guide. The table
is in the ROM at `$47FA`, as 128 ten-byte entries: eight bytes of password in the game's own text
encoding, then the round number, then a check byte. `read_passwords(romfile)` walks it until an
entry stops being a password whose round number follows the last one, which is both how the end is
found and how the parse checks itself. The text encoding runs letters from `$0A`, so `A` is `$0A`
and `Z` is `$23`, leaving `$00` to `$09` for the digits; `.` is `$24` in the ROM and `$8B` on
screen.

The title menu is `1PLAYER / 2PLAYERS / PASSWORD`, and the password screen is a 9×3 grid of
`A`–`Z` and `.` over a row of `NEXT / BACK / END`, with eight slots along the top. Selecting a
character fills the next slot and advances; `END` confirms and starts the round. Everything on
both screens that matters is a sprite:

| What | Where |
|---|---|
| Title-menu cursor | arrow sprite (tile `$AC`) at y = 88 / 104 / 120 |
| Entry cursor | the same arrow; cell = `(y - 80) / 16`, `(x - 16) / 16` |
| The eight slots | sprites at y = 48, x = 24, 40, 56, 72, 96, 112, 128, 144 |
| An empty slot | tile `$8C` |

`entry_cursor` reads where the cursor is and `entered_password` reads what has been typed. Every
cursor move is checked, and every character is checked against the slot it was meant to fill and
retried if it did not land. The screen does drop a press now and again, and a password one
character short is not an error: it is silently the wrong round.

A second route writes `$D003`, the stage index, through a hook on the loader at `0:0430`. It does
change the layout, and the two routes agree on the board, but it swaps the layout underneath a
game that still believes it is on round 1, so everything the cartridge derives from the round is
then wrong. We keep it only for a cartridge with no title menu, such as the synthetic test ROM,
which has a password table but no password screen. `info["boot_route"]` reports which of
`password`, `1player` or `tapped` was used.

## State

`PuzznicGBState` carries the entire emulator save-state (`gb_state`, the bytes from
`pyboy.save_state`) plus the board scraped from RAM. The save-state is what makes branching
possible, since applying an action loads the parent's bytes back into the emulator first, so
siblings expand from an identical machine. The cost is a state that is kilobytes rather than
bytes.

| Field | Source |
|---|---|
| `grid` | 12×10 tuple of cell type codes from `$DF00`, stride 20 per row, 2 bytes per cell |
| `blocks` | `(row, col, type, slot)` for every cell holding a block |
| `records` | live entries of the 6-byte record array at `$DD00` |
| `cursor` | `$D013` (row), `$D012` (column) |
| `total_blocks` | `$D018`: what the stage loaded with; never decremented |
| `blocks_remaining` | `$D019`: decremented once per block removed |
| `blocks_cleared` | `total_blocks - blocks_remaining` |
| `stage_types` | the block types present when the stage loaded |
| `stage_cleared` | sampled here, read by `is_goal` |
| `dead_end` | sampled here, read by `is_terminal` |

### Cell type codes

`grid` holds the game's own codes:

| Value | Meaning |
|---|---|
| `$00` | Empty; the only value that permits movement |
| `$01` | Transient: a block is clearing or in motion |
| `$02` | Solid ledge; blocks rest on it |
| `$03` | Outside the playfield |
| `$06` | Wall |
| `$08`–`$0F` | A block; its type is the value minus 7, so types 1–8 |

There are no per-stage width or height variables: every stage fills all 120 cells, and a small
stage is the same array with more `$03` around it, so `state.bounding_box()` derives the shape.

The stage data in ROM records only that a block occupies a cell, and the type is drawn by a PRNG
at load time, so expected types per stage cannot be hard-coded. The invariant that does hold is
that two blocks which look identical on screen carry identical type bytes.

### Literals

```
at(cursor, ROW, COL)
at(block-T, ROW, COL)
remaining(N)
all-blocks-matched(block-T)     # type T was in the stage and is now gone
goal-reached
terminal-state
```

Terrain is deliberately not in the literals, because it is static per stage and repeating the
walls in every state would swamp the atoms that do change; read `state.grid` for it. Depth is
absent for a different reason: with a step counter in the literals no successor could ever equal
its parent, and the self-loop filter in `successors` would be dead code.

Two states are equal when their grid and cursor match, which is the whole position, so search can
close over two ways of reaching the same board.

### Cross-checking

The grid and the record array reference each other, so the board can be read from either end, and
`state.is_consistent()` requires the grid scan, the record scan and `$D019` to agree. Note that
records are zeroed in place and slots are never compacted, so walking the record array until the
first `$00` type byte stops at the first hole and reports almost nothing; `decode_records`
iterates all `$D018` slots and skips the dead ones.

## Actions

`PuzznicGBAction` wraps a string of the form `"buttons,ticks"`, where buttons are `+`-joined and
ticks is how many frames to hold them.

| Action | Effect |
|---|---|
| `left` / `right` / `up` / `down` | Move the cursor one cell |
| `a+left` / `a+right` | Push the block under the cursor one cell sideways |

There is no `a+up` or `a+down`, because Puzznic slides blocks sideways and does not lift them. `B`
works as the modifier as well as `A`, and a direction on its own only walks the cursor off the
block; we checked both on the cartridge.

Cost is 1 for every action. `a` costs 0 in `action_cost_map` because it is a modifier that turns a
direction into a push rather than a move of its own.

Most of the branching is the cursor travelling to a block, which is worth knowing before pointing
a planner at this environment. Measured over the synthetic cartridge's stages:

| Stage | Branching factor | Successors that move a block |
|---|---|---|
| 2 blocks | 4.33 | 5.9% |
| 4 blocks | 4.06 | 3.3% |
| 7 blocks | 4.16 | 5.6% |

Over 90% of every expansion is the cursor travelling, because walking to a block is not a decision
but the overhead of making one. We leave that alone, since collapsing the walk into the push is a
search technique and search is the planner's side of the line. `cursor_path` is a breadth-first
search over the cells the cursor may occupy, exported for a planner that wants to build such
macro-actions; the actions this environment hands out are button presses.

### Calibration

How long to hold a button has two bounds, and neither is in the memory map. Too short and the
press is never sampled; too long and auto-repeat fires, so one action moves the cursor two cells
and the state the planner gets back is not the one its action described. The failure is quiet,
because `PuzznicGBAction("a+right,60")` looks like one push and reads like one action in a plan.
So `reset()` measures both bounds off the cartridge rather than trusting a constant:

```python
state, info = env.reset()
info["calibration"]
# Calibration(press_ticks=15, hold_window=(1, 30), push_scheme='modifier',
#             push_prefix='a', push_ticks=None, push_window=None)
```

`Puzznic (J)` repeats on frame 31, so any hold of 30 frames or fewer moves one cell and
`press_ticks` settles on 15. `measure_hold_window` presses a direction for 1, 2, 3… frames and
watches `$D012`/`$D013`, returning the closed range of holds that move the cursor exactly one
cell: the lower end is where presses start registering, and the upper end is one frame short of
auto-repeat. `press_ticks` is the middle of that range, far enough from either edge to survive a
frame of jitter in when the game samples input, at a cost of about 0.07 ms per spare frame.

Measured across the first eight stages of `Puzznic (J)`:

| Stage | Blocks | Cursor window | Push window |
|---|---|---|---|
| 0 | 6 | (1, 30) | not probeable |
| 1 | 10 | (1, 30) | not probeable |
| 2 | 8 | (1, 30) | not probeable |
| 3 | 8 | (1, 30) | not probeable |
| 4 | 9 | (1, 30) | (1, 30) |
| 5 | 7 | (1, 30) | not probeable |
| 6 | 15 | (2, 31) | not probeable |
| 7 | 8 | (1, 30) | (1, 30) |

Stage 6 shows why the middle of the window is the right choice rather than its lower edge, since
there a single-frame press is not sampled at all. Where the push window could be measured it
matches the cursor's, so on this cartridge one repeat routine serves both.

`measure_push_window` does the same for a held block, as a separate measurement because the two
need not agree: a cartridge is free to repeat a held block on its own schedule. Getting this wrong
is worse than getting the cursor wrong. Held past the repeat, a single `a+right` slides the block
two cells, and if the second cell puts it next to its own colour the pair clears, handing the
planner a board two blocks lighter than the action it applied ever asked for. Probing it needs a
block that can be slid two cells and still be found afterwards, so `push_probe_candidates` rejects
any block that would fall down a hole or land next to its own colour. On a cramped stage there is
no such block, `push_window` is `None`, and `push_ticks` falls back to `press_ticks`.

`probe_push_scheme` walks the cursor onto a real block with somewhere to go and tries each
candidate scheme:

| Scheme | Inputs |
|---|---|
| `modifier` | hold A and press a direction; one input |
| `grab` | press A to pick the block up, then a direction; two inputs |
| `direct` | a direction alone moves the block under the cursor |

Whichever scheme actually moves the block is the one the action model uses, so a cartridge that
works some other way is handled rather than silently mis-driven. On `Puzznic (J)` it comes out
`modifier`.

Calibration describes the game rather than the stage, so it runs once per environment and is
reused across resets, at a cost of about 0.15 s. `PuzznicGBEnv(rom, calibrate=False)` skips it and
falls back to `PRESS_TICKS` and the `modifier` scheme.

`button_actions` holds each action for its own window:

```python
['left,8', 'right,8', 'up,8', 'down,8', 'a+left,5', 'a+right,5']
```

### Waiting for the round to accept input

The stage loader fills work RAM before the round has finished announcing itself. For 210 frames on
`Puzznic (J)` the board is completely readable, with grid, records, counters and cursor all
correct and cross-checking, while every button is ignored, and nothing in RAM says so. A state
snapshotted in that window looks perfectly normal and answers no action, so a planner sees a stage
with no legal moves rather than an error.

`reset()` calls `wait_until_interactive`, which presses a direction from a snapshot at increasing
offsets until the cursor answers, then rewinds and replays only the waiting, so the cursor stays
where the loader put it. `info["intro_ticks"]` reports how long it took. It retries having pressed
START first, because START is the pause button once a round is running and the boot sequence taps
it while getting through the title screens.

### Applying an action

A push is not instantaneous. The block slides, blocks above it fall, a match clears through the
`$01` transient, and the fall can cascade into another match, so snapshotting straight after the
button press would capture the middle of that. `apply` presses the buttons and then runs `settle`:

```python
load_state(pyboy, state.gb_state)      # rewind to the parent
pyboy.button(button, hold)             # press
pyboy.tick(max(ticks) + 1)
settle(pyboy)                          # wait for the board to stop moving
```

The board counts as settled once no cell is mid-clear and the grid has been byte-identical for
`SETTLE_STABLE_TICKS` (4) frames, giving up after `SETTLE_MAX_TICKS` (600). Two things cut the
wait short: `$D019` reaching zero, because clearing the last block ends the stage and the
cartridge loads the next round over the top of it; and `$D018` changing, which means a new stage
has loaded. Both limits are configurable:

```python
env = PuzznicGBEnv(rom, settle_max_ticks=1200, settle_stable_ticks=8)
```

## Goal and terminal

- **Goal** (`is_goal`): `state.stage_cleared`, meaning the stage loaded with blocks, `$D019` is
  zero and the grid scan agrees.
- **Terminal** (`is_terminal`): `state.dead_end`, meaning some type has exactly one block left.
  Matching is pairwise, so a lone block can never be removed. Three blocks of a type are not a
  dead end, because Puzznic clears a group of three when they meet.

Both read flags captured on the passed state rather than live memory. Both are absorbing:
`successors` returns `[]` and `step`/`simulate` hand the state straight back, so an action applied
past a goal state cannot return a position from the next round.

There is no timer. Puzznic gives a time limit per stage and takes a life when it runs out, but the
timer address is not in the memory map, so a state that has run out of time is not terminal here.
Long plans are the case to watch.

## Rendering

`str(state)` draws the playfield, trimmed to its bounding box:

| Glyph | Cell |
|---|---|
| `#` | Wall |
| `=` | Ledge |
| `.` | Empty |
| `*` | Clearing |
| `1`–`8` | A block of that type |
| `c` | The cursor, on an empty cell |
| `¢` | The cursor, on top of something |

`env.render()` returns the console's own screen for every position `step` has played through, as
one PIL image per de-duplicated step, magnified four times by nearest neighbour so the frames
carry only the four shades the Game Boy drew. The text board above is printed alongside as a
caption, because a terminal cannot show a picture. Passing a target writes the frames instead, in
any format `render_trace` spells:

```python
env.render("play.gif")            # animated, one frame per step
env.render("play.png")            # a contact sheet
env.render("play-frames/")        # a directory of PNGs
```

`state.save(rom, path)` writes a PNG of a single state by booting a throwaway emulator to it; it
uses the null window, so it needs no display, but it does need Pillow.

`render_trace` supplies this environment's own cartridge, so the frames are real console
screenshots:

```python
import os
from planiverse.planners.width import IteratedBFWS
from planiverse.benchmark import measures

env = PuzznicGBEnv(romfile=os.environ["PLANIVERSE_PUZZNIC_ROM"])
env.fix_index(0)
env.reset()

result = IteratedBFWS(max_width=1000, progress=measures.puzznic_gb).solve(env)
trace = env.simulate(result.plan)

env.render_trace(trace, "puzznic_gb.gif")                                   # animated
env.render_trace(trace, "puzznic_gb.png", actions=result.plan, env=env,     # contact sheet
                 columns=4, max_states=12)
```

A cartridge frame is 640x576, so `max_states=` thins a long trace rather than tiling it in full.
See [docs/rendering.md](../rendering.md) for the other output formats.

## Notes and limits

- **No score.** The score lives in the status-panel tilemap shadow at `$D800` and the digit layout
  there is not worked out, so nothing reads it. `step` returns `blocks_cleared` in the reward
  slot.
- **No timer**; see [Goal and terminal](#goal-and-terminal).
- **The push window is not measurable on every stage.** It needs a block that can be slid two
  cells without falling or matching, and six of the first eight stages have none. There
  `push_ticks` is `None` and falls back to the cursor hold.
- **`successors` shares one emulator.** All expansion runs on `self.pyboy`, and correctness relies
  on each action reloading its parent's save-state first. Do not parallelise expansion over one
  environment.
- **`render=True` is for watching, not planning.** It opens an SDL2 window and slows expansion.
  The constructor keeps that flag as `self.render_window`, because `self.render` is the method
  that prints the history.
- **`load_state` advances a frame** after restoring, so it is not a byte-exact restore.

## Testing without the cartridge

[`tests/fake_puzznic_rom.py`](../../tests/fake_puzznic_rom.py) assembles a small homebrew Game Boy
cartridge, using [`tests/sm83.py`](../../tests/sm83.py), that puts the same facts at the same
addresses: a stage loader at `$0430` that reads `$D003`, a 12×10 grid at `$DF00`, records at
`$DD00`, counters at `$D018`/`$D019` and a cursor at `$D012`/`$D013`.

It is not a Puzznic clone: no gravity, no timer, no score, no cascades. A push that leaves two
same-typed blocks adjacent clears both after a three-frame `$01` transient. It does model cursor
auto-repeat, so `measure_hold_window` has a bound to find, and a 60-frame round intro, so
`wait_until_interactive` has a wait to discover.

The tests that need the real cartridge are opt-in:

```bash
PLANIVERSE_PUZZNIC_ROM="/path/to/Puzznic (J).gb" poetry run pytest tests/test_puzznic_gb.py
```

## Files

| Path | What |
|---|---|
| [`puzznic_gb.py`](../../planiverse/environments/gameboy/puzznic_gb.py) | `PuzznicGBEnv`, `PuzznicGBState`, `PuzznicGBAction`, the calibration and the RAM decoders |
| [`puzznic-gb-memory-map.md`](puzznic-gb-memory-map.md) | Every address the environment reads |
| [`tests/test_puzznic_gb.py`](../../tests/test_puzznic_gb.py) | Tests, against the synthetic cartridge and the real one |
| [`tests/fake_puzznic_rom.py`](../../tests/fake_puzznic_rom.py) | The synthetic cartridge |
| [`tests/sm83.py`](../../tests/sm83.py) | The assembler that builds it |
