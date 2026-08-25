# Puzznic (Game Boy)

Puzznic played on the cartridge. This environment drives the real Japanese Game Boy ROM inside
[PyBoy](https://github.com/Baekalfen/PyBoy) and reads the board straight out of the console's work
RAM, so the transition function is the game's own code rather than a reconstruction of its rules.
States are emulator save-states, so search can branch by rewinding the machine.

The sibling [`puzznic.py`](puzznic.md) re-implements the same game in pure Python with 50
hand-written levels. Use that one if you want a dependency-free benchmark; use this one if you want
the cartridge's actual behaviour.

Every address this environment reads is catalogued in the
[memory map](puzznic-gb-memory-map.md), including how each one was established and how much of it is
verified against live RAM rather than read off a disassembly.

- **Class:** `PuzznicGBEnv`
- **Import:** `from planiverse.problems.retro_games.puzznic_gb import PuzznicGBEnv, PuzznicGBPush`
- **Source:** [`planiverse/problems/retro_games/puzznic_gb.py`](../../planiverse/problems/retro_games/puzznic_gb.py)
- **Dependencies:** `pyboy` + a `Puzznic (J).gb` ROM you supply (`pillow` for screenshots)

## The ROM

**Not included, and cannot be.** Puzznic is Taito's copyrighted work; the repo ships no ROM and none
will be added. Supply your own legally-obtained dump and pass its path:

```python
env = PuzznicGBEnv("Puzznic (J).gb")
```

Every address below was read from one specific dump:

| | |
|---|---|
| File | `Puzznic (J).gb`, 65,536 bytes |
| MD5 | `9a777d82cd7a8913ba1aed2cc854fa50` |
| Cartridge | MBC1, 64 KiB, **no** cartridge RAM — all state is in work RAM |

Because the addresses are revision-specific, the constructor hashes the file and raises a
`UserWarning` when it is not that dump. Pass `verify_rom=False` to silence it.

## Quickstart

```python
from planiverse.problems.retro_games.puzznic_gb import PuzznicGBEnv, PuzznicGBAction

env = PuzznicGBEnv("Puzznic (J).gb", render=False)   # render=True opens an SDL2 window
env.fix_index(0)                                     # choose the stage *before* reset
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
# push_5_3_right 3 2      <- walk two cells, then push: three button presses
# push_5_6_left 2 2

trace = env.simulate([PuzznicGBPush(5, 3, "right"), PuzznicGBPush(5, 4, "right")])
print(env.is_goal(trace[-1]))
```

(The board above is illustrative — the stage layouts are in the ROM, and the block *types* are drawn
by a PRNG at load time, so what Round 1 prints is not fixed.)

`reset()` boots the ROM, taps through the title screens until a stage is on the playfield, waits for
the board to settle, and snapshots it.

## Stages

`fix_index(i)` selects the stage. The index is the raw value the cartridge's loader indexes its
pointer table with (`$D003`, one byte), so the accepted range is `0`–`255`. **How many of those
entries are real stages was never established** — an index past the end of the table will build
whatever bytes follow it. Index `0` is the first entry, which is Round 1 on a normal boot.

Without `fix_index`, `stage_index` stays `None` and the game boots wherever it normally would.

### How stage selection works

Naively you would write the index into `$D003` between frames. That does not work: a title screen can
reset the index and call the loader within the same frame, and the loader's read wins. Instead the
environment registers a PyBoy hook on the loader's entry point:

```python
self.pyboy.hook_register(0, STAGE_LOADER_ENTRY, _force_stage, (self.pyboy, self.stage_index))
```

`STAGE_LOADER_ENTRY` is `0:0430`. The hook fires the instant the loader is entered — after whatever
the menu wrote, before the loader reads `$D003` at `$045A` — so the write always lands in the
window that matters. The hook stays registered, so every subsequent stage load gets the same index
too, which keeps a cleared stage from rolling on into the next round.

## State representation

`PuzznicGBState` carries the **entire emulator save-state** (`gb_state`, the bytes from
`pyboy.save_state`) plus the board scraped from RAM. The save-state is what makes branching possible:
applying an action loads the parent's bytes back into the emulator first, so siblings expand from an
identical machine.

| Field | Source |
|---|---|
| `grid` | 12×10 tuple of cell type codes from `$DF00`, stride 20 per row, 2 bytes per cell |
| `blocks` | `(row, col, type, slot)` for every cell holding a block |
| `records` | live entries of the 6-byte record array at `$DD00` |
| `cursor` | `$D013` (row), `$D012` (column) |
| `total_blocks` | `$D018` — what the stage loaded with; never decremented |
| `blocks_remaining` | `$D019` — decremented once per block removed |
| `blocks_cleared` | `total_blocks - blocks_remaining` |
| `stage_types` | the block types present when the stage loaded |
| `stage_cleared` | sampled here, read by `is_goal` |
| `dead_end` | sampled here, read by `is_terminal` |

### Cell type codes

`grid` holds the game's own codes, not an abstraction of them:

| Value | Meaning |
|---|---|
| `$00` | Empty — the only value that permits movement |
| `$01` | Transient: a block is clearing or in motion |
| `$02` | Solid ledge — blocks rest on it |
| `$03` | Outside the playfield |
| `$06` | Wall |
| `$08`–`$0F` | A block; its type is the value minus 7, so types 1–8 |

There are **no per-stage width or height variables** — every stage fills all 120 cells and a small
stage is the same array with more `$03` around it. `state.bounding_box()` derives the shape.

### Block types are randomised per playthrough

The stage data in ROM records only *that* a block occupies a cell; the type is drawn by a PRNG at
load time. Never hard-code expected types per stage. The invariant that does hold: two blocks that
look identical on screen carry identical type bytes.

### Literals

```
at(cursor, ROW, COL)
at(block-T, ROW, COL)
remaining(N)
all-blocks-matched(block-T)     # type T was in the stage and is now gone
goal-reached
terminal-state
```

Terrain is deliberately *not* in the literals — it is static per stage, and repeating the walls in
every state would swamp the atoms that actually change. Read `state.grid` for it.

`depth` is also deliberately absent. With a step counter in the literals no successor could ever
equal its parent, and the self-loop filter in `successors` would be dead code — the trap the
[urban planning](urban-planning.md#known-quirks) environment fell into.

Two states are equal when their grid and cursor match, which is the whole position: depth and history
do not distinguish two ways of reaching the same board, so search can close.

### Cross-checking

The grid and the record array reference each other, so the board can be read from either end.
`state.is_consistent()` runs the memory map's own check — the grid scan, the record scan and `$D019`
must agree. If they do not, one of the three has drifted; the grid scan and `$D019` are the two that
were verified against live RAM.

Note that records are zeroed **in place** and slots are never compacted, so walking the record array
until the first `$00` type byte stops at the first hole and reports almost nothing. `decode_records`
iterates all `$D018` slots and skips the dead ones.

## Actions

There are two action models. `push` is the default and is the one to plan with; `button` is
the raw button presses underneath it.

```python
env = PuzznicGBEnv(rom)                      # push  — one action per legal push
env = PuzznicGBEnv(rom, actions="button")    # button — one action per button press
```

### Why the default is not the buttons

Measured on the synthetic cartridge's stages, expanding with the primitive button actions:

| Stage | Branching factor | Successors that move a block |
|---|---|---|
| 2 blocks | 4.33 | **5.9%** |
| 4 blocks | 4.06 | **3.3%** |
| 7 blocks | 4.16 | **5.6%** |

Over 90% of every expansion is the cursor walking. That is not a decision the planner is
making — walking to a block is the *overhead* of making one — but blind search pays for it
at every node, and it multiplies: a plan that pushes eight times across a 6×8 room is forty
or fifty button presses deep, at branching 4.

`PuzznicGBPush` collapses the walk into the push. Same stages, same breadth-first search,
25-second budget:

| Stage | Model | Branching | Productive | Result |
|---|---|---|---|---|
| 2 blocks | button | 4.33 | 5.9% | solved, 12 expansions, 1.8 s |
| | **push** | 2.50 | **100%** | **solved, 2 expansions, 0.1 s** |
| 4 blocks | button | 4.06 | 3.3% | **not solved** |
| | **push** | 6.03 | **100%** | **solved in 12 s**, 4 pushes |
| 7 blocks | button | 4.16 | 5.6% | not solved |
| | push | 11.28 | 100% | not solved — this one needs a heuristic |

Branching *rises* with the push model on a crowded board, because it is one action per legal
push and a crowded board has more of them. That is the right trade: every one of them
changes the position.

### `push` — the planning model

`PuzznicGBPush(row, col, direction)` walks the cursor onto the block at `(row, col)` and
pushes it one cell left or right. `direction` is only ever `left` or `right`: Puzznic slides
blocks sideways, it does not lift them.

`successors` offers one per **legal** push, worked out from the grid without running the
emulator at all — a block only moves into an empty cell, since the movement check at
`1:506E` rejects every non-zero cell value, so ledges and walls obstruct exactly as each
other. Illegal pushes never cost a frame of emulation.

The walk is adaptive, not a precomputed button sequence: it presses towards the target,
re-reads `$D012`/`$D013`, and stops when it arrives or when a press makes no progress. A
cartridge whose cursor cannot cross some cell therefore produces a no-op, which `successors`
filters out, rather than a plan that quietly does something else.

**Cost** is the number of button presses: the walk plus the push. A planner minimising it
prefers the block nearest the cursor, which is what a person would do.

### `button` — the raw presses

`PuzznicGBAction` wraps a string of the form `"buttons,ticks"`, where buttons are `+`-joined
and ticks is how many frames to hold them — the same spelling the
[Super Mario Land](super-mario-land.md#actions) environment uses.

| Action | Effect |
|---|---|
| `left` / `right` / `up` / `down` | Move the cursor one cell |
| `a+left` / `a+right` | Push the block under the cursor one cell sideways |

**Cost** is `1` for every action. `a` costs `0` in `action_cost_map` because it is a
modifier that turns a direction into a push, not a move of its own.

### Ticks are measured, not chosen

How long to hold a button has two bounds, and neither is in the memory map:

- **Too short** and the press is never sampled.
- **Too long** and the cursor's auto-repeat fires, so one action moves the cursor two cells
  and the state the planner gets back is not the one its action described.

So `reset()` measures them off the cartridge instead of trusting a constant:

```python
state, info = env.reset()
info["calibration"]
# Calibration(press_ticks=8, hold_window=(1, 16), push_scheme='modifier', push_prefix='a')
```

`measure_hold_window` presses a direction for 1, 2, 3… frames and watches `$D012`/`$D013`,
returning the closed range of holds that move the cursor **exactly one cell** — the lower
end is where presses start registering, the upper end is one frame short of auto-repeat.
`press_ticks` is the middle of that range, far enough from either edge to survive a frame of
jitter in when the game samples input. The spare frames cost about 0.07 ms each, so there is
nothing to win by shaving them.

`probe_push_scheme` then answers a question the memory map does not: **how this cartridge
moves a block.** It walks the cursor onto a real block with somewhere to go and tries each
candidate:

| Scheme | Inputs |
|---|---|
| `modifier` | hold A and press a direction — one input |
| `grab` | press A to pick the block up, then a direction — two inputs |
| `direct` | a direction alone moves the block under the cursor |

whichever actually moves the block is the one both action models then use. Until this
existed, `a+left` was an assumption; now it is a measurement, and a cartridge that works
some other way is handled rather than silently mis-driven.

Calibration describes the game, not the stage, so it runs once per environment and is reused
across resets. It costs about 0.15 s. `PuzznicGBEnv(rom, calibrate=False)` skips it and
falls back to `PRESS_TICKS` and the `modifier` scheme.

### Applying an action, and settling

A push is not instantaneous. The block slides, blocks above it fall, a match clears through the `$01`
transient, and the fall can cascade into another match. Snapshotting straight after the button press
would capture the middle of that, so `apply` presses the buttons and then runs `settle`:

```python
load_state(pyboy, state.gb_state)      # rewind to the parent
pyboy.button(button, hold)             # press
pyboy.tick(max(ticks) + 1)
settle(pyboy)                          # ...then wait for the board to stop moving
```

The board counts as settled once no cell is mid-clear and the grid has been byte-identical for
`SETTLE_STABLE_TICKS` (4) frames, giving up after `SETTLE_MAX_TICKS` (600). Two things cut the wait
short: `$D019` reaching zero, because clearing the last block ends the stage and the cartridge then
loads the next round straight over the top of it; and `$D018` changing, which means a new stage has
loaded and there is nothing left to wait for.

Both are configurable per environment:

```python
env = PuzznicGBEnv(rom, settle_max_ticks=1200, settle_stable_ticks=8)
```

## Goal and terminal

- **Goal** (`is_goal`) — `state.stage_cleared`: the stage loaded with blocks, `$D019` is zero and the
  grid scan agrees.
- **Terminal** (`is_terminal`) — `state.dead_end`: some type has exactly one block left. Matching is
  pairwise, so a lone block can never be removed and the stage can no longer be cleared. This is a
  property of the position that the cartridge does not flag; it is the same rule the pure-Python
  environment uses, and it is sound — three blocks of a type are *not* a dead end, because Puzznic
  clears a group of three when they meet.

Both read flags captured on the passed `state`, not live memory, which would describe whichever state
was applied last.

Both are also **absorbing**, as in the pure-Python environment: `successors` returns `[]` for them and
`step`/`simulate` hand the state straight back. That is not tidiness. Clearing the last block ends the
stage and the cartridge loads the next round over the top of it, so an action applied past a goal
state would quietly return a position from a *different* stage.

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

`env.render()` prints the de-duplicated history of `step` calls and returns it as a list of strings,
the same as the pure-Python environment. `state.save(rom, path)` writes a PNG of the state by booting
a throwaway emulator to it — it uses the null window, so unlike the Super Mario Land environment it
needs no display, but it does need Pillow.

## Planning

The generic `TreeSearchPlanner` from
[`super_mario_planner_gb.py`](../../planiverse/planners/super_mario_planner_gb.py) works against this
environment as-is; supply a heuristic and a cost function:

```python
from collections import defaultdict
from planiverse.planners.super_mario_planner_gb import TreeSearchPlanner

def heuristic(state):
    """Sum, over each type, of the distance between its two closest blocks."""
    by_type = defaultdict(list)
    for block in state.blocks:
        by_type[block.type].append(block)
    total = 0
    for blocks in by_type.values():
        pairs = [abs(a.row - b.row) + abs(a.col - b.col)
                 for i, a in enumerate(blocks) for b in blocks[i + 1:]]
        total += min(pairs) if pairs else 50      # a lone block is a dead end
    return total

plan = TreeSearchPlanner().search(state, env, heuristic, lambda states, actions: len(actions))
env.validate(plan)
```

`is_terminal` is worth checking during expansion: stranding a colour is a genuine dead end, and
pruning those branches is most of what makes this a planning problem rather than a reflex one.

The cursor-walking half of that problem is what the `push` action model fixes — see
[Actions](#actions) for the measurements. What is left is the heuristic: Manhattan distance between
same-typed blocks is a weak proxy, because it ignores whether the two can actually be brought
together, and a seven-block stage still defeats plain best-first search. A heuristic that counts
*pushes* rather than cells, and that recognises a block wedged where no push can reach the pair it
needs, is the piece still worth writing.

Note that `is_terminal` already prunes one whole class of dead end for free: a stage with a stranded
colour cannot be won, and `successors` returns nothing from such a state.

## Known quirks and gaps

- **The stage count is unknown.** `fix_index` accepts the whole `0`–`255` byte because that is the
  range the loader can be pointed at, not because there are 256 stages. Out-of-range indices read
  past the pointer table.
- **No score.** The Game Boy's score lives in the status-panel tilemap shadow at `$D800`, and the
  digit layout in there was never worked out, so nothing reads it. `step` returns `blocks_cleared` in
  the reward slot instead. The pure-Python environment's `score` is its own invented formula, so the
  two were never comparable anyway.
- **No timer.** Puzznic gives you a time limit per stage and takes a life when it runs out. The timer
  address is not in the memory map, so a state that has run out of time is not terminal here. Long
  plans are the case to watch.
- **Stage transitions were never observed** on the real cartridge. `$D018`/`$D019` should be
  re-initialised on entering the next round; `settle` stops the moment `$D019` hits zero precisely so
  that a cleared stage is observed before the next one loads over it, but this was not verified
  against the ROM.
- **`successors` shares one emulator.** All expansion runs on `self.pyboy`; correctness relies on
  each action reloading its parent's save-state first. Don't parallelise expansion over one env.
- **`render=True` is for watching, not planning.** It opens an SDL2 window and slows expansion. The
  constructor keeps that flag as `self.render_window`, because `self.render` is the method that
  prints the history.
- **`load_state` advances a frame** after restoring, mirroring the Super Mario Land environment.
  Nothing moves on its own in a settled Puzznic position, so it does not shift the state — but it is
  not a byte-exact restore.

## Testing without the cartridge

Everything above would be untestable in CI if it needed a copyrighted ROM, so the test suite builds
its own. [`tests/fake_puzznic_rom.py`](../../tests/fake_puzznic_rom.py) assembles a small homebrew
Game Boy cartridge — with [`tests/sm83.py`](../../tests/sm83.py), a minimal SM83 assembler — that
puts the same facts at the same addresses: a stage loader at `$0430` that reads `$D003`, a 12×10 grid
at `$DF00`, records at `$DD00`, counters at `$D018`/`$D019`, a cursor at `$D012`/`$D013`.

It is **not** a Puzznic clone: no gravity, no timer, no score, no cascades. A push that leaves two
same-typed blocks adjacent clears both after a three-frame `$01` transient, and that is the entire
rule set. It does model **cursor auto-repeat** — a direction held for more than 16 frames moves the
cursor again, and every 6 frames after that — because otherwise any hold length would look correct
and there would be nothing for `measure_hold_window` to find. What it exercises is the interface between the environment and a Game Boy — booting,
hooking the loader to force a stage, decoding the grid, waiting for a move to settle, spotting a
cleared stage — which is the part that otherwise could only be checked by hand.

Its title screen deliberately rewrites `$D003` and calls the loader in the same frame, so
`test_fix_index_selects_the_stage_the_loader_builds` fails if stage selection is ever changed back to
poking memory between frames.

The tests that need the real cartridge are opt-in:

```bash
PLANIVERSE_PUZZNIC_ROM="/path/to/Puzznic (J).gb" poetry run pytest tests/test_puzznic_gb.py
```

## Files

| Path | What |
|---|---|
| [`puzznic_gb.py`](../../planiverse/problems/retro_games/puzznic_gb.py) | `PuzznicGBEnv`, `PuzznicGBState`, `PuzznicGBPush`, `PuzznicGBAction`, the calibration, and the RAM decoders |
| [`tests/test_puzznic_gb.py`](../../tests/test_puzznic_gb.py) | Tests, against the synthetic cartridge and the real one |
| [`tests/fake_puzznic_rom.py`](../../tests/fake_puzznic_rom.py) | The synthetic cartridge |
| [`tests/sm83.py`](../../tests/sm83.py) | The assembler that builds it |
