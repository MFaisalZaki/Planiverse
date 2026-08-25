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
- **Import:** `from planiverse.problems.retro_games.puzznic_gb import PuzznicGBEnv, PuzznicGBAction`
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
# left_for_15 1 6
# right_for_15 1 6
# ...

ticks = info["calibration"].press_ticks          # measured; do not hard-code it
trace = env.simulate([PuzznicGBAction(f"left,{ticks}"), PuzznicGBAction(f"a+right,{ticks}")])
print(env.is_goal(trace[-1]))
```

(The board above is illustrative — the stage layouts are in the ROM, and the block *types* are drawn
by a PRNG at load time, so what Round 1 prints is not fixed.)

`reset()` boots the ROM, taps through the title screens until a stage is on the playfield, waits for
the board to settle, and snapshots it.

## Stages

`Puzznic (J)` has **128 rounds**, and `fix_index(i)` selects one of them zero-based —
`fix_index(3)` is round 4. `reset()` reaches it the way a player would: it picks `PASSWORD`
on the title menu and types the round's password.

```python
env.fix_index(49)
env.password_for(49)        # 'PASSWORD' — round 50's password really is that
state, info = env.reset()
info["boot_route"], info["stage_index"]
# ('password', 49)
```

### The passwords come out of the cartridge

They are not transcribed from a guide. The table is in the ROM at `$47FA`: 128 ten-byte
entries, eight bytes of password in the game's own text encoding, then the round number,
then a check byte. `read_passwords(romfile)` walks it until an entry stops being a password
whose round number follows the last one — which is both how the end is found and how the
parse checks itself.

Reading it rather than hard-coding it means the passwords always match the ROM in hand, and
it is what settles the round count: the table is 128 long, and round 128's password is
`SAISHUU.` — 最終, "final".

The text encoding is letters from `$0A`, so `A` is `$0A` and `Z` is `$23`, leaving `$00`–`$09`
for the digits. `.` is `$24` in the ROM and `$8B` on screen.

### How entry works

The title menu is `1PLAYER / 2PLAYERS / PASSWORD`, and the password screen is a 9×3 grid of
`A`–`Z` and `.` over a row of `NEXT / BACK / END`, with eight slots along the top. Selecting a
character fills the next slot and advances; `END` confirms and starts the round.

Everything on both screens that matters is a **sprite**, which is what makes this reliable
rather than hopeful:

| What | Where |
|---|---|
| Title-menu cursor | arrow sprite (tile `$AC`) at y = 88 / 104 / 120 |
| Entry cursor | the same arrow; cell = `(y - 80) / 16`, `(x - 16) / 16` |
| The eight slots | sprites at y = 48, x = 24, 40, 56, 72, 96, 112, 128, 144 |
| An empty slot | tile `$8C` |

So `entry_cursor` reads where the cursor actually is and `entered_password` reads what has
actually been typed. Every cursor move is checked, and every character is checked against the
slot it was meant to fill and retried if it did not land — the screen does drop a press now
and again, and a password one character short is not an error, it is silently the wrong
round.

### Why not just poke the loader

`$D003` is the stage index, and writing it through a hook on the loader at `0:0430` does
change the layout — the two routes agree, and `fix_index(3)` produces the same board either
way. But it swaps the layout underneath a game that still believes it is on round 1, so
everything the cartridge derives from the round is wrong, and the hook has to keep firing for
every later load. Typing the password puts the game on the round properly.

The poke is kept for one case: a cartridge with no title menu. The synthetic test ROM has a
password *table* but no password *screen*, so it takes that route, and `info["boot_route"]`
says which of `password`, `1player` or `tapped` was used.

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

`PuzznicGBAction` wraps a string of the form `"buttons,ticks"`, where buttons are `+`-joined
and ticks is how many frames to hold them — the same spelling the
[Super Mario Land](super-mario-land.md#actions) environment uses. That is the whole action
set: what the console has, and nothing above it.

| Action | Effect |
|---|---|
| `left` / `right` / `up` / `down` | Move the cursor one cell |
| `a+left` / `a+right` | Push the block under the cursor one cell sideways |

There is no `a+up`/`a+down`: Puzznic slides blocks sideways, it does not lift them. `B` works
as the modifier as well as `A`, and a direction on its own only walks the cursor off the
block — both checked on the cartridge.

**Cost** is `1` for every action. `a` costs `0` in `action_cost_map` because it is a modifier
that turns a direction into a push, not a move of its own.

### The branching is mostly cursor walking

Worth knowing before you point a planner at this. Measured over the synthetic cartridge's
stages:

| Stage | Branching factor | Successors that move a block |
|---|---|---|
| 2 blocks | 4.33 | 5.9% |
| 4 blocks | 4.06 | 3.3% |
| 7 blocks | 4.16 | 5.6% |

Over 90% of every expansion is the cursor travelling, because walking to a block is not a
decision — it is the overhead of making one. Breadth-first search over the buttons solves the
2-block stage and does not close the 4-block one inside 25 seconds.

That is a property of the game as the console presents it, and this environment leaves it
alone: collapsing the walk into the push is a search technique, and search is the planner's
side of the line. `cursor_path` is exported for a planner that wants to build such macros —
it is the reachability the cursor really has, walls and all — but the actions this
environment hands out are button presses.

### Ticks are measured, not chosen

How long to hold a button has two bounds, and neither is in the memory map:

- **Too short** and the press is never sampled.
- **Too long** and auto-repeat fires, so one action moves the cursor — or the block it is
  holding — two cells, and the state the planner gets back is not the one its action
  described.

The failure is quiet. `PuzznicGBAction("a+right,60")` looks like one push and reads like one
action in a plan, but it slides the block until the button comes up.

So `reset()` measures them off the cartridge instead of trusting a constant:

```python
state, info = env.reset()
info["calibration"]
# Calibration(press_ticks=15, hold_window=(1, 30), push_scheme='modifier',
#             push_prefix='a', push_ticks=None, push_window=None)
```

Those are the real cartridge's numbers. **`Puzznic (J)` repeats on frame 31**, so any hold of
30 frames or fewer moves one cell and `press_ticks` settles on 15. A hold of 60 — which looks
like one press and reads like one action in a plan — moves two.

or, without writing any code:

```bash
python -m planiverse.problems.retro_games.puzznic_gb "Puzznic (J).gb" --stage 0
```

`measure_hold_window` presses a direction for 1, 2, 3… frames and watches `$D012`/`$D013`,
returning the closed range of holds that move the cursor **exactly one cell** — the lower
end is where presses start registering, the upper end is one frame short of auto-repeat.
`press_ticks` is the middle of that range, far enough from either edge to survive a frame of
jitter in when the game samples input. The spare frames cost about 0.07 ms each, so there is
nothing to win by shaving them.

Measured across the first eight stages of `Puzznic (J)`:

| Stage | Blocks | Cursor window | Push window |
|---|---|---|---|
| 0 | 6 | (1, 30) | not probeable |
| 1 | 10 | (1, 30) | not probeable |
| 2 | 8 | (1, 30) | not probeable |
| 3 | 8 | (1, 30) | not probeable |
| 4 | 9 | (1, 30) | **(1, 30)** |
| 5 | 7 | (1, 30) | not probeable |
| 6 | 15 | **(2, 31)** | not probeable |
| 7 | 8 | (1, 30) | **(1, 30)** |

Stage 6 is the one that shows why the middle of the window is the right choice rather than
its lower edge: there a single-frame press is not sampled at all. And where the push window
*could* be measured it matches the cursor's, so on this cartridge one repeat routine serves
both — which is now a measurement rather than the assumption it used to be.

### The push has its own window

`measure_push_window` does the same for a **held block**, and it is a separate measurement
because the two need not agree — a cartridge is free to repeat a held block on its own
schedule. Getting this wrong is worse than getting the cursor wrong: held past the repeat, a
single `a+right` slides the block two cells, and if the second cell puts it next to its own
colour the pair clears. The planner is then handed a board two blocks lighter than the action
it applied ever asked for.

Probing it needs a block that can be slid two cells and still be *found* afterwards, so
`push_probe_candidates` rejects any block that would fall down a hole on the way, or land
next to its own colour and vanish. (These are calibration machinery, not an action model.) On a cramped stage there may be no such block, and then
`push_window` is `None` and `push_ticks` falls back to `press_ticks` — better than a number
read off a board that cleared halfway through the measurement.

`button_actions` holds each action for its own window:

```python
['left,8', 'right,8', 'up,8', 'down,8', 'a+left,5', 'a+right,5']
```

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

On `Puzznic (J)` it comes out `modifier`, which is what the environment had always assumed —
and probing by hand confirms the rest of the picture: `B` works as the modifier too, a
direction on its own only walks the cursor off the block, and `A` followed by a direction
does nothing, so there is no pick-up-and-carry mode to model.

Calibration describes the game, not the stage, so it runs once per environment and is reused
across resets. It costs about 0.15 s. `PuzznicGBEnv(rom, calibrate=False)` skips it and
falls back to `PRESS_TICKS` and the `modifier` scheme.

### The round has to start listening first

The stage loader fills work RAM before the round has finished announcing itself. For **210
frames** on `Puzznic (J)` the board is completely readable — grid, records, counters, cursor,
all correct and cross-checking — and every button is ignored. Nothing in RAM says "not yet".

A state snapshotted in that window is the worst kind of wrong: it looks perfectly normal and
answers no action, so a planner sees a stage with no legal moves rather than an error. So
`reset()` calls `wait_until_interactive`, which presses a direction from a snapshot at
increasing offsets until the cursor answers, then rewinds and replays only the waiting — the
state you get back still has the cursor exactly where the loader put it. `info["intro_ticks"]`
reports how long it took.

It retries having pressed START first, because **START is the pause button** once a round is
running, and the boot sequence taps it while getting through the title screens.

### Where the cursor may go

The cursor sits on blocks and crosses ledges, and cannot enter a wall or the outside — all
four verified on the cartridge. Stages are not rectangles (Round 1's bottom row is two cells
narrower than the row above it), so walking the rows and then the columns steps into a wall.
`cursor_path` is a breadth-first search over the cells the cursor may occupy — exported
because a planner reasoning about which blocks are even reachable needs it.

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

The cursor walking is most of what makes this hard — see
[Actions](#actions) for the measurements — and a planner that wants to collapse it into
macro-actions can, using `cursor_path` for the reachability. The other half is the heuristic:
Manhattan distance between
same-typed blocks is a weak proxy, because it ignores whether the two can actually be brought
together, and a seven-block stage still defeats plain best-first search. A heuristic that counts
*pushes* rather than cells, and that recognises a block wedged where no push can reach the pair it
needs, is the piece still worth writing.

Note that `is_terminal` already prunes one whole class of dead end for free: a stage with a stranded
colour cannot be won, and `successors` returns nothing from such a state.

## Verified against the cartridge

Everything below was checked by driving `Puzznic (J)` (MD5 `9a777d82cd7a8913ba1aed2cc854fa50`)
through this environment, not read off a disassembly:

- **Boot, stage selection and decoding.** The first eight stages all load, and on every one of
  them the grid scan, the record array and `$D019` agree (`state.is_consistent()`).
- **Round selection, both ways.** The password table parses to 128 rounds; typing rounds 1, 4,
  16, 50, 100 and 128 puts `$D003` on each one, and poking the loader agrees with it.
- **Cell semantics.** Ledges, walls and the outside marker behave as the memory map says, and
  the cursor's freedom of movement matches: on blocks and ledges, never into a wall.
- **The push scheme and both hold windows** — see [Actions](#actions).
- **Matching.** `a+right` on Round 1's block at row 8, column 3 slides it onto its own colour
  and clears the pair: `$D019` drops from 6 to 4, records zeroed in place, slots uncompacted.
- **Search.** Breadth-first over `push` actions solves Round 1 in **4 pushes from 10 states in
  about a second**, and `validate` replays it to an empty board.

## Known quirks and gaps

- **No score.** The Game Boy's score lives in the status-panel tilemap shadow at `$D800`, and the
  digit layout in there was never worked out, so nothing reads it. `step` returns `blocks_cleared` in
  the reward slot instead. The pure-Python environment's `score` is its own invented formula, so the
  two were never comparable anyway.
- **No timer.** Puzznic gives you a time limit per stage and takes a life when it runs out. The timer
  address is not in the memory map, so a state that has run out of time is not terminal here. Long
  plans are the case to watch.
- **Stage transitions are still unobserved.** `settle` stops the moment `$D019` hits zero precisely
  so that a cleared stage is seen before the next round loads over it, and Round 1 does get cleared
  and validated — but what the cartridge does *after* that, and whether `$D018`/`$D019` are
  re-initialised the way the memory map expects, has not been watched.
- **The push window is not measurable on every stage.** It needs a block that can be slid two cells
  without falling or matching, and dense boards have none — six of the first eight stages, in fact.
  There `push_ticks` is `None` and falls back to the cursor hold, which on this cartridge is the same
  number anyway.
- **Only a handful of rounds have been driven.** Rounds 1, 4, 16, 50, 100 and 128 were selected by
  password and decoded correctly, and the first eight by poking the loader. Nothing suggests the rest differ, but the intro
  length, the hold windows and the push scheme are all measured per-cartridge and not per-stage, so
  a stage that behaved differently would not be noticed.
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
rule set. It does model the two things that make driving a Game Boy awkward, because otherwise the
code for them would never run: **cursor auto-repeat** (a direction held more than 16 frames moves the
cursor again, and every 6 frames after that), so `measure_hold_window` has a real bound to find; and
a **round intro** of 60 frames during which the board is completely readable and every button is
ignored, so `wait_until_interactive` has a real wait to discover. What it exercises is the interface between the environment and a Game Boy — booting,
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
| [`puzznic_gb.py`](../../planiverse/problems/retro_games/puzznic_gb.py) | `PuzznicGBEnv`, `PuzznicGBState`, `PuzznicGBAction`, the calibration, and the RAM decoders |
| [`tests/test_puzznic_gb.py`](../../tests/test_puzznic_gb.py) | Tests, against the synthetic cartridge and the real one |
| [`tests/fake_puzznic_rom.py`](../../tests/fake_puzznic_rom.py) | The synthetic cartridge |
| [`tests/sm83.py`](../../tests/sm83.py) | The assembler that builds it |
