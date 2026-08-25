# Flipull (Game Boy)

Flipull — Taito's *Plotting* — played on the cartridge. This environment drives the real US Game Boy
ROM inside [PyBoy](https://github.com/Baekalfen/PyBoy) and reads the block field straight out of the
console's work RAM, so the transition function is the game's own code rather than a reconstruction of
its rules. States are emulator save-states, so search can branch by rewinding the machine.

The player stands at the right of a wall of blocks holding one of them. Throwing it sends it left
along a row: it destroys blocks of its own type as it goes, and swaps with the first block of a
different type, which becomes the new held block. Destroying a block drops its column. The stage is
finished when few enough blocks are left.

That makes the action set unusually small for a Game Boy game — **pick a row, throw** — while the
consequences of a throw run several moves deep. Compare
[Puzznic (Game Boy)](puzznic-gb.md), where over 90% of every expansion is the cursor walking to the
block you meant to move. Here there is no walking to speak of: at most fourteen rows to choose
between, and the branching factor is three.

Every address this environment reads is catalogued in the
[memory map](flipull-gb-memory-map.md), including how each one was established and how much of it is
verified against live RAM rather than read off a disassembly. That grading matters more here than it
does for Puzznic — see [What is and is not verified](#what-is-and-is-not-verified).

> **Read that section before trusting this.** Unlike the [Puzznic](puzznic-gb.md) environment, which
> was driven against its cartridge end to end, **this environment has never been run against the real
> Flipull dump.** It was built against the memory map's recorded observations and exercised against a
> synthetic cartridge that reproduces them. Every number printed below is from that synthetic
> cartridge unless it is one the map recorded.

- **Class:** `FlipullGBEnv`
- **Import:** `from planiverse.problems.retro_games.flipull_gb import FlipullGBEnv, FlipullGBAction`
- **Source:** [`planiverse/problems/retro_games/flipull_gb.py`](../../planiverse/problems/retro_games/flipull_gb.py)
- **Dependencies:** `pyboy` + a `Flipull (USA).gb` ROM you supply (`pillow` for screenshots)

## The ROM

**Not included, and cannot be.** Flipull is Taito's copyrighted work; the repo ships no ROM and none
will be added. Supply your own legally-obtained dump and pass its path:

```python
env = FlipullGBEnv("Flipull (USA).gb")
```

Every address below was read from one specific dump:

| | |
|---|---|
| File | `Flipull (USA).gb`, 32,768 bytes |
| MD5 | `4fcc13db8144687e6b28200387aed25c` |
| Cartridge | **No mapper** — the whole ROM is flat at `$0000–$7FFF`, and there is no cartridge RAM |

All state therefore lives in work RAM and HRAM, and Flipull leans on HRAM unusually heavily: nearly
every counter is there rather than in WRAM.

Because the addresses are revision-specific, the constructor hashes the file and raises a
`UserWarning` when it is not that dump. Pass `verify_rom=False` to silence it.

## Quickstart

```python
from planiverse.problems.retro_games.flipull_gb import FlipullGBEnv, FlipullGBAction

env = FlipullGBEnv("Flipull (USA).gb", render=False)   # render=True opens an SDL2 window
state, info = env.reset()

print(state)              # the field, trimmed to its bounding box.
                          # The blocks are stage 1 as the memory map recorded it; the
                          # staircase, the player's Y and the tick counts below are the
                          # synthetic cartridge's, and the real dump may differ:
# ################
# #=
# # =
# #  =
# #23343
# #23112
# #43234
# #22121
# #14441
# ################
# held: 1
# player: 104

print(state.blocks_remaining, state.clear_target)      # 25 9
for action, successor in env.successors(state):
    print(action, action.cost(), successor.blocks_remaining, successor.held_block)
# up_for_8 1 25 1
# down_for_8 1 25 1
# a_for_8 1 25 3      <- the throw met a block of another type: nothing destroyed,
#                        and what came back in hand is now a 3

ticks = info["calibration"].press_ticks                # measured; do not hard-code it
throw = info["calibration"].throw_button               # probed; likewise
trace = env.simulate([FlipullGBAction(f"down,{ticks}"), FlipullGBAction(f"{throw},{ticks}")])
print(env.is_goal(trace[-1]))
```

or, without writing any code:

```bash
python -m planiverse.problems.retro_games.flipull_gb "Flipull (USA).gb"
```

which prints the field, the measurements and the action set.

`reset()` boots the ROM, taps through the title screens until a field with blocks on it is up, waits
for the stage to start accepting input, calibrates, and snapshots.

## Stages

**`fix_index` accepts only `0`.** No way to select a stage has been established, and rather than
quietly starting stage 1 and calling it stage 9, anything else asserts:

```python
env.fix_index(3)
# AssertionError: Invalid index: no verified way to select a stage exists yet.
```

`$FFC6` looks like the stage number — it read `01` during stage 1 — but the memory map grades it
**unverified**, having never seen it change, and no password or level-select route has been looked
for. Puzznic's `fix_index` works because its password table was found in ROM at `$47FA` and typing a
password was watched putting `$D003` on the right round; nothing equivalent exists here yet. So index
0 is whatever stage the cartridge boots into, and `info["stage"]` reports what `$FFC6` says it is.

This is the single biggest gap in the environment. Until it is closed, Flipull is one problem
instance rather than a benchmark set.

## State representation

`FlipullGBState` carries the **entire emulator save-state** (`gb_state`, the bytes from
`pyboy.save_state`) plus the position scraped from RAM. The save-state is what makes branching
possible: applying an action loads the parent's bytes back into the emulator first, so siblings expand
from an identical machine.

| Field | Source |
|---|---|
| `field` | 14×16 tuple of cell values from `$C840`, stride `$20` per row |
| `blocks` | `(row, col, type)` for every cell holding a block |
| `staircase` | the fixed `$87` cells |
| `blocks_remaining` | `$FFCA`×10 + `$FFC9` — the live count |
| `blocks_initial` | `$FFC1`×10 + `$FFC0` — what the stage started with |
| `clear_target` | `$FFCF` — the `CLEAR` number |
| `timer_seconds` | `$FFCE`×60 + `$FFCC`×10 + `$FFCB` |
| `stage` | `$FFC6` |
| `held_block` | `$FFD4`, as a 1–4 type |
| `player_y` | the player sprite's Y in the OAM buffer |
| `stage_types` | the block types present when the stage loaded |
| `stage_cleared` | sampled here, read by `is_goal` |
| `out_of_time` | sampled here, read by `is_terminal` |

### Cell values

`field` holds the game's own codes, not an abstraction of them:

| Value | Meaning |
|---|---|
| `$00` | Outside the field — the row stride is 32 bytes but only 16 columns carry meaning |
| `$80` | Border: the ceiling, the floor, and the left wall |
| `$83`–`$86` | A playable block, four types |
| `$87` | The fixed staircase — structural, never clearable, and excluded from the count |

`decode_blocks` counts only `$83`–`$86`, which is what makes the field agree with the HUD.

### Counters are decimal digits, one per byte

Flipull stores every counter as **separate decimal digits, ones first** — not binary, and not packed
BCD like Super Mario Land's score. Searching RAM for 25 or `$19` finds nothing; the value is `05` and
`02` in adjacent bytes. `decode_digits(tens, ones)` is the whole of it, and it is the reason
`blocks_remaining` is not simply `pyboy.memory[...]`.

### Literals

```
at(player, Y)
at(block-T, ROW, COL)
holding(block-T)
remaining(N)
clear-target(N)
all-blocks-cleared(block-T)     # type T was in the stage and is now gone
goal-reached
terminal-state
```

Terrain — the border and the staircase — is deliberately *not* in the literals: it is static per
stage, and repeating it in every state would swamp the atoms that actually change. Read
`state.field` for it.

`at(player, Y)` is the sprite's Y, **not a row index**. The memory map has no row variable for the
player (`$C002` tracks vertical *input*, `89`/`8F`, not position), so where he is, is only readable
from where he is drawn. `calibration.row_pitch` is the measured pixels-per-row if you want to turn
one into the other.

Two states are equal when their field, held block and player row match. The player's row belongs in
that because **a throw from a different row does something different** — leave it out and `up` and
`down` become self-loops, `successors` filters them, and the environment offers a single action.

`depth` is deliberately absent, as in every other environment here: with a step counter in the
literals no successor could ever equal its parent.

### Cross-checking

`state.is_consistent()` runs the memory map's own check: the number of `$83`–`$86` cells in the field
against the HUD counter at `$FFC9`/`$FFCA`. That check is what the map itself leans on — 25 cells
against `BLOCK 25`, and 24 against 24 after a throw — and a mismatch means one of the two has
drifted. It is also what `stage_is_loaded` uses to decide the boot sequence has arrived somewhere
real.

## Actions

`FlipullGBAction` wraps a string of the form `"button,ticks"` — the same spelling the
[Puzznic](puzznic-gb.md#actions) and [Super Mario Land](super-mario-land.md#actions) environments
use. That is the whole action set: what the console has, and nothing above it.

| Action | Effect |
|---|---|
| `up` / `down` | Move the player one row |
| `a` (or whichever button `probe_throw_button` finds) | Throw the held block |

**Cost** is `1` for every action.

Three actions, so the branching factor is three, and every one of them does something: this is a
game whose console interface is already close to its planning interface. Contrast Puzznic, where the
same button-level honesty costs you a branching factor of four in which one expansion in twenty
moves a block.

### Ticks are measured, not chosen

How long to hold a button has two bounds, and neither is in the memory map:

- **Too short** and the press is never sampled.
- **Too long** and auto-repeat fires, so one action moves the player two rows, and the state the
  planner gets back is not the one its action described.

The failure is quiet: `FlipullGBAction("down,60")` looks like one press and reads like one action in
a plan. So `reset()` measures off the cartridge instead of trusting a constant:

```python
state, info = env.reset()
info["calibration"]
# Calibration(press_ticks=8, hold_window=(1, 16), throw_button='a',
#             throw_ticks=8, player_sprite=0, row_pitch=8)
```

Those numbers are the *synthetic* cartridge's — its auto-repeat fires on frame 17 by construction.
What the real dump repeats on is exactly what this measures and nobody has yet read off; Puzznic's
came out `(1, 30)`, and a `press_ticks` guessed from that would have been wrong on either.

`measure_hold_window` presses `down` for 1, 2, 3… frames and watches the player sprite move,
returning the closed range of holds that move him **exactly one row** — the lower end is where
presses start registering, the upper end is one frame short of auto-repeat. `press_ticks` is the
middle of that range, far enough from either edge to survive a frame of jitter in when the game
samples input.

Calibration describes the game, not the stage, so it runs once per environment and is reused across
resets. `FlipullGBEnv(rom, calibrate=False)` skips it and falls back to `PRESS_TICKS` and `a`.

### Which sprite is the player

Measuring a hold window needs something that moves, and Flipull gives you nothing in RAM that says
where the player is. So `probe_player_sprite` finds him: it snapshots, taps `up`, taps `down`, and
looks for the OAM entry that went **up for one and down for the other**.

Requiring opposite movement rather than merely *change* is not fussiness. The first version of this
accepted anything that moved, and on the synthetic cartridge it found two candidates — because that
ROM had parked a scratch variable inside the OAM DMA buffer at `$C010`, where the environment reads
sprites from. A probe that accepted the first mover would have silently tracked a counter instead of
the player, and every hold window measured after that would have been meaningless. It returns `None`
when the answer is not unique, and calibration degrades to the fallback rather than lying.

### Which button throws

`probe_throw_button` presses each of `A` and `B` and watches for a throw in two independent places:
the flags at `$FFD2`/`$FFD3` going up, and the field changing. A button that only moves the player
is therefore not mistaken for one that throws.

### The stage has to start listening first

As in Puzznic, a stage can be entirely readable while its intro is still running — Puzznic ignores
input for 210 frames after its field is in memory. A state snapshotted in that window is the worst
kind of wrong: it looks perfectly normal and answers no action, so a planner sees a stage with no
legal moves rather than an error.

So `reset()` calls `wait_until_interactive`, which probes from a snapshot at increasing offsets until
the player answers a button, then rewinds and replays only the waiting — the state you get back is
untouched. `info["intro_ticks"]` reports how long it took.

### Applying an action, and settling

A throw is not instantaneous. The block crosses the field destroying its own type as it goes, then
every column it emptied falls. Snapshotting straight after the button press would capture the middle
of that, so `apply` presses and then runs `settle`:

```python
load_state(pyboy, state.gb_state)      # rewind to the parent
pyboy.button(button, hold)             # press
pyboy.tick(max(ticks) + 1)
settle(pyboy)                          # ...then wait for the field to stop moving
```

Settled means no throw in flight (`$FFD2`/`$FFD3` both clear) **and** the field byte-identical for
`SETTLE_STABLE_TICKS` (6) frames, giving up after `SETTLE_MAX_TICKS` (900). Both are configurable:

```python
env = FlipullGBEnv(rom, settle_max_ticks=1800, settle_stable_ticks=8)
```

The in-flight flags matter because the field is briefly stable *while* a block is crossing it — the
block is a sprite, not a field cell, until it lands. Waiting on the field alone would call that
settled and snapshot a position mid-throw.

## Goal and terminal

- **Goal** (`is_goal`) — `state.stage_cleared`: the stage loaded with blocks and `blocks_remaining`
  is down to `clear_target`. Flipull finishes a stage at the `CLEAR` number rather than at zero; the
  HUD shows `BLOCK 25` against `CLEAR 09`.
- **Terminal** (`is_terminal`) — `state.out_of_time`: the clock at `$FFCB`/`$FFCC`/`$FFCE` reached
  zero.

Both read flags captured on the passed `state`, not live memory, which would describe whichever state
was applied last. Both are also **absorbing**: `successors` returns `[]` for them and `step` /
`simulate` hand the state straight back, so an action applied past the end of a stage cannot quietly
return a position from the next one.

Unlike Puzznic there is **no positional dead end** to detect. Puzznic's is sound and cheap — a type
with exactly one block left can never be matched — but Flipull's equivalent question is "can this
field still reach its target?", and answering it needs throw mechanics the memory map does not
describe. So a stage here is lost on time and nothing else, which means long plans are the case to
watch: a search that wanders will hit `is_terminal` eventually, but only after burning three minutes
of game clock.

## Rendering

`str(state)` draws the field, trimmed to its bounding box:

| Glyph | Cell |
|---|---|
| `#` | Border |
| `=` | Staircase |
| (space) | Outside the field |
| `1`–`4` | A block of that type |

with `held:` and `player:` as trailing lines. Those two are appended rather than drawn into the grid
because neither is on the field: the held block is in hand, and the player's position is only known
as a sprite Y that no row index here can honestly stand in for. They are in the text at all because
they are in `__eq__` — leave the player out and a rendered trace of a plan silently drops every move,
since two states that differ only by his row print identically.

`env.render()` prints the de-duplicated history of `step` calls and returns it as a list of strings,
the same as every other environment here. `state.save(rom, path)` writes a PNG of the state by
booting a throwaway emulator to it — it uses the null window, so unlike the Super Mario Land
environment it needs no display, but it does need Pillow.

## Planning

The generic `TreeSearchPlanner` from
[`super_mario_planner_gb.py`](../../planiverse/planners/super_mario_planner_gb.py) works against this
environment as-is; supply a heuristic and a cost function:

```python
from planiverse.planners.super_mario_planner_gb import TreeSearchPlanner

def heuristic(state):
    """How many blocks past the target are still standing."""
    return max(0, state.blocks_remaining - state.clear_target)

plan = TreeSearchPlanner().search(state, env, heuristic, lambda states, actions: len(actions))
env.validate(plan)
```

That heuristic is admissible-ish and weak: every throw removes at most a few blocks, but it says
nothing about *which* row to throw from, which is the entire decision. The piece worth writing is one
that looks down each row and asks what a throw from it would hit — the held block's type against the
first blocks it would meet — which is derivable from `state.field` and `state.held_block` without any
extra RAM reading. `column_blocks(field, col)` gives a column top-down, which is the direction it
collapses in.

The shape of the problem is worth knowing before you point a planner at it: three actions, a
three-minute clock, and effects that cascade. It is a much narrower tree than Puzznic's but a deeper
one — Puzznic's difficulty is in the branching, Flipull's is in the lookahead.

## What is and is not verified

The memory map behind this environment was derived behaviourally, and it is explicit about what that
did and did not establish. Because the confidence varies field by field, so does how much weight the
environment puts on each one. The map's own grading:

| Verified | Good | Moderate | Unverified |
|---|---|---|---|
| Field base, stride, 14 rows, left wall | Blocks tens digit | Initial block count | Stage number `$FFC6` |
| Cell encoding (`$80`/`$83`–`$86`/`$87`) | Timer tens and minutes | Clear target `$FFCF` | Upcoming-block queue at `$CA00` |
| Field count matching `BLOCK` at 25 and 24 | | Held block `$FFD4`, throw flags | |
| Per-column collapse on destruction | | In-flight X `$FFDF` | |
| Blocks ones `$FFC9`, timer seconds ones `$FFCB` | | | |

On top of that grading sits a second, blunter caveat: **the environment itself is unverified against
the cartridge.** The map's observations were made by recording RAM in a purpose-written emulator, not
by driving this code through PyBoy, so everything the environment adds on top of the map — booting,
finding the player, measuring the hold window, probing the throw button, settling a throw — has been
checked only against the synthetic cartridge. That cartridge was written to the map, so it cannot
disagree with it, which means it can confirm the code is self-consistent and cannot confirm the code
is right about Flipull.

Two consequences of the *map's* grading also run through the code:

- **`fix_index` refuses everything but 0**, because the stage number is the map's least-supported
  claim and a silently wrong stage is worse than a loud refusal.
- **`is_goal` leans on `$FFCF`**, which is graded moderate: it read `09` and never changed, which is
  consistent with a clear target but was never watched being *met*. The environment cross-checks what
  it can — `blocks_initial > 0` guards against calling an unloaded stage cleared — but the actual
  stage-complete transition has not been observed.

Also worth carrying: the **score was never located** (it advanced 0 → 100 with no candidate byte
isolated), so `step` returns blocks cleared in the reward slot; **only stage 1 was ever played**, and
only one block destroyed, so multi-block chains — the core of Flipull's scoring — were never
exercised; and the field's address calculator was **not found in code**, unlike Puzznic's at `0:29CE`.
The `$C840`/stride-32 geometry comes from the dump's structure, which is unambiguous but empirical.

## Known quirks and gaps

- **Never run against the real ROM.** See [What is and is not verified](#what-is-and-is-not-verified).
  Running `python -m planiverse.problems.retro_games.flipull_gb "Flipull (USA).gb"` against a real
  dump, and comparing what it prints to the map, is the cheapest way to close this.
- **One stage.** See [Stages](#stages). This is the next gap after that.
- **No score.** Not located in RAM. `step` returns `blocks_initial - blocks_remaining`.
- **No dead-end detection.** A field that can no longer reach its target is not terminal until the
  clock runs out.
- **Field width beyond column 5 is inferred.** Stage 1 uses columns 1–5; the full 16 comes from the
  ceiling and floor reading `$80` across that width. `bounding_box` derives the shape rather than
  assuming it, so a wider stage would decode correctly — but none has been seen.
- **`successors` shares one emulator.** All expansion runs on `self.pyboy`; correctness relies on
  each action reloading its parent's save-state first. Don't parallelise expansion over one env.
- **`render=True` is for watching, not planning.** It opens an SDL2 window and slows expansion. The
  constructor keeps that flag as `self.render_window`, because `self.render` is the method that
  prints the history.
- **`load_state` advances a frame** after restoring, mirroring the other two Game Boy environments.

## Testing without the cartridge

Everything above would be untestable in CI if it needed a copyrighted ROM, so the test suite builds
its own. [`tests/fake_flipull_rom.py`](../../tests/fake_flipull_rom.py) assembles a small homebrew
Game Boy cartridge — with [`tests/sm83.py`](../../tests/sm83.py), the same minimal SM83 assembler the
Puzznic test cartridge uses — that puts the same facts at the same addresses: a field at `$C840` with
stride `$20`, decimal-digit counters at `$FFC9`/`$FFCA`, a timer at `$FFCB`/`$FFCC`/`$FFCE`, a clear
target at `$FFCF`, a held block at `$FFD4` and throw flags at `$FFD2`/`$FFD3`.

Its stage 1 is the memory map's stage 1, byte for byte — the five rows recorded at `$C940`–`$C9C0` —
so `decode_field` is checked against a real observation rather than against itself.

It is **not** a Flipull clone: no chains, no score, no upcoming-block queue. A throw destroys the
first block of the held type or swaps with the first block of another, and a destroyed block collapses
its column, which is the entire rule set. It does model the three things that make driving a Game Boy
awkward, because otherwise the code for them would never run: **auto-repeat** on up and down, so
`measure_hold_window` has a real bound to find; a **stage intro** of 45 frames during which the field
is completely readable and every button is ignored, so `wait_until_interactive` has a real wait to
discover; and a **ticking clock**, so `is_terminal` is reachable.

What it exercises is the interface between the environment and a Game Boy — booting, decoding the
field, finding the player among the sprites, measuring a hold window, waiting for a throw to settle —
which is the part that otherwise could only be checked by hand.

The tests that need the real cartridge are opt-in:

```bash
PLANIVERSE_FLIPULL_ROM="/path/to/Flipull (USA).gb" poetry run pytest tests/test_flipull_gb.py
```

## Files

| Path | What |
|---|---|
| [`flipull_gb.py`](../../planiverse/problems/retro_games/flipull_gb.py) | `FlipullGBEnv`, `FlipullGBState`, `FlipullGBAction`, the calibration, and the RAM decoders |
| [`flipull-gb-memory-map.md`](flipull-gb-memory-map.md) | Every address, and how each was established |
| [`tests/test_flipull_gb.py`](../../tests/test_flipull_gb.py) | Tests, against the synthetic cartridge and the real one |
| [`tests/fake_flipull_rom.py`](../../tests/fake_flipull_rom.py) | The synthetic cartridge |
| [`tests/sm83.py`](../../tests/sm83.py) | The assembler that builds it |
