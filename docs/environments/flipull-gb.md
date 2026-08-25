# Flipull (Game Boy)

Flipull — Taito's *Plotting* — played on the cartridge. This environment drives the real US Game Boy
ROM inside [PyBoy](https://github.com/Baekalfen/PyBoy) and reads the block field straight out of the
console's work RAM, so the transition function is the game's own code rather than a reconstruction of
its rules. States are emulator save-states, so search can branch by rewinding the machine.

The player stands at the right of a wall of blocks holding one of them, and can move up and down the
twelve rows or throw. A throw sends the block left; blocks of its own type are destroyed, a
destroyed block drops its column, and something comes back into his hand. The stage is finished when
few enough blocks are left.

That is deliberately vague about what a throw hits, because
[nobody has established it](#what-a-throw-hits-is-not-modelled) and this environment does not
pretend to know — it asks the cartridge instead.

The action set is unusually small for a Game Boy game — **pick a row, throw** — while the
consequences of a throw run several moves deep. Compare
[Puzznic (Game Boy)](puzznic-gb.md), where over 90% of every expansion is the cursor walking to the
block you meant to move. Here there is no walking to speak of: twelve rows to choose between, and a
branching factor of at most three.

Every address this environment reads is catalogued in the
[memory map](flipull-gb-memory-map.md), including how each one was established and how much of it is
verified against live RAM rather than read off a disassembly. That grading matters more here than it
does for Puzznic — see [What is and is not verified](#what-is-and-is-not-verified).

It has since been driven against the cartridge end to end, and that is where most of what follows
comes from. It is also where four things the map said turned out to mean something else — see
[What the cartridge corrected](#what-the-cartridge-corrected).

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

print(state)              # the field, trimmed to its bounding box. Block types are drawn
                          # afresh each playthrough, so the digits are not fixed:
# ################
# #====
# #===
# #==
# #=
# #
# #
# #
# #31231
# #33122
# #44431
# #12243
# #24431
# ################
# held: 1
# player: row 12

print(state.blocks_remaining, state.clear_target)      # 25 9
for action, successor in env.successors(state):
    print(action, action.cost(), successor.blocks_remaining, successor.held_block)
# up_for_5 1 25 1
# a_for_5 1 24 3
#   ...and no down: he starts on the bottom row, so it changes nothing and is filtered

ticks = info["calibration"].press_ticks                # measured; do not hard-code it
throw = info["calibration"].throw_button               # probed; likewise
trace = env.simulate([FlipullGBAction(f"up,{ticks}"), FlipullGBAction(f"{throw},{ticks}")])
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
| `held_block` | the hand sprite's tile, as a 1–4 type — **not** `$FFD4` |
| `last_thrown` | `$FFD4`, which is the block *previously* in hand |
| `throws` | `$FFD2`/`$FFD3`, a count of completed throws |
| `player_y` | the player sprite's Y in the OAM buffer |
| `player_row` | that Y turned into a field row, via the measured row span |
| `row_blocks` | the blocks in the player's row, as `(col, type)` |
| `sprites` | the OAM buffer — a thrown block is a sprite, so this is part of the position |
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
at(player, ROW)
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

`at(player, ROW)` is a **field row**, and getting to that took a measurement. There is no row
variable in RAM — `$C002` tracks vertical *input* (`89`/`8F`), not position — so all the player
directly offers is a sprite Y. Calibration walks him to the top and to the bottom, which pins the
lowest row he can stand on to the row just above the floor, and `row_for_y` counts up from there in
`row_pitch` steps. Checked against what a throw actually does: a destroyed block collapses its
column, so the field rows that change run from the top of the wall down to the row that was hit,
and that bottom row is the one this says he is on.

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

Three actions, so the branching factor is at most three: this is a game whose console interface is
already close to its planning interface. Contrast Puzznic, where the same button-level honesty costs
you a branching factor of four in which one expansion in twenty moves a block.

"At most", because `successors` filters what changes nothing, and two of the three are routinely
filtered:

- **A move into a wall.** The player starts on the bottom row, so `down` does nothing until he has
  gone up.
- **A throw that does not connect.** Some throws play the whole animation — the block flies the
  width of the field and arcs back — and leave the position exactly as it was, down to the
  cartridge's own throw counter.

### What a throw hits is not modelled

Deliberately. The obvious rule — the block meets the rightmost block in the player's row, and needs
a match to do anything — is **wrong**. Driven across all twelve rows of stage 1, *every* row
connects, including the rows above the wall that contain no blocks at all, so the block plainly
travels further than its own row. Some later throws then connect from no row at all.

Rather than ship a plausible guess, the environment does what it exists to do and asks the
cartridge: `apply` presses the button and reads back what happened. `state.threw(parent)` answers
"did that connect?" from the cartridge's own counter rather than by inference, and `row_blocks` is
exported for a planner that wants to build its own model.

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
# Calibration(press_ticks=5, hold_window=(1, 10), throw_button='a', throw_ticks=5,
#             player_sprite=0, held_sprite=1, row_pitch=8, move_button='up',
#             row_span=(40, 128))
```

Those are the real cartridge's numbers. **`Flipull (USA)` repeats on frame 11**, so any hold of 10
frames or fewer moves one row and `press_ticks` settles on 5 — a far tighter window than Puzznic's
`(1, 30)`, and a good illustration of why this is measured per cartridge rather than shared. The
old hard-coded default of 8 frames happens to sit inside it, which is worse than being outside:
every test passed while the number was picked for a different game.

`measure_hold_window` presses `down` for 1, 2, 3… frames and watches the player sprite move,
returning the closed range of holds that move him **exactly one row** — the lower end is where
presses start registering, the upper end is one frame short of auto-repeat. `press_ticks` is the
middle of that range, far enough from either edge to survive a frame of jitter in when the game
samples input.

Calibration describes the game, not the stage, so it runs once per environment and is reused across
resets. `FlipullGBEnv(rom, calibrate=False)` skips it and falls back to `PRESS_TICKS` and `a`.

### Which sprites are the player and his block

Measuring a hold window needs something that moves, and Flipull gives you nothing in RAM that says
where the player is. So `probe_sprites` finds him: it snapshots, taps `up`, taps `down`, and looks
at which OAM entries moved and which way.

Two rules, and the cartridge taught both of them.

**A candidate must never move the wrong way** — up must not increase Y, down must not decrease it,
and at least one has to do something. That is what rejects a free-running counter that happens to
sit in the OAM DMA buffer, which is exactly what the synthetic cartridge did on the first attempt,
parking a scratch variable at `$C010` where the environment reads sprites from. A probe that took
the first thing that moved would have tracked a counter instead of the player, and every hold
window measured afterwards would have been meaningless.

**But it must not have to move both ways.** That was the original rule, and it finds nobody on the
real cartridge: `Flipull (USA)` starts the player on the bottom row, where `down` is the floor. So
a blocked direction is allowed, and calibration records which direction actually worked as
`move_button` — the hold window is then measured with that one. (Puzznic's hold measurement had the
same bug, probing *left* into a wall and concluding the cursor repeated at frame 41.)

**Two sprites move together**, because the block in the player's hand travels with him. They are
told apart by throwing: the block flies off across the field, the player does not. That also
identifies the hand sprite, which is worth having on its own — its tile is the only honest read of
what is in hand.

### The held block is a sprite, not `$FFD4`

`$FFD4` looks like the held block and is not. Driven across five throws it holds the block
*previously* in hand — the one just thrown — lagging the hand by one throw, and reading `$00` until
the first throw of a stage. The hand sprite's tile is the live value, and it carries the field's own
`$83`–`$86` encoding.

The stage's *opening* hand is the one case neither gives you: before the first throw the tile reads
`$82`, which is not a block value at all. So `probe_initial_hand` measures it — throw once from a
snapshot, read `$FFD4` back, rewind — and the throw never happened. Working the first throw
backwards from what it did to the field gives the same answer.

### Which button throws

`probe_throw_button` presses each of `A` and `B` and watches for a throw in two independent places:
the completed-throw count at `$FFD2`/`$FFD3` going up, and the field changing. A button that only
moves the player is therefore not mistaken for one that throws. Both are checked *after* settling,
because neither happens until the block lands, some thirty frames after the press. On
`Flipull (USA)` both `A` and `B` throw; the first that does is the one taken.

### The stage has to start listening first

As in Puzznic, a stage can be entirely readable while its intro is still running — Puzznic ignores
input for 210 frames after its field is in memory. A state snapshotted in that window is the worst
kind of wrong: it looks perfectly normal and answers no action, so a planner sees a stage with no
legal moves rather than an error.

So `reset()` calls `wait_until_interactive`, which probes from a snapshot at increasing offsets until
the player answers a button, then rewinds and replays only the waiting — the state you get back is
untouched. `info["intro_ticks"]` reports how long it took.

Each offset is tested by running the same frames **twice, once pressing and once not**, and asking
whether the button made any difference. Asking only "did the sprites change" is a different question
with a different answer: on any cartridge whose sprites animate, everything moves every frame
regardless of input, so the first offset looks interactive and the wait is skipped. `Flipull (USA)`
really does answer at frame 0 — `intro_ticks` is `0` — but it takes the two-run comparison to know
that rather than to assume it.

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

Settled means the field **and the sprites** byte-identical for `SETTLE_STABLE_TICKS` (10) frames,
giving up after `SETTLE_MAX_TICKS` (900). Both are configurable:

```python
env = FlipullGBEnv(rom, settle_max_ticks=1800, settle_stable_ticks=16)
```

**The sprites are the half that is easy to get wrong, and this got it wrong.** A thrown block is a
sprite until it lands, so the field sits perfectly still for the thirty-odd frames the block spends
crossing the screen. Waiting on the field alone calls that settled and snapshots a position that has
not happened yet — the throw is still in the air, and every successor is the parent.

The throw flags do not rescue it, because they are not throw flags: see
[What the cartridge corrected](#what-the-cartridge-corrected). Watching the sprites covers the whole
cycle — the flight out, the landing that changes the field and drops a column, and the arc back to
the player's hand. On `Flipull (USA)` that runs 61 frames from the bottom row and 169 from the worst
row measured, which is why `SETTLE_MAX_TICKS` is 900 and not 200.

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
field still reach its target?", and answering it needs the throw mechanics
[nobody has established](#what-a-throw-hits-is-not-modelled). So a stage here is lost on time and
nothing else, which means long plans are the case to watch: a search that wanders will hit
`is_terminal` eventually, but only after burning three minutes of game clock.

Positions with no move left do exist. Thrown repeatedly from the starting row, stage 1 connects
three times and then stops: the animation still plays and nothing changes, so `successors` offers
only the moves, and no sequence of moves alone can change the field. The environment does not
recognise that as terminal, because recognising it in general is the same unsolved question.

## Rendering

`str(state)` draws the field, trimmed to its bounding box:

| Glyph | Cell |
|---|---|
| `#` | Border |
| `=` | Staircase |
| (space) | Outside the field |
| `1`–`4` | A block of that type |

with `held:` and `player: row N` as trailing lines. Those two are appended rather than drawn into the
grid because neither is on the field: the held block is in hand, and the player stands beside it.
They are in the text at all because they are in `__eq__` — leave the player out and a rendered trace
of a plan silently drops every move, since two states that differ only by his row print identically.

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

That heuristic is weak, and the reason is the interesting part: it says nothing about *which* row to
throw from, which is the entire decision — and neither can anything else here, because
[what a throw hits is not modelled](#what-a-throw-hits-is-not-modelled). A planner finds out by
expanding. `row_blocks(field, row)` and `column_blocks(field, col)` are exported for anyone who wants
to work the rule out and build a real evaluation on it; that, rather than a better distance
heuristic, is the piece worth writing.

The shape of the problem is worth knowing before you point a planner at it: at most three actions, a
three-minute clock, and effects that cascade. It is a much narrower tree than Puzznic's but a deeper
one — Puzznic's difficulty is in the branching, Flipull's is in the lookahead. Search does move: from
the opening position, expanding greedily on block count clears three blocks in three throws, each
replaying consistently.

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

What the map got right, it got very right: the field geometry, the cell encoding, the digit-per-byte
counters and the column collapse all hold exactly as written. Everything under **The throw** did not
— see the next section.

Two consequences of the map's grading still run through the code:

- **`fix_index` refuses everything but 0**, because the stage number is the map's least-supported
  claim and a silently wrong stage is worse than a loud refusal.
- **`is_goal` leans on `$FFCF`**, which is graded moderate: it read `09` and never changed, which is
  consistent with a clear target but was never watched being *met*. The environment cross-checks what
  it can — `blocks_initial > 0` guards against calling an unloaded stage cleared — but the actual
  stage-complete transition has not been observed.

Also worth carrying: the **score was never located** (it advanced 0 → 100 with no candidate byte
isolated), so `step` returns blocks cleared in the reward slot; and the field's address calculator was
**not found in code**, unlike Puzznic's at `0:29CE`. The `$C840`/stride-32 geometry comes from the
dump's structure, which is unambiguous but empirical — and it decodes every stage-1 position this
environment has produced, cross-checking against the HUD counter every time.

## What the cartridge corrected

The map was built by recording RAM in a purpose-written emulator across a short scripted session:
one stage, a handful of throws. That is enough to identify a counter and miss what a byte *means*.
Driving this code through PyBoy against the same dump turned up four:

| The map said | The cartridge says |
|---|---|
| `$FFD2`/`$FFD3` — throw state flags, both `00`→`01` on release | A **count of completed throws**: `0,0 → 1,1 → 2,2 → 3,3`. It stays `0` for the entire flight and rises only when the block lands, so it is the opposite of the in-flight marker it was taken for — and it does not move at all for a throw that changes nothing. The map saw the first increment and read it as a flag. |
| `$FFD4` — held / in-flight block type | The block **previously** in hand — the one just thrown. It lags the hand by one throw and reads `$00` until the first throw of a stage. |
| `$FFDF` — in-flight block X, falling steadily as the block travels | A free-running counter. It falls by 17 a frame, wrapping through zero, whether or not anything is in flight. The map sampled it during a throw and read the fall as travel. |
| `TIME 2:59` at stage start | The clock starts at `3:00`; the map read it a second in. |

The first of those was not a documentation nit. `settle` used it to decide a throw was still in the
air, so it believed every throw had landed the moment it was pressed, and — because the field does
not move while the block is a sprite — happily returned a snapshot from mid-flight.

Two more corrections were the environment's own assumptions rather than the map's:

- **The player starts on the bottom row**, where `down` does nothing. The sprite probe required
  movement in both directions and therefore found no player at all, which cascaded into no hold
  window, no held block, and a calibration full of fallbacks.
- **The player and his block are two sprites**, not one, so "the sprite that moved" was never unique.

None of these would have been caught by the synthetic cartridge as it was, because it had been
written to the map. It now reproduces the corrected behaviour instead — see
[Testing without the cartridge](#testing-without-the-cartridge).

## Known quirks and gaps

- **One stage.** See [Stages](#stages). This is the gap to close first — until it is closed, Flipull
  is one problem instance rather than a benchmark set.
- **What a throw hits is not modelled.** See [Actions](#actions). A planner gets throw outcomes by
  expanding, never by predicting, so it cannot prune a throw without spending a state on it.
- **No score.** Not located in RAM. `step` returns `blocks_initial - blocks_remaining`.
- **No dead-end detection.** A field that can no longer reach its target is not terminal until the
  clock runs out — and since the throw is not modelled, there is no cheap way to recognise one. It
  does happen: thrown repeatedly from the starting row, stage 1 connects three times and then stops,
  and from there only the clock ends the stage.
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

It is **not** a Flipull clone, and it does not pretend to know the rule that decides what a throw
hits, because [nobody does](#what-a-throw-hits-is-not-modelled). It reproduces the *shape* of the
game as the environment sees it, which after the cartridge session means all four of the things that
were wrong before:

- the player starts on the bottom row, where `down` is a wall
- he and the block in his hand are two sprites that move together
- a thrown block is a sprite, so the field sits still for the whole flight and again for the arc back
- `$FFD2`/`$FFD3` count completed throws — 0 in flight, and never advanced by a throw that changes
  nothing

plus the three that make driving a Game Boy awkward at all: **auto-repeat** on up and down, so
`measure_hold_window` has a real bound to find (frame 17 here, against the cartridge's 11 — different
on purpose, because the point is that the code measures rather than assumes); a **stage intro** of 45
frames during which the field is completely readable and every button is ignored, so
`wait_until_interactive` has a real wait to discover; and a **ticking clock**, so `is_terminal` is
reachable.

Its throw rule — a different type in the first cell refuses the throw — is a *stand-in* chosen to
produce both outcomes, connecting and not. It is explicitly not Flipull's rule.

What it exercises is the interface between the environment and a Game Boy — booting, decoding the
field, telling the player from the block in his hand, measuring a hold window, waiting for a throw to
actually finish, and watching a column collapse — which is the part that otherwise could only be
checked by hand.

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
