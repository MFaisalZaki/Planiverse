# Flipull (Game Boy)

This environment plays Flipull, Taito's *Plotting*, on the cartridge. It drives the US Game Boy
ROM inside [PyBoy](https://github.com/Baekalfen/PyBoy) and reads the block field out of the
console's work RAM, so the transition function is the game's own code rather than a reconstruction
of its rules. States are emulator save-states (i.e., the whole machine serialised to bytes), so
search can branch by rewinding the machine.

The player stands at the right of a wall of blocks holding one of them, and can move up and down
the twelve rows or throw. A throw sends the block left, blocks of its own type are destroyed, a
destroyed block drops its column, and something comes back into his hand. The stage is finished
once few enough blocks are left.

That description is deliberately vague about what a throw hits, because we have not established it
and this environment does not pretend to know. It asks the cartridge instead, which costs a state
expansion per throw: a planner cannot prune a throw without spending one. See [Actions](#actions).

The action set is unusually small for a Game Boy game, being a choice of row plus a throw, while
the consequences of a throw run several moves deep. Compare [Puzznic (Game Boy)](puzznic-gb.md),
where over 90% of every expansion is the cursor walking to the block you meant to move; here there
is no walking to speak of, and a branching factor of at most three.

- **Class:** `FlipullGBEnv`
- **Import:** `from planiverse.environments.gameboy.flipull_gb import FlipullGBEnv, FlipullGBAction`
- **Source:** [`planiverse/environments/gameboy/flipull_gb.py`](../../planiverse/environments/gameboy/flipull_gb.py)
- **Instances:** 32 stages, indices `0`–`31`
- **Dependencies:** `pyboy` plus a `Flipull (USA).gb` ROM you supply; `pillow` for screenshots
- **Memory map:** [`flipull-gb-memory-map.md`](flipull-gb-memory-map.md)
- **Sibling:** [`FlipullGame`](flipull.md) is the dependency-free Python environment

## The ROM

The repo ships no ROM, because Flipull is Taito's copyrighted work, so you supply your own legally
obtained dump and pass its path:

```python
env = FlipullGBEnv("Flipull (USA).gb")
```

Every address below was read from one dump:

| | |
|---|---|
| File | `Flipull (USA).gb`, 32,768 bytes |
| MD5 | `4fcc13db8144687e6b28200387aed25c` |
| Cartridge | No mapper; the whole ROM is flat at `$0000–$7FFF`, and there is no cartridge RAM |

With no cartridge RAM, all state lives in work RAM and HRAM, and Flipull leans on HRAM unusually
heavily, keeping nearly every counter there rather than in WRAM.

The addresses are revision-specific, so the constructor hashes the file and raises a `UserWarning`
when it is not that dump. Pass `verify_rom=False` to silence it.

## Quickstart

```python
from planiverse.environments.gameboy.flipull_gb import FlipullGBEnv, FlipullGBAction

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

ticks = info["calibration"].press_ticks                # measured; do not hard-code it
throw = info["calibration"].throw_button               # probed; likewise
trace = env.simulate([FlipullGBAction(f"up,{ticks}"), FlipullGBAction(f"{throw},{ticks}")])
print(env.is_goal(trace[-1]))
```

Or from the command line, which prints the field, the measurements and the action set:

```bash
python -m planiverse.environments.gameboy.flipull_gb "Flipull (USA).gb"
```

`reset()` boots the ROM, taps through the title screens until a field with blocks on it is up,
waits for the stage to start accepting input, calibrates, and snapshots.

## Stages

`Flipull (USA)` has 32 stages, and `fix_index(i)` selects one of them zero-based, so
`fix_index(7)` is stage 8.

```python
env.fix_index(7)
state, info = env.reset()
info["stage"], state.blocks_remaining, state.clear_target
# (8, 36, 7)
```

```bash
python -m planiverse.environments.gameboy.flipull_gb "Flipull (USA).gb" --stage 7
```

The cartridge's stage table fixes each stage's block total and CLEAR target, while the arrangement
of block types is drawn from an RNG seeded by boot timing, so a deterministic boot always sees the
same draw. `reset` idles a fixed seven frames per selected index before booting, which gives every
index its own draw while keeping `reset` repeatable: the same index always builds the same board,
and the 32 indices give 32 distinct fields.

### The stage number is two decimal digits

The game keeps the stage number as two decimal digits: `$FFC6` is the ones and `$FFC7` the tens.
`0:1673` is the advance:

```
1673  ld hl,$FFC6
1676  inc (hl)          ; stage++
1677  ld a,(hl)
1678  cp $0A            ; ...and at ten,
167A  jr nz,$1681
167C  ld a,$00
167E  ld (hl+),a        ; zero the ones, step to $FFC7
167F  jr $1676          ; and carry into the tens
```

Stage *N* is `$FFC7 = N // 10`, `$FFC6 = N % 10`, and both feed the HUD, which is why the
on-screen `STAGE` reads `23` and not `3`.

### The stage table

`0:2D55` is the loader. It turns the two digits into `10*tens + ones - 1`, indexes a table of
pointers at `$3A0E`, and copies the three bytes each entry points at into the HUD counters:

```
2D64  ldh a,($FFC6)     ; ones
2D66  add a,b           ;   + 10 * tens
2D67  sub $01           ; zero-based
2D69  rlca              ; two bytes per pointer
2D6A  ld hl,$3A0E       ; the table
...
2D7B  ldh ($FFCA),a     ; blocks, tens digit
2D80  ldh ($FFC9),a     ; blocks, ones digit
2D85  ldh ($FFCF),a     ; the CLEAR target
```

`read_stage_table(romfile)` walks that table rather than hard-coding it, the same way the Puzznic
environment reads its password table out of ROM instead of transcribing a guide. Each descriptor
is `[clear target, blocks ones, blocks tens]`, and the parse stops when an entry stops looking
like a stage, which it must, because a second, shorter table follows immediately at `$3A4E`.

```python
env.stages[7]
# Stage(number=8, clear_target=7, blocks=36)
```

That table is also what makes the selection self-checking. Every stage has a block total of 25, 30
or 36 and a target between 5 and 9, so `reset` compares what came up against what the ROM's table
says stage *N* holds, and raises if they disagree:

```
RuntimeError: selecting stage 8 did not take: the cartridge came up on stage 1 with 25
blocks and a target of 9, where the ROM's own table at $3A0E says stage 8 has 36 blocks
and a target of 7.
```

### How the selection is made

This matters because a wrong stage is not obviously wrong; it is a perfectly plausible board.

`reset` registers a PyBoy hook on the loader at `0:2D55` and writes the two digits as execution
arrives there, before the first instruction reads them. The hook is disarmed as soon as the stage
is up, so the cartridge behaves normally from that point: a cleared stage advances to the next
one, and a lost life reloads this one. Left armed, it would rewrite the number every time the
loader ran and the game could never leave the chosen stage.

This is not the same as [Puzznic's loader poke](puzznic-gb.md#rounds), which we keep there only as
a fallback. That one swaps a layout in underneath a game that still believes it is on round 1.
Here the two bytes written are the game's own stage number, and they are written before anything
reads them, so every place that reads them agrees afterwards, including the field builder and the
HUD.

`fix_index` refuses anything outside `0`–`31`. Past the end of the table the loader reads whatever
follows it in ROM: the second table at `$3A4E` begins with a pointer back to stage 1's descriptor,
so stage 33 would be stage 1.

## State

`FlipullGBState` carries the entire emulator save-state (`gb_state`, the bytes from
`pyboy.save_state`) plus the position scraped from RAM. The save-state is what makes branching
possible, since applying an action loads the parent's bytes back into the emulator first, so
siblings expand from an identical machine. The cost is a state that is kilobytes rather than
bytes.

| Field | Source |
|---|---|
| `field` | 14×16 tuple of cell values from `$C840`, stride `$20` per row |
| `blocks` | `(row, col, type)` for every cell holding a block |
| `staircase` | the fixed `$87` cells |
| `blocks_remaining` | `$FFCA`×10 + `$FFC9`: the live count |
| `blocks_initial` | `$FFC1`×10 + `$FFC0`: what the stage started with |
| `clear_target` | `$FFCF`: the `CLEAR` number |
| `timer_seconds` | `$FFCE`×60 + `$FFCC`×10 + `$FFCB` |
| `stage` | `$FFC7`×10 + `$FFC6` |
| `held_block` | the hand sprite's tile, as a 1–4 type, not `$FFD4` |
| `last_thrown` | `$FFD4`, the block previously in hand |
| `throws` | `$FFD2`/`$FFD3`, a count of completed throws |
| `player_y` | the player sprite's Y in the OAM buffer |
| `player_row` | that Y turned into a field row, via the measured row span |
| `row_blocks` | the blocks in the player's row, as `(col, type)` |
| `sprites` | the OAM buffer; a thrown block is a sprite, so this is part of the position |
| `stage_types` | the block types present when the stage loaded |
| `stage_cleared` | sampled here, read by `is_goal` |
| `out_of_time` | sampled here, read by `is_terminal` |

### Cell values

`field` holds the game's own codes:

| Value | Meaning |
|---|---|
| `$00` | Outside the field; the row stride is 32 bytes but only 16 columns carry meaning |
| `$80` | Border: the ceiling, the floor and the left wall |
| `$83`–`$86` | A playable block, four types |
| `$87` | The fixed staircase: structural, never clearable, excluded from the count |

`decode_blocks` counts only `$83`–`$86`, which is what makes the field agree with the HUD.

### Counters are decimal digits, one per byte

Flipull stores every counter as separate decimal digits, ones first, rather than as binary or as
the packed BCD Super Mario Land uses for its score. A count of 25 is `05` and `02` in adjacent
bytes, so searching RAM for 25 or `$19` finds nothing. `decode_digits(tens, ones)` is the whole of
it, and it is the reason `blocks_remaining` is not simply `pyboy.memory[...]`.

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

Terrain, meaning the border and the staircase, is deliberately not in the literals, because it is
static per stage and repeating it in every state would swamp the atoms that do change; read
`state.field` for it. Depth is absent for a different reason: with a step counter in the literals
no successor could ever equal its parent, and the self-loop filter in `successors` would be dead
code.

`at(player, ROW)` is a field row, and getting to that took a measurement. There is no row variable
in RAM, since `$C002` tracks vertical input (`89`/`8F`) rather than position, so all the player
offers directly is a sprite Y. Calibration walks him to the top and to the bottom, which pins the
lowest row he can stand on to the row just above the floor, and `row_for_y` counts up from there
in `row_pitch` steps.

Two states are equal when their field, held block and player row match. The player's row belongs
in that because a throw from a different row does something different: leave it out and `up` and
`down` become self-loops, `successors` filters them, and the environment offers a single action.

### Cross-checking

`state.is_consistent()` compares the number of `$83`–`$86` cells in the field against the HUD
counter at `$FFC9`/`$FFCA`. It is also what `stage_is_loaded` uses to decide the boot sequence has
arrived somewhere real.

## Actions

`FlipullGBAction` wraps a string of the form `"button,ticks"`.

| Action | Effect |
|---|---|
| `up` / `down` | Move the player one row |
| `a` (or whichever button `probe_throw_button` finds) | Throw the held block |

Cost is 1 for every action, so the branching factor is at most three: this is a game whose console
interface is already close to its planning interface. It is at most three because `successors`
filters what changes nothing, and two of the three are routinely filtered. A move into a wall does
nothing, since the player starts on the bottom row and `down` changes nothing until he has gone
up. And a throw that does not connect plays the whole animation, sending the block the width of
the field and back, while leaving the position exactly as it was, down to the cartridge's own
throw counter.

### What a throw hits is not modelled

We leave this unmodelled deliberately. The obvious rule, that the block meets the rightmost block
in the player's row and needs a match to do anything, is wrong. Driven across all twelve rows of
stage 1, every row connects, including rows above the wall that contain no blocks, so the block
plainly travels further than its own row.

Rather than ship a plausible guess, the environment asks the cartridge. `apply` presses the button
and reads back what happened, and `state.threw(parent)` answers "did that connect?" from the
cartridge's own counter rather than by inference. `row_blocks(field, row)` and
`column_blocks(field, col)` are exported for a planner that wants to work the rule out and build
its own model.

### Calibration

How long to hold a button has two bounds, and neither is in the memory map. Too short and the
press is never sampled; too long and auto-repeat fires, so one action moves the player two rows
and the state the planner gets back is not the one its action described. The failure is quiet,
because `FlipullGBAction("down,60")` looks like one press and reads like one action in a plan. So
`reset()` measures both bounds off the cartridge rather than trusting a constant:

```python
state, info = env.reset()
info["calibration"]
# Calibration(press_ticks=5, hold_window=(1, 10), throw_button='a', throw_ticks=5,
#             player_sprite=0, held_sprite=1, row_pitch=8, move_button='up',
#             row_span=(40, 128))
```

`Flipull (USA)` repeats on frame 11, so any hold of 10 frames or fewer moves one row and
`press_ticks` settles on 5, a far tighter window than Puzznic's `(1, 30)` and a good illustration
of why this is measured per cartridge rather than shared. `measure_hold_window` presses `down` for
1, 2, 3… frames and watches the player sprite move, returning the closed range of holds that move
him exactly one row: the lower end is where presses start registering, and the upper end is one
frame short of auto-repeat. `press_ticks` is the middle of that range, far enough from either edge
to survive a frame of jitter in when the game samples input.

Measuring a hold window needs something that moves, and Flipull gives nothing in RAM that says
where the player is, so `probe_sprites` finds him: it snapshots, taps `up`, taps `down`, and looks
at which OAM entries moved and which way. Two rules apply, and the cartridge taught us both.

First, a candidate must never move the wrong way: up must not increase Y, down must not decrease
it, and at least one has to do something. That is what rejects a free-running counter that happens
to sit in the OAM DMA buffer, and a probe taking the first thing that moved would track a counter
instead of the player, leaving every hold window measured afterwards meaningless.

Second, a candidate need not move both ways. `Flipull (USA)` starts the player on the bottom row,
where `down` is the floor, so requiring movement in both directions finds nobody. Calibration
records which direction actually worked as `move_button`, and measures the hold window with that
one.

Two sprites move together, the player and the block in his hand, and we tell them apart by
throwing: the block flies off across the field, the player does not. That also identifies the hand
sprite, whose tile is the only honest read of what is in hand.

The held block is the hand sprite's tile, which carries the field's `$83`–`$86` encoding. It
should not be confused with `$FFD4`, which looks like the held block and is not: `$FFD4` holds the
block *previously* in hand, lagging the hand by one throw and reading `$00` until the first throw
of a stage.

The stage's opening hand is the one case neither gives you, since before the first throw the hand
tile reads `$82`, which is not a block value at all. `probe_initial_hand` measures it by throwing
once from a snapshot, reading `$FFD4` back, and rewinding, so the throw never happened.

`probe_throw_button` presses each of `A` and `B` and watches for a throw in two independent
places: the completed-throw count at `$FFD2`/`$FFD3` going up, and the field changing, so a button
that only moves the player is not mistaken for one that throws. Both are checked after settling,
because neither happens until the block lands, some thirty frames after the press. On `Flipull
(USA)` both `A` and `B` throw, and the first that does is the one taken.

Calibration describes the game rather than the stage, so it runs once per environment and is
reused across resets. `FlipullGBEnv(rom, calibrate=False)` skips it and falls back to
`PRESS_TICKS` and `a`.

### Waiting for the stage to accept input

As in Puzznic, a stage can be entirely readable while its intro is still running, and a state
snapshotted in that window looks perfectly normal while answering no action, so a planner sees a
stage with no legal moves rather than an error.

`reset()` calls `wait_until_interactive`, which probes from a snapshot at increasing offsets until
the player answers a button, then rewinds and replays only the waiting, so the state handed back
is untouched. `info["intro_ticks"]` reports how long it took.

Each offset is tested by running the same frames twice, once pressing and once not, and asking
whether the button made any difference. Asking only whether the sprites changed is a different
question with a different answer, because on any cartridge whose sprites animate everything moves
every frame regardless of input. `Flipull (USA)` does answer at frame 0, so `intro_ticks` is `0`,
but it takes the two-run comparison to know that rather than to assume it.

### Applying an action

A throw is not instantaneous. The block crosses the field destroying its own type as it goes, and
then every column it emptied falls, so snapshotting straight after the button press would capture
the middle of that. `apply` presses and then runs `settle`:

```python
load_state(pyboy, state.gb_state)      # rewind to the parent
pyboy.button(button, hold)             # press
pyboy.tick(max(ticks) + 1)
settle(pyboy)                          # wait for the field to stop moving
```

Settled means the field and the sprites byte-identical for `SETTLE_STABLE_TICKS` (10) frames,
giving up after `SETTLE_MAX_TICKS` (900).

The sprites are the half that is easy to get wrong. A thrown block is a sprite until it lands, so
the field sits perfectly still for the thirty-odd frames the block spends crossing the screen.
Waiting on the field alone would call that settled and snapshot a position that has not happened
yet, making every successor equal its parent. Watching the sprites covers the whole cycle: the
flight out, the landing that changes the field and drops a column, and the arc back to the
player's hand. On `Flipull (USA)` that runs 61 frames from the bottom row and 169 from the worst
row we measured, which is why `SETTLE_MAX_TICKS` is 900 rather than 200. Both limits are
configurable:

```python
env = FlipullGBEnv(rom, settle_max_ticks=1800, settle_stable_ticks=16)
```

## Goal and terminal

- **Goal** (`is_goal`): `state.stage_cleared`, meaning the stage loaded with blocks and
  `blocks_remaining` is down to `clear_target`. Flipull finishes a stage at the `CLEAR` number
  rather than at zero; the HUD shows `BLOCK 25` against `CLEAR 09`.
- **Terminal** (`is_terminal`): `state.out_of_time`, meaning the clock at `$FFCB`/`$FFCC`/`$FFCE`
  reached zero. The clock starts at `3:00`.

Both read flags captured on the passed state rather than live memory. Both are absorbing:
`successors` returns `[]` and `step`/`simulate` hand the state straight back, so an action applied
past the end of a stage cannot return a position from the next one.

Unlike Puzznic there is no positional dead-end test. Puzznic's is sound and cheap, since a type
with exactly one block left can never be matched, but Flipull's equivalent question is whether
this field can still reach its target, and answering it needs the throw mechanics we have not
established. So a stage here is lost on time and nothing else, which means long plans are the case
to watch: a search that wanders hits `is_terminal` eventually, but only after burning three
minutes of game clock.

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

with `held:` and `player: row N` as trailing lines. Both are appended rather than drawn into the
grid because neither is on the field: the held block is in hand, and the player stands beside it.
They are in the text at all because both are in `__eq__`, and leaving the player out would make a
rendered trace silently drop every move, since two states differing only by his row print
identically.

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

env = FlipullGBEnv(romfile=os.environ["PLANIVERSE_FLIPULL_ROM"])
env.fix_index(0)
env.reset()

result = IteratedBFWS(max_width=1000, progress=measures.flipull_gb).solve(env)
trace = env.simulate(result.plan)

env.render_trace(trace, "flipull_gb.gif")                                   # animated
env.render_trace(trace, "flipull_gb.png", actions=result.plan, env=env,     # contact sheet
                 columns=4, max_states=12)
```

A cartridge frame is 640x576, so `max_states=` thins a long trace rather than tiling it in full.
See [docs/rendering.md](../rendering.md) for the other output formats.

## Notes and limits

- **What a throw hits is not modelled.** A planner gets throw outcomes by expanding, never by
  predicting, so it cannot prune a throw without spending a state on it.
- **Only the 32 main stages.** A second, shorter table at `$3A4E` is picked from by a routine that
  consults the RNG at `$FFAF`. `fix_index` reaches the main table only.
- **No score.** Not located in RAM. `step` returns `blocks_initial - blocks_remaining`.
- **No dead-end detection.** A field that can no longer reach its target is not terminal until the
  clock runs out.
- **Field width beyond column 5 is inferred.** Stage 1 uses columns 1–5; the full 16 comes from
  the ceiling and floor reading `$80` across that width. `bounding_box` derives the shape rather
  than assuming it.
- **`successors` shares one emulator.** All expansion runs on `self.pyboy`, and correctness relies
  on each action reloading its parent's save-state first. Do not parallelise expansion over one
  environment.
- **`render=True` is for watching, not planning.** It opens an SDL2 window and slows expansion.
  The constructor keeps that flag as `self.render_window`, because `self.render` is the method
  that prints the history.
- **`load_state` advances a frame** after restoring, so it is not a byte-exact restore.

## Testing without the cartridge

[`tests/fake_flipull_rom.py`](../../tests/fake_flipull_rom.py) assembles a small homebrew Game Boy
cartridge, using [`tests/sm83.py`](../../tests/sm83.py), that puts the same facts at the same
addresses: a field at `$C840` with stride `$20`, decimal-digit counters at `$FFC9`/`$FFCA`, a
timer at `$FFCB`/`$FFCC`/`$FFCE`, a clear target at `$FFCF`, a held block at `$FFD4` and throw
counters at `$FFD2`/`$FFD3`. Its stage 1 is the memory map's stage 1, byte for byte.

It is not a Flipull clone, and its throw rule (a different type in the first cell refuses the
throw) is a stand-in chosen to produce both outcomes, connecting and not. It reproduces the shape
of the game as the environment sees it: the player starts on the bottom row where `down` is a
wall, he and the block in his hand are two sprites that move together, a thrown block is a sprite
so the field sits still for the whole flight, and `$FFD2`/`$FFD3` count completed throws. It also
models auto-repeat on up and down (frame 17 here, against the cartridge's 11), a stage intro of 45
frames, and a ticking clock, so `measure_hold_window`, `wait_until_interactive` and `is_terminal`
all have something real to work against.

The tests that need the real cartridge are opt-in:

```bash
PLANIVERSE_FLIPULL_ROM="/path/to/Flipull (USA).gb" poetry run pytest tests/test_flipull_gb.py
```

## Files

| Path | What |
|---|---|
| [`flipull_gb.py`](../../planiverse/environments/gameboy/flipull_gb.py) | `FlipullGBEnv`, `FlipullGBState`, `FlipullGBAction`, the calibration and the RAM decoders |
| [`flipull-gb-memory-map.md`](flipull-gb-memory-map.md) | Every address the environment reads |
| [`tests/test_flipull_gb.py`](../../tests/test_flipull_gb.py) | Tests, against the synthetic cartridge and the real one |
| [`tests/fake_flipull_rom.py`](../../tests/fake_flipull_rom.py) | The synthetic cartridge |
| [`tests/sm83.py`](../../tests/sm83.py) | The assembler that builds it |
