# Boxxle II (Game Boy)

Boxxle II played on the cartridge. This environment drives the real Game Boy ROM inside
[PyBoy](https://github.com/Baekalfen/PyBoy) and reads the board straight out of the console's
work RAM, so the transition function is the game's own code rather than a reconstruction of
its rules. States are emulator save-states, so search can branch by rewinding the machine.

The sibling [`boxxle2.py`](boxxle2.md) re-implements the same 120 levels in pure Python. Use
that one if you want a dependency-free benchmark; use this one if you want the cartridge's
actual behaviour. Unusually among the twins here, the two agree exactly — see
[Verified against the cartridge](#verified-against-the-cartridge).

Every address this environment reads is catalogued in the
[memory map](boxxle2-gb-memory-map.md), including how each one was established and how much of
it was checked against live RAM rather than read off a disassembly.

- **Class:** `Boxxle2GBEnv`
- **Import:** `from planiverse.environments.gameboy.boxxle2_gb import Boxxle2GBEnv, Boxxle2GBAction`
- **Source:** [`planiverse/environments/gameboy/boxxle2_gb.py`](../../planiverse/environments/gameboy/boxxle2_gb.py)
- **Dependencies:** `pyboy` + a `Boxxle II (USA, Europe).gb` ROM you supply (`pillow` for screenshots)

## The game

Sokoban. A warehouse keeper walks a grid one cell at a time and pushes boxes onto marked
squares. A box can only be pushed, never pulled; only one box moves at a time; and a box shoved
into a corner is there for good. Every level ships with exactly as many goals as boxes, so the
level is solved when every box is home.

The cartridge offers undo and restart. **Neither is in the action set**, and that is
deliberate: a Sokoban with undo has no dead ends, and dead ends are most of what makes it a
search problem.

## The ROM

**Not included, and cannot be.** Boxxle II is copyrighted; the repo ships no ROM and none will
be added. Supply your own legally-obtained dump and pass its path:

```python
env = Boxxle2GBEnv("Boxxle II (USA, Europe).gb")
```

Every address below was read from one specific dump:

| | |
|---|---|
| File | `Boxxle II (USA, Europe).gb`, 32,768 bytes |
| MD5 | `308abd707a48ee9d69c287d818469fd6` |
| Cartridge | ROM ONLY — no mapper, **no** cartridge RAM. All state is in work RAM. |

Because the addresses are revision-specific, the constructor hashes the file and raises a
`UserWarning` when it is not that dump. Pass `verify_rom=False` to silence it.

## Quickstart

```python
from planiverse.environments.gameboy.boxxle2_gb import Boxxle2GBEnv

env = Boxxle2GBEnv("Boxxle II (USA, Europe).gb", render=False)  # render=True opens a window
env.fix_index(0)                                                # choose the level *before* reset
state, info = env.reset()

print(state)
#  #######
#  # @ooo#
#  #   ####
# ###$    #
# #   #$# #
# # $ #   #
# #   #####
# #####

print(info)
# {'level_index': 0, 'level': '1-01', 'stage': 0, 'level_in_stage': 0,
#  'size': (9, 8), 'boxes': 3,
#  'calibration': Calibration(press_ticks=9, hold_window=(1, 18))}

for action, successor in env.successors(state):
    print(action, action.cost(), successor.boxes_home)
# left_for_9 1 0
# down_for_9 1 0
# right_for_9 1 0
```

`up` is missing from that list because there is a wall above the keeper, and `successors`
drops any action that leaves the position unchanged.

The alphabet a board is printed in is the usual one: `#` wall, `$` box, `o` goal, `*` box on a
goal, `@` keeper, `+` keeper on a goal, space floor. It is the same alphabet
[`boxxle2.py`](boxxle2.md) uses, so the two print identically.

## Levels

120 of them, twelve stages of ten, indexed 0–119.

```python
env.fix_index(37)
env.label_for()          # '4-08' — how the cartridge itself numbers it
```

They range from 6×5 with 3 boxes to 16×16 with 59.

### Reading the levels without booting anything

The level data is in the ROM, and the decoder is a pure function of the cartridge image:

```python
from planiverse.environments.gameboy.boxxle2_gb import read_levels, verify_level_table

levels = read_levels("Boxxle II (USA, Europe).gb")   # 120 tuples of ASCII rows
print("\n".join(levels[0]))

with open("Boxxle II (USA, Europe).gb", "rb") as handle:
    verify_level_table(handle.read())                # () — every record's size checks out
```

This is where [`boxxle2.py`](boxxle2.md)'s 120 boards came from. Nothing was transcribed by
hand, which is the difference between this pair of environments and the Puzznic pair, where
hand transcription put two quiet errors into the levels.

```bash
python -m planiverse.environments.gameboy.boxxle2_gb "Boxxle II (USA, Europe).gb" --dump
```

### How a level is selected

By hooking `LoadLevelHeader` at `$0F53` and writing the stage and level counters from inside
the hook body.

The obvious alternative — write `$C162` and `$C352` from outside, on a frame boundary — does
not work. The menu resets both counters and calls the loader within the same frame, so the
loader always reads the reset values. Hooking its entry puts the write between the reset at
the menu and the read at `$0F5D`.

The other alternative is the cartridge's own `PASSKEY` screen, which is what
[`puzznic_gb`](puzznic-gb.md) does. It is not used here because **the passkey encoding has not
been located in the ROM**: four characters from a 35-glyph alphabet, and no table anywhere in
the disassembly that maps them to a level. Third-party walkthroughs publish the passkeys if
you want to type one by hand; the hook reaches all 120 levels without them.

## Booting

Power-on to a playable board is roughly 1,370 frames — about 0.05 s of emulation, and about
0.17 s of wall clock including the calibration probe. The front end is three screens deep:

| `$C34E` | Screen | How it is passed |
|---|---|---|
| `$04` | title, "PUSH START KEY" | START |
| `$10` | MUSIC: BGM A / B / C | START |
| `$20` | MENU: PLAY / PASSKEY / CREATE | START, with PLAY already selected |
| `$06` | the story cutscene | waited out — about 970 frames, and no button shortens it |
| `$00` | playing | |

`boot` presses START only while `$C34E` names one of the first three. Tapping it on a timer
instead — which is what the Puzznic environment's fallback route does — is wrong here, because
**START during play opens the pause overlay**, so a stray press lands on exactly the screen
being aimed at.

`$00` also means "still booting" for the first sixty frames after power-on, so
`level_is_loaded` requires four things at once: the playing state, a board size the cartridge
could actually have loaded, a keeper somewhere on the board, and at least one box.

## State representation

`Boxxle2GBState` carries the emulator save-state plus what was read out of RAM when it was
snapshotted:

| Attribute | |
|---|---|
| `grid` | tuple of tuples of glyphs, `height` × `width` |
| `player` | `Position(row, col)` |
| `boxes`, `goals` | tuples of `Position` |
| `boxes_home` | how many boxes are on a goal |
| `width`, `height` | |
| `solved` | every box home |
| `stuck` | boxes wedged in a corner off a goal |
| `gb_state` | the PyBoy save-state, which is what lets search rewind |

Two states are equal when their grid and keeper match. Depth and history are not part of it,
so a position reached two ways compares equal and search closes.

### Where the board comes from

The cartridge decompresses each level into three plain byte planes in work RAM — goals at
`$C922`, boxes at `$CA8A`, walls at `$CBF2`, one byte per cell, 20 bytes per row — and the
environment reads them directly. No tile decoding, no sprite matching, no inference. The
keeper is a 16-bit offset into the same grid at `$C110:$C10F`.

That is why so much here is exact rather than approximate, and it is worth being clear that it
is a property of *this cartridge* rather than of the technique.

### Literals

```python
{'at(player, 1, 3)', 'boxes-home(0)',
 'at(box, 3, 3)', 'at(box, 4, 5)', 'at(box, 5, 2)',
 'goal(cell, 1, 4)', 'goal(cell, 1, 5)', 'goal(cell, 1, 6)'}
```

Goals are static and could have been left out; they are included because a width-based planner
partitions on atoms, and `boxes-home(n)` alone gives it a very coarse progress signal.
`goal-reached` and `terminal-state` are added when they apply.

Walls are *not* literals. They never change, there are hundreds of them on a large board, and
every one would be an atom that is true in every state — pure noise for a novelty measure.

### Consistency

```python
state.is_consistent()      # boxes and goals are equinumerous, as every level guarantees
```

The cartridge's own invariant, used as a cross-check: a position where the counts disagree
means one of the three plane addresses has drifted, or the board was read while the
level-cleared sequence was rewriting it.

## Actions

Four, one per direction, spelled `"button,ticks"`:

```python
env.get_actions()
# ['left,9', 'up,9', 'down,9', 'right,9']
```

Each costs 1. There is no modifier and no combination: on this cartridge a direction *is* the
move, and whether it walks or pushes is the game's decision, not the plan's.

### Ticks are measured, not chosen

Holding a direction for 1–19 frames moves the keeper exactly one cell. At 20 frames the d-pad
repeats and one press becomes two moves — which in a Sokoban is not a longer plan but a box
pushed somewhere nobody asked for, and a successor that does not match the action that
produced it.

That bound is not in the memory map, so `reset` measures it on the cartridge in hand:

```python
state, info = env.reset()
info["calibration"]
# Calibration(press_ticks=9, hold_window=(1, 18))
```

`measure_hold_window` finds a direction with two clear cells of room — one cell is not enough,
because with a wall in the way every hold looks identical and the window comes back as wide as
the probe — then walks the hold from 1 upwards, rewinding to the same state each time, and
reports the closed range that moves exactly one cell. `press_ticks` is the middle of it: far
enough above the low end to survive a frame of jitter in when the game samples the pad, far
enough below the high end not to trip the repeat.

Pass `calibrate=False` to skip the probe and use the documented default of 10.

```bash
python -m planiverse.environments.gameboy.boxxle2_gb "Boxxle II (USA, Europe).gb" --level 0
```

### Applying an action, and settling

`apply` rewinds the emulator to the parent state, presses the button, waits for the game to
stop moving, and snapshots. What "stop moving" means is the interesting part.

**The state is correct one frame after the press.** The plane buffers and the keeper offset
both update the frame the pad is read. **The keeper is not**: it slides one pixel per frame for
sixteen frames, and every press during the slide is ignored.

That slide is visible *only* in the shadow OAM at `$C000`. A byte-by-byte diff of
`$C0A0–$C460` across the animation finds nothing that is not also moving when the game is
idle — no counter, no flag, no direction byte. So `settle` watches the three planes, the keeper
offset **and** the 160-byte sprite buffer, and returns once all of them have held still for
three frames.

Watching only the planes drops roughly every second press, and the failure looks exactly like a
planner's action having no effect — which is the reason this is written down twice, here and in
the memory map.

Stability on its own is still not enough, and this is the part that took the longest to find.
The slide **pauses for a frame or two partway through**, and — worse — a hold long enough to
trip the d-pad's auto-repeat looks perfectly settled in the gap between the first move and the
second. So `settle` also refuses to return before frame 22, which is one past the frame the
repeat would have fired on. It costs nothing: the slide plus its stable frames already take
about that long.

```python
env = Boxxle2GBEnv(rom, settle_max_ticks=240, settle_stable_ticks=3)
```

### The two frames before the press

`apply` rewinds to the parent state before pressing, and `load_state` ticks exactly one frame
afterwards. **On some states that one frame is not enough**: the press lands before the main
loop next samples the pad, `ReadJoypad` never sees the edge, and the move is silently dropped.
The successor then comes back equal to its parent and `successors` deletes the action as a
no-op — a legal move disappearing from the search space with no error anywhere.

It is not every state, which is what makes it nasty: replaying a 500-move plan on the cartridge
failed at move 19, and that same move applied on its own worked perfectly. `Boxxle2GBAction`
therefore runs `LEAD_IN_TICKS = 2` idle frames after the rewind and before the press. The
constant lives here rather than in the shared `GBAction` because the other cartridges do not
need it.

With both fixes in place, all 40 stored plans clear their level on the ROM; before them, six
did not.

### Why settling stops early on a win

About 320 frames after the last box goes home, the cartridge switches to its
congratulation-and-replay sequence and **rewrites the plane buffers with something that is not
a Sokoban position** — boxes and goals scattered across the walls. A board snapshotted 30
frames after the winning push decodes as garbage.

So `settle` returns the instant every box is home, and `__advance__` treats a solved state as
absorbing: every action from it returns the state itself, and `successors`' self-loop filter
drops them all.

## Goal and terminal

```python
env.is_goal(state)        # every box on a goal
env.is_terminal(state)    # some box is wedged in a corner, off a goal
```

`is_terminal` is **sound and deliberately incomplete**. A box with a wall on one of its vertical
sides and one of its horizontal sides can never be moved again by anyone, so if it is not
already home the level is lost — that much is certain, and it is what `stuck_boxes` reports.
Positions that are dead for subtler reasons (a wall-hugging row with no goal on it, two boxes
frozen against each other) are not claimed, because a wrong `is_terminal` prunes a *solvable*
branch, and that is a much worse failure than letting a doomed one run.

This is one of the few cartridge environments that can answer the question at all. It can only
do so because the walls are sitting in work RAM in plain form.

`__score__`, which is what `step` reports, is `boxes_home`.

## Planning

```python
from planiverse.planners.width.bfws import BFWSSearch
from planiverse.planners.width.result import Budget

env = Boxxle2GBEnv(rom)
env.fix_index(0)
env.reset()

result = BFWSSearch(
    width=1,
    progress=lambda s: len(s.boxes) - s.boxes_home,
    heuristic=lambda s: sum(min(abs(b.row - g.row) + abs(b.col - g.col) for g in s.goals)
                            for b in s.boxes),
).solve(env, budget=Budget(max_expansions=20000))
```

Two things to expect. The branching factor is at most four and often two or three, which is
low; and the state space is nonetheless enormous, because Sokoban's difficulty is in the
ordering of the pushes rather than in the choice at any one step. Expect the early levels to
fall quickly and the later ones not to fall at all — the same shape the stored solutions have
(see below).

An expansion costs one save-state load, one button press and a settle: on the order of a
millisecond. `successors` on a typical position runs in about 50 ms.

## Verified against the cartridge

Three checks, all reproducible from a dump in hand.

**All 120 boards decode identically two ways.** Each level was loaded through the `$0F53` hook
and read out of the plane buffers, and compared cell for cell against the same level decoded
straight out of the ROM image. **0 mismatches.** That simultaneously verifies the level format,
the three plane addresses, the 20-byte stride, the keeper offset and the level-select hook.

**The Python twin matches move for move.** All 120 levels, twenty-five random moves each,
replayed on the cartridge and on [`boxxle2.py`](boxxle2.md) and compared after every one of the
3,000 moves. **0 divergences** — walls refusing a step, boxes refusing a push, boxes against
boxes, and boards in both cell-size modes.

**Stored plans still clear their levels.** `tests/data/boxxle2_gb_solutions.json` holds an
action sequence per solved level, and `tests/test_solutions.py` replays one on the ROM when a
ROM is present. Their provenance is in [`boxxle2.md`](boxxle2.md#where-the-solutions-came-from).

## Known quirks and gaps

**The passkey encoding is not decoded.** See [How a level is selected](#how-a-level-is-selected).
The consequence is that levels are reached by poking the loader rather than the way a player
would, so the cartridge's own progress counters are consistent but its save/restore path is
never exercised.

**A cleared level is absorbing rather than advancing.** On the cartridge, clearing a level
loads the next one over the top of it. Here the run ends, because the position a planner would
be handed next belongs to a different problem.

**`is_terminal` misses deadlocks.** By design; see [Goal and terminal](#goal-and-terminal).

**The story cutscene cannot be skipped**, so every `reset` pays about 970 frames for it. It is
still only 0.05 s of emulation, but it is the reason `boot_max_ticks` defaults to 4000.

## Testing without the cartridge

`tests/test_boxxle2_gb.py` runs in two tiers. The first needs no ROM at all: the level decoder,
the board decoder, the deadlock test and the settle predicate are pure functions over bytes,
and are tested against synthetic plane buffers. The second boots a real cartridge and is opt-in:

```bash
PLANIVERSE_BOXXLE2_ROM="/path/to/Boxxle II (USA, Europe).gb" poetry run pytest tests/test_boxxle2_gb.py
```

## Files

| | |
|---|---|
| [`planiverse/environments/gameboy/boxxle2_gb.py`](../../planiverse/environments/gameboy/boxxle2_gb.py) | the environment |
| [`planiverse/environments/gameboy/gb.py`](../../planiverse/environments/gameboy/gb.py) | shared PyBoy machinery |
| [`planiverse/environments/gameboy_py/boxxle2.py`](../../planiverse/environments/gameboy_py/boxxle2.py) | the pure-Python twin |
| [`docs/environments/boxxle2-gb-memory-map.md`](boxxle2-gb-memory-map.md) | every address, and how it was established |
| [`tests/test_boxxle2_gb.py`](../../tests/test_boxxle2_gb.py) | tests |
| [`tests/data/boxxle2_gb_solutions.json`](../../tests/data/boxxle2_gb_solutions.json) | plans replayed on the ROM |
