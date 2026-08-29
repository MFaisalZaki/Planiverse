"""Flipull on a Game Boy, driven through PyBoy.

Flipull (Taito's *Plotting*) sits the player at the right of a wall of blocks holding one
of them, free to move up and down the twelve rows or to throw. A throw sends the block left;
blocks of its own type are destroyed, a destroyed block drops its column, and something comes
back into his hand. The stage is finished when few enough blocks are left.

That is deliberately vague about what a throw hits, because nobody has established it: driven
across all twelve rows of stage 1, *every* row connects, empty ones included, so the block
travels further than its own row. This environment does not model it: it asks the cartridge,
which is the whole point of driving one. See `FlipullGBEnv.successors`.

The action set is small for a Game Boy game (pick a row, throw) while the *consequences* of
a throw run several moves deep, which is the shape a planner wants.

The addresses come from a reverse-engineering pass over `Flipull (USA).gb`
(MD5 `4fcc13db8144687e6b28200387aed25c`) and are documented in
`docs/environments/flipull-gb-memory-map.md`. That map was derived behaviourally (recording
WRAM and HRAM every frame against known on-screen values), so its confidence varies field by
field. The constants below carry that grading rather than pretending to it. Four of them
turned out to mean something else once this code was driven against the same dump; those
carry what the cartridge said instead.

    env = FlipullGBEnv("Flipull (USA).gb")
    state, info = env.reset()
    print(state)
    for action, successor in env.successors(state):
        ...
"""
import os
from collections import Counter, namedtuple

from planiverse.environments.gameboy.gb import (
    GBAction, GBEnv, GBState, create_pyboy, load_state, save_state,
    sprites as _oam_sprites,
)

#: The dump these addresses were read from. No mapper: the whole ROM is flat at $0000-$7FFF.
ROM_MD5 = "4fcc13db8144687e6b28200387aed25c"

# ------------------------------------------------------------------- the block field
# Verified: base, stride and the wall pattern at both ends. The address calculator was never
# found in code (unlike Puzznic's at `0:29CE`), so the geometry is read off the dump's
# structure: 14 evenly spaced rows with consistent borders.
FIELD_ADDR = 0xC840
ROW_STRIDE = 0x20                # 32 bytes per row...
FIELD_ROWS = 14
FIELD_COLS = 16                  # ...of which only the first 16 carry meaning
FIELD_BYTES = FIELD_ROWS * ROW_STRIDE

CELL_OUTSIDE = 0x00              # outside the field
CELL_BORDER = 0x80               # ceiling, floor, left wall
BLOCK_MIN = 0x83                 # $83-$86 are the four playable block types
BLOCK_MAX = 0x86
CELL_STAIRCASE = 0x87            # fixed structural diagonal; never clearable

CEILING_ROW = 0
FLOOR_ROW = FIELD_ROWS - 1
LEFT_WALL_COL = 0

CELL_GLYPHS = {CELL_OUTSIDE: " ", CELL_BORDER: "#", CELL_STAIRCASE: "="}

# --------------------------------------------------------------------- HRAM counters
# Flipull keeps its counters as separate decimal digits, ones first: not binary and not
# packed BCD. Searching for 25 or $19 finds nothing; the value is `05` and `02` in adjacent
# bytes. Nearly every counter is in HRAM rather than WRAM.
BLOCKS_ONES_ADDR = 0xFFC9        # verified: 05 -> 04 as the HUD went 25 -> 24
BLOCKS_TENS_ADDR = 0xFFCA        # good
INITIAL_ONES_ADDR = 0xFFC0       # moderate: the stage's *starting* total, not the live one
INITIAL_TENS_ADDR = 0xFFC1       # moderate
TIMER_SECONDS_ONES_ADDR = 0xFFCB  # verified: 09 -> 02 as the HUD went 2:59 -> 2:52
TIMER_SECONDS_TENS_ADDR = 0xFFCC  # good
TIMER_MINUTES_ADDR = 0xFFCE      # good
SUBSECOND_ADDR = 0xFFCD          # moderate: free-running tick
CLEAR_TARGET_ADDR = 0xFFCF       # moderate: the CLEAR number; never seen change

# ------------------------------------------------------------------------ the stage
# The map graded `$FFC6` unverified ("read 01 in stage 1, never seen change"), and it is
# half the answer. Disassembling the loader shows the stage number is kept as two decimal
# digits, and both are read straight into the HUD:
STAGE_ONES_ADDR = 0xFFC6         # verified: the on-screen STAGE number tracks these two
STAGE_TENS_ADDR = 0xFFC7         # across every stage this environment can select
STAGE_ADDR = STAGE_ONES_ADDR     # kept for the map's own name

def stage_digits(stage):
    """The two bytes the cartridge keeps a stage number in. `stage` is 1-based.

    `0:1673` is the advance: `inc ($FFC6)`, and at ten it zeroes that and carries into
    `$FFC7`. So the pair is a two-digit decimal number, tens then ones, and the loader's
    table index is `10*tens + ones - 1`.
    """
    return stage // 10, stage % 10


def stage_number(tens, ones):
    """The inverse of `stage_digits`."""
    return tens * 10 + ones


#: `0:2D55` is the loader. It reads both digits, indexes a table of 32 pointers at `$3A0E`,
#: and copies the three bytes each one points at into the HUD counters. Hooking it is how
#: `fix_index` selects a stage; see `FlipullGBEnv.fix_index`.
STAGE_LOADER_ADDR = 0x2D55
STAGE_TABLE_ADDR = 0x3A0E
STAGE_COUNT = 32
STAGE_DESCRIPTOR_BYTES = 3       # clear target, blocks ones, blocks tens
ROM_BANK = 0                     # no mapper: the whole ROM is flat at $0000-$7FFF

# ----------------------------------------------------------------------- the throw
# Three of the map's four entries here turned out to mean something else when the cartridge
# was actually driven. What they really do, measured frame by frame across a throw:
THROW_COUNT_ADDRS = (0xFFD2, 0xFFD3)  # a COUNT of completed throws, not a pair of flags:
                                      # 0,0 -> 1,1 -> 2,2 -> 3,3, and it does not advance
                                      # for a throw that changes nothing. The map read the
                                      # first increment as `00 -> 01 on release`. Note that
                                      # it stays 0 for the whole flight and rises only when
                                      # the block lands, so it is the opposite of the
                                      # in-flight marker it was taken for.
LAST_THROWN_ADDR = 0xFFD4        # the block *previously* in hand, i.e. the one just thrown;
                                 # it lags the held-block sprite by exactly one throw, and
                                 # reads $00 until the first throw. Not what is in hand now;
                                 # `held_sprite` is.
FREE_COUNTER_ADDR = 0xFFDF       # the map read this as the in-flight X because it falls
                                 # steadily. It does fall, by 17 a frame, wrapping through
                                 # zero, whether or not anything is in flight. A counter.
INFLIGHT_Y_ADDR = 0xFFDE         # low, and unused here for the same reason
PLAYER_STATE_ADDR = 0xC002       # tracked vertical input (89/8F); not a row index

#: OAM DMA source, confirmed on the cartridge: `$C000`-`$C09F` mirrors hardware OAM at
#: `$FE00` byte for byte. The player and the block in his hand are both sprites, and which
#: slots they occupy is discovered at runtime; see `probe_sprites`.
OAM_BUFFER_ADDR = 0xC000
OAM_BUFFER_BYTES = 160

# --------------------------------------------------------------------------- driving
PRESS_TICKS = 5                  # fallback only; `calibrate` measures the real window.
                                 # Flipull (USA) repeats on frame 11, so its window is
                                 # (1, 10) and the middle of it is 5, a far tighter bound
                                 # than Puzznic's (1, 30).
PROBE_MAX_HOLD = 60
SETTLE_MAX_TICKS = 900           # the longest throw measured took 169 frames: the block
SETTLE_STABLE_TICKS = 10         # crosses the field, lands, and then arcs back to the hand
BOOT_MAX_TICKS = 2400

#: Extra idle frames before boot, per selected stage index. The cartridge draws each
#: stage's block *arrangement* from an RNG seeded by boot timing; its stage table fixes
#: only the block total and the CLEAR target. A deterministic boot therefore always sees
#: the same draw, and without this delay the 32 stages collapse to three distinct boards
#: (one per block total). Seven frames per index gives every index its own draw, still
#: deterministically: the same index always builds the same board, so `reset` stays
#: repeatable. Verified against the cartridge: 32 indices, 32 distinct fields.
SEED_TICKS_PER_INDEX = 7
BOOT_PRESS_EVERY = 12
INTRO_MAX_TICKS = 900
INTRO_STEP_TICKS = 10

THROW_BUTTONS = ("a", "b")       # which one throws is probed, not assumed
MOVE_BUTTONS = ("up", "down")

action_cost_map = {"up": 1, "down": 1, "a": 1, "b": 1, "nop": 0}

Block = namedtuple("Block", ["row", "col", "type"])
Calibration = namedtuple("Calibration", ["press_ticks", "hold_window", "throw_button",
                                         "throw_ticks", "player_sprite", "held_sprite",
                                         "row_pitch", "move_button", "row_span"],
                         defaults=(None,) * 9)


def button_actions(calibration=None):
    """The buttons this game has: pick a row, throw."""
    ticks = (calibration.press_ticks if calibration and calibration.press_ticks
             else PRESS_TICKS)
    throw = (calibration.throw_button if calibration and calibration.throw_button
             else THROW_BUTTONS[0])
    throw_ticks = (calibration.throw_ticks if calibration and calibration.throw_ticks
                   else ticks)
    return [f"{button},{ticks}" for button in MOVE_BUTTONS] + [f"{throw},{throw_ticks}"]


action_list = button_actions()


# --------------------------------------------------------------------- pure decoding
# Split out from the emulator so they can be tested against synthetic RAM.

def decode_field(raw):
    """The bytes at `$C840` as a 14x16 field, dropping the unused half of each row."""
    return tuple(tuple(raw[ROW_STRIDE * row + col] for col in range(FIELD_COLS))
                 for row in range(FIELD_ROWS))


def is_playable(value):
    """`$83`-`$86` are the four block types. `$87` is the fixed staircase, not a block."""
    return BLOCK_MIN <= value <= BLOCK_MAX


def decode_blocks(field):
    """Every playable block on the field, top-left first."""
    return tuple(Block(row, col, field[row][col] - BLOCK_MIN + 1)
                 for row in range(FIELD_ROWS) for col in range(FIELD_COLS)
                 if is_playable(field[row][col]))


def decode_digits(tens, ones):
    """Flipull stores counters as separate decimal digits, ones first."""
    return tens * 10 + ones


def decode_timer(minutes, seconds_tens, seconds_ones):
    return minutes * 60 + seconds_tens * 10 + seconds_ones


def block_counts(blocks):
    return Counter(block.type for block in blocks)


def row_for_y(y, row_pitch, bottom_y):
    """Which field row a player sprite Y stands on.

    Anchored at the bottom rather than the top: the lowest row the player can reach is the
    one just above the floor, and every row above it is `row_pitch` pixels further up. Both
    anchors are measured on the cartridge, so nothing here is a magic screen coordinate.

    Verified by throwing from all twelve reachable rows of `Flipull (USA)` stage 1 and
    watching which field rows changed: a destroyed block collapses its column, so the rows
    that move are exactly row 8 down to the row that was hit.
    """
    if y is None or not row_pitch or bottom_y is None:
        return None
    return (FLOOR_ROW - 1) - (bottom_y - y) // row_pitch


Stage = namedtuple("Stage", ["number", "clear_target", "blocks"])


def row_blocks(field, row):
    """A row left to right, as `(col, type)` for the blocks in it.

    Offered as a convenience for a planner reasoning about a row, not as a model of what a
    throw does; see `FlipullGBEnv.successors` for why there is no such model here.
    """
    if row is None or not 0 <= row < FIELD_ROWS:
        return ()
    return tuple((col, field[row][col] - BLOCK_MIN + 1) for col in range(FIELD_COLS)
                 if is_playable(field[row][col]))


def column_blocks(field, col):
    """A column top-down, which is the direction it collapses in."""
    return tuple(field[row][col] for row in range(FIELD_ROWS))


def bounding_box(field):
    """The rows and columns that are part of the field at all.

    Only stage 1 was ever observed and it uses columns 1-5, so the full width is inferred
    from the ceiling and floor reading `$80` across 16 columns. Derive it rather than assume.
    """
    used = [(row, col) for row in range(FIELD_ROWS) for col in range(FIELD_COLS)
            if field[row][col] != CELL_OUTSIDE]
    if not used:
        return (0, -1), (0, -1)
    rows = [row for row, _ in used]
    cols = [col for _, col in used]
    return (min(rows), max(rows)), (min(cols), max(cols))


# ------------------------------------------------------------------------- emulation
# `create_pyboy`, `save_state` and `load_state` come from the shared `gb` module.

def read_field(pyboy):
    return pyboy.memory[FIELD_ADDR:FIELD_ADDR + FIELD_BYTES]


def read_blocks_remaining(pyboy):
    return decode_digits(pyboy.memory[BLOCKS_TENS_ADDR], pyboy.memory[BLOCKS_ONES_ADDR])


def read_timer(pyboy):
    return decode_timer(pyboy.memory[TIMER_MINUTES_ADDR],
                        pyboy.memory[TIMER_SECONDS_TENS_ADDR],
                        pyboy.memory[TIMER_SECONDS_ONES_ADDR])


def throw_count(pyboy):
    """How many throws have completed.

    Not a flag, and not an in-flight marker. Driving the cartridge shows `$FFD2`/`$FFD3`
    counting `0,0 -> 1,1 -> 2,2 -> 3,3`, staying put for a throw that changes nothing, and
    (the part that matters) remaining `0` for the whole flight, rising only when the block
    lands. Treating it as "a throw is in progress" gets the truth exactly backwards.
    """
    return max(pyboy.memory[addr] for addr in THROW_COUNT_ADDRS)


def sprites(pyboy):
    """The OAM DMA buffer, as `(y, x, tile)` for every sprite slot, parked ones included.

    Unlike Puzznic's, this keeps all forty entries: `probe_sprites` identifies the player
    by *index* into this list, so the indices must be stable across frames.
    """
    return _oam_sprites(pyboy, OAM_BUFFER_ADDR)


def stage_is_loaded(pyboy):
    """True once a field with blocks on it is up and agrees with the HUD counter."""
    remaining = read_blocks_remaining(pyboy)
    if remaining == 0:
        return False
    blocks = decode_blocks(decode_field(read_field(pyboy)))
    # The field and the HUD counter have to agree, which is the check the memory map
    # itself leans on: 25 cells in $83-$86 against BLOCK 25.
    return len(blocks) == remaining


def settle(pyboy, render=False, max_ticks=SETTLE_MAX_TICKS, stable_ticks=SETTLE_STABLE_TICKS):
    """Run the emulator until the game stops moving.

    Settled means the field **and the sprites** byte-identical for `stable_ticks` frames.

    Both halves are load-bearing, and the sprites are the half that is easy to miss: a
    thrown block is a *sprite* until it lands, so the field sits still for the
    thirty-odd frames it spends crossing the screen. Waiting on the field alone calls that
    settled and snapshots a position mid-throw. This function used to do that,
    because it also trusted `$FFD2`/`$FFD3` to say a throw was in flight when in fact they
    stay `0` until it lands.

    Watching the sprites covers the whole cycle: the flight out, the landing that changes
    the field and drops a column, and the arc back to the player's hand. On `Flipull (USA)`
    that runs 61 frames from the bottom row and 169 from the worst row measured.
    """
    previous, stable = None, 0
    for _ in range(max_ticks):
        pyboy.tick(1, render)
        now = (read_field(pyboy), tuple(sprites(pyboy)))
        if now == previous:
            stable += 1
            if stable >= stable_ticks:
                return True
        else:
            previous, stable = now, 0
    return False


def _tap(pyboy, button, hold, render=False, gap=2):
    pyboy.button(button, hold)
    pyboy.tick(hold + gap, render)


def probe_sprites(pyboy, render=False, press_ticks=PRESS_TICKS, throw_button=None,
                  **settle_kwargs):
    """Which OAM entries are the player and the block in his hand, and which way he can move.

    The memory map has no row index for the player (`$C002` tracks vertical *input*, not
    position), so the only thing that says where he is, is his sprite. Which sprite that is
    gets discovered rather than assumed, so it survives a different stage or revision.

    Two things the cartridge taught this probe, neither of which the synthetic one could:

    **The player may start against a wall.** `Flipull (USA)` opens with him on the bottom
    row, where `down` moves nothing at all. Demanding that a candidate move *both* ways
    (which is what this did) finds nobody. So a blocked direction is allowed, and what is
    required instead is that no candidate ever move the *wrong* way: up must not increase Y
    and down must not decrease it, and at least one of them has to do something. That still
    rejects a free-running counter that happens to live in the OAM buffer, which is the
    thing the both-ways rule was there to reject in the first place. (Puzznic's hold
    measurement had this same bug, probing left into a wall.)

    **More than one sprite moves.** The player and the block in his hand travel together, so
    there are two candidates, not one. They are told apart by throwing: the held block flies
    off across the field, the player does not move. That also identifies the held-block
    sprite, which is worth having: its tile is the only honest read of what is in hand.

    Returns `(player, held, move_button)`, any of which may be `None`.
    """
    snapshot = save_state(pyboy)
    before = sprites(pyboy)
    deltas, moved_by = {}, {}
    for button in MOVE_BUTTONS:
        load_state(pyboy, snapshot, render)
        _tap(pyboy, button, press_ticks, render)
        settle(pyboy, render, **settle_kwargs)
        after = sprites(pyboy)
        deltas[button] = [now[0] - was[0] for was, now in zip(before, after)]
        moved_by[button] = after != before
    load_state(pyboy, snapshot, render)

    up, down = deltas[MOVE_BUTTONS[0]], deltas[MOVE_BUTTONS[1]]
    candidates = [i for i in range(len(before))
                  if before[i][0] and up[i] <= 0 <= down[i] and (up[i] or down[i])]
    if not candidates:
        return None, None, None

    # Whichever direction actually got the player off the spot he starts on is the one the
    # hold window can be measured with; the other may be into a wall.
    move_button = next((b for b in MOVE_BUTTONS if moved_by[b]), MOVE_BUTTONS[0])
    if len(candidates) == 1:
        return candidates[0], None, move_button

    # Throw, and see who stays put. The one that leaves is the block.
    load_state(pyboy, snapshot, render)
    _tap(pyboy, throw_button or THROW_BUTTONS[0], press_ticks, render)
    pyboy.tick(20, render)
    midflight = sprites(pyboy)
    load_state(pyboy, snapshot, render)
    stayed = [i for i in candidates if midflight[i] == before[i]]
    left = [i for i in candidates if midflight[i] != before[i]]
    if len(stayed) == 1:
        return stayed[0], (left[0] if len(left) == 1 else None), move_button
    return None, None, move_button


def player_y(pyboy, sprite):
    """The player sprite's Y, or None when we never worked out which sprite he is."""
    return None if sprite is None else sprites(pyboy)[sprite][0]


def measure_row_span(pyboy, state_bytes, sprite, render=False, press_ticks=PRESS_TICKS,
                     max_rows=FIELD_ROWS + 2, **settle_kwargs):
    """The highest and lowest Y the player can stand on, walked out on the cartridge.

    Needed to turn his sprite Y into a field row: the bottom of the walk is the row just
    above the floor. Twelve rows on `Flipull (USA)`, which is exactly the field's 14 minus
    the ceiling and the floor.
    """
    if sprite is None:
        return None
    span = []
    for button in MOVE_BUTTONS:
        load_state(pyboy, state_bytes, render)
        y = player_y(pyboy, sprite)
        for _ in range(max_rows):
            _tap(pyboy, button, press_ticks, render)
            settle(pyboy, render, **settle_kwargs)
            moved = player_y(pyboy, sprite)
            if moved == y:
                break
            y = moved
        span.append(y)
    load_state(pyboy, state_bytes, render)
    return min(span), max(span)


def probe_initial_hand(pyboy, state_bytes, render=False, press_ticks=PRESS_TICKS,
                       throw_button=None, **settle_kwargs):
    """The block type in the player's hand at stage start.

    Every later state can read this off the hand sprite's tile, which carries the field's own
    `$83`-`$86` encoding. The *first* one cannot: at stage start the tile reads `$82`, which
    is not a block value at all, and inventing a type for it would be a guess.

    So it is measured instead. `$FFD4` holds the block *previously* in hand, so throwing once
    and reading it back names the block that was there before the throw, after which the
    probe rewinds and the throw never happened. Confirmed by working the first throw
    backwards from what it did to the field: it destroyed a `$83` and left a `$83` behind in
    the cell it swapped with, which is `$FFD4`'s answer too.
    """
    load_state(pyboy, state_bytes, render)
    _tap(pyboy, throw_button or THROW_BUTTONS[0], press_ticks, render)
    settle(pyboy, render, **settle_kwargs)
    thrown = pyboy.memory[LAST_THROWN_ADDR]
    load_state(pyboy, state_bytes, render)
    return thrown - BLOCK_MIN + 1 if is_playable(thrown) else None


def measure_hold_window(pyboy, state_bytes, sprite, render=False, max_hold=PROBE_MAX_HOLD,
                        button=None, **settle_kwargs):
    """The closed range of holds that move the player exactly one row.

    Same bound as every other Game Boy menu: too short and the press is never sampled, too
    long and auto-repeat moves two rows, so the state handed back is not the one the action
    described. Measured off the cartridge rather than guessed.

    `button` is the direction the player can actually move in from here, passed in rather
    than hard-coded, because on `Flipull (USA)` he starts on the bottom row and this used to
    probe `down` into the floor and conclude he never moved.
    """
    if sprite is None:
        return None, None
    step = None
    low = None
    for hold in range(1, max_hold + 1):
        load_state(pyboy, state_bytes, render)
        origin = player_y(pyboy, sprite)
        _tap(pyboy, button or MOVE_BUTTONS[0], hold, render)
        settle(pyboy, render, **settle_kwargs)
        moved = abs(player_y(pyboy, sprite) - origin)
        if moved == 0:
            continue
        if step is None:
            step, low = moved, hold
        elif moved >= 2 * step:
            return (low, hold - 1), step
    load_state(pyboy, state_bytes, render)
    return (None if low is None else (low, max_hold)), step


def probe_throw_button(pyboy, state_bytes, render=False, press_ticks=PRESS_TICKS,
                       **settle_kwargs):
    """Which button throws, found by pressing each and watching for a throw.

    A throw shows up in two independent places (the completed-throw count at
    `$FFD2`/`$FFD3` goes up, and the field changes), so a button that only moves the player
    is not mistaken for one. Both are checked *after* settling, because neither happens
    until the block lands, some thirty frames after the press.

    On `Flipull (USA)` both `A` and `B` throw; the first that does is the one taken.
    """
    for button in THROW_BUTTONS:
        load_state(pyboy, state_bytes, render)
        before, throws_before = read_field(pyboy), throw_count(pyboy)
        _tap(pyboy, button, press_ticks, render)
        settle(pyboy, render, **settle_kwargs)
        if throw_count(pyboy) != throws_before or read_field(pyboy) != before:
            load_state(pyboy, state_bytes, render)
            return button
    load_state(pyboy, state_bytes, render)
    return None


def calibrate(pyboy, state_bytes, render=False, max_hold=PROBE_MAX_HOLD, **settle_kwargs):
    """Measure how this cartridge wants to be driven. A property of the game, so once is
    enough: every probe rewinds to `state_bytes`, so nothing it does survives."""
    load_state(pyboy, state_bytes, render)
    # The throw button first: it needs nothing else, and telling the player apart from the
    # block in his hand needs a throw to separate them.
    throw = probe_throw_button(pyboy, state_bytes, render, PRESS_TICKS, **settle_kwargs)
    player, held, move_button = probe_sprites(pyboy, render, PRESS_TICKS, throw,
                                              **settle_kwargs)
    window, row_pitch = measure_hold_window(pyboy, state_bytes, player, render, max_hold,
                                            move_button, **settle_kwargs)
    press_ticks = (window[0] + window[1]) // 2 if window else PRESS_TICKS
    span = measure_row_span(pyboy, state_bytes, player, render, press_ticks,
                            **settle_kwargs)
    load_state(pyboy, state_bytes, render)
    return Calibration(press_ticks=press_ticks, hold_window=window,
                       throw_button=throw or THROW_BUTTONS[0], throw_ticks=press_ticks,
                       player_sprite=player, held_sprite=held, row_pitch=row_pitch,
                       move_button=move_button, row_span=span)


def wait_until_interactive(pyboy, render=False, max_ticks=INTRO_MAX_TICKS,
                           step=INTRO_STEP_TICKS, press_ticks=PRESS_TICKS):
    """Advance until the stage accepts a button, and report how many frames it took.

    A stage can be entirely readable while its intro is still running (Puzznic ignores input
    for 210 frames after its field is in memory), and a state snapshotted in that window
    looks normal and answers nothing. Probes from a snapshot at increasing offsets, then
    rewinds and replays only the waiting, so the field is untouched.

    Each offset is tested by running the same frames **twice, once pressing and once not**,
    and asking whether the button made any difference. Asking only "did the sprites change"
    is not the same question and gives the wrong answer on any cartridge whose sprites
    animate: everything moves every frame regardless of input, so the very first offset
    looks interactive and this returns 0 whether or not the game is listening.
    `Flipull (USA)` really does answer at 0, but it takes the two-run comparison to know
    that rather than to assume it.
    """
    snapshot = save_state(pyboy)
    for waited in range(0, max_ticks, step):
        load_state(pyboy, snapshot, render)
        if waited:
            pyboy.tick(waited, render)
        offset = save_state(pyboy)
        pyboy.tick(press_ticks + 14, render)
        idle = sprites(pyboy)
        for button in MOVE_BUTTONS + THROW_BUTTONS[:1]:
            load_state(pyboy, offset, render)
            _tap(pyboy, button, press_ticks, render)
            pyboy.tick(12, render)
            if sprites(pyboy) != idle:
                load_state(pyboy, snapshot, render)
                if waited:
                    pyboy.tick(waited, render)
                return waited
    load_state(pyboy, snapshot, render)
    return None


def boot(pyboy, render=False, max_ticks=BOOT_MAX_TICKS, press_every=BOOT_PRESS_EVERY):
    """Tap through the title screens until a stage is on the field."""
    for frame in range(0, max_ticks, press_every):
        pyboy.button("start" if (frame // press_every) % 2 == 0 else "a", 4)
        pyboy.tick(press_every, render)
        if stage_is_loaded(pyboy):
            return True
    return False


# ----------------------------------------------------------------------------- state

class FlipullGBState(GBState):
    """A settled position: the emulator save-state plus the facts read out of RAM."""

    def __init__(self, pyboy, depth, calibration=None, stage_types=None, held_hint=None):
        super().__init__(pyboy, depth)
        self.calibration = calibration
        self.__update__(pyboy, stage_types, held_hint)

    def __update__(self, pyboy, stage_types, held_hint=None):
        raw = read_field(pyboy)
        self.field = decode_field(raw)
        self.blocks = decode_blocks(self.field)
        self.staircase = self.decode_staircase(self.field)

        self.blocks_remaining = read_blocks_remaining(pyboy)
        self.blocks_initial = decode_digits(pyboy.memory[INITIAL_TENS_ADDR],
                                            pyboy.memory[INITIAL_ONES_ADDR])
        self.clear_target = pyboy.memory[CLEAR_TARGET_ADDR]
        self.timer_seconds = read_timer(pyboy)
        self.stage = stage_number(pyboy.memory[STAGE_TENS_ADDR],
                                  pyboy.memory[STAGE_ONES_ADDR])

        # Both read off sprites, because neither is in RAM: `$FFD4` lags a throw behind
        # what is in hand, and the player has no row variable at all.
        self.held_block = (self.held_block_type(pyboy, self.calibration.held_sprite)
                           if self.calibration else None)
        if self.held_block is None and held_hint is not None:
            # Only the stage's first state needs this: its hand sprite reads `$82`, which is
            # not a block encoding. `probe_initial_hand` measured what it really is.
            self.held_block = held_hint
        self.last_thrown = pyboy.memory[LAST_THROWN_ADDR]
        self.throws = throw_count(pyboy)
        # Kept because a thrown block *is* a sprite: the field alone cannot tell a settled
        # position from one with a block still in the air.
        self.sprites = tuple(sprites(pyboy))
        self.player_y = player_y(pyboy, self.calibration.player_sprite) if self.calibration else None
        self.player_row = (row_for_y(self.player_y, self.calibration.row_pitch,
                                     self.calibration.row_span[1])
                           if self.calibration and self.calibration.row_span else None)
        self.row_blocks = row_blocks(self.field, self.player_row)

        # The block types a stage started with, carried down the search tree rather than
        # re-derived; a type cleared out would otherwise look like one that never existed.
        self.stage_types = (frozenset(block.type for block in self.blocks)
                            if stage_types is None else stage_types)

        # Sampled here, while the emulator still holds this state: read later from live
        # memory they would describe whichever state was applied last.
        self.stage_cleared = (self.blocks_initial > 0
                              and self.blocks_remaining <= self.clear_target)
        self.out_of_time = self.timer_seconds == 0

        predicates = [f"remaining({self.blocks_remaining})",
                      f"clear-target({self.clear_target})"]
        if self.player_row is not None:
            predicates.append(f"at(player, {self.player_row})")
        elif self.player_y is not None:
            # Before the row span is measured all we have is where he is drawn.
            predicates.append(f"at(player, y{self.player_y})")
        predicates += [f"at(block-{block.type}, {block.row}, {block.col})"
                       for block in self.blocks]
        if self.held_block is not None:
            predicates.append(f"holding(block-{self.held_block})")
        matched = self.stage_types - {block.type for block in self.blocks}
        predicates += [f"all-blocks-cleared(block-{kind})" for kind in sorted(matched)]
        if self.stage_cleared:
            predicates.append("goal-reached")
        if self.out_of_time:
            predicates.append("terminal-state")
        self.literals = frozenset(predicates)

    def is_consistent(self):
        """Whether the field agrees with the HUD counter.

        The memory map's own cross-check: 25 cells in `$83`-`$86` against `BLOCK 25`, and 24
        against 24 after a throw. A mismatch means one of the two has drifted.
        """
        return len(self.blocks) == self.blocks_remaining

    def block_counts(self):
        return block_counts(self.blocks)

    def bounding_box(self):
        return bounding_box(self.field)

    # --------------------------------------------------------------- pure derivation
    # Static because they read synthetic RAM as happily as live memory; they live on the
    # class because this state is their only production caller.

    @staticmethod
    def decode_staircase(field):
        """The fixed `$87` cells, which are structural and can never be cleared."""
        return tuple((row, col) for row in range(FIELD_ROWS) for col in range(FIELD_COLS)
                     if field[row][col] == CELL_STAIRCASE)

    @staticmethod
    def held_block_type(pyboy, sprite):
        """The type of the block in the player's hand, read off its sprite.

        `$FFD4` looks like this and is not: driven across five throws it holds the block
        *previously* in hand (the one just thrown), lagging the sprite by one throw and
        reading `$00` until the first throw of a stage. The sprite's tile is the live
        value, and it uses the same `$83`-`$86` encoding the field does.
        """
        if sprite is None:
            return None
        tile = sprites(pyboy)[sprite][2]
        return tile - BLOCK_MIN + 1 if is_playable(tile) else None

    @staticmethod
    def render_field(field, held=None, player=None, trim=True):
        """The field as ASCII: `#` border, `=` staircase, `.` empty, digits for block types.

        `held` and `player` are appended as trailing lines rather than drawn into the
        grid: the held block is not on the field, and the player's position is only known
        as a sprite Y (the memory map has no row variable for him), which no row index
        here can honestly stand in for.
        """
        (top, bottom), (left, right) = (bounding_box(field) if trim
                                        else ((0, FIELD_ROWS - 1), (0, FIELD_COLS - 1)))
        lines = []
        for row in range(top, bottom + 1):
            line = []
            for col in range(left, right + 1):
                value = field[row][col]
                if is_playable(value):
                    line.append(str(value - BLOCK_MIN + 1))
                else:
                    line.append(CELL_GLYPHS.get(value, "?"))
            lines.append("".join(line))
        text = "\n".join(lines)
        if held is not None:
            text += f"\nheld: {held}"
        if player is not None:
            text += f"\nplayer: row {player}"
        return text

    def __eq__(self, other):
        # The position is the field, what is in hand, and which row the player is on: a
        # throw from a different row does something different, so it is not the same state.
        # Depth and history are not part of it, so a state reached two ways compares equal.
        # The clear target is part of it: stages can share a board and differ only in the
        # target (stages 1 and 3 do), so the same field can sit at two different distances
        # from the goal.
        return (isinstance(other, FlipullGBState) and self.field == other.field
                and self.held_block == other.held_block and self.player_y == other.player_y
                and self.clear_target == other.clear_target)

    def __hash__(self):
        return hash((self.field, self.held_block, self.player_y, self.clear_target))

    def threw(self, parent):
        """Whether the throw that produced this state connected.

        The cartridge's own answer, read from its completed-throw counter rather than
        inferred: a throw that changes nothing does not advance it.
        """
        return self.throws != parent.throws

    def __str__(self):
        # The clear target rides along under the board for the same reason it is part of
        # `__eq__`: without it, stages that share a board stringify identically.
        board = self.render_field(self.field, self.held_block,
                                  self.player_row if self.player_row is not None
                                  else self.player_y)
        return f"{board}\nclear target: {self.clear_target}"

    def __repr__(self):
        return (f"<FlipullGBState(depth={self.depth}, remaining={self.blocks_remaining}"
                f"/target {self.clear_target}, held={self.held_block})>")


# ---------------------------------------------------------------------------- actions

class FlipullGBAction(GBAction):
    """A button held for a number of frames, spelled `"button,ticks"`.

    The same spelling the Puzznic and Super Mario Land environments use. Flipull's whole
    input vocabulary is up, down and throw, so this is the console's interface with nothing
    layered over it.
    """

    cost_map = action_cost_map

    def __settle__(self, pyboy, render, **settle_kwargs):
        return settle(pyboy, render, **settle_kwargs)

    def __next_state__(self, pyboy, state):
        return FlipullGBState(pyboy, state.depth + 1, state.calibration, state.stage_types,
                              state.held_block)


# ------------------------------------------------------------------------ environment

class FlipullGBEnv(GBEnv):
    """Flipull, played on the cartridge.

    The ROM is copyrighted and is not distributed with this repo; pass the path to your own
    dump.

    `successors` routinely filters two of the three actions, and both for real reasons: a
    move into a wall (the player starts on the bottom row, so `down` does nothing until he
    has gone up), and a throw that does not connect: some throws play the whole animation
    and leave the position exactly as it was, down to the cartridge's own completed-throw
    counter. What decides which throws connect is **not modelled here**, deliberately. The
    obvious rule, that the block meets the rightmost block in the player's row and needs a
    match, is wrong: driven across all twelve rows of stage 1, every row connects,
    including rows with no blocks in them at all, so the block plainly travels further than
    its own row. Rather than ship a guess, this environment does what it exists to do and
    asks the cartridge. `row_blocks` is exported for a planner that wants to build its own
    model.
    """

    rom_md5 = ROM_MD5
    rom_name = "Flipull (USA)"
    action_class = FlipullGBAction

    @staticmethod
    def read_stage_table(romfile, base=STAGE_TABLE_ADDR, count=STAGE_COUNT):
        """The stage list, read out of the cartridge rather than transcribed.

        `$3A0E` is a table of `count` little-endian pointers; each points at three bytes:
        the CLEAR target, then the block total as ones and tens digits, the same
        digit-per-byte spelling the HUD counters use. The loader at `0:2D55` indexes it
        with `10*$FFC7 + $FFC6 - 1`.

        Reading it means the stage list always matches the ROM in hand, and it gives
        `reset` something independent to check a selected stage against: stage 8 has 36
        blocks and a target of 7, and if the cartridge says otherwise the selection did
        not take.
        """
        with open(romfile, "rb") as handle:
            rom = handle.read()
        stages = []
        for index in range(count):
            pointer = base + 2 * index
            if pointer + 1 >= len(rom):
                break
            target = rom[pointer] | (rom[pointer + 1] << 8)
            if target + STAGE_DESCRIPTOR_BYTES > len(rom):
                break
            clear, ones, tens = rom[target:target + STAGE_DESCRIPTOR_BYTES]
            # The table is followed in ROM by a shorter second one, so the parse has to
            # know where to stop rather than trust `count`. Digits that are not digits
            # mean we have walked off the end of it.
            if ones > 9 or tens > 9 or clear > 99 or not (tens * 10 + ones):
                break
            stages.append(Stage(index + 1, clear, tens * 10 + ones))
        return tuple(stages)

    def __init__(self, romfile, render=False, verify_rom=True, calibrate=True,
                 settle_max_ticks=SETTLE_MAX_TICKS, settle_stable_ticks=SETTLE_STABLE_TICKS,
                 boot_max_ticks=BOOT_MAX_TICKS):
        self.romfile = romfile
        # Not `self.render`: that name belongs to the method which prints the history.
        self.render_window = render
        self.pyboy = None
        self.stage_index = None
        self.state = None
        self.state_history = []
        self.calibration = None
        self.should_calibrate = calibrate
        self.intro_ticks = None
        self.actions = action_list
        self.settle_kwargs = {"max_ticks": settle_max_ticks, "stable_ticks": settle_stable_ticks}
        self.boot_max_ticks = boot_max_ticks
        self.stages = self.read_stage_table(romfile) if os.path.isfile(romfile) else ()
        self.forcing_stage = None
        if verify_rom:
            self.__verify_rom__()

    def stage_for(self, index):
        """What the cartridge says stage `index` (0-based) holds, before loading it."""
        return self.stages[index]

    def fix_index(self, index):
        """Select the stage, zero-based: `fix_index(7)` is stage 8.

        The cartridge keeps its stage number as two decimal digits (`$FFC7` tens, `$FFC6`
        ones), and the loader at `0:2D55` turns them into an index into a 32-entry table.
        `reset` hooks that loader and writes the digits as it arrives, so the stage the
        cartridge builds is the one asked for.

        This is not the same as poking a layout in behind the game's back, which is what
        Puzznic's fallback route does and why that one is kept for emergencies. Here the
        two bytes written *are* the game's own stage number, they are written before
        anything reads them, and the eleven places that read them (the field builder and
        the HUD among them) all then agree: the on-screen `STAGE` really does say 8.
        """
        if not self.stages:
            raise RuntimeError(
                f"no stage table could be read from {self.romfile} at ${STAGE_TABLE_ADDR:04X}. "
                f"Check the ROM is Flipull (USA) (MD5 {ROM_MD5}).")
        if not 0 <= index < len(self.stages):
            raise IndexError(
                f"Invalid index: {index}. {self.romfile} has {len(self.stages)} stages, so "
                f"the index must be 0-{len(self.stages) - 1}. Past the end of the table the "
                "loader reads whatever follows it in ROM and silently builds some other "
                "stage, so this refuses rather than doing that.")
        self.stage_index = index

    def reset(self):
        self.__restart_emulator__()
        # The arrangement RNG is seeded by boot timing (see SEED_TICKS_PER_INDEX), so this
        # per-index delay is what makes fix_index select a distinct board rather than only
        # a distinct CLEAR target.
        for _ in range(SEED_TICKS_PER_INDEX * (self.stage_index or 0)):
            self.pyboy.tick()
        self.__force_stage__()
        if not boot(self.pyboy, self.render_window, self.boot_max_ticks):
            raise RuntimeError(
                f"no stage was loaded within {self.boot_max_ticks} frames of booting "
                f"{self.romfile}. Check the ROM is Flipull (USA) (MD5 {ROM_MD5}).")
        # Stop forcing now that the stage is up, so the cartridge behaves normally from
        # here: a cleared stage advances to the next one, and a lost life reloads this one.
        self.forcing_stage = None
        self.__check_stage__()
        self.intro_ticks = wait_until_interactive(self.pyboy, self.render_window)
        settle(self.pyboy, self.render_window, **self.settle_kwargs)

        held_hint = None
        if self.should_calibrate and self.calibration is None:
            # A property of the cartridge, not of the stage, so once is enough.
            self.calibration = calibrate(self.pyboy, save_state(self.pyboy),
                                         self.render_window, **self.settle_kwargs)
            self.actions = button_actions(self.calibration)
        if self.calibration:
            # This one *is* per-stage: the opening hand's sprite tile is not a block value.
            held_hint = probe_initial_hand(self.pyboy, save_state(self.pyboy),
                                           self.render_window, self.calibration.press_ticks,
                                           self.calibration.throw_button, **self.settle_kwargs)

        self.state = FlipullGBState(self.pyboy, 0, self.calibration, held_hint=held_hint)
        self.state_history = [self.state]
        return self.state, {"stage": self.state.stage,
                            "blocks": self.state.blocks_remaining,
                            "clear_target": self.state.clear_target,
                            "intro_ticks": self.intro_ticks,
                            "held_block": self.state.held_block,
                            "calibration": self.calibration}

    def __force_stage__(self):
        """Write the stage digits each time the loader is reached, for the boot only."""
        if self.stage_index is None:
            return
        self.forcing_stage = self.stage_index + 1

        def force(_context):
            if self.forcing_stage is None:
                return
            tens, ones = stage_digits(self.forcing_stage)
            self.pyboy.memory[STAGE_TENS_ADDR] = tens
            self.pyboy.memory[STAGE_ONES_ADDR] = ones

        self.pyboy.hook_register(ROM_BANK, STAGE_LOADER_ADDR, force, None)

    def __check_stage__(self):
        """Whether the stage that loaded matches what the ROM's table said it would.

        The table gives a block total and a CLEAR target per stage, and no two neighbours
        share both, so this catches a selection that silently did not take, which is the
        failure worth catching, because a wrong stage still looks like a good one.
        """
        if self.stage_index is None:
            return
        want = self.stages[self.stage_index]
        got_stage = stage_number(self.pyboy.memory[STAGE_TENS_ADDR],
                                 self.pyboy.memory[STAGE_ONES_ADDR])
        got_blocks = read_blocks_remaining(self.pyboy)
        got_clear = self.pyboy.memory[CLEAR_TARGET_ADDR]
        if (got_stage, got_blocks, got_clear) != (want.number, want.blocks, want.clear_target):
            raise RuntimeError(
                f"selecting stage {want.number} did not take: the cartridge came up on "
                f"stage {got_stage} with {got_blocks} blocks and a target of {got_clear}, "
                f"where the ROM's own table at ${STAGE_TABLE_ADDR:04X} says stage "
                f"{want.number} has {want.blocks} blocks and a target of {want.clear_target}.")

    def is_goal(self, state):
        """Few enough blocks left.

        Flipull finishes a stage when the count is down to the `CLEAR` number rather than to
        zero: the HUD shows `BLOCK 25` against `CLEAR 09`. `$FFCF` is that number, and the
        map grades it moderate: it read 09 and never changed, which is consistent with a
        target but was never watched being met.
        """
        return state.stage_cleared

    def is_terminal(self, state):
        """The clock ran out.

        Unlike Puzznic there is no positional dead end to detect: a stage is lost on time,
        and whether a given field can still reach its target depends on the throw mechanics,
        which the map does not describe.
        """
        return state.out_of_time

    def __score__(self, state):
        return state.blocks_initial - state.blocks_remaining


# --------------------------------------------------------------------------- reporting

def _report(romfile, stage=None, render=False):
    """Print what this cartridge wants: the measurements, and the stage it loaded."""
    env = FlipullGBEnv(romfile, render=render)
    try:
        print(f"{len(env.stages)} stages in the table at ${STAGE_TABLE_ADDR:04X}\n")
        env.fix_index(0 if stage is None else stage)
        state, info = env.reset()
        calibration = info["calibration"]
        print(f"stage {info['stage']}: {state.blocks_remaining} blocks, "
              f"clear target {state.clear_target}, {state.timer_seconds}s on the clock\n")
        print(state, "\n")
        if calibration.player_sprite is None:
            print("  player       not found — no OAM entry tracked the direction pressed")
        else:
            rows = ("" if not calibration.row_span else
                    f", rows {row_for_y(calibration.row_span[1], calibration.row_pitch, calibration.row_span[1])}"
                    f"-{row_for_y(calibration.row_span[0], calibration.row_pitch, calibration.row_span[1])}")
            print(f"  player       OAM slot {calibration.player_sprite}, "
                  f"{calibration.row_pitch} pixels per row{rows}, "
                  f"free to move {calibration.move_button}")
        if calibration.held_sprite is None:
            print("  held block   not identified — nothing flew off when a throw went out")
        else:
            print(f"  held block   OAM slot {calibration.held_sprite}, "
                  f"currently type {state.held_block}")
        if calibration.hold_window is None:
            print("  move hold    not measurable — the player never moved")
        else:
            print(f"  move hold    {calibration.hold_window}  -> press_ticks "
                  f"{calibration.press_ticks}  (repeat fires on frame "
                  f"{calibration.hold_window[1] + 1})")
        print(f"  throw        {calibration.throw_button.upper()} held "
              f"{calibration.throw_ticks} frames")
        print(f"  stage intro  {info['intro_ticks']} frames of ignored input")
        print(f"\n  button actions: {button_actions(calibration)}")
    finally:
        env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure how a Flipull cartridge wants to be driven: which sprite the "
                    "player is, how long a button must be held to move exactly one row, "
                    "and which button throws.")
    parser.add_argument("rom", help="path to Flipull (USA).gb")
    parser.add_argument("--stage", type=int, default=None,
                        help="stage index to load, zero-based (--stage 7 is stage 8)")
    parser.add_argument("--render", action="store_true", help="open an SDL2 window")
    args = parser.parse_args()
    _report(args.rom, args.stage, args.render)
