"""Super Mario Land on a Game Boy, driven through PyBoy.

Addresses come from a reverse-engineering pass over `Super Mario Land (World) (Rev 1).gb`
(MD5 `b259feb41811c7e4e1dc200167985c84`), documented in
`docs/environments/super-mario-land.md`. That map was derived *behaviourally* — recording
work RAM every frame while driving scripted input — so its confidence varies field by field,
and the constants below carry that grading rather than pretending to it.
"""
from itertools import product, chain
from collections import namedtuple
from pyboy.utils import bcd_to_dec
from planiverse.environments.gb import (
    GBAction, GBEnv, GBState, create_pyboy, load_state,
)

#: The dump the addresses below were read from. Another revision will read garbage.
ROM_MD5 = "b259feb41811c7e4e1dc200167985c84"

# --------------------------------------------------------------- Mario's state struct
# $C200-$C21F. Verified: the struct starts at $C200 -- $C1F0-$C1FF was provably zero across
# every frame recorded.
MARIO_Y_ADDR = 0xC201            # verified; screen coordinates, not level
MARIO_X_ADDR = 0xC202            # verified; saturates at $51 once the camera takes over
MARIO_ANIMATION_ADDR = 0xC203    # good: $00 still, cycles $01-$04 moving
MARIO_FACING_ADDR = 0xC205       # verified: $00 right, $20 left
MARIO_JUMP_PHASE_ADDR = 0xC207   # good: $00 grounded, $01 -> $02 through a jump
MARIO_ON_GROUND_ADDR = 0xC20A    # verified: $01 grounded, $00 airborne
MARIO_SPEED_ADDR = 0xC20C        # good: $00 still, $06 walking, $19-$27 airborne
MARIO_DIRECTION_ADDR = 0xC20D    # verified: $00 still, $10 right, $20 left
MARIO_MOVING_ADDR = 0xC20F       # good: $00 still, $01 moving or airborne

FACING_LEFT = 0x20
DIRECTIONS = {0x00: "still", 0x10: "right", 0x20: "left"}

#: Mario's X stops here and the camera scrolls instead, so it is not a progress measure.
MARIO_X_SATURATES_AT = 0x51

# ------------------------------------------------------------------ object/enemy array
# $D100-$D19F, ten slots of $10 bytes. Verified: base, stride, the $FF empty marker, and
# +2/+3 as Y/X. Everything from +4 on comes from a single ground-walking enemy in 1-1 and
# will not generalise to flying, shelled or boss objects.
OBJECTS_ADDR = 0xD100
OBJECT_SLOTS = 10
OBJECT_STRIDE = 0x10
OBJECT_EMPTY = 0xFF
Enemy = namedtuple("Enemy", ["slot", "type", "x", "y", "animation"])

# ------------------------------------------------------------------- HUD and counters
TIMER_LOW_ADDR = 0xDA01          # verified: low two digits, BCD
TIMER_HIGH_ADDR = 0xDA02         # verified: high digit, BCD
LIVES_ADDR = 0xDA15              # moderate: matched MARIO x02 on screen, never seen change
WORLD_ADDR = 0xDA16              # unverified: read $01 in 1-1, no level transition observed

#: Hardware SCX. The map searched all of WRAM for a mirror of the scroll value and found
#: none, so the camera is only readable from the register itself.
SCX_ADDR = 0xFF43
SCY_ADDR = 0xFF42

#: Not in the memory map -- PyBoy's own Super Mario Land wrapper uses it as the level block,
#: and `level_progress` below is its formula. Kept because it is the only thing that gives a
#: number which keeps rising across screens; the map explicitly failed to find a 16-bit
#: level X anywhere in WRAM.
LEVEL_BLOCK_ADDR = 0xC0AB

#: Neither of these is in the memory map. $C000-$C09F is the OAM shadow, so both sit just
#: past it in territory the map does not cover, and neither was ever watched changing.
LEVEL_COMPLETE_ADDR = 0xDFE8     # unverified
MUSIC_TRACK_ADDR = 0xC0A4        # unverified
DEATH_MUSIC_TRACK = 0x39

forward_ticks = 10
image_resize_factor = 4
position = namedtuple("Position", ["x", "y"])
velocity = namedtuple("Velocity", ["x", "y"])


def read_timer(pyboy):
    """The HUD timer, from the two BCD bytes at `$DA01`/`$DA02`.

    `$DA02` holds the high digit and `$DA01` the low two, so `03`/`97` reads 397 — checked
    against the on-screen value, decrementing one BCD unit per second.
    """
    return decode_timer(pyboy.memory[TIMER_HIGH_ADDR], pyboy.memory[TIMER_LOW_ADDR])


def decode_timer(high, low):
    return (high & 0x0F) * 100 + (low >> 4) * 10 + (low & 0x0F)


def decode_objects(raw):
    """The live entries of the object array, from the 160 bytes at `$D100`.

    A slot goes live when an object scrolls into range and reverts to `$FF` when it leaves
    or dies, so this is what is on screen rather than everything in the level.
    """
    enemies = []
    for slot in range(OBJECT_SLOTS):
        base = slot * OBJECT_STRIDE
        fields = raw[base:base + OBJECT_STRIDE]
        if len(fields) < 4 or fields[0] == OBJECT_EMPTY:
            continue
        enemies.append(Enemy(slot=slot, type=fields[1], y=fields[2], x=fields[3],
                             animation=fields[4] if len(fields) > 4 else None))
    return tuple(enemies)


def read_objects(pyboy):
    return decode_objects(pyboy.memory[OBJECTS_ADDR:OBJECTS_ADDR + OBJECT_SLOTS * OBJECT_STRIDE])


action_list  = list()
action_list += list(chain.from_iterable([[f'{a},{t}' for a in ['a+left', 'a+right', 'b+left', 'b+right']] for t in [5,10,15]])) # [3, 5, 10]
action_list += list(chain.from_iterable([[f'{a},{t}' for a in ['nop', 'left', 'right', 'down']] for t in [3]])) #[2]

action_cost_map = {
    'a': 2,
    'b': 2,
    'left': 1,
    'right': 1,
    'down': 1,
    'nop': 0
}


# The emulator plumbing (`create_pyboy`, `load_state`, save-states) comes from the
# shared `gb` module.

class SuperMarioState(GBState):
    def __init__(self, pyboy, depth):
        super().__init__(pyboy, depth)
        self.__update__(pyboy)

    def __update__(self, pyboy):
        # --- Mario. $C201/$C202 are *screen* coordinates: X stops at $51 once he reaches
        # the scroll trigger and the camera takes over, so it is a position on the display,
        # never a measure of how far through the level he is. Use `level_progress` for that.
        self.mario_position = position(x=pyboy.memory[MARIO_X_ADDR], y=pyboy.memory[MARIO_Y_ADDR])
        self.mario_facing = "left" if pyboy.memory[MARIO_FACING_ADDR] == FACING_LEFT else "right"
        self.on_ground = pyboy.memory[MARIO_ON_GROUND_ADDR] == 0x01
        self.airborne = not self.on_ground
        self.jump_phase = pyboy.memory[MARIO_JUMP_PHASE_ADDR]
        self.animation_frame = pyboy.memory[MARIO_ANIMATION_ADDR]
        self.moving = pyboy.memory[MARIO_MOVING_ADDR] != 0

        # $C20C is a speed *magnitude* and $C20D a direction code -- not the x and y of a
        # velocity, which is what these two used to be read as. There is no vertical
        # velocity byte in the map at all; `jump_phase` and `on_ground` are what describe
        # vertical motion.
        self.mario_speed = pyboy.memory[MARIO_SPEED_ADDR]
        self.mario_direction = DIRECTIONS.get(pyboy.memory[MARIO_DIRECTION_ADDR], "unknown")

        # --- the camera. The map searched every byte of WRAM for a mirror of the scroll
        # value and found none, so SCX is only readable from the register.
        self.camera_x = pyboy.memory[SCX_ADDR]
        self.camera_y = pyboy.memory[SCY_ADDR]

        # --- objects. Ten slots at $D100, $FF meaning empty; a slot goes live when
        # something scrolls into range. Replaces counting sprites by tile id, which only
        # ever recognised one identifier.
        self.enemies = read_objects(pyboy)
        self.enemies_on_screen = len(self.enemies)
        self.touching_enemy = any(self.touching(self.mario_position, enemy)
                                  for enemy in self.enemies)
        self.collision = self.touching_enemy

        # Goal/terminal flags are sampled here, while the emulator still holds this state.
        # Reading them later from live memory would describe whichever state was applied last.
        # Neither address is in the memory map; both are inherited guesses -- see the docs.
        self.level_complete = pyboy.memory[LEVEL_COMPLETE_ADDR] == 0x01
        self.game_over = pyboy.memory[MUSIC_TRACK_ADDR] == DEATH_MUSIC_TRACK

        self.timeleft = read_timer(pyboy)
        self.lives_left = bcd_to_dec(pyboy.memory[LIVES_ADDR])
        self.world = pyboy.memory[WORLD_ADDR]

        # PyBoy's own wrapper formula, not the memory map's -- the map looked for a 16-bit
        # level X across all of WRAM and found nothing. SCX is taken at scanline 16 rather
        # than from $FF43, because the HUD splits the screen and the register holds whatever
        # the split left behind; scanline 16 is below it, so it is the playfield's scroll.
        level_block = pyboy.memory[LEVEL_BLOCK_ADDR]
        playfield_scx = pyboy.screen.tilemap_position_list[16][0]
        self.level_progress = level_block * 16 + (playfield_scx - 7) % 16 + self.mario_position.x

        blank = 300
        self.coins = self.__sum_number_on_screen__(pyboy, 9, 1, 2, blank, -256)
        self.score = self.__sum_number_on_screen__(pyboy, 0, 1, 6, blank, -256)

        predicates = [
            f'(supermario position {self.mario_position.x} {self.mario_position.y})',
            f'(supermario motion {self.mario_speed} {self.mario_direction})',
            f'(supermario grounded {int(self.on_ground)})',
            f'(progress {self.level_progress})',
            f'(depth {self.depth})',
            f'(coins {self.coins})',
            # f'(timeleft {self.timeleft})', # Ignore this one.
            f'(livesleft {self.lives_left})',
        ]
        predicates += [f'(enemy {enemy.type} {enemy.x} {enemy.y})' for enemy in self.enemies]
        self.literals |= frozenset(predicates)
    
    def __sum_number_on_screen__(self, pyboy, x, y, length, blank_tile_identifier, tile_identifier_offset):
        number = 0
        for i, x in enumerate(pyboy.tilemap_background[x : x + length, y]):
            if x != blank_tile_identifier: number += (x + tile_identifier_offset) * (10 ** (length - 1 - i))
        return number
    
    def __eq__(self, other):
        # Two states are equal if and only if mario is in the same position. 
        # I am afraid that this won't all mario to go back
        # So if mairo is in the same position and time difference is more than xx seconds then consider those two states are the same.
        return self.literals == other.literals and abs(self.timeleft - other.timeleft) < 5

    def __repr__(self):
        return (f'<SuperMarioState(depth={self.depth}, mario_position={self.mario_position}, '
                f'progress={self.level_progress}, enemies={self.enemies_on_screen})>')
    
    @staticmethod
    def touching(mario, enemy, reach=8):
        """Whether an object's box overlaps Mario's.

        Both are screen coordinates read on the same frame, so they are directly
        comparable. This is a proximity test, not a flag the game sets — the map found no
        damage byte — so it says "in contact", not "took a hit": what contact costs
        depends on power-up state, which the map could not confirm either.
        """
        return abs(mario.x - enemy.x) < reach and abs(mario.y - enemy.y) < reach

    def mario_damage(self):
        """1 when an object's box overlaps Mario's.

        A proximity test over the `$D100` array, not a flag the game sets: the map found no
        damage byte. Useful as a heuristic penalty, not as a death signal.
        """
        return 1 if self.touching_enemy else 0

    def save(self, gamerom, file, scale=2 * image_resize_factor):
        """Inherited behaviour, with this game's larger default scale.

        It used to open a real SDL2 window to take the screenshot, which needs a display and
        so failed on any headless machine — including every CI runner. The shared
        implementation uses the null window, which renders the frame just as well.
        """
        return super().save(gamerom, file, scale=scale)

class SuperMarioAction(GBAction):
    cost_map = action_cost_map

    def __cost__(self):
        # Unlike the siblings, cost scales with how long the buttons are held.
        return (sum(self.cost_map[button] for button, _ in self.actions_tick_list)
                * self.actions_tick_list[-1][1])

    def __lt__(self, other):
        s = self.actions_tick_list
        o = other.actions_tick_list
        return max(x[1] for x in s) < max(x[1] for x in o)

    # No `__settle__` override: this environment snapshots straight after the hold. The
    # siblings wait for the game to stop moving first; deciding what "settled" honestly
    # means for a scrolling platformer needs the real cartridge, so it is deliberately
    # not changed here.

    def __next_state__(self, pyboy, state):
        return SuperMarioState(pyboy, state.depth + 1)


class SuperMarioEnv(GBEnv):
    rom_md5 = ROM_MD5
    rom_name = "Super Mario Land (World) (Rev 1)"
    action_class = SuperMarioAction

    def __init__(self, romfile, render=False, verify_rom=True):
        self.romfile = romfile
        self.pyboy   = None
        # Not `self.render`: that name belongs to the method which prints the history.
        self.render_window = render
        self.world_level = None
        # Super Mario Land has 4 worlds of 3 levels each, both 1-indexed.
        self.world_level_map = {k:v for k, v in enumerate(product(range(1,5), range(1,4)))}
        self.actions = action_list
        self.state = None
        self.state_history = []
        self.settle_kwargs = {}
        if verify_rom:
            self.__verify_rom__()

    def reset(self):
        self.__restart_emulator__()
        self.game = self.pyboy.game_wrapper
        self.game.game_area_mapping(self.game.mapping_compressed, 0)
        # world_level is None until fix_index is called, which starts the game at its default level.
        self.game.start_game(world_level=self.world_level)
        self.game.set_lives_left(0) # to avoid replays
        self.pyboy.tick() # To render screen after `.start_game`
        self.game.post_tick()
        self.state = SuperMarioState(self.pyboy, 0)
        self.state_history = [self.state]
        return self.state, {}

    def fix_index(self, index):
        assert index in self.world_level_map.keys(), "Invalid index"
        self.world_level = self.world_level_map[index]

    def is_goal(self, state):
        # TODO: 0xDFE8 is our best guess at the level-complete flag; it needs confirming.
        return state.level_complete

    def is_terminal(self, state):
        """Mario died.

        Touching an enemy is deliberately not terminal. `state.touching_enemy` is a
        proximity test over the object array, and contact is only fatal to small Mario —
        the map could not confirm the power-up byte, so there is no way to tell the cases
        apart. Death itself is caught by the music track. Use `mario_damage()` if you want
        contact as a heuristic penalty.
        """
        return state.game_over

    def __score__(self, state):
        return state.level_progress

    def __advance__(self, state, action):
        """Apply one action.

        Unlike the siblings there is no absorbing rule for won or lost levels here, and no
        settle: both would change what this environment has always handed back, and
        verifying the change needs the real cartridge — so the alignment is deliberately
        deferred rather than slipped in.
        """
        if isinstance(action, str):
            action = SuperMarioAction(action)
        return action.apply(self.pyboy, state)
