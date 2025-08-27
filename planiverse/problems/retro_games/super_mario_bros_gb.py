import os
import io
from enum import Enum
from tkinter import Image
import numpy as np
from copy import deepcopy
from itertools import product, chain
from collections import namedtuple
from pyboy import PyBoy
from pyboy.utils import _bcd_to_dec, bcd_to_dec
from planiverse.problems.retro_games.base import RetroGame

forward_ticks = 10
image_resize_factor = 4
position = namedtuple("Position", ["x", "y"])
velocity = namedtuple("Velocity", ["x", "y"])
action_list  = list(chain.from_iterable([[f'{a},{t}' for a in ['a+left', 'a+right']] for t in [2, 10]])) # [3, 5, 10]
action_list += list(chain.from_iterable([[f'{a},{t}' for a in ['left', 'right']] for t in [3]])) #[2]

#  'b+left', 'b+right', 'down', 'a'

def create_pyboy(romfile, render):
    return PyBoy(romfile, sound_volume=0, window="SDL2" if render else "null")

def save_state(pyboy):
    with io.BytesIO() as f:
        pyboy.save_state(f)
        f.seek(0)
        state_bytes = f.getvalue()
    return state_bytes

def load_state(pyboy, state_bytes, render=False):
    with io.BytesIO(state_bytes) as f:
        pyboy.load_state(f)
        pyboy.tick(1, render)

class SuperMarioState:
    def __init__(self, pyboy, depth):
        self.depth     = depth
        self.literals  = frozenset() # Dummy
        self.gb_state  = save_state(pyboy)
        self.__update__(pyboy)
        
    def __lt__(self, other):
        return self.depth < other.depth
    
    def __update__(self, pyboy):
        # Do game updates.
        self.mario_position = position(x=pyboy.memory[0xC202], y=pyboy.memory[0xC201])
        self.mario_velocity = velocity(x=pyboy.memory[0xC20C], y=pyboy.memory[0xC20D])
        # self.collision = any(abs(a - self.mario_position.x) + abs(b - self.mario_position.y) <= 3 for a, b in [(pyboy.memory[a], pyboy.memory[a - 1]) for a in range(0xD103, 0xD194, 0x10)])

        self.collision = pyboy.memory[0xC0AC] != 0 # Check the Mario dead jump timer.
        
        # Do we need this ?! 
        self.max_mario_x_pos = self.mario_position.x + 175
        
        
        timeleft = _bcd_to_dec(pyboy.memory[0xDA01 : 0xDA01 + 2])
        self.timeleft = timeleft[0] + timeleft[1] * 100
        level_block = pyboy.memory[0xC0AB]
        mario_x = pyboy.memory[0xC202]
        scx = pyboy.screen.tilemap_position_list[16][0]
        self.level_progress = level_block * 16 + (scx - 7) % 16 + mario_x
        blank = 300
        self.coins = self._sum_number_on_screen(pyboy, 9, 1, 2, blank, -256)
        self.lives_left = bcd_to_dec(pyboy.memory[0xDA15])
        self.score = self._sum_number_on_screen(pyboy, 0, 1, 6, blank, -256)
        
        # So the state is constructed as the following:
        # (supermario XX YY)
        # (sprite ID XX YY)
        # (coins #count) <- ignore
        # (timeleft ?) <- ignore
        # (livesleft ?) <- ignore
        predicates  = []
        # predicates += list(map(lambda sprite: f'(sprite {sprite.tile_identifier} {sprite.x} {sprite.y})', filter(lambda s:s.on_screen, self.gamestate._sprites_on_screen())))
        # predicates += chain.from_iterable([[f'(tile {i}, {j}, {v})' for j, v in enumerate(line)] for i, line in enumerate(self.gamestate.game_area())])
        predicates += [
            f'(supermario {self.mario_position.x} {self.mario_position.y})',
            # f'(coins {self.coins})',
            # f'(timeleft {self.timeleft})', # Ignore this one.
            # f'(livesleft {self.lives_left})',
        ]
        self.literals |= frozenset(predicates)
    
    def _sum_number_on_screen(self, pyboy, x, y, length, blank_tile_identifier, tile_identifier_offset):
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
        return f'<SuperMarioState(depth={self.depth}, mario_position={self.mario_position}, velocity={self.mario_velocity}, collision={self.collision})>'
    
    def mario_damage(self):
        return 1 if self.collision else 0

    def save(self, gamerom, file):
        dummy_pyboy = create_pyboy(gamerom, True)
        load_state(dummy_pyboy, self.gb_state, True)
        dummy_pyboy.screen.image.resize((320*image_resize_factor,288*image_resize_factor)).save(file)
        dummy_pyboy.stop()

class SuperMarioAction:
    def __init__(self, action):
        self.action = action
        self.forward_ticks = forward_ticks

    def __lt__(self, other):
        s = self.__parse_action__(self.action)
        o = self.__parse_action__(other.action)
        return max(x[1] for x in s) < max(x[1] for x in o)

    def __str__(self):
        return self.action

    def __repr__(self):
        return str(self)

    def __parse_action__(self, act):
        # Parse the action string into a list of individual actions.
        actions, tick = act.split(',')
        return [(a, int(tick)) for a in actions.split('+')]

    def apply(self, pyboy, state):
        load_state(pyboy, state.gb_state)
        # apply the action n times to speed up the search.
        ticks_values = set()
        for act, ticks in self.__parse_action__(self.action):
            pyboy.button(act, ticks)
            ticks_values.add(ticks)
        pyboy.tick(max(ticks_values), False)
        ret_state = SuperMarioState(pyboy, state.depth + 1)
        return ret_state
    
class SuperMario(RetroGame):
    def __init__(self, romfile, render=False):
        self.romfile = romfile
        self.pyboy   = None
        self.render  = render
        self.world_level = None
        self.world_level_map = {k:v for k, v in enumerate(product([i for i in range(0,4)], repeat=2))}
        self.actions = action_list

    def reset(self):
        self.pyboy = create_pyboy(self.romfile, self.render)
        self.game = self.pyboy.game_wrapper
        self.game.game_area_mapping(self.game.mapping_compressed, 0)
        self.game.start_game()
        self.game.set_lives_left(0) # to avoid replays 
        self.pyboy.tick() # To render screen after `.start_game`
        self.game.post_tick()
        return SuperMarioState(self.pyboy, 0), {}

    def fix_index(self, index):
        assert index in self.world_level_map.keys(), "Invalid index"
        self.world_level = self.world_level_map[index]

    def is_goal(self, state):
        # TODO: We need to know what is special that we can use to know that we reached the goal
        return self.pyboy.memory[0xDFE8] == 0x01

    def is_terminal(self, state):
        # know this information from the music track requested.
        # according to: https://datacrystal.romhacking.net/wiki/Super_Mario_Land:RAM_map
        return self.pyboy.memory[0xC0A4] == 0x39 or state.collision 
    
    # Returns a list of [action, successor_state]
    def successors(self, state):
        ret = []
        for idx, actionstr in enumerate(self.actions):
            action = SuperMarioAction(actionstr)
            successor_state = action.apply(self.pyboy, state)
            if successor_state == state: continue
            ret.append((action, successor_state))
        return ret
    
    def simulate(self, plan):
        # this needs to be reimplemented
        state, _ = self.reset()
        state_trace = [state]
        for idx, action in enumerate(plan):
            state_trace.append(action.apply(self.pyboy, state_trace[-1]))
            if self.is_terminal(state_trace[-1]):
                break
        return state_trace
