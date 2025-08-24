import os
import io
from enum import Enum
from tkinter import Image
import numpy as np
from copy import deepcopy
from itertools import product
from collections import namedtuple
from pyboy import PyBoy
from planiverse.problems.retro_games.base import RetroGame

class Actions(Enum):
    IDLE = 0
    LEFT_MOVE_PRESS = 1
    RIGHT_MOVE_PRESS = 2
    UP_MOVE_PRESS = 3
    DOWN_MOVE_PRESS = 4
    LEFT_HOLD_PRESS = 5
    RIGHT_HOLD_PRESS = 6
    UP_HOLD_PRESS = 7
    DOWN_HOLD_PRESS = 8
    RELEASE = 9

position = namedtuple("Position", ["x", "y"])

def save_state(pyboy):
    with io.BytesIO() as f:
        pyboy.save_state(f)
        f.seek(0)
        state_bytes = f.getvalue()
    return state_bytes

def load_state(pyboy, state_bytes, render=False):
    with io.BytesIO(state_bytes) as f:
        pyboy.load_state(f)
        pyboy.tick(60, render)

class SuperMarioState:
    def __init__(self, pyboy, gamestate):
        self.gamestate = gamestate
        self.pyboy     = pyboy
        self.gb_state  = save_state(pyboy)
        self.mario_position = self.__get_mario_position__(pyboy)

    def __eq__(self, other):
        # Two states are equal if and only if mario is in the same position. 
        # I am afraid that this won't all mario to go back
        # So if mairo is in the same position and time difference is more than xx seconds then consider those two states are the same.
        return self.mario_position == other.mario_position and abs(self.gamestate.time_left - other.gamestate.time_left) > 5

    def __get_mario_position__(self, pyboy):
        return position(x=pyboy.memory[0xC202], y=pyboy.memory[0xC201])    

    @property
    def timeleft(self):
        return self.gamestate.time_left
    @property
    def lives_left(self):
        return self.gamestate.lives_left

class SuperMarioAction:
    def __init__(self, pyboy, action):
        self.pyboy = pyboy
        self.action = action

    def __str__(self):
        return self.action

    def __repr__(self):
        return str(self)

    def apply(self, state, game, t=60, render=False):
        current_state = save_state(self.pyboy)
        load_state(self.pyboy, state.gb_state)
        for act in self.action.split('+'):
            self.pyboy.button(act)
        self.pyboy.tick(t, render)
        ret_state = SuperMarioState(self.pyboy, game)
        load_state(self.pyboy, current_state) # restore the current state
        return ret_state
    
    def simulate(self, state):
        """
        Use this only to produce gifs.
        """
        factor = 3
        dummy_pyboy = PyBoy(self.pyboy.gamerom, sound_volume=0)
        dummy_pyboy.tick(60, True) # To render screen after `.start_game`
        load_state(dummy_pyboy, state.gb_state, True)
        for act in self.action.split('+'):
            dummy_pyboy.button(act)
        ret_state_imgs = []
        for _ in range(60): # Progress 60 frames and render every frame
            if not dummy_pyboy.tick(1, True): break
            ret_state_imgs.append(deepcopy(dummy_pyboy.screen.image).resize((dummy_pyboy.screen.image.width * factor, dummy_pyboy.screen.image.height * factor)))
        dummy_pyboy.stop()
        return ret_state_imgs
    
class SuperMario(RetroGame):
    def __init__(self, romfile, render=False):
        self.romfile = romfile
        self.pyboy   = None
        self.render  = render
        self.world_level = None
        self.world_level_map = {k:v for k, v in enumerate(product([i for i in range(0,4)], repeat=2))}
        self.actions = ['a', 'b', 'left', 'right', 'down', 'a+left', 'a+right', 'b+down']
        self.render_frame = 60

    def reset(self):
        self.pyboy = PyBoy(self.romfile, sound_volume=0, window="SDL2" if self.render else "null")
        # assert self.world_level is not None, "World level must be set before reset"
        # self.pyboy.game_wrapper.set_world_level(self.world_level[0], self.world_level[1])
        self.pyboy.set_emulation_speed(0)
        self.game = self.pyboy.game_wrapper
        self.game.game_area_mapping(self.game.mapping_compressed, 0)
        self.game.start_game()
        self.pyboy.tick(self.render_frame, self.render) # To render screen after `.start_game`
        init_state = SuperMarioState(self.pyboy, self.game)
        return init_state, {}

    def fix_index(self, index):
        assert index in self.world_level_map.keys(), "Invalid index"
        self.world_level = self.world_level_map[index]

    def is_goal(self, state):
        return self.pyboy.memory[0xDFE8] == 0x01

    def is_terminal(self, state):
        # know this information from the music track requested.
        # according to: https://datacrystal.romhacking.net/wiki/Super_Mario_Land:RAM_map
        # DFE8 1 Request a Music Track 
        # #  0x01 = level clear, 
        # #  0x02 = death,
        # #  0x10 = game over,
        return self.pyboy.memory[0xDFE8] in [0x02, 0x10]
    
    # Returns a list of [action, successor_state]
    def successors(self, state):
        ret = []
        for idx, actionstr in enumerate(self.actions):
            action = SuperMarioAction(self.pyboy, actionstr)
            successor_state = action.apply(state, self.game, self.render_frame, self.render)
            if successor_state == state: continue
            ret.append((action, successor_state))
        return ret
    
    def simulate(self, plan):
        assert False, "Not implemented"
    