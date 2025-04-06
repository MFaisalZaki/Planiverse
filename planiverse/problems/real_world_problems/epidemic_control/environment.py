

import os
import json
import numpy as np
from gym import spaces
from planiverse.problems.real_world_problems.epidemic_control.epipolicy.core.epidemic import construct_epidemic
from planiverse.problems.real_world_problems.epidemic_control.epipolicy.obj.act import construct_act, get_normalized_action
from planiverse.problems.real_world_problems.base import RealWorldProblem

PERIOD = 7

class EpiState:
    def __init__(self, state, depth):
        self.state = state
        self.depth = depth
        self.literals = frozenset([])
        self.__update__()

    def __update__(self):
        # Convert the np.array into at(x,y,val) literals.
        self.literals = frozenset([])


class EpiEnv(RealWorldProblem):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__("EpiEnv")

    def fix_index(self, index):
        #  list all json files in the directory
        # read all scenarios from json files
        index_scenario_map = {}
        for idx, json_file in enumerate([os.path.join(os.path.dirname(__file__), 'jsons', f) for f in os.listdir(os.path.join(os.path.dirname(__file__),'jsons')) if f.endswith('.json')]):
            with open(json_file, 'r') as f:
                data = json.load(f)
            index_scenario_map[idx] = data
        assert index in index_scenario_map, f"Scenario {index} not found in index_scenario_map"
        self.scenario = index_scenario_map[index]

    def __reset__(self, session, vac_starts):
        self.epi = construct_epidemic(session)
        total_population = np.sum(self.epi.static.default_state.obs.current_comp)
        obs_count = self.epi.static.compartment_count * self.epi.static.locale_count * self.epi.static.group_count
        action_count = 0
        action_param_count =  0
        for itv in self.epi.static.interventions:
            if not itv.is_cost:
                action_count += 1
                action_param_count += len(itv.cp_list)
        self.act_domain = np.zeros((action_param_count, 2), dtype=np.float32)
        index = 0
        for itv in self.epi.static.interventions:
            if not itv.is_cost:
                for cp in itv.cp_list:
                    self.act_domain[index, 0] = cp.min_value
                    self.act_domain[index, 1] = cp.max_value
                    index += 1
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=0, high=1, shape=(action_count,), dtype=np.float32)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=total_population, shape=(obs_count,), dtype=np.float32)

        self.time_passed = 0 # To keep track of how many days have passed 
        self.vac_starts = vac_starts # number of days to prepare a vaccination / make it available 
    
    def reset(self):
        self.__reset__(self.scenario, 10)
        state = self.epi.reset()
        return EpiState(state, 0), {}

    def is_goal(self, state):
        return state.depth >= self.epi.static.schedule.horizon

    def is_terminal(self, state):
        return False # there are stuck states in this environment.
    
    def successors(self, state):
        ret = []
        for action in self.actionslist:
            successor_state = self.env.generative_step(state, action)[0]
            if successor_state == state: continue
            ret.append((action, NASimState(successor_state, self.env.network)))
        return ret
    


    def step(self, action):
        if self.time_passed < self.vac_starts: 
            action[0] = 0 
        # print("================================================================")
        # print("time elapsed: ", self.time_passed) 
        # print("action: ", action) 
        # print("================================================================")

        self.time_passed += PERIOD 


        expanded_action = np.zeros(len(self.act_domain), dtype=np.float32)
        index = 0
        for i in range(len(self.act_domain)):
            if self.act_domain[i, 0] == self.act_domain[i, 1]:
                expanded_action[i] = self.act_domain[i, 0]
            else:
                expanded_action[i] = action[index]
                index += 1

        epi_action = []
        index = 0
        for itv_id, itv in enumerate(self.epi.static.interventions):
            if not itv.is_cost:
                epi_action.append(construct_act(itv_id, expanded_action[index:index+len(itv.cp_list)]))
                index += len(itv.cp_list)

        total_r = 0
        for i in range(PERIOD):
            state, r, done = self.epi.step(epi_action)
            total_r += r
            if done:
                self.time_passed = 0 
                break
        return state.obs.current_comp.flatten(), total_r, done, dict()

