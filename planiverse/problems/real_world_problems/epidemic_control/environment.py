

import os
import json
import numpy as np
from itertools import combinations
from gym import spaces
from planiverse.problems.real_world_problems.epidemic_control.epipolicy.core.epidemic import construct_epidemic
from planiverse.problems.real_world_problems.epidemic_control.epipolicy.obj.act import construct_act, get_normalized_action
from planiverse.problems.real_world_problems.base import RealWorldProblem

PERIOD = 7

class EpiState:
    def __init__(self, state, depth, static):
        self.state = state
        self.depth = depth
        self.static = static
        self.literals = frozenset([])
        self.__update__()

    def __update__(self):
        self.literals = frozenset([f'depth({self.depth})'])
        current_comp = self.state.obs.current_comp
        kpi_values = [(self.static.get_property_name('compartment', i), np.sum(current_comp[i])) for i in range(self.static.compartment_count)]
        self.literals |= frozenset(map(lambda kv: f'{kv[0].lower()}({str(kv[1]).replace(".","_").replace("+","_plus_").replace("-","minus")})', kpi_values))
        pass
    
    def __eq__(self, other):
        return self.state == other.state and self.depth == other.depth

class EpiEnv(RealWorldProblem):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__("EpiEnv")

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
        # I have my doubts about this actionlist, we need to recheck this.
        self.actionslist  = list(map(lambda a: np.array(a),combinations([i*0.1 for i in range(0,10)], 2)))
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=total_population, shape=(obs_count,), dtype=np.float32)
        self.time_passed = 0 # To keep track of how many days have passed 
        self.vac_starts = vac_starts # number of days to prepare a vaccination / make it available 
    
    def __perform_action__(self, state, action):
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
        
        for i in range(PERIOD):
            normalized_action = get_normalized_action(self.epi.static, epi_action)
            next_state, delta_parameter = self.epi.get_next_state(state.state, normalized_action)

        return EpiState(next_state, state.depth + PERIOD, self.epi.static)

    def reset(self):
        self.__reset__(self.scenario, 10)
        state = self.epi.reset()
        return EpiState(state, 0, self.epi.static), {}

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

    def is_goal(self, state):
        return state.depth >= self.epi.static.schedule.horizon

    def is_terminal(self, state):
        return False # there are stuck states in this environment.
    
    def successors(self, state):
        ret = []
        for action in self.actionslist:
            successor_state = self.__perform_action__(state, action)
            if successor_state == state: continue
            # we need to stringify the action for _BFS_SEARCH
            ret.append(('^'.join([f'({idx}={v})' for idx, v in enumerate(action)]), successor_state))
        return ret
    

