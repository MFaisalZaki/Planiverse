

import os
import json
import numpy as np
from itertools import combinations, product, chain
from gym import spaces
from planiverse.problems.real_world_problems.epidemic_control.epipolicy.core.epidemic import construct_epidemic
from planiverse.problems.real_world_problems.epidemic_control.epipolicy.obj.act import Act, construct_act, get_default_act, get_normalized_action
from planiverse.problems.real_world_problems.base import RealWorldProblem

PERIOD = 7

class EpiAction:
    def __init__(self, index, intervention_details):
        self.name          = intervention_details.name
        self.index         = index
        self.locale_regex  = '*' 
        self.cpv_list      = np.array([i.default_value for i in intervention_details.cp_list], dtype=np.float32)
        self.default_value = self.cpv_list[0]
        self.min_value     = self.cpv_list[1]
        self.max_value     = self.cpv_list[2]
        self.itv_details   = intervention_details
    
    def create_action(self, value):
        ret_action = EpiAction(self.index, self.itv_details)
        ret_action.cpv_list[0] = value
        ret_action.default_value = value
        return ret_action

class EpiCost:
    def __init__(self, index, intervention_details):
        self.name          = intervention_details.name
        self.index         = index
        self.locale_regex  = '*' 
        self.cpv_list      = np.array([i.default_value for i in intervention_details.cp_list], dtype=np.float32)
        self.itv_details   = intervention_details

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
        
        self.interventionslist = [] #list(map(lambda i: EpiAction(i[0], i[1]), enumerate(filter(lambda itv: not itv.is_cost, self.epi.static.interventions))))
        self.costs             = [] #list(map(lambda i: EpiCost(i[0], i[1]), enumerate(filter(lambda itv: itv.is_cost, self.epi.static.interventions))))
        self.interventions     = [] #list(chain.from_iterable([[itv.create_action(i) for i in np.linspace(itv.min_value, itv.max_value, 3)] for itv in self.interventionslist]))
        
        for idx, itv in enumerate(self.epi.static.interventions):
            if itv.is_cost:
                self.costs += [EpiCost(idx, itv)]
            else:
                act = EpiAction(idx, itv)
                self.interventions += [act.create_action(i) for i in np.linspace(act.min_value, act.max_value, 3)]

        self.time_passed = 0 # To keep track of how many days have passed 
        self.vac_starts = vac_starts # number of days to prepare a vaccination / make it available 
    
    def __perform_action__(self, state, action):
        next_state, delta_parameter = self.epi.get_next_state(state.state, action)
        for i in range(PERIOD-1):
            next_state, delta_parameter = self.epi.get_next_state(next_state, action)
        return EpiState(next_state, state.depth + PERIOD, self.epi.static)

    def reset(self):
        self.__reset__(self.scenario, 10)
        state = self.epi.reset()
        return EpiState(state, 0, self.epi.static), {}

    def fix_index(self, index):
        #  list all json files in the directory
        # read all scenarios from json files
        index_scenario_map = {}
        for idx, json_file in enumerate(sorted([os.path.join(os.path.dirname(__file__), 'jsons', f) for f in os.listdir(os.path.join(os.path.dirname(__file__),'jsons')) if f.endswith('.json')])):
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
        for action in self.interventions:
            successor_state = self.__perform_action__(state, [action] + self.costs)
            if successor_state == state: continue
            # we need to stringify the action for _BFS_SEARCH
            ret.append((action, successor_state))
        return ret
