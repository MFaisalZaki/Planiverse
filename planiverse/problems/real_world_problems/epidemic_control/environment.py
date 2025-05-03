

import os
import json
import numpy as np
from numpy import dot
from numpy.linalg import norm

from copy import deepcopy
from collections import defaultdict
from itertools import combinations, product, chain
from gym import spaces
from planiverse.problems.real_world_problems.epidemic_control.epipolicy.core.epidemic import construct_epidemic
from planiverse.problems.real_world_problems.epidemic_control.epipolicy.core.executor import Executor
from planiverse.problems.real_world_problems.epidemic_control.epipolicy.obj.control_parameter import ControlParameter
from planiverse.problems.real_world_problems.base import RealWorldProblem

PERIOD = 7

class EpiAction:
    def __init__(self, index, intervention_details):
        self.index         = index
        self.name          = intervention_details.name
        self.itv_details   = intervention_details
        self.locale_regex  = '*' 
        self.cpv_list      = np.array([round(i.default_value,2) for i in intervention_details.cp_list], dtype=np.float32)
        self.control_parameter_index = next((i for i in range(len(intervention_details.cp_list)) if intervention_details.cp_list[i].name in ["compliance", "degree", "percentage"]), None)
        self.min_value     = intervention_details.cp_list[self.control_parameter_index].min_value
        self.max_value     = intervention_details.cp_list[self.control_parameter_index].max_value
    
    def __str__(self):
        return f"{self.name} = {self.cpv_list[self.control_parameter_index]}"
    
    def create_action(self, value):
        ret_action = EpiAction(self.index, self.itv_details)
        assert value >= self.min_value and value <= self.max_value, f"Value {value} is out of bounds [{self.min_value}, {self.max_value}]"
        ret_action.cpv_list[self.control_parameter_index] = value
        return ret_action

class EpiCost:
    def __init__(self, index, intervention_details):
        self.name          = intervention_details.name
        self.index         = index
        self.locale_regex  = '*' 
        self.cpv_list      = np.array([round(i.default_value,2) for i in intervention_details.cp_list], dtype=np.float32)
        self.itv_details   = intervention_details

class EpiState:
    def __init__(self, state, depth, static):
        self.state = state
        self.depth = depth
        self.static = static
        self.literals = frozenset([])
        self.__update__()

    def __vectorize__(self):
        return sorted([(self.static.get_property_name('compartment', i), str(int(np.sum(self.state.obs.current_comp[i])))) for i in range(self.static.compartment_count)], key=lambda x: x[0].lower())

    def __update__(self):
        self.literals = frozenset([f'depth({self.depth})'])
        current_comp = self.state.obs.current_comp
        kpi_values = [(self.static.get_property_name('compartment', i), str(int(np.sum(current_comp[i])))) for i in range(self.static.compartment_count)]
        self.literals |= frozenset(map(lambda kv: f'{kv[0].lower()}({kv[1]})', kpi_values))
    
    def __eq__(self, other):
        s1 = np.array(list(map(lambda o:int(o[1]), self.__vectorize__())))
        s2 = np.array(list(map(lambda o:int(o[1]), other.__vectorize__())))
        norm1 = np.linalg.norm(s1)
        norm2 = np.linalg.norm(s2)
        if norm1 == 0 or norm2 == 0: return 0.0  # No similarity if one vector is all zeros
        sim = np.dot(s1, s2) / (norm1 * norm2)
        return max(0.0, min(1.0, sim)) >= 0.9
        return self.state == other.state and self.depth == other.depth
    
    def __repr__(self):
        sir_model_value = {self.static.get_property_name('compartment', i): str(int(np.sum(self.state.obs.current_comp[i]))) for i in range(self.static.compartment_count)}
        return f"EpiState(depth={self.depth}, {', '.join(map(lambda kv: f'{kv[0]}={kv[1]}', sir_model_value.items()))})"
    

class EpiEnv(RealWorldProblem):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__("EpiEnv")

    def __reset__(self, session, vac_starts):
        self.itv_split = 2

        # the easiest way is to modify the interventions before creating the pandemic env.
        updated_optimze_interventions = list()
        updated_interventions = list()
        for itv in session['optimize']['interventions']:
            control_param_index = next((i for i in range(len(itv['control_params'])) if itv['control_params'][i]['name'] in ["compliance", "degree", "percentage"]), None)
            intervention_index = next((i for i in range(len(session['interventions'])) if itv['name'] == session['interventions'][i]['name']), None)
            intervention_control_param_index = next((i for i in range(len(session['interventions'][intervention_index]['control_params'])) if session['interventions'][intervention_index]['control_params'][i]['name'] in ["compliance", "degree", "percentage"]), None)

            for i in np.linspace(itv['control_params'][control_param_index]['min_value'], itv['control_params'][control_param_index]['max_value'], self.itv_split):
                updated_itv = deepcopy(session['interventions'][intervention_index].copy())
                updated_itv['control_params'][intervention_control_param_index]['default_value'] = str(i)
                updated_itv['name'] = f"{itv['name']}_value_{str(i).replace('.','_')}"
                updated_interventions.append(updated_itv)
                
                updated_opt_itv = deepcopy(itv.copy())
                updated_opt_itv['name'] = f"{itv['name']}_value_{str(i).replace('.','_')}"
                updated_optimze_interventions.append(updated_opt_itv)
        
        session['schedules'] = [] # remove any schedules created for this session.
        # session['optimize']['interventions'] = updated_optimze_interventions[:]
        # session['interventions'] = updated_interventions[:]

        self.epi = construct_epidemic(session)
                
        self.interventionslist = [] #list(map(lambda i: EpiAction(i[0], i[1]), enumerate(filter(lambda itv: not itv.is_cost, self.epi.static.interventions))))
        self.costs             = [] #list(map(lambda i: EpiCost(i[0], i[1]), enumerate(filter(lambda itv: itv.is_cost, self.epi.static.interventions))))
        self.interventions     = [] #list(chain.from_iterable([[itv.create_action(i) for i in np.linspace(itv.min_value, itv.max_value, 3)] for itv in self.interventionslist]))

        self.costs = list()
        self.costs = list(map(lambda i: EpiCost(i[0], i[1]), filter(lambda itv: itv[-1].is_cost, enumerate(self.epi.static.interventions))))
        self.interventions = list(map(lambda i: EpiAction(i[0], i[1]), filter(lambda itv: not itv[-1].is_cost, enumerate(self.epi.static.interventions))))

        self.interventions = list(map(list, product(*[[act.create_action(i) for i in np.linspace(act.min_value, act.max_value, self.itv_split)] for act in self.interventions])))
        self.vac_starts = vac_starts # number of days to prepare a vaccination / make it available 
    
    def __disable_vaccination__(self, state, action):
        if state.depth >= self.vac_starts: return action
        vaccination_index = next((i for i in range(len(action)) if action[i].name == 'Vaccination'), None)
        ret_action = action[:]
        ret_action[vaccination_index].cpv_list[ret_action[vaccination_index].control_parameter_index].default_value = 0.0
        return ret_action

    def __perform_action__(self, state, action):
        # TODO: avoid vaccination if the time did not pass.
        # action = self.__disable_vaccination__(state, action)
        # print(f"{' ^ '.join(map(str, action))}")
        next_state, delta_parameter = self.epi.get_next_state(state.state, action)
        for i in range(PERIOD-1):
            # action = self.__disable_vaccination__(next_state, action)
            next_state, delta_parameter = self.epi.get_next_state(next_state, action)
        return EpiState(next_state, state.depth + PERIOD, self.epi.static)

    def reset(self):
        self.__reset__(self.scenario, 10)
        self.init_state = EpiState(self.epi.reset(), 0, self.epi.static)
        return self.init_state, {}

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
        if state.depth >= 356:
            pass # for development.
        return state.depth >= 356
        sir_model_value = {self.epi.static.get_property_name('compartment', i): np.sum(state.state.obs.current_comp[i]) for i in range(self.epi.static.compartment_count)}
        # I guess a goal state should be if there are no infected people.
        return sir_model_value['I'] == 0.0

    def is_terminal(self, state):
        # A better terminal state is the tree searched for 356 days.
        return state.depth >= 356
        sir_model_value = {self.epi.static.get_property_name('compartment', i): np.sum(state.state.obs.current_comp[i]) for i in range(self.epi.static.compartment_count)}
        # So if all ppls are infected then this is a terminal state.
        return False # there are stuck states in this environment.
    
    def successors(self, state):
        ret = []
        for idx, action in enumerate(self.interventions):
            # print(f"idx: {idx}")
            successor_state = self.__perform_action__(state, action + self.costs)
            if successor_state == state: continue
            # we need to stringify the action for _BFS_SEARCH
            ret.append((' ^ '.join(map(str, action)), successor_state))
        return ret
