

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
    def __str__(self):
        return f"{self.name} = {self.cpv_list[0]}"


class EpiState:
    def __init__(self, state, depth, static):
        self.state = state
        self.depth = depth
        self.static = static
        self.literals = frozenset([])
        self.__update__()
    
    def __hash__(self):
        # Combine the hash of literals and depth for uniqueness
        return hash((self.literals, self.depth))

    def __vectorize__(self):
        return sorted(filter(lambda c: c[0] in ['I', 'R'], [(self.static.get_property_name('compartment', i), int(np.sum(self.state.obs.current_comp[i]))) for i in range(self.static.compartment_count)]), key=lambda x: x[0].lower())
        return sorted([(self.static.get_property_name('compartment', i), int(np.sum(self.state.obs.current_comp[i]))) for i in range(self.static.compartment_count)], key=lambda x: x[0].lower())

    def __update__(self):
        # self.literals = frozenset([f'depth({self.depth})'])
        current_comp = self.state.obs.current_comp
        kpi_values = [(self.static.get_property_name('compartment', i), str(int(np.sum(current_comp[i])))) for i in range(self.static.compartment_count)]
        # self.literals |= frozenset(map(lambda kv: f'{kv[0].lower()}({kv[1]})', kpi_values))
        self.literals = frozenset([' ^ '.join(map(lambda kv: f'{kv[0].lower()}({kv[1]})', kpi_values + [('depth', str(self.depth)), 'hash', str(hash(self))]))])
        pass
    
    def __compute_cosine_similarity__(self, other):
        s1 = np.array(list(map(lambda o:int(o[1]), self.__vectorize__())))
        s2 = np.array(list(map(lambda o:int(o[1]), other.__vectorize__())))
        norm1 = np.linalg.norm(s1)
        norm2 = np.linalg.norm(s2)
        if norm1 == 0 or norm2 == 0: return 0.0  # No similarity if one vector is all zeros
        return np.dot(s1, s2) / (norm1 * norm2)

    def __eq__(self, other):
        v1 = np.array(list(map(lambda o: int(o[1]), self.__vectorize__())))
        v2 = np.array(list(map(lambda o: int(o[1]), other.__vectorize__())))

        # print(f"v1: {v1}, v2: {v2}, diff: {np.linalg.norm(v1 - v2, ord=1) < 50}")

        # I would say for every month let's increase the threshold by 10.
        initial_threshold = 50
        if self.depth > 20: initial_threshold += 10
        if self.depth > 30: initial_threshold += 5
        if self.depth > 60: initial_threshold += 10
        if self.depth > 90: initial_threshold += 10
        if self.depth > 120: initial_threshold += 10
        if self.depth > 150: initial_threshold += 10
        if self.depth > 180: initial_threshold += 10

        return np.linalg.norm(v1 - v2, ord=1) < initial_threshold
        return np.linalg.norm(v1 - v2, ord=p)
        
        [(i[1]-j[1])**2 for i,j in zip(self.__vectorize__(), other.__vectorize__())]
        
        
        return abs(sum(map(lambda c: c[1], self.__vectorize__())) - sum(map(lambda c: c[1], other.__vectorize__()))) < 10
        return self.__compute_cosine_similarity__(other) >= 0.99
        # If states are not in the same depth then they are not equal.
        if self.depth != other.depth: return False
        
        return max(0.0, min(1.0, self.__compute_cosine_similarity__(other))) >= 0.9
        return self.state == other.state and self.depth == other.depth
    
    def __repr__(self):
        sir_model_value = {self.static.get_property_name('compartment', i): str(int(np.sum(self.state.obs.current_comp[i]))) for i in range(self.static.compartment_count)}
        return f"EpiState(depth={self.depth}, {', '.join(map(lambda kv: f'{kv[0]}={kv[1]}', sir_model_value.items()))})"
    

class EpiEnv(RealWorldProblem):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, delay_vaccination_time, horizon):
        super().__init__("EpiEnv")
        self.vac_starts = delay_vaccination_time
        self.horizon = horizon

    def __reset__(self, session):
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
                
        # self.interventionslist = [] #list(map(lambda i: EpiAction(i[0], i[1]), enumerate(filter(lambda itv: not itv.is_cost, self.epi.static.interventions))))
        self.costs             = [] #list(map(lambda i: EpiCost(i[0], i[1]), enumerate(filter(lambda itv: itv.is_cost, self.epi.static.interventions))))
        self.interventions     = [] #list(chain.from_iterable([[itv.create_action(i) for i in np.linspace(itv.min_value, itv.max_value, 3)] for itv in self.interventionslist]))

        self.costs = list()
        self.costs = list(map(lambda i: EpiCost(i[0], i[1]), filter(lambda itv: itv[-1].is_cost, enumerate(self.epi.static.interventions))))
        self.basic_interventions = list(map(lambda i: EpiAction(i[0], i[1]), filter(lambda itv: not itv[-1].is_cost, enumerate(self.epi.static.interventions))))
        
        discretised_actions = [[act.create_action(i) for i in np.linspace(act.min_value, act.max_value, self.itv_split)] for act in self.basic_interventions]



        # discretised_actions = list(map(list, product(*[[act.create_action(i) for i in np.linspace(act.min_value, act.max_value, self.itv_split)] for act in self.basic_interventions])))
        # [combinations(self.basic_interventions, r) for r in range(1, len(self.basic_interventions)+1)]
        pass
        
        self.interventions = list(product(*discretised_actions))
        # for r in range(1, len(self.basic_interventions) + 1):  # r is the size of the combination
        #     # now for every combination, we need to remove the intervention with zero values
        #     self.interventions.extend(combinations(self.basic_interventions, r))
        self.interventions = list(map(list, self.interventions))[1:]
        # make sure that at least 50% of the interventions are not zero.
        self.interventions = list(filter(lambda i: [a.cpv_list[0] for a in i].count(0) >= len(self.basic_interventions)/2, self.interventions))
        
        self.action_str_map = {' ^ '.join(map(str, action)): action + self.costs for action in self.interventions + [self.__disable_vaccination__(0, a) for a in self.interventions][1:]}


        # remove the intervention where all values are 0.0
        # self.interventions = list(filter(lambda itv: not all([i == 0.0 for i in itv.cpv_list]), self.interventions))
        pass
        
    def __disable_vaccination__(self, depth, action):
        vaccination_index = next((i for i in range(len(action)) if action[i].name == 'Vaccination'), None)
        if depth >= self.vac_starts or vaccination_index is None: return action
        ret_action = action[:]
        # remove the vaccination action from the ret_action.
        ret_action.pop(vaccination_index)
        return ret_action

    def __perform_action__(self, state, action):
        _execute_action = list(filter(lambda a: isinstance(a, EpiCost) or a.cpv_list[0] > 0, action))
        # remove the actions with zero values to save computation time.
        next_state, delta_parameter = self.epi.get_next_state(state.state, _execute_action)
        for i in range(1, PERIOD+1):
            # if len(action) == 0: return EpiState(next_state, state.depth + i, self.epi.static)
            next_state, delta_parameter = self.epi.get_next_state(next_state, _execute_action)
        return EpiState(next_state, state.depth + PERIOD, self.epi.static)

    def reset(self):
        self.__reset__(self.scenario)
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
        if state.depth >= self.horizon:
            pass # for development.
        return state.depth >= self.horizon
        sir_model_value = {self.epi.static.get_property_name('compartment', i): np.sum(state.state.obs.current_comp[i]) for i in range(self.epi.static.compartment_count)}
        # I guess a goal state should be if there are no infected people.
        return sir_model_value['I'] == 0.0

    def is_terminal(self, state):
        # A better terminal state is the tree searched for 356 days.
        return False
        sir_model_value = {self.epi.static.get_property_name('compartment', i): np.sum(state.state.obs.current_comp[i]) for i in range(self.epi.static.compartment_count)}
        # So if all ppls are infected then this is a terminal state.
        return False # there are stuck states in this environment.
    
    def successors(self, state):
        ret = []
        performed_actions = set()
        for idx, action in enumerate(self.interventions):
            # print(f"idx: {idx}")
            action = self.__disable_vaccination__(state.depth, action)
            action_str = ' ^ '.join(map(str, action))
            # check if the action is already performed.
            if action_str in performed_actions: continue # when vaccination is not applied, then some actions will be repeated.
            if not any([a.cpv_list[0] != 0.0 for a in filter(lambda o: isinstance(o, EpiAction), action)] ): continue # avoid actions with no interventions applied.
            successor_state = self.__perform_action__(state, action + self.costs)
            if successor_state == state: continue
            # we need to stringify the action for _BFS_SEARCH
            ret.append((' ^ '.join(map(str, action + self.costs)), successor_state))
            performed_actions.add(action_str)
        return ret
    
    def simulate(self, plan):
        state, _ = self.reset()
        ret_states_trace = [state]
        for action in plan:
            state = self.__perform_action__(state, self.action_str_map[' ^ '.join(filter(lambda a: not ('cost' in a.lower()), action.split(' ^ ')))])
            ret_states_trace.append(state)
        return ret_states_trace