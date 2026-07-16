
# This implementation is based on this paper:
# "AI Agent as Urban Planner: Steering Stakeholder Dynamics in Urban Planning via Consensus-based Multi-Agent Reinforcement Learning"


# So a state is a graph representing the urban environment.
# The node represents a parcel which can be: residential zone 𝑟, office 𝑜, green space 𝑔, commercial zone 𝑐, facilities 𝑓
# The edges represents the spatial relationships between these parcels.

# As for the action, it is a reassignment of land use for a specific parcel.

# Note files should be named as: node_info.csv and node_pairs_knn4.csv. We used the same format as the original dataset.

# TODO: We need to figure out the numbers of the land type.

import os
import math
import pandas as pd
import networkx as nx
import numpy as np
from enum import Enum
from copy import deepcopy


from planiverse.problems.real_world_problems.base import RealWorldProblem

# Share of a land-use class rezoned by a single action.
CHANGE_RATIO = 0.05

class LandUseType(Enum):
    RESIDENTIAL = 'r'
    OFFICE      = 'o'
    GREEN_SPACE = 'g'
    COMMERCIAL  = 'c'
    FACILITIES  = 'f'
    EMPTY       = 'n'  # No land use, e.g., water bodies or undeveloped land

landuse_map = {
    -1.0 : LandUseType.EMPTY,      # confirmed
    0.0  : LandUseType.RESIDENTIAL,# confirmed
    1.0  : LandUseType.OFFICE,     # confirmed
    2.0  : LandUseType.COMMERCIAL, # confirmed
    3.0  : LandUseType.FACILITIES, # confirmed
    4.0  : LandUseType.GREEN_SPACE # confirmed
}

landuse_map_reverse = {v: k for k, v in landuse_map.items()}

def update_landuse(land, t):
    land['landuse_type'] = landuse_map_reverse[t]
    land['type'] = t

class UrbanEnvState:
    
    def __init__(self, urban_graph, depth):
        self.urban_graph = urban_graph
        self.depth = depth
        self.literals = frozenset([])
        self.__update__()

    def __compute_sustainability_score__(self):
        # \text{Sustainability} \propto \frac{\#g + \#c + \#f}{\text{non-empty parcels}}
        green_area_count = list(filter(lambda e: self.urban_graph.nodes[e]['type'] == LandUseType.GREEN_SPACE, self.urban_graph.nodes))
        commercial_area_count = list(filter(lambda e: self.urban_graph.nodes[e]['type'] == LandUseType.COMMERCIAL, self.urban_graph.nodes))
        facility_area_count = list(filter(lambda e: self.urban_graph.nodes[e]['type'] == LandUseType.FACILITIES, self.urban_graph.nodes))
        total_parcels = len(list(filter(lambda e: self.urban_graph.nodes[e]['type'] != LandUseType.EMPTY, self.urban_graph.nodes)))
        return round((len(green_area_count) + len(commercial_area_count) + len(facility_area_count)) / total_parcels, 1)

    def __compute_diversity_score__(self):
        landuse_freq = {k:len(list(filter(lambda e: self.urban_graph.nodes[e]['type'] == k, self.urban_graph.nodes))) for k in LandUseType}
        landuse_freq.pop(LandUseType.EMPTY) # don't want this.
        proportions = [v / sum(landuse_freq.values()) for v in landuse_freq.values()]
        shannon_diversity = -sum(p * np.log(p) for p in proportions if p > 0)
        normalised_shannon_diversity = shannon_diversity / np.log(len(landuse_freq)) if len(landuse_freq) > 1 else 0
        return round(normalised_shannon_diversity, 1)

    def __update__(self):
        # self.literals |= frozenset(map(lambda e: f'land_{int(e)}_is_{self.urban_graph.nodes[e]["type"].value}' , self.urban_graph.nodes))
        # A more abstract representation to speed up the search for iw by breaking symmetry
        self.literals |= frozenset(map(lambda e: f'{e[0]}_{e[1]}', {k.value:len(list(filter(lambda e: self.urban_graph.nodes[e]['type'] == k, self.urban_graph.nodes))) for k in LandUseType}.items()))
        self.literals |= frozenset([f'depth_{self.depth}'])
        self.sustainability_score = self.__compute_sustainability_score__()
        self.diversity_score      = self.__compute_diversity_score__()
    
    def __eq__(self, other):
        return other.literals == self.literals

# Use land actions.
class UrbanPlanAction:
    def __init__(self, landusetype):
        self.landusetype    = landusetype
        self.converted_nodes = []
        self.actionstr       = f'action_{landusetype.value}_noop'

    def __call__(self, state):
        return self.apply(state)

    def __str__(self):
        return self.actionstr

    def __get_lands_of_type__(self, g, landtype:LandUseType):
        # return filter(lambda n: g.nodes[n]['type'] == landtype, g.nodes)
        return list(filter(lambda n: g.nodes[n]['type'] == landtype, g.nodes))

    def __select_lands__(self, lands):
        """!
        The share of lands a single action rezones, rounded up so that a class with fewer
        than 1/CHANGE_RATIO parcels is still rezonable.
        """
        return lands[:math.ceil(len(lands) * CHANGE_RATIO)]

    def __split_evenly__(self, lands, landtypes):
        """!
        Deals the selected lands round-robin across landtypes, so each target type gets an
        even share. Returns a list of (land, new_type) pairs, one entry per land.
        """
        return [(land, landtypes[i % len(landtypes)]) for i, land in enumerate(lands)]

    def apply(self, state):
        # Conversions are recomputed on every call, so an action can be replayed against
        # any state rather than only the one it was generated from.
        self.converted_nodes = []
        landuse = state.urban_graph.copy()
        for land, type in  self.__get_lands_to_convert__(landuse):
            self.converted_nodes.append((land, landuse.nodes[land]['type'], type))
            update_landuse(landuse.nodes[land], type)
        self.actionstr = 'action_' + '__'.join(map(lambda a: f'{int(a[0])}_{a[2].value}', self.converted_nodes)) \
                         if self.converted_nodes else f'action_{self.landusetype.value}_noop'
        return UrbanEnvState(landuse, state.depth+1)

    def __get_lands_to_convert__(self, landuse):
        assert False, "This method should be implemented in the subclass."

class ConvertEmptyAction(UrbanPlanAction):
    def __init__(self):
        super().__init__(LandUseType.EMPTY)

    def __get_lands_to_convert__(self, landuse):
        # Split the developed empty spaces evenly between r, o, g, c, f.
        # Every land used to be paired with every type, and apply() kept the last assignment,
        # so all of them silently became facilities.
        to_convert_lands = self.__select_lands__(self.__get_lands_of_type__(landuse, LandUseType.EMPTY))
        return self.__split_evenly__(to_convert_lands, [LandUseType.RESIDENTIAL, LandUseType.OFFICE, LandUseType.GREEN_SPACE, LandUseType.COMMERCIAL, LandUseType.FACILITIES])

class ConvertGreenSpaceAction(UrbanPlanAction):
    def __init__(self):
        super().__init__(LandUseType.GREEN_SPACE)

    def __get_lands_to_convert__(self, landuse):
        # Split the converted green spaces evenly between facilities and commercial.
        to_convert_lands = self.__select_lands__(self.__get_lands_of_type__(landuse, LandUseType.GREEN_SPACE))
        return self.__split_evenly__(to_convert_lands, [LandUseType.FACILITIES, LandUseType.COMMERCIAL])

class ConvertOfficesAction(UrbanPlanAction):
    def __init__(self):
        super().__init__(LandUseType.OFFICE)

    def __get_lands_to_convert__(self, landuse):
        # Offices become commercial.
        to_convert_lands = self.__select_lands__(self.__get_lands_of_type__(landuse, LandUseType.OFFICE))
        return self.__split_evenly__(to_convert_lands, [LandUseType.COMMERCIAL])

class ConvertCommercialAction(UrbanPlanAction):
    def __init__(self):
        super().__init__(LandUseType.COMMERCIAL)

    def __get_lands_to_convert__(self, landuse):
        # Commercial areas become facilities.
        to_convert_lands = self.__select_lands__(self.__get_lands_of_type__(landuse, LandUseType.COMMERCIAL))
        return self.__split_evenly__(to_convert_lands, [LandUseType.FACILITIES])

class ConvertFacilitiesAction(UrbanPlanAction):
    def __init__(self):
        super().__init__(LandUseType.FACILITIES)

    def __get_lands_to_convert__(self, landuse):
        # Split the converted facilities evenly between green space and commercial.
        to_convert_lands = self.__select_lands__(self.__get_lands_of_type__(landuse, LandUseType.FACILITIES))
        return self.__split_evenly__(to_convert_lands, [LandUseType.GREEN_SPACE, LandUseType.COMMERCIAL])

class RemoveResidentialAction(UrbanPlanAction):
    def __init__(self):
        super().__init__(LandUseType.EMPTY)

    def __get_lands_to_convert__(self, landuse):
        # Residential areas are cleared back to empty land.
        to_convert_lands = self.__select_lands__(self.__get_lands_of_type__(landuse, LandUseType.RESIDENTIAL))
        return self.__split_evenly__(to_convert_lands, [LandUseType.EMPTY])

class UrbanPlanningEnv(RealWorldProblem):
    def __init__(self, horizon: int):
        self.index   = None
        self.horizon = horizon
        self.actions = [ConvertGreenSpaceAction, ConvertEmptyAction, ConvertOfficesAction, ConvertCommercialAction, ConvertFacilitiesAction, RemoveResidentialAction]

    def reset(self):
        # This is an initial map until I figure it out.
        dummylist = []
        self.graph = nx.Graph()
        for _, row in self.node_info.iterrows():
            attributes = row.to_dict().copy()
            node_id = attributes.pop('node_id', None)
            assert node_id is not None, "Node ID cannot be None."
            dummylist.append(attributes['landuse_type'])
            attr = attributes | {'type': landuse_map[attributes['landuse_type']]}
            self.graph.add_node(node_id, **attr)

        for _, row in self.node_pairs.iterrows():
            attributes = row.to_dict().copy()
            from_node = attributes.pop('node', None)
            to_node = attributes.pop('node_adj', None)
            assert from_node is not None and to_node is not None, "Node IDs cannot be None."
            # skip nodes that are not in the current graph
            if from_node not in self.graph or to_node not in self.graph:
                continue
            self.graph.add_edge(from_node, to_node, **attributes)

        landuse_keys = set(map(lambda n: self.graph.nodes[n]['type'], self.graph.nodes))
        self.statsitics = {
            'landuse': {l: len(list(filter(lambda t: t == l, [self.graph.nodes[n]['type'] for n in self.graph.nodes]))) for l in landuse_keys}
        }
        return UrbanEnvState(self.graph, 0), {}
    
    def fix_index(self, index):
        city_info_index_map = {
            0: {
                'name': 'Kendall Square',
                'info': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cities', 'Kendall_Square_data')
            },
            1 :{
                'name': 'St Andrews',
                'info': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cities', 'st_andrews_data')
            }
        }

        assert index in city_info_index_map, f"Index {index} not found in city_info_index_map."
        self.index = index
        self.urban_name = city_info_index_map[index]['name']
        urban_info = city_info_index_map[index]['info']
        self.node_info  = pd.read_csv(os.path.join(urban_info, 'node_info.csv'))
        self.node_pairs = pd.read_csv(os.path.join(urban_info, 'node_pairs_knn4.csv'))

    def is_goal(self, state):
        return state.depth >= self.horizon
    
    def is_terminal(self, state):
        return False
    
    # Returns a list of [action, successor_state]
    def successors(self, state):
        ret = []
        for idx, actiontype in enumerate(self.actions):
            action = actiontype()
            successor_state = action(state)
            # A state carries its depth in its literals, so a successor can never compare equal
            # to its parent and `successor_state == state` would never fire. Drop the actions
            # that rezoned nothing (their land-use class is absent) instead.
            if not action.converted_nodes: continue
            # we need to stringify the action for _BFS_SEARCH
            ret.append((action, successor_state))
        return ret
    
    def simulate(self, plan):
        state, _ = self.reset()
        ret_states_trace = [state]
        for action in plan:
            state = action.apply(state)
            ret_states_trace.append(state)
        return ret_states_trace
    