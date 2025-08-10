
# This implementation is based on this paper:
# "AI Agent as Urban Planner: Steering Stakeholder Dynamics in Urban Planning via Consensus-based Multi-Agent Reinforcement Learning"


# So a state is a graph representing the urban environment.
# The node represents a parcel which can be: residential zone 𝑟, office 𝑜, green space 𝑔, commercial zone 𝑐, facilities 𝑓
# The edges represents the spatial relationships between these parcels.

# As for the action, it is a reassignment of land use for a specific parcel.

# Note files should be named as: node_info.csv and node_pairs_knn4.csv. We used the same format as the original dataset.

# TODO: We need to figure out the numbers of the land type.

import os
import pandas as pd
import networkx as nx
from enum import Enum
from copy import deepcopy

from planiverse.problems.real_world_problems.base import RealWorldProblem

class LandUseType(Enum):
    RESIDENTIAL = 'r'
    OFFICE      = 'o'
    GREEN_SPACE = 'g'
    COMMERCIAL  = 'c'
    FACILITIES  = 'f'
    EMPTY       = 'n'  # No land use, e.g., water bodies or undeveloped land

class UrbanEnvState:
    
    def __init__(self, urban_graph, depth):
        self.urban_graph = urban_graph
        self.depth = depth
        self.literals = frozenset([])
        self.__update__()

    def __update__(self):
        self.literals |= frozenset(map(lambda e: f'land_{int(e)}_is_{self.urban_graph.nodes[e]["type"].value}' , self.urban_graph.nodes))
    
    def __eq__(self, other):
        if not isinstance(other, UrbanEnvState): return False
        # Two states are the same if they have the same land usage.
        return nx.utils.misc.graphs_equal(self.urban_graph, other.urban_graph)


# Use land actions.
class UrbanPlanAction:
    def __init__(self, landusetype):
        self.landusetype = landusetype
    
    def __call__(self, state):
        self.converted_nodes = []
        new_state = self.apply(state)
        self.actionstr = 'action_' + '__'.join(map(lambda a: f'{int(a[0])}_{a[2].value}', self.converted_nodes))
        return new_state
    
    def __str__(self):
        return self.actionstr
    
    def __get_lands_to_convert__(self, g, landtype:LandUseType):
        return list(filter(lambda n: g.nodes[n]['type'] == landtype, g.nodes))

    def apply(self, state):
        # split half evenly empty space between r, o, g, c, f
        landuse = deepcopy(state.urban_graph)
        to_convert_lands = self.__get_lands_to_convert__(landuse, LandUseType.EMPTY)
        if len(to_convert_lands) > 0: self.convert(landuse, to_convert_lands)
        return UrbanEnvState(landuse, state.depth+1)
    
    def convert(self, landuse, to_convert_lands):
        assert False, "This method should be implemented in the subclass."

class ConvertEmptyAction(UrbanPlanAction):
    def __init__(self):
        super().__init__(LandUseType.EMPTY)

    def convert(self, landuse, to_convert_lands):
        # Convert half of the empty spaces evenly between r, o, g, c, f
        to_convert_lands = to_convert_lands[:len(to_convert_lands)//2]
        for new_type in [LandUseType.RESIDENTIAL, LandUseType.OFFICE, LandUseType.GREEN_SPACE, LandUseType.COMMERCIAL, LandUseType.FACILITIES]:
            for land in to_convert_lands:
                self.converted_nodes.append((land, landuse.nodes[land]['type'], new_type))
                landuse.nodes[land]['type'] = new_type

class ConvertOfficesAction(UrbanPlanAction):
    def __init__(self):
        super().__init__(LandUseType.OFFICE)

    def convert(self, landuse, to_convert_lands):
        # So half of the offices will be splited 20% to be g and 80% to be commercial
        to_convert_lands = to_convert_lands[:len(to_convert_lands)//2]
        
        to_green_space = to_convert_lands[:int(len(to_convert_lands)*0.2)]
        to_commercial = to_convert_lands[int(len(to_convert_lands)*0.2):]
        
        for land in to_green_space:
            self.converted_nodes.append((land, landuse.nodes[land]['type'], LandUseType.GREEN_SPACE))
            landuse.nodes[land]['type'] = LandUseType.GREEN_SPACE
        
        for land in to_commercial:
            self.converted_nodes.append((land, landuse.nodes[land]['type'], LandUseType.COMMERCIAL))
            landuse.nodes[land]['type'] = LandUseType.COMMERCIAL

        
class ConvertCommercialAction(UrbanPlanAction):
    def __init__(self):
        super().__init__(LandUseType.COMMERCIAL)

    def convert(self, landuse, to_convert_lands):
        # So half of the c will be spliited to 20% to be g and 80% to be f.
        to_convert_lands = to_convert_lands[:len(to_convert_lands)//2]
        to_green_space = to_convert_lands[:int(len(to_convert_lands)*0.2)]
        to_facilities = to_convert_lands[int(len(to_convert_lands)*0.2):]

        for land in to_green_space:
            self.converted_nodes.append((land, landuse.nodes[land]['type'], LandUseType.GREEN_SPACE))
            landuse.nodes[land]['type'] = LandUseType.GREEN_SPACE

        for land in to_facilities:
            self.converted_nodes.append((land, landuse.nodes[land]['type'], LandUseType.FACILITIES))
            landuse.nodes[land]['type'] = LandUseType.FACILITIES

class ConvertFacilitiesAction(UrbanPlanAction):
    def __init__(self):
        super().__init__(LandUseType.FACILITIES)

    def convert(self, landuse, to_convert_lands):
        # so 20% of the facilities will be converted to 20% green space and 80% to commercial.
        to_convert_lands = to_convert_lands[:int(len(to_convert_lands)*0.2)]
        to_green_space = to_convert_lands[:int(len(to_convert_lands)*0.2)]
        to_commercial = to_convert_lands[int(len(to_convert_lands)*0.2):]

        for land in to_green_space:
            self.converted_nodes.append((land, landuse.nodes[land]['type'], LandUseType.GREEN_SPACE))
            landuse.nodes[land]['type'] = LandUseType.GREEN_SPACE

        for land in to_commercial:
            self.converted_nodes.append((land, landuse.nodes[land]['type'], LandUseType.COMMERCIAL))
            landuse.nodes[land]['type'] = LandUseType.COMMERCIAL

class UrbanPlanningEnv(RealWorldProblem):
    def __init__(self, horizon: int):
        self.index   = None
        self.horizon = horizon
        self.actions = [ConvertEmptyAction, ConvertOfficesAction, ConvertCommercialAction, ConvertFacilitiesAction]


    def reset(self):
        # This is an initial map until I figure it out.
        landuse_map = {
            -1.0 : LandUseType.EMPTY,
             0.0 : LandUseType.RESIDENTIAL,
             1.0 : LandUseType.OFFICE,
             2.0 : LandUseType.GREEN_SPACE,
             3.0 : LandUseType.COMMERCIAL,
             4.0 : LandUseType.FACILITIES
        }

        self.graph = nx.Graph()
        for _, row in self.node_info.iterrows():
            attributes = row.to_dict()
            node_id = attributes.pop('node_id', None)
            assert node_id is not None, "Node ID cannot be None."
            attributes['type'] = landuse_map[attributes['landuse_type']]
            self.graph.add_node(node_id, **attributes)
        
        for _, row in self.node_pairs.iterrows():
            attributes = row.to_dict()
            from_node = attributes.pop('node', None)
            to_node = attributes.pop('node_adj', None)
            assert from_node is not None and to_node is not None, "Node IDs cannot be None."
            self.graph.add_edge(from_node, to_node, **attributes)

        return UrbanEnvState(self.graph, 0), {}
    
    def fix_index(self, index):
        city_info_index_map = {
            0: {
                'name': 'Kendall Square',
                'info': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cities', 'Kendall_Square_data')
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
        # A better terminal state is the tree searched for 356 days.
        return False
    
    # Returns a list of [action, successor_state]
    def successors(self, state):
        ret = []
        for idx, actiontype in enumerate(self.actions):
            action = actiontype()
            successor_state = action(state)
            if successor_state == state: continue
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
    