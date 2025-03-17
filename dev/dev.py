import os
from absl import flags, app

import yaml
import pickle

from planiverse.problems.real_world_problems.urban_planning.envs.city import CityEnv, PlanClient, CityState

obj_filepath = '/Users/mustafafaisal/Developer/Planiverse/planiverse/problems/real_world_problems/urban_planning/cfg/test_data/synthetic/objectives_grid.yaml'
objectives_plan = yaml.safe_load(open(obj_filepath, 'r'))

init_plan_filepath = '/Users/mustafafaisal/Developer/Planiverse/planiverse/problems/real_world_problems/urban_planning/cfg/test_data/synthetic/init_plan_grid.pickle'
init_plan = pickle.load(open(init_plan_filepath, 'rb'))


state_cfg = {
    'objectives': objectives_plan,
    'init_plan': init_plan
}

init_state = CityState(cfg = state_cfg)





plc = PlanClient(objectives_plan, init_plan)






from planiverse.problems.real_world_problems.urban_planning.utils.config import Config


sandboxdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sandboxdev')
os.makedirs(sandboxdir, exist_ok=True)

cfg = Config("hlg", "111", sandboxdir, "root-dir", "gsca")

plan_objectives_filepath = ''
init_plan_filepath = ''





env = CityEnv(cfg)

ret = env.reset()

pass

# # We need a successor function that takes a state and an action and returns a new state
# # We need to define a goal state.

# # How the state is defined? 
# # urban geographic elements as nodes and spatial contiguity as edges. 

# # The problem is modelled as:
# # graph to describe the topology of cities in arbitrary forms and formulate urban planning as a sequential decision-making problem on the graph.

# # The problem is formulated as:
# # spatial planning as a sequential decision-making problem on the graph and conduct planning at the topology level instead of geometry.

# # We have an objective metrics.
# # evaluate the efficiency of the spatial layout according to the existing literature on service31, ecology32 and traffic33–35, which provide a comprehensive evaluation regarding the accessibility to basic urban services, the coverage of greenness and the efficiency and rationality of the road network.

# # Then the state will be an instance of the graph.

# # Now what the action space looks like:
# # Cartesian product of three sub-spaces, including:
# # 1. what to plan, 
# # 2. where to plan and 
# # 3. how to plan


pass
