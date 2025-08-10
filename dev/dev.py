

from planiverse.problems.real_world_problems.urban_planning.environment import UrbanPlanningEnv



env = UrbanPlanningEnv('Kendall Square', 
                       "/Users/mustafafaisal/Developer/Planiverse/venv/lib/python3.12/site-packages/planiverse/problems/real_world_problems/urban_planning/cities/Kendall_Square_data",
                       100)
init_state, _ = env.reset()

successors = env.successors(init_state)

for action, state in successors:
    print(f"Action: {action}, State: {state}")
    pass
