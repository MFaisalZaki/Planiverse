
import os
from planiverse.problems.retro_games.super_mario_bros_gb import SuperMario, SuperMarioAction
from planiverse.planners.super_mario_planner import SuperMarioPlanner


current_file_path = os.path.dirname(os.path.abspath(__file__))
sml_romfile = os.path.join(current_file_path, "sandbox", "SuperMarioLand.gb")
dumpstates_images = os.path.join(current_file_path, "sandbox", "dump")

env = SuperMario(sml_romfile)
env.fix_index(0)
init_state, _ = env.reset()

# # # # plan = [SuperMarioAction('right')] * 0 + [SuperMarioAction('a')] * 0 + [SuperMarioAction('a+right')] * 4
# plan = [SuperMarioAction('right+b')] * 30
# state_trace = env.simulate(plan)
# for idx, (act, state) in enumerate(zip(plan, state_trace)):
#     state.save(sml_romfile, os.path.join(dumpstates_images, f"0_t_{idx}_{str(act)}.png"))



pass


# print(f"Initial State: {init_state}, Final state: {state_trace[-1]}")

# print('##########')

# env = SuperMario(sml_romfile)
# env.fix_index(0)
# init_state, _ = env.reset()

# plan = [SuperMarioAction(env.pyboy, 'left')]*4
# state_trace = env.simulate(plan)
# print(f"Initial State: {init_state}, Final state: {state_trace[-1]}")
pass
# # plan = plan[:len(state_trace)-1] # trim the plan to the executed length.
# for idx, (action, successor_state) in enumerate(zip(plan, state_trace[1:])):
#     for frameidx, image in enumerate(action.simulate(successor_state)):
#         image.save(os.path.join(dumpstates_images, f"{idx}_{frameidx}.png"))

planner = SuperMarioPlanner(env)
plan = planner.search(env)

pass

# # This is how to dump an sequence of images.
# for idx, (action, successor_state) in enumerate(env.successors(init_state)):
    
#     


#     # okay now let's check if we can print those states into files.
#     # env.save_state_file(successor_state, os.path.join(dumpstates_images, f"state_{idx}_action_{action}.png"))
    
#     print(f"Action: {action}, Successor State: {successor_state.mario_position}")

pass


# from planiverse.problems.real_world_problems.urban_planning.environment import UrbanPlanningEnv
# env = UrbanPlanningEnv(100)
# env.fix_index(0)
# init_state, _ = env.reset()
# successors = env.successors(init_state)
# for action, state in successors:
#     print(f"Action: {action}, State: {state}")
#     pass
