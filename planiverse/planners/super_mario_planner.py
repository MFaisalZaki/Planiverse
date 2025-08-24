from typing import Any, Dict, List, Tuple
from planiverse.planners.tree_search_planners import TreeSearchPlanner, Heuristic, CostFunction


class SuperMarioPlanner(TreeSearchPlanner):
    # this is a reimplementation of Robin Baumgarten’s original agent (RBA0)
    def __init__(self, env) -> None:
        # override the is_goal method in the env since this is an iterative replanning.
        super().__init__()
        
        

    # def is_goal(self, state) -> bool:
    #     # So the goal state is moving mario for 175 on x-axis.
    #     return state.mario_position.x >= self.initial_state.mario_position.x+175

    def search(self, env):
        # We need to have some termination condition here for the overall game.
        root, _ = env.reset()
        env.is_goal = lambda state: state.mario_position.x >= root.mario_position.x+175
        costfn = lambda s,a: abs(s[0].timeleft - s[-1].timeleft)
        hfn = lambda s: 1.1*(abs(s.mario_position.x - root.max_mario_x_pos) // 5)
        # Here is is a tricky thing, we need to plan until the window ends,
        # execute the plan and then start a new plan.
        plan = super().search(root, env, hfn, costfn)
        pass





    #     pass