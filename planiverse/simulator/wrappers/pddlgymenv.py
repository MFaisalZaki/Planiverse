from pddlgym.structs import Literal, LiteralConjunction
from pddlgym.prolog_interface import PrologInterface
from pddlgym.core import get_successor_states

from planiverse.simulator.wrappers.base import SimulatorBase

class PDDLGymEnv(SimulatorBase):
    def __init__(self, name, envobj):
        super().__init__(name, envobj)
        self.init_state, self.init_info = self.reset()
        self.goal_state = self.init_state.goal
    
    def __apply_action__(self, state, action):
        next_state = get_successor_states(state, action, self.envobj.env.domain)
        if isinstance(next_state, list): return next_state.pop()
        elif isinstance(next_state, frozenset): return list(next_state).pop()
        else: return next_state

    def __check_goal__(self, state, goal):
        if isinstance(goal, Literal):
            if goal.is_negative and goal.positive in state.literals:
                return False
            if not goal.is_negative and goal not in state.literals:
                return False
            return True
        if isinstance(goal, LiteralConjunction):
            return all(self.__check_goal__(state, lit) for lit in goal.literals)
        prolog_interface = PrologInterface(state.literals, goal,
            max_assignment_count=2,
            allow_redundant_variables=True)
        assignments = prolog_interface.run()
        return len(assignments) > 0
    
    def reset(self):
        return self.envobj.reset()
    
    def successors(self, state):
        return [(a, self.__apply_action__(state, a)) for a in sorted(self.envobj.env.action_space.all_ground_literals(state), key=str)]
    
    def is_terminal(self, state):
        return False # The terminal state means that this state has no successors and is not a goal state.

    def is_goal(self, state):
        return self.__check_goal__(state, self.init_state.goal)

    def simulate(self, plan):
        state, info = self.reset()
        ret_states = [state]
        for action in plan:
            state = self.__apply_action__(state, action)
            ret_states.append(state)
        return ret_states
    
    def validate(self, plan):
        return self.is_terminal(self.simulate(plan)[-1])
    
    def goal(self):
        return self.goal_state
