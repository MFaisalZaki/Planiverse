from planiverse.simulator.wrappers.pddlgym.structs import Literal, LiteralConjunction
from planiverse.simulator.wrappers.pddlgym.prolog_interface import PrologInterface
from planiverse.simulator.wrappers.pddlgym.core import get_successor_states

from planiverse.simulator.wrappers.base import SimulatorBase

class PDDLGymEnv(SimulatorBase):
    def __init__(self, name, envobj):
        super().__init__(name, envobj)
        # PDDLGym draws a *random* problem out of the domain's problem files on every
        # reset unless the index is pinned. That makes reset() non-repeatable and lets
        # a plan be validated against a different problem than it was found for, so pin
        # the first problem here; fix_index selects a different one.
        self.fix_index(0)
        self.init_state, self.init_info = self.reset()
        self.goal_state = self.init_state.goal

    def fix_index(self, index):
        """!
        Selects which of the domain's problem files to load, like every other Planiverse
        environment's fix_index. Call it before reset().
        """
        problems = self.envobj.env.problems
        assert 0 <= index < len(problems), f"Index {index} not found, this domain has {len(problems)} problems."
        self.envobj.env.fix_problem_index(index)
    
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
        # Test the goal the state itself carries, not the one captured at construction.
        # PDDLGym cycles through a domain's problem files, so every reset() can load a
        # different problem with a different goal -- and often different objects. A cached
        # goal then refers to another problem entirely and can never be satisfied.
        return self.__check_goal__(state, state.goal)

    def simulate(self, plan):
        state, info = self.reset()
        ret_states = [state]
        for action in plan:
            state = self.__apply_action__(state, action)
            ret_states.append(state)
        return ret_states
    
    def validate(self, plan):
        return self.is_goal(self.simulate(plan)[-1])
