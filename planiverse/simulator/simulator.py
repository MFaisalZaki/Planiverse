from planiverse.environments.base import Environment, implements_contract

# The idea of the simulator is to give it an env and it provides a single interface for the
# planner to use.
class Simulator:
    def __init__(self, envobj):
        self.simulator = None
        if isinstance(envobj, Environment) or implements_contract(envobj):
            # Structural, not nominal. This used to be two isinstance checks against two
            # base classes whose only difference was which directory the environment lived
            # in, and which therefore did exactly the same thing. What matters is whether
            # the object answers the six methods a planner calls.
            self.simulator = envobj

        assert self.simulator is not None, f"Unsupported environment type: {type(envobj)}"
    
    def reset(self):
        return self.simulator.reset()
    
    def step(self, action):
        return self.simulator.step(action)
    
    def successors(self, state):
        return self.simulator.successors(state)
    
    def is_goal(self, state):
        return self.simulator.is_goal(state)

    def is_terminal(self, state):
        return self.simulator.is_terminal(state)
    
    def simulate(self, plan):
        return self.simulator.simulate(plan)
    
    def validate(self, plan):
        self.reset()
        return self.simulator.validate(plan)
    
    def get_actions(self):
        return self.simulator.get_actions()
