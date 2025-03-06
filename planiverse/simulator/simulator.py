
from gym.wrappers.order_enforcing import OrderEnforcing

from planiverse.simulator.wrappers.pddlgymenv import PDDLGymEnv

# The idea of the simulator is to give it an env: ppdlgym, pyboy, ... etc. And it provides a
# single interface for the planner to use.
class Simulator:
    def __init__(self, envobj):
        self.simulator = None
        if isinstance(envobj, OrderEnforcing):
            self.simulator = PDDLGymEnv(f"{envobj.env.domain.domain_name}", envobj)
        else:
            assert False, f"Unsupported environment type: {type(envobj)}"
    
    def reset(self):
        return self.simulator.reset()
    
    def step(self, action):
        return self.simulator.step(action)
    
    def successors(self, state):
        return self.simulator.successors(state)
    
    def is_terminal(self, state):
        return self.simulator.is_terminal(state)
    
    def simulate(self, plan):
        return self.simulator.simulate(plan)
    
    def validate(self, plan):
        return self.simulator.validate(plan)
    
