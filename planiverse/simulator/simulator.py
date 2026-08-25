from planiverse.problems.retro_games.base import RetroGame
from planiverse.problems.real_world_problems.base import RealWorldProblem

try:
    # gym and pddlgym come with the `pddl` extra, which needs Python <=3.12. Wrapping a
    # native Planiverse environment needs neither, so a missing pddlgym must not stop this
    # module importing — it only rules out the PDDLGym branch.
    from gym.wrappers.order_enforcing import OrderEnforcing
except ImportError:
    OrderEnforcing = None

# The idea of the simulator is to give it an env: ppdlgym, pyboy, ... etc. And it provides a
# single interface for the planner to use.
class Simulator:
    def __init__(self, envobj):
        self.simulator = None
        if OrderEnforcing is not None and isinstance(envobj, OrderEnforcing):
            # Imported here rather than at module scope: it pulls in pddlgym, which this
            # facade only needs when it is actually handed a PDDLGym environment.
            from planiverse.simulator.wrappers.pddlgymenv import PDDLGymEnv
            self.simulator = PDDLGymEnv(f"{envobj.env.domain.domain_name}", envobj)
        elif isinstance(envobj, RetroGame) or isinstance(envobj, RealWorldProblem):
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
