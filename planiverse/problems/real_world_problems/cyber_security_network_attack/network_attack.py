
import nasim

from nasim.envs.action import Exploit, PrivilegeEscalation, ServiceScan, OSScan, SubnetScan, ProcessScan, NoOp
from nasim.envs.state import State

from planiverse.problems.real_world_problems.base import RealWorldProblem

setattr(Exploit, "__hash__", lambda self: hash(str(self)))
setattr(PrivilegeEscalation, "__hash__", lambda self: hash(str(self)))
setattr(ServiceScan, "__hash__", lambda self: hash(str(self)))
setattr(OSScan, "__hash__", lambda self: hash(str(self)))
setattr(SubnetScan, "__hash__", lambda self: hash(str(self)))
setattr(ProcessScan, "__hash__", lambda self: hash(str(self)))
setattr(NoOp, "__hash__", lambda self: hash(str(self)))

class NASimState(State):
    def __init__(self, state):
        super().__init__(state.tensor, state.host_num_map)
        self.literals = frozenset([f'at({x},{y},{val})' for x, row in enumerate(self.tensor) for y, val in enumerate(row)])             

class EnvNASim(RealWorldProblem):
    def __init__(self, scenario_name=None, scenario_yaml=None):
        super().__init__("nasim")
        self.env           = None
        self.actionslist   = None
        self.scenario_name = scenario_name
        self.scenario_yaml = scenario_yaml
        
    def fix_index(self, index):
        # based on the value pick from the scenario.
        index_scenario_map = {
            0:"tiny",
            1:"tiny-hard",
            2:"tiny-small",
            3:"small",
            4:"small-honeypot",
            5:"small-linear",
            6:"medium",
            7:"medium-single-site",
            8:"medium-multi-site",
            9:"tiny-gen",
            10:"tiny-gen-rgoal",
            11:"small-gen",
            12:"small-gen-rgoal",
            13:"medium-gen",
            14:"large-gen",
            15:"huge-gen",
            16:"pocp-1-gen",
            17:"pocp-2-gen"
        }
        assert index in index_scenario_map, f"Index {index} not found in the index_scenario_map"
        self.scenario_name = index_scenario_map[index]

    def reset(self):
        assert self.scenario_name is not None or self.scenario_yaml is not None, "Scenario name or yaml is not set."
        # Check if we want to load the scenario from the yaml or from a name.
        self.env = nasim.make_benchmark(self.scenario_name)
        _, _ = self.env.reset(seed=0)
        self.actionslist = self.env.action_space.actions
        return NASimState(self.env.current_state), {}
    
    def is_goal(self, state):
        return self.env.network.all_sensitive_hosts_compromised(state)
    
    def successors(self, state):
        ret = []
        for action in self.actionslist:
            successor_state = self.env.generative_step(state, action)[0]
            if successor_state == state: continue
            ret.append((action, NASimState(successor_state)))
        return ret
    
    def is_terminal(self, state):
        return False # there are stuck states in this environment.
    
    def simulate(self, plan):
        state, _ = self.reset()
        ret_states_trace = [state]
        for action in plan:
            ret_states_trace.append(NASimState(self.env.generative_step(ret_states_trace[-1], action)[0]))
        return ret_states_trace