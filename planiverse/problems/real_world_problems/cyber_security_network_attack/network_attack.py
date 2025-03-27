
import nasim

from nasim.envs.action import ActionResult, Exploit, PrivilegeEscalation, ServiceScan, OSScan, SubnetScan, ProcessScan, NoOp
from nasim.envs.state  import State
from nasim.envs.utils  import AccessLevel
from nasim.envs.network import Network

from planiverse.problems.real_world_problems.base import RealWorldProblem

def perform_action(self, state, action):
    """Perform the given Action against the network.

    Arguments
    ---------
    state : State
        the current state
    action : Action
        the action to perform

    Returns
    -------
    State
        the state after the action is performed
    ActionObservation
        the result from the action
    """
    tgt_subnet, tgt_id = action.target
    assert 0 < tgt_subnet < len(self.subnets)
    assert tgt_id <= self.subnets[tgt_subnet]

    next_state = state.copy()

    if action.is_noop():
        return next_state, ActionResult(True)

    if not state.host_reachable(action.target) \
        or not state.host_discovered(action.target):
        result = ActionResult(False, 0.0, connection_error=True)
        return next_state, result

    has_req_permission = self.has_required_remote_permission(state, action)
    if action.is_remote() and not has_req_permission:
        result = ActionResult(False, 0.0, permission_error=True)
        return next_state, result

    if action.is_exploit() \
        and not self.traffic_permitted(
                state, action.target, action.service
        ):
        result = ActionResult(False, 0.0, connection_error=True)
        return next_state, result

    host_compromised = state.host_compromised(action.target)
    if action.is_privilege_escalation() and not host_compromised:
        result = ActionResult(False, 0.0, connection_error=True)
        return next_state, result

    if action.is_exploit() and host_compromised:
        # host already compromised so exploits don't fail due to randomness
        pass
    # elif np.random.rand() > action.prob:
    #     return next_state, ActionResult(False, 0.0, undefined_error=True)

    if action.is_subnet_scan():
        return self._perform_subnet_scan(next_state, action)

    if action.is_wiretapping():
        return self._perform_wiretapping(next_state, action)

    #if action.is_privilege_escalation() and t_host.is_running_process(action.process):
    #    self._perform_privilege_escalation(state,  action)
    #    self._perform_privilege_escalation(next_state, action)

    t_host = state.get_host(action.target)

    if action.is_privilege_escalation():
        has_proc = (
                action.process is None
                or t_host.is_running_process(action.process)
        )
        has_os = (
                action.os is None or t_host.is_running_os(action.os)
        )
        if has_os and has_proc and action.req_access <= t_host.access:
            self._perform_privilege_escalation(state, action)
            self._perform_privilege_escalation(next_state, action)

    next_host_state, action_obs = t_host.perform_action(action)
    next_state.update_host(action.target, next_host_state)
    self._update(next_state, action, action_obs)
    return next_state, action_obs


setattr(Exploit, "__hash__", lambda self: hash(str(self)))
setattr(PrivilegeEscalation, "__hash__", lambda self: hash(str(self)))
setattr(ServiceScan, "__hash__", lambda self: hash(str(self)))
setattr(OSScan, "__hash__", lambda self: hash(str(self)))
setattr(SubnetScan, "__hash__", lambda self: hash(str(self)))
setattr(ProcessScan, "__hash__", lambda self: hash(str(self)))
setattr(NoOp, "__hash__", lambda self: hash(str(self)))
setattr(Network, 'perform_action', perform_action)


class NASimState(State):
    def __init__(self, state, network):
        super().__init__(state.tensor, state.host_num_map)
        self.network  = network
        self.literals = frozenset([])
        self.__update__()

    def __update__(self):
        # Convert the np.array into at(x,y,val) literals.
        self.literals = frozenset([f'at({x},{y},{val})' for x, row in enumerate(self.tensor) for y, val in enumerate(row)])
        # Check which hosts are compromised.
        for addr in self.network.sensitive_addresses:
            if self.host_has_access(addr,AccessLevel.ROOT):
                self.literals |= frozenset([f'compromised_host_{self.host_num_map[addr]}'])

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
        return NASimState(self.env.current_state, self.env.network), {}
    
    def is_goal(self, state):
        return self.env.network.all_sensitive_hosts_compromised(state)
    
    def successors(self, state):
        ret = []
        for action in self.actionslist:
            successor_state = self.env.generative_step(state, action)[0]
            if successor_state == state: continue
            ret.append((action, NASimState(successor_state, self.env.network)))
        return ret
    
    def is_terminal(self, state):
        return False # there are stuck states in this environment.
    
    def simulate(self, plan):
        state, _ = self.reset()
        ret_states_trace = [state]
        for action in plan:
            ret_states_trace.append(NASimState(self.env.generative_step(ret_states_trace[-1], action)[0], self.env.network))
        return ret_states_trace