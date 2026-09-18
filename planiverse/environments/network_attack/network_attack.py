
import nasim

from nasim.envs.action import ActionResult, Exploit, PrivilegeEscalation, ServiceScan, OSScan, SubnetScan, ProcessScan, NoOp
from nasim.envs.state  import State
from nasim.envs.utils  import AccessLevel
from nasim.envs.network import Network

from planiverse.environments.base import Environment
from planiverse.environments.generation import rng


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
        # host already compromised so exploits do not fail due to randomness
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


#: Fixes both the generated networks and NASim's own reset. Nine of the eighteen benchmark
#: scenarios are generated rather than loaded, so without this the environment is a different
#: problem each time it is built, and no plan survives a replay.
SCENARIO_SEED = 0


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

#: NASim's benchmark scenarios, in the order `set_index` offers them.
BENCHMARKS = ("tiny", "tiny-hard", "tiny-small", "small", "small-honeypot", "small-linear",
              "medium", "medium-single-site", "medium-multi-site", "tiny-gen", "tiny-gen-rgoal",
              "small-gen", "small-gen-rgoal", "medium-gen", "large-gen", "huge-gen",
              "pocp-1-gen", "pocp-2-gen")


class EnvNASim(Environment):
    """Penetration testing against a NASim network.

    An instance is one of three things, all spelled as a dict: a benchmark scenario
    (`{"scenario": "tiny"}`, what `set_index` selects), a scenario file of your own
    (`{"yaml": path}`), or a network NASim generates from a seed and some sizes
    (`{"hosts": 5, "services": 3, "seed": 7, ...}`, what `generate_instance` draws).
    """

    def __init__(self, scenario_name=None, scenario_yaml=None):
        super().__init__("nasim")
        self.env           = None
        self.actionslist   = None
        self.scenario_index = None
        #: The instance `reset` builds; see the class docstring for the three shapes.
        self.instance = None
        if scenario_yaml is not None:
            self.set_instance({"yaml": scenario_yaml})
        elif scenario_name is not None:
            self.set_instance({"scenario": scenario_name})

    @property
    def scenario_name(self):
        """The benchmark scenario selected, or None for a file or a generated network."""
        return self.instance.get("scenario") if self.instance else None

    @property
    def scenario_yaml(self):
        return self.instance.get("yaml") if self.instance else None

    def set_index(self, index):
        if not 0 <= index < len(BENCHMARKS):
            raise IndexError(
                f"Invalid index: {index}. There are {len(BENCHMARKS)} scenarios, so the "
                f"index must be 0-{len(BENCHMARKS) - 1}.")
        self.set_instance({"scenario": BENCHMARKS[index]})
        self.scenario_index = index

    def set_instance(self, instance):
        """Select an instance: `{"scenario": name}`, `{"yaml": path}` or
        `{"hosts": n, "services": m, "seed": s, ...}` with any of NASim's generator options."""
        if not any(key in instance for key in ("scenario", "yaml", "hosts")):
            raise ValueError("an instance names a benchmark scenario, a yaml file, or the "
                             "hosts and services of a network to generate")
        if "hosts" in instance and "services" not in instance:
            raise ValueError("a generated network needs `services` as well as `hosts`")
        self.instance = dict(instance)
        self.scenario_index = None

    def generate_instance(self, seed=None, hosts=5, services=3, **options):
        """Draw a fresh network, select it, and return it as a dict.

        NASim builds the network: `hosts` and `services` fix its size, and `options` are
        passed straight to its scenario generator (`num_os`, `num_processes`,
        `num_exploits`, `num_privescs`, `r_sensitive`, `r_user`, `uniform`, `alpha_H`,
        `alpha_V`, `lambda_V`, and the rest; see `nasim.scenarios.generator`). The seed goes
        with it, so the same dict always builds the same network, and NASim's generated
        networks always have their sensitive hosts reachable, so there is nothing to check.
        """
        _, seed = rng(seed)
        instance = {"hosts": int(hosts), "services": int(services), "seed": seed, **options}
        self.set_instance(instance)
        return instance

    def reset(self):
        if self.instance is None:
            raise ValueError("Call set_index(), set_instance() or generate_instance() first.")
        # The seed is what makes a `*-gen` scenario a *problem* rather than a draw: NASim
        # generates those networks on demand, so an unseeded `make_benchmark` hands out a
        # different topology, service layout and OS layout on every call. Search would then
        # run on one network and `simulate` replay the plan against another, which is
        # exactly how a correct plan comes back INVALID. The fixed scenarios load from file
        # and ignore this, and a generated instance carries its own seed.
        if "yaml" in self.instance:
            self.env = nasim.load(self.instance["yaml"])
        elif "scenario" in self.instance:
            self.env = nasim.make_benchmark(self.instance["scenario"], seed=SCENARIO_SEED)
        else:
            options = {key: value for key, value in self.instance.items()
                       if key not in ("hosts", "services")}
            self.env = nasim.generate(self.instance["hosts"], self.instance["services"],
                                      **options)
        _, _ = self.env.reset(seed=SCENARIO_SEED)
        self.actionslist = self.env.action_space.actions
        return NASimState(self.env.current_state, self.env.network), {
            "instance": dict(self.instance), "scenario": self.scenario_index,
            "generated": "hosts" in self.instance}
    
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