import os
import json
import numpy as np
from enum import Enum
from copy import deepcopy

from planiverse.problems.real_world_problems.base import RealWorldProblem

# This environment is based on https://github.com/torayeff/mfgrl/tree/main

class ActionType(Enum):
    BUY_CFG               = 0
    CONTINUE_PRODUCTION   = 1
    BATCH_PRODUCTION      = 2
    FINISH_PRODUCTION_ALL = 3
    FINISH_PRODUCTION_CFG = 4


class ConfigurationAction:
    def __init__(self, cfg_id, action_type:ActionType, batch_size:int=-1):
        self.cfg = cfg_id
        self.action = action_type.value
        self.batch_size = batch_size

    def __str__(self):
        match self.action:
            case ActionType.BUY_CFG.value:
                return f'buy_cfg_{self.cfg}'
            case ActionType.CONTINUE_PRODUCTION.value:
                return f'continue_production_cfg_{self.cfg}'
            case ActionType.FINISH_PRODUCTION_ALL.value:
                return 'finish_production_all'
            case ActionType.FINISH_PRODUCTION_CFG.value:
                return f'finish_production_cfg_{self.cfg}'
            case ActionType.BATCH_PRODUCTION.value:
                return f'batch_production_cfg_{self.cfg}_size_{self.batch_size}'
            case _:
                raise ValueError(f'Unknown action type: {self.action}')
    
    def __repr__(self):
        return self.__str__()


class MfgState:
    def __init__(self, state, static_state, DEMAND):
        self.DEMAND        = DEMAND
        self.NUM_CFGS      = state["num_cfgs"]
        self.BUFFER_SIZE   = state["buffer_size"]
        self._state        = state.copy()
        self._static_state = static_state.copy()
        # self.buffer_idx    = np.where(state["incurred_costs"] == 0)[0][0] #len(state["incurred_costs"]) - 1
        self.literals      = frozenset([])
        self.__update__()

    def __update__(self):
        # frozenset([' ^ ' .join([f'{k}({str(v)})' for k, v in self._state.items()])])
        state_vars = []
        for cfg, values in self._state["configuration_costs"].items():
            state_vars.extend([f'{k}(cfg{cfg} {v})'.replace('.','').lower() for k, v in values.items()])
        state_vars.append(f'demand({self._state["demand"]})'.replace('.','').lower())
        state_vars.append(f'demand_time({self._state["demand_time"]})'.replace('.','').lower())

        self.literals = frozenset(state_vars)

    def __eq__(self, state):
        return isinstance(state, MfgState) and self.literals == state.literals

    def buy_cfg(self, cfg_id: int):
        """Buys new configuration.
        This does not update the environment's time.

        Args:
            cfg_id (int): The index of new configuration.

        Returns:
            an updated state.
        """
        ret_state = deepcopy(self._state)
        ret_static_state = deepcopy(self._static_state)

        # calculate reward
        reward = -1.0 * ret_state['configuration_costs'][cfg_id]["market_incurring_costs"]

        # buy new configuration
        # update inucrred costs
        ret_state["configuration_costs"][cfg_id]["incurred_costs"] = ret_state['configuration_costs'][cfg_id]["market_incurring_costs"]

        # update recurring costs
        ret_state["configuration_costs"][cfg_id]["recurring_costs"] = ret_state['configuration_costs'][cfg_id]["market_recurring_costs"]
        ret_static_state["recurring_costs"] = ret_state['configuration_costs'][cfg_id]["market_recurring_costs"]

        # update production rates
        ret_state["configuration_costs"][cfg_id]["production_rates"] = ret_state['configuration_costs'][cfg_id]["market_production_rates"]
        ret_static_state["production_rates"] = ret_state['configuration_costs'][cfg_id]["market_production_rates"]

        # update setup times
        ret_state["configuration_costs"][cfg_id]["setup_times"] = ret_state['configuration_costs'][cfg_id]["market_setup_times"]

        # update cfgs status
        ret_state["configuration_costs"][cfg_id]["cfgs_status"] = (1 / ret_state['configuration_costs'][cfg_id]["market_setup_times"])

        # update production
        ret_state["configuration_costs"][cfg_id]["produced_counts"] = 0

        ret_state["configuration_costs"][cfg_id]["bought"] = True

        return MfgState(ret_state, ret_static_state, self.DEMAND)
    
    def continue_production(self, cfg_id):
        """Continues production.
        This updates the environment's time.

        Returns:
            float: Reward as the sum of negative recurring costs.
        """
        ret_state = deepcopy(self._state)
        ret_static_state = deepcopy(self._static_state)

        # .astype(int) ensures that only ready machines contribute
        reward = -1.0 * np.sum(ret_state["configuration_costs"][cfg_id]["cfgs_status"].astype(int)* ret_state["configuration_costs"][cfg_id]["recurring_costs"])

        # produce products with ready configurations
        ret_state["configuration_costs"][cfg_id]["produced_counts"] += (ret_state["configuration_costs"][cfg_id]["cfgs_status"].astype(int) * ret_state["configuration_costs"][cfg_id]["production_rates"])

        # update cfgs status
        # update only ready or being prepared cfgs
        updates = np.ceil(ret_state["configuration_costs"][cfg_id]["cfgs_status"])
        # add small eps to deal with 0.999999xxx
        progress = 1 / ret_state["configuration_costs"][cfg_id]["setup_times"] + 1e-9 if ret_state["configuration_costs"][cfg_id]["setup_times"] != 0 else 0
        ret_state["configuration_costs"][cfg_id]["cfgs_status"] = np.clip(ret_state["configuration_costs"][cfg_id]["cfgs_status"] + updates * progress, a_min=0, a_max=1)

        # update observation
        ret_state["demand"] = self.DEMAND - np.sum(ret_state["configuration_costs"][cfg_id]["produced_counts"].astype(int))
        ret_state["demand_time"] -= 1

        return MfgState(ret_state, ret_static_state, self.DEMAND)
    
    def batch_production(self, cfg_id, items):
        next_state = self.copy()
        for i in range(items):
            next_state = next_state.continue_production(cfg_id)
            if next_state.is_goal() or next_state.is_terminal(): break
        return next_state

    def finish_production_all(self):
        ret_state = deepcopy(self._state)
        ret_static_state = deepcopy(self._static_state)

        for cfg_id in self._state["configuration_costs"].keys():
            # .astype(int) ensures that only ready machines contribute
            reward = -1.0 * np.sum(ret_state["configuration_costs"][cfg_id]["cfgs_status"].astype(int)* ret_state["configuration_costs"][cfg_id]["recurring_costs"])

            # produce products with ready configurations
            ret_state["configuration_costs"][cfg_id]["produced_counts"] += (ret_state["configuration_costs"][cfg_id]["cfgs_status"].astype(int) * ret_state["configuration_costs"][cfg_id]["production_rates"])

            # update cfgs status
            # update only ready or being prepared cfgs
            updates = np.ceil(ret_state["configuration_costs"][cfg_id]["cfgs_status"])
            # add small eps to deal with 0.999999xxx
            progress = 1 / ret_state["configuration_costs"][cfg_id]["setup_times"] + 1e-9 if ret_state["configuration_costs"][cfg_id]["setup_times"] != 0 else 0
            ret_state["configuration_costs"][cfg_id]["cfgs_status"] = np.clip(ret_state["configuration_costs"][cfg_id]["cfgs_status"] + updates * progress, a_min=0, a_max=1)

            # update observation
            ret_state["demand"] = self.DEMAND - np.sum(ret_state["configuration_costs"][cfg_id]["produced_counts"].astype(int))
            ret_state["demand_time"] -= 1

        return MfgState(ret_state, ret_static_state, self.DEMAND)

    # Let's consider the goal state as the state where the demand_time is out
    def is_goal(self) -> bool:
        return self._state["demand_time"] <= 0

    def is_terminal(self) -> bool:
        return False
    
    def copy(self):
        """Returns a copy of the state."""
        return MfgState(self._state.copy(), self._static_state.copy(), self.DEMAND)

    def apply_action(self, action:ConfigurationAction):
        next_state = self.copy()
        
        match action.action:
            case ActionType.BUY_CFG.value:
                next_state = next_state.buy_cfg(action.cfg)
            case ActionType.CONTINUE_PRODUCTION.value:
                next_state = next_state.continue_production(action.cfg)
            case ActionType.FINISH_PRODUCTION_CFG.value:
                # continue production for a specific configuration
                while not (next_state.is_goal() or next_state.is_terminal()):
                    next_state = next_state.continue_production(action.cfg)
            case ActionType.BATCH_PRODUCTION.value:
                # continue production for a specific configuration in batches
                next_state = next_state.batch_production(action.cfg, action.batch_size)
            case ActionType.FINISH_PRODUCTION_ALL.value:
                # continue production for all configurations
                if not (next_state.is_goal() or next_state.is_terminal()) and\
                list(filter(lambda b: not b[1]['bought'], self._state["configuration_costs"].items())).count(False) == 0:
                    while True:
                        next_state = next_state.finish_production_all()
                        if next_state.is_goal(): break
                        if next_state.is_terminal(): break
            case _:
                raise ValueError(f'Unknown action type: {action.action}')
        return next_state

class MfgEnv(RealWorldProblem):

    def __init__(self):
        # load the data files.
        self._load_setup_datafiles()
        
    def reset(self):
        _env_state = {
            "num_cfgs" : self.NUM_CFGS,
            "buffer_size": self.BUFFER_SIZE,
            "demand": self.DEMAND,
            "demand_time": self.DEMAND_TIME,
            # "incurred_costs": np.zeros(self.BUFFER_SIZE, dtype=np.float32),
            # "recurring_costs": np.zeros(self.BUFFER_SIZE, dtype=np.float32),
            # "production_rates": np.zeros(self.BUFFER_SIZE, dtype=np.float32),
            # "setup_times": np.zeros(self.BUFFER_SIZE, dtype=np.float32),
            # "cfgs_status": np.zeros(self.BUFFER_SIZE, dtype=np.float32),
            # "produced_counts": np.zeros(self.BUFFER_SIZE, dtype=np.float32),
            
            "configuration_costs" : {
                i: {
                    'id': i,
                    "bought": False,
                    "incurred_costs": 0,
                    "recurring_costs": 0,
                    "production_rates": 0,
                    "setup_times": 0,
                    "cfgs_status": 0,
                    "produced_counts": 0,
                    "market_incurring_costs": self.INCUR_COSTS[i],
                    "market_recurring_costs": self.RECUR_COSTS[i],
                    "market_production_rates": self.PRODN_RATES[i],
                    "market_setup_times": self.SETUP_TIMES[i],
                } for i in range(self.NUM_CFGS)
            },
            
            # "market_incurring_costs": self.INCUR_COSTS,
            # "market_recurring_costs": self.RECUR_COSTS,
            # "market_production_rates": self.PRODN_RATES,
            # "market_setup_times": self.SETUP_TIMES,
        }

        # static state are used for stochastic operations
        _static_state = {
            "recurring_costs": np.zeros(self.BUFFER_SIZE, dtype=np.float32),
            "production_rates": np.zeros(self.BUFFER_SIZE, dtype=np.float32),
        }

        return MfgState(_env_state, _static_state, self.DEMAND), {}
    
    def is_goal(self, state: MfgState) -> bool:
        return state.is_goal()
    
    def is_terminal(self, state: MfgState) -> bool:
        return state.is_terminal()
    
    def successors(self, state):
        successors = []
        
        # so a successor state can be the following:
        # 1. purchasing a new configuration
        # 2. continuing production for existing configurations
        # 3. or continuing production for all configurations <- don't like this one
        for cfg_id, values in filter(lambda b: not b[1]['bought'], state._state["configuration_costs"].items()):
            buy_cfg_action =  ConfigurationAction(cfg_id, ActionType.BUY_CFG)
            successors.append((buy_cfg_action, state.apply_action(buy_cfg_action)))
        
        for cfg_id, values in filter(lambda b: b[1]['bought'], state._state["configuration_costs"].items()):
            # continue_production_action = ConfigurationAction(cfg_id, ActionType.CONTINUE_PRODUCTION)
            # successors.append((continue_production_action, state.apply_action(continue_production_action)))
            for batch_size in [10, 20, 50, 100]:
                batch_production_action = ConfigurationAction(cfg_id, ActionType.BATCH_PRODUCTION, batch_size)
                # append the action to continue production for a specific configuration
                successors.append((batch_production_action, state.batch_production(cfg_id, batch_size)))


            # finish_production_action = ConfigurationAction(cfg_id, ActionType.FINISH_PRODUCTION_CFG)
            # append the action to continue production for a specific configuration
            # successors.append((finish_production_action, state.apply_action(finish_production_action)))
        
        # append the action to continue production for all configurations if we have bought all configurations
        if len(list(filter(lambda b: not b[1]['bought'], state._state["configuration_costs"].items()))) == 0:
            FINISH_PRODUCTION_ALL_action = ConfigurationAction(-1, ActionType.FINISH_PRODUCTION_ALL)
            # successors.append((FINISH_PRODUCTION_ALL_action, state.apply_action(FINISH_PRODUCTION_ALL_action)))
        return successors
        
    def fix_index(self, index):
        assert index in self.data_index, f"Index {index} not found in data index."
        self._setup_data(self.data_index[index])
    
    def simulate(self, plan):
        state, _ = self.reset()
        ret_states_trace = [state]
        for action in plan:
            ret_states_trace.append(ret_states_trace[-1].apply_action(action))
        return ret_states_trace

    def _check_problem_feasibility(self) -> bool:
        """Checks whether the problem is feasible.
        The problem is not feasible if the solution does not exist
        using the manufacturing configuration with highes capacity.

        Returns:
            bool: Feasibility of the problem.
        """
        idx = np.argmax(self.PRODN_RATES)
        return (self.DEMAND_TIME - self.SETUP_TIMES[idx]) * self.PRODN_RATES[idx] * self.BUFFER_SIZE > self.DEMAND

    def _setup_data(self, DATA_FILE):
        """Sets up the data."""
        with open(DATA_FILE, "r") as f:
            data = json.load(f)

        self.BUFFER_SIZE = data["buffer_size"]
        self.NUM_CFGS = len(data["configurations"])
        self.DEMAND = data["demand"]
        self.DEMAND_TIME = data["demand_time"]
        self.INCUR_COSTS = np.array([], dtype=np.float32)
        self.RECUR_COSTS = np.array([], dtype=np.float32)
        self.PRODN_RATES = np.array([], dtype=np.float32)
        self.SETUP_TIMES = np.array([], dtype=np.float32)
        for v in data["configurations"].values():
            self.INCUR_COSTS = np.append(self.INCUR_COSTS, v["incurring_cost"])
            self.RECUR_COSTS = np.append(self.RECUR_COSTS, v["recurring_cost"])
            self.PRODN_RATES = np.append(self.PRODN_RATES, v["production_rate"])
            self.SETUP_TIMES = np.append(self.SETUP_TIMES, v["setup_time"])

        assert (
            self._check_problem_feasibility()
        ), "Infeasible. Demand will not be satisfied even in the best case."

        # calculate the penalty K
        # K = max. possible incur. cost + max. possible recur. cost
        # max. possible incur.cost = purchasing the most expensive and filling buffer
        # max. possible recur. cost = running the most recur. cost equipment
        self.PENALTY_K = (
            self.INCUR_COSTS.max() + self.DEMAND_TIME * self.RECUR_COSTS.max()
        ) * self.BUFFER_SIZE

    def _load_setup_datafiles(self):
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.data_index = {i:os.path.join(data_dir, f) for (i,f) in enumerate(map(lambda x: os.path.join(data_dir, x), os.listdir(data_dir)))}
