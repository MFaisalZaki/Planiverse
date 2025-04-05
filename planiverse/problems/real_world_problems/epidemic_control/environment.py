import gym
from gym import spaces
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain

from planiverse.problems.real_world_problems.base import RealWorldProblem
from .create_players import CreatePlayers

sns.set_style("whitegrid")

class PandemicState:
    def __init__(self, players_lattice, pandemic_length):
        self.players_lattice = players_lattice
        self.pandemic_length = pandemic_length
        self.literals = frozenset([])
        self.__update__()

    def __update__(self):
        self.literals = frozenset(chain.from_iterable([[f'at({x},{y},{val})' for y,val in enumerate(row)] for x, row in enumerate(self.players_lattice.build_matrix_strategy(self.pandemic_length))]))
        pass


class PandemicEnv(RealWorldProblem):
    """Custom environment for simulating a pandemic.

    This environment follows the gym interface and allows the user to
    control the spread of a pandemic by adjusting the contact rate.

    Parameters:
    m (int): The number of rows in the lattice.
    n (int): The number of columns in the lattice.
    weight_vac (float): The weight for the vaccine strategy.
    weight_inf (float): The weight for the infection strategy.
    weight_recov (float): The weight for the recovery strategy.
    seed_strategy (int): The initial strategy for the seed player.
    cost_vaccine (float): The cost of using the vaccine strategy.
    cost_infection (float): The cost of using the infection strategy.
    cost_recover (float): The cost of using the recovery strategy.
    lockdown_cost (float): The cost of implementing a lockdown.

    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__("pandemic")
        
        # Define action and observation space
        self.scenario          = None
        self.action_space      = None
        self.observation_space = None
        self.players_lattice   = None
        self.iteration         = 1
        self.pandemic_length   = None

        self.infected_num_list, self.vaccinated_num_list, self.recovered_num_list = (
            [],
            [],
            [],
        )
        self.reward_list, self.actions_taken = [], []
        self.avg_infected_epi, self.avg_vaccinated_epi, self.avg_recovered_epi = (
            [],
            [],
            [],
        )
    
    def fix_index(self, index):
        self.scenarios = {
            0: {'m' : 50, 'n' : 50, 'weight_vac' : 0.05, 'weight_inf' : 0.1, 'weight_recov' : 0.5, 'seed_strategy': 4, 'cost_vaccine' : 10, 'cost_infection': 1000, 'cost_recover': 0.1, 'lockdown_cost': 10000, 'transmission_rate': 0.5, 'sensitivity': 3, 'reward_factor': 2}
        }
        assert index in self.scenarios, f"Scenario {index} not found."
        self.scenario = self.scenarios[index]

    def reset(self):
        """Reset the environment and return the initial state.

        Returns:
        np.ndarray: An m x n array representing the initial state of the lattice.
        """
        self.pandemic_length   = 0
        assert self.scenario is not None, "Scenario is not set."
        self.action_space      = spaces.Discrete(2)
        self.actionslist       = list(range(self.action_space.n))
        self.observation_space = spaces.Box(low=0, high=3, shape=(self.scenario['m'], self.scenario['n']), dtype=np.int8)
        
        players_lattice        = CreatePlayers(**self.scenario)
        players_lattice.get_strategy() # we need to have a fixed strategy for the players not random.
        players_lattice.state_zero()

        self.state             = PandemicState(players_lattice, 0)
        self.pandemic_length = 0
        return self.state, {}

    def is_goal(self, state):
        return state.players_lattice.count_num_strategy(2) <= 0
    
    def is_terminal(self, state):
        # I guess a terminal state would be all players are infected or all players are recovered.
        return False # there are stuck states in this environment.

    def successors(self, state):
        ret = []
        for action in self.actionslist:
            successor_state = self.generative_step(state, action)
            if successor_state == state: continue
            ret.append((action, successor_state))
        return ret

    def generative_step(self, state, action):
        contact_rate = self.take_action(action)
        successor_state = copy.copy(state)
        successor_state.pandemic_length += self.iteration
        successor_state.players_lattice.update_lattice(self.iteration, contact_rate, successor_state.pandemic_length)
        return successor_state

    def step(self, action):
        """Execute one time step within the environment.

        Parameters:
        action (int): The action to be taken.
            0: Increase contact rate.
            1: Decrease contact rate.

        Returns:
        tuple: A tuple containing the observation, reward, done flag, and metadata.
        """

        # iteration = 1
        # contact_rate = self.take_action(action)
        pandemic_time = self.pandemic_length
        self.players_lattice.update_lattice(iteration, contact_rate, pandemic_time)
        state = self.players_lattice.build_matrix_strategy(pandemic_time)
        self.pandemic_length += iteration

        # actions
        self.actions_taken.append(action)
        num_infected = self.players_lattice.count_num_strategy(2)

        # print("Infections for step {}: {} ".format(self.pandemic_length, num_infected))
        self.infected_num_list.append(num_infected)
        self.vaccinated_num_list.append(self.players_lattice.count_num_strategy(1))
        self.recovered_num_list.append(self.players_lattice.count_num_strategy(3))

        reward = self.players_lattice.calc_reward(
            contact_rate, self.pandemic_length, self.reward_factor
        )
        self.reward_list.append(reward)

        # if self.pandemic_length >= 100:
        # if self.players_lattice.count_num_strategy(2) <= 0.01*(self.m*self.n):
        if num_infected <= 0:
            # # plot charts showing change in values as the episode runs
            # if self.plot_title is not None:
            #     self.players_lattice.plot_episode_changes(
            #         self.plot_title,
            #         self.pandemic_length,
            #         self.infected_num_list,
            #         self.vaccinated_num_list,
            #         self.recovered_num_list,
            #         self.reward_list,
            #         self.actions_taken
            #     )

            # reset list values
            (
                self.infected_num_list,
                self.vaccinated_num_list,
                self.recovered_num_list,
            ) = ([], [], [])
            self.reward_list, self.actions_taken = [], []

            done = True
        else:
            done = False
        # obs = np.append(state)
        # obs = np.append(state, axis=0)
        obs = state

        return obs, reward, done, {}

    def take_action(self, action) -> float:
        """Adjust the contact rate based on the given action.

        Parameters:
        action (int): The action to be taken.
            0: Increase contact rate.
            1: Decrease contact rate.

        Returns:
        float: The new contact rate.
        """
        return 1.0 if action == 0 else 0.5

    def render(self, mode="human", close=False):
        """
        Render the current state of the lattice.
        """
        self.players_lattice.draw_matrix_strategy(-1)
