
class SimulatorBase:
    """!
    The main role of this base class is to provide a common interface for the simulator class to use.
    """
    def __init__(self, name, envobj):
        self.name = name
        self.envobj = envobj

    def reset(self):
        """!
        Resets the environment to the initial state. It returns the initial state and info dictionary.
        """
        raise NotImplementedError

    def step(self, action):
        """!
        Applies the action to the environment and returns the next state, reward, done, and info.
        """
        raise NotImplementedError

    def successors(self, state):
        """!
        Returns a list of possible successors and their actions of the given state.
        """
        raise NotImplementedError

    def is_terminal(self, state):
        """!
        Returns True if the given state is terminal, otherwise False.
        """
        raise NotImplementedError

    def simulate(self, plan):
        """!
        Simulates the given plan and returns the sequence of states info.
        """
        raise NotImplementedError
    
    def validate(self, plan):
        """!
        Validates the given plan and returns True if it is valid, otherwise False.
        """
        raise NotImplementedError

    