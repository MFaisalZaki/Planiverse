
import nasim

class EnvNASim:
    def __init__(self, scenario_name):
        self.env = nasim.make_benchmark(scenario_name)

class NASimState:
    def __init__(self):
        pass
