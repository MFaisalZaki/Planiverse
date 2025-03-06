
import pddlgym

from planiverse.simulator.simulator import Simulator

env = pddlgym.make("PDDLEnvDepot-v0")
env.env.fix_problem_index(0)
env.env.seed(0)

test_simulator = Simulator(env)


pass