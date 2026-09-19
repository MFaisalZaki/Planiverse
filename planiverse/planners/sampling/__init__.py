"""Planners that sample action sequences, populations or swarms rather than expand a tree.

None of them trains anything, and none is Monte Carlo *tree* search: each optimises a set
of action sequences, or a swarm of walkers, against the simulator and commits.

| Planner | Optimises |
|---|---|
| `RollingHorizonEvolution` | a population of fixed-horizon sequences, evolved |
| `CrossEntropyPlanner`, `RandomShooting` | a distribution over sequences, refitted on elites |
| `RolloutPlanner` | one step, each child valued by a base policy's rollout |
| `NestedMonteCarloSearch` | nested random playouts, the best sequence kept |
| `FractalMonteCarlo` | a swarm of walkers cloning toward reward and diversity |
| `GoExplore` | an archive of cells, each restored and explored from |
| `MAPElitesPlanner` | an archive of sequences, one elite per descriptor cell |
| `KinodynamicTree` | a tree grown by random propagation: EST, KPIECE or SST |
| `PlanLocalSearch` | one sequence, by simulated annealing or iterated local search |

All of them replay sequences through `SuccessorCache`, so a prefix evaluated a hundred
times costs the simulator once.
"""
from planiverse.planners.sampling.cem import CrossEntropyPlanner, RandomShooting
from planiverse.planners.sampling.fractal import FractalMonteCarlo
from planiverse.planners.sampling.goexplore import GoExplore
from planiverse.planners.sampling.kinodynamic import KinodynamicTree
from planiverse.planners.sampling.localsearch import PlanLocalSearch
from planiverse.planners.sampling.nested import NestedMonteCarloSearch
from planiverse.planners.sampling.policy_rollout import RolloutPlanner
from planiverse.planners.sampling.qd import MAPElitesPlanner
from planiverse.planners.sampling.rhea import RollingHorizonEvolution
from planiverse.planners.sampling.sequences import Evaluation, evaluate_sequence

__all__ = [
    "CrossEntropyPlanner", "Evaluation", "FractalMonteCarlo", "GoExplore", "KinodynamicTree",
    "MAPElitesPlanner", "NestedMonteCarloSearch", "PlanLocalSearch", "RandomShooting",
    "RollingHorizonEvolution", "RolloutPlanner", "evaluate_sequence",
]
