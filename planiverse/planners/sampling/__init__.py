"""Planners that sample action sequences, populations or swarms rather than expand a tree.

None of them trains anything, none is Monte Carlo *tree* search, and none takes a reward:
each optimises a set of action sequences against the simulator, scoring a sequence by the
same `progress` heuristic the width planners take (how far from the goal its last state
is), and commits. The two surveyed planners whose definition is an accumulated reward,
Fractal Monte Carlo and the rollout algorithm, are deliberately not here.

| Planner | Optimises |
|---|---|
| `RollingHorizonEvolution` | a population of fixed-horizon sequences, evolved |
| `CrossEntropyPlanner`, `RandomShooting` | a distribution over sequences, refitted on elites |
| `NestedMonteCarloSearch` | nested random playouts, the best sequence kept |
| `GoExplore` | an archive of cells, each restored and explored from |
| `MAPElitesPlanner` | an archive of sequences, one elite per descriptor cell |
| `KinodynamicTree` | a tree grown by random propagation: EST, KPIECE or SST |
| `PlanLocalSearch` | one sequence, by simulated annealing or iterated local search |

All of them replay sequences through `SuccessorCache`, so a prefix evaluated a hundred
times costs the simulator once.
"""
from planiverse.planners.sampling.cem import CrossEntropyPlanner, RandomShooting
from planiverse.planners.sampling.goexplore import GoExplore
from planiverse.planners.sampling.kinodynamic import KinodynamicTree
from planiverse.planners.sampling.localsearch import PlanLocalSearch
from planiverse.planners.sampling.nested import NestedMonteCarloSearch
from planiverse.planners.sampling.qd import MAPElitesPlanner
from planiverse.planners.sampling.rhea import RollingHorizonEvolution
from planiverse.planners.sampling.sequences import Evaluation, evaluate_sequence

__all__ = [
    "CrossEntropyPlanner", "Evaluation", "GoExplore", "KinodynamicTree", "MAPElitesPlanner",
    "NestedMonteCarloSearch", "PlanLocalSearch", "RandomShooting", "RollingHorizonEvolution",
    "evaluate_sequence",
]
