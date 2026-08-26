"""Progress measures, one per environment.

`SIWSearch` and `BFWSSearch` take a `progress(state)` callback standing in for the unachieved
-goal count that classical width-based planners lean on. Against a simulator that count does
not exist — `is_goal` is a black-box predicate — so the measure has to be supplied per
environment, and a benchmark that omits it is not measuring BFWS so much as BFWS with its main
input removed.

So the measures are here, written down, one per environment, **lower is better**. Two things
follow from putting them in the benchmark rather than in the environments:

- They are a property of how you choose to *search* an environment, not of the environment.
  Two people can disagree about the right measure for the water network without either of
  them being wrong about what the water network is.
- They are visible. `planiverse-bench environments` prints which environments have one, so a
  weak result on an environment with no measure is legible as such instead of looking like a
  weak planner.

`DEFAULT_MEASURE` is what an environment without an entry gets, and it is worth being blunt
about what it does: nothing. It returns 0 everywhere, which turns BFWS into breadth-first
search ordered by novelty alone and turns SIW into a single IW call. That is a real result, not
a broken one — but it is a different experiment from the ones with a measure, and the reports
mark it.
"""


def _count(state, prefix):
    return sum(1 for literal in state.literals if str(literal).startswith(prefix))


def puzznic(state):
    """Blocks left. Clearing every block is the goal, so this is exactly the distance."""
    return _count(state, "at(box")


def flipull(state):
    """Blocks left above the stage's clear target."""
    return max(0, state.blocks_remaining - state.clear_target)


def platformer(state):
    """Columns between Mario and the flag. Dead states score worst.

    Being dead is not "far from the goal", it is "not going to arrive", so it is pinned above
    any live distance rather than left to compare as a number.
    """
    if getattr(state, "dead", False):
        return len(state.tiles[0]) + 1
    return max(0, state.goal[0] - state.tile_x)


def water_network(state):
    """Junctions still contaminated. Service loss is the constraint, not the objective."""
    return state.contaminated


def power_grid(state):
    """Steps still to survive. A blackout is worse than any of them."""
    horizon = getattr(state, "horizon", None) or 10
    if state.blackout:
        return horizon + 1
    return max(0, horizon - state.step)


def crop_management(state):
    """Irrigation decisions still to make.

    The season is a fixed-length sequence of choices and the yield is only known at the end,
    so there is nothing better available: progress here is elapsed time, not closeness.
    """
    return max(0, 10 - state.depth)


def puzznic_gb(state):
    return state.blocks_remaining


def flipull_gb(state):
    return state.blocks_remaining


def super_mario_land(state):
    """Distance still to run. `level_progress` counts up, and this counts down."""
    return -state.level_progress


def network_attack(state):
    """Sensitive hosts not yet rooted."""
    return -_count(state, "compromised_host")


#: environment name -> `progress(state) -> number`, lower is better.
MEASURES = {
    "puzznic": puzznic,
    "puzznic_gb": puzznic_gb,
    "flipull": flipull,
    "flipull_gb": flipull_gb,
    "platformer": platformer,
    "super_mario_land": super_mario_land,
    "water_network": water_network,
    "power_grid": power_grid,
    "crop_management": crop_management,
    "network_attack": network_attack,
}

#: Environments with no measure. Named rather than merely absent, so that "we have not written
#: one" is distinguishable from "we forgot this environment exists".
#:
#: - `epidemic` — the goal is a threshold on a compartment trajectory, and any monotone
#:   stand-in we tried preferred doing nothing.
#: - `manufacturing` — the objective is a cost over a whole schedule, not a distance.
#: - `urban_planning` — multi-objective by construction; collapsing it to one number is the
#:   research question, not a benchmark detail.
WITHOUT_MEASURE = ("epidemic", "manufacturing", "urban_planning")


def DEFAULT_MEASURE(state):
    """No information. See the module docstring for what this costs."""
    return 0


def measure_for(environment_name):
    """The measure for an environment, or `DEFAULT_MEASURE`."""
    return MEASURES.get(environment_name, DEFAULT_MEASURE)


def has_measure(environment_name):
    return environment_name in MEASURES
