"""Progress measures, one per environment.

`SIWSearch` and `BFWSSearch` take a `progress(state)` callback standing in for the unachieved
-goal count that classical width-based planners lean on. Against a simulator that count does
not exist (`is_goal` is a black-box predicate), so the measure has to be supplied per
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
a broken one, but it is a different experiment from the ones with a measure, and the reports
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


def lolo(state):
    """Heart framers still to collect, plus one while Lolo is not on the door.

    Collecting the last heart opens the door but does not clear the room, so a measure that
    stopped at the heart count would call every position with the door still to reach equally
    finished. Dying is not "far from the goal" but "not going to arrive", so it is pinned above
    any live distance rather than left to compare as a number.
    """
    if getattr(state, "dead", False) or getattr(state, "died", False):
        return 99
    return state.hearts_left + (0 if getattr(state, "solved", False) else 1)


def super_mario_land(state):
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


def lolo_gb(state):
    """The same measure as its Python twin, off the cartridge's own heart counter."""
    return lolo(state)


def amazing_tater(state):
    """Taters still out, plus how far the one under the controls has to walk.

    The tater count alone is a poor guide and in most rooms a useless one: it is 1 until the
    single tater steps onto the flag, and then it is 0. The straight-line distance breaks that
    plateau up without pretending to be admissible: turnstiles and pits mean the real route is
    often much longer, and it is a search guide, not a heuristic with a proof attached.
    """
    if not state.taters:
        return 0
    where = dict(state.taters)[state.active]
    flag = state.level.exit
    return 2 * len(state.taters) + abs(where[0] - flag[0]) + abs(where[1] - flag[1])


def amazing_tater_gb(state):
    """The same measure as its Python twin, off the board the cartridge composed."""
    if not state.taters or state.exit is None:
        return 0
    where = state.taters.get(state.active) or next(iter(state.taters.values()))
    return (2 * len(state.taters)
            + abs(where.row - state.exit.row) + abs(where.col - state.exit.col))


def super_mario_land_gb(state):
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
    "lolo": lolo,
    "lolo_gb": lolo_gb,
    "amazing_tater": amazing_tater,
    "amazing_tater_gb": amazing_tater_gb,
    "super_mario_land": super_mario_land,
    "super_mario_land_gb": super_mario_land_gb,
    "water_network": water_network,
    "power_grid": power_grid,
    "crop_management": crop_management,
    "network_attack": network_attack,
}

#: Environments with no measure. Named rather than merely absent, so that "we have not written
#: one" is distinguishable from "we forgot this environment exists". Empty since the
#: manufacturing environment was withdrawn; the reporting path that flags an unmeasured
#: environment is kept, because the next environment added may well need it.
WITHOUT_MEASURE = ()


def DEFAULT_MEASURE(state):
    """No information. See the module docstring for what this costs."""
    return 0


def measure_for(environment_name):
    """The measure for an environment, or `DEFAULT_MEASURE`."""
    return MEASURES.get(environment_name, DEFAULT_MEASURE)


def has_measure(environment_name):
    return environment_name in MEASURES
