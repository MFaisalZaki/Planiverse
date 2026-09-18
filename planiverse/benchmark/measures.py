"""Progress measures, one per environment, lower is better.

SIW and BFWS take a `progress(state)` callback in place of the unachieved-goal count that
classical width-based planners lean on. Against a simulator that count does not exist, so the
measure is written here, per environment: it is a property of how an environment is searched
rather than of the environment, and it is a search guide, not an admissible heuristic.
"""


def _count(state, prefix):
    return sum(1 for literal in state.literals if str(literal).startswith(prefix))


def puzznic(state):
    """Blocks left; clearing every block is the goal."""
    return _count(state, "at(box")


def flipull(state):
    """Blocks left above the stage's clear target."""
    return max(0, state.blocks_remaining - state.clear_target)


def lolo(state):
    """Hearts still to collect, plus one until Lolo is on the door.

    Dying is not "far from the goal" but "not going to arrive", so it is pinned above any live
    distance rather than left to compare as a number.
    """
    if getattr(state, "dead", False) or getattr(state, "died", False):
        return 99
    return state.hearts_left + (0 if getattr(state, "solved", False) else 1)


def super_mario_land(state):
    """Columns between Mario and the flag; a dead state scores worst."""
    if getattr(state, "dead", False):
        return len(state.tiles[0]) + 1
    return max(0, state.goal[0] - state.tile_x)


def water_network(state):
    """Junctions still contaminated."""
    return state.contaminated


def power_grid(state):
    """Steps still to survive; a blackout is worse than any of them."""
    horizon = getattr(state, "horizon", None) or 10
    return horizon + 1 if state.blackout else max(0, horizon - state.step)


def crop_management(state):
    """Irrigation decisions still to make: the yield is only known once the season ends."""
    return max(0, 10 - state.depth)


def amazing_tater(state):
    """Taters still out, plus the active one's straight-line distance to the flag.

    The count alone is 1 until the last tater steps onto the flag; the distance breaks that
    plateau up without pretending to be admissible, since turnstiles and pits make the real
    route much longer.
    """
    if not state.taters:
        return 0
    where = dict(state.taters)[state.active]
    flag = state.level.exit
    return 2 * len(state.taters) + abs(where[0] - flag[0]) + abs(where[1] - flag[1])


def network_attack(state):
    """Sensitive hosts not yet rooted."""
    return -_count(state, "compromised_host")


def emulated(state):
    """What the wrapper or the goal says is left: the state carries its own measure."""
    return state.progress


#: environment name -> `progress(state)`, lower is better.
MEASURES = {
    "puzznic": puzznic,
    "flipull": flipull,
    "lolo": lolo,
    "amazing_tater": amazing_tater,
    "super_mario_land": super_mario_land,
    "water_network": water_network,
    "power_grid": power_grid,
    "crop_management": crop_management,
    "network_attack": network_attack,
    "game_boy": emulated,
    "retro": emulated,
}
