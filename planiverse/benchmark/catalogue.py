"""The planners a benchmark can run, as data.

Each entry says how to build one from a JSON parameter block, so `planners/bfws-2.json` can
name a planner and set its knobs without the harness importing anything until it has to. The
signatures are documented here because a planner config is the one place where a typo is
silent: an unknown parameter would otherwise be accepted and ignored.
"""
import importlib

#: `name -> (import path, class, accepted parameters)`. `progress` and `heuristic` are not
#: listed: they are callables and come from `measures`, not from JSON.
PLANNERS = {
    "iw": ("planiverse.planners.width", "IWSearch",
           ("width", "strict", "novelty_rule")),
    "iterated_width": ("planiverse.planners.width", "IteratedWidth",
                       ("max_width", "strict", "novelty_rule")),
    "siw": ("planiverse.planners.width", "SIWSearch",
            ("width", "max_width", "max_rounds", "strict", "avoid_dead_ends")),
    # `bfws` deliberately does not expose `prune`: the pruned variant is incomplete, and a
    # config could not be told apart from the complete search the COMPLETE tuple below
    # vouches for. k-BFWS is benchmarked as the rounds of `iterated_bfws` instead.
    "bfws": ("planiverse.planners.width", "BFWSSearch",
             ("width", "strict")),
    "iterated_bfws": ("planiverse.planners.width", "IteratedBFWS",
                      ("max_width", "strict", "final_complete")),
    "fsx": ("planiverse.planners.fsx", "FSXPlanner",
            ("horizon", "walkers", "measure", "max_steps", "seed", "temperature")),
    "mcts": ("planiverse.planners.mcts", "MCTSPlanner",
             ("iterations", "exploration", "rollout_depth", "backup", "seed",
              "length_penalty")),
}

#: Planners that take a `progress` callback. The rest ignore one, and passing it would be a
#: `TypeError` rather than a no-op, so the harness has to know which is which.
TAKES_PROGRESS = ("siw", "bfws", "iterated_bfws")

#: Planners whose behaviour depends on a random seed. The harness records the seed in every
#: result so an unseeded run cannot be mistaken for a reproducible one.
RANDOMISED = ("fsx", "mcts")

#: Planners that are complete: if they finish without a plan, there is no plan.
#:
#: Only BFWS, and only because it uses novelty as a sort key rather than as a filter, so
#: nothing is ever discarded. The others all stop short of the reachable space:
#:
#: - `iw` prunes by novelty at a fixed width, so IW(k) misses any problem of width > k.
#: - `iterated_width` is complete only for problems of width <= `max_width`.
#: - `siw` commits irrevocably to the first improvement each leg finds. Iterating a leg's
#:   width fixes which states a leg can *see*, not that it commits to the first it likes.
#: - `fsx` and `mcts` are sampling planners and stop when their own step or iteration count
#:   runs out, which is not the same as having looked everywhere.
#:
#: This is what stops `UNSOLVED` from being read as "unsolvable": for everything outside this
#: tuple it means "this planner did not find one", and the reports say so.
COMPLETE = ("bfws",)

#: Planners that prove unsolvability only when they say so. `IteratedWidth` reports
#: `exhausted` when one of its widths covered the whole reachable space without discarding
#: anything for novelty; at that point no larger width can reach further, and there is no
#: plan. `IteratedBFWS` reserves the word the same way: a pruned round that covered the
#: space, or the unpruned final round emptying its frontier, and nothing else. Any other way
#: either stops (the budget, or reaching `max_width` with the filter still biting) proves
#: nothing, so completeness here is a property of the individual run rather than of the
#: planner. `iterated_bfws` sits here rather than in COMPLETE because `final_complete` is a
#: config knob: with it off, the planner is only as complete as its pruned rounds, which is
#: to say not at all.
COMPLETE_WHEN_EXHAUSTED = ("iterated_width", "iterated_bfws")


def names():
    return tuple(sorted(PLANNERS))


def describe(name):
    """`(import path, class name, accepted parameters)` for a planner."""
    if name not in PLANNERS:
        raise KeyError(
            f"unknown planner {name!r}. Available: {', '.join(names())}")
    return PLANNERS[name]


def build(name, params=None, progress=None):
    """Construct a planner.

    Unknown parameters raise rather than being dropped: a benchmark that silently ignores
    `"widht": 2` reports a width-1 result under a width-2 name, and nothing downstream can
    tell.
    """
    module_name, class_name, accepted = describe(name)
    params = dict(params or {})

    unknown = sorted(set(params) - set(accepted))
    if unknown:
        raise ValueError(
            f"planner {name!r} does not take {', '.join(unknown)}. "
            f"It takes: {', '.join(accepted) or '(nothing)'}")

    if progress is not None and name in TAKES_PROGRESS:
        params["progress"] = progress
    return getattr(importlib.import_module(module_name), class_name)(**params)


def takes_progress(name):
    return name in TAKES_PROGRESS


def is_randomised(name):
    return name in RANDOMISED


def is_complete(name, search_status=None):
    """Does a no-plan answer prove there is no plan?

    `search_status` is the planner's own status for one run. Without it this answers for the
    planner in general; with it, a conditionally complete planner can be judged on what it
    actually did.
    """
    if name in COMPLETE:
        return True
    return name in COMPLETE_WHEN_EXHAUSTED and search_status == "exhausted"
