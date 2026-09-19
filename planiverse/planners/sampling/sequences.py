"""Scoring an action sequence against the simulator, the way every sampling planner does.

A sequence is scored by where it ends, with the goal-distance heuristic every planner in
the library takes and nothing else: the negated `progress` of its last state, `+inf` when it reaches a goal, `-inf` when it walks into a dead end. A
gene that is not applicable where it lands is **skipped**, the way a game's forward model
treats an invalid input as a no-op, which is what makes random sequences over a per-state
action set workable; `actions` in the result lists only the genes that were applied.
`best_along=True` scores the best state on the way instead of the last one, which helps
when a good state is transient.
"""
from dataclasses import dataclass, field


@dataclass
class Evaluation:
    score: float
    trace: list = field(default_factory=list)   #: states, starting with the root
    actions: list = field(default_factory=list)  #: the actions that were applied
    goal: bool = False
    dead: bool = False


def state_value(env, progress, state):
    if env.is_goal(state):
        return float("inf")
    if env.is_terminal(state):
        return float("-inf")
    return -float(progress(state)) if progress else 0.0


def evaluate_sequence(env, cache, root, actions, progress=None, best_along=False):
    """Replay `actions` from `root` through `cache` and score the result."""
    trace, applied = [root], []
    node = root
    best = state_value(env, progress, root)
    for action in actions:
        if cache.exhausted() or env.is_goal(node) or env.is_terminal(node):
            break
        successor = cache.apply(node, action)
        if successor is None:
            continue
        node = successor
        trace.append(node)
        applied.append(action)
        value = state_value(env, progress, node)
        if env.is_goal(node):
            return Evaluation(float("inf"), trace, applied, goal=True)
        best = max(best, value)
        if env.is_terminal(node):
            return Evaluation(float("-inf") if not best_along else best, trace, applied,
                              dead=True)
    return Evaluation(best if best_along else state_value(env, progress, node), trace,
                      applied)


def commit(env, cache, state, applied, rng):
    """The action to execute after a decision: the first gene the best sequence applied,
    else a random child, a live one when there is any."""
    if applied:
        successor = cache.apply(state, applied[0])
        if successor is not None:
            return applied[0], successor
    children = cache.expand(state)
    live = [(a, s) for a, s in children if not env.is_terminal(s) or env.is_goal(s)]
    return rng.choice(live or children)
