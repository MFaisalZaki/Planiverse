"""What the planners added after the tool paper share.

Every environment offers `successors(state)` and nothing cheaper, so a planner that wants to
apply one action, or replay an action sequence, has to expand the whole state and pick the
child it asked for. `SuccessorCache` makes that cost once per state rather than once per
visit, which is what turns the sampling planners (rolling-horizon evolution, the cross-entropy
method, rollouts, Go-Explore) from hopeless into ordinary against these simulators: an
evolutionary population re-evaluates the same prefix hundreds of times.

The cache is keyed on `literals`, the same identity every closed list in the library uses,
so two states an environment spells the same are the same state here too.
"""
# Imported inside the functions: `planiverse.planners.width` imports this module, and this
# module wants its `result` types, so a module-level import here would be a cycle.


def remaining(budget, spent):
    """A sub-budget with `spent` expansions and the elapsed time already taken off.

    The iterated searches all cut the same slice; kept here so the new ones do not each
    re-derive it.
    """
    from planiverse.planners.width.result import Budget
    return Budget(
        max_expansions=(None if budget.max_expansions is None
                        else max(0, budget.max_expansions - spent)),
        max_seconds=(None if budget.max_seconds is None
                     else max(0.0, budget.max_seconds - budget.elapsed())))


def finish(status, plan, trace, statistics, budget, width=None):
    """A `SearchResult` with the elapsed time filled in, the way every planner ends."""
    from planiverse.planners.width.result import SearchResult
    statistics.elapsed = budget.elapsed()
    return SearchResult(plan=plan, states=trace, status=status,
                        width=width if status == "solved" else None, statistics=statistics)


class SuccessorCache:
    """`env.successors` memoised on `literals`, counting expansions once.

    `expand(state)` is the cached expansion; `apply(state, action)` the successor an action
    leads to, or `None` when the action does nothing here (the environment filters
    self-loops out of `successors`, so an inapplicable action is simply absent).
    """

    def __init__(self, env, statistics=None, budget=None, pruner=None):
        """`pruner`: a `DominatedActionPruner` that observes every expansion and drops
        actions it has found to duplicate others (see `pruning.py`)."""
        from planiverse.planners.width.result import SearchStatistics
        self.env = env
        self.statistics = statistics if statistics is not None else SearchStatistics()
        self.budget = budget
        self.pruner = pruner
        self.table = {}

    def expand(self, state):
        key = state.literals
        children = self.table.get(key)
        if children is None:
            children = self.env.successors(state)
            if self.pruner is not None:
                children = self.pruner.filter(children)
            self.table[key] = children
            self.statistics.expansions += 1
            self.statistics.generated += len(children)
        return children

    def apply(self, state, action):
        for candidate, successor in self.expand(state):
            if candidate == action:
                return successor
        return None

    def replay(self, state, actions, stop_at_goal=True):
        """Apply `actions` in order from `state`. Returns the trace, starting with `state`.

        Stops at the first inapplicable action, at a goal (when `stop_at_goal`), at a dead
        end, or when the budget runs out, so the trace can be shorter than the plan.
        """
        trace = [state]
        for action in actions:
            if self.exhausted():
                break
            node = trace[-1]
            if (stop_at_goal and self.env.is_goal(node)) or self.env.is_terminal(node):
                break
            successor = self.apply(node, action)
            if successor is None:
                break
            trace.append(successor)
        return trace

    def exhausted(self):
        return self.budget is not None and self.budget.exhausted(self.statistics.expansions)

    def __len__(self):
        return len(self.table)


def action_vocabulary(env, state, cache=None):
    """The actions a sampling planner may draw from.

    `get_actions()` when the environment offers one, else the actions applicable in `state`,
    which is the best a per-state action set allows. Planners that learn the vocabulary as
    they go widen it with every expansion they see.
    """
    if env.provides("get_actions") if hasattr(env, "provides") else hasattr(env, "get_actions"):
        try:
            actions = list(env.get_actions())
            if actions:
                return actions
        except Exception:
            pass
    children = cache.expand(state) if cache is not None else env.successors(state)
    return [action for action, _ in children]


def dead_end_progress(progress, env, state, penalty=float("inf")):
    """`progress(state)`, or `penalty` for a dead end: no goal is reachable, so no
    measure of distance to one applies."""
    if env.is_terminal(state) and not env.is_goal(state):
        return penalty
    return progress(state) if progress is not None else 0.0
