"""What the planners share.

Every environment offers `successors(state)` and nothing cheaper, so a planner that wants to
apply one action, or replay an action sequence, has to expand the whole state and pick the
child it asked for. `SuccessorCache` makes that cost once per state rather than once per
visit, which is what makes the planners that come back to the same states affordable against
these simulators: Go-Explore and KPIECE explore from the same cells over and over, and the
macro planner replays every macro from every state it expands.

The cache is keyed on the state itself, which every environment makes hashable and
comparable on its identity. `literals` are not that identity: they are the atoms the novelty
tests count, and an environment may spell two different states the same way (two orders of the
same measures at different costs, say), so a cache keyed on them would hand one state the
other's children.
"""
# Imported inside the functions: `planiverse.planners.width` imports this module, and this
# module wants its `result` types, so a module-level import here would be a cycle.


def finish(status, plan, trace, statistics, budget, width=None):
    """A `SearchResult` with the elapsed time filled in, the way every planner ends."""
    from planiverse.planners.width.result import SearchResult
    statistics.elapsed = budget.elapsed()
    return SearchResult(plan=plan, states=trace, status=status,
                        width=width if status == "solved" else None, statistics=statistics)


class SuccessorCache:
    """`env.successors` memoised on the state, counting expansions once.

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
        key = state
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

