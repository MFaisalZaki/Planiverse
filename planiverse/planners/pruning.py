"""Dominated action pruning: stop sampling actions that never do anything different.

Jinnai and Fukunaga, *Learning to Prune Dominated Action Sequences in Online Black-Box
Planning*, AAAI 2017, and the 2022 extended version. Atari's action set has 18 actions of
which, in most games, many are duplicates of others (fire without a gun, diagonals that
resolve to a cardinal); an action sequence is *dominated* when another sequence always
reaches the same state, and the paper learns which are, from the search's own expansions,
and stops generating them (DASP; DASA extends it to context-dependent sequences).

Under this library's `successors` contract every child of a state is generated at once,
and duplicate children are already caught on `literals`, so pruning single actions saves
nothing in a tree search. It pays for the **sampling** planners, which draw actions from a
vocabulary and would otherwise spend draws on actions that duplicate others. That is where
this plugs in: give a `DominatedActionPruner` to `SuccessorCache`, and the cache observes
every expansion it performs and drops from each state's children any action that, after
`threshold` observations, led to the same state as a lower-ranked action at least a
`(1 - tolerance)` fraction of the time. The statistics come from the run being pruned;
nothing is learned before it starts.
"""


class DominatedActionPruner:
    """Online detection of actions equivalent to another action."""

    def __init__(self, threshold=20, tolerance=0.0):
        if threshold < 1 or not 0.0 <= tolerance < 1.0:
            raise ValueError("threshold must be >= 1 and tolerance in [0, 1)")
        self.threshold = threshold
        self.tolerance = tolerance
        self.counts = {}            # (a, b) -> [times both applicable, times same child]
        self.pruned = 0

    def observe(self, children):
        """Record, for every pair of applicable actions, whether they agreed."""
        items = sorted(((str(a), a, s.literals) for a, s in children), key=lambda t: t[0])
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                key = (items[i][0], items[j][0])
                seen, same = self.counts.get(key, (0, 0))
                self.counts[key] = (seen + 1, same + (items[i][2] == items[j][2]))

    def dominated(self, action, others):
        """Is `action` a duplicate of some earlier-ranked action in `others`?"""
        name = str(action)
        for other in others:
            key = (str(other), name)
            if key[0] >= name:
                continue
            seen, same = self.counts.get(key, (0, 0))
            if seen >= self.threshold and same / seen >= 1.0 - self.tolerance:
                return True
        return False

    def filter(self, children):
        """`children` without dominated actions; observes them first."""
        self.observe(children)
        actions = [a for a, _ in children]
        kept = [(a, s) for a, s in children if not self.dominated(a, actions)]
        self.pruned += len(children) - len(kept)
        return kept
