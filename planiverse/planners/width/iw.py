"""IW(k) and Iterated Width, against a simulator.

IW(k) is breadth-first search with one addition: a state is discarded unless its novelty is
at most `k`. That single filter is what makes it work: it turns an exponential frontier into
one bounded by the number of atom tuples of size ≤ k, so IW(1) expands at most as many states
as there are atoms.

The price is that IW(k) is **incomplete** for fixed k: a problem needing a state of novelty
3 is unsolvable by IW(2) no matter how long it runs. Iterated Width answers that by running
IW(1), then IW(2), and so on. `BFWS` in the sibling module answers it differently, by using
novelty to *order* rather than to discard.

Adapting this to a simulator rather than a PDDL task changes three things:

* **There is no goal decomposition.** `is_goal` is a black-box predicate, so there is no
  count of achieved subgoals to stop at or sort by. IW needs none; `BFWS` takes a
  `progress` callback in its place.
* **Expansions are expensive** (seconds each in the power grid environment), so every search
  takes a `Budget` and reports what it spent.
* **Dead ends are real.** A PDDL task usually has none; several environments here do, and
  `is_terminal` states are dropped rather than expanded.
"""
from collections import deque

from planiverse.planners.width.novelty import NoveltyTable, path_novelty
from planiverse.planners.width.result import Budget, SearchResult, SearchStatistics


class IWSearch:
    """IW(k) at one fixed width: breadth-first search that keeps only states of novelty ≤ k.

    The component `IW` is built from, not a planner in its own right: the literature's IW
    is the iteration over widths, which is `IW` below. It is not exported from the
    package; `IW(width=k)` runs one width.
    """

    def __init__(self, width=1, strict=True, novelty_rule="standard", atoms=None):
        """`novelty_rule` picks how novelty is measured; `atoms(state)` what it is measured
        over (`state.literals` when `None`).

        `"standard"` is the textbook definition: the smallest tuple of atoms in the state not
        seen anywhere in this search. `"path"` is the rule the pyBehaviourPlanningLTL
        reference uses (how many atoms are new relative to the path taken to the state), and
        is kept for comparability. The two agree at width 1 and part company above it; see
        `novelty.path_novelty`.
        """
        if novelty_rule not in ("standard", "path"):
            raise ValueError(f"novelty_rule must be 'standard' or 'path', got {novelty_rule!r}")
        self.width = width
        self.strict = strict
        self.novelty_rule = novelty_rule
        self.atoms = atoms

    def __atoms__(self, state):
        return state.literals if self.atoms is None else self.atoms(state)

    def solve(self, env, budget=None, state=None):
        """Search from `state`, or from `env.reset()` when it is not given."""
        budget = (budget or Budget()).start()
        statistics = SearchStatistics(widths_tried=(self.width,))
        table = NoveltyTable(self.width, strict=self.strict)

        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return self.__result__("solved", [], [state], statistics, budget)

        # The initial state is always kept, whatever its novelty: pruning it would leave
        # nothing to search.
        table.evaluate_and_record(self.__atoms__(state))
        frontier = deque([(state, [], [state], frozenset(self.__atoms__(state)))])
        closed = {state.literals}

        while frontier:
            if budget.exhausted(statistics.expansions):
                return self.__result__("out_of_budget", None, [], statistics, budget, table)

            node, plan, trace, path_atoms = frontier.popleft()
            statistics.expansions += 1

            for action, successor in env.successors(node):
                statistics.generated += 1

                if successor.literals in closed:
                    statistics.pruned_duplicate += 1
                    continue

                atoms = self.__atoms__(successor)
                if self.novelty_rule == "standard":
                    novel = table.evaluate_and_record(atoms) <= self.width
                else:
                    novel = path_novelty(atoms, path_atoms) >= self.width
                if not novel:
                    statistics.pruned_novelty += 1
                    continue

                closed.add(successor.literals)
                successor_plan = plan + [action]
                successor_trace = trace + [successor]

                # Checked before the terminal test: a goal that is also terminal (which is
                # every absorbing goal state in this library) must still be reported solved.
                if env.is_goal(successor):
                    return self.__result__("solved", successor_plan, successor_trace,
                                           statistics, budget, table)
                if env.is_terminal(successor):
                    statistics.pruned_terminal += 1
                    continue

                frontier.append((successor, successor_plan, successor_trace,
                                 path_atoms | frozenset(atoms)))

        return self.__result__("exhausted", None, [], statistics, budget, table)

    def __result__(self, status, plan, trace, statistics, budget, table=None):
        statistics.elapsed = budget.elapsed()
        if table is not None:
            statistics.novelty_evaluations = table.evaluations
            statistics.tuples_enumerated = table.tuples_enumerated
        return SearchResult(plan=plan, states=trace, status=status,
                            width=self.width if status == "solved" else None,
                            statistics=statistics)


class IW:
    """Iterated Width: IW(1), IW(2), … until one solves it, or until no larger width could help.

    Lipovetzky and Geffner, *Width and Serialization of Classical Planning Problems*, ECAI
    2012, and *Classical Planning with Simulators: Results on the Atari Video Games*, IJCAI
    2015, where IW(1) runs against a black-box simulator. The iteration is the standard answer
    to IW(k)'s incompleteness, and the right way to run IW when you do not already know the
    problem's width; `width=k` runs that one width only.

    ```python
    from planiverse.planners.width import IW

    env.set_index(0)
    result = IW(max_width=2).solve(env)     # IW(1), then IW(2)
    result = IW(width=2).solve(env)         # IW(2) alone
    if result:
        env.validate(result.plan)
    ```
 Note what it costs against a simulator: each width
    restarts from scratch and re-expands everything the previous one did, and here an
    expansion is a hydraulic solve or a power-flow solve rather than a bitmask update. The
    budget is shared across widths rather than granted afresh to each.

    `max_width` is a **bound, not a plan**. Set it as high as you like: the loop almost never
    reaches it, because it stops as soon as one of three things happens.

    1. A width solves the problem.
    2. The budget runs out, the usual outcome above width 2, since IW(k) enumerates every
       k-tuple of every state's atoms.
    3. **A width exhausts the reachable space without pruning anything for novelty.** That is
       the interesting one: if nothing was ever discarded for being unnovel, then IW(k) saw
       the whole reachable space, and no larger width can see more. The problem has no
       solution, and iterating further would re-run the identical search. This is what makes
       `IW` complete when it gets that far, and what makes a bound of 1000 harmless rather
       than a thousand wasted restarts.

    Above `strict`'s practical width the tuple enumeration gets expensive fast, so widths
    beyond 2 need `strict=False`; `NoveltyTable` refuses them otherwise.
    """

    def __init__(self, max_width=2, width=None, strict=True, novelty_rule="standard",
                 atoms=None):
        """`width` pins one width, so IW(k) runs on its own; `max_width` is otherwise the
        bound the iteration stops at."""
        if width is not None and width < 1:
            raise ValueError(f"width must be at least 1, got {width}")
        if max_width < 1:
            raise ValueError(f"max_width must be at least 1, got {max_width}")
        self.width = width
        self.max_width = max_width
        self.strict = strict
        self.novelty_rule = novelty_rule
        self.atoms = atoms

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        totals = SearchStatistics()
        last = None

        widths = ([self.width] if self.width is not None
                  else range(1, self.max_width + 1))
        for width in widths:
            search = IWSearch(width, strict=self.strict, novelty_rule=self.novelty_rule,
                              atoms=self.atoms)
            remaining = Budget(
                max_expansions=(None if budget.max_expansions is None
                                else max(0, budget.max_expansions - totals.expansions)),
                max_seconds=(None if budget.max_seconds is None
                             else max(0.0, budget.max_seconds - budget.elapsed())))
            try:
                result = search.solve(env, remaining, state)
            except ValueError:
                # `strict` refused this width. Stop rather than raise: the widths already
                # tried are a real result, and turning them into an exception would throw
                # them away.
                break
            totals.merge(result.statistics)
            last = result
            if result.solved:
                result.statistics = totals
                return result
            if result.status == "out_of_budget":
                break
            if result.status == "exhausted" and not result.statistics.pruned_novelty:
                # Nothing was ever discarded for novelty, so this width searched the whole
                # reachable space. No larger one can reach further, and this is a proof that
                # there is no plan, reported as `exhausted` to keep it distinguishable from
                # the widths simply running out.
                return self.__failed__("exhausted", totals)

        if last is None:
            return self.__failed__("failed", totals)
        # A width that emptied its frontier *while pruning* proves nothing: a wider one
        # would have seen more, and the loop only stopped trying because `max_width` (or
        # `strict`) said so. Passing that width's "exhausted" through would collide with
        # the completeness proof above, which the benchmark reads as "there is no plan"
        # (`catalogue.is_complete`), so hitting the ceiling reports "failed" instead.
        # (`last is not None`, not `if last`: SearchResult.__bool__ is `solved`, so a plain
        # truth test on an unsolved result is always False.)
        return self.__failed__("failed" if last.status == "exhausted" else last.status,
                               totals)

    @staticmethod
    def __failed__(status, totals):
        result = SearchResult(status=status)
        result.statistics = totals
        return result

