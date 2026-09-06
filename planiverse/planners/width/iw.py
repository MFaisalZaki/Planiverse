"""IW(k), Iterated Width and Serialised IW, against a simulator.

IW(k) is breadth-first search with one addition: a state is discarded unless its novelty is
at most `k`. That single filter is what makes it work: it turns an exponential frontier into
one bounded by the number of atom tuples of size ≤ k, so IW(1) expands at most as many states
as there are atoms.

The price is that IW(k) is **incomplete** for fixed k: a problem needing a state of novelty
3 is unsolvable by IW(2) no matter how long it runs. Iterated Width answers that by running
IW(1), then IW(2), and so on. `BFWS` in the sibling module answers it differently, by using
novelty to *order* rather than to discard.

Adapting this to a simulator rather than a PDDL task changes three things:

* **There is no goal decomposition.** `is_goal` is a black-box predicate, so Serialised IW's
  "make one more subgoal true" has nothing to count. `SIWSearch` takes a `progress` callback
  instead, and says plainly what it degrades to without one.
* **Expansions are expensive** (seconds each in the power grid environment), so every search
  takes a `Budget` and reports what it spent.
* **Dead ends are real.** A PDDL task usually has none; several environments here do, and
  `is_terminal` states are dropped rather than expanded.
"""
from collections import deque

from planiverse.planners.width.novelty import NoveltyTable, path_novelty
from planiverse.planners.width.result import Budget, SearchResult, SearchStatistics


class IWSearch:
    """IW(k): breadth-first search that keeps only states of novelty ≤ k.

    ```python
    from planiverse.planners.width import IWSearch

    env.set_index(0)
    result = IWSearch(width=1).solve(env)
    if result:
        env.validate(result.plan)
    ```
    """

    def __init__(self, width=1, strict=True, novelty_rule="standard"):
        """`novelty_rule` picks how novelty is measured.

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
        table.evaluate_and_record(state.literals)
        frontier = deque([(state, [], [state], frozenset(state.literals))])
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

                if self.novelty_rule == "standard":
                    novel = table.evaluate_and_record(successor.literals) <= self.width
                else:
                    novel = path_novelty(successor.literals, path_atoms) >= self.width
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
                                 path_atoms | frozenset(successor.literals)))

        return self.__result__("exhausted", None, [], statistics, budget, table)

    def __result__(self, status, plan, trace, statistics, budget, table=None):
        statistics.elapsed = budget.elapsed()
        if table is not None:
            statistics.novelty_evaluations = table.evaluations
            statistics.tuples_enumerated = table.tuples_enumerated
        return SearchResult(plan=plan, states=trace, status=status,
                            width=self.width if status == "solved" else None,
                            statistics=statistics)


class IteratedWidth:
    """Run IW(1), IW(2), … until one solves it, or until no larger width could help.

    The standard answer to IW(k)'s incompleteness, and the right way to run IW when you do
    not already know the problem's width. Note what it costs against a simulator: each width
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
       `IteratedWidth` complete when it gets that far, and what makes a bound of 1000
       harmless rather than a thousand wasted restarts.

    Above `strict`'s practical width the tuple enumeration gets expensive fast, so widths
    beyond 2 need `strict=False`; `NoveltyTable` refuses them otherwise.
    """

    def __init__(self, max_width=2, strict=True, novelty_rule="standard"):
        if max_width < 1:
            raise ValueError(f"max_width must be at least 1, got {max_width}")
        self.max_width = max_width
        self.strict = strict
        self.novelty_rule = novelty_rule

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        totals = SearchStatistics()
        last = None

        for width in range(1, self.max_width + 1):
            search = IWSearch(width, strict=self.strict, novelty_rule=self.novelty_rule)
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


class SIWSearch:
    """Serialised IW: chain short IW searches, each ending as soon as progress is made.

    Classically "progress" means one more of the goal conjunction is true, and SIW is what
    makes width-based planning scale: it decomposes a problem no single IW call could reach.

    **A simulator has no goal conjunction to count.** `is_goal` is opaque, so there is nothing
    to be one-closer to. Supply `progress(state) -> comparable`, lower being better, and SIW
    will chain searches that each strictly reduce it:

    ```python
    SIWSearch(progress=lambda s: s.blocks_remaining).solve(env)
    ```

    Without one this degrades to plain IW(k) (it says so in `status` rather than pretending
    otherwise), because every intermediate search would have no stopping condition short of
    the goal itself.
    """

    def __init__(self, width=1, progress=None, max_rounds=50, strict=True,
                 avoid_dead_ends=True, max_width=None):
        """`max_width` makes each leg iterate its width instead of running at a fixed one.

        SIW inherits IW's width sensitivity exactly, because each leg *is* an IW search:
        novelty is a filter there, so if no state within IW(k)'s pruned reach improves
        progress, the leg fails and the whole search fails, even though a wider leg would
        have found one. Pinning `width` reports SIW at a width someone chose rather
        than at the width the problem needs, which is the same mistake as pinning IW.

        With `max_width` set, a leg tries IW(`width`), then IW(`width` + 1), and so on until
        it makes progress, the budget runs out, or a width covers everything reachable from
        the leg's start without discarding anything for novelty, at which point no wider leg
        can help either and the search is genuinely stuck. The budget is shared across the
        widths within a leg, so a stubborn leg cannot spend more than its allowance.

        Left as `None` this behaves exactly as before, at the single `width`.

        `avoid_dead_ends` stops a leg committing to progress that ends the problem.

        SIW is incomplete because each leg commits irrevocably to the first improvement it
        finds, and greedy progress can be a trap. It really happens here: on Puzznic level 1
        the first leg clears a pair and lands on a board with one block of a colour left,
        which can never be matched: progress was made and the problem is over. IW(2) and
        BFWS solve that level because they never commit.

        A simulator that computes `is_terminal` gives the leg a cheap way to refuse, which is
        sound and costs nothing: a dead end is not progress. Set it False for the classical
        behaviour.
        """
        if max_width is not None and max_width < width:
            raise ValueError(
                f"max_width {max_width} is below width {width}: a leg cannot iterate down")
        self.width = width
        self.progress = progress
        self.max_rounds = max_rounds
        self.strict = strict
        self.avoid_dead_ends = avoid_dead_ends
        self.max_width = max_width

    @property
    def ceiling(self):
        """The widest a leg may go. Equal to `width` when `max_width` is not set."""
        return self.width if self.max_width is None else self.max_width

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        if self.progress is None:
            search = (IWSearch(self.width, strict=self.strict) if self.max_width is None
                      else IteratedWidth(self.ceiling, strict=self.strict))
            result = search.solve(env, budget, state)
            if not result.solved:
                result.status = f"{result.status} (no progress measure; SIW degraded to IW)"
            return result

        totals = SearchStatistics(widths_tried=(self.width,))
        if state is None:
            state, _ = env.reset()
        plan, trace = [], [state]
        best = self.progress(state)
        # The width each leg actually needed. A problem whose hardest leg needed IW(2) is a
        # different problem from one every leg solved at IW(1), and the summary should say so.
        widths_used = []

        for _ in range(self.max_rounds):
            if env.is_goal(state):
                return self.__solved__(plan, trace, widths_used, totals, budget)
            if budget.exhausted(totals.expansions):
                break

            remaining = Budget(
                max_expansions=(None if budget.max_expansions is None
                                else max(0, budget.max_expansions - totals.expansions)),
                max_seconds=(None if budget.max_seconds is None
                             else max(0.0, budget.max_seconds - budget.elapsed())))
            leg = self.__reach_progress__(env, state, best, remaining, totals)
            if leg is None:
                break
            state, leg_plan, leg_trace, leg_width = leg
            widths_used.append(leg_width)
            plan += leg_plan
            trace += leg_trace
            best = self.progress(state)

        if env.is_goal(state):
            return self.__solved__(plan, trace, widths_used, totals, budget)
        totals.elapsed = budget.elapsed()
        if widths_used:
            totals.widths_tried = tuple(sorted(set(widths_used)))
        return SearchResult(status="failed", statistics=totals)

    def __solved__(self, plan, trace, widths_used, totals, budget):
        totals.elapsed = budget.elapsed()
        if widths_used:
            totals.widths_tried = tuple(sorted(set(widths_used)))
        return SearchResult(plan, trace, "solved", max(widths_used, default=self.width),
                            totals)

    def __reach_progress__(self, env, start, baseline, budget, totals):
        """One leg, widening until it gets somewhere.

        Returns `(state, plan, trace, width)` or `None`. The widths share one budget: a leg
        that keeps widening must not spend more than a leg that succeeded at the first width.
        """
        budget = budget.start()
        spent = 0
        for width in range(self.width, self.ceiling + 1):
            try:
                outcome, status, pruned, spent = self.__leg_at_width__(
                    env, start, baseline, budget, totals, width, spent)
            except ValueError:
                break               # `strict` refused this width; the leg stops here
            if outcome is not None:
                return outcome + (width,)
            if status == "out_of_budget":
                return None
            if not pruned:
                # Nothing was discarded for novelty, so this width saw everything reachable
                # from here and none of it was progress. A wider leg would see the same.
                return None
        return None

    def __leg_at_width__(self, env, start, baseline, budget, totals, width, spent):
        """IW(`width`) from `start`, stopping at the first state that improves on `baseline`.

        A goal also ends the leg, so the caller's loop sees it and reports solved. Returns
        `(outcome, status, pruned_for_novelty, spent)`, where `pruned_for_novelty` is what
        tells the caller whether a wider leg could see anything this one could not.
        """
        table = NoveltyTable(width, strict=self.strict)
        table.evaluate_and_record(start.literals)
        frontier = deque([(start, [], [])])
        closed = {start.literals}
        pruned = 0

        while frontier:
            # `spent` is this leg's own count, carried across widths. The budget handed in
            # has already had the previous legs' expansions subtracted from it, so checking
            # the cumulative total against it would subtract them a second time and starve
            # the search after half its allowance.
            if budget.exhausted(spent):
                return None, "out_of_budget", pruned, spent
            node, plan, trace = frontier.popleft()
            spent += 1
            totals.expansions += 1

            for action, successor in env.successors(node):
                totals.generated += 1
                if successor.literals in closed:
                    totals.pruned_duplicate += 1
                    continue
                if table.evaluate_and_record(successor.literals) > width:
                    totals.pruned_novelty += 1
                    pruned += 1
                    continue
                closed.add(successor.literals)
                successor_plan = plan + [action]
                successor_trace = trace + [successor]

                if env.is_goal(successor):
                    return (successor, successor_plan, successor_trace), "progress", \
                        pruned, spent
                if env.is_terminal(successor):
                    totals.pruned_terminal += 1
                    # A dead end is not progress. Committing to one ends the whole search,
                    # which is how SIW loses levels that IW solves.
                    if self.avoid_dead_ends:
                        continue
                    if self.progress(successor) < baseline:
                        return (successor, successor_plan, successor_trace), "progress", \
                            pruned, spent
                    continue
                if self.progress(successor) < baseline:
                    return (successor, successor_plan, successor_trace), "progress", \
                        pruned, spent
                frontier.append((successor, successor_plan, successor_trace))
        return None, "exhausted", pruned, spent
