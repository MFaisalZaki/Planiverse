"""IW(k), Iterated Width and Serialised IW, against a simulator.

IW(k) is breadth-first search with one addition: a state is discarded unless its novelty is
at most `k`. That single filter is what makes it work — it turns an exponential frontier into
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
* **Expansions are expensive** — seconds each in the power grid environment — so every search
  takes a `Budget` and reports what it spent.
* **Dead ends are real.** A PDDL task usually has none; three environments here do, and
  `is_terminal` states are dropped rather than expanded.
"""
from collections import deque

from planiverse.planners.width.novelty import NoveltyTable, path_novelty
from planiverse.planners.width.result import Budget, SearchResult, SearchStatistics


class IWSearch:
    """IW(k): breadth-first search that keeps only states of novelty ≤ k.

    ```python
    from planiverse.planners.width import IWSearch

    env.fix_index(0)
    result = IWSearch(width=1).solve(env)
    if result:
        env.validate(result.plan)
    ```
    """

    def __init__(self, width=1, strict=True, novelty_rule="standard"):
        """`novelty_rule` picks how novelty is measured.

        `"standard"` is the textbook definition: the smallest tuple of atoms in the state not
        seen anywhere in this search. `"path"` is the rule the pyBehaviourPlanningLTL
        reference uses — how many atoms are new relative to the path taken to the state — and
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

                # Checked before the terminal test: a goal that is also terminal — which is
                # every absorbing goal state in this library — must still be reported solved.
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
    """Run IW(1), IW(2), … until one solves it.

    The standard answer to IW(k)'s incompleteness. Note what it costs against a simulator:
    each width restarts from scratch and re-expands everything the previous one did, and here
    an expansion is a hydraulic solve or a power-flow solve rather than a bitmask update. The
    budget is therefore shared across widths rather than granted afresh to each.
    """

    def __init__(self, max_width=2, strict=True, novelty_rule="standard"):
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
            result = search.solve(env, remaining, state)
            totals.merge(result.statistics)
            last = result
            if result.solved:
                result.statistics = totals
                return result
            if result.status == "out_of_budget":
                break

        failed = SearchResult(status=last.status if last else "failed")
        failed.statistics = totals
        return failed


class SIWSearch:
    """Serialised IW: chain short IW searches, each ending as soon as progress is made.

    Classically "progress" means one more of the goal conjunction is true, and SIW is what
    makes width-based planning scale — it decomposes a problem no single IW call could reach.

    **A simulator has no goal conjunction to count.** `is_goal` is opaque, so there is nothing
    to be one-closer to. Supply `progress(state) -> comparable`, lower being better, and SIW
    will chain searches that each strictly reduce it:

    ```python
    SIWSearch(progress=lambda s: s.blocks_remaining).solve(env)
    ```

    Without one this degrades to plain IW(k) — it says so in `status` rather than pretending
    otherwise — because every intermediate search would have no stopping condition short of
    the goal itself.
    """

    def __init__(self, width=1, progress=None, max_rounds=50, strict=True,
                 avoid_dead_ends=True):
        """`avoid_dead_ends` stops a leg committing to progress that ends the problem.

        SIW is incomplete because each leg commits irrevocably to the first improvement it
        finds, and greedy progress can be a trap. It really happens here: on Puzznic level 1
        the first leg clears a pair and lands on a board with one block of a colour left,
        which can never be matched — progress was made and the problem is over. IW(2) and
        BFWS solve that level because they never commit.

        A simulator that computes `is_terminal` gives the leg a cheap way to refuse, which is
        sound and costs nothing: a dead end is not progress. Set it False for the classical
        behaviour.
        """
        self.width = width
        self.progress = progress
        self.max_rounds = max_rounds
        self.strict = strict
        self.avoid_dead_ends = avoid_dead_ends

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        if self.progress is None:
            result = IWSearch(self.width, strict=self.strict).solve(env, budget, state)
            if not result.solved:
                result.status = f"{result.status} (no progress measure; SIW degraded to IW)"
            return result

        totals = SearchStatistics(widths_tried=(self.width,))
        if state is None:
            state, _ = env.reset()
        plan, trace = [], [state]
        best = self.progress(state)

        for _ in range(self.max_rounds):
            if env.is_goal(state):
                totals.elapsed = budget.elapsed()
                return SearchResult(plan, trace, "solved", self.width, totals)
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
            state, leg_plan, leg_trace = leg
            plan += leg_plan
            trace += leg_trace
            best = self.progress(state)

        totals.elapsed = budget.elapsed()
        if env.is_goal(state):
            return SearchResult(plan, trace, "solved", self.width, totals)
        return SearchResult(status="failed", statistics=totals)

    def __reach_progress__(self, env, start, baseline, budget, totals):
        """One leg: IW from `start`, stopping at the first state that improves on `baseline`.

        A goal also ends the leg, so the caller's loop sees it and reports solved.
        """
        budget = budget.start()
        table = NoveltyTable(self.width, strict=self.strict)
        table.evaluate_and_record(start.literals)
        frontier = deque([(start, [], [])])
        closed = {start.literals}
        # This leg's own count. The budget handed in has already had the previous legs'
        # expansions subtracted from it, so checking the cumulative total against it would
        # subtract them a second time and starve the search after half its allowance.
        spent = 0

        while frontier:
            if budget.exhausted(spent):
                return None
            node, plan, trace = frontier.popleft()
            spent += 1
            totals.expansions += 1

            for action, successor in env.successors(node):
                totals.generated += 1
                if successor.literals in closed:
                    totals.pruned_duplicate += 1
                    continue
                if table.evaluate_and_record(successor.literals) > self.width:
                    totals.pruned_novelty += 1
                    continue
                closed.add(successor.literals)
                successor_plan = plan + [action]
                successor_trace = trace + [successor]

                if env.is_goal(successor):
                    return successor, successor_plan, successor_trace
                if env.is_terminal(successor):
                    totals.pruned_terminal += 1
                    # A dead end is not progress. Committing to one ends the whole search,
                    # which is how SIW loses levels that IW solves.
                    if self.avoid_dead_ends:
                        continue
                    if self.progress(successor) < baseline:
                        return successor, successor_plan, successor_trace
                    continue
                if self.progress(successor) < baseline:
                    return successor, successor_plan, successor_trace
                frontier.append((successor, successor_plan, successor_trace))
        return None
