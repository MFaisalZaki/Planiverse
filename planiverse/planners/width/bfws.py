"""Best-First Width Search.

IW uses novelty as a **filter**: states that fail the test are thrown away, which bounds the
frontier beautifully and makes IW(k) incomplete. BFWS uses novelty as a **sort key**: nothing
is discarded, so completeness is kept, and the search still goes where the novelty is.

The other half of the idea is that novelty is measured *within a partition* of the state
space rather than globally. Plain novelty runs out — once every atom has been seen somewhere,
no state is ever novel again and the ordering goes flat. Partitioning by something that
measures progress gives each partition its own novelty budget, so arriving somewhere new
renews exploration instead of ending it.

Classically the partition is the number of unachieved goals, giving the evaluation function
`f5 = <w_{#g}, #g>`. **A simulator has no goal conjunction to count** — `is_goal` is a black
box — so `#g` has to come from somewhere else. `BFWS` takes a `progress` callback for it, and
degrades to `<w, h>` without one, which is a weaker search rather than a broken one.

There is also a **pruned variant**, k-BFWS in Lipovetzky and Geffner's 2017 paper: keep the
ordering but discard states whose novelty exceeds the width, the way IW does. That trades
completeness back for IW's bounded frontier, and it is the round `IteratedBFWS` runs at
width 1, then 2, …, before falling back on one unpruned — complete — round. The result is
the polynomial-first, complete-last shape of the paper's Dual-BFWS.
"""
from heapq import heappop, heappush
from itertools import count

from planiverse.planners.width.novelty import PartitionedNovelty
from planiverse.planners.width.result import Budget, SearchResult, SearchStatistics


class BFWSSearch:
    """Best-first search ordered by `<novelty, progress, heuristic>`.

    ```python
    from planiverse.planners.width import BFWSSearch

    result = BFWSSearch(
        width=1,
        progress=lambda s: s.blocks_remaining,      # stands in for #unachieved-goals
    ).solve(env, budget=Budget(max_expansions=500))
    ```

    Both callbacks are optional and both are worth supplying. With neither, the key is
    novelty alone and ties break FIFO, which is a breadth-first search that prefers novel
    states — respectable, but it has no idea which way the goal is.
    """

    def __init__(self, width=1, progress=None, heuristic=None, partition=None, strict=True,
                 prune=False):
        """
        - `progress(state)` — lower is better; stands in for the unachieved-goal count.
        - `heuristic(state)` — lower is better; breaks ties among equally-progressed states.
        - `partition(state)` — what novelty is measured within. Defaults to `progress`, which
          is the classical choice; pass something else to partition on more than progress.
        - `prune` — discard states whose novelty exceeds `width` instead of merely sorting
          them last. This is k-BFWS: IW's bounded frontier with BFWS's ordering inside it,
          and IW's incompleteness back with it. It exists to be a round of `IteratedBFWS`;
          leave it False for the complete search this class is named after.
        """
        self.width = width
        self.progress = progress
        self.heuristic = heuristic
        self.partition = partition
        self.strict = strict
        self.prune = prune

    def __partition_of__(self, state):
        if self.partition is not None:
            return self.partition(state)
        if self.progress is not None:
            return self.progress(state)
        return 0            # one partition: novelty measured globally

    def __key__(self, state, novelty):
        key = [novelty]
        if self.progress is not None:
            key.append(self.progress(state))
        if self.heuristic is not None:
            key.append(self.heuristic(state))
        return tuple(key)

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics(widths_tried=(self.width,))
        novelty = PartitionedNovelty(self.width, strict=self.strict)
        tiebreak = count()          # FIFO among equal keys, and keeps states out of compares

        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return self.__result__("solved", [], [state], statistics, budget, novelty)

        heap = []
        opened = novelty.evaluate_and_record(self.__partition_of__(state), state.literals)
        heappush(heap, (self.__key__(state, opened), next(tiebreak), state, [], [state]))
        closed = set()

        while heap:
            if budget.exhausted(statistics.expansions):
                return self.__result__("out_of_budget", None, [], statistics, budget, novelty)

            _, _, node, plan, trace = heappop(heap)
            if node.literals in closed:
                statistics.pruned_duplicate += 1
                continue
            closed.add(node.literals)
            statistics.expansions += 1

            for action, successor in env.successors(node):
                statistics.generated += 1
                if successor.literals in closed:
                    statistics.pruned_duplicate += 1
                    continue

                successor_plan = plan + [action]
                successor_trace = trace + [successor]

                # Before the terminal test, because every absorbing goal state in this
                # library is terminal as well as a goal.
                if env.is_goal(successor):
                    return self.__result__("solved", successor_plan, successor_trace,
                                           statistics, budget, novelty)
                if env.is_terminal(successor):
                    statistics.pruned_terminal += 1
                    continue

                # Novelty orders; it never discards. That is the whole difference from IW,
                # and it is why BFWS stays complete at any width — unless `prune` was asked
                # for, which turns this back into a filter (k-BFWS) and gives IW's
                # incompleteness back with IW's bounded frontier.
                score = novelty.evaluate_and_record(self.__partition_of__(successor),
                                                    successor.literals)
                if self.prune and score > self.width:
                    statistics.pruned_novelty += 1
                    continue
                heappush(heap, (self.__key__(successor, score), next(tiebreak),
                                successor, successor_plan, successor_trace))

        return self.__result__("exhausted", None, [], statistics, budget, novelty)

    def __result__(self, status, plan, trace, statistics, budget, novelty):
        statistics.elapsed = budget.elapsed()
        statistics.novelty_evaluations = novelty.evaluations
        statistics.tuples_enumerated = novelty.tuples_enumerated
        return SearchResult(plan=plan, states=trace, status=status,
                            width=self.width if status == "solved" else None,
                            statistics=statistics)


class IteratedBFWS:
    """Pruned BFWS at width 1, then 2, …, then one unpruned round as the safety net.

    Plain `BFWSSearch` is already complete, so this is not `IteratedWidth`'s cure for
    incompleteness — it is a *budget* strategy. A pruned round (k-BFWS: BFWS's ordering
    inside IW's novelty filter) has IW's bounded frontier, so it is cheap, and its ordering
    means it usually finds the goal long before IW(k) would. The rounds escalate width only
    when the filter was genuinely too tight, and if every allowed width fails, the last of
    the budget goes to one unpruned round, which is complete. This is the shape of
    Lipovetzky and Geffner's Dual-BFWS: polynomial first, complete last.

    ```python
    from planiverse.planners.width import IteratedBFWS

    result = IteratedBFWS(
        max_width=2,
        progress=lambda s: s.blocks_remaining,
    ).solve(env, budget=Budget(max_expansions=500))
    ```

    The budget is shared across the rounds, and the loop stops early the same three ways
    `IteratedWidth` does: a round solves it, the budget runs out, or a pruned round empties
    its frontier **without discarding anything for novelty** — at which point it saw the
    whole reachable space, no wider or unpruned round can see more, and the `exhausted` it
    reports is a proof that there is no plan. The unpruned round's `exhausted` is the same
    proof, because nothing was discarded there by construction. Every other way of stopping
    proves nothing and is reported as `failed` or `out_of_budget`, never `exhausted` — the
    benchmark reads `exhausted` as unsolvability (`catalogue.is_complete`), so the word is
    reserved for when it is true.

    `statistics.widths_tried` lists the rounds in order; a trailing `1` after the ceiling is
    the unpruned round, which always runs at width 1 because completeness there costs the
    same at every width and the tuple enumeration does not.
    """

    def __init__(self, max_width=1000, progress=None, heuristic=None, partition=None,
                 strict=True, final_complete=True):
        """`final_complete` is the unpruned round. Turning it off leaves only the pruned
        rounds — cheaper, and incomplete the way IW is."""
        if max_width < 1:
            raise ValueError(f"max_width must be at least 1, got {max_width}")
        self.max_width = max_width
        self.progress = progress
        self.heuristic = heuristic
        self.partition = partition
        self.strict = strict
        self.final_complete = final_complete

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        totals = SearchStatistics()

        for width in range(1, self.max_width + 1):
            try:
                result = self.__round__(env, budget, state, totals, width, prune=True)
            except ValueError:
                # `strict` refused this width. The rounds already run are a real result;
                # fall through to the unpruned round rather than throw them away.
                break
            if result.solved:
                return result
            if result.status == "out_of_budget":
                return self.__failed__("out_of_budget", totals)
            if result.status == "exhausted" and not result.statistics.pruned_novelty:
                # Nothing was ever discarded for novelty, so this round saw the whole
                # reachable space and the unpruned round would re-run the identical search.
                # This is the proof there is no plan.
                return self.__failed__("exhausted", totals)

        if self.final_complete and not budget.exhausted(totals.expansions):
            result = self.__round__(env, budget, state, totals, width=1, prune=False)
            if result.solved:
                return result
            # Unpruned BFWS discards nothing, so its "exhausted" is the same proof the
            # pruned rounds could only reach by luck; "out_of_budget" passes through.
            return self.__failed__(result.status, totals)

        # The ceiling was hit with the filter still biting, and no unpruned round ran.
        # That proves nothing, so it must not be called "exhausted".
        return self.__failed__("failed", totals)

    def __round__(self, env, budget, state, totals, width, prune):
        search = BFWSSearch(width, progress=self.progress, heuristic=self.heuristic,
                            partition=self.partition, strict=self.strict, prune=prune)
        remaining = Budget(
            max_expansions=(None if budget.max_expansions is None
                            else max(0, budget.max_expansions - totals.expansions)),
            max_seconds=(None if budget.max_seconds is None
                         else max(0.0, budget.max_seconds - budget.elapsed())))
        result = search.solve(env, remaining, state)
        totals.merge(result.statistics)
        if result.solved:
            result.statistics = totals
        return result

    @staticmethod
    def __failed__(status, totals):
        result = SearchResult(status=status)
        result.statistics = totals
        return result
