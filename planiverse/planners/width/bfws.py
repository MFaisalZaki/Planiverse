"""Best-First Width Search, and Dual BFWS.

IW uses novelty as a **filter**: states that fail the test are thrown away, which bounds the
frontier beautifully and makes IW(k) incomplete. BFWS uses novelty as a **sort key**: nothing
is discarded, so completeness is kept, and the search still goes where the novelty is
(Lipovetzky and Geffner, *Best-First Width Search: Exploration and Exploitation in Classical
Planning*, AAAI 2017).

The other half of the idea is that novelty is measured *within a partition* of the state
space rather than globally. Plain novelty runs out: once every atom has been seen somewhere,
no state is ever novel again and the ordering goes flat. Partitioning by something that
measures progress gives each partition its own novelty budget, so arriving somewhere new
renews exploration instead of ending it.

Classically the partition is the number of unachieved goals `#g`, giving the evaluation
function `f5 = <w_{#g}, #g>`, and the strongest variant adds `#r`, the number of atoms of a
*relevant* set `R` reached along the path: `<w_{(#g, #r)}, #g, #r>`. **A simulator has no goal
conjunction to count** (`is_goal` is a black box), so `#g` comes from a `progress` callback,
and `R` cannot be read off a relaxed plan. Francès, Ramírez, Lipovetzky and Geffner (*Purely
Declarative Action Descriptions are Overrated: Classical Planning with Simulators*, IJCAI
2017) compute `R` without a model by running IW first and taking the atoms on the paths it
finds to states that achieve something; `BFWS(relevant="iw")` does the same, "achieves
something" read as strictly improving `progress`. Without `progress` the search degrades to
`<w, h>`, which is a weaker search rather than a broken one.

There is also a **pruned variant**, k-BFWS: keep the ordering but discard states whose
novelty exceeds the width, the way IW does. That trades completeness back for IW's bounded
frontier, and it is the round `DualBFWS` runs at width 1, then 2, …, before falling back on
one unpruned (complete) round: the polynomial-first, complete-last shape of Dual-BFWS
(Lipovetzky and Geffner, *A Polynomial Planning Algorithm that Beats LAMA and FF*, ICAPS
2017).
"""
from heapq import heappop, heappush
from itertools import count

from planiverse.planners.width.novelty import NoveltyTable, PartitionedNovelty
from planiverse.planners.width.result import Budget, SearchResult, SearchStatistics

RELEVANT = (None, "iw")


class BFWS:
    """Best-first search ordered by `<novelty, progress[, #r][, heuristic]>`.

    ```python
    from planiverse.planners.width import BFWS

    result = BFWS(
        width=1,
        progress=lambda s: s.blocks_remaining,      # stands in for #unachieved-goals
    ).solve(env, budget=Budget(max_expansions=500))
    BFWS(width=1, progress=boxes, relevant="iw").solve(env)   # BFWS(R), R from IW(1)
    ```

    Both callbacks are optional and both are worth supplying. With neither, the key is
    novelty alone and ties break FIFO, which is a breadth-first search that prefers novel
    states: respectable, but it has no idea which way the goal is.
    """

    def __init__(self, width=1, progress=None, heuristic=None, partition=None, strict=True,
                 prune=False, atoms=None, relevant=None, r_width=1, r_share=0.25):
        """
        - `progress(state)`: lower is better; stands in for the unachieved-goal count.
        - `heuristic(state)`: lower is better; breaks ties among equally-progressed states.
        - `partition(state)`: what novelty is measured within. Defaults to `progress` (and
          `#r` when `relevant` is set), which is the classical choice.
        - `atoms(state)`: what novelty is measured over; `state.literals` when `None`. The
          hook `BoundaryExtensionFeatures` (see `bee.py`) plugs into.
        - `prune`: discard states whose novelty exceeds `width` instead of merely sorting
          them last. This is k-BFWS: IW's bounded frontier with BFWS's ordering inside it,
          and IW's incompleteness back with it. It exists to be a round of `DualBFWS`;
          leave it False for the complete search this class is named after.
        - `relevant`: `"iw"` computes the relevant atoms `R` with an IW(`r_width`) pre-search
          on `r_share` of the expansion budget and adds `#r` to the partition and the key;
          if that pre-search reaches the goal, its plan is returned. `None` is BFWS with
          `R` empty, the `BFWS(R_0)` of the IJCAI 2017 paper.
        """
        if relevant not in RELEVANT:
            raise ValueError(f"relevant must be one of {RELEVANT}, got {relevant!r}")
        if not 0.0 < r_share < 1.0:
            raise ValueError(f"r_share must be in (0, 1), got {r_share}")
        self.width = width
        self.progress = progress
        self.heuristic = heuristic
        self.partition = partition
        self.strict = strict
        self.prune = prune
        self.atoms = atoms
        self.relevant = relevant
        self.r_width = r_width
        self.r_share = r_share
        self.relevant_atoms = frozenset()

    def __atoms__(self, state):
        return state.literals if self.atoms is None else self.atoms(state)

    def __partition_of__(self, state, reached=frozenset()):
        if self.partition is not None:
            return self.partition(state)
        base = self.progress(state) if self.progress is not None else 0
        return (base, len(reached)) if self.relevant else base

    def __key__(self, state, novelty, reached=frozenset()):
        key = [novelty]
        if self.progress is not None:
            key.append(self.progress(state))
        if self.relevant:
            key.append(-len(reached))
        if self.heuristic is not None:
            key.append(self.heuristic(state))
        return tuple(key)

    # ------------------------------------------------------------------ the pre-search

    def __find_relevant__(self, env, state, budget, statistics):
        """IW(`r_width`) from `state`. Returns `(plan, trace)` if it reaches a goal, else None.

        Fills `relevant_atoms` with the atoms on the paths to each state that improved the
        best progress seen so far.
        """
        table = NoveltyTable(self.r_width, strict=self.strict)
        table.evaluate_and_record(self.__atoms__(state))
        best = self.progress(state)
        queue = [(state, [], [state], frozenset(state.literals))]
        closed = {state.literals}
        relevant = set()
        head = 0
        while head < len(queue):
            if budget.exhausted(statistics.expansions):
                break
            node, plan, trace, path_atoms = queue[head]
            head += 1
            statistics.expansions += 1
            for action, successor in env.successors(node):
                statistics.generated += 1
                if successor.literals in closed:
                    statistics.pruned_duplicate += 1
                    continue
                if table.evaluate_and_record(self.__atoms__(successor)) > self.r_width:
                    statistics.pruned_novelty += 1
                    continue
                closed.add(successor.literals)
                atoms = path_atoms | frozenset(successor.literals)
                if env.is_goal(successor):
                    self.relevant_atoms = frozenset(relevant | atoms)
                    return plan + [action], trace + [successor]
                if env.is_terminal(successor):
                    statistics.pruned_terminal += 1
                    continue
                value = self.progress(successor)
                if value < best:
                    best = value
                    relevant |= atoms
                queue.append((successor, plan + [action], trace + [successor], atoms))
        statistics.novelty_evaluations += table.evaluations
        statistics.tuples_enumerated += table.tuples_enumerated
        self.relevant_atoms = frozenset(relevant)
        return None

    # ------------------------------------------------------------------ the search

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics(widths_tried=(self.width,))
        novelty = PartitionedNovelty(self.width, strict=self.strict)
        tiebreak = count()          # FIFO among equal keys, and keeps states out of compares

        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return self.__result__("solved", [], [state], statistics, budget, novelty)

        self.relevant_atoms = frozenset()
        if self.relevant == "iw" and self.progress is not None:
            share = (None if budget.max_expansions is None
                     else max(1, int(budget.max_expansions * self.r_share)))
            found = self.__find_relevant__(env, state,
                                           Budget(share, budget.max_seconds).start(), statistics)
            if found is not None:
                plan, trace = found
                statistics.widths_tried = (self.r_width,)
                return self.__result__("solved", plan, trace, statistics, budget, novelty,
                                       width=self.r_width)

        heap = []
        reached = frozenset(state.literals) & self.relevant_atoms
        opened = novelty.evaluate_and_record(self.__partition_of__(state, reached),
                                             self.__atoms__(state))
        heappush(heap, (self.__key__(state, opened, reached), next(tiebreak), state, [],
                        [state], reached))
        closed = set()

        while heap:
            if budget.exhausted(statistics.expansions):
                return self.__result__("out_of_budget", None, [], statistics, budget, novelty)

            _, _, node, plan, trace, reached = heappop(heap)
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

                now = reached | (frozenset(successor.literals) & self.relevant_atoms)
                # Novelty orders; it never discards. That is the whole difference from IW,
                # and it is why BFWS stays complete at any width, unless `prune` was asked
                # for, which turns this back into a filter (k-BFWS) and gives IW's
                # incompleteness back with IW's bounded frontier.
                score = novelty.evaluate_and_record(self.__partition_of__(successor, now),
                                                    self.__atoms__(successor))
                if self.prune and score > self.width:
                    statistics.pruned_novelty += 1
                    continue
                heappush(heap, (self.__key__(successor, score, now), next(tiebreak),
                                successor, successor_plan, successor_trace, now))

        return self.__result__("exhausted", None, [], statistics, budget, novelty)

    def __result__(self, status, plan, trace, statistics, budget, novelty, width=None):
        statistics.elapsed = budget.elapsed()
        statistics.novelty_evaluations += novelty.evaluations
        statistics.tuples_enumerated += novelty.tuples_enumerated
        if status != "solved" and self.relevant and self.progress is None:
            status = f"{status} (no progress measure; R is empty and BFWS(R) is BFWS)"
        return SearchResult(plan=plan, states=trace, status=status,
                            width=(width or self.width) if status == "solved" else None,
                            statistics=statistics)


class DualBFWS:
    """Pruned BFWS at width 1, then 2, …, then one unpruned round as the safety net.

    Plain `BFWS` is already complete, so this is not `IW`'s cure for incompleteness: it is a
    *budget* strategy. A pruned round (k-BFWS: BFWS's ordering inside IW's novelty filter)
    has IW's bounded frontier, so it is cheap, and its ordering means it usually finds the
    goal long before IW(k) would. The rounds escalate width only when the filter was
    genuinely too tight, and if every allowed width fails, the last of the budget goes to
    one unpruned round, which is complete. This is the shape of Lipovetzky and Geffner's
    Dual-BFWS: polynomial first, complete last.

    ```python
    from planiverse.planners.width import DualBFWS

    result = DualBFWS(
        max_width=2,
        progress=lambda s: s.blocks_remaining,
    ).solve(env, budget=Budget(max_expansions=500))
    ```

    The budget is shared across the rounds, and the loop stops early the same three ways
    `IW` does: a round solves it, the budget runs out, or a pruned round empties its
    frontier **without discarding anything for novelty**, at which point it saw the whole
    reachable space, no wider or unpruned round can see more, and the `exhausted` it
    reports is a proof that there is no plan. The unpruned round's `exhausted` is the same
    proof, because nothing was discarded there by construction. Every other way of stopping
    proves nothing and is reported as `failed` or `out_of_budget`, never `exhausted`: the
    benchmark reads `exhausted` as unsolvability, so the word is reserved for when it is
    true.

    `statistics.widths_tried` lists the rounds in order; a trailing `1` after the ceiling is
    the unpruned round, which always runs at width 1 because completeness there costs the
    same at every width and the tuple enumeration does not.
    """

    def __init__(self, max_width=1000, progress=None, heuristic=None, partition=None,
                 strict=True, final_complete=True, atoms=None, relevant=None):
        """`final_complete` is the unpruned round. Turning it off leaves only the pruned
        rounds: cheaper, and incomplete the way IW is. `relevant` is passed to every round,
        each of which computes its own `R`."""
        if max_width < 1:
            raise ValueError(f"max_width must be at least 1, got {max_width}")
        self.max_width = max_width
        self.progress = progress
        self.heuristic = heuristic
        self.partition = partition
        self.strict = strict
        self.final_complete = final_complete
        self.atoms = atoms
        self.relevant = relevant

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
        search = BFWS(width, progress=self.progress, heuristic=self.heuristic,
                      partition=self.partition, strict=self.strict, prune=prune,
                      atoms=self.atoms, relevant=self.relevant)
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
