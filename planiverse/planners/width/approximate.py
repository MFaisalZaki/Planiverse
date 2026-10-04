"""Approximate novelty: Bloom filters, sampled tuples, and skipped expansions.

Singh, Lipovetzky, Ramírez and Segovia-Aguas, *Approximate Novelty Search*, ICAPS 2021.
Exact novelty at width `k` enumerates every `k`-tuple of every state's atoms and remembers
all of them, which is why `NoveltyTable` refuses widths above 2. The paper trades exactness
for a polynomial bound in three places:

* **Bloom filters** hold the seen tuples: fixed memory, no false negatives, a false-positive
  rate that rises as the filter fills (a state may be called non-novel when it is novel,
  never the reverse). A bank of filters serves the partitions; when there are more
  partitions than filters, partitions share one at random, as the paper does.
* **Sampled tuples**: when a state has more `k`-tuples than `sample`, only that many,
  drawn uniformly, are tested and recorded.
* **An adaptive policy skips expansions**: the paper forgoes expanding some open-list nodes
  under a bound on the search space, and its exact rule could not be read from here. The
  stand-in is stated rather than hidden: a node whose novelty exceeds `width` is expanded
  with probability `1 - generated / space_bound`, so nothing is skipped early and non-novel
  nodes are increasingly skipped as the search approaches the bound. `space_bound=None`
  disables it.

The ordering is BFWS's `<novelty, progress>`, novelty measured within the `progress`
partition. With the policy on, the search is incomplete.
"""
import hashlib
import random
from heapq import heappop, heappush
from itertools import combinations, count
from math import comb

from planiverse.planners.common import finish
from planiverse.planners.width.result import Budget, SearchStatistics


class BloomFilter:
    """A plain Bloom filter over hashable tuples; `bits` wide, `hashes` probes per key."""

    def __init__(self, bits=1 << 20, hashes=3):
        self.bits = bits
        self.hashes = hashes
        self.array = bytearray((bits + 7) // 8)
        self.entries = 0

    def __positions__(self, key):
        digest = hashlib.blake2b(repr(key).encode(), digest_size=16).digest()
        a = int.from_bytes(digest[:8], "big")
        b = int.from_bytes(digest[8:], "big") | 1
        return [(a + i * b) % self.bits for i in range(self.hashes)]

    def __contains__(self, key):
        return all(self.array[p >> 3] & (1 << (p & 7)) for p in self.__positions__(key))

    def add(self, key):
        for p in self.__positions__(key):
            self.array[p >> 3] |= 1 << (p & 7)
        self.entries += 1


class BloomNoveltyTable:
    """Novelty up to `width` with sampled tuples remembered in a Bloom filter."""

    def __init__(self, width=1, bits=1 << 20, hashes=3, sample=None, rng=None):
        if width < 1:
            raise ValueError(f"width must be at least 1, got {width}")
        self.width = width
        self.sample = sample
        self.random = rng or random.Random(0)
        self.filter = BloomFilter(bits, hashes)
        self.evaluations = 0
        self.tuples_enumerated = 0

    def __tuples__(self, atoms, size):
        if self.sample is None or comb(len(atoms), size) <= self.sample:
            return combinations(atoms, size)
        return (tuple(sorted(self.random.sample(atoms, size))) for _ in range(self.sample))

    def evaluate_and_record(self, literals):
        self.evaluations += 1
        atoms = sorted(literals)
        novelty = self.width + 1
        fresh = []
        for size in range(1, min(self.width, len(atoms)) + 1):
            for combo in self.__tuples__(atoms, size):
                self.tuples_enumerated += 1
                if combo not in self.filter:
                    fresh.append(combo)
                    novelty = min(novelty, size)
        for combo in fresh:
            self.filter.add(combo)
        return novelty


class ApproximateNoveltySearch:
    """BFWS with Bloom-filter novelty, sampled tuples and an expansion-skipping policy.

    ```python
    from planiverse.planners.width import ApproximateNoveltySearch

    result = ApproximateNoveltySearch(width=3, progress=boxes, sample=500,
                                      space_bound=50_000, seed=0).solve(env, budget)
    ```

    `width` above 2 is allowed here without `strict=False`: bounding the cost is the point.
    """

    def __init__(self, width=2, progress=None, sample=1000, bits=1 << 20, hashes=3,
                 filters=64, space_bound=None, seed=None):
        if width < 1:
            raise ValueError(f"width must be at least 1, got {width}")
        self.width = width
        self.progress = progress
        self.sample = sample
        self.bits = bits
        self.hashes = hashes
        self.filters = filters
        self.space_bound = space_bound
        self.random = random.Random(seed)

    def solve(self, env, budget=None, state=None):
        budget = (budget or Budget()).start()
        statistics = SearchStatistics(widths_tried=(self.width,))
        bank, assigned = [], {}

        def table_for(node):
            key = self.progress(node) if self.progress else 0
            if key not in assigned:
                if len(bank) < self.filters:
                    bank.append(BloomNoveltyTable(self.width, self.bits, self.hashes,
                                                  self.sample, self.random))
                    assigned[key] = bank[-1]
                else:
                    assigned[key] = self.random.choice(bank)
            return assigned[key]

        if state is None:
            state, _ = env.reset()
        if env.is_goal(state):
            return finish("solved", [], [state], statistics, budget, self.width)
        tiebreak = count()
        heap = [((table_for(state).evaluate_and_record(state.literals),
                  self.progress(state) if self.progress else 0), next(tiebreak),
                 state, [], [state])]
        closed = set()
        skipped = 0
        while heap:
            if budget.exhausted(statistics.expansions):
                return self.__done__("out_of_budget", statistics, budget, bank, skipped)
            (novelty, _), _, node, plan, trace = heappop(heap)
            if node.literals in closed:
                statistics.pruned_duplicate += 1
                continue
            if (self.space_bound and novelty > self.width
                    and self.random.random() < min(1.0, statistics.generated
                                                   / self.space_bound)):
                skipped += 1
                continue
            closed.add(node.literals)
            statistics.expansions += 1
            for action, successor in env.successors(node):
                statistics.generated += 1
                if successor.literals in closed:
                    statistics.pruned_duplicate += 1
                    continue
                successor_plan, successor_trace = plan + [action], trace + [successor]
                if env.is_goal(successor):
                    return finish("solved", successor_plan, successor_trace, statistics,
                                  budget, self.width)
                if env.is_terminal(successor):
                    statistics.pruned_terminal += 1
                    continue
                score = table_for(successor).evaluate_and_record(successor.literals)
                heappush(heap, ((score, self.progress(successor) if self.progress else 0),
                                next(tiebreak), successor, successor_plan, successor_trace))
        return self.__done__("failed" if skipped else "exhausted", statistics, budget, bank,
                             skipped)

    def __done__(self, status, statistics, budget, bank, skipped):
        statistics.novelty_evaluations = sum(t.evaluations for t in bank)
        statistics.tuples_enumerated = sum(t.tuples_enumerated for t in bank)
        statistics.pruned_novelty = skipped
        return finish(status, None, [], statistics, budget)
