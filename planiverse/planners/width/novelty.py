"""Novelty, the measure width-based search is built on.

The novelty of a state is the size of the **smallest tuple of atoms** it contains that has
not appeared in any state seen before. A state with a brand-new atom has novelty 1; one
whose atoms have all been seen individually but which combines two of them in a new way has
novelty 2, and so on. IW(k) throws away everything with novelty above k; BFWS sorts by it
instead of discarding.

Two things about applying this to a *simulator* rather than a PDDL task:

**The atoms are whatever `literals` says they are.** A planner cannot see inside the
transition, so the modelling decision each environment made about how coarsely to spell its
state is exactly the decision that fixes the width. The water network buckets its
contamination into twentieths; had it kept the raw float, every state would carry a new atom
and every state would have novelty 1, which makes novelty useless as a filter. Environments
here bucket deliberately for this reason.

**Width 2 costs O(n²) per state and width 3 costs O(n³).** With a hundred atoms that is
5,000 pairs and 160,000 triples, per state, and a simulator's successor is already
expensive, so `NoveltyTable` refuses widths above `MAX_PRACTICAL_WIDTH` unless asked
explicitly, and reports what it cost.
"""
from itertools import combinations

#: Beyond this, the tuple enumeration is usually more expensive than the simulator it is
#: meant to be saving. Pass `strict=False` to go higher anyway.
MAX_PRACTICAL_WIDTH = 2


class NoveltyTable:
    """Remembers every atom tuple up to `width` that any state has shown.

    Not thread-safe and not shareable between searches: the table *is* the search's memory
    of what it has seen, so two searches sharing one would prune each other's states.
    """

    def __init__(self, width=1, strict=True):
        if width < 1:
            raise ValueError(f"width must be at least 1, got {width}")
        if strict and width > MAX_PRACTICAL_WIDTH:
            raise ValueError(
                f"width {width} enumerates every {width}-tuple of every state's atoms, "
                f"which is rarely worth it against a simulator whose successors are already "
                f"expensive. Pass strict=False if you mean it.")
        self.width = width
        # `{size: set of tuples}`, filled on demand. A list of `width` sets would allocate
        # one per level whether or not any state has that many atoms, so a width of 1000
        # (what `IteratedWidth` is given as a bound) cost a thousand allocations per level
        # tried, for nine atoms' worth of work.
        self.seen = {}
        self.evaluations = 0
        self.tuples_enumerated = 0

    def evaluate(self, literals):
        """The novelty of `literals`, without recording it.

        Returns `width + 1` when nothing new is found, which is the conventional way of
        saying "not novel at this width" and sorts after every real novelty.
        """
        self.evaluations += 1
        atoms = sorted(literals)
        # A tuple longer than the state has atoms does not exist, so widths above that add
        # nothing. Capping here is what lets `IteratedWidth` take a bound of 1000 without
        # spinning through 990 empty combination ranges per state.
        for size in range(1, self.__ceiling__(atoms) + 1):
            seen = self.seen.get(size, ())
            for combo in combinations(atoms, size):
                self.tuples_enumerated += 1
                if combo not in seen:
                    return size
        return self.width + 1

    def record(self, literals):
        """Remember every tuple of `literals` up to `width`."""
        atoms = sorted(literals)
        for size in range(1, self.__ceiling__(atoms) + 1):
            self.seen.setdefault(size, set()).update(combinations(atoms, size))

    def __ceiling__(self, atoms):
        """The largest tuple size worth looking at for this state.

        A tuple longer than the state has atoms does not exist, so sizes above that are empty
        ranges, and with a width in the hundreds they are the whole cost.
        """
        return min(self.width, len(atoms))

    def evaluate_and_record(self, literals):
        """The usual pairing: score the state, then remember it.

        Kept as one call because doing them in the other order always yields `width + 1`:
        a state is never novel with respect to itself.
        """
        novelty = self.evaluate(literals)
        self.record(literals)
        return novelty

    def is_novel(self, literals):
        return self.evaluate_and_record(literals) <= self.width

    def __len__(self):
        return sum(len(level) for level in self.seen.values())

    def __repr__(self):
        return f"<NoveltyTable(width={self.width}, tuples={len(self)})>"


class PartitionedNovelty:
    """One novelty table per partition, which is what makes BFWS work.

    Plain novelty runs out: once every atom has been seen somewhere, nothing is novel again
    and the search stalls. Partitioning by something that measures progress (how many goals
    are left, how deep the plan is) gives each partition its own budget, so reaching a new
    partition renews exploration instead of ending it.
    """

    def __init__(self, width=1, strict=True):
        self.width = width
        self.strict = strict
        self.tables = {}

    def table_for(self, key):
        if key not in self.tables:
            self.tables[key] = NoveltyTable(self.width, strict=self.strict)
        return self.tables[key]

    def evaluate_and_record(self, key, literals):
        return self.table_for(key).evaluate_and_record(literals)

    @property
    def evaluations(self):
        return sum(table.evaluations for table in self.tables.values())

    @property
    def tuples_enumerated(self):
        return sum(table.tuples_enumerated for table in self.tables.values())

    def __len__(self):
        return sum(len(table) for table in self.tables.values())

    def __repr__(self):
        return f"<PartitionedNovelty(width={self.width}, partitions={len(self.tables)})>"


def path_novelty(literals, seen_on_path):
    """The rule the pyBehaviourPlanningLTL reference uses, kept for comparability.

    It counts how many atoms of the state are new **relative to the path taken to it**, and
    IW(k) keeps the state when that count is at least k. That is not the standard measure and
    the reference's own source flags it as unverified. At k=1 the two agree: "has a new
    atom". At k=2 they diverge: this asks for *two* new atoms, while standard novelty asks
    for one new *pair*, and a state can satisfy either without the other.

    It is also *path*-based rather than search-based, so the same state can be novel down one
    branch and not another. That makes IW under this rule explore more, and makes its results
    depend on visit order.
    """
    return len(frozenset(literals) - frozenset(seen_on_path))
