"""What a width-based search hands back.

Reporting is not decoration here. Against a simulator one expansion can cost seconds (the
power grid environment spends 8 to 19 of them per node), so "it found nothing" and "it ran
out of budget having looked at four nodes" are completely different answers and a planner
that cannot tell them apart is not usable.
"""
import time
from dataclasses import dataclass, field


@dataclass
class SearchStatistics:
    """What the search did, in enough detail to tell why it stopped."""

    expansions: int = 0             #: states whose successors were generated
    generated: int = 0              #: successors produced
    pruned_novelty: int = 0         #: discarded for failing the novelty test (IW only)
    pruned_terminal: int = 0        #: discarded as dead ends
    pruned_duplicate: int = 0       #: already in the closed list
    novelty_evaluations: int = 0
    tuples_enumerated: int = 0
    elapsed: float = 0.0
    widths_tried: tuple = ()        #: for iterated searches

    def merge(self, other):
        """Accumulate another search's counters, for the iterated and serialised planners."""
        for name in ("expansions", "generated", "pruned_novelty", "pruned_terminal",
                     "pruned_duplicate", "novelty_evaluations", "tuples_enumerated",
                     "elapsed"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.widths_tried = tuple(self.widths_tried) + tuple(other.widths_tried)
        return self

    def __str__(self):
        return (f"{self.expansions} expansions, {self.generated} generated, "
                f"{self.pruned_novelty} pruned by novelty, "
                f"{self.pruned_terminal} dead ends, {self.elapsed:.2f}s")


@dataclass
class SearchResult:
    """A plan, or a precise account of why there is not one."""

    plan: list = None               #: actions, or None
    states: list = field(default_factory=list)   #: the trace, one longer than the plan
    status: str = "failed"          #: solved | failed | exhausted | out_of_budget
    width: int = None               #: the width it was found at
    statistics: SearchStatistics = field(default_factory=SearchStatistics)

    @property
    def solved(self):
        return self.status == "solved"

    @property
    def cost(self):
        """Summed action cost when the actions report one, else the plan length."""
        if not self.plan:
            return 0
        if all(callable(getattr(action, "cost", None)) for action in self.plan):
            return sum(action.cost() for action in self.plan)
        return len(self.plan)

    def __len__(self):
        return len(self.plan or ())

    def __iter__(self):
        return iter(self.plan or ())

    def __bool__(self):
        return self.solved

    def __str__(self):
        if not self.solved:
            return f"<no plan: {self.status} — {self.statistics}>"
        actions = " → ".join(str(action) for action in self.plan) or "(empty plan)"
        return (f"plan of {len(self.plan)} actions, cost {self.cost}, "
                f"width {self.width}\n  {actions}\n  {self.statistics}")


class Budget:
    """A wall-clock and node allowance, checked as the search runs.

    Both default to unlimited. Against an expensive simulator the node budget is usually the
    one that matters, because time is dominated by however many expansions you allowed.
    """

    def __init__(self, max_expansions=None, max_seconds=None):
        self.max_expansions = max_expansions
        self.max_seconds = max_seconds
        self.started = None

    def start(self):
        self.started = time.monotonic()
        return self

    def exhausted(self, expansions):
        if self.max_expansions is not None and expansions >= self.max_expansions:
            return True
        if self.max_seconds is not None and self.started is not None:
            return time.monotonic() - self.started >= self.max_seconds
        return False

    def elapsed(self):
        return 0.0 if self.started is None else time.monotonic() - self.started
