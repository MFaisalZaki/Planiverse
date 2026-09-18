"""A small best-first tree search that works against any environment implementing the contract.

    planner = TreeSearchPlanner()
    plan = planner.search(state, env, hfn, costfn)

The two things you supply are a `Heuristic` and a `CostFunction`, both callables: the
heuristic scores a state, the cost function scores the trace so far, and the search expands
the node with the smallest sum. `PriorityQueue` pushes `(priority, item)` tuples, so ties
compare the items themselves, which is why state and action classes define `__lt__`.
"""
import heapq
from typing import List


class Heuristic:
    def __init__(self, env) -> None:
        self.env = env

    def __call__(self, state) -> float:
        raise NotImplementedError("a Heuristic scores a state; subclass and implement __call__")

    def is_dead_state(self, state) -> bool:
        return False


class CostFunction:
    def __init__(self, env) -> None:
        self.env = env

    def __call__(self, state_trace: List, action_trace: List) -> float:
        return len(action_trace)          # the default: plan length


class PriorityQueue:
    """A priority queue with O(1) access to the lowest-priority item.

    Priorities cannot be changed once an item is in, but the same item may be inserted more
    than once with different priorities.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        heapq.heappush(self.heap, (priority, item))
        self.count += 1

    def pop(self):
        (_, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0


class TreeSearchPlanner:
    """Best-first search over `env.successors`, keyed on `hfn(state) + costfn(trace)`."""

    def search(self, state, env, hfn, costfn):
        queue = PriorityQueue()
        visited = set()
        queue.push(([state], [], []), 0)
        while not queue.isEmpty():
            state_trace, action_trace, ltl_trace = queue.pop()
            state = state_trace[0]
            if env.is_goal(state):
                return action_trace
            if state.literals in visited:
                continue
            visited.add(state.literals)
            for action, successor_state in env.successors(state):
                successor_state_trace = [successor_state] + state_trace
                successor_action_trace = action_trace + [action]
                key = hfn(successor_state) + costfn(successor_state_trace,
                                                   successor_action_trace)
                queue.push((successor_state_trace, successor_action_trace, ltl_trace), key)
        return []
