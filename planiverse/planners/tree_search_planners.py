
import random
import math
import heapq
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from heapq import heappop, heappush
from tqdm import tqdm

@dataclass()
class TreeNode:
	state: object  = None
	action: object = None
	child_nodes:Dict = field(default_factory=dict)  # mapping from action sets to visited nodes
	after_novelty_pruning_child_nodes:Dict = field(default_factory=dict)  # mapping from action sets to visited nodes

class Heuristic:
	def __init__(self, env) -> None:
		self.env = env
	def __call__(self, state) -> float:
		assert False, "This should be implemented by the agent."
	def is_dead_state(self, state) -> bool:
		return False

class CostFunction:
	def __init__(self, env) -> None:
		self.env = env
	def __call__(self, state_trace:List, action_trace:List) -> float:
		return len(action_trace) # this is the default cost function

class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.

      Note that this PriorityQueue does not allow you to change the priority
      of an item.  However, you may insert the same item multiple times with
      different priorities.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        # FIXME: restored old behaviour to check against old results better
        # FIXED: restored to stable behaviour
        # entry = (priority, self.count, item)
        entry = (priority, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        # (_, _, item) = heapq.heappop(self.heap)
        (_, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

class TreeSearchPlanner:
	def __init__(self):
		pass
	
	def search(self, state, env, hfn, costfn):
		queue = PriorityQueue()
		visited = set()
		# init_state, info = self.env.reset()
		queue.push(([state], [], []), 0)
		while not queue.isEmpty():
			state_trace, action_trace, ltl_trace = queue.pop()
			state = state_trace[0]
			if env.is_goal(state):  return action_trace
			if state.literals in visited: continue
			visited.add(state.literals)
			for action, successor_state in env.successors(state):
				successor_state_trace  = [successor_state] + state_trace
				successor_action_trace = action_trace + [action]
				key = hfn(successor_state) + costfn(successor_state_trace, successor_action_trace)
				# print(f'Debug: {state}, {successor_state}, {action}, h={hfn(successor_state)}, g={costfn(successor_state_trace, successor_action_trace)}, f={key}')
				queue.push((successor_state_trace, successor_action_trace, ltl_trace), key)
		return []
