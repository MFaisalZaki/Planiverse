from typing import Any, Dict, List, Tuple
# from planiverse.planners.tree_search_planners import TreeSearchPlanner, Heuristic, CostFunction
import random
import math
import heapq
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
from heapq import heappop, heappush
# from tqdm import tqdm

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
				print(f'Debug: {successor_state}, {action}, h={hfn(successor_state)}, g={costfn(successor_state_trace, successor_action_trace)}, f={key}')
				queue.push((successor_state_trace, successor_action_trace, ltl_trace), key)
		return []



class SuperMarioPlanner(TreeSearchPlanner):
	# this is a reimplementation of Robin Baumgarten’s original agent (RBA0)
	def __init__(self, env) -> None:
		# override the is_goal method in the env since this is an iterative replanning.
		super().__init__()
		
	def __is_goal__(self, state):
		# Against level_progress, not mario_position.x. Mario's X is a *screen* coordinate
		# that stops at $51 (81) once the camera takes over, and he starts around $32 (50),
		# so a window of 175 on it could never be reached and this search never terminated.
		return state.level_progress >= self.root.level_progress + 175
	
	def __cost_fn__(self, state_trace, action_trace):
		combined_action = 2 if '+' in action_trace[-1].action else 1
		return 1.0*abs(state_trace[0].timeleft - state_trace[-1].timeleft) + state_trace[0].depth + combined_action

	def __hueristic_fn__(self, state):
		# Same reason as __is_goal__: mario_position.x saturates, level_progress does not.
		distance_delta = state.level_progress - self.root.level_progress
		damage_penalty = state.mario_damage() * 100000
		if distance_delta <= 0 : return 100000 + damage_penalty
		return 1.2*(-1 * distance_delta + damage_penalty)

	def search(self, env):
		# We need to have some termination condition here for the overall game.
		self.root, _ = env.reset()
		env.is_goal = self.__is_goal__
		costfn = self.__cost_fn__
		hfn = self.__hueristic_fn__
		# hfn = lambda s: 0
		# Here is a tricky thing: we need to plan until the window ends,
		# execute the plan and then start a new plan.
		plan = super().search(self.root, env, hfn, costfn)
		pass





	#     pass