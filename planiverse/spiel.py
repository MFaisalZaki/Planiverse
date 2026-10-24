"""OpenSpiel as the interface: every simulator a `pyspiel.Game`, every planner a function of one.

    import pyspiel
    import planiverse.spiel as spiel

    game = pyspiel.load_game("planiverse_lolo", {"index": 3})   # or {"seed": 7}
    actions, result = spiel.solve(game, "bfws")                 # a plan, as action ids

    state = game.new_initial_state()
    for action in actions:
        state.apply_action(action)
    assert state.returns() == [1.0]

Two adapters, and nothing else, so that neither the simulators nor the planners are written
twice.

`PlaniverseGame` puts an `Environment` behind OpenSpiel's API: a single-player, deterministic,
perfect-information game with terminal reward 1 at a goal and 0 anywhere else. A goal and a
dead end are both terminal, and a terminal state has no legal actions. Each environment is
registered as `planiverse_<name>`, with `index` (a bundled instance) or `seed` (a generated
one) as game parameters. The state carries what OpenSpiel has no slot for, the `literals`
the width planners need, and hands any other attribute through to the simulator's own state,
so a progress measure written for one reads the other.

`SpielEnvironment` goes the other way: it gives any single-player `pyspiel.Game` the
planners' contract, so every planner in the library runs on OpenSpiel's own games as well as
ours. A state's atoms are its `literals` when it has them, else one `obs(index)=value` per
nonzero entry of its observation tensor (the RAM-byte atoms IW used on Atari, Lipovetzky,
Ramírez and Geffner 2015) or else its string, plus its returns and whether it is terminal.
Chance nodes are resolved by a generator seeded on the game's seed and the history, which
keeps expansion deterministic, as the contract requires.

`solve(game, planner)` is the planners in OpenSpiel's own pattern: a function of a game that
returns what it found, here a plan of action ids rather than a policy.
"""
import inspect
import random

import pyspiel

from planiverse.environments.base import Environment
from planiverse.environments.registry import REGISTRY, make
from planiverse.planners.common import SuccessorCache

_NO_SEED = -1


def _game_type(name, long_name):
    return pyspiel.GameType(
        short_name=name,
        long_name=long_name,
        dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
        chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
        information=pyspiel.GameType.Information.PERFECT_INFORMATION,
        utility=pyspiel.GameType.Utility.GENERAL_SUM,
        reward_model=pyspiel.GameType.RewardModel.TERMINAL,
        max_num_players=1,
        min_num_players=1,
        provides_information_state_string=False,
        provides_information_state_tensor=False,
        provides_observation_string=False,
        provides_observation_tensor=False,
        parameter_specification={"index": 0, "seed": _NO_SEED, "max_actions": 100_000,
                                 "max_game_length": 100_000})


class PlaniverseGame(pyspiel.Game):
    """A Planiverse environment as a single-player OpenSpiel game.

    Action ids index the environment's `get_actions()` when it offers one, so they mean the
    same thing in every game built from the same instance. Environments whose actions are
    built per state number them in the order they are first seen; ids are then stable within
    one game object, which is all a plan needs, and `max_actions` is the declared bound.
    """

    #: The registry entry this game builds; set on the per-environment subclass.
    spec = None

    def __init__(self, params=None):
        spec = self.spec
        params = params or {}
        seed = params.get("seed", _NO_SEED)
        self.env_name = spec.name
        self.env = make(spec.name, **({"seed": seed} if seed != _NO_SEED
                                      else {"index": params.get("index", 0)}))
        self.env.reset()                  # some action lists are read off the reset state
        self.actions = []
        if self.env.provides("get_actions"):
            self.actions = list(dict.fromkeys(self.env.get_actions()))
        self.ids = {action: i for i, action in enumerate(self.actions)}
        self.cache = SuccessorCache(self.env)
        bound = max(len(self.actions), params.get("max_actions", 100_000))
        info = pyspiel.GameInfo(num_distinct_actions=bound, max_chance_outcomes=0,
                                num_players=1, min_utility=0.0, max_utility=1.0,
                                utility_sum=None,
                                max_game_length=params.get("max_game_length", 100_000))
        super().__init__(_game_type(f"planiverse_{spec.name}", spec.summary), info, params)

    def new_initial_state(self):
        return PlaniverseState(self, self.env.reset()[0])

    def action_id(self, action):
        if action not in self.ids:
            if len(self.actions) >= self.num_distinct_actions():
                raise ValueError(f"{self.env.name} has more than {len(self.actions)} distinct "
                                 "actions; raise the max_actions game parameter")
            self.ids[action] = len(self.actions)
            self.actions.append(action)
        return self.ids[action]

    def expand(self, inner):
        """`{action id: successor}`, from one simulator expansion per state."""
        return {self.action_id(action): child for action, child in self.cache.expand(inner)}


class PlaniverseState(pyspiel.State):
    """An environment's state behind OpenSpiel's API. `inner` is the simulator's own."""

    def __init__(self, game, inner):
        super().__init__(game)
        self.inner = inner
        self.literals = inner.literals

    def __getattr__(self, name):
        # Only reached for what neither OpenSpiel nor this class defines.
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def __deepcopy__(self, memo):
        # What `clone()` calls. The simulator's states are values, so a clone shares them,
        # and the game (and the simulator behind it) is never copied.
        return PlaniverseState(self.get_game(), self.inner)

    def current_player(self):
        return pyspiel.PlayerId.TERMINAL if self.is_terminal() else 0

    def is_terminal(self):
        env = self.get_game().env
        return env.is_goal(self.inner) or env.is_terminal(self.inner)

    def _legal_actions(self, player):
        return sorted(self.get_game().expand(self.inner))

    def _apply_action(self, action):
        self.inner = self.get_game().expand(self.inner)[action]
        self.literals = self.inner.literals

    def _action_to_string(self, player, action):
        return str(self.get_game().actions[action])

    def returns(self):
        return [1.0 if self.get_game().env.is_goal(self.inner) else 0.0]

    def __str__(self):
        return str(self.inner)


def _register():
    for spec in REGISTRY:
        name = f"planiverse_{spec.name}"
        if name not in pyspiel.registered_names():
            # A class, not a closure: OpenSpiel keeps the factory past interpreter shutdown,
            # and a registered lambda segfaults the process on exit.
            pyspiel.register_game(_game_type(name, spec.summary),
                                  type(f"PlaniverseGame_{spec.name}", (PlaniverseGame,),
                                       {"spec": spec}))


_register()


# --------------------------------------------------------------------- the other direction

class SpielNode:
    """A `pyspiel.State` with the identity the planners key on: hashable `literals`."""

    __slots__ = ("state", "literals")

    def __init__(self, state):
        self.state = state
        literals = getattr(state, "literals", None)
        if literals is None:
            try:
                literals = frozenset(f"obs({i})={v:g}"
                                     for i, v in enumerate(state.observation_tensor(0)) if v)
            except pyspiel.SpielError:
                literals = frozenset([str(state)])
            # The return and the end are part of an OpenSpiel state even where the observation
            # leaves them out, and a width planner must see a goal as new to reach it.
            literals |= {f"returns{tuple(state.returns())}", f"terminal({state.is_terminal()})"}
        self.literals = literals

    def __getattr__(self, name):
        return getattr(self.state, name)

    def __eq__(self, other):
        return isinstance(other, SpielNode) and self.literals == other.literals

    def __hash__(self):
        return hash(self.literals)

    def __lt__(self, other):
        # Only ever asked to break a tie in a priority queue.
        return id(self) < id(other)

    def __str__(self):
        return str(self.state)


class SpielEnvironment(Environment):
    """Any single-player `pyspiel.Game`, through the contract the planners are written to.

    `goal(state)` decides which terminal states are goals; by default a terminal state with a
    positive return, which is every goal of a `planiverse_*` game.
    """

    def __init__(self, game, goal=None, seed=0):
        if game.num_players() != 1:
            raise ValueError("planning needs a single-player game; "
                             f"{game.get_type().short_name} has {game.num_players()}")
        super().__init__(game.get_type().short_name)
        self.game = game
        self.goal = goal or (lambda state: state.returns()[0] > 0)
        self.seed = seed

    def _settle(self, state):
        """Resolve chance nodes, the same way every time for the same history."""
        while state.is_chance_node():
            outcomes, probabilities = zip(*state.chance_outcomes())
            draw = random.Random(hash((self.seed, tuple(state.history()))))
            state.apply_action(draw.choices(outcomes, probabilities)[0])
        return state

    def reset(self):
        return SpielNode(self._settle(self.game.new_initial_state())), {}

    def __advance__(self, node, action):
        state = node.state.clone()
        state.apply_action(action)
        return SpielNode(self._settle(state))

    def successors(self, node):
        if node.state.is_terminal():
            return []
        children = [(action, self.__advance__(node, action))
                    for action in node.state.legal_actions()]
        return [(action, child) for action, child in children
                if child.literals != node.literals]

    def is_goal(self, node):
        return node.state.is_terminal() and self.goal(node.state)

    def is_terminal(self, node):
        return node.state.is_terminal()


def solve(game, planner="bfws", budget=None, progress=None, goal=None, **options):
    """Plan in `game`: `(actions, result)`, the action ids of the plan (`None` if none was
    found) and the planner's `SearchResult`.

    `planner` is a benchmark tag (`"bfws"`, `"iw"`, `"gbfs"`, ...) or a planner object.
    `progress(state)`, lower is better, goes to the planners that take one; for a
    `planiverse_*` game it defaults to the benchmark's measure for that environment.
    """
    from planiverse.benchmark.measures import MEASURES
    from planiverse.benchmark.planners import PLANNERS

    env = SpielEnvironment(game, goal=goal)
    if isinstance(planner, str):
        cls, defaults = PLANNERS[planner]
        if progress is None and isinstance(game, PlaniverseGame):
            progress = MEASURES.get(game.env_name)
        kwargs = {**defaults, **options}
        if progress is not None and "progress" in inspect.signature(cls).parameters:
            kwargs["progress"] = progress
        planner = cls(**kwargs)
    result = planner.solve(env, budget=budget)
    return result.plan, result
