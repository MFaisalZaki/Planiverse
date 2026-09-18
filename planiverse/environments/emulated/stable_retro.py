"""Any Stable-Retro integration as a planning environment.

Stable-Retro wraps a shelf of console emulators behind one Gym interface and, for each game
it integrates, a save state to start from and a list of RAM variables (`data.json`: a score,
a lives counter, a position) with a reward and a done condition over them
(`scenario.json`). That is nearly a planning environment already. What it lacks is a way
back to a state, which the emulators' save states give (`em.get_state()` and
`em.set_state()`), and a goal, which is a threshold on one of the variables or a number of
actions survived.

    env = RetroEnv("Airstriker-Genesis-v0")   # the game Stable-Retro ships a ROM for
    state, info = env.reset()
    for action, successor in env.successors(state):
        print(action, successor.variables)

Any integration works: Stable-Retro's own, with a ROM you import into it, or a custom one
(`inttype=` and `stable_retro.data.Integrations`). This module reads nothing of a game but
what its integration declares.

## Instances, and generating them

A bundled instance is one of the integration's save states (`Airstriker-Genesis-v0` ships
one, `Level1`), with the game's default goal. A generated one draws a save state, an opening
of the environment's own actions played from it, and a goal, so the planner starts somewhere
in the level. The emulator is deterministic given the save state and the inputs (checked on
every reset by the tests), so the same seed always gives the same instance, and the instance
is plain data that `set_instance` replays exactly.

## What a state is

`RetroState` carries a save state (compressed), the integration's variables, and literals
built from them: `var(name, value)` for each, and `ram(hash)`, a hash of the console's whole
RAM. Two states with the same RAM are the same state: an integration's variables are a few
counters (Airstriker names a score, a lives count and a game-over flag, and nothing about
where the ship is), so on their own they would fold every position with the same score into
one. `identity="variables"` does exactly that, for a game whose variables do say enough,
and it is the coarser, faster search.

## One emulator per process

Stable-Retro allows one live emulator per process. Environments here share it: constructing
a second `RetroEnv` is fine, and whichever one is used next takes the emulator over, the
other reopening it on its next call. Save states survive the handover, so a state expanded
under one environment can be expanded under another.
"""
import hashlib
import zlib

from planiverse.environments.base import Environment
from planiverse.environments.generation import bounded_search, draw_until, rng

#: The game Stable-Retro ships a freely redistributable ROM for, and so the default.
DEFAULT_GAME = "Airstriker-Genesis-v0"

#: Frames a press lasts. Four is the frame skip Stable-Retro's own baselines use.
DEFAULT_HOLD = 4

#: Per game: the button combinations worth offering a planner, and the default goal. An
#: integration's discrete action space runs to every combination its buttons allow (126 on
#: the Genesis), most of them doing nothing a planner needs.
DEFAULT_ACTIONS = {
    "Airstriker-Genesis-v0": ("", "LEFT", "RIGHT", "B", "LEFT+B", "RIGHT+B"),
}
DEFAULT_GOALS = {
    # Airstriker scores nothing for long stretches and loses a life every 300 frames or so
    # to whatever it is not dodging, so the problem is staying alive.
    "Airstriker-Genesis-v0": {"survive": 100},
}


def _retro():
    try:
        import stable_retro
    except ImportError:                       # the package's old import name
        import retro as stable_retro
    return stable_retro


#: The environment holding the process's one emulator, if any.
_LIVE = None


def _holds(spec, variables, start, done, steps, terminal=False):
    """Does a goal or dead-end `spec` hold?

    A spec is a dict with one of: `variable` with `at_least`, `at_most`, `equals`, or
    `delta` (relative to the start of the instance), or, for a dead end, `drop` (a lost
    life); `survive`, a number of actions taken without a dead end; or, for a dead end,
    `done`, the integration's own done condition.
    """
    if not spec:
        return False
    if spec.get("done") and done:
        return True
    if "survive" in spec:
        return steps >= spec["survive"]
    if "variable" not in spec:
        return False
    value, base = variables.get(spec["variable"]), start.get(spec["variable"])
    if value is None:
        return False
    if terminal and spec.get("drop"):
        return base is not None and value < base
    if "delta" in spec:
        return base is not None and value >= base + spec["delta"]
    if "at_least" in spec:
        return value >= spec["at_least"]
    if "at_most" in spec:
        return value <= spec["at_most"]
    if "equals" in spec:
        return value == spec["equals"]
    return False


class RetroAction:
    """A button combination, spelled the way Stable-Retro names it: `LEFT+B`, or `` for none."""

    def __init__(self, name, action_id):
        self.name = name
        self.action_id = action_id

    def cost(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, RetroAction) and self.action_id == other.action_id

    def __hash__(self):
        return hash(self.action_id)

    def __lt__(self, other):
        return self.action_id < other.action_id

    def __str__(self):
        return self.name or "nop"

    __repr__ = __str__


class RetroState:
    """A save state, and what the integration's variables say it is."""

    def __init__(self, snapshot, variables, ram_hash, steps, goal, terminal, progress,
                 identity="ram", count_steps=False, depth=0):
        self.snapshot = snapshot
        self.variables = dict(variables)
        self.ram_hash = ram_hash
        self.steps = steps
        self.goal = goal
        self.terminal = terminal
        self.progress = progress
        self.depth = depth
        self.key = (tuple(sorted(self.variables.items())),
                    ram_hash if identity == "ram" else None,
                    steps if count_steps else None)
        literals = [f"var({name}, {value})" for name, value in sorted(self.variables.items())]
        if identity == "ram":
            literals.append(f"ram({ram_hash})")
        if count_steps:
            literals.append(f"steps({steps})")
        if goal:
            literals.append("goal-reached")
        if terminal:
            literals.append("terminal-state")
        self.literals = frozenset(literals)

    def __eq__(self, other):
        return isinstance(other, RetroState) and self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        status = "goal" if self.goal else "dead end" if self.terminal else f"step {self.steps}"
        return ", ".join(f"{name}={value}" for name, value in sorted(self.variables.items())) \
            + f"  [{status}]"

    def __repr__(self):
        return f"<RetroState(steps={self.steps}, variables={self.variables}, goal={self.goal})>"


class RetroEnv(Environment):
    """A Stable-Retro game as a planning problem. Needs `stable-retro`; the default game
    ships with it."""

    def __init__(self, game=DEFAULT_GAME, actions=None, hold=DEFAULT_HOLD, goal=None,
                 terminal=None, identity="ram", inttype=None):
        """`game` is an integration name. `actions` are button combinations as Stable-Retro
        spells them (`"LEFT+B"`, `""` for none), defaulting to `DEFAULT_ACTIONS` for the game
        or every combination; `hold` is frames per press. `goal` and `terminal` are specs
        (see `_holds`); the default dead end is the integration's own done condition and a
        lost life where the integration counts lives. `identity` is `"ram"` (two states are
        the same when the console's RAM is) or `"variables"` (when the integration's
        variables are)."""
        if identity not in ("ram", "variables"):
            raise ValueError(f"identity must be 'ram' or 'variables', not {identity!r}")
        super().__init__("retro")
        self.game = game
        self.hold = hold
        self.identity = identity
        self.inttype = inttype
        self._actions_override = tuple(actions) if actions is not None else None
        self._goal_override = goal
        self._terminal_override = terminal
        self._env = None
        self._names = ()                      # every discrete action's button combination
        self._ids = {}                        # combination name -> discrete action id
        self._states = None                   # the integration's save states, sorted

        self.index = 0
        #: The instance `reset` builds: `{"state": name, "warmup": [...], "goal": ...}`.
        self.instance = None
        self.witness = None
        self.witness_expansions = None
        self.state = None
        self.state_history = []
        self._start_variables = {}
        self._initial = None

    # ------------------------------------------------------------------ the emulator

    def __env__(self):
        """The Stable-Retro environment, opened on first use and taken over from whichever
        `RetroEnv` held the process's one emulator before."""
        global _LIVE
        if self._env is None:
            if _LIVE is not None and _LIVE is not self:
                _LIVE.close()
            retro = _retro()
            kwargs = {"use_restricted_actions": retro.Actions.DISCRETE, "render_mode": None}
            if self.inttype is not None:
                kwargs["inttype"] = self.inttype
            self._env = retro.make(self.game, **kwargs)
            _LIVE = self
            self._env.reset(seed=0)
            if not self._names:
                self._names = tuple("+".join(self._env.get_action_meaning(a))
                                    for a in range(self._env.action_space.n))
                self._ids = {name: a for a, name in enumerate(self._names)}
        return self._env

    def states(self):
        """The integration's save states, which are the bundled instances."""
        if self._states is None:
            retro = _retro()
            kwargs = {} if self.inttype is None else {"inttype": self.inttype}
            self._states = tuple(sorted(retro.data.list_states(self.game, **kwargs)))
        return self._states

    def close(self):
        """Release the emulator. Save states and the selected instance survive; the next
        call reopens it."""
        global _LIVE
        if self._env is not None:
            self._env.close()
            self._env = None
            if _LIVE is self:
                _LIVE = None

    # ------------------------------------------------------------------ instances

    @property
    def actions(self):
        if self._actions_override is not None:
            return self._actions_override
        if self.game in DEFAULT_ACTIONS:
            return DEFAULT_ACTIONS[self.game]
        self.__env__()
        return self._names

    def __default_goal__(self):
        if self._goal_override is not None:
            return self._goal_override
        if self.game in DEFAULT_GOALS:
            return DEFAULT_GOALS[self.game]
        variables = self.__env__().data.lookup_all()
        return {"variable": "score", "delta": 1} if "score" in variables else {"survive": 100}

    def __default_terminal__(self):
        if self._terminal_override is not None:
            return self._terminal_override
        variables = self.__env__().data.lookup_all()
        return {"done": True, "variable": "lives", "drop": True} if "lives" in variables \
            else {"done": True}

    def __instance__(self, state_name, warmup, goal=None):
        return {"state": state_name, "warmup": list(warmup),
                "goal": goal if goal is not None else self.__default_goal__(),
                "terminal": self.__default_terminal__()}

    def set_index(self, index):
        """Select the integration's `index`th save state, from its first frame."""
        states = self.states()
        if not 0 <= index < len(states):
            raise IndexError(
                f"Invalid index: {index}. {self.game} ships {len(states)} save states, so "
                f"the index must be 0-{len(states) - 1}.")
        self.set_instance(self.__instance__(states[index], []))
        self.index = index

    def set_instance(self, instance):
        """Select an instance: `{"state": save state name, "warmup": [actions], "goal":
        spec, "terminal": spec}`."""
        if instance.get("state") not in self.states():
            raise ValueError(f"{instance.get('state')!r} is not a save state of {self.game}: "
                             f"choose from {self.states()}")
        self.instance = {"state": instance["state"], "warmup": list(instance.get("warmup", [])),
                         "goal": instance.get("goal", self.__default_goal__()),
                         "terminal": instance.get("terminal", self.__default_terminal__())}
        self.index = None
        self.witness = self.witness_expansions = None
        self._initial = None

    def generate_instance(self, seed=None, warmup=(0, 40), goal=None, solvable=False,
                          search_limit=2000, attempts=20):
        """Draw a fresh instance, select it, and return it as a dict.

        One of the integration's save states at random, and an opening of `warmup` (a range)
        actions from the vocabulary played from it. A draw that is already won or lost after
        its opening is redrawn. With `solvable` the draw is also searched, best-first under
        the goal's own measure, for up to `search_limit` expansions; off by default, since
        an emulator expansion costs milliseconds and a survival goal is deep. The emulator
        is deterministic given the save state and the inputs, so the same seed always gives
        the same instance.
        """
        random_, seed = rng(seed)
        found = {}

        def draw(attempt):
            opening = [random_.choice(self.actions) for _ in range(random_.randint(*warmup))]
            instance = self.__instance__(random_.choice(self.states()), opening, goal)
            instance["seed"] = seed
            return instance

        def accept(instance):
            self.set_instance(instance)
            state, _ = self.reset()
            if self.is_goal(state) or self.is_terminal(state):
                return False
            if solvable:
                outcome = bounded_search(self, search_limit, progress=lambda s: s.progress)
                if outcome.plan is None:
                    return False
                found.update(plan=outcome.plan, expansions=outcome.expansions)
            return True

        instance = draw_until(draw, accept, attempts, f"{self.game} instance")
        self.set_instance(instance)
        if solvable:
            self.witness, self.witness_expansions = found["plan"], found["expansions"]
        return instance

    # ------------------------------------------------------------------ the contract

    def reset(self):
        if self.instance is None:
            self.set_index(0)
        env = self.__env__()
        if self._initial is None:
            env.load_state(self.instance["state"], **({} if self.inttype is None
                                                       else {"inttype": self.inttype}))
            env.reset(seed=0)
            for name in self.instance["warmup"]:
                self.__press__(self.__action__(name))
            # Relative goals (`delta`, `drop`) are measured from the instance's own initial
            # state, which is after its opening, not from the save state it was played from.
            self._start_variables = {name: int(value)
                                     for name, value in env.data.lookup_all().items()}
            self._initial = self.__snapshot__()
        else:
            self.__load__(self._initial)
        self.state = self.__observe__(0, 0)
        self.state_history = [self.state]
        return self.state, {"game": self.game,
                            "state": self.instance["state"],
                            "index": self.index,
                            "generated": self.index is None,
                            "warmup": len(self.instance["warmup"]),
                            "goal": self.instance["goal"],
                            "actions": len(self.actions),
                            "variables": dict(self.state.variables)}

    def is_goal(self, state):
        return state.goal

    def is_terminal(self, state):
        return state.terminal and not state.goal

    def successors(self, state):
        if state.goal or state.terminal:
            return []
        successors = []
        for name in self.actions:
            action = self.__action__(name)
            successor = self.__advance__(state, action)
            if successor == state:
                continue
            successors.append((action, successor))
        return successors

    def __advance__(self, state, action):
        if state.goal or state.terminal:
            return state
        if not isinstance(action, RetroAction):
            action = self.__action__(str(action))
        self.__load__(state.snapshot)
        self.__press__(action)
        return self.__observe__(state.steps + 1, state.depth + 1)

    def simulate(self, plan):
        state, _ = self.reset()
        trace = [state]
        for action in plan:
            trace.append(self.__advance__(trace[-1], action))
        return trace

    def step(self, action):
        if self.state is None:
            raise ValueError("Game not initialized. Call reset() first.")
        before = self.state.progress
        self.state = self.__advance__(self.state, action)
        self.state_history.append(self.state)
        return self.state, before - self.state.progress

    def get_actions(self):
        return [self.__action__(name) for name in self.actions]

    def render(self, target=None, **kwargs):
        """The console's own frames for the positions `step` played through, as arrays, or
        written to `target` through `render_trace` when one is given."""
        frames = []
        for state in self.state_history:
            self.__load__(state.snapshot)
            frames.append(self.__env__().get_screen())
        if target is None:
            return frames
        from planiverse.rendering import render_trace

        return render_trace(self.state_history, target, **kwargs)

    # ------------------------------------------------------------------ emulation

    def __action__(self, name):
        self.__env__()
        name = "" if name == "nop" else name
        if name not in self._ids:
            wanted = frozenset(name.split("+")) if name else frozenset()
            for candidate, action_id in self._ids.items():
                if frozenset(candidate.split("+")) - {""} == wanted:
                    self._ids[name] = action_id
                    break
            else:
                raise ValueError(f"{name!r} is not a button combination of {self.game}; "
                                 f"the buttons are {self.__env__().buttons}")
        return RetroAction(name, self._ids[name])

    def __press__(self, action):
        env = self.__env__()
        for _ in range(self.hold):
            env.step(action.action_id)

    def __snapshot__(self):
        return zlib.compress(self.__env__().em.get_state(), 1)

    def __load__(self, snapshot):
        env = self.__env__()
        env.em.set_state(zlib.decompress(snapshot))
        env.data.update_ram()                 # `step` does this; a bare `set_state` does not

    def __observe__(self, steps, depth):
        env = self.__env__()
        variables = {name: int(value) for name, value in env.data.lookup_all().items()}
        ram_hash = hashlib.blake2b(env.get_ram().tobytes(), digest_size=8).hexdigest()
        done = bool(env.data.is_done())
        goal_spec, terminal_spec = self.instance["goal"], self.instance["terminal"]
        goal = _holds(goal_spec, variables, self._start_variables, done, steps)
        terminal = _holds(terminal_spec, variables, self._start_variables, done, steps,
                          terminal=True)
        if "variable" in goal_spec and goal_spec["variable"] in variables:
            target = goal_spec.get("at_least",
                                   self._start_variables.get(goal_spec["variable"], 0)
                                   + goal_spec.get("delta", 0))
            progress = target - variables[goal_spec["variable"]]
        else:
            progress = goal_spec.get("survive", 0) - steps
        return RetroState(self.__snapshot__(), variables, ram_hash, steps, goal, terminal,
                          progress, identity=self.identity,
                          count_steps="survive" in goal_spec, depth=depth)
