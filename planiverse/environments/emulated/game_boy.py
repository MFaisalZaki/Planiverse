"""Any Game Boy cartridge as a planning environment, through PyBoy and its game wrappers.

One environment for every cartridge. What a planner needs from an emulator is the same
whatever is running on it: a way back to a state to expand it again, which a save state
gives; something to read a state *as*, which PyBoy's game wrappers give (`game_area()`, a
matrix of tile identifiers, and whatever the wrapper measures: blocks left, hearts left,
level progress, score); and a goal, which the wrapper answers too (`stage_cleared()`,
`level_solved()`, `room_solved()`) or which is a threshold on one of those measurements.
The knowledge of a particular game, where its counters live and how its menus are walked,
is PyBoy's business and stays in its wrappers; nothing here reads a game's memory map.

    env = GameBoyEnv("puzznic.gb")          # or PLANIVERSE_GB_ROM=... and GameBoyEnv()
    env.set_index(7)                          # the wrapper's eighth stage, level or room
    state, info = env.reset()
    for action, successor in env.successors(state):
        print(action, successor.fields)

A cartridge PyBoy has no wrapper for still works: the state is the background map and
whatever bytes of memory `watch=` names, and the goal is whatever `goal=` says about them.
That is what the test suite uses, on a cartridge it assembles itself.

## What a wrapper gives the environment

`PROFILES` says, per wrapper class, how to reach instance `i` (a `start_game` argument: a
stage, a level, a room, a world and level, a timer seed), what to read as the state's
fields, what the goal and the dead end are, which buttons make sense, and how long to hold
one. It covers the wrappers PyBoy ships and the ones added to it for Puzznic, Flipull,
Amazing Tater and Adventures of Lolo; an unlisted wrapper gets the generic profile, which
reads every number the wrapper exposes and asks for a goal to be named.

## Instances, and generating them

A bundled instance is one of the wrapper's stages, levels or rooms, started with the
timer's DIV register at zero. A generated one draws a stage, a timer seed and an opening:
a random run of the environment's own actions played from the stage's first frame, so the
planner starts somewhere in the level rather than at its door. The emulator is
deterministic given the cartridge, the timer seed and the inputs, so the same seed always
gives the same instance, and the instance is plain data (`{"start": ..., "timer_div": ...,
"warmup": [...], "goal": ...}`) that `set_instance` replays exactly.

## What a state is

`GBState` carries a save state (compressed; PyBoy's are 200 KB of mostly zeros), the
`game_area()` tiles, the wrapper's fields, and literals built from both: `tile(row, col,
id)` for every cell and `field(name, value)` for every measurement. Two states with the same
tiles and fields are the same state, whatever their save states say about timers, so search
closes on positions rather than on frame counts.
"""
import io
import os
import zlib

from planiverse.environments.base import Environment
from planiverse.environments.generation import bounded_search, draw_until, rng

#: Where the cartridge path comes from when the constructor is not given one. The file is
#: copyrighted for every commercial title and cannot ship, so it can only come from the user.
ROM_VARIABLE = "PLANIVERSE_GB_ROM"

#: PyBoy's button names. An action is one of these, or several joined with `+`, or `nop`.
BUTTONS = ("a", "b", "start", "select", "up", "down", "left", "right")
DIRECTIONS = ("left", "right", "up", "down")

#: Frames the generic start runs before the first playable frame: PyBoy's boot ROM takes
#: about eighty frames to hand over to the cartridge.
BOOT_FRAMES = 120


class Profile:
    """What the environment knows about one game wrapper.

    `select(index)` gives the `start_game` keyword arguments that reach bundled instance
    `index`, of which there are `instances`. `fields` are the wrapper attributes (or methods)
    read into every state. `goal` and `terminal` are specs (see `_holds`). `actions` and
    `hold` are the default action vocabulary and how many frames a press lasts. `progress`
    maps a state's fields to a number that falls as the goal nears.
    """

    def __init__(self, instances, select, fields, goal, terminal, actions, hold,
                 progress=None, random_timer=True):
        self.instances = instances
        self.select = select
        self.fields = tuple(fields)
        self.goal = goal
        self.terminal = terminal
        self.actions = tuple(actions)
        self.hold = hold
        self.progress = progress
        self.random_timer = random_timer


def _delta(name, amount):
    return {"attribute": name, "delta": amount}


PROFILES = {
    "GameWrapperPuzznic": Profile(
        128, lambda i: {"stage": i},
        ("stage", "blocks_remaining", "blocks_total", "cursor"),
        {"method": "stage_cleared"}, {"method": "game_over"},
        DIRECTIONS + ("a+left", "a+right"), 6,
        progress=lambda f, s: f.get("blocks_remaining", 0)),
    "GameWrapperFlipull": Profile(
        32, lambda i: {"stage": i + 1},          # the wrapper counts stages from one
        ("stage", "blocks_remaining", "clear_target", "time_left", "player_row", "held_block"),
        {"method": "stage_cleared"}, {"method": "game_over"},
        ("up", "down", "throw"), 5,
        progress=lambda f, s: max(0, f.get("blocks_remaining", 0) - f.get("clear_target", 0))),
    "GameWrapperAmazingTater": Profile(
        105, lambda i: {"level": i},
        ("level", "taters_left", "taters_home", "active_tater"),
        {"method": "level_solved"}, {"method": "game_over"},
        DIRECTIONS + ("select",), 5,
        progress=lambda f, s: f.get("taters_left", 0)),
    "GameWrapperAdventuresOfLolo": Profile(
        163, lambda i: {"room": i},
        ("room", "hearts_left", "magic_shots", "lolo"),
        {"method": "room_solved"}, {"method": "game_over"},
        DIRECTIONS + ("a",), 20,
        progress=lambda f, s: f.get("hearts_left", 0) + 1),
    "GameWrapperSuperMarioLand": Profile(
        12, lambda i: {"world_level": (i // 3 + 1, i % 3 + 1)},
        ("level_progress", "lives_left", "coins", "time_left", "world"),
        _delta("level_progress", 250), {"method": "game_over", "attribute": "lives_left"},
        ("right", "a+right", "left", "a+left", "a", "nop"), 8,
        progress=lambda f, s: -f.get("level_progress", 0)),
    "GameWrapperSuperMarioBrosDeluxe": Profile(
        1, lambda i: {},
        ("level_progress", "lives_left", "coins", "time_left", "world"),
        _delta("level_progress", 250), {"method": "game_over", "attribute": "lives_left"},
        ("right", "a+right", "left", "a+left", "a", "nop"), 8,
        progress=lambda f, s: -f.get("level_progress", 0)),
    "GameWrapperTetris": Profile(
        16, lambda i: {"timer_div": 16 * i},      # the timer seed is what tells games apart
        ("score", "level", "lines", "next_tetromino"),
        _delta("lines", 1), {"method": "game_over"},
        ("left", "right", "down", "a", "b", "nop"), 4,
        progress=lambda f, s: -f.get("lines", 0), random_timer=False),
    "GameWrapperKirbyDreamLand": Profile(
        1, lambda i: {},
        ("score", "health", "lives_left"),
        _delta("score", 100), {"method": "game_over", "attribute": "lives_left"},
        ("right", "a+right", "left", "a+left", "b", "a", "nop"), 8,
        progress=lambda f, s: -f.get("score", 0)),
}

#: For a cartridge PyBoy has no wrapper for, or one this module has no profile for: every
#: number the wrapper exposes is a field, the buttons are all of them, and there is no goal
#: until one is named, other than lasting a while.
GENERIC = Profile(1, lambda i: {"ticks": BOOT_FRAMES}, (), {"survive": 20},
                  {"method": "game_over"}, BUTTONS + ("nop",), 8, progress=None)


def _holds(spec, wrapper, fields, start, memory, steps, terminal=False):
    """Does a goal or dead-end `spec` hold?

    A spec is a dict with one of: `method`, a wrapper method returning a bool
    (`stage_cleared`, `game_over`); `attribute` with `at_least`, `at_most`, `equals`, or
    `delta` (relative to the start of the instance), or, for a dead end, `drop` (the value
    fell below its start, a lost life); `memory` with the same comparisons on a byte of
    memory; or `survive`, a number of actions taken without a dead end. A spec naming
    both a `method` and an `attribute` holds when either does.
    """
    if not spec:
        return False
    if "method" in spec:
        method = getattr(wrapper, spec["method"], None)
        try:
            if method is not None and method():
                return True
        except NotImplementedError:
            pass
        if "attribute" not in spec and "memory" not in spec:
            return False
    if "survive" in spec:
        return steps >= spec["survive"]
    if "attribute" in spec:
        name = spec["attribute"]
        value, base = fields.get(name), start.get(name)
    else:
        value, base = int(memory[spec["memory"]]), start.get(f"@{spec['memory']:#x}")
    if value is None:
        return False
    if terminal and spec.get("drop", "attribute" in spec and "method" in spec):
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


class GBAction:
    """Buttons held for the environment's `hold` frames: `right`, `a+right`, `nop`."""

    def __init__(self, name):
        self.name = name
        self.buttons = () if name == "nop" else tuple(name.split("+"))

    def cost(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, GBAction) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    __repr__ = __str__


class GBState:
    """A save state, and what the wrapper and the background map say it is."""

    def __init__(self, snapshot, tiles, fields, steps, goal, terminal, progress,
                 count_steps=False, depth=0):
        self.snapshot = snapshot              # zlib-compressed PyBoy save state
        self.tiles = tiles                    # game_area(), as a tuple of tuples
        self.fields = dict(fields)
        self.steps = steps
        self.goal = goal
        self.terminal = terminal
        self.progress = progress
        self.depth = depth
        # Only a survival goal makes the step count part of what a state is.
        self.key = (tiles, tuple(sorted(self.fields.items())), steps if count_steps else None)

        literals = [f"tile({row}, {col}, {tile})"
                    for row, cells in enumerate(tiles) for col, tile in enumerate(cells)]
        literals += [f"field({name}, {value})" for name, value in sorted(self.fields.items())]
        if count_steps:
            literals.append(f"steps({steps})")
        if goal:
            literals.append("goal-reached")
        if terminal:
            literals.append("terminal-state")
        self.literals = frozenset(literals)

    def __eq__(self, other):
        return isinstance(other, GBState) and self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def __lt__(self, other):
        return self.depth < other.depth

    def __str__(self):
        rows = [" ".join(f"{tile:3d}" for tile in cells) for cells in self.tiles]
        fields = ", ".join(f"{name}={value}" for name, value in sorted(self.fields.items()))
        status = "goal" if self.goal else "dead end" if self.terminal else f"step {self.steps}"
        return "\n".join(rows + [f"{fields}  [{status}]" if fields else f"[{status}]"])

    def __repr__(self):
        return f"<GBState(steps={self.steps}, fields={self.fields}, goal={self.goal})>"


class GameBoyEnv(Environment):
    """A cartridge under PyBoy as a planning problem. Needs `pyboy`, and a ROM you supply."""

    def __init__(self, rom=None, actions=None, hold=None, settle=0, watch=None, goal=None,
                 terminal=None, profile=None):
        """`rom` is the cartridge, or `PLANIVERSE_GB_ROM` is read. `actions` overrides the
        profile's button vocabulary and `hold` how many frames a press lasts; `settle` is
        frames to run after a press on a wrapper with no `settle()` of its own. `watch` names
        bytes of memory to read into every state as fields, `{"counter": 0xC000}`, and `goal`
        and `terminal` are specs (see `_holds`) that override the profile's."""
        super().__init__("game_boy")
        self.rom = rom or os.environ.get(ROM_VARIABLE)
        if not self.rom or not os.path.isfile(self.rom):
            raise FileNotFoundError(
                "a Game Boy cartridge is needed and cannot ship with this repository: pass "
                f"rom= or point {ROM_VARIABLE} at one")
        self._pyboy = None
        self._wrapper = None
        self._booted = None                   # the start the live emulator was booted with
        self.profile = profile
        self.watch = dict(watch or {})
        self.settle = settle
        self._actions_override = tuple(actions) if actions else None
        self._hold_override = hold
        self._goal_override = goal
        self._terminal_override = terminal

        self.index = 0
        #: The instance `reset` builds; see the module docstring for its shape.
        self.instance = None
        #: The plan `generate_instance` accepted the current instance on, when it checked
        #: one, and what the search spent finding it.
        self.witness = None
        self.witness_expansions = None
        self.state = None
        self.state_history = []
        self._start_fields = {}
        self._initial = None                  # (instance key, compressed snapshot)
        self.__boot__({"ticks": BOOT_FRAMES}, 0)   # to learn the wrapper, and so the profile
        self.set_index(0)

    # ------------------------------------------------------------------ the emulator

    def __boot__(self, start, timer_div):
        """A fresh emulator, booted to instance `start`.

        Fresh because a wrapper's `start_game` may be called once per emulator, and it is
        what reaches a stage or room; a different one needs a new boot. The boot itself
        costs a second or two, and `reset` only pays it when the start changes.
        """
        from pyboy import PyBoy

        self.__stop__()
        pyboy = PyBoy(self.rom, window="null", sound_emulated=False, log_level="ERROR")
        wrapper = pyboy.game_wrapper
        if self.profile is None:
            self.profile = PROFILES.get(type(wrapper).__name__, GENERIC)
        start = dict(start)
        ticks = start.pop("ticks", None)
        if type(wrapper).__name__ == "PyBoyGameWrapper" or ticks is not None:
            pyboy.tick(BOOT_FRAMES if ticks is None else ticks, False)
        wrapper.start_game(timer_div=start.pop("timer_div", timer_div), **start)
        self._pyboy, self._wrapper = pyboy, wrapper
        self._booted = (self.__frozen__(start), timer_div)

    def __stop__(self):
        if self._pyboy is not None:
            self._pyboy.stop(save=False)
            self._pyboy = None
            self._wrapper = None
            self._booted = None

    @staticmethod
    def __frozen__(start):
        return tuple(sorted((key, tuple(value) if isinstance(value, list) else value)
                            for key, value in start.items()))

    def close(self):
        self.__stop__()

    # ------------------------------------------------------------------ instances

    @property
    def actions(self):
        return self._actions_override or self.profile.actions

    @property
    def hold(self):
        if self._hold_override is not None:
            return self._hold_override
        return getattr(self._wrapper, "press_ticks", None) or self.profile.hold

    def __instance__(self, index, start, timer_div, warmup):
        return {"index": index, "start": dict(start), "timer_div": int(timer_div),
                "warmup": list(warmup),
                "goal": self._goal_override or self.profile.goal,
                "terminal": self._terminal_override or self.profile.terminal}

    def set_index(self, index):
        """Select the wrapper's `index`th stage, level or room, from its first frame."""
        count = self.profile.instances
        if not 0 <= index < count:
            raise IndexError(
                f"Invalid index: {index}. This cartridge's wrapper offers {count} "
                f"instances, so the index must be 0-{count - 1}.")
        start = self.profile.select(index)
        timer_div = start.pop("timer_div", 0) if "timer_div" in start else 0
        self.set_instance(self.__instance__(index, start, timer_div, []))
        self.index = index

    def set_instance(self, instance):
        """Select an instance: `{"start": start_game kwargs, "timer_div": n, "warmup":
        [actions], "goal": spec, "terminal": spec}`."""
        for name in instance.get("warmup", []):
            GBAction(name)
        self.instance = {"index": instance.get("index"), "start": dict(instance.get("start", {})),
                         "timer_div": int(instance.get("timer_div", 0)),
                         "warmup": list(instance.get("warmup", [])),
                         "goal": instance.get("goal", self._goal_override or self.profile.goal),
                         "terminal": instance.get("terminal",
                                                  self._terminal_override or self.profile.terminal)}
        self.index = None
        self.witness = self.witness_expansions = None
        self._initial = None

    def generate_instance(self, seed=None, warmup=(0, 30), index=None, goal=None,
                          solvable=False, search_limit=2000, attempts=20):
        """Draw a fresh instance, select it, and return it as a dict.

        A stage, level or room (`index`, or one of the wrapper's at random), a timer seed
        for the games whose randomness reads the DIV register, and an opening of `warmup`
        (a range) actions drawn from the action vocabulary and played from the stage's
        first frame. A draw that is already won or lost after its opening is redrawn. With
        `solvable`, the draw is also searched, best-first under the profile's progress
        measure, for up to `search_limit` expansions, which on an emulator is minutes rather
        than seconds, so it is off by default. The emulator is deterministic given the
        cartridge, the timer seed and the inputs, so the same seed always gives the same
        instance.
        """
        random_, seed = rng(seed)
        count = self.profile.instances
        found = {}

        def draw(attempt):
            chosen = index if index is not None else random_.randrange(count)
            start = self.profile.select(chosen)
            timer_div = start.pop("timer_div") if "timer_div" in start else (
                random_.randrange(256) if self.profile.random_timer else 0)
            opening = [random_.choice(self.actions)
                       for _ in range(random_.randint(*warmup))]
            instance = self.__instance__(chosen, start, timer_div, opening)
            if goal is not None:
                instance["goal"] = goal
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

        instance = draw_until(draw, accept, attempts, "Game Boy instance")
        self.set_instance(instance)
        if solvable:
            self.witness, self.witness_expansions = found["plan"], found["expansions"]
        return instance

    # ------------------------------------------------------------------ the contract

    def reset(self):
        if self.instance is None:
            self.set_index(0)
        instance = self.instance
        key = (self.__frozen__(instance["start"]), instance["timer_div"])
        if self._initial is not None and self._initial[0] == key and self._pyboy is not None:
            self.__load__(self._initial[1])
        else:
            if self._booted != key:
                self.__boot__(instance["start"], instance["timer_div"])
            else:
                self._wrapper.reset_game(timer_div=instance["timer_div"])
            for name in instance["warmup"]:
                self.__press__(GBAction(name))
            # Relative goals (`delta`, `drop`) are measured from the instance's own initial
            # state, which is after its opening, not from the stage's first frame.
            self._start_fields = self.__start__()
            self._initial = (key, self.__snapshot__())
        self.state = self.__observe__(0, 0)
        self.state_history = [self.state]
        return self.state, {"rom": os.path.basename(self.rom),
                            "wrapper": type(self._wrapper).__name__,
                            "index": instance["index"],
                            "generated": self.index is None,
                            "start": dict(instance["start"]),
                            "timer_div": instance["timer_div"],
                            "warmup": len(instance["warmup"]),
                            "goal": instance["goal"],
                            "fields": dict(self.state.fields)}

    def is_goal(self, state):
        return state.goal

    def is_terminal(self, state):
        return state.terminal and not state.goal

    def successors(self, state):
        if state.goal or state.terminal:
            return []
        successors = []
        for name in self.actions:
            action = GBAction(name)
            successor = self.__advance__(state, action)
            if successor == state:
                continue
            successors.append((action, successor))
        return successors

    def __advance__(self, state, action):
        if state.goal or state.terminal:
            return state
        if not isinstance(action, GBAction):
            action = GBAction(str(action))
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
        return self.state, (before - self.state.progress) if before is not None else 0

    def get_actions(self):
        return [GBAction(name) for name in self.actions]

    def render(self, target=None, **kwargs):
        """The console's own frames for the positions `step` played through, as PIL images,
        or written to `target` through `render_trace` when one is given."""
        frames = []
        for state in self.state_history:
            self.__load__(state.snapshot)
            self._pyboy.tick(1, True)
            frames.append(self._pyboy.screen.image.copy())
        if target is None:
            return frames
        from planiverse.rendering import render_trace

        return render_trace(self.state_history, target, **kwargs)

    # ------------------------------------------------------------------ emulation

    def __press__(self, action):
        """Hold the action's buttons for `hold` frames, then let the game settle."""
        pyboy, wrapper = self._pyboy, self._wrapper
        buttons = []
        for button in action.buttons:
            if button == "throw":
                button = getattr(wrapper, "throw_button", None) or "a"
            elif button == "switch":
                button = "select"
            buttons.append(button)
        for button in buttons:
            pyboy.button_press(button)
        pyboy.tick(self.hold, False)
        for button in buttons:
            pyboy.button_release(button)
        settle = getattr(wrapper, "settle", None)
        if callable(settle):
            settle()
        elif self.settle:
            pyboy.tick(self.settle, False)
        else:
            pyboy.tick(1, False)

    def __snapshot__(self):
        buffer = io.BytesIO()
        self._pyboy.save_state(buffer)
        return zlib.compress(buffer.getvalue(), 1)

    def __load__(self, snapshot):
        self._pyboy.load_state(io.BytesIO(zlib.decompress(snapshot)))
        post_tick = getattr(self._wrapper, "post_tick", None)
        if callable(post_tick):
            post_tick()

    def __fields__(self):
        """Every measurement the wrapper (and `watch`) offers, as plain numbers."""
        wrapper, fields = self._wrapper, {}
        names = self.profile.fields
        if not names:
            names = tuple(name for name in dir(wrapper)
                          if not name.startswith("_") and name not in _NOT_FIELDS
                          and isinstance(getattr(wrapper, name, None), (int, float)))
        for name in names:
            try:
                value = getattr(wrapper, name)
                if callable(value):
                    value = value()
            except Exception:
                continue
            if isinstance(value, (tuple, list)):
                for position, item in enumerate(value):
                    if isinstance(item, (int, float)):
                        fields[f"{name}.{position}"] = _plain(item)
            elif isinstance(value, (int, float)):
                fields[name] = _plain(value)
        for name, address in self.watch.items():
            fields[name] = int(self._pyboy.memory[address])
        return fields

    def __start__(self):
        """The fields at the instance's initial state, plus the bytes of memory its goal and
        dead end name, keyed `@0xc000`: what `delta` and `drop` are measured from."""
        start = self.__fields__()
        for spec in (self.instance["goal"], self.instance["terminal"]):
            if spec and "memory" in spec:
                start[f"@{spec['memory']:#x}"] = int(self._pyboy.memory[spec["memory"]])
        return start

    def __observe__(self, steps, depth):
        fields = self.__fields__()
        tiles = tuple(tuple(int(tile) for tile in row) for row in self._wrapper.game_area())
        start, memory = self._start_fields, self._pyboy.memory
        goal = _holds(self.instance["goal"], self._wrapper, fields, start, memory, steps)
        terminal = _holds(self.instance["terminal"], self._wrapper, fields, start, memory,
                          steps, terminal=True)
        progress = (self.profile.progress(fields, start) if self.profile.progress
                    else -steps)
        return GBState(self.__snapshot__(), tiles, fields, steps, goal, terminal, progress,
                       count_steps="survive" in (self.instance["goal"] or {}), depth=depth)


#: Wrapper attributes that are numbers but not measurements of the game.
_NOT_FIELDS = frozenset({"sprite_offset", "tilemap_use_background", "game_has_started",
                         "game_area_follow_scxy", "shape", "argv"})


def _plain(value):
    return int(value) if isinstance(value, bool) or float(value).is_integer() else round(value, 3)
