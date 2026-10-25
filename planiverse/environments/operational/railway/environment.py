"""Train dispatching on Flatland: a few cities joined by single-track lines, a handful of trains
waiting in their stations, and the decision of when each one leaves; get every train to its
target before the horizon with the total of their arrival steps under the target.

Flatland (https://github.com/flatland-association/flatland-rl, MIT; Mohanty et al. 2020,
"Flatland-RL: multi-agent reinforcement learning on trains", https://arxiv.org/abs/2012.05893)
is the railway scheduling simulator begun by SBB with AIcrowd and now kept by the Flatland
Association. It is a dependency installed from PyPI; nothing of it is included here. The
network is drawn by its own `sparse_rail_generator` and the trains' stations and targets by its
`sparse_line_generator`, both from the instance's seed, with malfunctions off.

The decisions are the dispatcher's: let one waiting train depart, or wait while the trains
already out run on. A train that has departed follows its shortest route to its target, taking
at each switch the branch Flatland's distance map says is nearest. Everything else is
Flatland's: one train to a cell, a train that finds the next cell taken stops until it is free,
and two trains that meet head-on on a single track can never move again.

## Trajectory constraints as a goal

A deadlock (trains that can never move again) is `is_terminal`, and so is the horizon passing
with a train not arrived. The total of the trains' arrival steps is accumulated in the state as
train-steps (each step adds one for every train not yet arrived), and the goal is every train
arrived with that total under the target; a total over the target is a dead end, since it only
grows. The target is the least total any of the scripted dispatching rules achieves, so the
planner is asked to do at least as well as the best rule by choosing the departures.

## Determinism and state

Flatland with malfunctions off is deterministic, so a state is what the simulator holds after
the departures so far: the step, each train's state, cell and direction, which trains have been
let go, and the train-steps. Two dispatching histories that leave the same are one state. Each
state carries a pickled snapshot of Flatland's agents to expand from, which restores in a fraction
of a millisecond.
"""
import pickle
import warnings
from collections import namedtuple

from planiverse.environments.base import Environment
from planiverse.environments.generation import draw_until, rng

#: Steps a `wait` runs while a train is still to depart.
PERIOD = 4
#: What the generator draws when not told: the cities, the trains and the side of the grid.
CITIES = (2, 3, 4)
TRAINS = (2, 3, 4, 5)
SIZES = (25, 30, 35)
#: One track between cities and one pair of platforms in each: lines are single track.
RAILS_BETWEEN_CITIES, PLATFORM_PAIRS = 1, 1
#: The train-steps the literals band together.
BAND = 10


def _flatland():
    from flatland.core.env_observation_builder import DummyObservationBuilder
    from flatland.envs.line_generators import sparse_line_generator
    from flatland.envs.rail_env import RailEnv
    from flatland.envs.rail_env_action import RailEnvActions
    from flatland.envs.rail_generators import sparse_rail_generator
    from flatland.envs.step_utils.states import TrainState

    return {"RailEnv": RailEnv, "rail": sparse_rail_generator, "line": sparse_line_generator,
            "obs": DummyObservationBuilder, "actions": RailEnvActions, "states": TrainState}


def build(instance):
    """Flatland's `RailEnv` for an instance, reset, with every train ready to leave at step 0.

    The network and the trains are Flatland's from the seed; the departures are the plan's, so
    the timetable's earliest departures are cleared and its episode length replaced by the
    instance's horizon.
    """
    fl = _flatland()
    seed = instance["seed"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = fl["RailEnv"](
            width=instance["size"], height=instance["size"],
            rail_generator=fl["rail"](max_num_cities=instance["cities"], seed=seed, grid_mode=False,
                                      max_rails_between_cities=RAILS_BETWEEN_CITIES,
                                      max_rail_pairs_in_city=PLATFORM_PAIRS),
            line_generator=fl["line"](seed=seed), number_of_agents=instance["trains"],
            obs_builder_object=fl["obs"](), random_seed=seed)
        env.reset(random_seed=seed)
    for agent in env.agents:
        agent.earliest_departure = 0
    return env


class RailwayAction:
    """`depart(i)` lets train `i` leave its station; `wait` lets the trains out run on."""

    def __init__(self, train=None):
        self.train = train
        self.name = "wait" if train is None else f"depart({train})"

    @classmethod
    def parse(cls, text):
        text = str(text).strip()
        if text == "wait":
            return cls()
        if text.startswith("depart(") and text.endswith(")"):
            return cls(int(text[len("depart("):-1]))
        raise ValueError(f"unknown action: {text!r}")

    def cost(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, RailwayAction) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    __repr__ = __str__


WAIT = RailwayAction()

#: What a train's Flatland state is called here.
STATUS = {0: "waiting", 1: "waiting", 2: "waiting", 3: "running", 4: "stopped", 5: "stopped",
          6: "arrived"}


class RailwayState:
    """The network after the departures so far: the step, each train's status, cell and
    direction, the trains let go, the train-steps so far, the cells still to run and whether a
    deadlock has set in. Identity is the step, the trains and the train-steps; `snapshot` is
    Flatland's agents, kept to expand from, and `layout`, a `RailMap`, is the track `str`
    draws the trains on. Neither is compared, hashed or in the literals."""

    def readings(self):
        """The state's numbers by name, which its OpenSpiel game gives as the named pieces
        of its observation."""
        count = {status: sum(1 for t in self.trains if t[0] == status)
                 for status in ("waiting", "running", "stopped", "arrived")}
        return {**count, "cells to go": self.to_go, "train-steps": self.train_steps,
                "target": self.target}

    def __init__(self, step, trains, released, train_steps, to_go=0, deadlocked=(), snapshot=None,
                 target=0, horizon=0, depth=0, layout=None):
        self.step = step
        self.layout = layout
        self.trains = tuple(tuple(t) for t in trains)          # (status, cell or None, direction or None)
        self.released = tuple(sorted(released))
        self.train_steps, self.to_go = train_steps, to_go
        self.deadlocked = tuple(sorted(deadlocked))
        self.snapshot = snapshot
        self.target, self.horizon, self.depth = target, horizon, depth
        self.key = (self.step, self.trains, self.released, self.train_steps)
        literals = [f"step({step})", f"train_steps({train_steps // BAND * BAND})"]
        for i, (status, cell, _) in enumerate(self.trains):
            literals.append(f"train({i}, {status})")
            if cell is not None:
                literals.append(f"at({i}, {cell[0]}, {cell[1]})")
        literals += [f"released({i})" for i in self.released]
        literals += [f"deadlocked({i})" for i in self.deadlocked]
        self.literals = frozenset(literals)

    @property
    def arrived(self):
        return sum(1 for t in self.trains if t[0] == "arrived")

    @property
    def left(self):
        return len(self.trains) - self.arrived

    def __eq__(self, other):
        return isinstance(other, RailwayState) and self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def __lt__(self, other):
        return self.depth < other.depth

    def __repr__(self):
        return (f"<RailwayState(step={self.step}, arrived={self.arrived}, "
                f"train_steps={self.train_steps})>")

    def __str__(self):
        """Flatland's track from above in box-drawing lines, two columns to a cell: each
        train's target as ◎, a train out on the line as an arrow in its direction with its
        number beside it, a train still waiting as its number at its start; then the step, the
        trains arrived, running and waiting, and the train-steps against the target."""
        from planiverse.rendering.text import Canvas

        count = self.readings()
        status = (f"step {self.step}   arrived {count['arrived']}  running {count['running']}  "
                  f"waiting {count['waiting']}")
        if count["stopped"]:
            status += f"  stopped {count['stopped']}"
        status += f"   train-steps {self.train_steps} of {self.target}"
        if self.deadlocked:
            status += "   deadlocked"
        if self.layout is None:
            return status
        canvas = Canvas(2 * self.layout.width, self.layout.height)
        for row, line in enumerate(self.layout.track):
            canvas.write(0, row, line)
        for row, column in self.layout.targets:
            canvas.put(2 * column, row, "◎")
        waiting = {}
        for train, (status_, cell, direction) in enumerate(self.trains):
            if cell is None and status_ == "waiting":
                waiting.setdefault(self.layout.starts[train], []).append(str(train))
        for (row, column), trains in waiting.items():
            canvas.write(2 * column, row, "".join(trains)[:2])
        for train, (status_, cell, direction) in enumerate(self.trains):
            if cell is not None and status_ != "arrived":
                canvas.write(2 * cell[1], cell[0], HEADINGS[direction] + str(train)[-1])
        return f"{canvas}\n{status}"


class RailMap(namedtuple("RailMap", "width height track targets starts")):
    """What a picture of a state needs from Flatland's network and nothing a state is
    compared on: the track as lines of text, two columns to a cell, each train's target cell
    and each train's start cell. `RailMap.of(env)` reads it from Flatland's transition map."""

    __slots__ = ()

    @classmethod
    def of(cls, env):
        from planiverse.rendering.text import JOINS

        rail = env.rail
        lines = []
        for row in range(rail.height):
            line = []
            for column in range(rail.width):
                sides = _sides(rail.get_full_transitions(row, column))
                if not sides:
                    line.append("  ")
                    continue
                key = tuple(int(side in sides) for side in range(4))
                line.append(JOINS[key] + ("─" if 1 in sides else " "))
            lines.append("".join(line))
        targets = tuple(sorted({(int(cell[0]), int(cell[1]))
                                for agent in env.agents for cell, _ in agent.targets}))
        starts = tuple((int(agent.initial_configuration[0][0]), int(agent.initial_configuration[0][1]))
                       for agent in env.agents)
        return cls(rail.width, rail.height, tuple(lines), targets, starts)


#: A train's direction as Flatland numbers it (north, east, south, west), as an arrow.
HEADINGS = "▲▶▼◀"


def _sides(transitions):
    """The sides of a cell (0 north, 1 east, 2 south, 3 west) its track leaves by, from
    Flatland's sixteen-bit transitions: four bits of exits for each heading a train may have
    on entering, north first."""
    sides = set()
    for heading in range(4):
        for exit_ in range(4):
            if (transitions >> (4 * (3 - heading) + (3 - exit_))) & 1:
                sides.update((exit_, (heading + 2) % 4))
    return sides


def reference_policies(trains, order):
    """The scripted dispatching rules the target is set off, as `(name, rule)` pairs. A rule
    maps a state to the trains to let go before the next `wait`: all at once; one at a time,
    each when the last has arrived, in index order and shortest route first; and one every
    `wait`, in the same two orders."""

    def at_once(state):
        return [i for i in range(trains) if i not in state.released]

    def one_at_a_time(sequence):
        def rule(state):
            if any(state.trains[i][0] != "arrived" for i in state.released):
                return []
            return [i for i in sequence if i not in state.released][:1]
        return rule

    def staggered(sequence):
        def rule(state):
            return [i for i in sequence if i not in state.released][:1]
        return rule

    index = list(range(trains))
    return [("all at once", at_once), ("one at a time", one_at_a_time(index)),
            ("one at a time, shortest first", one_at_a_time(order)),
            ("one a wait", staggered(index)), ("one a wait, shortest first", staggered(order))]


class RailwayEnv(Environment):
    """Every train to its target before the horizon with the train-steps under the target."""

    def __init__(self):
        super().__init__("railway")
        self.instance = None
        self.index = None
        self.state = None
        self.witness = None
        self.witness_expansions = None
        self.__env__ = None

    # ------------------------------------------------------------------ instances

    def set_index(self, index):
        if not 0 <= index < len(TIMETABLES):
            raise IndexError(f"Invalid index: {index}. There are {len(TIMETABLES)} timetables, so "
                             f"the index must be 0-{len(TIMETABLES) - 1}.")
        self.set_instance(TIMETABLES[index])
        self.index = index

    def set_instance(self, instance):
        """Select a timetable: `seed` (Flatland's, for the network and the trains), `cities`,
        `trains`, `size` (the grid's side), `horizon` (steps) and `target` (train-steps), with
        `solved_at`, the length of the plan it was accepted on, if known."""
        for key in ("seed", "cities", "trains", "size", "horizon", "target"):
            if key not in instance:
                raise ValueError(f"a timetable needs `{key}`")
        self.instance = {"seed": int(instance["seed"]), "cities": int(instance["cities"]),
                         "trains": int(instance["trains"]), "size": int(instance["size"]),
                         "horizon": int(instance["horizon"]), "target": int(instance["target"])}
        if instance.get("solved_at") is not None:
            self.instance["solved_at"] = int(instance["solved_at"])
        self.index = None
        self.witness = self.witness_expansions = None
        self.__env__ = None

    def generate_instance(self, seed=None, cities=None, trains=None, size=None, attempts=60):
        """Draw a timetable, select it, and return it as the dict `set_instance` takes.

        The cities, trains and grid side come from `CITIES`, `TRAINS` and `SIZES` unless given,
        and Flatland draws the network and the trains from a seeded seed. The horizon is
        Flatland's own episode length for the network, lengthened where needed to the step at
        which running the trains one at a time finishes. The rules of `reference_policies` are
        then run; the target is the least train-steps any of them gets every train home with,
        and that rule's plan is the witness. A draw is thrown back when Flatland cannot build
        it, when a train has no route, when no rule gets everyone home, or when letting every
        train go at once does as well as any, since then nothing needs deciding.
        """
        random_, _ = rng(seed)
        found = {}

        def draw(attempt):
            return {"seed": random_.randrange(1, 2 ** 30), "cities": cities or random_.choice(CITIES),
                    "trains": trains or random_.choice(TRAINS), "size": size or random_.choice(SIZES),
                    "horizon": 0, "target": 0}

        def accept(instance):
            self.set_instance(instance)
            try:
                env = self._env(probe=True)
            except Exception:
                return False                     # Flatland could not lay the network out
            if len(env.agents) != instance["trains"] or any(d == float("inf") for d in self.__routes__):
                return False
            order = sorted(range(instance["trains"]), key=lambda i: (self.__routes__[i], i))
            self.instance["target"] = 10 ** 9
            self.instance["horizon"] = 10 ** 6
            _, sequential = self.run_rule(reference_policies(instance["trains"], order)[1][1])
            if sequential is None:
                return False
            horizon = max(self.__episode__, sequential.step)
            self.instance["horizon"] = horizon
            outcomes = [self.run_rule(rule) for _, rule in reference_policies(instance["trains"], order)]
            finished = [(last.train_steps, k, plan) for k, (plan, last) in enumerate(outcomes)
                        if last is not None]
            if not finished:
                return False
            best = min(finished)
            if outcomes[0][1] is not None and outcomes[0][1].train_steps <= best[0]:
                return False                     # letting every train go at once is already best
            instance.update(horizon=horizon, target=best[0], solved_at=len(best[2]))
            found["plan"], found["measured"] = best[2], len(outcomes)
            return True

        instance = draw_until(draw, accept, attempts, "railway timetable")
        self.set_instance(instance)
        self.witness, self.witness_expansions = found["plan"], found["measured"]
        return instance

    def run_rule(self, rule):
        """Dispatch by `rule` from the start and return `(plan, final state)`, the state `None`
        unless every train arrives inside the horizon and the target."""
        state, _ = self.reset()
        plan = []
        while not self.is_goal(state) and not self.is_terminal(state):
            for train in rule(state):
                plan.append(RailwayAction(train))
                state = self.__advance__(state, plan[-1])
            plan.append(WAIT)
            state = self.__advance__(state, WAIT)
        return plan, (state if self.is_goal(state) else None)

    # ------------------------------------------------------------------ the simulator

    def _env(self, probe=False):
        if self.__env__ is None:
            env = build(self.instance)
            fl = _flatland()
            self.__fl__ = fl
            self.__distance__ = env.distance_map.get()
            self.__episode__ = env._max_episode_steps          # Flatland's own episode length
            self.__routes__ = [float(self.__distance__[a.handle][cell][direction])
                               for a in env.agents for cell, direction in [a.initial_configuration]]
            if not probe and len(env.agents) != self.instance["trains"]:
                raise RuntimeError(f"Flatland drew {len(env.agents)} trains for a timetable of "
                                   f"{self.instance['trains']}")
            env._max_episode_steps = self.instance["horizon"]
            self.__env__ = env
            self.__layout__ = RailMap.of(env)
            self.__start__ = self._save(())
        return self.__env__

    def _save(self, released):
        env = self.__env__
        return pickle.dumps((env.agents, env._elapsed_steps, env.dones, env.agent_positions,
                             tuple(sorted(released)), 0))

    def _load(self, snapshot):
        env = self._env()
        agents, step, dones, positions, released, _ = pickle.loads(snapshot)
        env.agents, env._elapsed_steps, env.dones, env.agent_positions = agents, step, dones, positions
        env._max_episode_steps = self.instance["horizon"]
        return set(released)

    def _next(self, agent):
        """The configuration a train moves to next on its shortest route, and the action."""
        actions = self.__fl__["actions"]
        if agent.current_configuration is None:
            return agent.initial_configuration, actions.MOVE_FORWARD
        best = None
        for action in (actions.MOVE_FORWARD, actions.MOVE_LEFT, actions.MOVE_RIGHT):
            transition = self.__env__.rail.apply_action_independent(action, agent.current_configuration)
            if transition is None:
                continue
            (cell, direction), _ = transition
            distance = self.__distance__[agent.handle][cell[0]][cell[1]][direction]
            if best is None or distance < best[0]:
                best = (distance, action, (cell, direction))
        return best[2], best[1]

    def _deadlocked(self):
        """The trains that can never move again: each one waits on the train in the cell it
        needs next, and a chain of waits that closes on itself never clears."""
        agents = [a for a in self.__env__.agents if STATUS[int(a.state)] != "arrived"]
        holder = {a.current_configuration[0]: a.handle for a in agents
                  if a.current_configuration is not None}
        waits_on = {a.handle: holder.get(self._next(a)[0][0]) for a in agents}
        stuck = set()
        for train in waits_on:
            seen, current = set(), train
            while current is not None and current not in seen:
                seen.add(current)
                current = waits_on[current]
            if current is not None:
                stuck.add(train)
        return stuck

    def _step(self, released):
        env = self.__env__
        actions = self.__fl__["actions"]
        moves = {}
        for agent in env.agents:
            if STATUS[int(agent.state)] == "arrived":
                continue
            moves[agent.handle] = self._next(agent)[1] if agent.handle in released else actions.DO_NOTHING
        running = sum(1 for a in env.agents if STATUS[int(a.state)] != "arrived")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env.step(moves)
        return running

    def _wait(self, released, train_steps):
        """Run `PERIOD` steps, or to the end once every train has been let go, stopping early
        at a deadlock, at every train arrived, at the horizon or past the target."""
        env = self.__env__
        everyone = len(released) == len(env.agents)
        steps = 0
        while not env.dones["__all__"]:
            train_steps += self._step(released)
            steps += 1
            if (train_steps > self.instance["target"] or self._deadlocked()
                    or (not everyone and steps >= PERIOD)):
                break
        return train_steps

    def _read(self, released, train_steps, depth):
        env = self.__env__
        trains, to_go = [], 0
        for agent in env.agents:
            status = STATUS[int(agent.state)]
            configuration = agent.current_configuration
            if configuration is None:
                trains.append((status, None, None))
            else:
                (row, column), direction = configuration
                trains.append((status, (int(row), int(column)), int(direction)))
            if status != "arrived":
                cell, direction = configuration or agent.initial_configuration
                to_go += int(self.__distance__[agent.handle][cell[0]][cell[1]][direction])
        stuck = self._deadlocked()
        snapshot = pickle.dumps((env.agents, env._elapsed_steps, env.dones, env.agent_positions,
                                 tuple(sorted(released)), train_steps))
        return RailwayState(env._elapsed_steps, trains, released, train_steps, to_go, stuck, snapshot,
                            target=self.instance["target"], horizon=self.instance["horizon"], depth=depth,
                            layout=self.__layout__)

    def close(self):
        self.__env__ = None

    # ------------------------------------------------------------------- contract

    def reset(self):
        if self.instance is None:
            self.set_index(0)
        self._env()
        self._load(self.__start__)
        self.state = self._read(set(), 0, 0)
        return self.state, {"timetable": self.index, "trains": self.instance["trains"],
                            "cities": self.instance["cities"], "horizon": self.instance["horizon"],
                            "target": self.instance["target"], "generated": self.index is None}

    def is_goal(self, state):
        return state.left == 0 and state.train_steps <= self.instance["target"]

    def is_terminal(self, state):
        return not self.is_goal(state) and (
            bool(state.deadlocked) or state.train_steps > self.instance["target"]
            or state.step >= self.instance["horizon"] or state.left == 0)

    def get_actions(self, state=None):
        trains = self.instance["trains"] if self.instance else 0
        return [RailwayAction(i) for i in range(trains)] + [WAIT]

    def applicable(self, state):
        """`depart(i)` for each train not yet let go, in index order, and `wait`."""
        return [RailwayAction(i) for i, (status, _, _) in enumerate(state.trains)
                if i not in state.released and status != "arrived"] + [WAIT]

    def successors(self, state):
        if self.is_goal(state) or self.is_terminal(state):
            return []
        children = []
        for action in self.applicable(state):
            child = self._apply(state, action)
            if child != state:
                children.append((action, child))
        return children

    def _apply(self, state, action):
        released = self._load(state.snapshot)
        _, _, _, _, _, train_steps = pickle.loads(state.snapshot)
        if action.train is not None:
            released.add(action.train)
        else:
            train_steps = self._wait(released, train_steps)
        return self._read(released, train_steps, state.depth + 1)

    def __advance__(self, state, action):
        if self.is_goal(state) or self.is_terminal(state):
            return state
        if not isinstance(action, RailwayAction):
            action = RailwayAction.parse(action)
        if action not in self.applicable(state):
            return state
        return self._apply(state, action)


#: The bundled timetables: `generate_instance(seed)` for the seed beside each, embedded as the
#: plain data `set_instance` takes, with the plan each was accepted on in
#: `tests/data/railway_solutions.json`.
TIMETABLES = (
    # seed 9000; 3 cities, 5 trains, 25 by 25, 35-step plan
    {"seed": 157630427, "cities": 3, "trains": 5, "size": 25, "horizon": 157, "target": 406, "solved_at": 35},
    # seed 9001; 3 cities, 4 trains, 25 by 25, 14-step plan
    {"seed": 39830315, "cities": 3, "trains": 4, "size": 25, "horizon": 47, "target": 117, "solved_at": 14},
    # seed 9002; 4 cities, 3 trains, 25 by 25, 6-step plan
    {"seed": 620881865, "cities": 4, "trains": 3, "size": 25, "horizon": 139, "target": 146, "solved_at": 6},
    # seed 9003; 4 cities, 2 trains, 25 by 25, 13-step plan
    {"seed": 610606394, "cities": 4, "trains": 2, "size": 25, "horizon": 76, "target": 113, "solved_at": 13},
    # seed 9004; 3 cities, 5 trains, 30 by 30, 32-step plan
    {"seed": 1063345461, "cities": 3, "trains": 5, "size": 30, "horizon": 143, "target": 380, "solved_at": 32},
    # seed 9005; 3 cities, 2 trains, 30 by 30, 10-step plan
    {"seed": 376437504, "cities": 3, "trains": 2, "size": 30, "horizon": 55, "target": 83, "solved_at": 10},
    # seed 9006; 2 cities, 5 trains, 25 by 25, 30-step plan
    {"seed": 185122146, "cities": 2, "trains": 5, "size": 25, "horizon": 123, "target": 344, "solved_at": 30},
    # seed 9007; 4 cities, 3 trains, 35 by 35, 42-step plan
    {"seed": 224251655, "cities": 4, "trains": 3, "size": 35, "horizon": 215, "target": 434, "solved_at": 42},
    # seed 9008; 3 cities, 5 trains, 30 by 30, 32-step plan
    {"seed": 648715749, "cities": 3, "trains": 5, "size": 30, "horizon": 163, "target": 390, "solved_at": 32},
    # seed 9009; 4 cities, 5 trains, 25 by 25, 33-step plan
    {"seed": 391298913, "cities": 4, "trains": 5, "size": 25, "horizon": 153, "target": 400, "solved_at": 33},
    # seed 9010; 3 cities, 4 trains, 35 by 35, 29-step plan
    {"seed": 1069820629, "cities": 3, "trains": 4, "size": 35, "horizon": 183, "target": 367, "solved_at": 29},
    # seed 9011; 4 cities, 4 trains, 30 by 30, 32-step plan
    {"seed": 573281517, "cities": 4, "trains": 4, "size": 30, "horizon": 181, "target": 353, "solved_at": 32},
    # seed 9012; 2 cities, 5 trains, 25 by 25, 25-step plan
    {"seed": 357445393, "cities": 2, "trains": 5, "size": 25, "horizon": 101, "target": 268, "solved_at": 25},
    # seed 9013; 3 cities, 2 trains, 30 by 30, 8-step plan
    {"seed": 890600818, "cities": 3, "trains": 2, "size": 30, "horizon": 38, "target": 57, "solved_at": 8},
    # seed 9014; 3 cities, 3 trains, 25 by 25, 6-step plan
    {"seed": 852624802, "cities": 3, "trains": 3, "size": 25, "horizon": 54, "target": 63, "solved_at": 6},
    # seed 9015; 4 cities, 2 trains, 35 by 35, 8-step plan
    {"seed": 306407235, "cities": 4, "trains": 2, "size": 35, "horizon": 36, "target": 53, "solved_at": 8},
    # seed 9016; 3 cities, 4 trains, 25 by 25, 22-step plan
    {"seed": 448293656, "cities": 3, "trains": 4, "size": 25, "horizon": 111, "target": 209, "solved_at": 22},
    # seed 9017; 4 cities, 3 trains, 30 by 30, 33-step plan
    {"seed": 996360513, "cities": 4, "trains": 3, "size": 30, "horizon": 175, "target": 344, "solved_at": 33},
    # seed 9018; 3 cities, 5 trains, 25 by 25, 32-step plan
    {"seed": 284830276, "cities": 3, "trains": 5, "size": 25, "horizon": 127, "target": 340, "solved_at": 32},
    # seed 9019; 4 cities, 3 trains, 30 by 30, 24-step plan
    {"seed": 517958469, "cities": 4, "trains": 3, "size": 30, "horizon": 118, "target": 235, "solved_at": 24},
    # seed 9020; 4 cities, 4 trains, 25 by 25, 27-step plan
    {"seed": 975312175, "cities": 4, "trains": 4, "size": 25, "horizon": 144, "target": 285, "solved_at": 27},
    # seed 9021; 2 cities, 5 trains, 30 by 30, 32-step plan
    {"seed": 85217924, "cities": 2, "trains": 5, "size": 30, "horizon": 145, "target": 358, "solved_at": 32},
    # seed 9022; 3 cities, 5 trains, 25 by 25, 34-step plan
    {"seed": 11904788, "cities": 3, "trains": 5, "size": 25, "horizon": 192, "target": 445, "solved_at": 34},
    # seed 9023; 4 cities, 2 trains, 35 by 35, 8-step plan
    {"seed": 1036363007, "cities": 4, "trains": 2, "size": 35, "horizon": 39, "target": 59, "solved_at": 8},
    # seed 9024; 4 cities, 4 trains, 35 by 35, 43-step plan
    {"seed": 680983499, "cities": 4, "trains": 4, "size": 35, "horizon": 217, "target": 487, "solved_at": 43},
    # seed 9025; 4 cities, 5 trains, 30 by 30, 64-step plan
    {"seed": 360345570, "cities": 4, "trains": 5, "size": 30, "horizon": 312, "target": 795, "solved_at": 64},
    # seed 9026; 3 cities, 4 trains, 30 by 30, 31-step plan
    {"seed": 942561203, "cities": 3, "trains": 4, "size": 30, "horizon": 161, "target": 363, "solved_at": 31},
    # seed 9027; 4 cities, 2 trains, 30 by 30, 18-step plan
    {"seed": 978860285, "cities": 4, "trains": 2, "size": 30, "horizon": 119, "target": 179, "solved_at": 18},
    # seed 9028; 4 cities, 4 trains, 25 by 25, 25-step plan
    {"seed": 422354421, "cities": 4, "trains": 4, "size": 25, "horizon": 139, "target": 265, "solved_at": 25},
    # seed 9029; 4 cities, 4 trains, 35 by 35, 43-step plan
    {"seed": 344135044, "cities": 4, "trains": 4, "size": 35, "horizon": 216, "target": 491, "solved_at": 43},
    # seed 9030; 3 cities, 4 trains, 30 by 30, 26-step plan
    {"seed": 321373061, "cities": 3, "trains": 4, "size": 30, "horizon": 117, "target": 283, "solved_at": 26},
    # seed 9031; 4 cities, 4 trains, 35 by 35, 38-step plan
    {"seed": 1059605557, "cities": 4, "trains": 4, "size": 35, "horizon": 161, "target": 401, "solved_at": 38},
    # seed 9032; 4 cities, 5 trains, 25 by 25, 46-step plan
    {"seed": 992213298, "cities": 4, "trains": 5, "size": 25, "horizon": 216, "target": 561, "solved_at": 46},
    # seed 9033; 4 cities, 2 trains, 25 by 25, 16-step plan
    {"seed": 276491288, "cities": 4, "trains": 2, "size": 25, "horizon": 101, "target": 151, "solved_at": 16},
    # seed 9034; 4 cities, 5 trains, 35 by 35, 57-step plan
    {"seed": 848217002, "cities": 4, "trains": 5, "size": 35, "horizon": 257, "target": 742, "solved_at": 57},
    # seed 9035; 4 cities, 4 trains, 30 by 30, 23-step plan
    {"seed": 714976752, "cities": 4, "trains": 4, "size": 30, "horizon": 119, "target": 247, "solved_at": 23},
    # seed 9036; 4 cities, 4 trains, 35 by 35, 43-step plan
    {"seed": 1049550267, "cities": 4, "trains": 4, "size": 35, "horizon": 219, "target": 487, "solved_at": 43},
    # seed 9037; 4 cities, 4 trains, 30 by 30, 32-step plan
    {"seed": 961625233, "cities": 4, "trains": 4, "size": 30, "horizon": 146, "target": 347, "solved_at": 32},
    # seed 9038; 4 cities, 3 trains, 25 by 25, 11-step plan
    {"seed": 25569470, "cities": 4, "trains": 3, "size": 25, "horizon": 56, "target": 97, "solved_at": 11},
    # seed 9039; 2 cities, 5 trains, 30 by 30, 26-step plan
    {"seed": 943283422, "cities": 2, "trains": 5, "size": 30, "horizon": 97, "target": 286, "solved_at": 26},
    # seed 9040; 3 cities, 5 trains, 35 by 35, 74-step plan
    {"seed": 56394861, "cities": 3, "trains": 5, "size": 35, "horizon": 309, "target": 982, "solved_at": 74},
    # seed 9041; 2 cities, 5 trains, 35 by 35, 40-step plan
    {"seed": 815210155, "cities": 2, "trains": 5, "size": 35, "horizon": 174, "target": 495, "solved_at": 40},
    # seed 9042; 3 cities, 4 trains, 30 by 30, 26-step plan
    {"seed": 245364516, "cities": 3, "trains": 4, "size": 30, "horizon": 149, "target": 289, "solved_at": 26},
    # seed 9043; 4 cities, 4 trains, 35 by 35, 23-step plan
    {"seed": 273441536, "cities": 4, "trains": 4, "size": 35, "horizon": 118, "target": 245, "solved_at": 23},
    # seed 9044; 4 cities, 5 trains, 35 by 35, 58-step plan
    {"seed": 696776955, "cities": 4, "trains": 5, "size": 35, "horizon": 269, "target": 762, "solved_at": 58},
    # seed 9045; 3 cities, 4 trains, 30 by 30, 29-step plan
    {"seed": 803278543, "cities": 3, "trains": 4, "size": 30, "horizon": 135, "target": 325, "solved_at": 29},
    # seed 9046; 3 cities, 4 trains, 25 by 25, 32-step plan
    {"seed": 596847022, "cities": 3, "trains": 4, "size": 25, "horizon": 162, "target": 361, "solved_at": 32},
    # seed 9047; 3 cities, 5 trains, 25 by 25, 46-step plan
    {"seed": 65190784, "cities": 3, "trains": 5, "size": 25, "horizon": 196, "target": 591, "solved_at": 46},
    # seed 9048; 4 cities, 4 trains, 25 by 25, 35-step plan
    {"seed": 295537948, "cities": 4, "trains": 4, "size": 25, "horizon": 160, "target": 399, "solved_at": 35},
    # seed 9049; 3 cities, 5 trains, 30 by 30, 10-step plan
    {"seed": 684449591, "cities": 3, "trains": 5, "size": 30, "horizon": 156, "target": 193, "solved_at": 10},
    # seed 9050; 3 cities, 5 trains, 30 by 30, 48-step plan
    {"seed": 528476589, "cities": 3, "trains": 5, "size": 30, "horizon": 218, "target": 623, "solved_at": 48},
    # seed 9051; 4 cities, 2 trains, 30 by 30, 17-step plan
    {"seed": 196542150, "cities": 4, "trains": 2, "size": 30, "horizon": 113, "target": 167, "solved_at": 17},
    # seed 9052; 4 cities, 5 trains, 35 by 35, 51-step plan
    {"seed": 36992498, "cities": 4, "trains": 5, "size": 35, "horizon": 207, "target": 676, "solved_at": 51},
    # seed 9053; 2 cities, 5 trains, 25 by 25, 30-step plan
    {"seed": 404236284, "cities": 2, "trains": 5, "size": 25, "horizon": 125, "target": 326, "solved_at": 30},
    # seed 9054; 3 cities, 5 trains, 25 by 25, 25-step plan
    {"seed": 957003503, "cities": 3, "trains": 5, "size": 25, "horizon": 133, "target": 292, "solved_at": 25},
    # seed 9055; 4 cities, 2 trains, 35 by 35, 17-step plan
    {"seed": 800918436, "cities": 4, "trains": 2, "size": 35, "horizon": 111, "target": 167, "solved_at": 17},
    # seed 9056; 2 cities, 4 trains, 25 by 25, 21-step plan
    {"seed": 240960256, "cities": 2, "trains": 4, "size": 25, "horizon": 104, "target": 221, "solved_at": 21},
    # seed 9057; 2 cities, 5 trains, 35 by 35, 40-step plan
    {"seed": 1015081419, "cities": 2, "trains": 5, "size": 35, "horizon": 175, "target": 500, "solved_at": 40},
    # seed 9058; 3 cities, 3 trains, 25 by 25, 16-step plan
    {"seed": 776845085, "cities": 3, "trains": 3, "size": 25, "horizon": 87, "target": 156, "solved_at": 16},
    # seed 9059; 4 cities, 4 trains, 30 by 30, 35-step plan
    {"seed": 8983463, "cities": 4, "trains": 4, "size": 30, "horizon": 176, "target": 387, "solved_at": 35},
    # seed 9060; 4 cities, 4 trains, 25 by 25, 21-step plan
    {"seed": 976201707, "cities": 4, "trains": 4, "size": 25, "horizon": 128, "target": 245, "solved_at": 21},
    # seed 9061; 4 cities, 3 trains, 30 by 30, 24-step plan
    {"seed": 217361879, "cities": 4, "trains": 3, "size": 30, "horizon": 136, "target": 243, "solved_at": 24},
    # seed 9062; 4 cities, 4 trains, 35 by 35, 32-step plan
    {"seed": 347468887, "cities": 4, "trains": 4, "size": 35, "horizon": 186, "target": 397, "solved_at": 32},
    # seed 9063; 3 cities, 4 trains, 25 by 25, 21-step plan
    {"seed": 470107789, "cities": 3, "trains": 4, "size": 25, "horizon": 126, "target": 243, "solved_at": 21},
    # seed 9064; 3 cities, 5 trains, 30 by 30, 40-step plan
    {"seed": 7887745, "cities": 3, "trains": 5, "size": 30, "horizon": 153, "target": 502, "solved_at": 40},
    # seed 9065; 3 cities, 4 trains, 30 by 30, 23-step plan
    {"seed": 625277176, "cities": 3, "trains": 4, "size": 30, "horizon": 109, "target": 245, "solved_at": 23},
    # seed 9066; 4 cities, 3 trains, 25 by 25, 28-step plan
    {"seed": 408229125, "cities": 4, "trains": 3, "size": 25, "horizon": 147, "target": 282, "solved_at": 28},
    # seed 9067; 4 cities, 2 trains, 25 by 25, 14-step plan
    {"seed": 222515115, "cities": 4, "trains": 2, "size": 25, "horizon": 85, "target": 127, "solved_at": 14},
    # seed 9068; 3 cities, 3 trains, 25 by 25, 12-step plan
    {"seed": 51399698, "cities": 3, "trains": 3, "size": 25, "horizon": 60, "target": 107, "solved_at": 12},
    # seed 9069; 2 cities, 2 trains, 30 by 30, 8-step plan
    {"seed": 1008089559, "cities": 2, "trains": 2, "size": 30, "horizon": 39, "target": 59, "solved_at": 8},
    # seed 9070; 2 cities, 2 trains, 30 by 30, 7-step plan
    {"seed": 95551074, "cities": 2, "trains": 2, "size": 30, "horizon": 28, "target": 41, "solved_at": 7},
    # seed 9071; 2 cities, 4 trains, 30 by 30, 27-step plan
    {"seed": 1051066146, "cities": 2, "trains": 4, "size": 30, "horizon": 124, "target": 301, "solved_at": 27},
    # seed 9072; 4 cities, 5 trains, 30 by 30, 35-step plan
    {"seed": 121040083, "cities": 4, "trains": 5, "size": 30, "horizon": 137, "target": 434, "solved_at": 35},
    # seed 9073; 4 cities, 4 trains, 25 by 25, 38-step plan
    {"seed": 829910765, "cities": 4, "trains": 4, "size": 25, "horizon": 161, "target": 433, "solved_at": 38},
    # seed 9074; 4 cities, 2 trains, 30 by 30, 15-step plan
    {"seed": 678783883, "cities": 4, "trains": 2, "size": 30, "horizon": 94, "target": 141, "solved_at": 15},
    # seed 9075; 3 cities, 3 trains, 25 by 25, 12-step plan
    {"seed": 454107677, "cities": 3, "trains": 3, "size": 25, "horizon": 66, "target": 109, "solved_at": 12},
    # seed 9076; 3 cities, 5 trains, 30 by 30, 42-step plan
    {"seed": 34196863, "cities": 3, "trains": 5, "size": 30, "horizon": 197, "target": 512, "solved_at": 42},
    # seed 9077; 3 cities, 4 trains, 25 by 25, 25-step plan
    {"seed": 294962623, "cities": 3, "trains": 4, "size": 25, "horizon": 160, "target": 305, "solved_at": 25},
    # seed 9078; 3 cities, 4 trains, 30 by 30, 50-step plan
    {"seed": 35711606, "cities": 3, "trains": 4, "size": 30, "horizon": 217, "target": 495, "solved_at": 50},
    # seed 9079; 3 cities, 5 trains, 30 by 30, 43-step plan
    {"seed": 146580488, "cities": 3, "trains": 5, "size": 30, "horizon": 185, "target": 582, "solved_at": 43},
    # seed 9080; 4 cities, 4 trains, 25 by 25, 35-step plan
    {"seed": 508828721, "cities": 4, "trains": 4, "size": 25, "horizon": 163, "target": 389, "solved_at": 35},
    # seed 9081; 3 cities, 5 trains, 25 by 25, 48-step plan
    {"seed": 84645752, "cities": 3, "trains": 5, "size": 25, "horizon": 241, "target": 598, "solved_at": 48},
    # seed 9082; 4 cities, 3 trains, 25 by 25, 6-step plan
    {"seed": 877098909, "cities": 4, "trains": 3, "size": 25, "horizon": 112, "target": 119, "solved_at": 6},
    # seed 9083; 3 cities, 4 trains, 35 by 35, 21-step plan
    {"seed": 399316842, "cities": 3, "trains": 4, "size": 35, "horizon": 132, "target": 249, "solved_at": 21},
    # seed 9084; 4 cities, 5 trains, 35 by 35, 37-step plan
    {"seed": 475067026, "cities": 4, "trains": 5, "size": 35, "horizon": 179, "target": 450, "solved_at": 37},
    # seed 9085; 4 cities, 5 trains, 30 by 30, 38-step plan
    {"seed": 485457192, "cities": 4, "trains": 5, "size": 30, "horizon": 191, "target": 500, "solved_at": 38},
    # seed 9086; 4 cities, 4 trains, 30 by 30, 23-step plan
    {"seed": 49174321, "cities": 4, "trains": 4, "size": 30, "horizon": 111, "target": 227, "solved_at": 23},
    # seed 9087; 4 cities, 4 trains, 25 by 25, 22-step plan
    {"seed": 118950239, "cities": 4, "trains": 4, "size": 25, "horizon": 97, "target": 215, "solved_at": 22},
    # seed 9088; 4 cities, 4 trains, 35 by 35, 42-step plan
    {"seed": 268538496, "cities": 4, "trains": 4, "size": 35, "horizon": 221, "target": 469, "solved_at": 42},
    # seed 9089; 4 cities, 2 trains, 30 by 30, 20-step plan
    {"seed": 1028699807, "cities": 4, "trains": 2, "size": 30, "horizon": 132, "target": 197, "solved_at": 20},
    # seed 9090; 4 cities, 3 trains, 35 by 35, 23-step plan
    {"seed": 923658003, "cities": 4, "trains": 3, "size": 35, "horizon": 132, "target": 245, "solved_at": 23},
    # seed 9091; 3 cities, 4 trains, 30 by 30, 28-step plan
    {"seed": 767218348, "cities": 3, "trains": 4, "size": 30, "horizon": 134, "target": 291, "solved_at": 28},
    # seed 9092; 4 cities, 5 trains, 25 by 25, 36-step plan
    {"seed": 88075309, "cities": 4, "trains": 5, "size": 25, "horizon": 146, "target": 487, "solved_at": 36},
    # seed 9093; 4 cities, 2 trains, 35 by 35, 10-step plan
    {"seed": 583838555, "cities": 4, "trains": 2, "size": 35, "horizon": 55, "target": 83, "solved_at": 10},
    # seed 9094; 3 cities, 2 trains, 25 by 25, 8-step plan
    {"seed": 177825972, "cities": 3, "trains": 2, "size": 25, "horizon": 37, "target": 55, "solved_at": 8},
    # seed 9095; 3 cities, 2 trains, 35 by 35, 4-step plan
    {"seed": 411955586, "cities": 3, "trains": 2, "size": 35, "horizon": 110, "target": 113, "solved_at": 4},
    # seed 9096; 3 cities, 4 trains, 25 by 25, 25-step plan
    {"seed": 68672395, "cities": 3, "trains": 4, "size": 25, "horizon": 142, "target": 289, "solved_at": 25},
    # seed 9097; 3 cities, 5 trains, 25 by 25, 35-step plan
    {"seed": 640707517, "cities": 3, "trains": 5, "size": 25, "horizon": 136, "target": 467, "solved_at": 35},
    # seed 9098; 4 cities, 5 trains, 25 by 25, 10-step plan
    {"seed": 109407533, "cities": 4, "trains": 5, "size": 25, "horizon": 192, "target": 227, "solved_at": 10},
    # seed 9099; 3 cities, 4 trains, 35 by 35, 32-step plan
    {"seed": 237182022, "cities": 3, "trains": 4, "size": 35, "horizon": 125, "target": 335, "solved_at": 32},
)
