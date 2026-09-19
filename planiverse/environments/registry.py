"""What each environment is, as data rather than as a package path.

The old layout encoded one fact about an environment ("real world problem" or "retro game")
in the directory it lived in, which meant the fact could not be queried, could not be
combined with any other fact, and could not be wrong without moving files. Everything a
planner might actually select on is here instead, and the old distinction survives as
family tags, alongside finer ones: `game`, `operational` (an agent running a system it is
responsible for), and `security`.

    >>> from planiverse.environments import list_environments, make
    >>> [spec.name for spec in list_environments(tag="operational")]
    ['crop_management', 'power_grid', 'water_network']
    >>> env = make("puzznic")

Nothing here imports an environment module. The specs are declarative and `make` imports
lazily, so listing the catalogue costs nothing even though some of it needs grid2op, WNTR or
PCSE to run.
"""
import importlib
from dataclasses import dataclass, field

#: How a state is identified, which is the thing that decides whether search can branch.
#:
#: - `value`:    the state carries its own contents; expanding is pure.
#: - `path`:     the state is the decision sequence, replayed on demand. Sound only because
#:               the simulator is deterministic.
#: - `snapshot`: the state carries a serialised emulator image (a save state), and is told
#:               apart from others by what the emulator's game wrapper reads off it.
STATE_IDENTITIES = ("value", "path", "snapshot")


@dataclass(frozen=True)
class EnvironmentSpec:
    """Everything about an environment that does not require importing it."""

    name: str
    factory: str                     #: "module:ClassName", imported on demand
    summary: str
    instances: str                   #: how many bundled problems it offers, in words
    deterministic: bool
    state_identity: str
    requires: tuple = ()             #: third-party modules needed to run it
    docs: str = ""
    tags: frozenset = field(default_factory=frozenset)
    #: What `generate_instance` draws at random, in words. Every bundled environment has a
    #: generator; this says what one of its fresh instances varies in.
    generates: str = ""
    #: Constructor keyword arguments, as `(name, value)` pairs, for the environments whose
    #: `__init__` takes a required argument. Without these `make(name)` works for most of the
    #: catalogue and raises a `TypeError` for the rest, which makes "build every registered
    #: environment" (what the benchmark does) impossible to write generically.
    #: Pairs rather than a dict so the spec stays hashable.
    defaults: tuple = ()

    def load(self):
        """Import and return the class."""
        module_name, class_name = self.factory.split(":")
        return getattr(importlib.import_module(module_name), class_name)

    def build(self, **kwargs):
        """Construct the environment, with `defaults` filled in and `kwargs` winning."""
        return self.load()(**{**dict(self.defaults), **kwargs})

    def available(self):
        """Can this environment run here (are its dependencies importable)?"""
        for module_name in self.requires:
            try:
                importlib.import_module(module_name)
            except ImportError:
                return False
        return True


REGISTRY = (
    EnvironmentSpec(
        name="puzznic",
        factory="planiverse.environments.games.puzznic:PuzznicGame",
        summary="Sliding block puzzle, re-implemented in pure Python",
        instances="100 levels: the first 100 of the cartridge's 128 rounds",
        generates="board size, wall layout, block colours and pairs",
        deterministic=True,
        state_identity="value",
        docs="docs/environments/puzznic.md",
        tags=frozenset({"game", "puzzle", "dependency-free"}),
    ),
    EnvironmentSpec(
        name="flipull",
        factory="planiverse.environments.games.flipull:FlipullGame",
        summary="Flipull-like throwing puzzle, re-implemented in pure Python",
        instances="100 stages: 32 to the cartridge's table, then 68 generated",
        generates="wall size, block types, arrangement and clear target",
        deterministic=True,
        state_identity="value",
        docs="docs/environments/flipull.md",
        tags=frozenset({"game", "puzzle", "dependency-free"}),
    ),
    EnvironmentSpec(
        name="amazing_tater",
        factory="planiverse.environments.games.amazing_tater:AmazingTaterGame",
        summary="Amazing Tater's blocks, pits and turnstiles, re-implemented in pure Python",
        instances="100 rooms: 41 puzzle-mode, then the first 59 of 64 beginner and action-mode",
        generates="room size, walls, blocks, pits, turnstiles and taters",
        deterministic=True,
        state_identity="value",
        docs="docs/environments/amazing-tater.md",
        tags=frozenset({"game", "puzzle", "dependency-free"}),
    ),
    EnvironmentSpec(
        name="lolo",
        factory="planiverse.environments.games.lolo:LoloGame",
        summary="Adventures of Lolo's block-and-heart puzzle, re-implemented in pure Python",
        instances="100 rooms: the first 100 of the cartridge's 163",
        generates="terrain, hearts, Emerald Framers, Snakeys and Medusas",
        deterministic=True,
        state_identity="value",
        docs="docs/environments/lolo.md",
        tags=frozenset({"game", "puzzle", "dependency-free"}),
    ),
    EnvironmentSpec(
        name="slingshot",
        factory="planiverse.environments.slingshot.environment:SlingshotEnv",
        summary="A slingshot physics puzzle: knock every target down within the shots given",
        instances="100 levels, generated",
        generates="the structures, their materials, where the targets sit, and the shots",
        deterministic=True,
        state_identity="value",
        requires=("pymunk",),
        docs="docs/environments/slingshot.md",
        tags=frozenset({"game", "physics"}),
    ),
    EnvironmentSpec(
        name="artillery",
        factory="planiverse.environments.artillery.environment:ArtilleryEnv",
        summary="Artillery over a hilly field: lob shells through the wind to destroy every target",
        instances="100 fields, generated",
        generates="the terrain, where the targets dig in, the wind, and the shells",
        deterministic=True,
        state_identity="value",
        docs="docs/environments/artillery.md",
        tags=frozenset({"game", "physics", "dependency-free"}),
    ),
    EnvironmentSpec(
        name="tower_defence",
        factory="planiverse.environments.tower_defence.environment:TowerDefenceEnv",
        summary="Tower defence: build between waves so that the last wave leaves you alive",
        instances="100 maps, generated",
        generates="the path, the building slots, the waves, the gold and the lives",
        deterministic=True,
        state_identity="value",
        docs="docs/environments/tower-defence.md",
        tags=frozenset({"game", "dependency-free"}),
    ),
    EnvironmentSpec(
        name="fluid",
        factory="planiverse.environments.fluid.environment:FluidEnv",
        summary="A cellular fluid puzzle: dig channels so that enough water reaches the basin",
        instances="100 caves, generated",
        generates="the cave, the spring, the basin, the drain, the water needed and the digs allowed",
        deterministic=True,
        state_identity="value",
        docs="docs/environments/fluid.md",
        tags=frozenset({"game", "dependency-free"}),
    ),
    EnvironmentSpec(
        name="billiards",
        factory="planiverse.environments.billiards.environment:BilliardsEnv",
        summary="Billiards on pooltool: pot every ball within the shots given without sinking the cue ball",
        instances="100 tables, generated",
        generates="where the balls lie, how many there are, and the shots",
        deterministic=True,
        state_identity="value",
        requires=("pooltool",),
        docs="docs/environments/billiards.md",
        tags=frozenset({"game", "physics"}),
    ),
    EnvironmentSpec(
        name="lemmings",
        factory="planiverse.environments.lemmings.environment:LemmingsEnv",
        summary="A Lemmings-like: steer a crowd of walkers to the exit with a few skills, in time",
        instances="100 levels, generated",
        generates="the platforms, gaps and walls, the entrance and exit, the crowd, the quota and the skills",
        deterministic=True,
        state_identity="value",
        docs="docs/environments/lemmings.md",
        tags=frozenset({"game", "dependency-free"}),
    ),
    EnvironmentSpec(
        name="micropolis",
        factory="planiverse.environments.micropolis.environment:MicropolisEnv",
        summary="A city on the Micropolis engine: zone a site a year and reach the population by the horizon",
        instances="100 cities, generated",
        generates="the map, the layout, the horizon and the population target",
        deterministic=True,
        state_identity="path",
        requires=("micropolisengine",),
        docs="docs/environments/micropolis.md",
        tags=frozenset({"operational", "city"}),
    ),
    EnvironmentSpec(
        name="game_boy",
        factory="planiverse.environments.emulated.game_boy:GameBoyEnv",
        summary="Any Game Boy cartridge, through PyBoy and its game wrappers",
        instances="one per stage, level or room the cartridge's wrapper reaches",
        generates="the stage, the timer seed, and an opening played from its first frame",
        deterministic=True,
        state_identity="snapshot",
        requires=("pyboy",),
        docs="docs/environments/game-boy.md",
        tags=frozenset({"game", "emulator"}),
    ),
    EnvironmentSpec(
        name="retro",
        factory="planiverse.environments.emulated.stable_retro:RetroEnv",
        summary="Any Stable-Retro integration, from a save state to a goal on its variables",
        instances="one per save state the integration ships (Airstriker: 1)",
        generates="the save state, an opening played from it, and the goal",
        deterministic=True,
        state_identity="snapshot",
        requires=("stable_retro",),
        docs="docs/environments/stable-retro.md",
        tags=frozenset({"game", "emulator"}),
    ),
    EnvironmentSpec(
        name="network_attack",
        factory="planiverse.environments.network_attack.network_attack:EnvNASim",
        summary="Penetration testing against a simulated enterprise network",
        instances="100 networks: NASim's 18 benchmarks, then 82 generated",
        generates="network topology, hosts, services, OSs and exploits",
        deterministic=True,
        state_identity="value",
        requires=("nasim",),
        docs="docs/environments/network-attack.md",
        tags=frozenset({"security", "policy"}),
    ),
    EnvironmentSpec(
        name="water_network",
        factory="planiverse.environments.water_network.environment:WaterNetworkEnv",
        summary="Containing a contaminant in a water network without cutting off supply",
        instances="100 scenarios: 9 chosen, then 91 generated",
        generates="the network, and the junction the contaminant enters at",
        deterministic=True,
        state_identity="value",
        requires=("wntr",),
        docs="docs/environments/water-distribution.md",
        tags=frozenset({"operational", "infrastructure", "solver-in-the-loop"}),
    ),
    EnvironmentSpec(
        name="power_grid",
        factory="planiverse.environments.power_grid.environment:PowerGridEnv",
        summary="Restoring grid security by substation topology after a line trips",
        instances="100 contingencies: 9 chosen, then 91 generated",
        generates="the time series, its starting step, and the line that trips",
        deterministic=True,
        state_identity="path",
        requires=("grid2op",),
        docs="docs/environments/power-grid.md",
        tags=frozenset({"operational", "infrastructure", "solver-in-the-loop"}),
    ),
    EnvironmentSpec(
        name="flood_transport",
        factory="planiverse.environments.flood_transport.environment:FloodTransportEnv",
        summary="Protecting a flooding city's roads: which zones to adapt, and when",
        instances="100 scenarios: 15 chosen, then 85 generated",
        generates="the city, its storms, the horizon and the measures on offer",
        deterministic=True,
        state_identity="path",
        docs="docs/environments/flood-transport.md",
        tags=frozenset({"operational", "infrastructure", "climate"}),
    ),
    EnvironmentSpec(
        name="crop_management",
        factory="planiverse.environments.crop_management.environment:CropEnv",
        summary="Scheduling irrigation across a growing season",
        instances="100 seasons: 22 years, then 78 with the sowing moved",
        generates="the year's weather and the sowing date",
        deterministic=True,
        state_identity="path",
        requires=("pcse",),
        docs="docs/environments/crop-management.md",
        tags=frozenset({"operational", "agriculture", "continuous-dynamics"}),
    ),
)

_BY_NAME = {spec.name: spec for spec in REGISTRY}


def list_environments(tag=None, available_only=False):
    """The catalogue, optionally filtered.

    `tag` selects on the family tags (`game`, `operational`, `security`), but also on
    properties that never had a home, like `continuous-dynamics` or `solver-in-the-loop`.
    `available_only` drops the ones whose dependencies are not installed here.
    """
    specs = [spec for spec in REGISTRY if tag is None or tag in spec.tags]
    if available_only:
        specs = [spec for spec in specs if spec.available()]
    return sorted(specs, key=lambda spec: spec.name)


def get_spec(name):
    if name not in _BY_NAME:
        raise KeyError(
            f"Unknown environment: {name!r}. Known: {', '.join(sorted(_BY_NAME))}")
    return _BY_NAME[name]


def make(name, index=None, seed=None, **kwargs):
    """Build an environment by name, optionally selecting its instance.

    ```python
    env = make("water_network", index=8)       # the ninth bundled scenario
    env = make("puzznic", seed=7)              # a freshly generated level
    state, info = env.reset()
    ```

    `index` selects a bundled instance through `set_index`; `seed` draws a new one through
    `generate_instance`. Pass one or the other.
    """
    if index is not None and seed is not None:
        raise ValueError("pass index= for a bundled instance or seed= for a generated one, "
                         "not both")
    environment = get_spec(name).build(**kwargs)
    if index is not None:
        environment.set_index(index)
    elif seed is not None:
        environment.generate_instance(seed=seed)
    return environment


def tags():
    """Every tag in use, so a caller can discover what it may filter on."""
    return frozenset().union(*(spec.tags for spec in REGISTRY))
