"""What each environment is, as data rather than as a package path.

The old layout encoded one fact about an environment — "real world problem" or "retro game" —
in the directory it lived in, which meant the fact could not be queried, could not be
combined with any other fact, and could not be wrong without moving files. Everything a
planner might actually select on is here instead, and the old distinction survives as one
tag among several.

    >>> from planiverse.environments import list_environments, make
    >>> [spec.name for spec in list_environments(tag="infrastructure")]
    ['power_grid', 'water_network']
    >>> env = make("puzznic")

Nothing here imports an environment module. The specs are declarative and `make` imports
lazily, so listing the catalogue costs nothing even though half of it needs PyBoy, grid2op or
numba to run.
"""
import importlib
from dataclasses import dataclass, field

#: How a state is identified, which is the thing that decides whether search can branch.
#:
#: - `value`    — the state carries its own contents; expanding is pure.
#: - `path`     — the state is the decision sequence, replayed on demand. Sound only because
#:                the simulator is deterministic.
#: - `snapshot` — the state carries a serialised simulator image (a Game Boy save-state).
STATE_IDENTITIES = ("value", "path", "snapshot")


@dataclass(frozen=True)
class EnvironmentSpec:
    """Everything about an environment that does not require importing it."""

    name: str
    factory: str                     #: "module:ClassName", imported on demand
    summary: str
    instances: str                   #: how many problems it offers, in words
    deterministic: bool
    state_identity: str
    requires: tuple = ()             #: third-party modules needed to run it
    needs_rom: bool = False          #: needs a copyrighted file the user supplies
    docs: str = ""
    tags: frozenset = field(default_factory=frozenset)

    def load(self):
        """Import and return the class."""
        module_name, class_name = self.factory.split(":")
        return getattr(importlib.import_module(module_name), class_name)

    def available(self):
        """Are this environment's dependencies importable here?"""
        for module_name in self.requires:
            try:
                importlib.import_module(module_name)
            except ImportError:
                return False
        return True


REGISTRY = (
    EnvironmentSpec(
        name="puzznic",
        factory="planiverse.environments.puzznic:PuzznicGame",
        summary="Sliding block puzzle, re-implemented in pure Python",
        instances="50 levels",
        deterministic=True,
        state_identity="value",
        docs="docs/environments/puzznic.md",
        tags=frozenset({"game", "puzzle", "dependency-free"}),
    ),
    EnvironmentSpec(
        name="puzznic_gb",
        factory="planiverse.environments.puzznic_gb:PuzznicGBEnv",
        summary="Puzznic played on the Game Boy cartridge, through PyBoy",
        instances="128 rounds",
        deterministic=True,
        state_identity="snapshot",
        requires=("pyboy",),
        needs_rom=True,
        docs="docs/environments/puzznic-gb.md",
        tags=frozenset({"game", "puzzle", "emulator"}),
    ),
    EnvironmentSpec(
        name="flipull_gb",
        factory="planiverse.environments.flipull_gb:FlipullGBEnv",
        summary="Flipull (Taito's Plotting) on the Game Boy, through PyBoy",
        instances="32 stages",
        deterministic=True,
        state_identity="snapshot",
        requires=("pyboy",),
        needs_rom=True,
        docs="docs/environments/flipull-gb.md",
        tags=frozenset({"game", "puzzle", "emulator"}),
    ),
    EnvironmentSpec(
        name="super_mario_land",
        factory="planiverse.environments.super_mario_land:SuperMarioEnv",
        summary="Super Mario Land on the Game Boy, through PyBoy",
        instances="12 levels",
        deterministic=True,
        state_identity="snapshot",
        requires=("pyboy",),
        needs_rom=True,
        docs="docs/environments/super-mario-land.md",
        tags=frozenset({"game", "platformer", "emulator"}),
    ),
    EnvironmentSpec(
        name="epidemic",
        factory="planiverse.environments.epidemic_control.environment:EpiEnv",
        summary="Vaccination and lockdown policy over a compartmental epidemic model",
        instances="7 scenarios",
        deterministic=True,
        state_identity="value",
        requires=("numba", "sympy", "numpy"),
        docs="docs/environments/epidemic-control.md",
        tags=frozenset({"health", "policy", "continuous-dynamics"}),
    ),
    EnvironmentSpec(
        name="network_attack",
        factory="planiverse.environments.network_attack.network_attack:EnvNASim",
        summary="Penetration testing against a simulated enterprise network",
        instances="18 NASim benchmarks",
        deterministic=True,
        state_identity="value",
        requires=("nasim",),
        docs="docs/environments/network-attack.md",
        tags=frozenset({"security", "policy"}),
    ),
    EnvironmentSpec(
        name="manufacturing",
        factory="planiverse.environments.manufacturing.mfenv:MfgEnv",
        summary="Buying and scheduling machine configurations against a demand deadline",
        instances="7 demand/capacity instances",
        deterministic=True,
        state_identity="value",
        requires=("numpy",),
        docs="docs/environments/manufacturing.md",
        tags=frozenset({"operations", "scheduling"}),
    ),
    EnvironmentSpec(
        name="urban_planning",
        factory="planiverse.environments.urban_planning.environment:UrbanPlanningEnv",
        summary="Land-use decisions balancing stakeholder objectives across a city",
        instances="2 cities",
        deterministic=True,
        state_identity="value",
        requires=("pandas", "networkx"),
        docs="docs/environments/urban-planning.md",
        tags=frozenset({"policy", "operations"}),
    ),
    EnvironmentSpec(
        name="water_network",
        factory="planiverse.environments.water_network.environment:WaterNetworkEnv",
        summary="Containing a contaminant in a water network without cutting off supply",
        instances="9 scenarios",
        deterministic=True,
        state_identity="value",
        requires=("wntr",),
        docs="docs/environments/water-distribution.md",
        tags=frozenset({"infrastructure", "solver-in-the-loop"}),
    ),
    EnvironmentSpec(
        name="power_grid",
        factory="planiverse.environments.power_grid.environment:PowerGridEnv",
        summary="Restoring grid security by substation topology after a line trips",
        instances="9 contingencies",
        deterministic=True,
        state_identity="path",
        requires=("grid2op",),
        docs="docs/environments/power-grid.md",
        tags=frozenset({"infrastructure", "solver-in-the-loop"}),
    ),
    EnvironmentSpec(
        name="crop_management",
        factory="planiverse.environments.crop_management.environment:CropEnv",
        summary="Scheduling irrigation across a growing season",
        instances="22 seasons",
        deterministic=True,
        state_identity="path",
        requires=("pcse",),
        docs="docs/environments/crop-management.md",
        tags=frozenset({"agriculture", "continuous-dynamics"}),
    ),
)

_BY_NAME = {spec.name: spec for spec in REGISTRY}


def list_environments(tag=None, available_only=False):
    """The catalogue, optionally filtered.

    `tag` selects on the tags that used to be package directories — `game`, `infrastructure`,
    and so on — but also on properties that never had a home, like `continuous-dynamics`.
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


def make(name, index=None, **kwargs):
    """Build an environment by name, optionally selecting its instance.

    ```python
    env = make("water_network", index=8)
    state, info = env.reset()
    ```
    """
    environment = get_spec(name).load()(**kwargs)
    if index is not None:
        environment.fix_index(index)
    return environment


def tags():
    """Every tag in use, so a caller can discover what it may filter on."""
    return frozenset().union(*(spec.tags for spec in REGISTRY))
