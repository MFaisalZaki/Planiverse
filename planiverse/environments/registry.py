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
lazily, so listing the catalogue costs nothing even though half of it needs PyBoy, grid2op or
numba to run.
"""
import importlib
import os
from dataclasses import dataclass, field

#: How a state is identified, which is the thing that decides whether search can branch.
#:
#: - `value`:    the state carries its own contents; expanding is pure.
#: - `path`:     the state is the decision sequence, replayed on demand. Sound only because
#:               the simulator is deterministic.
#: - `snapshot`: the state carries a serialised simulator image (a Game Boy save-state).
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
    #: Constructor keyword arguments, as `(name, value)` pairs, for the environments whose
    #: `__init__` takes a required argument. Without these `make(name)` works for most of the
    #: catalogue and raises a `TypeError` for the rest, which makes "build every registered
    #: environment" (what the benchmark harness does) impossible to write generically.
    #: Pairs rather than a dict so the spec stays hashable.
    defaults: tuple = ()
    #: For a `needs_rom` environment, the environment variable holding the path to the
    #: cartridge, and the constructor argument to pass it as. The file is copyrighted and
    #: cannot ship, so the path can only come from the user. Without somewhere to put it
    #: these environments cannot be constructed by name at all, which makes them invisible to
    #: anything generic (the benchmark harness, most obviously).
    rom_variable: str = ""
    rom_argument: str = "romfile"

    def load(self):
        """Import and return the class."""
        module_name, class_name = self.factory.split(":")
        return getattr(importlib.import_module(module_name), class_name)

    def build(self, **kwargs):
        """Construct the environment, with `defaults` filled in and `kwargs` winning.

        For a ROM environment the cartridge path is read from `rom_variable` unless the
        caller passes one.
        """
        arguments = dict(self.defaults)
        if self.needs_rom and self.rom_argument not in kwargs:
            rom = self.rom_path()
            if rom is None:
                raise FileNotFoundError(
                    f"{self.name} needs a cartridge, which cannot ship with this repo. "
                    f"Point {self.rom_variable} at one, or pass "
                    f"{self.rom_argument}= yourself.")
            arguments[self.rom_argument] = rom
        return self.load()(**{**arguments, **kwargs})

    def rom_path(self):
        """The cartridge for this environment, or None. Set `rom_variable` to point at one."""
        if not self.rom_variable:
            return None
        path = os.environ.get(self.rom_variable)
        return path if path and os.path.isfile(path) else None

    def rom_flag(self):
        """The short name this environment's cartridge goes by on a command line.

        Derived from `rom_variable` rather than stored, so the flag and the variable are the
        same name in two spellings and cannot drift apart:
        `PLANIVERSE_PUZZNIC_ROM` is `--rom-puzznic`.
        """
        if not self.rom_variable:
            return None
        name = self.rom_variable
        for prefix in ("PLANIVERSE_",):
            if name.startswith(prefix):
                name = name[len(prefix):]
        if name.endswith("_ROM"):
            name = name[:-len("_ROM")]
        return name.lower().replace("_", "-")

    def available(self):
        """Can this environment run here (dependencies importable, and a ROM if it needs one)?"""
        for module_name in self.requires:
            try:
                importlib.import_module(module_name)
            except ImportError:
                return False
        return not self.needs_rom or self.rom_path() is not None


REGISTRY = (
    EnvironmentSpec(
        name="puzznic",
        factory="planiverse.environments.gameboy_py.puzznic:PuzznicGame",
        summary="Sliding block puzzle, re-implemented in pure Python",
        instances="128 levels",
        deterministic=True,
        state_identity="value",
        docs="docs/environments/puzznic.md",
        tags=frozenset({"game", "puzzle", "dependency-free"}),
    ),
    EnvironmentSpec(
        name="puzznic_gb",
        factory="planiverse.environments.gameboy.puzznic_gb:PuzznicGBEnv",
        summary="Puzznic played on the Game Boy cartridge, through PyBoy",
        instances="128 rounds",
        deterministic=True,
        state_identity="snapshot",
        requires=("pyboy",),
        needs_rom=True,
        rom_variable="PLANIVERSE_PUZZNIC_ROM",
        docs="docs/environments/puzznic-gb.md",
        tags=frozenset({"game", "puzzle", "emulator"}),
    ),
    EnvironmentSpec(
        name="flipull",
        factory="planiverse.environments.gameboy_py.flipull:FlipullGame",
        summary="Flipull-like throwing puzzle, re-implemented in pure Python",
        instances="32 stages, matching the cartridge's sizes and CLEAR targets",
        deterministic=True,
        state_identity="value",
        docs="docs/environments/flipull.md",
        tags=frozenset({"game", "puzzle", "dependency-free"}),
    ),
    EnvironmentSpec(
        name="flipull_gb",
        factory="planiverse.environments.gameboy.flipull_gb:FlipullGBEnv",
        summary="Flipull (Taito's Plotting) on the Game Boy, through PyBoy",
        instances="32 stages",
        deterministic=True,
        state_identity="snapshot",
        requires=("pyboy",),
        needs_rom=True,
        rom_variable="PLANIVERSE_FLIPULL_ROM",
        docs="docs/environments/flipull-gb.md",
        tags=frozenset({"game", "puzzle", "emulator"}),
    ),
    EnvironmentSpec(
        name="amazing_tater",
        factory="planiverse.environments.gameboy_py.amazing_tater:AmazingTaterGame",
        summary="Amazing Tater's blocks, pits and turnstiles, re-implemented in pure Python",
        instances="105 rooms",
        deterministic=True,
        state_identity="value",
        docs="docs/environments/amazing-tater.md",
        tags=frozenset({"game", "puzzle", "dependency-free"}),
    ),
    EnvironmentSpec(
        name="amazing_tater_gb",
        factory="planiverse.environments.gameboy.amazing_tater_gb:AmazingTaterGBEnv",
        summary="Amazing Tater played on the Game Boy cartridge, through PyBoy",
        instances="105 rooms",
        deterministic=True,
        state_identity="snapshot",
        requires=("pyboy",),
        needs_rom=True,
        rom_variable="PLANIVERSE_AMAZING_TATER_ROM",
        docs="docs/environments/amazing-tater-gb.md",
        tags=frozenset({"game", "puzzle", "emulator"}),
    ),
    EnvironmentSpec(
        name="lolo",
        factory="planiverse.environments.gameboy_py.lolo:LoloGame",
        summary="Adventures of Lolo's block-and-heart puzzle, re-implemented in pure Python",
        instances="163 rooms",
        deterministic=True,
        state_identity="value",
        docs="docs/environments/lolo.md",
        tags=frozenset({"game", "puzzle", "dependency-free"}),
    ),
    EnvironmentSpec(
        name="lolo_gb",
        factory="planiverse.environments.gameboy.lolo_gb:LoloGBEnv",
        summary="Adventures of Lolo played on the Game Boy cartridge, through PyBoy",
        instances="163 rooms",
        deterministic=True,
        state_identity="snapshot",
        requires=("pyboy",),
        needs_rom=True,
        rom_variable="PLANIVERSE_LOLO_ROM",
        docs="docs/environments/lolo-gb.md",
        tags=frozenset({"game", "puzzle", "emulator"}),
    ),
    EnvironmentSpec(
        name="super_mario_land",
        factory="planiverse.environments.gameboy_py.super_mario_land:SuperMarioLandGame",
        summary="Super Mario Land-style platformer with approximated physics, in pure Python",
        instances="8 levels",
        deterministic=True,
        state_identity="value",
        docs="docs/environments/super-mario-land.md",
        tags=frozenset({"game", "platformer", "dependency-free"}),
    ),
    EnvironmentSpec(
        name="super_mario_land_gb",
        factory="planiverse.environments.gameboy.super_mario_land_gb:SuperMarioLandGBEnv",
        summary="Super Mario Land played on the Game Boy cartridge, through PyBoy",
        instances="12 levels",
        deterministic=True,
        state_identity="snapshot",
        requires=("pyboy",),
        needs_rom=True,
        rom_variable="PLANIVERSE_SUPER_MARIO_LAND_ROM",
        docs="docs/environments/super-mario-land-gb.md",
        tags=frozenset({"game", "platformer", "emulator"}),
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
        name="water_network",
        factory="planiverse.environments.water_network.environment:WaterNetworkEnv",
        summary="Containing a contaminant in a water network without cutting off supply",
        instances="9 scenarios",
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
        instances="9 contingencies",
        deterministic=True,
        state_identity="path",
        requires=("grid2op",),
        docs="docs/environments/power-grid.md",
        tags=frozenset({"operational", "infrastructure", "solver-in-the-loop"}),
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


def make(name, index=None, **kwargs):
    """Build an environment by name, optionally selecting its instance.

    ```python
    env = make("water_network", index=8)
    state, info = env.reset()
    ```
    """
    environment = get_spec(name).build(**kwargs)
    if index is not None:
        environment.fix_index(index)
    return environment


def tags():
    """Every tag in use, so a caller can discover what it may filter on."""
    return frozenset().union(*(spec.tags for spec in REGISTRY))
