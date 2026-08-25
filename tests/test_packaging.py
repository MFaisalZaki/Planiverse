"""The declared dependencies, checked against what the library actually imports.

Two things went wrong before these existed. `gym` was imported by the simulator facade and
the epidemic environment but never declared, so it only ever worked because pddlgym happened
to pull it in. And `pddlgym 0.0.7` requires `pillow <10`, which has no Python 3.13 wheels and
whose 9.5.0 sdist cannot build there — Pillow's `setup.py` reads its own version out of
`locals()` after an `exec`, which PEP 667 stopped working in 3.13 — so `pip install .` died
on 3.13 for everyone, whether or not they wanted PDDL support.
"""
import ast
import pathlib
import re
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = REPO / "pyproject.toml"

pytestmark = pytest.mark.skipif(
    not PYPROJECT.is_file(), reason="not running from a source checkout")

# One module per environment, plus the facade and the planner: whatever these reach is what
# an install has to provide.
ENTRY_POINTS = [
    "planiverse.problems.retro_games.puzznic",
    "planiverse.problems.retro_games.puzznic_gb",
    "planiverse.problems.retro_games.super_mario_bros_gb",
    "planiverse.problems.real_world_problems.epidemic_control.environment",
    "planiverse.problems.real_world_problems.urban_planning.environment",
    "planiverse.problems.real_world_problems.cyber_security_network_attack.network_attack",
    "planiverse.problems.real_world_problems.manufacturing_environment.mfenv",
    "planiverse.simulator.simulator",
    "planiverse.planners.super_mario_planner_gb",
]


@pytest.fixture(scope="module")
def project():
    import tomllib

    with open(PYPROJECT, "rb") as handle:
        return tomllib.load(handle)["project"]


@pytest.fixture(scope="module")
def declared(project):
    """The distribution names in `dependencies`, normalised."""
    return {re.split(r"[\s<>=!~@\[(;]", line.strip(), maxsplit=1)[0].lower().replace("_", "-")
            for line in project["dependencies"]}


def module_path(module):
    parts = module.split(".")
    for candidate in (REPO.joinpath(*parts).with_suffix(".py"),
                      REPO.joinpath(*parts, "__init__.py")):
        if candidate.is_file():
            return candidate
    return None


def imported_names(path):
    """Every module named by a module-level import, relative ones resolved to absolute."""
    names = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package = ".".join(path.parent.relative_to(REPO).parts)
                names.append(f"{package}.{node.module}" if node.module else package)
            elif node.module:
                names.append(node.module)
    return names


def import_closure():
    """Third-party roots reachable from the entry points, and who pulls each one in.

    Walks the library's own modules rather than importing them, so it works on an install
    that is missing something — which is the case this is here to catch.
    """
    third_party, seen, queue = {}, set(), list(ENTRY_POINTS)
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        seen.add(module)
        path = module_path(module)
        if path is None:
            continue
        for name in imported_names(path):
            if name.startswith("planiverse"):
                # `from pkg.mod import Thing`: Thing may be a module or a name in pkg.mod.
                queue += [name, name.rsplit(".", 1)[0]]
            else:
                root = name.split(".")[0]
                if root not in sys.stdlib_module_names:
                    third_party.setdefault(root, module)
    return third_party


def test_every_entry_point_exists():
    for module in ENTRY_POINTS:
        assert module_path(module) is not None, f"{module} moved or was renamed"


def test_the_import_closure_is_fully_declared(declared):
    """Everything the environments import has to be in `dependencies`.

    `gym` was missing from it for a long time and only worked because pddlgym happened to
    depend on it, which broke the moment pddlgym stopped being installed.
    """
    undeclared = {root: importer for root, importer in import_closure().items()
                  if root.lower().replace("_", "-") not in declared}
    assert not undeclared, "imported but not declared in pyproject.toml: " + ", ".join(
        f"{root} (from {importer})" for root, importer in sorted(undeclared.items()))


def test_an_install_covers_every_environment(declared):
    """A single `pip install .` is meant to give you all of them, not a subset."""
    for requirement in ("pyboy", "nasim", "numba", "pandas", "networkx", "gym"):
        assert requirement in declared


def test_pddlgym_is_declared_only_below_python_313(project):
    """Without the marker, `pip install .` dies building Pillow 9.5.0 on 3.13."""
    declared = [line for line in project["dependencies"] if line.startswith("pddlgym")]
    assert declared, "pddlgym is no longer declared"
    for line in declared:
        assert "python_version < '3.13'" in line, \
            f"{line!r} would be resolved on 3.13, where pddlgym cannot install"


def test_nothing_else_carries_that_marker(project):
    """Only pddlgym is held back; everything else installs on every supported Python."""
    for line in project["dependencies"]:
        if "python_version" in line:
            assert line.startswith("pddlgym"), f"unexpected version marker: {line!r}"


def test_pddlgym_is_absent_exactly_where_it_cannot_be_installed():
    """The marker's whole point: on 3.13 the wrapper is unavailable, not broken."""
    try:
        import pddlgym  # noqa: F401
        available = True
    except ImportError:
        available = False
    if sys.version_info >= (3, 13):
        assert not available, "pddlgym should not be installable on 3.13"


def test_the_simulator_facade_imports_without_pddlgym():
    """Wrapping a native environment needs no solver, so a missing pddlgym must not stop the
    module importing — it only rules out the PDDLGym branch."""
    from planiverse.problems.retro_games.puzznic import PuzznicGame
    from planiverse.simulator.simulator import Simulator

    env = PuzznicGame()
    env.fix_index(0)
    assert Simulator(env).simulator is env
