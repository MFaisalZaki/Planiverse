"""The declared dependencies, checked against what the library actually imports.

The failure mode this guards against: a module imports something that only works because a
dependency happens to pull it in, and breaks the moment that dependency changes or leaves.
The import-closure test walks the library's own source, so it catches the gap on an install
that is missing the package, which is exactly the case it exists for.
"""
import ast
import pathlib
import re
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = REPO / "pyproject.toml"

# Import roots whose distribution goes by another name.
DISTRIBUTION_OF = {"pil": "pillow", "yaml": "pyyaml",
                   "sklearn": "scikit-learn", "cv2": "opencv-python"}

pytestmark = pytest.mark.skipif(
    not PYPROJECT.is_file(), reason="not running from a source checkout")

# One module per environment, plus the facade and the planner: whatever these reach is what
# an install has to provide.
ENTRY_POINTS = [
    "planiverse.environments.gameboy_py.puzznic",
    "planiverse.environments.gameboy.puzznic_gb",
    "planiverse.environments.gameboy_py.flipull",
    "planiverse.environments.gameboy.flipull_gb",
    "planiverse.environments.water_network.environment",
    "planiverse.environments.power_grid.environment",
    "planiverse.environments.crop_management.environment",
    "planiverse.environments.gameboy_py.super_mario_land",
    "planiverse.environments.gameboy.super_mario_land_gb",
    "planiverse.environments.network_attack.network_attack",
    "planiverse.benchmark.cli",
    "planiverse.rendering.trace",
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
    that is missing something, which is the case this is here to catch.
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


def test_the_console_script_points_at_something_callable(project):
    """The generated SLURM jobs invoke `planiverse-bench` by name, so a broken entry point
    fails on the cluster rather than here, after the whole benchmark has been submitted."""
    import importlib

    scripts = project.get("scripts", {})
    assert "planiverse-bench" in scripts, "the benchmark CLI has to be installed"
    module_name, _, attribute = scripts["planiverse-bench"].partition(":")
    assert callable(getattr(importlib.import_module(module_name), attribute))


def test_the_import_closure_is_fully_declared(declared):
    """Everything the environments import has to be in `dependencies`."""
    undeclared = {}
    for root, importer in import_closure().items():
        name = root.lower().replace("_", "-")
        if DISTRIBUTION_OF.get(name, name) not in declared:
            undeclared[root] = importer
    assert not undeclared, "imported but not declared in pyproject.toml: " + ", ".join(
        f"{root} (from {importer})" for root, importer in sorted(undeclared.items()))


def test_an_install_covers_every_environment(declared):
    """A single `pip install .` is meant to give you all of them, not a subset."""
    for requirement in ("pyboy", "nasim", "pandas", "networkx"):
        assert requirement in declared


def test_nothing_carries_a_python_version_marker(project):
    """Version markers hide a dependency from part of the supported range; a package that
    cannot install everywhere should be dropped or vendored instead."""
    for line in project["dependencies"]:
        assert "python_version" not in line, f"unexpected version marker: {line!r}"
