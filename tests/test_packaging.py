"""The dependency extras, checked against the metadata they are declared in.

Splitting the dependencies into extras is what lets the package install on Python 3.13,
where `pddlgym 0.0.7` cannot: it requires `pillow <10`, Pillow shipped no 3.13 wheels, and
Pillow 9.5.0's `setup.py` reads its own version out of `locals()` after an `exec`, which
PEP 667 stopped working in 3.13. These tests pin the arrangement down so it does not drift
back.
"""
import pathlib
import sys
import tomllib

import pytest

PYPROJECT = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"

pytestmark = pytest.mark.skipif(
    not PYPROJECT.is_file(), reason="not running from a source checkout")

ENVIRONMENT_EXTRAS = ["retro", "manufacturing", "urban", "network-attack", "epidemic", "pddl"]


@pytest.fixture(scope="module")
def project():
    with open(PYPROJECT, "rb") as handle:
        return tomllib.load(handle)["project"]


@pytest.fixture(scope="module")
def extras(project):
    return project["optional-dependencies"]


def test_a_bare_install_pulls_in_nothing(project):
    """Puzznic and the interface need only the standard library, so an install that wants
    neither an emulator nor a solver should not be made to download JAX."""
    assert project["dependencies"] == []


def test_every_environment_has_an_extra(extras):
    for name in ENVIRONMENT_EXTRAS + ["dev", "all"]:
        assert name in extras, f"missing extra: {name}"


def test_all_is_the_union_of_the_environment_extras(extras):
    """`all` is spelled out by hand because Poetry rejects `planiverse[retro,...]`."""
    union = {requirement for name in ENVIRONMENT_EXTRAS for requirement in extras[name]}
    assert set(extras["all"]) == union, (
        "the `all` extra drifted from the extras it is supposed to cover:\n"
        f"  missing: {sorted(union - set(extras['all']))}\n"
        f"  extra:   {sorted(set(extras['all']) - union)}")


def test_dev_is_not_swept_into_all(extras):
    """`all` means every environment, not every dependency: pytest is not one."""
    assert not set(extras["dev"]) & set(extras["all"])


def test_the_pddl_extra_is_held_below_python_313(extras):
    """Without the marker, `pip install` dies building Pillow 9.5.0 on 3.13."""
    declared = [line for line in extras["pddl"] if line.startswith("pddlgym")]
    assert declared, "the pddl extra no longer declares pddlgym"
    for line in declared:
        assert "python_version < '3.13'" in line, \
            f"{line!r} would be resolved on 3.13, where pddlgym cannot install"


def test_gym_is_not_held_back_with_it(extras):
    """gym installs on every supported Python, and `Simulator` uses it to recognise a
    PDDLGym environment, so only pddlgym itself carries the marker."""
    declared = [line for line in extras["pddl"] if line.startswith("gym")]
    assert declared and not any("python_version" in line for line in declared)


def test_pddlgym_is_only_ever_declared_with_that_marker(extras):
    for name, requirements in extras.items():
        for line in requirements:
            if line.startswith("pddlgym"):
                assert "python_version < '3.13'" in line, \
                    f"the {name!r} extra declares {line!r} without the 3.13 marker"


def test_pddlgym_is_absent_exactly_when_it_cannot_be_installed():
    """The marker's whole point: the wrapper is unavailable on 3.13, not broken."""
    try:
        import pddlgym  # noqa: F401
        available = True
    except ImportError:
        available = False
    if sys.version_info >= (3, 13):
        assert not available, "pddlgym should not be installable on 3.13"


def test_the_simulator_facade_imports_without_pddlgym():
    """Wrapping a native environment needs no solver, so a missing pddlgym must not stop
    the module importing — it only rules out the PDDLGym branch."""
    from planiverse.problems.retro_games.puzznic import PuzznicGame
    from planiverse.simulator.simulator import Simulator

    env = PuzznicGame()
    env.fix_index(0)
    assert Simulator(env).simulator is env
