#!/usr/bin/env python3
"""Install Flatland and what the railway environment needs of it, for
`planiverse.environments.operational.railway`.

Flatland (https://github.com/flatland-association/flatland-rl, MIT) declares `numpy<2`, and
upper bounds on matplotlib, networkx, contourpy and rpds-py, all of which the rest of this
library's dependencies need newer versions of; numpy 1.x also has no wheels for Python 3.13.
So the simulator is installed without its dependency list, and the list beside it with those
five bounds dropped. The railway environment's tests run Flatland on numpy 2.

    python scripts/install_flatland.py [--quiet]

The install lands in the environment of the interpreter that runs this, so run it with the
same Python the library was installed into. It is a Python file rather than a shell script
because Windows has no bash.
"""
import importlib
import importlib.metadata
import re
import subprocess
import sys

FLATLAND = "flatland-rl>=4.3"

#: Flatland's bounds that the library's other dependencies cannot meet; the names stay.
UNBOUNDED = {"numpy", "matplotlib", "networkx", "contourpy", "rpds-py"}


def pip(*args):
    quiet = ["--quiet"] if "--quiet" in sys.argv[1:] else []
    subprocess.check_call([sys.executable, "-m", "pip", "install", *quiet, *args])


def main():
    pip("--no-deps", FLATLAND)
    importlib.invalidate_caches()
    # Flatland's own list, read from the copy just installed so it tracks the version, with
    # the bounds above dropped and the extras' lines left out.
    requirements = []
    for requirement in importlib.metadata.requires("flatland-rl") or []:
        if "extra ==" in requirement:
            continue
        name = re.split(r"[\s<>=!~;\[(]", requirement, maxsplit=1)[0]
        requirements.append(name if name.lower().replace("_", "-") in UNBOUNDED else requirement)
    # pip would otherwise report Flatland's bounds as conflicts with what is installed, which
    # is the point of this script.
    pip("--no-warn-conflicts", *requirements)
    check = ("from flatland.envs.rail_env import RailEnv; import flatland; "
             "print('flatland installed:', flatland.__file__)")
    subprocess.check_call([sys.executable, "-c", check])


if __name__ == "__main__":
    main()
