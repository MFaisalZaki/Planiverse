#!/usr/bin/env python3
"""Install OR-Gym and what the inventory environment needs of it, for
`planiverse.environments.operational.inventory`.

OR-Gym (https://github.com/hubbs5/or-gym, MIT) declares `gym<=0.19.0`, whose package metadata
current pip refuses to read, so a plain `pip install or-gym` fails. OR-Gym's environments run on
the Gym 0.26 that python-microgrid already brings in, so the simulator is installed without its
dependency list and the list beside it with Gym's bound dropped. The two names numpy 2 removed
that OR-Gym uses (`numpy.Inf`, `numpy.product`) are restored by the inventory environment itself.

    python scripts/install_or_gym.py [--quiet]

The install lands in the environment of the interpreter that runs this, so run it with the
same Python the library was installed into. It is a Python file rather than a shell script
because Windows has no bash.
"""
import importlib
import importlib.metadata
import re
import subprocess
import sys

OR_GYM = "or-gym>=0.5"


def pip(*args):
    quiet = ["--quiet"] if "--quiet" in sys.argv[1:] else []
    subprocess.check_call([sys.executable, "-m", "pip", "install", *quiet, *args])


def main():
    pip("--no-deps", OR_GYM)
    importlib.invalidate_caches()
    # OR-Gym's own list, read from the copy just installed, with Gym's bound dropped.
    requirements = []
    for requirement in importlib.metadata.requires("or-gym") or []:
        if "extra ==" in requirement:
            continue
        name = re.split(r"[\s<>=!~;\[(]", requirement, maxsplit=1)[0]
        requirements.append("gym>=0.26" if name.lower() == "gym" else requirement)
    pip("--no-warn-conflicts", *requirements)
    check = "import or_gym; print('or_gym installed:', or_gym.__file__)"
    subprocess.check_call([sys.executable, "-c", check])


if __name__ == "__main__":
    main()
