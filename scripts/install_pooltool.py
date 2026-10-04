#!/usr/bin/env python3
"""Install pooltool and what the billiards environment needs of it, for
`planiverse.environments.games.billiards`.

pooltool (https://github.com/ekiefl/pooltool, Apache-2.0) declares, on Linux and Windows, a
dependency on a Panda3D development build (`panda3d==1.11.0.dev3702`) that PyPI does not carry,
so a plain `pip install pooltool-billiards` fails there with "No matching distribution found for
panda3d"; on macOS it takes the released 1.10. So the simulator is installed without its
dependency list and the real ones beside it, with the released Panda3D 1.10 from PyPI on every
platform. Panda3D is pooltool's renderer; the physics the environment runs does not touch it.
See THIRD-PARTY-NOTICES.md.

    python scripts/install_pooltool.py [--quiet]

The install lands in the environment of the interpreter that runs this, so run it with the
same Python the library was installed into. It is a Python file rather than a shell script
because Windows has no bash.
"""
import importlib
import importlib.metadata
import re
import subprocess
import sys

POOLTOOL = "pooltool-billiards>=0.6"
PANDA3D = "panda3d>=1.10.13,<1.11"       # the released line, which has wheels everywhere


def pip(*args):
    quiet = ["--quiet"] if "--quiet" in sys.argv[1:] else []
    subprocess.check_call([sys.executable, "-m", "pip", "install", *quiet, *args])


def main():
    pip("--no-deps", POOLTOOL)
    importlib.invalidate_caches()
    # pooltool's own list, read from the copy just installed so it tracks the version, with
    # every Panda3D line replaced by the released one and the extras' lines left out.
    requirements = [PANDA3D]
    for requirement in importlib.metadata.requires("pooltool-billiards") or []:
        name = re.split(r"[\s<>=!~;\[(]", requirement, maxsplit=1)[0]
        if name.lower() == "panda3d" or "extra ==" in requirement:
            continue
        requirements.append(requirement)
    # pip would otherwise report pooltool's Panda3D pin as a conflict with what was just
    # installed, which is the point of this script.
    pip("--no-warn-conflicts", *requirements)
    check = ("import pooltool as pt; s = pt.System.example(); pt.simulate(s, inplace=True); "
             "print('pooltool', pt.__version__, 'installed:', pt.__file__)")
    subprocess.check_call([sys.executable, "-c", check])


if __name__ == "__main__":
    main()
