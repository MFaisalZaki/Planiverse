#!/usr/bin/env python3
"""Say which environments the install in front of you can run, and which it cannot and why.

    python tools/check_install.py
    python tools/check_install.py --expect-missing micropolis,factory,airspace

Every registered environment is listed with the third-party modules it needs and whether each
imports here. With `--expect-missing`, the exit status is non-zero if any environment *other*
than those named is missing a dependency, which is how the CI workflow checks that an install
on each platform covers what the README says it covers.
"""
import argparse
import importlib
import sys

from planiverse.environments import REGISTRY


def missing_modules(spec):
    """The modules in `spec.requires` that do not import, with the error for each."""
    missing = {}
    for module_name in spec.requires:
        try:
            importlib.import_module(module_name)
        except Exception as exc:          # an ImportError, or a broken binary dependency
            missing[module_name] = f"{type(exc).__name__}: {exc}"
    return missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--expect-missing", default="", metavar="NAMES",
                        help="comma-separated environments allowed to be unavailable here")
    args = parser.parse_args(argv)
    expected = {name for name in args.expect_missing.split(",") if name}

    unexpected = []
    width = max(len(spec.name) for spec in REGISTRY)
    for spec in sorted(REGISTRY, key=lambda spec: spec.name):
        missing = missing_modules(spec)
        if not missing:
            needs = ", ".join(spec.requires) or "nothing beyond the library"
            print(f"{spec.name:{width}}  available   ({needs})")
            continue
        print(f"{spec.name:{width}}  MISSING     " + "; ".join(
            f"{module}: {error}" for module, error in missing.items()))
        if spec.name not in expected:
            unexpected.append(spec.name)

    available = sum(1 for spec in REGISTRY if not missing_modules(spec))
    print(f"\n{available} of {len(REGISTRY)} environments available on {sys.platform}, "
          f"Python {sys.version.split()[0]}")
    if unexpected:
        print("not expected to be missing here: " + ", ".join(unexpected), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
