# Vendored PDDLGym

This directory is a copy of [PDDLGym](https://github.com/tomsilver/pddlgym) **0.0.7**, the
version published on PyPI, MIT licensed (see `LICENSE.md`). It is vendored rather than
installed for one reason: the published package cannot be installed on Python 3.13.

## Why it is here

`pddlgym 0.0.7` declares `pillow <10`. Pillow published no wheels below 10 for Python 3.13,
so pip falls back to building 9.5.0 from source, and that build fails in Pillow's own
`setup.py`:

```python
def get_version():
    version_file = "src/PIL/_version.py"
    with open(version_file, encoding="utf-8") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__version__"]        # KeyError on 3.13
```

[PEP 667](https://peps.python.org/pep-0667/) made function-scope `locals()` return an
independent snapshot in 3.13, so what `exec` writes into one call is not visible to the
next. `pip download "pillow<10"` fails on 3.13 before any compiling starts, which took the
whole of Planiverse down with it — PDDL support or not.

The pin itself existed for a single line in `rendering/utils.py`: `Image.ANTIALIAS`, removed
in Pillow 10. pddlgym's own `TODO` above that line asked for the fix that is now applied
here, so the pin is no longer needed at all.

pip has no way to override another package's requirements — `--constraint` can only narrow a
range, never widen one — so patching from the outside was not an option. The choices were a
fork or a vendored copy; this repo already vendors EpiPolicy under
`epidemic_control/epipolicy/`, so it follows that precedent.

## What was changed

The copy is otherwise verbatim. Every edit is marked with a `Vendored change:` comment, so
`grep -rn "Vendored change" .` lists them all.

**1. Import paths retargeted** (16 files). The package now lives at
`planiverse.simulator.wrappers.pddlgym`, so its absolute self-imports and the `entry_point`
strings it registers with gym were rewritten to match:

| Upstream | Here |
|---|---|
| `from pddlgym.structs import ...` | `from planiverse.simulator.wrappers.pddlgym.structs import ...` |
| `import pddlgym` | `import planiverse.simulator.wrappers.pddlgym as pddlgym` |
| `entry_point='pddlgym.core:PDDLEnv'` | `entry_point='planiverse.simulator.wrappers.pddlgym.core:PDDLEnv'` |

Its relative imports (`from .utils import ...`, the majority) are untouched.

**2. The Pillow 10 resize fix**, in `rendering/utils.py` — `Image.ANTIALIAS` becomes
`Image.Resampling.LANCZOS`, exactly as pddlgym's `TODO` there specified. This is the change
that removes the need for the `pillow <10` pin, and it is why Planiverse declares
`pillow >=9.1` (`Image.Resampling` arrived in 9.1).

**3. Seven `import ipdb; ipdb.set_trace()` calls removed.** `ipdb` is not a dependency of
pddlgym or of Planiverse, so every one of these raised `ModuleNotFoundError` rather than
opening a debugger. Four of them sat directly above the `raise` they were masking, and were
simply deleted; one was a debug hook under `if verbose:` with nothing else in the block. The
two in `rendering/hanoi.py` and `rendering/minecraft.py` guarded fall-through branches with
no error of their own, so they became a `ValueError` describing the state that got there.

**4. One Python 2 implicit relative import**, `import pddl_parser` in
`downward_translate/instantiate.py`, which raises `ModuleNotFoundError` on Python 3 —
`pddl_parser` is a sibling module. It is now `from . import pddl_parser`.

Changes 3 and 4 were not planned; `tests/test_packaging.py` walks the import graph from every
environment's entry point and asserts each third-party module it reaches is a declared
dependency, and it found them. That test, plus checks that no file here still imports the
top-level `pddlgym` name and that `Image.ANTIALIAS` is gone, is what holds this arrangement
in place.

## Re-syncing

Upstream 0.0.7 is the last PyPI release. If a newer one appears, re-vendoring is:

```bash
pip download --no-deps pddlgym==<version> -d /tmp/pg && unzip -q /tmp/pg/pddlgym-*.whl -d /tmp/pg/x
# copy /tmp/pg/x/pddlgym over this directory, then reapply the four changes above
```

Then run `pytest tests/test_packaging.py tests/test_simulator.py`, which will tell you if any
of them were missed.

## Dependencies

Vendoring the code does not vendor its dependencies. `matplotlib`, `imageio` and
`scikit-image` are declared in `pyproject.toml` on PDDLGym's behalf, alongside `gym`,
`networkx` and `pillow`, which Planiverse needed anyway. `pddlgym_planners`, referenced from
`demo_planning.py`, is not a Planiverse dependency — that demo is dead weight kept only so
this copy stays diffable against upstream.
