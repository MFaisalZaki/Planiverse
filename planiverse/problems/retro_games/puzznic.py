"""Compatibility shim: this module moved to `planiverse.environments.puzznic`."""
import warnings

warnings.warn(
    "planiverse.problems is deprecated; import from planiverse.environments.puzznic instead.",
    DeprecationWarning, stacklevel=2)

from planiverse.environments.puzznic import *          # noqa: F401,F403,E402
