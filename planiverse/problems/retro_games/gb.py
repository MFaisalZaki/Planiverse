"""Compatibility shim: this module moved to `planiverse.environments.gb`."""
import warnings

warnings.warn(
    "planiverse.problems is deprecated; import from planiverse.environments.gb instead.",
    DeprecationWarning, stacklevel=2)

from planiverse.environments.gb import *          # noqa: F401,F403,E402
