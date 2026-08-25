"""Compatibility shim: this module moved to `planiverse.environments.base`."""
import warnings

warnings.warn(
    "planiverse.problems is deprecated; import from planiverse.environments.base instead.",
    DeprecationWarning, stacklevel=2)

from planiverse.environments.base import *          # noqa: F401,F403,E402
