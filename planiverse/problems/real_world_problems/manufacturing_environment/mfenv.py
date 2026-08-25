"""Compatibility shim: this module moved to `planiverse.environments.manufacturing.mfenv`."""
import warnings

warnings.warn(
    "planiverse.problems is deprecated; import from planiverse.environments.manufacturing.mfenv instead.",
    DeprecationWarning, stacklevel=2)

from planiverse.environments.manufacturing.mfenv import *          # noqa: F401,F403,E402
