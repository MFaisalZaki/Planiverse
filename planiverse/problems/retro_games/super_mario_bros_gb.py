"""Compatibility shim: this module moved to `planiverse.environments.super_mario_land`."""
import warnings

warnings.warn(
    "planiverse.problems is deprecated; import from planiverse.environments.super_mario_land instead.",
    DeprecationWarning, stacklevel=2)

from planiverse.environments.super_mario_land import *          # noqa: F401,F403,E402
