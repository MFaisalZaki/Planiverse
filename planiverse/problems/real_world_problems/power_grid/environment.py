"""Compatibility shim: this module moved to `planiverse.environments.power_grid.environment`."""
import warnings

warnings.warn(
    "planiverse.problems is deprecated; import from planiverse.environments.power_grid.environment instead.",
    DeprecationWarning, stacklevel=2)

from planiverse.environments.power_grid.environment import *          # noqa: F401,F403,E402
