"""Compatibility shim: this module moved to `planiverse.environments.water_network.environment`."""
import warnings

warnings.warn(
    "planiverse.problems is deprecated; import from planiverse.environments.water_network.environment instead.",
    DeprecationWarning, stacklevel=2)

from planiverse.environments.water_network.environment import *          # noqa: F401,F403,E402
