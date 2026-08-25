"""Compatibility shim: this module moved to `planiverse.environments.epidemic_control.environment`."""
import warnings

warnings.warn(
    "planiverse.problems is deprecated; import from planiverse.environments.epidemic_control.environment instead.",
    DeprecationWarning, stacklevel=2)

from planiverse.environments.epidemic_control.environment import *          # noqa: F401,F403,E402
