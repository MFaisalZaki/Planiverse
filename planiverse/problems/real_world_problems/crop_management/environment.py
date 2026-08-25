"""Compatibility shim: this module moved to `planiverse.environments.crop_management.environment`."""
import warnings

warnings.warn(
    "planiverse.problems is deprecated; import from planiverse.environments.crop_management.environment instead.",
    DeprecationWarning, stacklevel=2)

from planiverse.environments.crop_management.environment import *          # noqa: F401,F403,E402
