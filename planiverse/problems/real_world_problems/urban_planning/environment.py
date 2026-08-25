"""Compatibility shim: this module moved to `planiverse.environments.urban_planning.environment`."""
import warnings

warnings.warn(
    "planiverse.problems is deprecated; import from planiverse.environments.urban_planning.environment instead.",
    DeprecationWarning, stacklevel=2)

from planiverse.environments.urban_planning.environment import *          # noqa: F401,F403,E402
