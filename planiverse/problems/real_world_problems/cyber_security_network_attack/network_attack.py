"""Compatibility shim: this module moved to `planiverse.environments.network_attack.network_attack`."""
import warnings

warnings.warn(
    "planiverse.problems is deprecated; import from planiverse.environments.network_attack.network_attack instead.",
    DeprecationWarning, stacklevel=2)

from planiverse.environments.network_attack.network_attack import *          # noqa: F401,F403,E402
