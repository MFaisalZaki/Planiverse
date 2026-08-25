"""Compatibility shim: this module moved to `planiverse.environments.puzznic_gb`."""
import warnings

warnings.warn(
    "planiverse.problems is deprecated; import from planiverse.environments.puzznic_gb instead.",
    DeprecationWarning, stacklevel=2)

from planiverse.environments.puzznic_gb import *          # noqa: F401,F403,E402
