"""Compatibility shim. `RealWorldProblem` is now `planiverse.environments.base.Environment`.

The two base classes were merged: the split said where an environment came from rather than
what a planner could do with it, and nothing could usefully dispatch on it. See
`planiverse/environments/base.py`.
"""
import warnings

from planiverse.environments.base import Environment

warnings.warn(
    "planiverse.problems is deprecated; import from planiverse.environments instead. "
    "`RealWorldProblem` is now `planiverse.environments.base.Environment`.",
    DeprecationWarning, stacklevel=2)

RealWorldProblem = Environment
