"""Planiverse environments: simulators a planner can search.

One flat namespace, one base class, and a registry that carries the differences as data:

    >>> from planiverse.environments import list_environments, make
    >>> [spec.name for spec in list_environments(tag="operational")]
    ['crop_management', 'power_grid', 'water_network']
    >>> env = make("puzznic", index=0)
"""
import sys

from planiverse.environments.base import (
    OPTIONAL_METHODS, REQUIRED_METHODS, Environment, implements_contract,
)
from planiverse.environments.registry import (
    REGISTRY, STATE_IDENTITIES, EnvironmentSpec, get_spec, list_environments, make, tags,
)

if sys.platform == "win32":
    # libsumo's Windows wheel loads only once the SUMO libraries are in the process; see the
    # module for why, and what it costs.
    from planiverse.environments._windows import preload_sumo_libraries
    preload_sumo_libraries()

__all__ = [
    "Environment", "EnvironmentSpec", "REGISTRY", "REQUIRED_METHODS", "OPTIONAL_METHODS",
    "STATE_IDENTITIES", "get_spec", "implements_contract", "list_environments", "make",
    "tags",
]
