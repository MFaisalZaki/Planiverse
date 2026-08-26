"""Benchmarking Planiverse planners, with SLURM job generation.

An experiment is a directory of JSON, a sandbox is a directory of results, and the stages
between them run independently — prepare on a laptop, run on a cluster, analyse anywhere:

    planiverse-bench init --exp-dir experiment
    planiverse-bench discover --exp-dir experiment --sandbox-dir sandbox
    planiverse-bench generate --exp-dir experiment --sandbox-dir sandbox
    bash sandbox/slurm/submit_all.sh        # or: bash sandbox/run_local.sh 8
    planiverse-bench analyze --sandbox-dir sandbox
    planiverse-bench report --sandbox-dir sandbox

The design follows [pyPMTEvalToolkit](https://github.com/pyPMT/pyPMTEvalToolkit). See
`docs/benchmark.md`.
"""
from planiverse.benchmark.config import (
    ExperimentConfig, Limits, PlannerSpec, SlurmConfig, TaskSelection,
)
from planiverse.benchmark.discovery import discover, pair_up, parse_task_id, task_id
from planiverse.benchmark.runner import STATUSES, solve

__all__ = [
    "ExperimentConfig", "Limits", "PlannerSpec", "STATUSES", "SlurmConfig", "TaskSelection",
    "discover", "pair_up", "parse_task_id", "solve", "task_id",
]
