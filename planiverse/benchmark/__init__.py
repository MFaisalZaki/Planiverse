"""`planiverse-bench`: the library's evaluation protocol, as code.

    planiverse-bench generate [--sandbox-dir sandbox] [--partition P] [--qos Q] [--account A]
    bash sandbox/slurm/submit_all.sh             # or: bash sandbox/run_local.sh 8
    planiverse-bench report   [--sandbox-dir sandbox]

The harness is pyPMTEvalToolkit's (https://github.com/pyPMT/pyPMTEvalToolkit): its sandbox
layout, its command files and SLURM job arrays, its local runner, and its `analyze` and `report`
stages, which read the result files back into a CSV, coverage and outcome tables, IPC scores
and survival plots. What is Planiverse's own is the one stage the toolkit cannot do for a
simulator, `solve`: one planner on one instance under the limits, written out in the toolkit's
result format whatever happens. The toolkit's own `solve` parses PDDL and runs a Unified
Planning engine; here there is a simulator and a planner object instead.

`generate` asks every registered environment how many instances it has and writes one command
per (planner, instance, seed) for every planner the library has, through the toolkit's writers;
`--reference` keeps it to the four reference configurations. A seeded planner runs once per
seed, each under its own tag (`fsx-s3`), so the toolkit's tables list the seeds side by side.

The protocol is the constants below. There is no configuration file: a run that changed a limit
would not be the same experiment, and one that did not has nothing to configure.
"""
import argparse
import inspect
import json
import os
import platform
import shlex
import signal
import sys
import time
import traceback

try:
    import resource
except ImportError:     # Windows: no address-space cap, the Budget alone bounds a run
    resource = None

from planiverse.benchmark.measures import MEASURES
from planiverse.benchmark.planners import PLANNERS, REFERENCE
from planiverse.environments import REGISTRY, get_spec
from planiverse.planners.width import Budget

#: Per run: 30 minutes of wall clock, 8 GB of address space, 500,000 expansions.
LIMITS = {"seconds": 1800, "bytes": 8 * 1024 ** 3, "expansions": 500_000}

#: `REFERENCE` and `PLANNERS` (every configuration, by tag) are in `planners.py`. This is a
#: planning library: nothing there takes a reward, and nothing learns before it plans.

#: The seeds a planner whose constructor takes one runs under. Every (instance, seed) is a
#: full run under the same limits; the environments are deterministic, so the seed is the only
#: source of variance. The three width planners have no seed and run once.
SEEDS = range(5)

#: The most elements a job array may have. SLURM caps an array at the site's `MaxArraySize`,
#: commonly 1001 (elements 0 to 1000) and often less, so a group with more instances than
#: this is submitted as several arrays, each over its own slice of the command file.
MAX_ARRAY = 1000

#: The toolkit's tasks have a suite and a track. The suite is this library, and every
#: environment is a classical problem in the toolkit's sense (deterministic, one agent, a goal
#: to reach), so its tables, which know the classical, numeric and temporal tracks, report
#: them all under the first; the per-domain tables give the environments one by one.
SUITE, TRACK = "planiverse", "classical"

#: How a run can end, in the toolkit's words, which its `analyze` and `report` count. A plan
#: that does not replay is SOLVED with `validated` false, which the toolkit keeps out of
#: coverage and lists on its own. A search that stopped on its own, or at the expansion cap,
#: is EXHAUSTED; which of the two is in the result's `metrics`.
SOLVED, EXHAUSTED, TIMEOUT, MEMOUT, UNSUPPORTED, ERROR = (
    "SOLVED", "EXHAUSTED", "TIMEOUT", "MEMOUT", "UNSUPPORTED", "ERROR")


def _seeds(tag):
    """The seeds a planner runs under: SEEDS if its constructor takes one, else a single None."""
    return list(SEEDS) if "seed" in inspect.signature(PLANNERS[tag][0]).parameters else [None]


def _group(tag, seed):
    """The toolkit's planner tag for one planner under one seed: its results directory."""
    return tag if seed is None else f"{tag}-s{seed}"


def _task(environment, index):
    """An instance as the toolkit names it."""
    from pypmt_eval_toolkit.tasks import Task

    return Task(suite=SUITE, domain=environment, instance=str(index), domain_file="",
                problem_file="", track=TRACK)


def _experiment(partition=None, qos=None, account=None, parallel=50):
    """The protocol's limits as a toolkit experiment, which its SLURM writer reads."""
    from pypmt_eval_toolkit.config import Experiment

    slurm = {"partition": partition, "qos": qos, "account": account,
             "max-parallel-jobs": parallel, "max-array-size": MAX_ARRAY}
    return Experiment(name="planiverse", path="", planners=[], details={"cfgs": {
        "timelimit": LIMITS["seconds"], "memorylimit": LIMITS["bytes"] // 1024 ** 2,
        "slurm": slurm}})


def generate(sandbox, partition=None, qos=None, account=None, parallel=50, reference=False):
    """Count every environment's instances, then write the commands and the arrays to run them."""
    from pypmt_eval_toolkit.generator import (
        Sandbox, _write_command_files, _write_local_runner, _write_slurm_arrays,
    )
    from pypmt_eval_toolkit.tasks import summarize

    planners = REFERENCE if reference else PLANNERS
    layout = Sandbox(sandbox)
    layout.create()
    counts = {}
    for spec in REGISTRY:
        try:
            env = spec.build()
        except Exception as exc:
            print(f"  {spec.name:22} skipped: {exc}", file=sys.stderr)
            continue
        # The registry says how many instances there are in words; the environment says so
        # in code, by refusing the first index it does not have.
        for count in range(10_000):
            try:
                env.set_index(count)
            except Exception:
                break
        getattr(env, "close", lambda: None)()
        counts[spec.name] = count
        print(f"  {spec.name:22} {count:>4} instances")

    tasks = [_task(env, index) for env, n in counts.items() for index in range(n)]
    # Seed 0 of every planner first, so it is the first to finish.
    groups = {_group(tag, seed): (tag, seed) for seed in [None, *SEEDS] for tag in planners
              if seed in _seeds(tag)}
    # `expected` is what the toolkit's `analyze` counts a run against, so a job that never ran
    # is MISSING rather than silently absent.
    with open(layout.tasks_file, "w") as handle:
        json.dump({"summary": summarize(tasks),
                   "expected": {group: [task.task_id for task in tasks] for group in groups},
                   "tasks": [task.to_dict() for task in tasks]}, handle, indent=1)
    commands = {
        group: [f"{shlex.quote(sys.executable)} -m planiverse.benchmark solve "
                f"--sandbox-dir {shlex.quote(layout.root)} {tag} {task.domain}@{task.instance}"
                + ("" if seed is None else f" --seed {seed}") for task in tasks]
        for group, (tag, seed) in groups.items()}
    _write_command_files(layout, commands)
    arrays = _write_slurm_arrays(layout, _experiment(partition, qos, account, parallel), commands)
    _write_local_runner(layout, commands, 8)
    print(f"{len(tasks)} instances x {len(groups)} groups "
          f"({', '.join(f'{tag} x{len(_seeds(tag))}' for tag in planners)}) "
          f"= {len(tasks) * len(groups)} runs in {len(arrays)} arrays of at most {MAX_ARRAY}\n"
          f"  submit:  bash {layout.slurm_dir}/submit_all.sh\n"
          f"  or here: bash {layout.root}/run_local.sh 8")


#: Whether this platform has the interval timer the per-run alarm is set on.
ALARM = hasattr(signal, "setitimer") and hasattr(signal, "SIGALRM")


def _alarm(*_):
    raise TimeoutError


def solve(sandbox, tag, task, seed=None):
    """Run one planner on one instance under the limits and write down what happened.

    The result is written even when the run fails, which is the point: a benchmark that only
    records its successes cannot say that a planner crashed on a third of the set. It is the
    toolkit's result format, so the toolkit's `analyze` and `report` read it as their own.
    """
    name, index = task.rsplit("@", 1)
    seed = _seeds(tag)[0] if seed is None else seed   # a seeded planner run by hand gets its first
    instance = _task(name, index)
    record = {"task": {"task-id": instance.task_id, "suite": SUITE, "domain": name,
                       "instance": index, "track": TRACK, "ipc": None},
              "planner": {"tag": _group(tag, seed), "engine": tag, "seed": seed,
                          "params": PLANNERS[tag][1]},
              "limits": {"time-seconds": LIMITS["seconds"],
                         "memory-mb": LIMITS["bytes"] // 1024 ** 2,
                         "expansions": LIMITS["expansions"]},
              "run": {"host": platform.node(), "pid": os.getpid(), "started": time.time(),
                      "slurm-job-id": os.environ.get("SLURM_JOB_ID"),
                      "slurm-array-task-id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                      "python": platform.python_version()},
              "status": ERROR, "timings": {}, "plan": None, "metrics": {}, "logs": []}
    _mark(sandbox, record)
    # An address-space cap turns an overrun into a MemoryError the run can record, instead of
    # an OOM kill that leaves no file. macOS refuses the call and Windows has no `resource`
    # module; the cap is for the Linux cluster.
    if resource is not None:
        try:
            resource.setrlimit(resource.RLIMIT_AS,
                               (LIMITS["bytes"], resource.getrlimit(resource.RLIMIT_AS)[1]))
        except (ValueError, OSError):
            pass
    env = None
    try:
        try:
            env = get_spec(name).build()
            env.set_index(int(index))
        except Exception as exc:
            return _write(sandbox, record, UNSUPPORTED, note=f"{type(exc).__name__}: {exc}")
        cls, params = PLANNERS[tag]
        if seed is not None:
            params = {**params, "seed": seed}
        if "progress" in inspect.signature(cls).parameters:
            params = {**params, "progress": MEASURES.get(name)}
        planner = cls(**params)
        # The Budget is checked between expansions; the alarm catches the one expansion that
        # itself overruns (a power-grid step can take twenty seconds), so the run records a
        # TIMEOUT rather than being killed by SLURM and leaving no file. Windows has no
        # interval timer, so there the Budget alone bounds the run.
        if ALARM:
            signal.signal(signal.SIGALRM, _alarm)
            signal.setitimer(signal.ITIMER_REAL, 1.02 * LIMITS["seconds"])
        started = time.perf_counter()
        try:
            out = planner.solve(env, Budget(max_expansions=LIMITS["expansions"],
                                            max_seconds=LIMITS["seconds"]))
        except TimeoutError:
            return _write(sandbox, record, TIMEOUT, time.perf_counter() - started)
        except MemoryError:
            return _write(sandbox, record, MEMOUT, time.perf_counter() - started)
        finally:
            if ALARM:
                signal.setitimer(signal.ITIMER_REAL, 0)
        elapsed = time.perf_counter() - started
        record["plan"] = None if out.plan is None else [str(action) for action in out.plan]
        record["metrics"].update({
            "search-status": out.status, "width": out.width,
            "expansions": out.statistics.expansions, "generated": out.statistics.generated,
            "widths-tried": list(out.statistics.widths_tried),
            # Zero for the width planners; the sampling planners count what they ran before
            # they succeeded.
            "rollouts": out.statistics.rollouts, "episodes": out.statistics.episodes})
        if out.solved:
            record["metrics"]["plan-length"] = len(out.plan)
            try:
                record["metrics"]["validated"] = bool(env.validate(out.plan))
            except Exception:
                record["metrics"]["validated"] = False
            status = SOLVED
        elif out.status in ("failed", "exhausted", "step_limit", "dead_end"):
            # The search stopped on its own: nothing left, or FSX at its step cap or a dead end.
            record["metrics"]["stopped"] = "on its own"
            status = EXHAUSTED
        elif elapsed >= 0.95 * LIMITS["seconds"]:
            status = TIMEOUT
        else:
            # The expansion cap, or an iterated search whose per-width allowances ran out.
            record["metrics"]["stopped"] = "at the expansion cap"
            status = EXHAUSTED
        return _write(sandbox, record, status, elapsed)
    except MemoryError:
        return _write(sandbox, record, MEMOUT)
    except Exception as exc:
        record["traceback"] = traceback.format_exc()
        return _write(sandbox, record, ERROR, note=f"{type(exc).__name__}: {exc}")
    finally:
        getattr(env, "close", lambda: None)()


def _path(sandbox, record, suffix):
    from pypmt_eval_toolkit.generator import Sandbox
    from pypmt_eval_toolkit.tasks import Task

    slug = Task(suite=SUITE, domain=record["task"]["domain"],
                instance=record["task"]["instance"], domain_file="", problem_file="",
                track=TRACK).slug
    return Sandbox(sandbox).result_file(record["planner"]["tag"], slug)[:-len(".json")] + suffix


def _mark(sandbox, record):
    """The toolkit's breadcrumb: a run SLURM kills before it can write is then KILLED, not
    MISSING."""
    from pypmt_eval_toolkit.runner import _write_marker

    os.makedirs(os.path.dirname(_path(sandbox, record, ".running")), exist_ok=True)
    _write_marker(_path(sandbox, record, ".running"), record)


def _write(sandbox, record, status, seconds=None, note=None):
    from pypmt_eval_toolkit.runner import _peak_memory_mb, _write_json

    record["status"] = status
    record["timings"]["solve-seconds"] = seconds
    record["timings"]["total-seconds"] = time.time() - record["run"]["started"]
    record["metrics"]["peak-memory-mb"] = _peak_memory_mb()
    if note:
        record["logs"].append(note)
    path = _path(sandbox, record, ".json")
    try:
        _write_json(path, record)
    except MemoryError:
        # The write can be what runs out: a MEMOUT run still holds the search that filled the
        # address space. The cap has done its job by then; lift it to the hard limit and write
        # again.
        if resource is not None:
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (hard, hard))
        _write_json(path, record)
    try:
        os.remove(_path(sandbox, record, ".running"))
    except OSError:
        pass
    print(f"{record['task']['task-id']:28} {record['planner']['tag']:8} {status:12} "
          f"{record['timings']['total-seconds']:.2f}s")
    return record


def report(sandbox, per_domain=True, plots=True):
    """The toolkit's `analyze` and `report` over a sandbox: `analysis/results.csv` and the
    coverage summary, then `report/` with the tables, their LaTeX, and the figures."""
    from pypmt_eval_toolkit.analyzer import analyze
    from pypmt_eval_toolkit.report import report as tables

    analyze(argparse.Namespace(sandbox_dir=sandbox, results_dir=None, output_dir=None,
                               per_domain=per_domain))
    tables(argparse.Namespace(sandbox_dir=sandbox, results_dir=None, output_dir=None,
                              no_plots=not plots, formats="pdf,png"))


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--sandbox-dir", default="sandbox", help="where the runs go")
    parser = argparse.ArgumentParser(prog="planiverse-bench",
                                     description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    generate_ = commands.add_parser("generate", parents=[common],
                                    help="write the commands and the SLURM arrays")
    for option in ("partition", "qos", "account"):
        generate_.add_argument(f"--{option}", help=f"SLURM {option}")
    generate_.add_argument("--reference", action="store_true",
                           help="only the four reference planners, not every planner")
    generate_.add_argument("--parallel", type=int, default=50,
                           help="array elements running at once (default: 50)")
    solve_ = commands.add_parser("solve", parents=[common],
                                 help="run one planner on one instance")
    solve_.add_argument("planner", choices=list(PLANNERS))
    solve_.add_argument("task", help="environment@index")
    solve_.add_argument("--seed", type=int,
                        help="for the seeded planners; the generated commands set it")
    report_ = commands.add_parser("report", parents=[common],
                                  help="the toolkit's analysis, tables and figures")
    report_.add_argument("--no-plots", action="store_true", help="the tables alone")
    args = parser.parse_args(argv)
    if args.command == "generate":
        generate(args.sandbox_dir, args.partition, args.qos, args.account, args.parallel,
                 args.reference)
    elif args.command == "solve":
        solve(args.sandbox_dir, args.planner, args.task, args.seed)
    else:
        report(args.sandbox_dir, plots=not args.no_plots)
    return 0
