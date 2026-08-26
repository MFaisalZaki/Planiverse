"""Running one (planner, task) pair, under limits, and writing down what happened.

This is the part that runs inside a SLURM array element: one planner, one task, one JSON file
out. Everything else in the package prepares for it or reads what it produced.

The point of a harness rather than a `for` loop is that a failure has to be *recorded* rather
than raised. A planner that runs out of memory on instance 12 must not take the other 400 runs
with it, and "we never ran this" must look different in the results from "we ran it and it
found nothing". So every run ends in a status:

| Status | Meaning |
|---|---|
| `SOLVED` | a plan, and — if `validate-plans` — one that replays to a goal |
| `INVALID` | a plan that does not replay to a goal. A planner bug, reported as one |
| `UNSOLVED` | the planner stopped of its own accord without a plan |
| `TIMEOUT` | the wall-clock limit ran out first |
| `NODEOUT` | the expansion limit ran out first |
| `MEMOUT` | the memory limit was hit |
| `ERROR` | it raised |
| `UNSUPPORTED` | the environment could not be built here (missing dependency, missing ROM) |
| `MISSING` | no result file — assigned at analysis time, never written by a run |

`TIMEOUT` and `NODEOUT` are separated because they say different things about the same
planner: one is too slow per node, the other is looking in the wrong place.

`UNSOLVED` deliberately does **not** say "unsolvable". Only BFWS among these planners is
complete — see `catalogue.COMPLETE` — so for everything else it means no more than "this
planner stopped looking". Each result records the planner's raw `search_status` alongside, and
`complete` says whether the answer is a proof, so the two cases stay distinguishable in the
tables instead of being averaged together.
"""
import json
import os
import platform
import resource
import signal
import time
import traceback

from planiverse.benchmark import catalogue
from planiverse.benchmark.discovery import parse_task_id, task_filename
from planiverse.benchmark.measures import DEFAULT_MEASURE, has_measure, measure_for
from planiverse.environments import get_spec
from planiverse.planners.width.result import Budget

#: Statuses that count as the planner having answered the question.
CONCLUSIVE = ("SOLVED", "INVALID", "UNSOLVED")

#: Every status, in the order reports should list them.
STATUSES = ("SOLVED", "INVALID", "UNSOLVED", "TIMEOUT", "NODEOUT", "MEMOUT", "ERROR",
            "UNSUPPORTED", "MISSING")


class _TimeLimit:
    """A hard wall-clock stop, on top of the planner's own `Budget`.

    The `Budget` is checked between expansions, which is enough right up until one expansion
    is itself slow — the power grid environment spends 8 to 19 seconds in a single one, and
    the Game Boy environments can spend longer. Without an alarm on top, a job asked for 30
    minutes can overrun by one expansion and be killed by SLURM instead, losing the result
    file and turning a legible TIMEOUT into a missing row.

    SIGALRM is POSIX-only. On a platform without it this degrades to the `Budget` alone and
    says so rather than pretending to enforce a limit.
    """

    def __init__(self, seconds):
        self.seconds = seconds
        self.available = hasattr(signal, "SIGALRM") and seconds and seconds > 0
        self.previous = None

    def __enter__(self):
        if self.available:
            self.previous = signal.signal(signal.SIGALRM, self.__fire__)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, *_):
        if self.available:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.previous or signal.SIG_DFL)
        return False

    @staticmethod
    def __fire__(_signum, _frame):
        raise TimeoutError("wall-clock limit reached")


def apply_memory_limit(limit_bytes):
    """Cap the address space, so an overrun is a `MemoryError` we can catch and record.

    Without it the kernel's OOM killer ends the process and the run leaves no trace at all.
    The cap is on `RLIMIT_AS` rather than `RLIMIT_DATA` because that is what actually bounds
    a Python process's growth, mmapped arrays included.

    Returns whether the limit was applied: it can only ever be lowered, so a job already
    running under a tighter one keeps it.
    """
    if not limit_bytes or not hasattr(resource, "RLIMIT_AS"):
        return False
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY and limit_bytes > hard:
        return False
    try:
        resource.setrlimit(resource.RLIMIT_AS, (int(limit_bytes), hard))
        return True
    except (ValueError, OSError):
        return False


def peak_memory_bytes():
    """Peak RSS of this process. `ru_maxrss` is kB on Linux and bytes on macOS."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if platform.system() == "Darwin" else peak * 1024


def result_path(sandbox_dir, planner_tag, task):
    return os.path.join(sandbox_dir, "results", planner_tag, f"{task_filename(task)}.json")


def solve(planner_spec, task, limits, sandbox_dir=None, seed=None):
    """Run one pair and return the result dictionary, writing it out if given a sandbox.

    The result is written even when the run fails, which is the whole point: a benchmark that
    only records its successes cannot tell you that a planner crashed on a third of the set.
    """
    environment_name, index = parse_task_id(task)
    started_wall = time.time()
    record = {
        "task": task,
        "environment": environment_name,
        "index": index,
        "planner": planner_spec.tag,
        "planner_class": planner_spec.planner,
        "params": dict(planner_spec.params),
        "limits": {"time": limits.time, "memory": limits.memory,
                   "max_expansions": limits.max_expansions},
        "seed": seed,
        "has_progress_measure": has_measure(environment_name),
        "randomised": catalogue.is_randomised(planner_spec.planner),
        "complete": catalogue.is_complete(planner_spec.planner),
        "host": platform.node(),
        "started": started_wall,
    }
    if catalogue.is_randomised(planner_spec.planner) and seed is not None:
        record["params"].setdefault("seed", seed)

    memory_limited = apply_memory_limit(limits.bytes())
    record["memory_limit_applied"] = memory_limited

    environment = None
    try:
        spec = get_spec(environment_name)
        if not spec.available():
            return _finish(record, "UNSUPPORTED", started_wall, sandbox_dir,
                           note=f"{environment_name} needs {', '.join(spec.requires)}")
        try:
            environment = spec.build()
            environment.fix_index(index)
        except Exception as exc:
            return _finish(record, "UNSUPPORTED", started_wall, sandbox_dir,
                           note=f"{type(exc).__name__}: {exc}")

        progress = measure_for(environment_name)
        planner = catalogue.build(planner_spec.planner, record["params"],
                                  progress=None if progress is DEFAULT_MEASURE else progress)

        seconds = limits.seconds()
        budget = Budget(max_expansions=limits.max_expansions, max_seconds=seconds)
        limiter = _TimeLimit(seconds + max(1.0, 0.02 * seconds))
        record["hard_timeout_armed"] = limiter.available

        started = time.perf_counter()
        try:
            with limiter:
                outcome = planner.solve(environment, budget)
        except TimeoutError:
            return _finish(record, "TIMEOUT", started_wall, sandbox_dir,
                           elapsed=time.perf_counter() - started,
                           note="stopped by the hard wall-clock limit, not by the budget")
        except MemoryError:
            return _finish(record, "MEMOUT", started_wall, sandbox_dir,
                           elapsed=time.perf_counter() - started)
        elapsed = time.perf_counter() - started

        record["statistics"] = _statistics(outcome)
        record["search_status"] = outcome.status
        record["plan_length"] = len(outcome.plan) if outcome.plan is not None else None
        record["plan_cost"] = outcome.cost if outcome.plan else None
        record["width"] = outcome.width
        if outcome.plan is not None:
            record["plan"] = [str(action) for action in outcome.plan]

        status = _classify(outcome, limits, elapsed)
        if status == "SOLVED" and limits.validate_plans:
            valid, note = _validate(environment, outcome.plan)
            record["validated"] = valid
            if not valid:
                return _finish(record, "INVALID", started_wall, sandbox_dir,
                               elapsed=elapsed, note=note)
        return _finish(record, status, started_wall, sandbox_dir, elapsed=elapsed)

    except MemoryError:
        return _finish(record, "MEMOUT", started_wall, sandbox_dir)
    except Exception as exc:
        record["traceback"] = traceback.format_exc()
        return _finish(record, "ERROR", started_wall, sandbox_dir,
                       note=f"{type(exc).__name__}: {exc}")
    finally:
        close = getattr(environment, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def _classify(outcome, limits, elapsed):
    """Which limit ran out, when the search says only that one did.

    `SearchResult.status` reports `out_of_budget` without saying which half of the budget was
    spent, and the distinction is the useful part — too slow per node is a different problem
    from looking in the wrong place. So it is recovered here by comparing what was spent
    against what was allowed.
    """
    if outcome.solved:
        return "SOLVED"
    if outcome.status in ("failed", "exhausted"):
        return "UNSOLVED"
    expansions = outcome.statistics.expansions
    if limits.max_expansions and expansions >= limits.max_expansions:
        return "NODEOUT"
    if limits.seconds() and elapsed >= 0.95 * limits.seconds():
        return "TIMEOUT"
    # Out of budget with neither limit reached: an iterated search whose leg budgets ran out.
    return "NODEOUT" if limits.max_expansions else "TIMEOUT"


def _validate(environment, plan):
    """Replay the plan and check it lands on a goal.

    Cheap, and it catches the failure mode a benchmark is least able to notice otherwise: a
    planner that reports a plan it cannot reproduce. `validate` is optional in the contract,
    so replaying by hand is the fallback rather than an assumption.
    """
    try:
        validator = getattr(environment, "validate", None)
        if callable(validator):
            return bool(validator(plan)), "validate() rejected the plan"
        trace = environment.simulate(plan)
        return bool(trace) and environment.is_goal(trace[-1]), \
            "replaying the plan did not reach a goal"
    except Exception as exc:
        return False, f"validation raised {type(exc).__name__}: {exc}"


def _statistics(outcome):
    statistics = outcome.statistics
    return {
        "expansions": statistics.expansions,
        "generated": statistics.generated,
        "pruned_novelty": statistics.pruned_novelty,
        "pruned_terminal": statistics.pruned_terminal,
        "pruned_duplicate": statistics.pruned_duplicate,
        "search_seconds": statistics.elapsed,
        "widths_tried": list(statistics.widths_tried),
    }


def _finish(record, status, started_wall, sandbox_dir, elapsed=None, note=None):
    record["status"] = status
    record["seconds"] = elapsed if elapsed is not None else time.time() - started_wall
    record["wall_seconds"] = time.time() - started_wall
    record["peak_memory_bytes"] = peak_memory_bytes()
    if note:
        record["note"] = note
    record.setdefault("statistics", _empty_statistics())
    if sandbox_dir:
        write_result(sandbox_dir, record)
    return record


def _empty_statistics():
    return {"expansions": 0, "generated": 0, "pruned_novelty": 0, "pruned_terminal": 0,
            "pruned_duplicate": 0, "search_seconds": 0.0, "widths_tried": []}


def write_result(sandbox_dir, record):
    path = result_path(sandbox_dir, record["planner"], record["task"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(record, handle, indent=2, default=str)
        handle.write("\n")
    return path
