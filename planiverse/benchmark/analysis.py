"""Reading a sandbox full of result files back into tables.

The one thing this does that a `glob` would not is account for **missing** results. A
benchmark's most common failure is a job that never ran (cancelled, evicted, submitted to a
partition that does not exist), and if the analysis only reads the files that are there, a
planner that crashed on half the set shows up with excellent coverage over the half it
survived. So the expected set of pairs comes from `tasks.json`, and anything without a file is
`MISSING`.
"""
import csv
import json
import math
import os
from collections import Counter, defaultdict

from planiverse.benchmark.catalogue import is_complete
from planiverse.benchmark.discovery import read_tasks, task_filename
from planiverse.benchmark.runner import STATUSES


def load_results(sandbox_dir, pairs=None):
    """Every expected result, with `MISSING` rows for the ones that are not there."""
    if pairs is None:
        pairs = read_tasks(sandbox_dir).get("pairs", [])

    records = []
    for pair in pairs:
        path = os.path.join(sandbox_dir, "results", pair["planner"],
                            f"{task_filename(pair['task'])}.json")
        if not os.path.isfile(path):
            records.append({"planner": pair["planner"], "task": pair["task"],
                            "environment": pair["environment"], "index": pair["index"],
                            "status": "MISSING", "seconds": None, "plan_length": None,
                            "statistics": {}})
            continue
        try:
            with open(path) as handle:
                records.append(json.load(handle))
        except (OSError, json.JSONDecodeError) as exc:
            # A truncated file is what a job killed mid-write leaves behind. That is a real
            # outcome and reporting it as ERROR beats crashing the analysis over it.
            records.append({"planner": pair["planner"], "task": pair["task"],
                            "environment": pair["environment"], "index": pair["index"],
                            "status": "ERROR", "seconds": None, "plan_length": None,
                            "note": f"unreadable result file: {exc}", "statistics": {}})
    return records


def coverage(records):
    """Per planner: how many solved, out of how many, and over what time.

    Runtime is summarised over *solved* runs only. Averaging in the timeouts would reward a
    planner for failing quickly, which is the opposite of what a benchmark should do.
    """
    grouped = defaultdict(list)
    for record in records:
        grouped[record["planner"]].append(record)

    rows = []
    for planner, entries in sorted(grouped.items()):
        counts = Counter(entry.get("status", "MISSING") for entry in entries)
        solved = [entry for entry in entries if entry.get("status") == "SOLVED"]
        times = sorted(entry["seconds"] for entry in solved
                       if isinstance(entry.get("seconds"), (int, float)))
        lengths = [entry["plan_length"] for entry in solved
                   if isinstance(entry.get("plan_length"), int)]
        expansions = [entry.get("statistics", {}).get("expansions", 0) for entry in solved]
        # Completeness is a property of the run, not of the planner: `IteratedWidth` proves
        # unsolvability on the runs where it exhausted the space and proves nothing on the
        # ones the budget stopped. So the flag here answers the question the footnote asks,
        # "can I read this planner's UNSOLVED rows as proofs?", which is only yes when every
        # one of them was.
        unsolved = [entry for entry in entries if entry.get("status") == "UNSOLVED"]
        rows.append({
            "planner": planner,
            "class": entries[0].get("planner_class", ""),
            "complete": all(
                entry.get("complete",
                          is_complete(entry.get("planner_class", ""),
                                      entry.get("search_status")))
                for entry in unsolved),
            "proofs": sum(1 for entry in unsolved if entry.get("complete")),
            "unsolved": len(unsolved),
            "total": len(entries),
            "solved": len(solved),
            "coverage": len(solved) / len(entries) if entries else 0.0,
            "timeouts": counts.get("TIMEOUT", 0),
            "memouts": counts.get("MEMOUT", 0),
            "statuses": {status: counts.get(status, 0) for status in STATUSES
                         if counts.get(status)},
            "total_seconds": sum(times),
            "median_seconds": _median(times),
            "mean_seconds": sum(times) / len(times) if times else None,
            "std_seconds": _std(times),
            "mean_plan_length": sum(lengths) / len(lengths) if lengths else None,
            "total_expansions": sum(expansions),
        })
    return rows


def per_environment(records):
    """Solved counts split by environment, which is where planners actually differ.

    Mean runtime is over the *solved* runs only, for the same reason `coverage` summarises
    that way: averaging in the failures rewards failing fast.
    """
    grouped = defaultdict(Counter)
    times = defaultdict(list)
    for record in records:
        key = (record["planner"], record["environment"])
        grouped[key]["total"] += 1
        if record.get("status") == "SOLVED":
            grouped[key]["solved"] += 1
            if isinstance(record.get("seconds"), (int, float)):
                times[key].append(record["seconds"])

    rows = []
    for (planner, environment), counts in sorted(grouped.items()):
        solved_times = times[(planner, environment)]
        rows.append({"planner": planner, "environment": environment,
                     "solved": counts["solved"], "total": counts["total"],
                     "coverage": counts["solved"] / counts["total"] if counts["total"] else 0,
                     "mean_seconds": (sum(solved_times) / len(solved_times)
                                      if solved_times else None),
                     "std_seconds": _std(solved_times),
                     "has_progress_measure": next(
                         (r.get("has_progress_measure") for r in records
                          if r["environment"] == environment
                          and r.get("has_progress_measure") is not None), None)})
    return rows


def solver_sets(records, planners):
    """Per environment: how many tasks each exact subset of `planners` solved.

    The unit is the subset, not the planner, so the rows partition the tasks: a task solved
    by BFWS and IW counts once, under "bfws+iw", and not under either planner's own key. That
    is what makes the counts stackable, and it is the question the coverage table cannot
    answer, whether the planners solve the same tasks or different ones. Keys are the
    planners joined with "+", and "" is the tasks none of them solved. A final row with
    `environment` None sums over every domain. Only tasks every one of `planners` has a
    record for are counted.
    """
    planners = list(planners)
    by_task = defaultdict(dict)
    environment_of = {}
    for record in records:
        if record["planner"] in planners:
            by_task[record["task"]][record["planner"]] = record.get("status") == "SOLVED"
            environment_of[record["task"]] = record["environment"]

    sets = defaultdict(Counter)
    for task, statuses in by_task.items():
        if any(planner not in statuses for planner in planners):
            continue
        key = "+".join(planner for planner in planners if statuses[planner])
        sets[environment_of[task]][key] += 1
        sets[None][key] += 1
    return [{"environment": environment, "tasks": sum(sets[environment].values()),
             "sets": dict(sets[environment])}
            for environment in sorted(sets, key=lambda name: (name is None, name or ""))]


def write_csv(sandbox_dir, records):
    """One row per run, for anyone who would rather use pandas than these tables."""
    directory = os.path.join(sandbox_dir, "analysis")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "results.csv")
    columns = ["planner", "environment", "index", "task", "status", "seconds",
               "plan_length", "plan_cost", "width", "expansions", "generated",
               "pruned_terminal", "peak_memory_bytes", "has_progress_measure", "complete",
               "randomised", "seed", "search_status", "note"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            statistics = record.get("statistics") or {}
            writer.writerow({**record,
                             "expansions": statistics.get("expansions"),
                             "generated": statistics.get("generated"),
                             "pruned_terminal": statistics.get("pruned_terminal")})
    return path


def summarise(sandbox_dir, records=None):
    """Everything the `analyze` stage produces, as one dictionary."""
    records = load_results(sandbox_dir) if records is None else records
    return {
        "runs": len(records),
        "coverage": coverage(records),
        "per_environment": per_environment(records),
        "statuses": dict(Counter(record.get("status", "MISSING") for record in records)),
    }


def write_summary(sandbox_dir, summary):
    directory = os.path.join(sandbox_dir, "analysis")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "summary.json")
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return path


def _median(values):
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2


def _std(values):
    """Population standard deviation; None until there are two values to spread."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
