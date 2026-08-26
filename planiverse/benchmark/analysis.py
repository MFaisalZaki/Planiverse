"""Reading a sandbox full of result files back into tables.

The one thing this does that a `glob` would not is account for **missing** results. A
benchmark's most common failure is a job that never ran — cancelled, evicted, submitted to a
partition that does not exist — and if the analysis only reads the files that are there, a
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
        rows.append({
            "planner": planner,
            "class": entries[0].get("planner_class", ""),
            "complete": entries[0].get("complete",
                                       is_complete(entries[0].get("planner_class", ""))),
            "total": len(entries),
            "solved": len(solved),
            "coverage": len(solved) / len(entries) if entries else 0.0,
            "statuses": {status: counts.get(status, 0) for status in STATUSES
                         if counts.get(status)},
            "total_seconds": sum(times),
            "median_seconds": _median(times),
            "mean_plan_length": sum(lengths) / len(lengths) if lengths else None,
            "total_expansions": sum(expansions),
        })
    return rows


def per_environment(records):
    """Solved counts split by environment, which is where planners actually differ."""
    grouped = defaultdict(Counter)
    totals = Counter()
    for record in records:
        key = (record["planner"], record["environment"])
        grouped[key]["total"] += 1
        totals[record["environment"]] += 1
        if record.get("status") == "SOLVED":
            grouped[key]["solved"] += 1

    rows = []
    for (planner, environment), counts in sorted(grouped.items()):
        rows.append({"planner": planner, "environment": environment,
                     "solved": counts["solved"], "total": counts["total"],
                     "coverage": counts["solved"] / counts["total"] if counts["total"] else 0,
                     "has_progress_measure": next(
                         (r.get("has_progress_measure") for r in records
                          if r["environment"] == environment
                          and r.get("has_progress_measure") is not None), None)})
    return rows


def head_to_head(records):
    """For each pair of planners: tasks solved by one and not the other.

    Coverage totals hide this. Two planners can each solve 40 of 60 and have only 20 in
    common, which is a completely different situation from solving the same 40.
    """
    by_task = defaultdict(dict)
    for record in records:
        by_task[record["task"]][record["planner"]] = record.get("status") == "SOLVED"

    planners = sorted({record["planner"] for record in records})
    rows = []
    for left in planners:
        for right in planners:
            if left >= right:
                continue
            both = only_left = only_right = neither = 0
            for statuses in by_task.values():
                if left not in statuses or right not in statuses:
                    continue
                l, r = statuses[left], statuses[right]
                both += l and r
                only_left += l and not r
                only_right += r and not l
                neither += not l and not r
            rows.append({"left": left, "right": right, "both": both,
                         "only_left": only_left, "only_right": only_right,
                         "neither": neither})
    return rows


def ipc_score(records, agile=False):
    """IPC-style scores in [0, 1] per run, summed per planner.

    Quality: `best_length / this_length` over the plans found for a task, so a planner scores
    1 on a task where it ties the shortest plan anyone found. Agile: `1 / (1 + log10(t/t*))`
    on runtime, clipped at 1, which is the IPC agile-track rule.

    Both are relative to the field, so a score only means something next to the other planners
    in the same table — adding a planner changes everyone's numbers.
    """
    best = {}
    for record in records:
        if record.get("status") != "SOLVED":
            continue
        key = record["task"]
        value = record["seconds"] if agile else record["plan_length"]
        if not isinstance(value, (int, float)) or value is None:
            continue
        value = max(float(value), 1e-3 if agile else 1.0)
        best[key] = min(best.get(key, value), value)

    scores = Counter()
    for record in records:
        if record.get("status") != "SOLVED":
            continue
        reference = best.get(record["task"])
        value = record["seconds"] if agile else record["plan_length"]
        if reference is None or not isinstance(value, (int, float)):
            continue
        value = max(float(value), 1e-3 if agile else 1.0)
        if agile:
            scores[record["planner"]] += min(1.0, 1.0 / (1.0 + math.log10(value / reference)))
        else:
            scores[record["planner"]] += reference / value
    return dict(scores)


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
        "head_to_head": head_to_head(records),
        "ipc_quality": ipc_score(records, agile=False),
        "ipc_agile": ipc_score(records, agile=True),
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
