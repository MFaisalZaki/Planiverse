"""Tables and plots from an analysed sandbox.

Three outputs, because three audiences: `results.txt` to read in a terminal, `coverage.tex` to
paste into a paper, and two plots. The plots are the conventional pair from the planning
literature, a survival ("cactus") plot and a runtime scatter, and they are conventional for
a good reason: coverage alone cannot distinguish a planner that solves 40 tasks quickly from
one that solves the same 40 just inside the limit.
"""
import json
import os

from planiverse.benchmark.runner import STATUSES

#: Plot rendering needs matplotlib. It is a dependency of the library already, but the tables
#: are the useful part and they should not fail because a headless box has no backend.
try:  # pragma: no cover - exercised by whether matplotlib imports
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot
    PLOTTING = True
except Exception:  # pragma: no cover
    PLOTTING = False


def text_tables(summary, records):
    """The whole report as plain text."""
    lines = [f"Planiverse benchmark — {summary['runs']} runs", ""]
    lines += _coverage_table(summary)
    lines += [""] + _status_table(summary)
    lines += [""] + _environment_table(summary)
    lines += [""] + _head_to_head_table(summary)
    lines += [""] + _score_table(summary)
    lines += [""] + _caveats(summary, records)
    return "\n".join(lines) + "\n"


def _rule(widths):
    return "  ".join("-" * width for width in widths)


def _row(cells, widths, align=None):
    align = align or ["<"] * len(cells)
    return "  ".join(f"{str(cell):{a}{w}}" for cell, w, a in zip(cells, widths, align))


def _coverage_table(summary):
    widths = (18, 10, 9, 9, 12, 12, 11)
    header = ("planner", "solved", "of", "coverage", "total time", "median", "plan len")
    lines = ["Coverage", _row(header, widths), _rule(widths)]
    for row in sorted(summary["coverage"], key=lambda r: (-r["solved"], r["planner"])):
        lines.append(_row((
            row["planner"] + ("" if row["complete"] else " *"),
            row["solved"], row["total"], f"{row['coverage']:.0%}",
            f"{row['total_seconds']:.1f}s",
            "-" if row["median_seconds"] is None else f"{row['median_seconds']:.2f}s",
            "-" if row["mean_plan_length"] is None else f"{row['mean_plan_length']:.1f}",
        ), widths))
    lines.append("")
    lines.append("* at least one UNSOLVED row from this planner is not a proof that there is")
    lines.append("  no plan — it only means the planner stopped looking.")
    return lines


def _status_table(summary):
    present = [status for status in STATUSES
               if any(row["statuses"].get(status) for row in summary["coverage"])]
    widths = [18] + [max(8, len(status)) for status in present]
    lines = ["Outcomes", _row(["planner"] + present, widths), _rule(widths)]
    for row in sorted(summary["coverage"], key=lambda r: r["planner"]):
        lines.append(_row([row["planner"]] +
                          [row["statuses"].get(status, 0) for status in present], widths))
    return lines


def _environment_table(summary):
    planners = sorted({row["planner"] for row in summary["per_environment"]})
    environments = sorted({row["environment"] for row in summary["per_environment"]})
    solved = {(row["planner"], row["environment"]): row for row in summary["per_environment"]}
    widths = [22] + [max(9, len(planner)) for planner in planners]
    lines = ["Solved per environment", _row(["environment"] + planners, widths),
             _rule(widths)]
    for environment in environments:
        cells = []
        measured = True
        for planner in planners:
            row = solved.get((planner, environment))
            cells.append("-" if row is None else f"{row['solved']}/{row['total']}")
            if row is not None and row.get("has_progress_measure") is False:
                measured = False
        lines.append(_row([environment + ("" if measured else " †")] + cells, widths))
    lines.append("")
    lines.append("† no progress measure: SIW and BFWS run without their main input here.")
    return lines


def _head_to_head_table(summary):
    if not summary["head_to_head"]:
        return ["Head to head", "(needs at least two planners)"]
    widths = (18, 18, 7, 11, 11, 9)
    lines = ["Head to head",
             _row(("planner A", "planner B", "both", "only A", "only B", "neither"), widths),
             _rule(widths)]
    for row in summary["head_to_head"]:
        lines.append(_row((row["left"], row["right"], row["both"], row["only_left"],
                           row["only_right"], row["neither"]), widths))
    return lines


def _score_table(summary):
    quality, agile = summary["ipc_quality"], summary["ipc_agile"]
    if not quality:
        return ["IPC scores", "(nothing solved)"]
    widths = (18, 12, 12)
    lines = ["IPC scores (relative to this field only)",
             _row(("planner", "quality", "agile"), widths), _rule(widths)]
    for planner in sorted(quality, key=lambda p: -quality[p]):
        lines.append(_row((planner, f"{quality[planner]:.2f}",
                           f"{agile.get(planner, 0):.2f}"), widths))
    return lines


def _caveats(summary, records):
    """The things a reader would otherwise have to work out for themselves."""
    lines = ["Notes"]
    missing = summary["statuses"].get("MISSING", 0)
    if missing:
        lines.append(f"- {missing} runs produced no result file. Coverage percentages count "
                     f"them as failures; check sandbox/logs before believing the totals.")
    unsupported = summary["statuses"].get("UNSUPPORTED", 0)
    if unsupported:
        lines.append(f"- {unsupported} runs could not build their environment here.")
    randomised = sorted({record["planner"] for record in records
                         if record.get("randomised") and record.get("seed") is None})
    if randomised:
        lines.append(f"- {', '.join(randomised)} are randomised and ran unseeded, so these "
                     f"numbers will not reproduce exactly.")
    unmeasured = sorted({record["environment"] for record in records
                         if record.get("has_progress_measure") is False})
    if unmeasured:
        lines.append(f"- No progress measure for {', '.join(unmeasured)}; SIW and BFWS "
                     f"degrade to novelty-ordered search there.")
    if len(lines) == 1:
        lines.append("- nothing to flag.")
    return lines


def latex_coverage(summary):
    """A coverage table as a LaTeX tabular, ready to \\input."""
    planners = sorted({row["planner"] for row in summary["per_environment"]})
    environments = sorted({row["environment"] for row in summary["per_environment"]})
    lookup = {(row["planner"], row["environment"]): row
              for row in summary["per_environment"]}

    lines = [
        "% Generated by planiverse-bench report. Needs \\usepackage{booktabs}.",
        "\\begin{tabular}{l" + "r" * len(planners) + "}",
        "\\toprule",
        "Environment & " + " & ".join(_escape(p) for p in planners) + " \\\\",
        "\\midrule",
    ]
    for environment in environments:
        cells = []
        for planner in planners:
            row = lookup.get((planner, environment))
            cells.append("--" if row is None else f"{row['solved']}/{row['total']}")
        lines.append(_escape(environment) + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\midrule"]
    totals = {row["planner"]: row for row in summary["coverage"]}
    lines.append("Total & " + " & ".join(
        f"{totals[p]['solved']}/{totals[p]['total']}" if p in totals else "--"
        for p in planners) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def _escape(text):
    return str(text).replace("_", "\\_")


def _subplots(**kwargs):
    # The Agg selection at the top of this module is process-global state that any import
    # can overwrite (nasim switches the process to TkAgg, which segfaults headless), so
    # it is re-asserted at every figure creation rather than trusted.
    if matplotlib.get_backend().lower() != "agg":
        pyplot.switch_backend("Agg")
    return pyplot.subplots(**kwargs)


def cactus_plot(records, path):
    """Tasks solved against time: the standard way to see who is fast and who is thorough.

    Each planner's solved runtimes are sorted and plotted cumulatively, so a line that climbs
    steeply and stops early is a fast planner with poor coverage, and a line that keeps
    climbing is a slow one that gets there.
    """
    if not PLOTTING:
        return None
    series = {}
    for record in records:
        if record.get("status") != "SOLVED":
            continue
        if isinstance(record.get("seconds"), (int, float)):
            series.setdefault(record["planner"], []).append(max(record["seconds"], 1e-3))
    if not series:
        return None

    figure, axes = _subplots(figsize=(7, 4.5))
    for planner in sorted(series):
        times = sorted(series[planner])
        axes.plot(times, range(1, len(times) + 1), marker=".", label=planner)
    axes.set_xscale("log")
    axes.set_xlabel("time (s, log scale)")
    axes.set_ylabel("tasks solved")
    axes.set_title("Survival plot")
    axes.grid(alpha=0.3)
    axes.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    pyplot.close(figure)
    return path


def scatter_plot(records, left, right, path):
    """Two planners' runtimes on the same tasks, log-log.

    Failures are drawn on the border at the time limit rather than dropped, because a scatter
    of only the tasks both planners solved flatters whichever one fails more.
    """
    if not PLOTTING:
        return None
    by_task = {}
    for record in records:
        by_task.setdefault(record["task"], {})[record["planner"]] = record
    limit = max((r.get("seconds") or 0) for r in records) or 1.0

    points = []
    for statuses in by_task.values():
        a, b = statuses.get(left), statuses.get(right)
        if a is None or b is None:
            continue
        x = a["seconds"] if a.get("status") == "SOLVED" else limit
        y = b["seconds"] if b.get("status") == "SOLVED" else limit
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        both = a.get("status") == "SOLVED" and b.get("status") == "SOLVED"
        points.append((max(x, 1e-3), max(y, 1e-3), both))
    if not points:
        return None

    figure, axes = _subplots(figsize=(5, 5))
    for solved, marker, face in ((True, "o", "tab:blue"), (False, "o", "none")):
        chosen = [(x, y) for x, y, both in points if both is solved]
        if chosen:
            axes.scatter([x for x, _ in chosen], [y for _, y in chosen], marker=marker,
                         facecolors=face, edgecolors="tab:blue", alpha=0.7,
                         label="both solved" if solved else "at least one failed")
    span = [1e-3, limit * 1.5]
    axes.plot(span, span, "k--", linewidth=0.8)
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel(f"{left} (s)")
    axes.set_ylabel(f"{right} (s)")
    axes.set_title(f"{left} vs {right}")
    axes.grid(alpha=0.3)
    axes.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    pyplot.close(figure)
    return path


def write_report(sandbox_dir, summary, records):
    """Everything, into `sandbox/report/`."""
    directory = os.path.join(sandbox_dir, "report")
    os.makedirs(directory, exist_ok=True)
    written = {}

    text_path = os.path.join(directory, "results.txt")
    with open(text_path, "w") as handle:
        handle.write(text_tables(summary, records))
    written["text"] = text_path

    tex_path = os.path.join(directory, "coverage.tex")
    with open(tex_path, "w") as handle:
        handle.write(latex_coverage(summary))
    written["latex"] = tex_path

    json_path = os.path.join(directory, "report.json")
    with open(json_path, "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    written["json"] = json_path

    cactus = cactus_plot(records, os.path.join(directory, "cactus.png"))
    if cactus:
        written["cactus"] = cactus
    planners = sorted({record["planner"] for record in records})
    if len(planners) >= 2:
        scatter = scatter_plot(records, planners[0], planners[1],
                               os.path.join(directory, "scatter.png"))
        if scatter:
            written["scatter"] = scatter
    return written
