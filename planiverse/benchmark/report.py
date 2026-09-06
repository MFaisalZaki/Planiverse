"""Tables and plots from an analysed sandbox.

Three outputs, because three audiences: `results.txt` to read in a terminal, `coverage.tex` to
paste into a paper, and two plots. The plots are the conventional pair from the planning
literature, a survival ("cactus") plot and a runtime scatter, and they are conventional for
a good reason: coverage alone cannot distinguish a planner that solves 40 tasks quickly from
one that solves the same 40 just inside the limit.
"""
import json
import os
from collections import Counter
from itertools import combinations

from planiverse.benchmark.analysis import solver_sets
from planiverse.benchmark.config import parse_duration
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


def text_tables(summary):
    """The whole report as plain text."""
    lines = [f"Planiverse benchmark — {summary['runs']} runs", ""]
    lines += _coverage_table(summary)
    lines += [""] + _status_table(summary)
    lines += [""] + _environment_table(summary)
    return "\n".join(lines) + "\n"


def _rule(widths):
    return "  ".join("-" * width for width in widths)


def _row(cells, widths, align=None):
    align = align or ["<"] * len(cells)
    return "  ".join(f"{str(cell):{a}{w}}" for cell, w, a in zip(cells, widths, align))


def _coverage_table(summary):
    widths = (18, 10, 9, 9, 9, 9, 18, 12, 11)
    header = ("planner", "solved", "of", "coverage", "timeout", "memout", "mean ± std",
              "median", "plan len")
    lines = ["Coverage", _row(header, widths), _rule(widths)]
    for row in sorted(summary["coverage"], key=lambda r: (-r["solved"], r["planner"])):
        mean, std = row.get("mean_seconds"), row.get("std_seconds")
        lines.append(_row((
            row["planner"] + ("" if row["complete"] else " *"),
            row["solved"], row["total"], f"{row['coverage']:.0%}",
            row.get("timeouts", 0), row.get("memouts", 0),
            "-" if mean is None else
            f"{mean:.2f}s" + ("" if std is None else f" ± {std:.2f}"),
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
    lines.append(_rule(widths))
    totals = {row["planner"]: row for row in summary["coverage"]}
    lines.append(_row(["Total"] + [
        f"{totals[p]['solved']}/{totals[p]['total']}" if p in totals else "-"
        for p in planners], widths))
    lines.append("")
    lines.append("† no progress measure: SIW and BFWS run without their main input here.")
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
    _assert_agg()
    return pyplot.subplots(**kwargs)


def _assert_agg():
    if matplotlib.get_backend().lower() != "agg":
        pyplot.switch_backend("Agg")


#: Told apart without colour: planners by dash pattern, domains by marker shape, each shape
#: in black and again in mid-grey. Either cue survives a colour-vision deficiency and a
#: black-and-white printer, and the shapes stay distinct drawn hollow as well as filled.
_LINESTYLES = ("-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1)))
_MARKERS = ("o", "s", "^", "v", "D", "P", "X")
_SHADES = ("black", "0.55")


def time_limit_seconds(records):
    """The per-run wall-clock limit the experiment declared, from the records themselves.

    Every result carries the limits it ran under, so the penalty for a failed run comes from
    the experiment rather than from a constant here. Falls back to the largest observed
    runtime when no record declares one (synthetic records in tests, hand-made sandboxes).
    """
    for record in records:
        declared = (record.get("limits") or {}).get("time")
        if declared:
            try:
                return parse_duration(declared)
            except ValueError:
                continue
    return max((r.get("seconds") or 0) for r in records) or 1.0


def cactus_plot(records, path):
    """Time against tasks solved: the standard way to see who is fast and who is thorough.

    Each planner's solved runtimes are sorted and plotted cumulatively: the n-th point is the
    time its n-th easiest instance took. Runs that hit the time or memory limit are charged
    the full wall-clock limit and appended, so every planner's line spans its whole task set
    and a planner that fails often pays for it visibly instead of vanishing from the plot.
    """
    if not PLOTTING:
        return None
    limit = time_limit_seconds(records)
    solved, failed = {}, {}
    for record in records:
        planner = record["planner"]
        if record.get("status") == "SOLVED" and isinstance(record.get("seconds"),
                                                           (int, float)):
            solved.setdefault(planner, []).append(max(record["seconds"], 1e-3))
        elif record.get("status") in ("TIMEOUT", "MEMOUT"):
            failed[planner] = failed.get(planner, 0) + 1
    if not solved:
        return None

    figure, axes = _subplots(figsize=(7, 4.5))
    for index, planner in enumerate(sorted(solved)):
        times = sorted(solved[planner]) + [limit] * failed.get(planner, 0)
        # Dash pattern and a hollow marker every few dozen points: either cue alone
        # separates the lines on screen, and the marker still does when the dashes blur
        # at print size or the lines climb the same wall together.
        axes.plot(range(1, len(times) + 1), times, color="black",
                  linestyle=_LINESTYLES[index % len(_LINESTYLES)], linewidth=1.3,
                  marker=_MARKERS[index % len(_MARKERS)], markevery=(index * 9, 45),
                  markersize=5, markerfacecolor="white", markeredgewidth=0.9,
                  label=planner.upper())
    axes.axhline(limit, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    axes.set_xlabel("instances solved")
    axes.set_ylabel("time (s)")
    axes.set_title(f"Survival plot (time-outs and memory-outs charged {limit / 60:.0f}m)")
    axes.grid(alpha=0.3)
    axes.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    pyplot.close(figure)
    return path


def _domain_styles(environments):
    return {environment: (_MARKERS[index % len(_MARKERS)],
                          _SHADES[(index // len(_MARKERS)) % len(_SHADES)])
            for index, environment in enumerate(environments)}


def _penalised_seconds(record, limit):
    """A run's time on a scatter axis: its runtime if solved, the limit if not."""
    if record.get("status") == "SOLVED" and isinstance(record.get("seconds"), (int, float)):
        return max(record["seconds"], 1e-3)
    return limit


def runtime_twin_plots(records, directory):
    """One scatter per triple of planners, on a shared x-axis and two y-axes.

    The x-axis is the first planner's time; the left y-axis is the second's, the right y-axis
    the third's, on identical scales. Every instance is drawn twice: a filled marker against
    the left planner and a hollow one against the right, so the two comparisons sit on the
    same page and a point that moves between them shows what the third planner changes.
    Domains keep their marker and colour across every file. As in `scatter_plot`, a run that
    did not solve its instance sits on the limit, so failures stay in the picture.

    Within a triple the planner that solved the most instances takes the x-axis, then the
    next takes the left axis: the strongest planner is the natural reference, and putting it
    on x keeps the other two's failures on the top edge rather than the right one.
    """
    if not PLOTTING:
        return {}
    from matplotlib.lines import Line2D

    solved = Counter(record["planner"] for record in records
                     if record.get("status") == "SOLVED")

    limit = time_limit_seconds(records)
    by_task = {}
    for record in records:
        by_task.setdefault(record["task"], {})[record["planner"]] = record
    planners = sorted({record["planner"] for record in records})
    environments = sorted({record["environment"] for record in records})
    styles = _domain_styles(environments)
    span = (1e-3 / 1.5, limit * 1.5)

    written = {}
    for chosen in combinations(planners, 3):
        triple = sorted(chosen, key=lambda planner: (-solved[planner], planner))
        points = {}
        for entries in by_task.values():
            if any(planner not in entries for planner in triple):
                continue
            coordinates = [_penalised_seconds(entries[p], limit) for p in triple]
            points.setdefault(entries[triple[0]]["environment"], []).append(coordinates)
        if not points:
            continue

        figure, left = _subplots(figsize=(11, 6))
        right = left.twinx()
        for environment in environments:
            if environment not in points:
                continue
            xs, ys, zs = zip(*points[environment])
            marker, colour = styles[environment]
            left.scatter(xs, ys, marker=marker, color=colour, s=22, alpha=0.8,
                         label=environment)
            right.scatter(xs, zs, marker=marker, facecolors="none", edgecolors=colour,
                          s=22, alpha=0.8, linewidths=0.9)
        for axes in (left, right):
            axes.set_xscale("log")
            axes.set_yscale("log")
            axes.set_xlim(*span)
            axes.set_ylim(*span)
        left.plot(span, span, "k--", linewidth=0.8, alpha=0.6)
        left.axhline(limit, color="grey", linestyle=":", linewidth=0.8)
        left.axvline(limit, color="grey", linestyle=":", linewidth=0.8)
        left.set_xlabel(f"{triple[0].upper()} (s)")
        left.set_ylabel(f"{triple[1].upper()} (s)  — filled markers")
        right.set_ylabel(f"{triple[2].upper()} (s)  — hollow markers")
        left.set_title(" / ".join(p.upper() for p in triple)
                       + f": time per instance (unsolved at the {limit / 60:.0f}m limit)",
                       fontsize=10)
        left.grid(alpha=0.3)

        handles = [Line2D([], [], linestyle="", marker=marker, color=colour, label=env)
                   for env, (marker, colour) in styles.items() if env in points]
        handles += [Line2D([], [], linestyle="", marker="o", color="black",
                           label=f"vs {triple[1].upper()} (left axis)"),
                    Line2D([], [], linestyle="", marker="o", markerfacecolor="none",
                           color="black", label=f"vs {triple[2].upper()} (right axis)")]
        left.legend(handles=handles, fontsize="x-small", loc="center left",
                    bbox_to_anchor=(1.16, 0.5), title="domain", title_fontsize="small",
                    borderaxespad=1.5)
        figure.tight_layout()
        tag = "runtime_" + "_".join(triple)
        path = os.path.join(directory, tag + ".pdf")
        figure.savefig(path, bbox_inches="tight")
        pyplot.close(figure)
        written[tag] = path
    return written


def _overlap_segments(overlap):
    """The subsets to stack, in order, each as (label, keys it absorbs).

    The strongest planner (most tasks solved overall) anchors the split: everything, then it
    with each other planner, then it alone, then whatever it did not solve at all. With three
    planners that is five segments instead of seven, and the last one is the question the
    figure exists to answer: what do the others add.

    With two there are no "strongest + other" pairs to draw, because the only pair is the
    whole set and drawing it twice would count those tasks twice, so the split is just
    both / strongest only / the other one.
    """
    planners = overlap["planners"]
    total = next(row for row in overlap["rows"] if row["environment"] is None)
    solved_by = Counter()
    for key, count in total["sets"].items():
        for planner in key.split("+") if key else ():
            solved_by[planner] += count
    strongest = max(planners, key=lambda planner: (solved_by[planner], planner))
    others = [planner for planner in planners if planner != strongest]

    def key_for(subset):
        return "+".join(planner for planner in planners if planner in subset)

    segments = [(f"all {len(planners)}", [key_for(planners)])]
    if len(planners) > 2:
        for other in others:
            segments.append((f"{strongest.upper()} + {other.upper()}",
                             [key_for({strongest, other})]))
    segments.append((f"{strongest.upper()} only", [key_for({strongest})]))
    absorbed = {key for _, keys in segments for key in keys}
    rest = [key for key in _all_keys(planners) if key and key not in absorbed]
    segments.append((f"not {strongest.upper()} ("
                     + " / ".join(other.upper() for other in others) + " only)", rest))
    return segments


def _all_keys(planners):
    keys = []
    for size in range(len(planners) + 1):
        for subset in combinations(planners, size):
            keys.append("+".join(subset))
    return keys


#: Monochrome fills for the stacked segments: greys first, then hatched white. A set of k
#: planners needs k + 2 of them, so the two lists together cover every combination a run of
#: this size produces without reaching for colour.
_SEGMENT_FILLS = ("black", "0.35", "0.55", "0.75", "0.88")
_SEGMENT_HATCHES = ("///", "...", "xxx", "+++", "ooo")


def _segment_styles(count):
    styles = [{"facecolor": fill} for fill in _SEGMENT_FILLS[:count]]
    shortfall = max(0, count - len(_SEGMENT_FILLS))
    styles += [{"facecolor": "white", "hatch": hatch}
               for hatch in _SEGMENT_HATCHES[:shortfall]]
    return styles


def overlap_bars_plot(overlap, path):
    """One bar per domain, split by which subset of `overlap["planners"]` solved each task.

    Lengths are fractions of the domain, because domain sizes here range from 8 to 163 and
    absolute bars would let the two largest domains crowd out the picture; the size is on
    the label instead. The hollow remainder is what none of them solved.
    """
    if not PLOTTING or not overlap:
        return None
    segments = _overlap_segments(overlap)
    rows = [row for row in overlap["rows"] if row["environment"] is not None and row["tasks"]]
    if not rows:
        return None

    def fraction(row, keys):
        return sum(row["sets"].get(key, 0) for key in keys) / row["tasks"]

    rows.sort(key=lambda row: (sum(fraction(row, keys) for _, keys in segments),
                               row["environment"]))
    figure, axes = _subplots(figsize=(9, 0.42 * len(rows) + 1.8))
    positions = range(len(rows))
    left = [0.0] * len(rows)
    for (label, keys), style in zip(segments, _segment_styles(len(segments))):
        widths = [fraction(row, keys) for row in rows]
        axes.barh(positions, widths, left=left, edgecolor="black", linewidth=0.6,
                  label=label, **style)
        left = [start + width for start, width in zip(left, widths)]
    axes.barh(positions, [1.0 - start for start in left], left=left, facecolor="white",
              edgecolor="black", linewidth=0.6, linestyle=(0, (2, 2)), label="unsolved by all")
    axes.set_yticks(list(positions))
    axes.set_yticklabels([f"{row['environment']} (n={row['tasks']})" for row in rows],
                         fontsize=8)
    axes.set_xlim(0, 1)
    axes.set_xlabel("fraction of the domain's instances")
    axes.set_title("Instances solved, by which of "
                   + " / ".join(p.upper() for p in overlap["planners"]) + " solved them",
                   fontsize=10)
    axes.legend(fontsize="x-small", loc="lower right", framealpha=1.0)
    figure.tight_layout()
    figure.savefig(path)
    pyplot.close(figure)
    return path


def overlap_bars_plots(records, directory):
    """One `overlap_bars_plot` per combination of two or more planners.

    Which planners are worth comparing is a property of the run, so every combination is
    drawn rather than the width family alone. Within each one the planner that solved the
    most anchors the split and leads the file name, as in `runtime_twin_plots`.
    """
    if not PLOTTING:
        return {}
    solved = Counter(record["planner"] for record in records
                     if record.get("status") == "SOLVED")
    planners = sorted({record["planner"] for record in records})
    written = {}
    for size in range(2, len(planners) + 1):
        for chosen in combinations(planners, size):
            ordered = sorted(chosen, key=lambda planner: (-solved[planner], planner))
            overlap = {"planners": ordered, "rows": solver_sets(records, ordered)}
            tag = "overlap_" + "_".join(ordered)
            path = overlap_bars_plot(overlap, os.path.join(directory, tag + ".pdf"))
            if path:
                written[tag] = path
    return written


def write_report(sandbox_dir, summary, records):
    """Everything, into `sandbox/report/`."""
    directory = os.path.join(sandbox_dir, "report")
    os.makedirs(directory, exist_ok=True)
    written = {}

    text_path = os.path.join(directory, "results.txt")
    with open(text_path, "w") as handle:
        handle.write(text_tables(summary))
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

    cactus = cactus_plot(records, os.path.join(directory, "cactus.pdf"))
    if cactus:
        written["cactus"] = cactus
    written.update(runtime_twin_plots(records, directory))
    written.update(overlap_bars_plots(records, directory))
    return written
