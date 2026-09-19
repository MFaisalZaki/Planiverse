"""Charts for the environments whose state is a handful of readings.

A frame is the readings of every state up to the one shown, one panel per quantity,
against the step of the plan, with the action that produced each state written along the
bottom, the target as a dashed line across the panel it belongs to, and the goal or dead
end marked where the plan ends. A GIF grows the chart a state at a time; the sheet is the
whole plan on one image. matplotlib draws it (a dependency already, for the benchmark's
figures) through its object API, so no backend or display is involved.
"""
import math

from planiverse.rendering.readings import readings_of

#: Chart chrome: the surface, the inks, the grid and the axis, and the series and status
#: colours, in a fixed order so a reading keeps its colour whatever else is drawn.
SURFACE, INK, SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
DPI = 110


def readings_table(trace, readings=None):
    """The readings of every state of a trace, one dict per state."""
    readings = readings or readings_of(trace[0])
    if readings is None:
        raise ValueError(f"no readings are registered for {type(trace[0]).__name__}; it renders as text")
    return [readings.values(state) for state in trace]


def _limits(rows, panel):
    """A panel's y range over the whole trace: from zero when the readings sit near it, else
    padded round the data, so a line loading of 1.01 is not a flat line at the top; with
    headroom for the legend when there is one and for a target line above the data."""
    names = panel.series + ((panel.target,) if panel.target else ())
    values = [row[name] for row in rows for name in names if row.get(name) is not None]
    if not values:
        return 0, 1
    low, high = min(values), max(values)
    if low >= 0 and low <= 0.5 * high:
        low = 0
    span = high - low or abs(high) or 1
    headroom = 0.32 if len(panel.series) > 1 else 0.14
    return low - (0.08 * span if low != 0 else 0), high + headroom * span


def _spread(positions, gap):
    """Positions moved apart, in order, until no two are closer than `gap`, keeping the
    group centred where it was: how end labels dodge one another where series converge."""
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    moved = [positions[i] for i in order]
    for k in range(1, len(moved)):
        moved[k] = max(moved[k], moved[k - 1] + gap)
    shift = (sum(moved) - sum(positions[i] for i in order)) / max(1, len(moved))
    moved = [value - shift for value in moved]
    out = list(positions)
    for k, i in enumerate(order):
        out[i] = moved[k]
    return out


def _step_labels(trace, actions):
    labels = ["start"]
    for index in range(1, len(trace)):
        action = actions[index - 1] if actions and index - 1 < len(actions) else None
        labels.append(f"{index}. {action}" if action is not None else f"step {index}")
    return labels


def _note(env, state):
    if env is None:
        return None, INK
    if env.is_goal(state):
        return "goal", GOOD
    if env.is_terminal(state):
        return "dead end", CRITICAL
    return None, INK


def chart_image(trace, actions=None, env=None, upto=None, readings=None, dpi=DPI):
    """One image: the readings of the states up to `upto` (the whole trace by default),
    drawn on axes sized for the whole trace so that a sequence of these does not jump."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator
    from PIL import Image

    readings = readings or readings_of(trace[0])
    rows = readings_table(trace, readings)
    count = len(rows)
    upto = count - 1 if upto is None else upto
    labels = _step_labels(trace, actions)
    longest = max(len(label) for label in labels)
    widest_name = max(len(name) for panel in readings.panels for name in panel.series)
    width = max(7.5, 0.26 * count + 1.6)
    height = 1.5 * len(readings.panels) + 0.55 + 0.062 * longest
    figure = Figure(figsize=(width, height), dpi=dpi, facecolor=SURFACE)
    axes = figure.subplots(len(readings.panels), 1, sharex=True)
    axes = list(axes) if len(readings.panels) > 1 else [axes]
    note, colour = _note(env, trace[upto])
    title = labels[upto] if upto < count else labels[-1]
    figure.suptitle(title, x=0.01, y=0.995, ha="left", va="top", fontsize=10,
                    fontweight="bold", color=colour)
    if note:
        figure.text(0.99, 0.995, note, ha="right", va="top", fontsize=9, color=colour)

    for axis, panel in zip(axes, readings.panels):
        axis.set_facecolor(SURFACE)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(AXIS)
            axis.spines[side].set_linewidth(0.8)
        axis.yaxis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", length=0, labelsize=8, colors=SECONDARY)
        axis.set_title(panel.title, loc="left", fontsize=8.5, color=SECONDARY, pad=4)
        axis.set_xlim(-0.6, count - 0.4)
        low, high = _limits(rows, panel)
        axis.set_ylim(low, high)
        if all(float(row[name]).is_integer() for row in rows for name in panel.series
               if row.get(name) is not None):
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        if panel.target and rows[0].get(panel.target) is not None:
            target = rows[0][panel.target]
            axis.axhline(target, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
            axis.text(-0.45, target, f"{panel.target} {target:.4g}", ha="left", va="bottom",
                      fontsize=7.5, color=MUTED)
        ends = []
        for k, name in enumerate(panel.series):
            ys = [rows[i].get(name) for i in range(upto + 1)]
            xs = [i for i, y in enumerate(ys) if y is not None]
            ys = [y for y in ys if y is not None]
            if not xs:
                continue
            axis.plot(xs, ys, color=SERIES[k % len(SERIES)], linewidth=1.6, marker="o",
                      markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                      solid_capstyle="round", solid_joinstyle="round", label=name)
            ends.append((name, xs[-1], ys[-1]))
        # end labels, moved apart where the series converge, each led back to its line
        gap = 0.1 * (high - low)
        placed = _spread([y for _, _, y in ends], gap)
        for (name, x, y), at in zip(ends, placed):
            moved = abs(at - y) > 1e-9
            axis.annotate(name, (x, y), xytext=(8, (at - y) / (high - low) * axis.bbox.height / dpi * 72),
                          textcoords="offset points", fontsize=7.5, color=SECONDARY,
                          va="center", arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.8,
                                                        shrinkA=0, shrinkB=3) if moved else None)
        if len(panel.series) > 1:
            axis.legend(loc="upper left", frameon=False, fontsize=7.5, ncol=len(panel.series),
                        handlelength=1.4, labelcolor=SECONDARY, borderaxespad=0.2)
        if upto < count - 1:
            axis.axvline(upto, color=AXIS, linewidth=1)
        elif note:
            axis.axvline(count - 1, color=colour, linewidth=1.2)

    axes[-1].set_xticks(range(count))
    shown = labels if count <= 60 else [
        label if i % math.ceil(count / 60) == 0 or i == count - 1 else "" for i, label in enumerate(labels)]
    axes[-1].set_xticklabels(shown, rotation=90, fontsize=7.5, color=SECONDARY)
    figure.subplots_adjust(left=0.07 if width < 9 else 0.05,
                           right=1 - (0.07 * widest_name + 0.2) / width, top=0.93,
                           bottom=min(0.6, (0.062 * longest + 0.35) / height), hspace=0.55)
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    image = Image.frombuffer("RGBA", canvas.get_width_height(), canvas.buffer_rgba()).convert("RGB")
    return image


def chart_frames(trace, actions=None, env=None, indices=None, readings=None, dpi=DPI):
    """One image per state (or per index in `indices`), each the chart up to that state."""
    readings = readings or readings_of(trace[0])
    indices = range(len(trace)) if indices is None else indices
    return [chart_image(trace, actions, env, upto=index, readings=readings, dpi=dpi)
            for index in indices]
