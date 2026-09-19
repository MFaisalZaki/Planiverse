"""Turning a plan into something you can look at.

    from planiverse.rendering import render_trace

    trace = env.simulate(plan)
    render_trace(trace, "plan.png", actions=plan, env=env)   # every frame on one sheet
    render_trace(trace, "plan.pdf", actions=plan, env=env)   # paginated, for a long plan
    render_trace(trace, "plan.gif")                          # one frame at a time
    render_trace(trace, "plan-frames/")                      # one PNG per state
"""
from planiverse.rendering.charts import chart_frames, chart_image, readings_table
from planiverse.rendering.readings import READINGS, Panel, Readings, readings_of
from planiverse.rendering.trace import (
    DEFAULT_COLUMNS, contact_sheet, kept_indices, render_state, render_trace, trace_frames,
)

__all__ = ["DEFAULT_COLUMNS", "Panel", "READINGS", "Readings", "chart_frames", "chart_image",
           "contact_sheet", "kept_indices", "readings_of", "readings_table", "render_state",
           "render_trace", "trace_frames"]
