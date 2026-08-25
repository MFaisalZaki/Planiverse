"""Turning a plan into something you can look at.

    from planiverse.rendering import render_trace

    trace = env.simulate(plan)
    render_trace(trace, "plan.pdf", actions=plan)
    render_trace(trace, "plan.png", actions=plan)
"""
from planiverse.rendering.trace import (
    DEFAULT_COLUMNS, contact_sheet, render_state, render_trace, trace_frames,
)

__all__ = ["DEFAULT_COLUMNS", "contact_sheet", "render_state", "render_trace",
           "trace_frames"]
