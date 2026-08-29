"""Turning a plan into something you can look at.

    from planiverse.rendering import render_trace

    trace = env.simulate(plan)
    render_trace(trace, "plan.gif")        # an animated GIF
    render_trace(trace, "plan-frames/")    # one PNG per state
"""
from planiverse.rendering.trace import render_state, render_trace

__all__ = ["render_state", "render_trace"]
