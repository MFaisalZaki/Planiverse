# Rendering a trace

A plan is a list of actions, which is not much to look at. A **trace** is what those actions
did, and this turns one into a picture.

```python
from planiverse.rendering import render_trace

trace = env.simulate(plan)
render_trace(trace, "plan.png", actions=plan, env=env)   # a contact sheet
render_trace(trace, "plan.pdf", actions=plan, env=env)   # one state per page
```

The extension decides the format. `.pdf` gets a multi-page document — one state per page, or
`per_page=6` to tile several onto each, which is usually what you want for a long plan.

- **Dependencies:** Pillow, which the library already requires. Not matplotlib, though the
  monospace font is borrowed from it so that there is always one.

## Two sources of pixels

**A real screenshot**, when the state can produce one. The Game Boy states have
`save(rom, path)`, which boots a throwaway emulator to the save-state and grabs the screen.
Pass `gamerom=` and you get the actual console output:

```python
render_trace(trace, "flipull.png", actions=plan, env=env, gamerom="Flipull (USA).gb")
```

**The state's own text**, otherwise — `str(state)`, typeset in a monospace font. That is not
really a fallback: an ASCII board is what most of these environments were designed to be read
as, and the simulator environments describe themselves in a few lines of numbers.

If a screenshot is asked for and cannot be produced, the renderer falls back to the text and
**warns**. An earlier version fell back silently, which meant asking for screenshots quietly
returned text.

## What the captions carry

Each frame is titled with the action that produced it — the trace is one longer than the
plan, so frame 0 is `start`. Passing `env=` also marks goals in green and dead ends in red,
which is usually the thing you are actually looking for in a trace.

`max_states=N` thins a long trace to the first, the last and an even spread between. A
128-step wander from a goal-free planner is not worth 128 pages. The captions then carry the
real step numbers (`state 37 of 128`), because dropping the middle silently would be worse
than saying so.

## A bug this found

Nothing had ever looked at the pixels of `GBState.save`, and it had always produced a **blank
white rectangle**. `load_state` ticks the emulator with rendering *off* — correct everywhere
else, since search never looks at the screen and drawing it is wasted work — but that leaves
the frame buffer unfilled, so the screenshot captured nothing.

Super Mario Land's own `save` had a second problem: it opened a real SDL2 window to take the
shot, so it needed a display and failed on any headless machine, CI included. It now uses the
shared implementation with the null window.

Both are fixed, and `tests/test_rendering.py` asserts that a rendered frame contains more
than one colour — which is the assertion that would have caught it.

## Files

| Path | What |
|---|---|
| [`trace.py`](../planiverse/rendering/trace.py) | `render_trace`, `trace_frames`, `contact_sheet`, `render_state` |
| [`tests/test_rendering.py`](../tests/test_rendering.py) | Tests, including the blank-image check |
