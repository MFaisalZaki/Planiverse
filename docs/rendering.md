# Rendering a trace

A plan is a list of actions, which is not much to look at. A **trace** is what those actions
did, and this turns one into pictures. Rendering is deliberately nothing more than one image
per state, written to disk:

```python
trace = env.simulate(plan)
env.render_trace(trace, "plan.gif")        # an animated GIF, one frame per state
env.render_trace(trace, "plan-frames/")    # a directory of independent PNGs
```

A `.gif` target gets the animation (`duration_ms=` per frame, looping); anything else is
treated as a directory receiving `state-000.png` onward, in trace order. There used to be
captions, contact sheets, PDF pagination and trace thinning here; they dressed the frames up
without showing any state the frames did not already show, so they are gone.

- **Dependencies:** Pillow, which the library already requires. Not matplotlib, though the
  monospace font is borrowed from it so that there is always one.

## Two sources of pixels

**A real screenshot**, when the state can produce one. The Game Boy states have
`save(rom, path)`, which boots a throwaway emulator to the save-state and grabs the screen.
`env.render_trace` passes the environment's own cartridge automatically; the free-standing
`render_trace` takes `gamerom=`:

```python
from planiverse.rendering import render_trace

render_trace(trace, "flipull.gif", gamerom="Flipull (USA).gb")
```

**The state's own text**, otherwise: `str(state)`, typeset in a monospace font. That is not
really a fallback: an ASCII board is what most of these environments were designed to be read
as, and the simulator environments describe themselves in a few lines of numbers. A GIF is
pixels, and typesetting is the one step that turns a text board into them, which is the only
reason a font appears in the module at all.

If a screenshot is asked for and cannot be produced, the renderer falls back to the text and
**warns**. An earlier version fell back silently, which meant asking for screenshots quietly
returned text.

## A bug this found

Nothing had ever looked at the pixels of `GBState.save`, and it had always produced a **blank
white rectangle**. `load_state` ticks the emulator with rendering *off* (correct everywhere
else, since search never looks at the screen and drawing it is wasted work), but that leaves
the frame buffer unfilled, so the screenshot captured nothing.

Super Mario Land's own `save` had a second problem: it opened a real SDL2 window to take the
shot, so it needed a display and failed on any headless machine, CI included. It now uses the
shared implementation with the null window.

Both are fixed, and `tests/test_rendering.py` asserts that a rendered frame contains more
than one colour, which is the assertion that would have caught it.

## Files

| Path | What |
|---|---|
| [`trace.py`](../planiverse/rendering/trace.py) | `render_trace`, `render_state` |
| [`tests/test_rendering.py`](../tests/test_rendering.py) | Tests, including the blank-image check |
