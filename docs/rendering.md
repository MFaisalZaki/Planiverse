# Rendering a trace

A plan is a list of actions, which is not much to look at. A **trace** is what those actions
did, and this turns one into pictures. The target's extension picks the format, and the
choice is really about how many frames you want in front of you at once:

```python
trace = env.simulate(plan)
env.render_trace(trace, "plan.png", actions=plan, env=env)   # every frame on one sheet
env.render_trace(trace, "plan.pdf", actions=plan, env=env)   # paginated, for a long plan
env.render_trace(trace, "plan.gif")                          # one frame at a time
env.render_trace(trace, "plan-frames/")                      # a directory of PNGs
```

A **contact sheet** is the one that answers "what did the planner actually do": the whole
plan is visible at once, and passing `actions=` and `env=` captions each frame with its step
number, the action that produced it, and a green `goal` or red `dead end` note. A `.pdf`
paginates instead (`per_page=` tiles several frames onto a page). A `.gif` animates
(`duration_ms=` per frame, looping), which is the right thing for watching a plan and the
wrong thing for reading one. Anything without an extension is treated as a directory
receiving `state-000.png` onward, in trace order.

`max_states=` thins a long trace to the first, the last, and an even spread between —
worth having when a frame is a 640x576 console screenshot. The captions keep the real step
numbers, so a thinned sheet still says which step is which.

Every environment's own page carries a rendered plan for its first instance and the exact
snippet that produced it; the images live in [`docs/renders/`](renders/).

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
| [`renders/`](renders/) | The rendered plans the environment pages embed |
