# Rendering a trace

A plan is a list of actions, which is not much to look at. A *trace* (i.e., the sequence of states
those actions produced) is what `render_trace` turns into pictures. The target's extension picks
the format, and the choice is about how many frames you want in front of you at once:

```python
trace = env.simulate(plan)
env.render_trace(trace, "plan.png", actions=plan, env=env)   # every frame on one sheet
env.render_trace(trace, "plan.pdf", actions=plan, env=env)   # paginated, for a long plan
env.render_trace(trace, "plan.gif")                          # one frame at a time
env.render_trace(trace, "plan-frames/")                      # a directory of PNGs
```

A contact sheet is the format that answers what the planner actually did, since the whole plan is
visible at once. Passing `actions=` and `env=` captions each frame with its step number, the
action that produced it, and a green `goal` or red `dead end` note. A `.pdf` paginates instead,
and `per_page=` tiles several frames onto a page. A `.gif` animates, with `duration_ms=` per
frame, looping, which is the right thing for watching a plan and the wrong thing for reading one.
Anything without an extension is treated as a directory receiving `state-000.png` onward, in trace
order.

`max_states=` thins a long trace to the first, the last, and an even spread between. The
captions keep the real step numbers, so a thinned sheet still says which step is which.

Rendered plans for the first instance of each environment live in [`docs/renders/`](renders/).

- **Dependencies:** Pillow, which the library already requires. Not matplotlib, though the
  monospace font is borrowed from it so that there is always one.

## Where the pixels come from

Every frame is **the state's own text**: `str(state)`, typeset in a monospace font. That is not
a fallback, since an ASCII board is what most of these environments were designed to be read as,
and the simulator environments describe themselves in a few lines of numbers. A GIF is pixels,
and typesetting is the one step that turns a text board into them, which is the only reason a
font appears in the module at all.

## Files

| Path | What |
|---|---|
| [`trace.py`](../planiverse/rendering/trace.py) | `render_trace`, `render_state` |
| [`tests/test_rendering.py`](../tests/test_rendering.py) | Tests |
| [`renders/`](renders/) | Rendered plans, one per environment |
