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

`max_states=` thins a long trace to the first, the last, and an even spread between, which is
worth having when a frame is a 640x576 console screenshot. The captions keep the real step
numbers, so a thinned sheet still says which step is which.

Rendered plans for the first instance of each environment live in [`docs/renders/`](renders/).

- **Dependencies:** Pillow, which the library already requires. Not matplotlib, though the
  monospace font is borrowed from it so that there is always one.

## Two sources of pixels

**A real screenshot**, where the state can produce one. The Game Boy states carry `save(rom,
path)`, which boots a throwaway emulator to the save-state and grabs the screen.
`env.render_trace` passes the environment's own cartridge automatically, while the free-standing
`render_trace` takes `gamerom=`:

```python
from planiverse.rendering import render_trace

render_trace(trace, "flipull.gif", gamerom="Flipull (USA).gb")
```

Frames are magnified by nearest neighbour rather than by Pillow's default bicubic resample. A Game
Boy frame is 160x144 pixels in four shades, and bicubic invents shades between them, which turns
the block edges these environments read their grids from into grey mush.

`planiverse.environments.gameboy.gb.screens(rom, states)` is the batch form, and what
`env.render()` uses: one throwaway emulator for the whole list rather than one cold boot per
state, since a save-state carries everything and loading is the cheap half.

**The state's own text**, otherwise: `str(state)`, typeset in a monospace font. That is not really
a fallback, since an ASCII board is what most of these environments were designed to be read as,
and the simulator environments describe themselves in a few lines of numbers. A GIF is pixels, and
typesetting is the one step that turns a text board into them, which is the only reason a font
appears in the module at all.

If a screenshot is asked for and cannot be produced, the renderer falls back to the text and warns
rather than falling back silently.

## Playing a cartridge, as opposed to searching it

`env.render()` on a Game Boy environment returns the console's own frames for every position
`step` has played through, de-duplicated. A cartridge draws the position, and the text board is a
reading of RAM taken next to it rather than a picture of it, so the text is still printed as a
caption, because a terminal cannot show a picture.

```python
env.reset()
env.step("left,6")
env.step("a+right,16")

frames = env.render()             # PIL images, 640x576, one per de-duplicated step
env.render("play.gif")            # or write them: animated
env.render("play.png")            # a contact sheet
env.render("play-frames/")        # a directory of PNGs
```

With a target it hands the played history to `render_trace`, so the same captions and formats are
available: `env.render("play.png", actions=plan, env=env)`.

## Files

| Path | What |
|---|---|
| [`trace.py`](../planiverse/rendering/trace.py) | `render_trace`, `render_state` |
| [`tests/test_rendering.py`](../tests/test_rendering.py) | Tests |
| [`renders/`](renders/) | Rendered plans, one per environment |
