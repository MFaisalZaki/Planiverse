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
action that produced it, and a green `goal` or red `dead end` note. The sheet stacks the frames
top to bottom, each caption in a column to the left of its state; the caption column is one width
for the whole trace and every picture is padded to one size, so the states line up one under
another, and a long action name wraps within its column. `columns=` sets the frames side by side
instead, each pinned to the top left of its cell. A `.pdf` paginates instead,
and `per_page=` tiles several frames onto a page. A `.gif` animates, with `duration_ms=` per
frame, looping, which is the right thing for watching a plan and the wrong thing for reading one.
Anything without an extension is treated as a directory receiving `state-000.png` onward, in trace
order.

`max_states=` thins a long trace to the first, the last, and an even spread between. The
captions keep the real step numbers, so a thinned sheet still says which step is which.

Rendered plans for the first instance of each environment live in [`docs/renders/`](renders/).

- **Dependencies:** Pillow for the frames and the files, and matplotlib for the monospace face it
  ships (DejaVu Sans Mono); the library already requires both.

## Where the pixels come from

A frame is **the state's text**, `str(state)`, typeset in a monospace font, black on white.
Every environment but the two emulators draws its state as a picture in text, and what it draws
depends on the environment:

- a board (Puzznic, Lolo, Lemmings and the other grid games) has its walls joined into
  box-drawing lines and its pieces as symbols, two columns to a cell so a cell is about square;
- a field or a map (Slingshot, Billiards, the tower defence, the railway, the factory, the
  Micropolis city, the flood city) is laid out on a grid of characters;
- a system (the reservoirs, the power grid, the microgrid, the water network, the supply chain,
  the airspace, the town of an outbreak, the streets of the traffic grid, the crop's field, the
  attacked network) is drawn as a schematic of its parts.

Each picture ends in one status line, the quantities that matter at that state. A picture keeps
its size across a trace, so a GIF does not jump. The same text is what `print(state)` shows and
what the state's OpenSpiel `observation_string` is; the operational states also carry
`readings()`, their numbers by name, which the OpenSpiel observation tensor holds. A GIF is pixels, and typesetting is
the one step that turns the text into them, which is the only reason a font appears in the module.

The pictures draw in the helpers of `planiverse.rendering.text`, so that every environment draws
a wall, a gauge or a box the same way:

```python
from planiverse.rendering.text import Canvas, bar, boxed, missing_glyphs, side_by_side, tank

boxed(["###", "#@#", "###"], glyphs={"@": "☻"})   # walls joined into lines
canvas = Canvas(40, 12)                            # a grid of characters to draw on
canvas.rect(0, 0, 10, 4, label="tank"); canvas.hline(10, 30, 2); str(canvas)
bar(3.5, 10, 20)                                   # '███████░░░░░░░░░░░░░'
tank(0.6, 5, 6)                                    # a vessel filled to 60%, as lines
side_by_side([block_a, block_b])                   # blocks of text set next to each other
missing_glyphs(text)                               # characters the face has no glyph for
```

Every character a picture uses is in DejaVu Sans Mono, so a picture prints in a terminal as it
renders in a GIF; `missing_glyphs` is how the tests check it.

## Rendering OpenSpiel states

A trace can be OpenSpiel states as well as an environment's, from applying a plan to
`game.new_initial_state()`:

```python
game = pyspiel.load_game("planiverse_tower_defence", {"index": 0})
actions, _ = spiel.solve(game, "bfws")
trace = [game.new_initial_state()]
for action in actions:
    trace.append(trace[-1].clone())
    trace[-1].apply_action(action)
render_trace(trace, "plan.png")
```

A frame is then what OpenSpiel says about the state. Its text is `str(state)`, which for a
`planiverse_*` game is the simulator state's own picture.
Without `env=`, the caption reads the end of the plan off the game itself: a terminal state with
a positive return is a goal, and any other terminal state is a dead end. Any of OpenSpiel's own
games renders the same way, as its text.

For the two emulators (the Game Boy and Stable-Retro) a frame is **the console's own screen**
at that state, which the environment reads back by loading the state's snapshot and drawing
one frame; their `render_trace` passes those pictures through `frames=`, one per state, and
the captions go on as for text. `frames=` is open to any caller with a picture per state.

## Files

| Path | What |
|---|---|
| [`trace.py`](../planiverse/rendering/trace.py) | `render_trace`, `render_state`, `outcome` |
| [`text.py`](../planiverse/rendering/text.py) | `boxed`, `Canvas`, `bar`, `tank`, `side_by_side`, `missing_glyphs`: what pictures draw in |
| [`tests/test_rendering.py`](../tests/test_rendering.py) | Tests |
| [`renders/`](renders/) | Rendered plans, one per environment |
