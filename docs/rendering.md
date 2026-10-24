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

- **Dependencies:** Pillow for the text frames and the files, and matplotlib for the readings
  charts; the library already requires both.

## Where the pixels come from

For a board, a frame is **the state's own text**: `str(state)`, typeset in a monospace font.
That is not a fallback, since an ASCII board is what most of the games were designed to be read
as. A GIF is pixels, and typesetting is the one step that turns a text board into them, which
is the only reason a font appears in the module at all.

For an environment whose state is a handful of readings (a city's population and funds, a
factory's plates and coal, a crop's biomass, a grid's line loadings, a water network's
contamination, a flood city's costs, a tower defence's lives and gold) a line of numbers per
frame is read better as a chart of those numbers over the plan, so a frame there is **a chart
of the readings of every state up to the one shown**: one panel per quantity, the series that
share a unit on one axis and never two scales on one panel, the action that produced each state
along the bottom, a target as a dashed line, and the goal or dead end marked where the plan
ends. The GIF grows the chart a state at a time; the sheet is the whole plan on one figure.
The readings are part of the state: such a state carries `readings()`, its numbers by name, and
`panels`, which of them share an axis (`Panel(title, series, target)`, from
`planiverse.environments`). `render_trace` charts whatever has them and needs no telling;
`charts=False` asks for the text anyway. matplotlib draws it through its object API, so no
display is involved.

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

A frame is then what OpenSpiel says about the state. Its text is `str(state)`, and for a
`planiverse_*` game that is the simulator state's own text, the same as its
`observation_string`. Its readings are the ones the game's observation tensor holds, named in
the `dict` of OpenSpiel's observer (`open_spiel.python.observation.make_observation(game)`).
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
| [`charts.py`](../planiverse/rendering/charts.py) | `chart_image`, `chart_frames`: the readings charts |
| [`tests/test_rendering.py`](../tests/test_rendering.py) | Tests |
| [`renders/`](renders/) | Rendered plans, one per environment |
