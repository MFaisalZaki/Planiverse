"""Render a state trace: a contact sheet, a PDF, an animated GIF or numbered PNGs.

The target's extension picks the format, and the choice is about how many frames you want
in front of you at once:

    render_trace(trace, "plan.png", actions=plan, env=env)   # every frame on one sheet
    render_trace(trace, "plan.pdf", actions=plan, env=env)   # paginated, for a long plan
    render_trace(trace, "plan.gif")                          # one frame at a time
    render_trace(trace, "plan-frames/")                      # one PNG per state

A sheet is the one that answers "what did the planner actually do": the whole plan is
visible at once and each frame is captioned with the step number, the action that produced
it, and whether the state is a goal or a dead end. A GIF shows one frame at a time, which
is the right thing for an animation and the wrong thing for reading a plan.

Two sources of pixels per state, tried in that order:

1. **A real screenshot**, when the state can produce one. The Game Boy states have
   `save(rom, path)`, which boots a throwaway emulator to the save-state and grabs the
   screen; pass `gamerom=` and you get the actual console output.
2. **The state's own text**, otherwise: `str(state)`, typeset in a monospace font. Most
   of these environments were designed to be read as an ASCII board, and a GIF is pixels:
   typesetting is the one step that turns the board into them, which is the only reason a
   font appears in this file at all.
"""
import os

#: Columns in a contact sheet. Chosen so a ten-step plan lands on two tidy rows.
DEFAULT_COLUMNS = 5

#: Room around a typeset text state, and for the caption above each frame.
PADDING = 12
CAPTION_HEIGHT = 34

_BACKGROUND = (255, 255, 255)
_INK = (17, 17, 17)
_CAPTION_INK = (90, 90, 90)
_GOAL = (26, 122, 62)
_TERMINAL = (176, 42, 42)
_BORDER = (208, 208, 208)


def _font(size, bold=False):
    """A monospace face, whatever is installed.

    DejaVu Sans Mono ships inside matplotlib, so on any machine that can install this
    library it is present. Pillow's built-in bitmap font is the last resort and is tiny,
    so text states rendered with it are legible but cramped.
    """
    from PIL import ImageFont

    candidates = []
    try:
        import matplotlib

        root = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
        candidates.append(os.path.join(
            root, "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"))
    except ImportError:
        pass
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono%s.ttf" % ("-Bold" if bold else ""),
        "/usr/share/fonts/truetype/liberation/LiberationMono-%s.ttf"
        % ("Bold" if bold else "Regular"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def render_state(state, gamerom=None, font_size=14, min_width=160):
    """One state as an image.

    A screenshot when the state can produce one and `gamerom` is given; otherwise the
    state's own text, typeset.
    """
    from PIL import Image, ImageDraw

    if gamerom is not None and callable(getattr(state, "save", None)):
        import tempfile
        import warnings

        # A real file with a .png suffix, not a BytesIO: the Game Boy states' `save` ends in
        # `image.save(file)`, and Pillow cannot infer a format from a stream. Passing one
        # raises, and the first version of this caught that silently, so asking for
        # screenshots quietly returned text instead. Falling back is right; doing it without
        # saying so is not.
        handle, temporary = tempfile.mkstemp(suffix=".png", prefix="planiverse-frame-")
        os.close(handle)
        try:
            state.save(gamerom, temporary)
            with Image.open(temporary) as shot:
                return shot.convert("RGB")
        except Exception as exc:
            warnings.warn(
                f"could not screenshot {type(state).__name__} from {gamerom}: "
                f"{type(exc).__name__}: {exc}. Falling back to the state's text.",
                RuntimeWarning, stacklevel=2)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    text = str(state)
    font = _font(font_size)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    left, top, right, bottom = measure.multiline_textbbox((0, 0), text or " ", font=font)
    image = Image.new(
        "RGB",
        (max(min_width, right - left + 2 * PADDING), bottom - top + 2 * PADDING),
        _BACKGROUND)
    ImageDraw.Draw(image).multiline_text((PADDING, PADDING), text, font=font, fill=_INK)
    return image


def _uniform(frames):
    """Pad frames onto a shared canvas: GIF frames must agree on a size."""
    from PIL import Image

    width = max(frame.width for frame in frames)
    height = max(frame.height for frame in frames)
    padded = []
    for frame in frames:
        if frame.size == (width, height):
            padded.append(frame)
            continue
        canvas = Image.new("RGB", (width, height), _BACKGROUND)
        canvas.paste(frame, ((width - frame.width) // 2, (height - frame.height) // 2))
        padded.append(canvas)
    return padded


def _caption(image, title, subtitle=None, colour=_INK):
    """Put a caption above a frame and a hairline around it."""
    from PIL import Image, ImageDraw

    title_font, subtitle_font = _font(13, bold=True), _font(11)
    canvas = Image.new("RGB", (image.width + 2, image.height + CAPTION_HEIGHT + 2),
                       _BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((1, 2), title, font=title_font, fill=colour)
    if subtitle:
        draw.text((1, 18), subtitle, font=subtitle_font, fill=_CAPTION_INK)
    canvas.paste(image, (1, CAPTION_HEIGHT))
    draw.rectangle([0, CAPTION_HEIGHT - 1, canvas.width - 1, canvas.height - 1],
                   outline=_BORDER)
    return canvas


def _spelling(action):
    """How the environment itself spells this action.

    Most environments hand out plain strings and this is a no-op. The Game Boy actions are
    objects built from `"a+right,16"`, and that is the spelling their environment offers and
    accepts back -- but their `__str__` renders it as `a_with_right_for_16`, which nothing
    parses. A caption is for reading a plan against the environment you would replay it in,
    so it uses the spelling that environment answers to.
    """
    return getattr(action, "action", action)


def trace_frames(trace, actions=None, env=None, gamerom=None, max_states=None,
                 font_size=14, captions=True):
    """Every state of a trace as an image, captioned unless you ask otherwise.

    `actions` labels each frame with the action that produced it — the trace is one longer
    than the plan, so frame 0 is captioned "start". `env` lets the caption say which states
    are goals and which are dead ends, which is usually the thing you are looking for.

    `max_states` thins a long trace by keeping the first, the last, and an even spread
    between: a 128-step wander from a goal-free planner is not worth 128 pages, and dropping
    the middle silently would be worse than saying so, which the captions do by keeping the
    real step numbers.
    """
    states = list(trace)
    if not states:
        raise ValueError("nothing to render: the trace is empty")

    indices = list(range(len(states)))
    if max_states is not None and len(states) > max_states:
        if max_states < 2:
            indices = [0]
        else:
            step = (len(states) - 1) / (max_states - 1)
            indices = sorted({int(round(i * step)) for i in range(max_states)})

    frames = []
    for index in indices:
        state = states[index]
        image = render_state(state, gamerom=gamerom, font_size=font_size)
        if not captions:
            frames.append(image)
            continue

        if index == 0:
            title = "start"
        else:
            action = actions[index - 1] if actions and index - 1 < len(actions) else None
            title = f"{index}. {_spelling(action)}" if action is not None \
                else f"step {index}"

        colour, note = _INK, None
        if env is not None:
            if env.is_goal(state):
                colour, note = _GOAL, "goal"
            elif env.is_terminal(state):
                colour, note = _TERMINAL, "dead end"
        if len(indices) != len(states):
            note = f"{note} · state {index} of {len(states) - 1}" if note \
                else f"state {index} of {len(states) - 1}"

        frames.append(_caption(image, title, note, colour))
    return frames


def contact_sheet(frames, columns=DEFAULT_COLUMNS, gap=PADDING):
    """Tile frames into a single image, left to right and top to bottom."""
    from PIL import Image

    if not frames:
        raise ValueError("nothing to tile")
    columns = max(1, min(columns, len(frames)))
    rows = (len(frames) + columns - 1) // columns
    cell_width = max(frame.width for frame in frames)
    cell_height = max(frame.height for frame in frames)

    sheet = Image.new("RGB",
                      (columns * cell_width + (columns + 1) * gap,
                       rows * cell_height + (rows + 1) * gap),
                      _BACKGROUND)
    for position, frame in enumerate(frames):
        row, column = divmod(position, columns)
        sheet.paste(frame, (gap + column * (cell_width + gap),
                            gap + row * (cell_height + gap)))
    return sheet


def render_trace(trace, target, actions=None, env=None, gamerom=None, duration_ms=400,
                 font_size=14, max_states=None, columns=DEFAULT_COLUMNS, per_page=None,
                 captions=None):
    """Write every state of a trace to `target`. The extension decides the format.

    - `<name>.png` (or any other single-image extension): a **contact sheet**, `columns`
      wide, every frame captioned. Returns the path written.
    - `<name>.pdf`: a multi-page document, one frame per page, or `per_page` frames tiled
      onto each page. Returns the path written.
    - `<name>.gif`: an animated GIF, one frame per state, `duration_ms` per frame, looping.
      Returns the path written.
    - anything else: treated as a directory (created if needed) receiving one independent
      PNG per state, `state-000.png` onward. Returns the list of paths, in trace order.

    `captions` defaults to on for the sheet and the PDF, where a frame without its step
    number is not much use, and off for the GIF and the directory unless you passed
    `actions` or `env`, which is a fair sign you want them labelled.
    """
    extension = os.path.splitext(str(target))[1].lower()
    if captions is None:
        captions = extension in (".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff") \
            or actions is not None or env is not None

    frames = trace_frames(trace, actions=actions, env=env, gamerom=gamerom,
                          max_states=max_states, font_size=font_size, captions=captions)

    if extension == ".gif":
        first, *rest = _uniform(frames)
        first.save(target, save_all=True, append_images=rest,
                   duration=duration_ms, loop=0)
        return target

    if extension == ".pdf":
        if per_page and per_page > 1:
            pages = [contact_sheet(frames[start:start + per_page], columns=columns)
                     for start in range(0, len(frames), per_page)]
        else:
            pages = frames
        # Pillow writes multi-page PDFs directly, so there is no plotting library here.
        # RGB is required: PDF has no alpha channel to save into.
        first, rest = pages[0].convert("RGB"), [page.convert("RGB") for page in pages[1:]]
        first.save(target, "PDF", save_all=True, append_images=rest, resolution=150.0)
        return target

    if extension:
        contact_sheet(frames, columns=columns).save(target)
        return target

    os.makedirs(target, exist_ok=True)
    digits = max(3, len(str(len(frames) - 1)))
    paths = []
    for index, frame in enumerate(frames):
        path = os.path.join(str(target), f"state-{index:0{digits}d}.png")
        frame.save(path)
        paths.append(path)
    return paths
