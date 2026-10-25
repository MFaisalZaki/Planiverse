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

Every frame is **the state's own text**, `str(state)`, typeset in a monospace font in black on
white: a board in box-drawing characters, a field from the side or a system as a schematic, with
a status line under it, and for one of OpenSpiel's own games the text OpenSpiel prints for it. A GIF is pixels, and typesetting is the
one step that turns the text into them, which is the only reason a font appears in this file
at all. The characters pictures share are in `text.py`.

A trace is any list of states: an environment's, from `env.simulate(plan)`, or an OpenSpiel
game's, from applying the plan to `game.new_initial_state()` and cloning as it goes.
"""
import math
import os

#: Columns in a contact sheet: None picks them for the plan, so the sheet comes out about
#: half as wide again as it is tall (`SHEET_ASPECT`) rather than a strip either way.
DEFAULT_COLUMNS = None
SHEET_ASPECT = 1.5

#: Room around a typeset text state, and between a caption and its state.
PADDING = 12
#: The widest a caption column grows, in pixels, before a long action name wraps.
CAPTION_MAX_WIDTH = 280

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


def render_state(state, font_size=14, min_width=160):
    """One state as an image: its own text, typeset."""
    from PIL import Image, ImageDraw

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
        canvas.paste(frame, (0, 0))
        padded.append(canvas)
    return padded


def _caption_lines(title, width):
    """The title broken into lines no wider than `width` pixels, at spaces where it can be."""
    from PIL import Image, ImageDraw

    font = _font(13, bold=True)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    fits = lambda text: measure.textlength(text, font=font) <= width  # noqa: E731
    lines, line = [], ""
    for word in title.split(" "):
        candidate = f"{line} {word}" if line else word
        if fits(candidate):
            line = candidate
            continue
        if line:
            lines.append(line)
        while not fits(word):
            cut = max(1, next(k for k in range(len(word), 0, -1) if fits(word[:k])))
            lines.append(word[:cut])
            word = word[cut:]
        line = word
    return lines + ([line] if line else [])


def _caption_width(titles, notes=()):
    """One caption column for every frame of a trace: as wide as its longest title or note,
    up to `CAPTION_MAX_WIDTH`, so that every state starts the same distance from the left."""
    from PIL import Image, ImageDraw

    font, small = _font(13, bold=True), _font(11)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    widest = max([measure.textlength(title, font=font) for title in titles]
                 + [measure.textlength(note, font=small) for note in notes if note] + [0])
    return math.ceil(min(CAPTION_MAX_WIDTH, max(widest, measure.textlength("dead end", font=font))))


def _caption(image, title, subtitle=None, colour=_INK, width=None):
    """Set a caption in a column to the left of a frame, and a hairline round the frame.

    `width` is the caption column's, shared by every frame of a trace so they line up; a
    title too long for it wraps.
    """
    from PIL import Image, ImageDraw

    width = width or _caption_width([title])
    title_font, subtitle_font = _font(13, bold=True), _font(11)
    lines = _caption_lines(title, width)
    caption_height = 16 * len(lines) + (16 if subtitle else 0)
    left = width + PADDING
    canvas = Image.new("RGB", (left + image.width + 2,
                               max(image.height + 2, caption_height + PADDING)), _BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    for k, line in enumerate(lines):
        draw.text((0, PADDING + 16 * k), line, font=title_font, fill=colour)
    if subtitle:
        draw.text((0, PADDING + 16 * len(lines)), subtitle, font=subtitle_font, fill=_CAPTION_INK)
    canvas.paste(image, (left + 1, 1))
    draw.rectangle([left, 0, left + image.width + 1, image.height + 1], outline=_BORDER)
    return canvas


def outcome(state, env=None):
    """"goal", "dead end" or None: what a caption says about a state.

    `env` answers when it is given. Otherwise an OpenSpiel state answers for itself: a terminal
    state with a positive return is a goal, any other terminal state a dead end.
    """
    if env is not None:
        return "goal" if env.is_goal(state) else "dead end" if env.is_terminal(state) else None
    is_terminal, returns = getattr(state, "is_terminal", None), getattr(state, "returns", None)
    if callable(is_terminal) and callable(returns) and is_terminal():
        return "goal" if returns()[0] > 0 else "dead end"
    return None


def kept_indices(count, max_states=None):
    """Which states of a trace of `count` to show: all of them, or the first, the last and
    an even spread between when `max_states` is fewer."""
    if max_states is None or count <= max_states:
        return list(range(count))
    if max_states < 2:
        return [0]
    step = (count - 1) / (max_states - 1)
    return sorted({int(round(i * step)) for i in range(max_states)})


def trace_frames(trace, actions=None, env=None, max_states=None, font_size=14,
                 captions=True, frames=None):
    """Every state of a trace as an image, captioned unless you ask otherwise.

    `actions` labels each frame with the action that produced it — the trace is one longer
    than the plan, so frame 0 is captioned "start". `env` lets the caption say which states
    are goals and which are dead ends, which is usually the thing you are looking for.

    `frames` supplies the picture of each state, one PIL image or `(height, width, 3)` array
    per state of the trace, for an environment whose states have a screen of their own (the
    emulators); without it a state is typeset from its text.

    `max_states` thins a long trace by keeping the first, the last, and an even spread
    between: a 128-step wander from a goal-free planner is not worth 128 pages, and dropping
    the middle silently would be worse than saying so, which the captions do by keeping the
    real step numbers.
    """
    states = list(trace)
    if not states:
        raise ValueError("nothing to render: the trace is empty")
    if frames is not None:
        frames = list(frames)
        if len(frames) != len(states):
            raise ValueError(f"{len(frames)} frames for {len(states)} states")
    indices = kept_indices(len(states), max_states)

    def title(index):
        if index == 0:
            return "start"
        action = actions[index - 1] if actions and index - 1 < len(actions) else None
        return f"{index}. {action}" if action is not None else f"step {index}"

    # One size for every picture of the trace, so that their frames line up.
    images = _uniform([_picture(frames[index]) if frames is not None
                       else render_state(states[index], font_size=font_size)
                       for index in indices])
    if not captions:
        return images

    said = {index: outcome(states[index], env) for index in indices}

    def note(index):
        if len(indices) != len(states):
            where = f"state {index} of {len(states) - 1}"
            return f"{said[index]} · {where}" if said[index] else where
        return said[index]

    width = _caption_width([title(index) for index in indices], [note(i) for i in indices])
    pictures = []
    for index, image in zip(indices, images):
        colour = {"goal": _GOAL, "dead end": _TERMINAL}.get(said[index], _INK)
        pictures.append(_caption(image, title(index), note(index), colour, width=width))
    return pictures


def _picture(frame):
    """A supplied frame as an RGB PIL image: it may be one already, or an array."""
    from PIL import Image

    if not isinstance(frame, Image.Image):
        frame = Image.fromarray(frame)
    return frame.convert("RGB")


def _columns(frames, gap):
    """The columns that bring a sheet of `frames` nearest `SHEET_ASPECT`, wide to tall; of
    two as near, the one leaving fewer cells empty on the last row."""
    width = max(frame.width for frame in frames) + gap
    height = max(frame.height for frame in frames) + gap
    count = len(frames)

    def badness(columns):
        rows = -(-count // columns)
        aspect = columns * width / (rows * height)
        return abs(math.log(aspect / SHEET_ASPECT)), rows * columns - count

    return min(range(1, count + 1), key=badness)


def contact_sheet(frames, columns=DEFAULT_COLUMNS, gap=PADDING):
    """Tile frames into a single image, left to right and top to bottom, each pinned to the
    top left of its cell. A row is as tall as its tallest frame and a column as wide as its
    widest; `columns` left as None is picked so the sheet is about `SHEET_ASPECT` wide to
    tall."""
    from PIL import Image

    if not frames:
        raise ValueError("nothing to tile")
    if columns is None:
        columns = _columns(frames, gap)
    columns = max(1, min(columns, len(frames)))
    rows = [frames[start:start + columns] for start in range(0, len(frames), columns)]
    widths = [max(row[k].width for row in rows if k < len(row)) for k in range(columns)]
    heights = [max(frame.height for frame in row) for row in rows]

    sheet = Image.new("RGB", (sum(widths) + (columns + 1) * gap, sum(heights) + (len(rows) + 1) * gap),
                      _BACKGROUND)
    y = gap
    for row, height in zip(rows, heights):
        x = gap
        for frame, width in zip(row, widths):
            sheet.paste(frame, (x, y))
            x += width + gap
        y += height + gap
    return sheet


def render_trace(trace, target, actions=None, env=None, duration_ms=400, font_size=14,
                 max_states=None, columns=DEFAULT_COLUMNS, per_page=None, captions=None,
                 frames=None):
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

    `frames`, one picture per state, replaces the typeset text with the state's own screen
    (see `trace_frames`); the emulator environments pass their consoles' frames this way.
    """
    extension = os.path.splitext(str(target))[1].lower()
    if captions is None:
        captions = extension in (".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff") \
            or actions is not None or env is not None

    frames = trace_frames(trace, actions=actions, env=env, max_states=max_states,
                          font_size=font_size, captions=captions, frames=frames)

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

