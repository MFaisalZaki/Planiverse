"""Render a state trace to an animated GIF or a sequence of image files.

Rendering a trace is deliberately nothing more than

    frames = [render_state(state) for state in trace]

written to disk: an animated GIF when the target ends in `.gif`, one numbered PNG per
state when the target is a directory. There used to be captions, contact sheets, PDF
pagination and trace thinning here; they dressed the frames up without adding any state
the frames did not already show, so they are gone.

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

#: Room around a typeset text state.
PADDING = 12

_BACKGROUND = (255, 255, 255)
_INK = (17, 17, 17)


def _font(size):
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
        candidates.append(os.path.join(root, "DejaVuSansMono.ttf"))
    except ImportError:
        pass
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
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


def render_trace(trace, target, gamerom=None, duration_ms=400, font_size=14):
    """Write every state of a trace to `target`.

    - `<name>.gif`: an animated GIF, one frame per state, `duration_ms` per frame,
      looping. Returns the path written.
    - anything else: treated as a directory (created if needed) receiving one
      independent PNG per state, `state-000.png` onward. Returns the list of paths,
      in trace order.
    """
    states = list(trace)
    if not states:
        raise ValueError("nothing to render: the trace is empty")
    frames = [render_state(state, gamerom=gamerom, font_size=font_size)
              for state in states]

    if str(target).lower().endswith(".gif"):
        first, *rest = _uniform(frames)
        first.save(target, save_all=True, append_images=rest,
                   duration=duration_ms, loop=0)
        return target

    os.makedirs(target, exist_ok=True)
    digits = max(3, len(str(len(frames) - 1)))
    paths = []
    for index, frame in enumerate(frames):
        path = os.path.join(str(target), f"state-{index:0{digits}d}.png")
        frame.save(path)
        paths.append(path)
    return paths
