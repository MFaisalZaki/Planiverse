"""Tests for the trace renderer."""
import pytest

pytest.importorskip("PIL", reason="Pillow is not installed")

from PIL import Image  # noqa: E402

from planiverse.environments.games.puzznic import PuzznicGame  # noqa: E402
from planiverse.rendering import render_state, render_trace  # noqa: E402


@pytest.fixture
def env():
    game = PuzznicGame()
    game.set_index(0)
    return game


@pytest.fixture
def trace(env):
    state, _ = env.reset()
    actions = [action for action, _ in env.successors(state)][:3]
    return env.simulate(actions)


# ------------------------------------------------------------------------ one state

def test_a_text_state_renders_to_something_with_ink_in_it(env):
    """The blank-image trap: an image of the right size proves nothing if it is all
    background."""
    state, _ = env.reset()
    image = render_state(state)
    assert image.width > 0 and image.height > 0
    colours = image.convert("RGB").getcolors(maxcolors=100000)
    assert len(colours) > 1, "a rendered board must not be a blank rectangle"


def test_rendering_does_not_need_a_display(env):
    state, _ = env.reset()
    assert render_state(state).mode == "RGB"


# ----------------------------------------------------------------------------- files

def test_a_gif_gets_one_frame_per_state(tmp_path, trace):
    path = render_trace(trace, tmp_path / "plan.gif")
    with Image.open(path) as gif:
        assert gif.format == "GIF"
        assert gif.n_frames == len(trace), "one frame per state, none dropped"
        assert gif.info.get("loop") == 0, "the animation loops"


def test_a_directory_gets_one_independent_file_per_state(tmp_path, trace):
    paths = render_trace(trace, tmp_path / "frames")
    assert len(paths) == len(trace)
    assert paths == sorted(paths), "filenames sort in trace order"
    for path in paths:
        with Image.open(path) as frame:
            assert frame.format == "PNG"
            colours = frame.convert("RGB").getcolors(maxcolors=100000)
            assert len(colours) > 1, "no frame may be a blank rectangle"


def test_frames_of_different_sizes_share_one_gif_canvas(tmp_path):
    """GIF frames must agree on a size, and states are under no such obligation."""

    class Sized:
        def __init__(self, text):
            self.text = text

        def __str__(self):
            return self.text

    path = render_trace([Sized("ab"), Sized("a much longer state\nover two lines")],
                        tmp_path / "plan.gif")
    with Image.open(path) as gif:
        assert gif.n_frames == 2


def test_an_empty_trace_is_refused(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        render_trace([], tmp_path / "plan.gif")


def test_the_environment_offers_render_trace_as_a_convenience(tmp_path, env, trace):
    """`env.render_trace(trace, target)` delegates to the free-standing `render_trace`."""
    path = env.render_trace(trace, tmp_path / "plan.gif")
    with Image.open(path) as gif:
        assert gif.format == "GIF"
        assert gif.n_frames == len(trace)


# ---------------------------------------------------------------------------- charts

def tower_trace():
    """A short tower defence trace: a pure-Python environment whose state is readings."""
    from planiverse.environments.games.tower_defence.environment import TowerDefenceEnv

    game = TowerDefenceEnv()
    game.set_index(0)
    state, _ = game.reset()
    actions = [action for action, _ in game.successors(state)][:2]
    return game, actions, game.simulate(actions)


def test_a_readings_state_is_charted_and_a_board_is_not(env):
    from planiverse.rendering import readings_of

    board, _ = env.reset()
    assert readings_of(board) is None, "a board renders as its own text"
    game, _, trace = tower_trace()
    readings = readings_of(trace[0])
    assert readings is not None
    for panel in readings.panels:
        row = readings.values(trace[-1])
        assert all(isinstance(row[name], (int, float)) for name in panel.series), panel.title
        assert panel.target is None or panel.target in row


def test_every_registered_reading_names_a_state_class_that_exists():
    """The registry is keyed by class, so a rename would silently drop a chart."""
    import importlib

    from planiverse.rendering import READINGS

    checked = 0
    for module, name in READINGS:
        try:
            loaded = importlib.import_module(module)
        except ImportError:
            continue                      # a simulator not installed here
        assert hasattr(loaded, name), f"{module} has no {name}"
        checked += 1
    assert checked >= 1


def test_a_charted_gif_grows_one_frame_per_state(tmp_path):
    game, actions, trace = tower_trace()
    path = render_trace(trace, tmp_path / "plan.gif", actions=actions, env=game)
    with Image.open(path) as gif:
        assert gif.format == "GIF" and gif.n_frames == len(trace)
        sizes = set()
        for index in range(gif.n_frames):
            gif.seek(index)
            sizes.add(gif.size)
        assert len(sizes) == 1, "the axes are sized for the whole trace, so frames agree"


def test_a_charted_sheet_is_one_figure_with_ink_in_it(tmp_path):
    game, actions, trace = tower_trace()
    path = render_trace(trace, tmp_path / "plan.png", actions=actions, env=game)
    with Image.open(path) as sheet:
        assert sheet.format == "PNG"
        colours = sheet.convert("RGB").getcolors(maxcolors=1000000)
        assert len(colours) > 10, "a chart has lines, markers and text in it"
        assert sheet.width < 1400, "one figure, not a strip of tiles"


def test_the_text_can_still_be_asked_for(tmp_path):
    from planiverse.rendering import trace_frames

    game, actions, trace = tower_trace()
    path = render_trace(trace, tmp_path / "text.png", actions=actions, env=game, charts=False)
    frames = trace_frames(trace, actions=actions, env=game)
    with Image.open(path) as sheet:
        assert sheet.height >= frames[0].height, "tiles of the typeset text"
    with pytest.raises(ValueError, match="no readings"):
        render_trace([object()], tmp_path / "none.png", charts=True)


def test_supplied_frames_stand_in_for_the_typeset_text(tmp_path):
    import numpy as np
    from planiverse.rendering import trace_frames

    trace = ["first", "second"]
    screens = [np.full((40, 60, 3), 200, dtype=np.uint8), Image.new("RGB", (60, 40), (10, 10, 10))]
    frames = trace_frames(trace, frames=screens, captions=False)
    assert [frame.size for frame in frames] == [(60, 40), (60, 40)]
    assert frames[0].getpixel((0, 0)) == (200, 200, 200)
    path = render_trace(trace, tmp_path / "screens.gif", frames=screens)
    with Image.open(path) as gif:
        assert gif.n_frames == 2
    with pytest.raises(ValueError, match="frames for"):
        trace_frames(trace, frames=screens[:1])


def test_kept_indices_thin_a_trace_but_keep_its_ends():
    from planiverse.rendering import kept_indices

    assert kept_indices(5) == [0, 1, 2, 3, 4]
    assert kept_indices(11, max_states=3) == [0, 5, 10]
    assert kept_indices(11, max_states=1) == [0]
