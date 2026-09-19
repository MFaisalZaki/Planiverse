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
