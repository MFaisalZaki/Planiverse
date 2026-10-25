"""Tests for the trace renderer."""
import pytest

pytest.importorskip("PIL", exc_type=ImportError, reason="Pillow is not installed")

from PIL import Image  # noqa: E402

from planiverse.environments.games.puzznic.environment import PuzznicGame  # noqa: E402
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


# ---------------------------------------------------------------------------- pictures

class Drawn:
    """A state that draws a picture of itself, and spells itself otherwise."""

    def __str__(self):
        return "spelled"

    def picture(self):
        return "┌──┐\n│☻ │\n└──┘"


def test_a_frame_typesets_the_states_picture_where_it_draws_one(env):
    from planiverse.rendering import state_text

    assert state_text(Drawn()) == Drawn().picture()
    board, _ = env.reset()
    assert state_text(board) == (board.picture() if hasattr(board, "picture") else str(board))
    assert render_state(Drawn()).size != render_state("spelled").size


def test_the_text_helpers_join_their_lines():
    from planiverse.rendering.text import Canvas, bar, boxed, side_by_side, tank

    assert boxed(["###", "#@#", "###"], glyphs={"@": "☻"}) == "┌───┐\n│ ☻ │\n└───┘"
    canvas = Canvas(5, 3)
    canvas.rect(0, 0, 4, 2)
    canvas.vline(2, 0, 2)
    assert str(canvas) == "┌─┬─┐\n│ │ │\n└─┴─┘"
    assert bar(5, 10, 4) == "██░░" and bar(20, 10, 2) == "██" and bar(1, 0, 3) == "░░░"
    assert tank(1.0, 2, 2) == ["│██│", "│██│", "└──┘"]
    assert side_by_side(["a\nb", "cc"], gap=1) == "a cc\nb"


def test_an_openspiel_trace_renders_and_its_end_is_captioned(tmp_path):
    """A trace of OpenSpiel states renders like an environment's, and the caption reads the
    goal off the game's own return."""
    pyspiel = pytest.importorskip("pyspiel")
    import planiverse.spiel as spiel

    game = pyspiel.load_game("planiverse_tower_defence", {"index": 0})
    actions, result = spiel.solve(game, "bfws")
    assert result.status == "solved"
    trace = [game.new_initial_state()]
    for action in actions:
        trace.append(trace[-1].clone())
        trace[-1].apply_action(action)
    from planiverse.rendering import outcome

    assert outcome(trace[-1]) == "goal" and outcome(trace[0]) is None
    assert render_trace(trace, tmp_path / "plan.png").exists()
    board = pyspiel.load_game("deep_sea").new_initial_state()
    assert render_trace([board], tmp_path / "deep_sea.png").exists(), "OpenSpiel's own text"


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
