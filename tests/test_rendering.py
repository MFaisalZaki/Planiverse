"""Tests for the trace renderer."""
import pytest

pytest.importorskip("PIL", reason="Pillow is not installed")

from PIL import Image  # noqa: E402

from planiverse.environments.gameboy_py.puzznic import PuzznicGame  # noqa: E402
from planiverse.rendering import render_state, render_trace  # noqa: E402

from conftest import puzznic_rom_path  # noqa: E402


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


def test_rendering_does_not_need_a_rom_or_a_display(env):
    state, _ = env.reset()
    assert render_state(state, gamerom=None).mode == "RGB"


def test_a_broken_screenshot_falls_back_but_says_so(env):
    """It used to fall back silently, so asking for screenshots quietly returned text."""
    state, _ = env.reset()

    class Unscreenshotable:
        literals = state.literals

        def __str__(self):
            return "some board"

        def save(self, gamerom, file, scale=4):
            raise RuntimeError("no emulator here")

    with pytest.warns(RuntimeWarning, match="could not screenshot"):
        image = render_state(Unscreenshotable(), gamerom="pretend.gb")
    assert image.width > 0


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
    """`env.render_trace(trace, target)` delegates, filling in the environment's own
    cartridge when it has one; this environment has none, so it renders as text."""
    path = env.render_trace(trace, tmp_path / "plan.gif")
    with Image.open(path) as gif:
        assert gif.format == "GIF"
        assert gif.n_frames == len(trace)


# ------------------------------------------------------------------- the real console

@pytest.mark.skipif(puzznic_rom_path() is None,
                    reason="set PLANIVERSE_PUZZNIC_ROM to render real screenshots")
def test_a_game_boy_state_screenshots_the_actual_console():
    """`GBState.save` ticked the emulator with rendering *off*, so the frame buffer was
    never filled and every screenshot came out a blank white rectangle. It had always done
    that; nothing looked at the pixels until now."""
    pytest.importorskip("pyboy", reason="pyboy is not installed")
    from planiverse.environments.gameboy.puzznic_gb import PuzznicGBEnv

    rom = puzznic_rom_path()
    game = PuzznicGBEnv(rom)
    try:
        game.set_index(0)
        state, _ = game.reset()
        image = render_state(state, gamerom=rom)
        colours = image.convert("RGB").getcolors(maxcolors=100000)
        assert len(colours) > 1, "a Game Boy screenshot must not be blank"
        assert image.width >= 160 and image.height >= 144
    finally:
        game.close()


@pytest.mark.skipif(puzznic_rom_path() is None,
                    reason="set PLANIVERSE_PUZZNIC_ROM to render real screenshots")
def test_env_render_trace_supplies_its_own_cartridge(tmp_path):
    """On a cartridge-backed environment, `env.render_trace` needs no `gamerom=`: the
    frames come out at console resolution, not as typeset text."""
    pytest.importorskip("pyboy", reason="pyboy is not installed")
    from planiverse.environments.gameboy.puzznic_gb import PuzznicGBEnv

    game = PuzznicGBEnv(puzznic_rom_path())
    try:
        game.set_index(0)
        state, _ = game.reset()
        action, _next = game.successors(state)[0]
        paths = game.render_trace(game.simulate([action]), tmp_path / "frames")
        with Image.open(paths[0]) as frame:
            assert frame.width >= 160 and frame.height >= 144
    finally:
        game.close()
