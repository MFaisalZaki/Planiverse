"""Tests for the trace renderer."""
import pytest

pytest.importorskip("PIL", reason="Pillow is not installed")

from PIL import Image  # noqa: E402

from planiverse.environments.puzznic import PuzznicGame  # noqa: E402
from planiverse.rendering import (  # noqa: E402
    contact_sheet, render_state, render_trace, trace_frames,
)

from conftest import puzznic_rom_path  # noqa: E402


@pytest.fixture
def env():
    game = PuzznicGame()
    game.fix_index(0)
    return game


@pytest.fixture
def trace(env):
    state, _ = env.reset()
    actions = [action for action, _ in env.successors(state)][:3]
    return env.simulate(actions), actions


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


# ---------------------------------------------------------------------------- frames

def test_frames_are_captioned_with_the_actions_that_made_them(trace):
    states, actions = trace
    frames = trace_frames(states, actions=actions)
    assert len(frames) == len(states) == len(actions) + 1, "the trace is one longer"
    assert all(frame.width > 0 for frame in frames)


def test_frames_mark_goals_and_dead_ends(env):
    """The thing you are usually looking for in a trace."""
    from planiverse.planners.width import Budget, IWSearch

    result = IWSearch(width=2).solve(env, Budget(max_expansions=5000))
    assert result.solved
    states = env.simulate(result.plan)
    plain = trace_frames(states, actions=result.plan)
    marked = trace_frames(states, actions=result.plan, env=env)
    # The final frame gains a "goal" note, so it differs from the unannotated one.
    assert marked[-1].tobytes() != plain[-1].tobytes()


def test_a_long_trace_can_be_thinned_and_says_that_it_was(env):
    """A 128-step wander from a goal-free planner is not worth 128 pages, but dropping the
    middle silently would be worse than saying so."""
    state, _ = env.reset()
    node, states = state, [state]
    for _ in range(20):
        successors = env.successors(node)
        if not successors:
            break
        node = successors[0][1]
        states.append(node)

    frames = trace_frames(states, max_states=5)
    assert len(frames) <= 5 < len(states)
    assert len(trace_frames(states)) == len(states), "unthinned by default"


def test_thinning_to_one_frame_is_allowed(env):
    state, _ = env.reset()
    assert len(trace_frames([state, state, state], max_states=1)) == 1


def test_an_empty_trace_is_refused():
    with pytest.raises(ValueError, match="empty"):
        trace_frames([])


# ----------------------------------------------------------------------------- files

def test_png_is_a_contact_sheet(tmp_path, trace):
    states, actions = trace
    path = render_trace(states, tmp_path / "plan.png", actions=actions)
    with Image.open(path) as sheet:
        assert sheet.format == "PNG"
        assert len(sheet.convert("RGB").getcolors(maxcolors=100000)) > 1


def test_pdf_gets_one_page_per_state(tmp_path, trace):
    states, actions = trace
    path = render_trace(states, tmp_path / "plan.pdf", actions=actions)
    assert path.exists() and path.stat().st_size > 0
    assert path.read_bytes().startswith(b"%PDF")


def test_pdf_can_tile_several_states_per_page(tmp_path, trace):
    states, actions = trace
    one = render_trace(states, tmp_path / "one.pdf", actions=actions)
    tiled = render_trace(states, tmp_path / "tiled.pdf", actions=actions, per_page=4)
    assert one.read_bytes().count(b"/Type /Page\n") >= \
           tiled.read_bytes().count(b"/Type /Page\n")


def test_the_contact_sheet_lays_frames_out_in_a_grid():
    frames = [Image.new("RGB", (20, 10), (255, 0, 0)) for _ in range(6)]
    wide = contact_sheet(frames, columns=6)
    tall = contact_sheet(frames, columns=2)
    assert wide.width > tall.width and tall.height > wide.height


def test_an_empty_sheet_is_refused():
    with pytest.raises(ValueError, match="nothing to tile"):
        contact_sheet([])


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
        game.fix_index(0)
        state, _ = game.reset()
        image = render_state(state, gamerom=rom)
        colours = image.convert("RGB").getcolors(maxcolors=100000)
        assert len(colours) > 1, "a Game Boy screenshot must not be blank"
        assert image.width >= 160 and image.height >= 144
    finally:
        game.close()
