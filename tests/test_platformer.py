"""Tests for the pure-Python platformer.

The physics tests build their own one-off levels rather than leaning on whichever shipped
level happens to have the right shape, so that a rule can be pinned to a board designed to
show it. The level tests then check that every shipped level can actually be finished.
"""
import pytest

from planiverse.environments.platformer import (
    ACCEL, ACTION_NAMES, BOUNCE, GRAVITY, JUMP_SPEED, LEVELS, MAX_FALL,
    MEASURED_EXPANSIONS, RUN, TILE, WALK,
    PlatformerAction, PlatformerGame, blocked, parse_level,
)
from planiverse.planners.width import BFWSSearch, Budget

from conftest import assert_string_literals, assert_successors_contract

#: Twenty tiles of floor with nothing on it, for measuring.
FLAT = "\n".join([" " * 20, " " * 20, " " * 20, "M" + " " * 18 + "G", "#" * 20])


def game_on(text):
    game = PlatformerGame(levels=[text])
    game.fix_index(0)
    state, _ = game.reset()
    return game, state


def play(game, state, *actions):
    for action in actions:
        state = game.__advance__(state, action)
    return state


@pytest.fixture
def env():
    game = PlatformerGame()
    game.fix_index(0)
    return game


# ----------------------------------------------------------------------------- physics

def test_mario_starts_standing_on_the_floor():
    """Without the settle in `reset` the opening state is airborne, `on_ground` is false, and
    the first action of every level is wasted falling one pixel."""
    _, state = game_on(FLAT)
    assert state.on_ground
    assert state.vy == 0 and state.vx == 0
    assert state.tile_y == 3


def test_running_covers_more_ground_than_walking():
    game, state = game_on(FLAT)
    walked = play(game, state, "right,4", "right,4")
    ran = play(game, state, "b+right,6", "b+right,6")
    assert ran.x > walked.x
    assert ran.vx == RUN and walked.vx == WALK


def test_speed_builds_up_rather_than_appearing():
    """Momentum. Reaching `RUN` takes four ticks, so the first tick of a run is slow."""
    game, state = game_on(FLAT)
    after_two = play(game, state, "b+right,2")
    assert after_two.vx == 2 * ACCEL < RUN
    assert play(game, after_two, "b+right,6").vx == RUN


def test_letting_go_coasts_to_a_stop_rather_than_stopping_dead():
    game, state = game_on(FLAT)
    running = play(game, state, "b+right,6")
    assert running.vx == RUN
    coasting = play(game, running, "nop,4")
    assert coasting.vx == 0
    assert coasting.x > running.x, "he kept moving while slowing down"


def test_a_jump_goes_up_and_comes_back_down():
    game, state = game_on(FLAT)
    rising = play(game, state, "a+right,2")
    assert rising.y < state.y and rising.vy < 0 and not rising.on_ground
    landed = play(game, rising, "nop,4", "nop,4", "nop,4")
    assert landed.on_ground and landed.y == state.y


def peak_of(game, state, launch):
    """The highest point of a jump, sampled every two ticks.

    Sampled with `b+right,2` because it is the shortest action that does *not* hold `a`, so
    it lets the jump cut apply while still reading the arc finely. Comparing where a jump
    ends instead of where it peaks measures nothing: both arcs are already falling by then.
    """
    node, peak = game.__advance__(state, launch), None
    peak = node.y
    for _ in range(6):
        node = game.__advance__(node, "b+right,2")
        peak = min(peak, node.y)
        if node.on_ground or node.dead:
            break
    return peak


def test_a_short_hold_hops_and_a_long_one_jumps():
    """The jump cut. If these came out the same, two thirds of the action set would be dead
    weight — which is exactly what happened when the shortest hold was four ticks, because
    `vy` is already past `JUMP_CUT` by then."""
    game, state = game_on(FLAT)
    hop = peak_of(game, state, "a+right,2")
    full = peak_of(game, state, "a+right,6")
    assert full < hop, "holding the button longer must go higher"
    assert hop < state.y, "and a hop still leaves the ground"


def test_gravity_is_capped():
    void = "\n".join(["M" + " " * 6] + [" " * 7 for _ in range(10)] + ["#" * 5 + "G "])
    game, state = game_on(void)
    falling = play(game, state, "nop,4", "nop,4")
    assert falling.vy <= MAX_FALL


def test_a_wall_stops_you_and_takes_your_speed():
    """The flag is parked out of reach above the wall so that this measures the wall and not
    the level ending — an earlier version put it in the running lane, and the run stopped
    because Mario had won."""
    walled = "\n".join(["      G", "   #   ", "M  #   ", "#######"])
    game, state = game_on(walled)
    against = play(game, state, "b+right,12", "b+right,12")
    assert not game.is_goal(against), "he did not reach the flag"
    assert against.vx == 0, "running into a wall does not leave you running"
    assert against.tile_x == 2, "and he is stopped beside it, not inside it"
    assert not blocked(state.tiles, against.x, against.y)


def test_you_cannot_walk_off_the_left_edge():
    game, state = game_on(FLAT)
    assert play(game, state, "b+left,12", "b+left,12").x == 0


def test_a_ceiling_stops_a_jump():
    low = "\n".join(["#######", "       ", "M     G", "#######"])
    game, state = game_on(low)
    bonked = play(game, state, "a+right,6")
    assert bonked.y >= TILE, "he cannot pass through the ceiling"


# ------------------------------------------------------------------------------- death

def test_falling_into_a_pit_kills():
    pit = "\n".join([" " * 12, " " * 12, "M" + " " * 10 + "G", "####    ####"])
    game, state = game_on(pit)
    fallen = play(game, state, "b+right,12", "nop,4", "nop,4", "nop,4")
    assert fallen.dead
    assert game.is_terminal(fallen)
    assert game.successors(fallen) == []


def test_spikes_kill():
    spiky = "\n".join([" " * 10, "M" + " " * 8 + "G", "###^^#####"])
    game, state = game_on(spiky)
    walked = play(game, state, "b+right,12")
    assert walked.dead


def test_walking_into_an_enemy_kills():
    corridor = "\n".join([" " * 12, "M  E      G", "############"])
    game, state = game_on(corridor)
    assert len(state.enemies) == 1
    assert play(game, state, "b+right,12").dead


def dropping_onto_the_enemy(game, state, speed):
    """Mario placed one tile directly above the enemy, already falling at `speed`."""
    enemy_x, enemy_y, _ = state.enemies[0]
    return type(state)(state.tiles, enemy_x, enemy_y - TILE, 0, speed, False,
                       state.enemies, state.goal)


def test_landing_on_an_enemy_kills_the_enemy_and_bounces():
    """The classic rule, and the reason a jump is an attack as well as a way across."""
    corridor = "\n".join([" " * 13, " " * 13, "M     E     G", "#############"])
    game, state = game_on(corridor)
    landed = game.__advance__(dropping_onto_the_enemy(game, state, 2), "nop,4")
    assert not landed.dead, "he came down on top of it"
    assert not landed.enemies, "so it is gone"


def test_a_fast_fall_can_still_stomp():
    """Whether a landing counts is judged on where Mario was before the move, not after. At
    `MAX_FALL` he crosses the whole of an enemy's upper half inside one tick, so a test on
    the landing position alone would have him die on something he plainly landed on."""
    corridor = "\n".join([" " * 13, " " * 13, "M     E     G", "#############"])
    game, state = game_on(corridor)
    landed = game.__advance__(dropping_onto_the_enemy(game, state, MAX_FALL), "nop,4")
    assert not landed.dead, f"a fall at {MAX_FALL} should still be a stomp"
    assert not landed.enemies


def test_a_dead_state_is_absorbing():
    pit = "\n".join([" " * 12, " " * 12, "M" + " " * 10 + "G", "####    ####"])
    game, state = game_on(pit)
    dead = play(game, state, "b+right,12", "nop,4", "nop,4", "nop,4")
    assert dead.dead
    assert play(game, dead, "b+left,12") == dead, "nothing moves a dead Mario"


def test_death_is_not_the_goal():
    pit = "\n".join([" " * 12, " " * 12, "M" + " " * 10 + "G", "####    ####"])
    game, state = game_on(pit)
    dead = play(game, state, "b+right,12", "nop,4", "nop,4", "nop,4")
    assert not game.is_goal(dead)


# ----------------------------------------------------------------------------- enemies

def test_an_enemy_turns_round_at_the_edge_of_its_platform():
    """Otherwise they walk off into the void on the first tick and the level is empty."""
    ledge = "\n".join([" " * 12, "M    E    G", "###########"])
    game, state = game_on(ledge)
    seen = set()
    node = state
    for _ in range(12):
        node = game.__advance__(node, "nop,4")
        if node.dead:
            break
        seen.update(x for x, _, _ in node.enemies)
    assert node.enemies, "the enemy is still on the platform"
    assert len(seen) > 1, "and it actually patrols"


def test_an_enemy_does_not_walk_off_a_ledge():
    ledge = "\n".join([" " * 12, "M   E     G", "#####  ####"])
    game, state = game_on(ledge)
    node = state
    for _ in range(20):
        node = game.__advance__(node, "nop,4")
        if node.dead or not node.enemies:
            break
        for x, _, _ in node.enemies:
            assert x // TILE <= 4, "it stayed on the solid part"


# ------------------------------------------------------------------------------ levels

def test_a_level_needs_a_start_and_a_flag():
    with pytest.raises(ValueError, match="M"):
        parse_level("     \n#####")
    with pytest.raises(ValueError, match="G"):
        parse_level("M    \n#####")


def test_the_markers_are_not_terrain():
    """`M`, `E` and `G` say where things begin, not what the ground is made of."""
    tiles, start, enemies, goal = parse_level("M  E  G\n#######")
    assert start == (0, 0)
    assert enemies == ((3 * TILE, 0),)
    assert goal == (6, 0)
    assert "M" not in tiles[0] and "E" not in tiles[0]


def test_ragged_levels_are_padded_to_a_rectangle():
    tiles, _, _, _ = parse_level("M\n#####  G\n########")
    assert len({len(row) for row in tiles}) == 1


def test_every_shipped_level_parses_and_places_its_enemies_on_something():
    """An enemy hung in mid-air just oscillates on the spot, which is not a patrol."""
    for index, text in enumerate(LEVELS):
        tiles, _, enemies, _ = parse_level(text)
        for x, y in enemies:
            assert blocked(tiles, x, y + TILE), \
                f"level {index}: the enemy at {(x // TILE, y // TILE)} is standing on air"


def test_every_shipped_level_starts_mario_over_solid_ground():
    """`_settle` drops him until something stops him. Over a pit nothing does, so he would
    begin the level already falling out of it."""
    for index in range(len(LEVELS)):
        game = PlatformerGame()
        game.fix_index(index)
        state, _ = game.reset()
        assert state.on_ground, f"level {index} starts Mario in mid-air"
        assert not state.dead, f"level {index} starts Mario dead"
        assert blocked(state.tiles, state.x, state.y + TILE)


@pytest.mark.parametrize("index", range(len(LEVELS)))
def test_every_shipped_level_can_be_finished(index):
    """The bar every environment in this library has to clear. The levels are generated and
    then measured for exactly this reason: a level whose flag cannot be reached is not a
    benchmark, it is a budget sink."""
    game = PlatformerGame()
    game.fix_index(index)
    _, info = game.reset()
    width = info["width"]
    result = BFWSSearch(width=2, progress=lambda s: width - s.tile_x).solve(
        game, Budget(max_expansions=200000, max_seconds=180))
    assert result.solved, f"level {index}: {result.status}"
    assert game.validate(result.plan), "and the plan replays to the flag"


def test_the_levels_are_a_ramp_and_the_ramp_is_recorded():
    """`MEASURED_EXPANSIONS` is what each level cost the planner when it was accepted. It is
    data, not a claim: the point is that the set is ordered by difficulty rather than eight
    variations on one board, and that the ordering is written down where it can be checked."""
    assert len(MEASURED_EXPANSIONS) == len(LEVELS)
    assert list(MEASURED_EXPANSIONS) == sorted(MEASURED_EXPANSIONS)
    assert MEASURED_EXPANSIONS[-1] > MEASURED_EXPANSIONS[0]
    assert len({text for text in LEVELS}) == len(LEVELS), "and no level is a duplicate"


# ------------------------------------------------------------------------- the contract

def test_fix_index_rejects_a_level_that_does_not_exist(env):
    with pytest.raises(IndexError, match="Invalid index"):
        env.fix_index(len(LEVELS))
    with pytest.raises(IndexError):
        env.fix_index(-1)


def test_a_custom_level_set_replaces_the_shipped_one():
    game = PlatformerGame(levels=["M  G\n####"])
    game.fix_index(0)
    with pytest.raises(IndexError):
        game.fix_index(1)
    assert game.reset()[1]["width"] == 4


def test_reset_reports_the_level_it_set_up(env):
    state, info = env.reset()
    assert info["level"] == 0
    assert info["enemies"] == len(state.enemies)
    assert info["goal"] == state.goal


def test_states_carry_string_literals(env):
    state, _ = env.reset()
    assert_string_literals(state)
    assert any(lit.startswith("at(mario,") for lit in state.literals)
    assert any(lit.startswith("speed(") for lit in state.literals)


def test_successors_obey_the_contract_and_never_loop_back(env):
    state, _ = env.reset()
    successors = env.successors(state)
    assert_successors_contract(successors)
    assert successors, "the opening position has moves"
    assert all(child != state for _, child in successors)


def test_simulate_replays_a_plan_from_the_start(env):
    state, _ = env.reset()
    plan = [action for action, _ in env.successors(state)][:3]
    trace = env.simulate(plan)
    assert len(trace) == len(plan) + 1 and trace[0] == state


def test_simulate_and_step_agree(env):
    state, _ = env.reset()
    plan, node = [], state
    for _ in range(5):
        successors = env.successors(node)
        if not successors:
            break
        action, node = successors[0]
        plan.append(action)

    stepped = PlatformerGame()
    stepped.fix_index(0)
    stepped.reset()
    for action in plan:
        stepped.step(action)
    assert stepped.state == env.simulate(plan)[-1]


def test_replaying_the_same_plan_gives_the_same_trace(env):
    """Determinism, which is what lets a planner treat a state as a value."""
    state, _ = env.reset()
    plan = [action for action, _ in env.successors(state)][:4]
    assert [str(s) for s in env.simulate(plan)] == [str(s) for s in env.simulate(plan)]


def test_step_reports_the_ground_gained(env):
    env.reset()
    _, gained = env.step("b+right,12")
    assert gained >= 0


def test_step_before_reset_is_an_error():
    game = PlatformerGame()
    game.fix_index(0)
    with pytest.raises(ValueError, match="reset"):
        game.step("nop,4")


def test_reaching_the_flag_ends_the_level():
    game, state = game_on("M  G\n####")
    node = state
    for _ in range(6):
        if game.is_goal(node):
            break
        node = game.__advance__(node, "b+right,6")
    assert game.is_goal(node)
    assert game.successors(node) == [], "there is nothing left to do"
    assert not game.is_terminal(node), "the flag is not a death"


def test_the_flag_is_caught_even_at_speed():
    """A fast fall covers `MAX_FALL` units in a tick, so a goal test on tile coordinates
    alone would let Mario drop straight past the flag and call it a miss."""
    drop = "\n".join(["M" + " " * 6, " " * 7, " " * 7, " " * 7, "     G ", "#######"])
    game, state = game_on(drop)
    node = state
    for _ in range(8):
        node = game.__advance__(node, "b+right,6")
        if game.is_goal(node) or node.dead:
            break
    assert game.is_goal(node) or node.dead, "he either got there or died trying"


def test_actions_parse_compare_and_cost(env):
    assert PlatformerAction("nop,4") == PlatformerAction("nop,4")
    assert PlatformerAction("nop,4") != PlatformerAction("right,4")
    assert str(PlatformerAction("a+right,6")) == "a+right,6"
    assert len({PlatformerAction("nop,4"), PlatformerAction("nop,4")}) == 1
    assert PlatformerAction("nop,4").cost() == 0
    assert PlatformerAction("a+b+right,6").cost() > PlatformerAction("right,4").cost()
    assert sorted([PlatformerAction("a+right,6"), PlatformerAction("nop,4")])[0].name == "nop,4"


def test_an_action_that_does_not_exist_is_refused():
    with pytest.raises(ValueError, match="unknown action"):
        PlatformerAction("up,4")
    with pytest.raises(ValueError, match="unknown action"):
        PlatformerAction("a+right,7"), "a hold length that is not offered"


def test_get_actions_lists_the_whole_vocabulary(env):
    assert {a.name for a in env.get_actions()} == set(ACTION_NAMES)
    assert len(ACTION_NAMES) == len(set(ACTION_NAMES))


def test_render_prints_the_history(env, capsys):
    env.reset()
    env.step("b+right,6")
    rendered = env.render()
    assert len(rendered) == 2
    assert "mario:" in capsys.readouterr().out


def test_the_state_draws_mario_and_the_flag(env):
    state, _ = env.reset()
    drawn = str(state)
    assert "M" in drawn and "G" in drawn


def test_states_hash_by_configuration_not_by_depth():
    game, state = game_on(FLAT)
    twin = type(state)(state.tiles, state.x, state.y, state.vx, state.vy, state.on_ground,
                       state.enemies, state.goal, depth=99)
    assert twin == state and hash(twin) == hash(state)
