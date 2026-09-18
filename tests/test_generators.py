"""The instance generators, checked uniformly across every environment.

Every bundled environment can draw a fresh instance from a seed. What a planner is entitled
to assume about that is the same everywhere: the draw is a function of the seed, the result
is plain data that round-trips through a file, it loads through `set_instance` into the same
initial state, a bundled instance is still there afterwards, and the instance handed out
has a plan, kept as the witness it was accepted on. Per-environment layout details live
with the environment.
"""
import json

import pytest

from planiverse.environments import (
    Environment, get_spec, implements_contract, list_environments, make,
)
from planiverse.environments.generation import (
    GenerationError, bounded_search, draw_until, place, rng, scatter, walled_grid,
    width_search,
)

from conftest import assert_string_literals, assert_successors_contract

#: Options that keep a draw quick, so the suite stays fast: small boards, small networks.
FAST = {
    "puzznic": dict(width=4, height=4, colours=2, min_plan_length=2, search_limit=3000),
    "flipull": dict(width=4, height=3, types=3),
    "lolo": dict(hearts=2, framers=1, snakeys=1, min_plan_length=4, search_limit=5000),
    "amazing_tater": dict(width=6, height=5, blocks=1, pits=1, turnstiles=1,
                          min_plan_length=3, search_limit=5000),
    "super_mario_land": dict(width=24, gaps=1, platforms=1, enemies=1, hazards=0),
    "network_attack": dict(hosts=4, services=2),
    "water_network": dict(network="Net1.inp"),
    "power_grid": dict(),
    "crop_management": dict(),
    # The emulators check a draw only when asked (`solvable`), since one expansion is a
    # frame's worth of emulation per action; a near survival goal keeps the check short.
    "game_boy": dict(solvable=True, warmup=(0, 4), goal={"survive": 5}, search_limit=100),
    "retro": dict(solvable=True, warmup=(0, 5), goal={"survive": 5}, search_limit=100),
}



def environments():
    params = []
    for spec in list_environments():
        marks = [pytest.mark.slow] if spec.name == "power_grid" else []
        if not spec.available():
            marks.append(pytest.mark.skip(reason=f"{', '.join(spec.requires)} not installed"))
        params.append(pytest.param(spec.name, id=spec.name, marks=marks))
    return params


def fresh(name, seed, **options):
    env = get_spec(name).build()
    instance = env.generate_instance(seed=seed, **{**FAST[name], **options})
    return env, instance


# ------------------------------------------------------------------------- the contract

def test_every_bundled_environment_offers_the_generator():
    """The generator is contract, not capability: `implements_contract` refuses an
    environment without one, and every bundled environment passes it."""
    for spec in list_environments():
        if spec.available():
            cls = spec.load()
            assert cls.provides("generate_instance") and cls.provides("set_instance"), \
                f"{spec.name} has no instance generator"
            assert implements_contract(cls)
    assert not implements_contract(Environment), "the base only explains"
    assert not Environment.provides("generate_instance")


@pytest.mark.parametrize("name", environments())
def test_a_generated_instance_is_a_working_instance(name):
    env, instance = fresh(name, seed=1)
    state, info = env.reset()
    assert_string_literals(state)
    assert info["generated"] is True
    assert not env.is_goal(state), "a fresh instance should not start solved"
    assert not env.is_terminal(state), "nor dead"
    successors = env.successors(state)
    assert successors, "the initial state must have successors"
    assert_successors_contract(successors)
    getattr(env, "close", lambda: None)()


@pytest.mark.parametrize("name", environments())
def test_the_same_seed_gives_the_same_instance(name):
    """The draw is a function of the seed: two machines asking for seed 7 get one instance."""
    first, instance = fresh(name, seed=7)
    second, again = fresh(name, seed=7)
    assert instance == again
    assert first.reset()[0].literals == second.reset()[0].literals
    for env in (first, second):
        getattr(env, "close", lambda: None)()


@pytest.mark.parametrize("name", environments())
def test_an_instance_is_plain_data_that_round_trips_through_json(name):
    """What `generate_instance` returns can be written to a file and handed back."""
    env, instance = fresh(name, seed=3)
    expected = env.reset()[0].literals
    restored = json.loads(json.dumps(instance))
    other = get_spec(name).build()
    other.set_instance(restored)
    assert other.reset()[0].literals == expected
    for candidate in (env, other):
        getattr(candidate, "close", lambda: None)()


@pytest.mark.parametrize("name", environments())
def test_a_bundled_instance_is_still_there_afterwards(name):
    """Generating must not disturb `set_index`: the catalogue is the same before and after."""
    env, _ = fresh(name, seed=2)
    env.set_index(0)
    state, info = env.reset()
    assert info["generated"] is False
    pristine = get_spec(name).build()
    pristine.set_index(0)
    assert pristine.reset()[0].literals == state.literals
    for candidate in (env, pristine):
        getattr(candidate, "close", lambda: None)()


@pytest.mark.parametrize("name", environments())
def test_a_checked_draw_comes_with_the_plan_that_checked_it(name):
    """Every generator hands out an instance with a plan: found by search, or, for the crop
    season, the reference schedule the target is measured off."""
    env, instance = fresh(name, seed=4)
    assert env.witness, "a checked draw keeps the plan it was accepted on"
    assert env.validate(env.witness)
    assert env.witness_expansions >= 0
    if "solved_at" in instance:
        assert instance["solved_at"] == len(env.witness), \
            "a simulator scenario records the depth it was solved at, like the bundled ones"
    env.set_index(0)
    assert env.witness is None, "selecting a bundled instance clears it"
    getattr(env, "close", lambda: None)()


@pytest.mark.parametrize("name", ("puzznic", "lolo", "amazing_tater"))
def test_different_seeds_give_different_instances(name):
    drawn = {json.dumps(fresh(name, seed=seed)[1]) for seed in range(4)}
    assert len(drawn) > 1


def test_make_takes_a_seed_as_well_as_an_index():
    env = make("puzznic", seed=5, )
    assert env.index is None and env.reset()[1]["generated"]
    with pytest.raises(ValueError, match="not both"):
        make("puzznic", index=0, seed=5)


# ------------------------------------------------------------------ the shared machinery

def test_the_seed_is_the_only_source_of_randomness():
    first, seed = rng(11)
    second, _ = rng(11)
    assert seed == 11 and first.random() == second.random()
    unseeded, drawn = rng(None)
    assert isinstance(drawn, int), "an unseeded draw still records the seed it used"


def test_bounded_search_finds_a_shortest_plan_or_says_it_ran_out():
    env = make("puzznic", index=1)
    outcome = bounded_search(env, 10_000, key=str)
    assert outcome.plan is not None and env.validate(outcome.plan)
    assert len(outcome.plan) == 8, "breadth-first, so the plan is a shortest one"
    starved = bounded_search(env, 1, key=str)
    assert starved.plan is None and not starved.exhausted


def test_bounded_search_reports_an_exhausted_space():
    """A board with two colours of one block each is born terminal: nothing to search."""
    env = make("puzznic")
    env.set_instance("#####\n#1 2#\n# c #\n#####")
    outcome = bounded_search(env, 100, key=str)
    assert outcome.plan is None and outcome.exhausted


def test_draw_until_refuses_to_hand_out_an_unchecked_draw():
    with pytest.raises(GenerationError, match="no acceptable board in 3 draws"):
        draw_until(lambda attempt: attempt, lambda candidate: False, 3, "board")
    assert draw_until(lambda attempt: attempt, lambda candidate: candidate == 2, 5) == 2


# ------------------------------------------------------------------ per-environment knobs

def test_flipull_targets_are_the_fewest_blocks_a_stage_reaches():
    """The bundled stages were made the way the generator makes them, so the two agree."""
    from planiverse.environments.gameboy_py.flipull import STAGES, fewest_blocks_reachable

    for index in (0, 7, 31):
        fewest, exhausted, plan, _ = fewest_blocks_reachable(STAGES[index][0])
        assert exhausted and fewest == STAGES[index][1]
        env = make("flipull", index=index)
        assert env.validate(plan), "and the plan that reaches the fewest clears the stage"
    env, (text, target) = fresh("flipull", seed=1, clear_target=6)
    assert target == 6 and fewest_blocks_reachable(text)[0] <= 6


def test_the_generators_refuse_impossible_options():
    with pytest.raises(ValueError):
        make("lolo").generate_instance(seed=0, hearts=1, magic_hearts=2)
    with pytest.raises(ValueError):
        make("amazing_tater").generate_instance(seed=0, taters=0)
    with pytest.raises(ValueError):
        make("super_mario_land").generate_instance(seed=0, width=8)
    with pytest.raises(GenerationError):
        make("puzznic").generate_instance(seed=0, width=3, height=3, colours=2,
                                          min_plan_length=500, attempts=2, search_limit=50)


def test_an_unchecked_draw_is_allowed_but_is_asked_for():
    env = make("puzznic")
    level = env.generate_instance(seed=0, solvable=False, **FAST["puzznic"])
    assert env.witness is None and env.instance == level


def test_mario_levels_are_checked_the_way_the_shipped_ones_were():
    """BFWS(w=2) under the distance to the flag accepted the shipped levels and recorded what
    it spent as `MEASURED_EXPANSIONS`; a generated level gets the same check and the same
    number, so `min_expansions` is a floor on that ramp."""
    from planiverse.environments.gameboy_py.super_mario_land import MEASURED_EXPANSIONS

    env, _ = fresh("super_mario_land", seed=1)
    assert env.witness_expansions >= 1
    outcome = width_search(env, 20_000, lambda s: s.goal[0] - s.tile_x)
    assert outcome.expansions == env.witness_expansions, "the check is BFWS, not a stand-in"
    floor = MEASURED_EXPANSIONS[3]
    env, _ = fresh("super_mario_land", seed=2, width=60, gaps=3, platforms=3, enemies=3,
                   hazards=2, min_expansions=floor)
    assert env.witness_expansions >= floor


def test_amazing_tater_solve_and_the_generator_share_one_search():
    """`solve` found the stored solutions; the generator checks a drawn room with the same
    breadth-first search, so an accepted room passes exactly the test the bundled ones did."""
    from planiverse.environments.gameboy_py.amazing_tater import solve

    assert len(solve(0)) == 38 and solve(0, limit=3) is None
    env, _ = fresh("amazing_tater", seed=4)
    assert bounded_search(env, 5000).plan == env.witness


def test_the_board_helpers_draw_what_the_games_share():
    grid = walled_grid(3, 2, "#", ".")
    assert ["".join(row) for row in grid] == ["#####", "#...#", "#...#", "#####"]
    random_, _ = rng(0)
    cells = [(r, c) for r in (1, 2) for c in (1, 2, 3)]
    rocks = scatter(grid, random_, cells, "R", 0.5)
    assert len(rocks) == 3 and all(grid[r][c] == "R" for r, c in rocks)
    assert place(grid, random_, cells, ((0, 0, "a"), (0, 1, "b")), ".") in (True, False)
    assert not place(grid, random_, cells, ((0, 0, "x"), (0, 1, "y"), (0, 2, "z"), (0, 3, "w")),
                     "."), "a shape wider than the room never fits"


def test_a_generated_lolo_room_is_always_modelled_exactly():
    env, _ = fresh("lolo", seed=0)
    assert env.reset()[1]["exact"], "only Snakey and Medusa, the two enemies that never move"
