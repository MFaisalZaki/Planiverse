"""The instance generators, checked uniformly across every environment.

Every bundled environment can draw a fresh instance from a seed. What a planner is entitled
to assume about that is the same everywhere: the draw is a function of the seed, the result
is plain data that round-trips through a file, it loads through `set_instance` into the same
initial state, a bundled instance is still there afterwards, and, for the puzzles, the
instance handed out has a plan. Per-environment layout details live with the environment.
"""
import json

import pytest

from planiverse.environments import Environment, get_spec, list_environments, make
from planiverse.environments.generation import (
    GenerationError, bounded_search, draw_until, rng,
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
}

#: The environments whose generator checks each draw by search and keeps the plan it found.
PUZZLES = ("puzznic", "flipull", "lolo", "amazing_tater", "super_mario_land")


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
    for spec in list_environments():
        if spec.available():
            assert {"generate_instance", "set_instance"} <= spec.load().capabilities(), \
                f"{spec.name} has no instance generator"
    assert "generate_instance" not in Environment.capabilities(), "the base only explains"


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


@pytest.mark.parametrize("name", [n for n in PUZZLES])
def test_a_checked_puzzle_comes_with_the_plan_that_checked_it(name):
    env, _ = fresh(name, seed=4)
    assert env.witness, "a checked draw keeps the plan it was accepted on"
    assert env.validate(env.witness)
    env.set_index(0)
    assert env.witness is None, "selecting a bundled instance clears it"


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
        fewest, exhausted, plan = fewest_blocks_reachable(STAGES[index][0])
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


def test_a_generated_lolo_room_is_always_modelled_exactly():
    env, _ = fresh("lolo", seed=0)
    assert env.reset()[1]["exact"], "only Snakey and Medusa, the two enemies that never move"
