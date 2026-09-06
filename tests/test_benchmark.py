"""Tests for the benchmark harness.

Almost none of these run a real planner on a real environment: that is what the harness is
*for*, and a test suite that did it would take an hour. What is tested here is the harness's
own job: resolving a task list, writing jobs a cluster will accept, classifying an outcome,
and accounting for results that never arrived.
"""
import json
import os
import pathlib
import stat

import pytest

from planiverse.benchmark import analysis, catalogue, discovery, report, runner, slurm
from planiverse.benchmark.config import (
    ExperimentConfig, Limits, PlannerSpec, SlurmConfig, TaskSelection,
    format_slurm_time, parse_duration, parse_size,
)
from planiverse.benchmark.measures import MEASURES, WITHOUT_MEASURE, measure_for
from planiverse.environments import REGISTRY
from planiverse.environments.registry import EnvironmentSpec


# ------------------------------------------------------------------------------- config

@pytest.mark.parametrize("text,seconds", [
    ("30m", 1800), ("00:30:00", 1800), ("1:02:03", 3723), ("45s", 45), ("2h", 7200),
    ("1d", 86400), (90, 90), ("12:30", 750),
])
def test_durations_are_read_in_both_spellings(text, seconds):
    """SLURM writes HH:MM:SS and humans write 30m. Both turn up in a config file."""
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text,size", [
    ("8GB", 8 * 1024 ** 3), ("8G", 8 * 1024 ** 3), ("512M", 512 * 1024 ** 2),
    ("2TB", 2 * 1024 ** 4), (1024, 1024), ("1.5GB", int(1.5 * 1024 ** 3)),
])
def test_sizes_are_read(text, size):
    assert parse_size(text) == size


def test_nonsense_limits_are_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="duration"):
        parse_duration("soon")
    with pytest.raises(ValueError, match="size"):
        parse_size("lots")


def test_slurm_time_rounds_up():
    """Rounding down would hand SLURM a shorter limit than the harness is enforcing."""
    assert format_slurm_time(1800.4) == "00:30:01"
    assert format_slurm_time(59) == "00:00:59"
    assert format_slurm_time(3661) == "01:01:01"


def test_an_experiment_round_trips_through_disk(tmp_path):
    experiment = ExperimentConfig(
        name="demo",
        limits=Limits(time="5m", memory="2GB", max_expansions=1234),
        tasks=TaskSelection(tags=("puzzle",), max_instances_per_environment=3),
        slurm=SlurmConfig(partition="short", max_array_size=500),
        planners=(PlannerSpec(tag="bfws-2", planner="bfws", params={"width": 2}),),
    )
    experiment.save(tmp_path)
    loaded = ExperimentConfig.load(tmp_path)

    assert loaded.name == "demo"
    assert loaded.limits.seconds() == 300 and loaded.limits.bytes() == 2 * 1024 ** 3
    assert loaded.limits.max_expansions == 1234
    assert loaded.tasks.tags == ("puzzle",)
    assert loaded.slurm.partition == "short" and loaded.slurm.max_array_size == 500
    assert [(p.tag, p.planner, p.params) for p in loaded.planners] == \
           [("bfws-2", "bfws", {"width": 2})]


def test_the_json_uses_kebab_case(tmp_path):
    """The file is meant to be edited by hand, and it matches pyPMTEvalToolkit's spelling."""
    ExperimentConfig().save(tmp_path)
    with open(tmp_path / "exp-details.json") as handle:
        details = json.load(handle)
    assert "max-expansions" in details["limits"]
    assert "max-array-size" in details["slurm"]
    assert not any("_" in key for section in details.values()
                   if isinstance(section, dict) for key in section)


def test_a_config_from_a_later_version_still_loads(tmp_path):
    """Unknown keys are dropped rather than raising, so an old checkout can read a new file."""
    ExperimentConfig().save(tmp_path)
    path = tmp_path / "exp-details.json"
    details = json.loads(path.read_text())
    details["limits"]["something-invented-later"] = 7
    path.write_text(json.dumps(details))
    assert ExperimentConfig.load(tmp_path).limits.max_expansions == 1000000


def test_a_missing_experiment_says_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="init"):
        ExperimentConfig.load(tmp_path / "nope")


def test_a_planner_tag_has_to_be_safe_as_a_filename():
    """It becomes a filename, an sbatch job name and a column header."""
    with pytest.raises(ValueError, match="filesystem-safe"):
        PlannerSpec(tag="bfws/2", planner="bfws")
    with pytest.raises(ValueError, match="filesystem-safe"):
        PlannerSpec(tag="", planner="bfws")
    assert PlannerSpec(tag="bfws-2.strict_v1", planner="bfws").tag


def test_two_planner_files_cannot_share_a_tag(tmp_path):
    """Results are keyed on the tag, so a duplicate would have one planner overwrite the
    other's files and the report would quietly be wrong."""
    ExperimentConfig(planners=(PlannerSpec(tag="a", planner="bfws"),)).save(tmp_path)
    with open(tmp_path / "planners" / "b.json", "w") as handle:
        json.dump({"tag": "a", "planner": "iw"}, handle)
    with pytest.raises(ValueError, match="share a tag"):
        ExperimentConfig.load(tmp_path)


def test_the_defaults_cover_every_instance_and_both_versions_of_a_game():
    """A benchmark that samples a tenth of each environment is reporting on a sample it
    chose, and one that leaves out the cartridge-backed environments cannot compare a Game
    Boy environment against its pure-Python twin, which is most of the point of having
    both."""
    selection = TaskSelection()
    assert selection.max_instances_per_environment == 0, "0 means every instance"
    assert selection.include_rom_environments is True


def test_rom_paths_live_in_the_experiment_so_a_cluster_job_gets_them(tmp_path, monkeypatch):
    """A variable exported in the shell that ran `generate` is not there on the compute
    node, and the whole array would come back UNSUPPORTED."""
    from planiverse.environments import get_spec

    cartridge = tmp_path / "rom.gb"
    cartridge.write_bytes(b"\x00" * 32768)
    experiment = ExperimentConfig(roms={"puzznic_gb": str(cartridge)})
    experiment.save(tmp_path / "exp")

    loaded = ExperimentConfig.load(tmp_path / "exp")
    assert loaded.roms == {"puzznic_gb": str(cartridge)}

    spec = get_spec("puzznic_gb")
    monkeypatch.delenv(spec.rom_variable, raising=False)
    assert loaded.rom_for(spec) == str(cartridge), "the experiment's copy, with no variable"


def test_a_recorded_rom_path_that_is_not_there_falls_back_to_the_variable(tmp_path,
                                                                         monkeypatch):
    """A path recorded on the machine that wrote the config is a promise about a different
    filesystem until it is checked."""
    from planiverse.environments import get_spec

    spec = get_spec("puzznic_gb")
    elsewhere = tmp_path / "elsewhere.gb"
    elsewhere.write_bytes(b"\x00" * 32768)
    monkeypatch.setenv(spec.rom_variable, str(elsewhere))

    experiment = ExperimentConfig(roms={"puzznic_gb": str(tmp_path / "gone.gb")})
    assert experiment.rom_for(spec) == str(elsewhere)


def test_disabled_planners_are_left_out():
    experiment = ExperimentConfig(planners=(
        PlannerSpec(tag="on", planner="bfws"),
        PlannerSpec(tag="off", planner="iw", enabled=False),
    ))
    assert [spec.tag for spec in experiment.active_planners()] == ["on"]


# ---------------------------------------------------------------------------- catalogue

def test_every_catalogued_planner_can_be_built():
    for name in catalogue.names():
        assert catalogue.build(name, {}) is not None


def test_a_misspelled_parameter_is_refused_not_ignored():
    """A benchmark that drops `"widht": 2` reports a width-1 result under a width-2 name, and
    nothing downstream can tell."""
    with pytest.raises(ValueError, match="widht"):
        catalogue.build("bfws", {"widht": 2})


def test_an_unknown_planner_lists_the_ones_that_exist():
    with pytest.raises(KeyError, match="bfws"):
        catalogue.build("bwfs", {})


def test_progress_only_goes_to_the_planners_that_take_one():
    """The others would raise a TypeError, not ignore it."""
    sentinel = lambda state: 0
    assert catalogue.build("bfws", {}, progress=sentinel).progress is sentinel
    assert catalogue.build("siw", {}, progress=sentinel).progress is sentinel
    assert catalogue.build("iterated_bfws", {}, progress=sentinel).progress is sentinel
    assert catalogue.build("iw", {}, progress=sentinel) is not None


def test_only_bfws_claims_completeness():
    """`UNSOLVED` means "no plan found". Reading it as "unsolvable" is only sound for a
    complete planner, and among these that is BFWS alone. The iterated planners earn it
    per run, when they report "exhausted", the word they reserve for having covered the
    reachable space."""
    assert catalogue.is_complete("bfws")
    for name in ("iw", "iterated_width", "iterated_bfws", "siw", "fsx", "mcts"):
        assert not catalogue.is_complete(name), f"{name} is not complete"
    assert catalogue.is_complete("iterated_width", "exhausted")
    assert catalogue.is_complete("iterated_bfws", "exhausted")
    assert not catalogue.is_complete("iterated_bfws", "failed")


def test_the_randomised_planners_are_the_ones_taking_a_seed():
    for name in catalogue.names():
        takes_seed = "seed" in catalogue.describe(name)[2]
        assert catalogue.is_randomised(name) == takes_seed, name


# ----------------------------------------------------------------------------- measures

def test_every_registered_environment_is_measured_or_declared_unmeasured():
    """Otherwise "we have not written one" is indistinguishable from "we forgot this
    environment exists", and the second is the one that silently weakens a benchmark."""
    registered = {spec.name for spec in REGISTRY}
    accounted = set(MEASURES) | set(WITHOUT_MEASURE)
    assert registered - accounted == set(), "unaccounted environments"
    assert accounted - registered == set(), "measures for environments that do not exist"


def test_a_measure_is_a_number_that_falls_as_the_goal_nears():
    from planiverse.environments.gameboy_py.flipull import FlipullGame

    env = FlipullGame()
    env.set_index(0)
    state, _ = env.reset()
    measure = measure_for("flipull")
    start = measure(state)
    assert isinstance(start, (int, float)) and start > 0

    for action, successor in env.successors(state):
        if str(action) == "throw":
            assert measure(successor) < start, "clearing blocks is progress"


def test_a_dead_platformer_state_scores_worse_than_any_live_one():
    """Being dead is not "far from the goal", it is "not going to arrive"."""
    from planiverse.environments.gameboy_py.super_mario_land import SuperMarioLandGame

    env = SuperMarioLandGame()
    env.set_index(0)
    state, _ = env.reset()
    measure = measure_for("super_mario_land")
    dead = type(state)(state.tiles, state.x, state.y, 0, 0, False, state.enemies,
                       state.goal, dead=True)
    assert measure(dead) > measure(state)


def test_an_environment_without_a_measure_gets_a_flat_one():
    measure = measure_for("an_environment_with_no_measure")
    assert measure(object()) == 0


# ---------------------------------------------------------------------------- discovery

def test_task_ids_round_trip():
    assert discovery.parse_task_id(discovery.task_id("puzznic", 7)) == ("puzznic", 7)
    assert discovery.task_filename("puzznic@7") == "puzznic__7"


def test_a_malformed_task_id_is_refused():
    for bad in ("puzznic", "@3", "puzznic@x", ""):
        with pytest.raises(ValueError, match="not a task id"):
            discovery.parse_task_id(bad)


def test_even_selection_spreads_across_the_range():
    """Instance 0 of most of these environments is a tutorial, so a prefix measures little."""
    assert discovery.choose(50, 5, "even") == (0, 12, 24, 37, 49)
    assert discovery.choose(50, 5, "first") == (0, 1, 2, 3, 4)
    assert discovery.choose(3, 10, "even") == (0, 1, 2), "asking for more than there is"
    assert discovery.choose(0, 5) == ()
    assert discovery.choose(9, 1) == (0,)


def test_an_unknown_selection_strategy_is_refused():
    with pytest.raises(ValueError, match="even"):
        discovery.choose(10, 3, "random")


def test_a_rom_environment_finds_its_cartridge_through_an_environment_variable(monkeypatch,
                                                                              tmp_path):
    """The cartridges are copyrighted and cannot ship, so the path can only come from the
    user. Without somewhere to put it these environments cannot be constructed by name at
    all, which makes them invisible to anything generic, the harness included."""
    from planiverse.environments import get_spec

    spec = get_spec("puzznic_gb")
    assert spec.rom_variable == "PLANIVERSE_PUZZNIC_ROM"

    monkeypatch.delenv(spec.rom_variable, raising=False)
    assert spec.rom_path() is None and not spec.available()
    with pytest.raises(FileNotFoundError, match="cartridge"):
        spec.build()

    monkeypatch.setenv(spec.rom_variable, str(tmp_path / "not-there.gb"))
    assert spec.rom_path() is None, "a path that does not exist is not a cartridge"

    cartridge = tmp_path / "rom.gb"
    cartridge.write_bytes(b"\x00" * 32768)
    monkeypatch.setenv(spec.rom_variable, str(cartridge))
    assert spec.rom_path() == str(cartridge) and spec.available()


def test_instance_counts_are_probed_from_the_environment():
    """Probed rather than declared, so the count cannot drift out of step with the code."""
    from planiverse.environments import get_spec

    assert discovery.count_instances(get_spec("flipull")) == 32
    assert discovery.count_instances(get_spec("super_mario_land")) == 12
    assert discovery.count_instances(get_spec("puzznic")) == 128


def fake_spec(name, **kwargs):
    return EnvironmentSpec(name=name, factory="x:Y", summary="", instances="",
                           deterministic=True, state_identity="value", **kwargs)


def test_eligibility_says_why_something_was_left_out():
    """"Skipped" and "skipped because PyBoy is not installed" read very differently in a
    report."""
    selection = TaskSelection()
    ok, reason = discovery.eligible(fake_spec("rommy", needs_rom=True), selection, rom=None)
    assert not ok and "cartridge" in reason, "no cartridge is its own reason"

    ok, reason = discovery.eligible(
        fake_spec("rommy", needs_rom=True),
        TaskSelection(include_rom_environments=False), rom="/some/rom.gb")
    assert not ok and "include-rom-environments" in reason, "turned off is a different reason"

    assert discovery.eligible(fake_spec("rommy", needs_rom=True), selection,
                              rom="/some/rom.gb") == (True, "")

    ok, reason = discovery.eligible(fake_spec("nope", requires=("no_such_module",)),
                                    selection)
    assert not ok and "no_such_module" in reason

    ok, reason = discovery.eligible(
        fake_spec("tagged", tags=frozenset({"game"})), TaskSelection(tags=("policy",)))
    assert not ok and "tags" in reason

    ok, reason = discovery.eligible(
        fake_spec("skipme"), TaskSelection(exclude_environments=("skipme",)))
    assert not ok and "exclude" in reason

    assert discovery.eligible(fake_spec("fine"), selection) == (True, "")


def test_pair_up_respects_a_planners_own_restrictions():
    tasks = [{"id": "puzznic@0", "environment": "puzznic", "index": 0},
             {"id": "power_grid@0", "environment": "power_grid", "index": 0}]
    planners = [PlannerSpec(tag="everywhere", planner="bfws"),
                PlannerSpec(tag="games", planner="iw", tags=("game",)),
                PlannerSpec(tag="not-grid", planner="siw",
                            exclude_environments=("power_grid",))]
    pairs = discovery.pair_up(tasks, planners)
    covered = {(pair["planner"], pair["environment"]) for pair in pairs}
    assert ("everywhere", "power_grid") in covered
    assert ("games", "power_grid") not in covered, "power_grid is not tagged 'game'"
    assert ("games", "puzznic") in covered
    assert ("not-grid", "power_grid") not in covered


def test_explicit_tasks_bypass_the_selection():
    selection = TaskSelection(selected_tasks=("puzznic@3", "flipull@1"),
                              max_instances_per_environment=1)
    discovered = discovery.discover(selection)
    assert [task["id"] for task in discovered["tasks"]] == ["puzznic@3", "flipull@1"]


def test_an_explicit_task_for_an_unknown_environment_is_reported_not_run():
    discovered = discovery.discover(TaskSelection(selected_tasks=("nosuch@0",)))
    assert discovered["tasks"] == []
    assert discovered["skipped"][0]["reason"] == "not in the registry"


def test_the_task_list_is_written_and_read_back(tmp_path):
    discovered = discovery.discover(TaskSelection(selected_tasks=("puzznic@0",)))
    pairs = discovery.pair_up(discovered["tasks"],
                              [PlannerSpec(tag="bfws-2", planner="bfws")])
    discovery.write_tasks(tmp_path, discovered, pairs, tmp_path / "experiment")
    loaded = discovery.read_tasks(tmp_path)
    assert loaded["pairs"] == pairs
    assert "experiment" in loaded


def test_reading_a_missing_task_list_says_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="discover"):
        discovery.read_tasks(tmp_path)


# -------------------------------------------------------------------------------- slurm

@pytest.fixture
def experiment(tmp_path):
    config = ExperimentConfig(
        name="demo",
        limits=Limits(time="10m", memory="4GB", max_expansions=500),
        slurm=SlurmConfig(max_array_size=10, max_parallel_jobs=3, partition="short",
                          setup_commands=("module load python",)),
        planners=(PlannerSpec(tag="bfws-2", planner="bfws", params={"width": 2}),),
    )
    config.save(tmp_path / "experiment")
    return config


def pairs_for(count, planner="bfws-2"):
    return [{"planner": planner, "task": f"puzznic@{i}", "environment": "puzznic",
             "index": i} for i in range(count)]


def test_arrays_are_split_at_the_site_limit():
    """`MaxArraySize` is commonly 1001, and an array over it is rejected at submission with a
    message that does not name the cause."""
    assert slurm.array_chunks(500, 1000) == ((0, 500),)
    assert slurm.array_chunks(1000, 1000) == ((0, 1000),)
    assert slurm.array_chunks(1001, 1000) == ((0, 1000), (1000, 1))
    assert slurm.array_chunks(2500, 1000) == ((0, 1000), (1000, 1000), (2000, 500))
    assert slurm.array_chunks(0, 1000) == ()
    assert slurm.array_chunks(50, 0) == ((0, 50),), "no limit configured"


def test_a_long_command_list_becomes_several_sbatch_files(tmp_path, experiment):
    written = slurm.generate(tmp_path / "sandbox", pairs_for(25), experiment,
                             tmp_path / "experiment")
    assert len(written["scripts"]) == 3, "25 runs at 10 per array"
    offsets = []
    for script in written["scripts"]:
        body = open(script).read()
        offsets.append(int(body.split("OFFSET=")[1].split("\n")[0]))
    assert offsets == [0, 10, 20], "each slice knows where it starts"


def test_the_array_is_throttled(tmp_path, experiment):
    written = slurm.generate(tmp_path / "sandbox", pairs_for(5), experiment,
                             tmp_path / "experiment")
    body = open(written["scripts"][0]).read()
    assert "#SBATCH --array=0-4%3" in body, "a benchmark must not take the whole partition"


def test_slurm_limits_sit_above_the_harness_limits(tmp_path, experiment):
    """If SLURM kills the job at the same instant the harness times out, the TIMEOUT row is
    never written and a slow planner looks like an infrastructure failure."""
    written = slurm.generate(tmp_path / "sandbox", pairs_for(2), experiment,
                             tmp_path / "experiment")
    body = open(written["scripts"][0]).read()
    assert "#SBATCH --time=00:15:00" in body, "10m limit + 5m headroom"
    assert "#SBATCH --mem=5120M" in body, "4GB limit + 1GB headroom"


def test_site_directives_are_passed_through(tmp_path, experiment):
    written = slurm.generate(tmp_path / "sandbox", pairs_for(2), experiment,
                             tmp_path / "experiment")
    body = open(written["scripts"][0]).read()
    assert "#SBATCH --partition=short" in body
    assert "module load python" in body
    assert "#SBATCH --account" not in body, "unset directives are omitted, not left empty"


def test_qos_reaches_the_job(tmp_path):
    """Sites that gate submission on a QoS reject every job without one."""
    config = ExperimentConfig(
        name="q", slurm=SlurmConfig(qos="debug", partition="short"),
        planners=(PlannerSpec(tag="bfws-2", planner="bfws"),))
    written = slurm.generate(tmp_path / "sandbox", pairs_for(2), config,
                             tmp_path / "experiment")
    assert "#SBATCH --qos=debug" in open(written["scripts"][0]).read()


def test_setup_commands_run_before_the_job_and_keep_their_order(tmp_path):
    """`module load` before `source .../activate` is not the same as the other way round."""
    config = ExperimentConfig(
        name="s",
        slurm=SlurmConfig(setup_commands=("module load python/3.11",
                                          "source /shared/venv/bin/activate")),
        planners=(PlannerSpec(tag="bfws-2", planner="bfws"),))
    written = slurm.generate(tmp_path / "sandbox", pairs_for(2), config,
                             tmp_path / "experiment")
    body = open(written["scripts"][0]).read()
    assert body.index("module load python/3.11") < body.index("source /shared/venv")
    assert body.index("source /shared/venv") < body.index('eval "$COMMAND"')


def test_every_path_in_a_job_is_absolute(tmp_path, experiment, monkeypatch):
    """sbatch inherits the submitting shell's cwd, which holds until someone submits from
    somewhere else, and then a relative path fails a thousand array elements at once."""
    monkeypatch.chdir(tmp_path)
    written = slurm.generate("sandbox", pairs_for(3), experiment, "experiment")
    body = open(written["scripts"][0]).read()
    for line in body.split("\n"):
        if line.startswith(("COMMANDS=", "#SBATCH --output=", "#SBATCH --error=")):
            assert "=/" in line, f"relative path in {line!r}"
    assert open(written["commands"]["bfws-2"]["path"]).readline().count(" /") >= 2


def test_the_command_file_line_number_is_the_array_index(tmp_path, experiment):
    """What makes a failed element re-runnable by hand: line 412 is what index 412 ran."""
    written = slurm.generate(tmp_path / "sandbox", pairs_for(4), experiment,
                             tmp_path / "experiment")
    lines = open(written["commands"]["bfws-2"]["path"]).read().strip().split("\n")
    assert len(lines) == 4
    for index, line in enumerate(lines):
        assert f"--task puzznic@{index}" in line
    body = open(written["scripts"][0]).read()
    assert 'sed -n "$(( INDEX + 1 ))p"' in body, "sed is 1-indexed, the array is 0-indexed"


def test_a_job_run_by_hand_defaults_to_the_first_element(tmp_path, experiment):
    """`${SLURM_ARRAY_TASK_ID:-0}` is how you debug one element without a scheduler."""
    written = slurm.generate(tmp_path / "sandbox", pairs_for(3), experiment,
                             tmp_path / "experiment")
    assert "${SLURM_ARRAY_TASK_ID:-0}" in open(written["scripts"][0]).read()


def test_the_generated_scripts_are_executable(tmp_path, experiment):
    written = slurm.generate(tmp_path / "sandbox", pairs_for(3), experiment,
                             tmp_path / "experiment")
    for path in written["scripts"] + [written["submit_all"], written["run_local"]]:
        assert os.stat(path).st_mode & stat.S_IXUSR, f"{path} is not executable"


def test_submit_all_submits_every_script(tmp_path, experiment):
    written = slurm.generate(tmp_path / "sandbox", pairs_for(25), experiment,
                             tmp_path / "experiment")
    lines = [line for line in open(written["submit_all"]).read().split("\n")
             if line.startswith("sbatch ")]
    assert len(lines) == len(written["scripts"]) == 3


def test_per_task_scripts_are_offered_for_sites_without_arrays(tmp_path, experiment):
    written = slurm.generate(tmp_path / "sandbox", pairs_for(4), experiment,
                             tmp_path / "experiment", per_task_scripts=True)
    assert len(written["scripts"]) == 4
    body = open(written["scripts"][2]).read()
    assert "#SBATCH --array=" not in body, "a single job, not an array of one"
    assert "OFFSET=2" in body


def test_run_local_is_written_for_people_without_a_cluster(tmp_path, experiment):
    written = slurm.generate(tmp_path / "sandbox", pairs_for(3), experiment,
                             tmp_path / "experiment")
    body = open(written["run_local"]).read()
    assert "parallel" in body and "xargs" in body, "xargs is the fallback"


def test_run_local_runs_the_same_setup_as_the_cluster_jobs(tmp_path):
    """`setup_benchmark.sh` puts its virtualenv activation in setup-commands. A local run that
    ignored them used whichever interpreter was on PATH, which is exactly the difference that
    makes two runs of "the same" benchmark disagree."""
    config = ExperimentConfig(
        name="v",
        slurm=SlurmConfig(setup_commands=(". /shared/venv/bin/activate",)),
        planners=(PlannerSpec(tag="bfws-2", planner="bfws"),))
    written = slurm.generate(tmp_path / "sandbox", pairs_for(2), config,
                             tmp_path / "experiment")
    local = open(written["run_local"]).read()
    assert ". /shared/venv/bin/activate" in local
    assert local.index("activate") < local.index("FILES="), "before anything runs"
    assert ". /shared/venv/bin/activate" in open(written["scripts"][0]).read()


def test_a_seed_reaches_every_generated_command(tmp_path, experiment):
    written = slurm.generate(tmp_path / "sandbox", pairs_for(2), experiment,
                             tmp_path / "experiment", seed=17)
    assert all("--seed 17" in line for line in
               open(written["commands"]["bfws-2"]["path"]).read().strip().split("\n"))


def test_a_path_with_a_space_in_it_is_quoted(tmp_path, experiment):
    sandbox = tmp_path / "some sandbox"
    written = slurm.generate(sandbox, pairs_for(1), experiment, tmp_path / "experiment")
    line = open(written["commands"]["bfws-2"]["path"]).readline()
    assert "'" in line and "some sandbox" in line


# ------------------------------------------------------------------------------- runner

class Tiny:
    """A three-state chain, so the runner can be tested without a real environment."""

    def __init__(self, behaviour="solvable"):
        self.behaviour = behaviour
        self.states = [_TinyState(i) for i in range(3)]

    def set_index(self, index):
        if index:
            raise IndexError("one instance only")

    def reset(self):
        if self.behaviour == "explodes":
            raise RuntimeError("this environment is broken")
        return self.states[0], {}

    def successors(self, state):
        if self.behaviour == "explodes-later" and state.number == 1:
            raise RuntimeError("blew up mid-search")
        return [] if state.number == 2 else [("go", self.states[state.number + 1])]

    def is_goal(self, state):
        return self.behaviour != "unsolvable" and state.number == 2

    def is_terminal(self, state):
        return self.behaviour == "unsolvable" and state.number == 2

    def simulate(self, plan):
        return self.states[:len(plan) + 1]


class _TinyState:
    def __init__(self, number):
        self.number = number
        self.literals = frozenset({f"at({number})"})


@pytest.fixture
def tiny(monkeypatch):
    """Point the registry lookup at `Tiny` for the name `tiny`."""
    def install(behaviour="solvable"):
        spec = EnvironmentSpec(name="tiny", factory="x:Y", summary="", instances="",
                               deterministic=True, state_identity="value")
        monkeypatch.setattr(runner, "get_spec", lambda name: spec)
        monkeypatch.setattr(type(spec), "build", lambda self, **kw: Tiny(behaviour),
                            raising=False)
        return spec
    return install


def bfws_spec(tag="bfws-2"):
    return PlannerSpec(tag=tag, planner="bfws", params={"width": 2})


def test_a_solved_run_records_the_plan_and_the_statistics(tiny, tmp_path):
    tiny("solvable")
    record = runner.solve(bfws_spec(), "tiny@0", Limits(time="30s", max_expansions=100),
                          sandbox_dir=tmp_path)
    assert record["status"] == "SOLVED"
    assert record["plan_length"] == 2 and record["plan"] == ["go", "go"]
    assert record["statistics"]["expansions"] > 0
    assert record["validated"] is True
    assert os.path.isfile(runner.result_path(tmp_path, "bfws-2", "tiny@0"))


def test_an_unsolved_run_is_recorded_not_raised(tiny, tmp_path):
    tiny("unsolvable")
    record = runner.solve(bfws_spec(), "tiny@0", Limits(time="30s", max_expansions=100),
                          sandbox_dir=tmp_path)
    assert record["status"] == "UNSOLVED"
    assert record["plan_length"] is None
    assert record["complete"] is True, "BFWS, so this one really is a proof"


def test_a_crash_becomes_a_row_with_a_traceback(tiny, tmp_path):
    """A planner that dies on instance 12 must not take the other 400 runs with it."""
    tiny("explodes-later")
    record = runner.solve(bfws_spec(), "tiny@0", Limits(time="30s", max_expansions=100),
                          sandbox_dir=tmp_path)
    assert record["status"] == "ERROR"
    assert "blew up mid-search" in record["traceback"]
    assert os.path.isfile(runner.result_path(tmp_path, "bfws-2", "tiny@0"))


def test_an_environment_that_breaks_during_search_is_an_error(tiny, tmp_path):
    """It was built, so it is not UNSUPPORTED: it is broken, and that is a bug to fix
    rather than a platform it cannot run on."""
    tiny("explodes")
    record = runner.solve(bfws_spec(), "tiny@0", Limits(time="30s"), sandbox_dir=tmp_path)
    assert record["status"] == "ERROR"
    assert "broken" in record["note"]


def test_an_environment_whose_dependencies_are_missing_is_unsupported(tmp_path, monkeypatch):
    """UNSUPPORTED means "cannot run here", and it has to stay distinguishable from a
    planner failing: one is a machine without grid2op, the other is a result."""
    spec = EnvironmentSpec(name="tiny", factory="x:Y", summary="", instances="",
                           deterministic=True, state_identity="value",
                           requires=("no_such_module",))
    monkeypatch.setattr(runner, "get_spec", lambda name: spec)
    record = runner.solve(bfws_spec(), "tiny@0", Limits(time="30s"), sandbox_dir=tmp_path)
    assert record["status"] == "UNSUPPORTED"
    assert "no_such_module" in record["note"]


def test_an_index_out_of_range_is_unsupported(tiny, tmp_path):
    tiny("solvable")
    record = runner.solve(bfws_spec(), "tiny@5", Limits(time="30s"), sandbox_dir=tmp_path)
    assert record["status"] == "UNSUPPORTED"
    assert "IndexError" in record["note"]


def test_a_plan_that_does_not_replay_is_reported_as_invalid(tiny, tmp_path, monkeypatch):
    """The failure a benchmark is least able to notice otherwise: a planner reporting a plan
    it cannot reproduce."""
    tiny("solvable")
    monkeypatch.setattr(runner, "_validate", lambda env, plan: (False, "made it up"))
    record = runner.solve(bfws_spec(), "tiny@0", Limits(time="30s", max_expansions=100),
                          sandbox_dir=tmp_path)
    assert record["status"] == "INVALID"
    assert record["note"] == "made it up"
    assert record["plan_length"] == 2, "the plan is kept so it can be looked at"


def test_validation_can_be_turned_off(tiny, tmp_path, monkeypatch):
    tiny("solvable")
    monkeypatch.setattr(runner, "_validate", lambda env, plan: (False, "made it up"))
    record = runner.solve(bfws_spec(), "tiny@0",
                          Limits(time="30s", max_expansions=100, validate_plans=False),
                          sandbox_dir=tmp_path)
    assert record["status"] == "SOLVED"


def test_running_out_of_nodes_and_running_out_of_time_are_different_answers():
    """One is too slow per node, the other is looking in the wrong place."""
    from planiverse.planners.width.result import SearchResult, SearchStatistics

    out_of_nodes = SearchResult(status="out_of_budget",
                                statistics=SearchStatistics(expansions=500))
    assert runner._classify(out_of_nodes, Limits(time="30s", max_expansions=500), 0.4) \
        == "NODEOUT"

    out_of_time = SearchResult(status="out_of_budget",
                               statistics=SearchStatistics(expansions=12))
    assert runner._classify(out_of_time, Limits(time="30s", max_expansions=500), 30.0) \
        == "TIMEOUT"


def test_a_seed_is_recorded_for_a_randomised_planner(tiny, tmp_path):
    tiny("solvable")
    record = runner.solve(PlannerSpec(tag="mcts", planner="mcts",
                                      params={"iterations": 50}),
                          "tiny@0", Limits(time="30s", max_expansions=500),
                          sandbox_dir=tmp_path, seed=11)
    assert record["randomised"] is True
    assert record["seed"] == 11 and record["params"]["seed"] == 11


def test_the_hard_timeout_is_armed_on_posix(tiny, tmp_path):
    """The Budget is only checked between expansions, and the power grid spends up to 19
    seconds inside one."""
    tiny("solvable")
    record = runner.solve(bfws_spec(), "tiny@0", Limits(time="30s"), sandbox_dir=tmp_path)
    assert record["hard_timeout_armed"] is True


def test_a_memory_limit_can_only_be_tightened():
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        assert runner.apply_memory_limit(hard * 4) is False
    assert runner.apply_memory_limit(0) is False, "no limit configured"


# ----------------------------------------------------------------------------- analysis

def records_for(rows):
    """`(planner, task, status, seconds, plan_length)` tuples into result dictionaries."""
    return [{"planner": planner, "task": task,
             "environment": task.split("@")[0], "index": int(task.split("@")[1]),
             "status": status, "seconds": seconds, "plan_length": length,
             "planner_class": "bfws", "complete": True, "randomised": False,
             "has_progress_measure": True, "statistics": {"expansions": 10}}
            for planner, task, status, seconds, length in rows]


def test_a_run_that_never_happened_is_missing_not_absent(tmp_path):
    """A benchmark's most common failure is a job that never ran. Reading only the files that
    exist gives a planner that crashed on half the set excellent coverage over the half it
    survived."""
    pairs = [{"planner": "bfws-2", "task": "puzznic@0", "environment": "puzznic", "index": 0},
             {"planner": "bfws-2", "task": "puzznic@1", "environment": "puzznic", "index": 1}]
    runner.write_result(tmp_path, {"planner": "bfws-2", "task": "puzznic@0",
                                   "environment": "puzznic", "index": 0,
                                   "status": "SOLVED", "seconds": 1.0, "plan_length": 3,
                                   "statistics": {"expansions": 5}})
    loaded = analysis.load_results(tmp_path, pairs)
    assert [record["status"] for record in loaded] == ["SOLVED", "MISSING"]
    assert analysis.coverage(loaded)[0]["coverage"] == 0.5


def test_a_truncated_result_file_is_an_error_not_a_crash(tmp_path):
    """What a job killed mid-write leaves behind."""
    pairs = [{"planner": "p", "task": "puzznic@0", "environment": "puzznic", "index": 0}]
    path = os.path.join(tmp_path, "results", "p", "puzznic__0.json")
    os.makedirs(os.path.dirname(path))
    with open(path, "w") as handle:
        handle.write('{"status": "SOL')
    loaded = analysis.load_results(tmp_path, pairs)
    assert loaded[0]["status"] == "ERROR" and "unreadable" in loaded[0]["note"]


def test_runtime_is_summarised_over_solved_runs_only():
    """Averaging in the timeouts would reward a planner for failing quickly."""
    rows = analysis.coverage(records_for([
        ("fast", "a@0", "SOLVED", 1.0, 5),
        ("fast", "a@1", "TIMEOUT", 600.0, None),
        ("fast", "a@2", "SOLVED", 3.0, 7),
    ]))
    assert rows[0]["solved"] == 2
    assert rows[0]["total_seconds"] == 4.0, "the 600s timeout is not in there"
    assert rows[0]["median_seconds"] == 2.0
    assert rows[0]["mean_plan_length"] == 6.0


def test_the_status_breakdown_only_lists_what_happened():
    rows = analysis.coverage(records_for([
        ("p", "a@0", "SOLVED", 1.0, 3), ("p", "a@1", "NODEOUT", 9.0, None),
    ]))
    assert rows[0]["statuses"] == {"SOLVED": 1, "NODEOUT": 1}


def test_the_csv_has_a_row_per_run(tmp_path):
    import csv as csv_module

    records = records_for([("p", "a@0", "SOLVED", 1.0, 3), ("p", "a@1", "MISSING", None, None)])
    path = analysis.write_csv(tmp_path, records)
    with open(path) as handle:
        rows = list(csv_module.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["status"] == "SOLVED" and rows[0]["expansions"] == "10"


# ------------------------------------------------------------------------------- report

def test_the_text_report_covers_every_section():
    records = records_for([
        ("bfws-2", "puzznic@0", "SOLVED", 1.0, 4),
        ("bfws-2", "puzznic@1", "TIMEOUT", 9.0, None),
        ("iw-1", "puzznic@0", "SOLVED", 0.2, 6),
        ("iw-1", "puzznic@1", "SOLVED", 0.3, 8),
    ])
    summary = {**analysis.summarise("", records)}
    text = report.text_tables(summary)
    for heading in ("Coverage", "Outcomes", "Solved per environment"):
        assert heading in text, heading
    assert "bfws-2" in text and "iw-1" in text


def test_an_incomplete_planner_is_marked_in_the_table():
    records = records_for([("iw-1", "a@0", "UNSOLVED", 1.0, None)])
    records[0]["complete"] = False
    summary = analysis.summarise("", records)
    text = report.text_tables(summary)
    assert "iw-1 *" in text
    assert "not a proof" in text


def test_missing_runs_are_called_out_rather_than_buried():
    """A job that never ran must not be invisible: it gets its own Outcomes column."""
    records = records_for([("p", "a@0", "SOLVED", 1.0, 2), ("p", "a@1", "MISSING", None, None)])
    summary = analysis.summarise("", records)
    text = report.text_tables(summary)
    assert "MISSING" in text
    assert summary["statuses"]["MISSING"] == 1


def test_an_unmeasured_environment_is_flagged():
    records = records_for([("p", "unmeasured@0", "UNSOLVED", 1.0, None)])
    records[0]["has_progress_measure"] = False
    summary = analysis.summarise("", records)
    text = report.text_tables(summary)
    assert "†" in text and "unmeasured" in text


def test_the_latex_table_escapes_underscores():
    records = records_for([("bfws_2", "water_network@0", "SOLVED", 1.0, 4)])
    latex = report.latex_coverage(analysis.summarise("", records))
    assert "water\\_network" in latex and "bfws\\_2" in latex
    assert "\\begin{tabular}" in latex and "\\bottomrule" in latex


def test_the_report_writes_every_artefact(tmp_path):
    records = records_for([
        ("bfws-2", "puzznic@0", "SOLVED", 1.0, 4),
        ("bfws-2", "puzznic@1", "SOLVED", 2.0, 5),
        ("iw-1", "puzznic@0", "SOLVED", 0.5, 6),
        ("iw-1", "puzznic@1", "TIMEOUT", 9.0, None),
    ])
    written = report.write_report(tmp_path, analysis.summarise("", records), records)
    assert os.path.isfile(written["text"])
    assert os.path.isfile(written["latex"])
    assert os.path.isfile(written["json"])
    if report.PLOTTING:
        assert os.path.getsize(written["cactus"]) > 0
        # One overlap figure per combination. Two planners, so exactly the one pair.
        assert os.path.getsize(written["overlap_bfws-2_iw-1"]) > 0


def test_every_combination_of_planners_gets_an_overlap_figure(tmp_path):
    """Which planners are worth comparing is a property of the run, so all of them are drawn.

    Three planners means three pairs and one triple, plus one twin-runtime plot per triple.
    """
    records = records_for([
        ("bfws", "puzznic@0", "SOLVED", 1.0, 4), ("bfws", "puzznic@1", "SOLVED", 2.0, 5),
        ("iw", "puzznic@0", "SOLVED", 0.5, 6), ("iw", "puzznic@1", "TIMEOUT", 9.0, None),
        ("aaa", "puzznic@0", "TIMEOUT", 9.0, None), ("aaa", "puzznic@1", "TIMEOUT", 9.0, None),
    ])
    written = report.write_report(tmp_path, analysis.summarise("", records), records)
    if not report.PLOTTING:
        pytest.skip("matplotlib did not import")
    assert {key for key in written if key.startswith("overlap_")} == {
        "overlap_bfws_iw", "overlap_bfws_aaa", "overlap_iw_aaa", "overlap_bfws_iw_aaa"}
    assert {key for key in written if key.startswith("runtime_")} == {"runtime_bfws_iw_aaa"}
    assert "scatter" not in written
    for key, path in written.items():
        assert os.path.getsize(path) > 0, key


def test_the_stronger_planner_anchors_the_overlap_split(tmp_path):
    """`aaa` sorts first but solves nothing, so it must not be the anchor."""
    records = records_for([
        ("aaa", "a@0", "TIMEOUT", 9.0, None),
        ("zzz", "a@0", "SOLVED", 1.0, 3),
    ])
    written = report.write_report(tmp_path, analysis.summarise("", records), records)
    if not report.PLOTTING:
        pytest.skip("matplotlib did not import")
    assert "overlap_zzz_aaa" in written


def test_a_pair_does_not_double_count_the_tasks_both_solved():
    """For two planners the only pair IS the whole set; stacking both would exceed the domain."""
    overlap = {"planners": ["a", "b"],
               "rows": [{"environment": None, "tasks": 3,
                         "sets": {"a+b": 1, "a": 1, "b": 1}}]}
    segments = report._overlap_segments(overlap)
    keys = [key for _, group in segments for key in group]
    assert len(keys) == len(set(keys)), keys
    assert sorted(keys) == ["a", "a+b", "b"]


def test_plots_are_skipped_rather_than_crashing_when_nothing_solved(tmp_path):
    records = records_for([("p", "a@0", "TIMEOUT", 9.0, None)])
    written = report.write_report(tmp_path, analysis.summarise("", records), records)
    assert "cactus" not in written
    assert os.path.isfile(written["text"])


# ---------------------------------------------------------------------------------- CLI

def run_cli(*arguments):
    from planiverse.benchmark.cli import main
    return main(list(arguments))


def test_init_writes_an_experiment_and_refuses_to_clobber_it(tmp_path, capsys):
    assert run_cli("init", "--exp-dir", str(tmp_path / "exp")) == 0
    assert os.path.isfile(tmp_path / "exp" / "exp-details.json")
    assert run_cli("init", "--exp-dir", str(tmp_path / "exp")) == 1, "no silent overwrite"
    assert run_cli("init", "--exp-dir", str(tmp_path / "exp"), "--force") == 0


def test_the_default_experiment_is_a_spread_of_planners(tmp_path):
    run_cli("init", "--exp-dir", str(tmp_path / "exp"))
    loaded = ExperimentConfig.load(tmp_path / "exp")
    families = {spec.planner for spec in loaded.planners}
    assert {"iterated_width", "siw", "iterated_bfws", "fsx", "mcts"} <= families, \
        "one of each, so a report says something"
    assert not any(spec.planner in ("iw", "bfws") for spec in loaded.planners), \
        "fixed-width planners are not defaults; their iterated versions replace them"
    for spec in loaded.planners:
        assert catalogue.build(spec.planner, spec.params) is not None


def test_every_default_width_planner_is_the_iterated_version(tmp_path):
    """No default runs at a width we picked. IW and SIW iterate because novelty is a filter
    in both: a width too low loses states outright, and the right width is a property of
    the problem, so their ceiling is a bound that is never reached. BFWS iterates for a
    different reason: it is complete at every width, so its rounds are a budget strategy,
    cheap pruned rounds first, one complete round last. Its bound is the same 1000, but
    `strict` stays on, unlike the others': the strict refusal stops the pruned rounds at
    width 2, which is what hands the leftover budget to the complete round instead of
    spending it on tuple enumeration."""
    run_cli("init", "--exp-dir", str(tmp_path / "exp"))
    planners = {spec.tag: spec for spec in ExperimentConfig.load(tmp_path / "exp").planners}

    assert planners["iw"].planner == "iterated_width"
    assert planners["iw"].params["max_width"] >= 1000
    assert planners["siw"].planner == "siw"
    assert planners["siw"].params["max_width"] >= 1000
    assert planners["bfws"].planner == "iterated_bfws"
    assert planners["bfws"].params["max_width"] >= 1000
    assert planners["bfws"].params.get("strict", True) is True, \
        "strict stops the pruned rounds at width 2, saving budget for the complete round"
    assert "width" not in planners["bfws"].params, "no pinned width anywhere"


def test_init_records_an_environment_selection(tmp_path, capsys):
    """`--environments` is how the setup script's "which environments?" answer travels: it
    lands in the experiment as `include_environments`, which discovery already honours."""
    assert run_cli("init", "--exp-dir", str(tmp_path / "exp"),
                   "--environments", "puzznic, super_mario_land") == 0
    selection = ExperimentConfig.load(tmp_path / "exp").tasks
    assert selection.include_environments == ("puzznic", "super_mario_land"), \
        "recorded, with the space after the comma forgiven"


def test_init_refuses_an_unknown_environment(tmp_path, capsys):
    """Refused at init, not discovery: a typo that silently selects nothing costs an empty
    experiment, the same argument as refusing a bad cartridge path there and then."""
    assert run_cli("init", "--exp-dir", str(tmp_path / "exp"),
                   "--environments", "puzznic,super_marioland") == 2
    assert "super_marioland" in capsys.readouterr().err
    assert not os.path.exists(tmp_path / "exp" / "exp-details.json"), \
        "nothing written on a refusal"


def test_the_qos_and_setup_command_flags_reach_the_experiment(tmp_path, capsys):
    assert run_cli("init", "--exp-dir", str(tmp_path / "exp"), "--qos", "debug",
                   "--setup-command", "module load python",
                   "--setup-command", "source /shared/venv/bin/activate") == 0
    slurm_config = ExperimentConfig.load(tmp_path / "exp").slurm
    assert slurm_config.qos == "debug"
    assert slurm_config.setup_commands == ("module load python",
                                           "source /shared/venv/bin/activate")


def test_iw_iterates_its_width_rather_than_running_at_one_we_picked(tmp_path):
    """Pinning IW at a width reports it at a configuration we chose rather than the one the
    algorithm defines. The bound is a bound: IteratedWidth stops when a width solves it, the
    budget runs out, or a width covers the reachable space without pruning for novelty."""
    run_cli("init", "--exp-dir", str(tmp_path / "exp"))
    loaded = ExperimentConfig.load(tmp_path / "exp")
    iterated = [spec for spec in loaded.planners if spec.planner == "iterated_width"]
    assert iterated, "there has to be one"
    assert iterated[0].params["max_width"] >= 1000
    assert iterated[0].params["strict"] is False, \
        "strict refuses widths above 2, so it would stop at the cap we were trying to lift"


def test_environments_and_planners_list_without_a_config(capsys):
    assert run_cli("environments") == 0
    listed = capsys.readouterr().out
    assert "puzznic" in listed and "measure" in listed
    assert run_cli("planners") == 0
    assert "bfws" in capsys.readouterr().out


def test_the_whole_pipeline_runs_on_one_tiny_task(tmp_path, capsys):
    """init -> discover -> generate -> solve -> analyze -> report, for real."""
    experiment_dir, sandbox_dir = tmp_path / "exp", tmp_path / "sandbox"
    ExperimentConfig(
        name="smoke",
        limits=Limits(time="60s", memory="4GB", max_expansions=3000),
        tasks=TaskSelection(selected_tasks=("flipull@0", "flipull@1")),
        planners=(PlannerSpec(tag="bfws-2", planner="bfws", params={"width": 2}),),
    ).save(experiment_dir)

    assert run_cli("discover", "--exp-dir", str(experiment_dir),
                   "--sandbox-dir", str(sandbox_dir)) == 0
    assert run_cli("generate", "--exp-dir", str(experiment_dir),
                   "--sandbox-dir", str(sandbox_dir)) == 0
    capsys.readouterr()

    for task in ("flipull@0", "flipull@1"):
        assert run_cli("solve", "--exp-dir", str(experiment_dir),
                       "--sandbox-dir", str(sandbox_dir),
                       "--planner", "bfws-2", "--task", task) == 0
    assert "SOLVED" in capsys.readouterr().out

    assert run_cli("analyze", "--sandbox-dir", str(sandbox_dir)) == 0
    assert "Coverage" in capsys.readouterr().out
    assert run_cli("report", "--sandbox-dir", str(sandbox_dir)) == 0

    with open(sandbox_dir / "analysis" / "summary.json") as handle:
        summary = json.load(handle)
    assert summary["coverage"][0]["solved"] == 2
    assert os.path.isfile(sandbox_dir / "report" / "results.txt")


def test_solve_exits_zero_even_when_the_run_fails(tmp_path, capsys):
    """A non-zero exit would make SLURM mark the array element failed, hiding a legitimate
    result among genuine infrastructure errors."""
    experiment_dir, sandbox_dir = tmp_path / "exp", tmp_path / "sandbox"
    ExperimentConfig(
        limits=Limits(time="5s", max_expansions=1),
        planners=(PlannerSpec(tag="iw-1", planner="iw", params={"width": 1}),),
    ).save(experiment_dir)
    assert run_cli("solve", "--exp-dir", str(experiment_dir),
                   "--sandbox-dir", str(sandbox_dir),
                   "--planner", "iw-1", "--task", "puzznic@30") == 0
    assert os.path.isfile(runner.result_path(sandbox_dir, "iw-1", "puzznic@30"))


def test_solve_refuses_a_planner_tag_that_is_not_in_the_experiment(tmp_path, capsys):
    experiment_dir = tmp_path / "exp"
    ExperimentConfig(planners=(PlannerSpec(tag="a", planner="bfws"),)).save(experiment_dir)
    assert run_cli("solve", "--exp-dir", str(experiment_dir),
                   "--sandbox-dir", str(tmp_path / "sandbox"),
                   "--planner", "nope", "--task", "puzznic@0") == 2


def test_a_missing_cartridge_and_a_missing_dependency_read_differently():
    """One is fixed by installing something, the other by supplying a file only you have."""
    selection = TaskSelection()
    _, no_rom = discovery.eligible(fake_spec("g", needs_rom=True), selection, rom=None)
    _, no_dep = discovery.eligible(fake_spec("d", requires=("no_such_module",)), selection)
    assert "cartridge" in no_rom and "no_such_module" in no_dep
    assert no_rom != no_dep


def test_the_runner_builds_a_rom_environment_from_the_experiments_path(tmp_path,
                                                                      monkeypatch):
    built = {}

    class Fake:
        def set_index(self, index): raise IndexError("far enough — it was constructed")

    spec = EnvironmentSpec(name="romy", factory="x:Y", summary="", instances="",
                           deterministic=True, state_identity="value", needs_rom=True,
                           rom_variable="NOT_SET_ANYWHERE")
    monkeypatch.setattr(runner, "get_spec", lambda name: spec)
    monkeypatch.setattr(type(spec), "build",
                        lambda self, **kw: built.update(kw) or Fake(), raising=False)

    cartridge = tmp_path / "rom.gb"
    cartridge.write_bytes(b"\x00")
    record = runner.solve(bfws_spec(), "romy@0", Limits(time="10s"),
                          roms={"romy": str(cartridge)})
    assert built == {"romfile": str(cartridge)}, "the path reached the constructor"
    assert record["status"] == "UNSUPPORTED" and "IndexError" in record["note"]


def test_a_rom_environment_with_no_cartridge_anywhere_is_unsupported(tmp_path, monkeypatch):
    spec = EnvironmentSpec(name="romy", factory="x:Y", summary="", instances="",
                           deterministic=True, state_identity="value", needs_rom=True,
                           rom_variable="NOT_SET_ANYWHERE")
    monkeypatch.setattr(runner, "get_spec", lambda name: spec)
    record = runner.solve(bfws_spec(), "romy@0", Limits(time="10s"), sandbox_dir=tmp_path)
    assert record["status"] == "UNSUPPORTED"
    assert "cartridge" in record["note"]


def test_iterated_width_is_complete_only_on_the_runs_where_it_says_so(tiny, tmp_path):
    """It proves unsolvability when a width covers the reachable space without pruning for
    novelty, and proves nothing when the budget stops it. So completeness is a property of
    the run, not of the planner."""
    tiny("unsolvable")
    spec = PlannerSpec(tag="iw", planner="iterated_width",
                       params={"max_width": 1000, "strict": False})
    proved = runner.solve(spec, "tiny@0", Limits(time="30s", max_expansions=500),
                          sandbox_dir=tmp_path)
    assert proved["status"] == "UNSOLVED"
    assert proved["search_status"] == "exhausted"
    assert proved["complete"] is True

    stopped = runner.solve(spec, "tiny@0", Limits(time="30s", max_expansions=1),
                           sandbox_dir=tmp_path)
    assert stopped["status"] in ("NODEOUT", "TIMEOUT")
    assert stopped["complete"] is False


# ---------------------------------------------------------------------------- rom flags

def test_every_cartridge_environment_gets_its_own_flag():
    """Generated from the registry, so a new Game Boy environment gets a flag by existing."""
    from planiverse.benchmark.cli import rom_environments

    flags = dict((spec.name, flag) for spec, flag in rom_environments())
    assert flags == {"puzznic_gb": "puzznic", "flipull_gb": "flipull",
                     "lolo_gb": "lolo", "amazing_tater_gb": "amazing-tater",
                     "super_mario_land_gb": "super-mario-land"}


def test_the_flag_and_the_environment_variable_are_the_same_name():
    """`PLANIVERSE_PUZZNIC_ROM` is `--rom-puzznic`. Derived rather than stored, so the two
    spellings cannot drift apart."""
    from planiverse.benchmark.cli import rom_environments

    for spec, flag in rom_environments():
        assert spec.rom_variable == f"PLANIVERSE_{flag.replace('-', '_').upper()}_ROM"


def test_a_named_flag_records_the_cartridge(tmp_path, capsys):
    cartridge = tmp_path / "Puzznic.gb"
    cartridge.write_bytes(b"\x00" * 32768)
    assert run_cli("init", "--exp-dir", str(tmp_path / "exp"),
                   "--rom-puzznic", str(cartridge)) == 0
    assert ExperimentConfig.load(tmp_path / "exp").roms == \
        {"puzznic_gb": str(cartridge)}


def test_the_mario_alias_reaches_the_same_place(tmp_path, capsys):
    """`--rom-mario` stays as the short spelling of `--rom-super-mario-land`."""
    cartridge = tmp_path / "Mario.gb"
    cartridge.write_bytes(b"\x00" * 32768)
    assert run_cli("init", "--exp-dir", str(tmp_path / "exp"),
                   "--rom-mario", str(cartridge)) == 0
    assert ExperimentConfig.load(tmp_path / "exp").roms == \
        {"super_mario_land_gb": str(cartridge)}


def test_a_relative_path_and_a_tilde_are_resolved(tmp_path, monkeypatch, capsys):
    """The recorded path has to survive being read on a compute node in another directory."""
    cartridge = tmp_path / "rom.gb"
    cartridge.write_bytes(b"\x00" * 32768)
    monkeypatch.chdir(tmp_path)
    assert run_cli("init", "--exp-dir", "exp", "--rom-puzznic", "rom.gb") == 0
    recorded = ExperimentConfig.load(tmp_path / "exp").roms["puzznic_gb"]
    assert os.path.isabs(recorded) and os.path.isfile(recorded)


def test_a_cartridge_that_is_not_there_is_refused_at_the_flag(tmp_path, capsys):
    """A typo found while typing costs a second; one found after submitting four thousand
    jobs costs rather more."""
    assert run_cli("init", "--exp-dir", str(tmp_path / "exp"),
                   "--rom-flipull", str(tmp_path / "nope.gb")) == 2
    assert "no file at" in capsys.readouterr().err


def test_the_environment_keyed_form_still_works_for_scripting(tmp_path, capsys):
    cartridge = tmp_path / "rom.gb"
    cartridge.write_bytes(b"\x00" * 32768)
    assert run_cli("init", "--exp-dir", str(tmp_path / "exp"),
                   "--rom", f"flipull_gb={cartridge}") == 0
    assert ExperimentConfig.load(tmp_path / "exp").roms == {"flipull_gb": str(cartridge)}


def test_a_malformed_environment_keyed_rom_is_refused(tmp_path, capsys):
    assert run_cli("init", "--exp-dir", str(tmp_path / "exp"), "--rom", "nonsense") == 2
    assert "ENV=PATH" in capsys.readouterr().err


def test_several_cartridges_at_once(tmp_path, capsys):
    first, second = tmp_path / "a.gb", tmp_path / "b.gb"
    for path in (first, second):
        path.write_bytes(b"\x00" * 32768)
    assert run_cli("init", "--exp-dir", str(tmp_path / "exp"),
                   "--rom-puzznic", str(first), "--rom-flipull", str(second)) == 0
    assert ExperimentConfig.load(tmp_path / "exp").roms == {
        "puzznic_gb": str(first), "flipull_gb": str(second)}


def no_cartridges(monkeypatch):
    """Unset every cartridge variable the registry knows about.

    Derived from the registry rather than listed, for the same reason the tests below give:
    a second list of cartridge names falls out of step with the first one. A developer with
    `PLANIVERSE_LOLO_ROM` exported should not see a different result from CI.
    """
    for spec in REGISTRY:
        if spec.needs_rom:
            monkeypatch.delenv(spec.rom_variable, raising=False)


def test_the_missing_cartridge_hint_names_the_flags(tmp_path, capsys, monkeypatch):
    no_cartridges(monkeypatch)
    run_cli("init", "--exp-dir", str(tmp_path / "exp"))
    printed = capsys.readouterr().out
    assert "--rom-puzznic" in printed and "--rom-flipull" in printed


# ------------------------------------------------------------------------- setup script

SETUP_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "setup_benchmark.sh"


@pytest.mark.skipif(not SETUP_SCRIPT.is_file(), reason="not a source checkout")
def test_the_setup_script_is_executable_and_parses():
    import subprocess

    assert os.stat(SETUP_SCRIPT).st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(SETUP_SCRIPT)], check=True)


@pytest.mark.skipif(not SETUP_SCRIPT.is_file(), reason="not a source checkout")
def test_the_setup_script_asks_for_every_cartridge():
    """The three copyrighted ones. An experiment that silently skipped them would be
    benchmarking half of what it claims to."""
    from planiverse.benchmark.cli import rom_environments

    body = SETUP_SCRIPT.read_text()
    for spec, flag in rom_environments():
        assert spec.name in body, spec.name
        assert spec.rom_variable in body, spec.rom_variable
        assert f"--rom-{flag}" in body or f" {flag} " in body, flag
    assert "ask_rom" in body


@pytest.mark.skipif(not SETUP_SCRIPT.is_file(), reason="not a source checkout")
def test_the_setup_script_takes_the_same_rom_flags(tmp_path):
    """`--rom-*` is matched by shape and handed to `init`, so the script never holds a second
    list of cartridge names that could fall out of step with the registry."""
    import subprocess

    cartridge = tmp_path / "rom.gb"
    cartridge.write_bytes(b"\x00" * 32768)
    result = subprocess.run(
        ["bash", str(SETUP_SCRIPT), "--yes",
         "--exp-dir", str(tmp_path / "exp"), "--sandbox-dir", str(tmp_path / "sandbox"),
         "--rom-puzznic", str(cartridge),
         "--max-instances", "1", "--time", "5s",
         "--entry-point", "python -m planiverse.benchmark.cli"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(SETUP_SCRIPT.parent),
             # Every cartridge variable, taken from the registry: this test asserts on the
             # exact set of ROMs the script resolved, so one left set in the developer's own
             # shell puts an extra entry in it and fails a run that CI passes.
             **{spec.rom_variable: "" for spec in REGISTRY if spec.needs_rom}})
    assert result.returncode == 0, result.stderr[-2000:]
    assert "given on the command line" in result.stdout
    assert ExperimentConfig.load(tmp_path / "exp").roms == {"puzznic_gb": str(cartridge)}


@pytest.mark.skipif(not SETUP_SCRIPT.is_file(), reason="not a source checkout")
def test_skipping_a_cartridge_does_not_end_the_script():
    """`set -e` plus a bare `return` carrying a failed test's status ended the run on the
    first environment the user had no ROM for."""
    import subprocess

    result = subprocess.run(
        ["bash", str(SETUP_SCRIPT), "--yes", "--help"],
        capture_output=True, text=True)
    assert result.returncode == 0
    assert "--rom" not in result.stdout or "cartridge" in result.stdout.lower() \
        or "ROM" in result.stdout


def test_generate_says_so_rather_than_writing_nothing(tmp_path, capsys):
    experiment_dir = tmp_path / "exp"
    ExperimentConfig(
        tasks=TaskSelection(include_environments=("nothing-matches-this",)),
        planners=(PlannerSpec(tag="a", planner="bfws"),),
    ).save(experiment_dir)
    assert run_cli("generate", "--exp-dir", str(experiment_dir),
                   "--sandbox-dir", str(tmp_path / "sandbox")) == 1
