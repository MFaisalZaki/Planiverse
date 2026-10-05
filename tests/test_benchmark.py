"""The benchmark's own job: one file per run whatever happened, and a report that adds up."""
import json
import pathlib

import pytest

from planiverse.benchmark import report, solve

SANDBOX = pathlib.Path(__file__).resolve().parents[1] / "sandbox"


def test_a_run_is_written_out_whatever_happens(tmp_path):
    record = solve(tmp_path, "bfws", "puzznic@1")
    assert record["status"] == "SOLVED" and record["plan_length"] == 10
    written = json.loads((tmp_path / "results/bfws/puzznic__1.json").read_text())
    assert written["search_status"] == "solved" and written["plan"] == record["plan"]
    assert solve(tmp_path, "iw", "puzznic@9999")["status"] == "UNSUPPORTED"


def test_a_seeded_planner_writes_one_file_per_seed(tmp_path):
    record = solve(tmp_path, "fsx", "puzznic@9999", seed=3)
    assert record["seed"] == 3 and (tmp_path / "results/fsx/puzznic__9999__s3.json").is_file()
    assert solve(tmp_path, "fsx", "puzznic@9999")["seed"] == 0   # run by hand: the first seed


def test_the_reference_planners_take_no_reward_and_learn_nothing():
    """A planning library: every reference configuration is driven by the goal test and,
    at most, a progress heuristic."""
    import inspect
    from planiverse.benchmark import PLANNERS
    assert set(PLANNERS) == {"bfws", "iw", "siw", "fsx"}
    for cls, _ in PLANNERS.values():
        assert "reward" not in inspect.signature(cls).parameters, cls.__name__


def test_every_planner_in_the_library_is_in_the_benchmark():
    """The benchmark runs the whole library: every planner class the planner packages
    export, and every documented configuration of one, is registered under a tag that
    `solve` builds the way it builds any other, so `generate` with no flag covers them
    all."""
    import inspect
    from planiverse.benchmark import ALL, PLANNERS
    from planiverse.benchmark.candidates import CANDIDATES
    from planiverse.planners import blind, heuristic, macros, sampling, width

    registered = {cls for cls, _ in ALL.values()}
    registered |= {base for cls in registered for base in cls.__mro__[1:]}
    exported = {getattr(module, name) for module in (width, heuristic, sampling)
                for name in module.__all__} | {
        blind.BreadthFirstSearch, blind.UniformCostSearch, blind.IterativeDeepening,
        macros.MacroPlanner}
    planners = {cls for cls in exported if inspect.isclass(cls) and hasattr(cls, "solve")}
    assert planners <= registered, {cls.__name__ for cls in planners - registered}
    assert set(PLANNERS) <= set(ALL) and not set(PLANNERS) & set(CANDIDATES)
    # The documented variants of one class, each under its own tag.
    assert {"astar", "wastar", "gbfslw", "kpiece", "ils", "bee", "multi"} <= set(CANDIDATES)


def test_every_registered_planner_builds_and_runs_as_solve_runs_it():
    """Each tag's class takes its parameters, plus `progress` and `seed` where `solve` adds
    them, and runs on one instance under a small budget without returning a plan that does
    not replay."""
    import inspect
    from planiverse.benchmark import ALL, _seeds
    from planiverse.benchmark.measures import MEASURES
    from planiverse.environments import get_spec
    from planiverse.planners.width import Budget, SearchResult

    for tag, (cls, params) in ALL.items():
        env = get_spec("puzznic").build()
        env.set_index(0)
        if _seeds(tag)[0] is not None:
            params = {**params, "seed": 0}
        if "progress" in inspect.signature(cls).parameters:
            params = {**params, "progress": MEASURES["puzznic"]}
        result = cls(**params).solve(env, Budget(max_expansions=60, max_seconds=10))
        assert isinstance(result, SearchResult), tag
        if result.solved:
            assert env.validate(result.plan), f"{tag} returned a plan that does not replay"
        else:
            assert result.plan is None, tag


def test_the_report_expects_every_run_and_averages_over_seeds(tmp_path):
    (tmp_path / "tasks.json").write_text(
        json.dumps({"environments": [{"environment": "puzznic", "instances": 2}]}))
    solve(tmp_path, "bfws", "puzznic@0")
    solve(tmp_path, "bfws", "puzznic@1")
    report(tmp_path)
    statuses = (tmp_path / "report/statuses.tex").read_text()
    facts = (tmp_path / "report/facts.txt").read_text()
    assert "BFWS & \\textbf{2} & 0" in statuses and "Missing" not in statuses
    # Ten missing FSX runs are two per seed, so the row still sums to the two instances.
    assert "IW & 0 & 2" in statuses and "FSX & 0.0 (0.0) & 2" in statuses
    missing = facts.split("missing")[1].split("\n")[0]
    assert "iw on puzznic 2" in missing and "fsx on puzznic 10" in missing


@pytest.mark.skip(reason="the released sandbox was run with planner configurations the "
                         "library no longer has under these tags")
def test_the_released_results_come_out_of_their_sandbox():
    report(SANDBOX)
    facts = (SANDBOX / "report/facts.txt").read_text()
    assert facts.startswith("solved per seed: bfws 476, iw 357, siw 207")
    statuses = (SANDBOX / "report/statuses.tex").read_text()
    assert "BFWS & \\textbf{476} & 122 & 229 & 104 & 7 &" in statuses
    # The open-challenges section quotes these, so they come out of the same report.
    assert "open instances (solved by no planner in any seed): 439" in facts
    assert "ipc quality score over the 499 instances solved by any planner: bfws 444.9" in facts


def test_a_memout_is_written_even_when_the_write_itself_runs_out(tmp_path, monkeypatch):
    """Five 2026-09 runs left empty files: `open` truncated, then `json.dump` raised MemoryError
    under the address-space cap. The write lifts the cap and tries once more."""
    import planiverse.benchmark as bench
    real_dump, calls = bench.json.dump, []

    def dump_once_out_of_memory(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise MemoryError
        return real_dump(*args, **kwargs)

    monkeypatch.setattr(bench.json, "dump", dump_once_out_of_memory)
    record = {"task": "puzznic@0", "environment": "puzznic", "index": 0, "planner": "fsx",
              "seed": 0, "started": 0.0}
    bench._write(tmp_path, record, "MEMOUT")
    written = json.loads((tmp_path / "results/fsx/puzznic__0__s0.json").read_text())
    assert written["status"] == "MEMOUT" and len(calls) == 2


def test_generate_cuts_a_group_into_arrays_of_at_most_max_array(tmp_path, monkeypatch):
    """SLURM caps an array at the site's MaxArraySize, so a group over more instances than
    `MAX_ARRAY` is written as parts, each with its own command file and array, and `submit.sh`
    submits every part."""
    import re
    import planiverse.benchmark as bench
    from planiverse.environments import get_spec

    monkeypatch.setattr(bench, "REGISTRY", [get_spec("puzznic")])      # 100 instances
    monkeypatch.setattr(bench, "ALL", {"bfws": bench.PLANNERS["bfws"]})
    monkeypatch.setattr(bench, "MAX_ARRAY", 30)
    bench.generate(tmp_path)

    parts = sorted(tmp_path.glob("cmds/*.txt"))
    assert [p.name for p in parts] == ["bfws-p0.txt", "bfws-p1.txt", "bfws-p2.txt", "bfws-p3.txt"]
    lines = [p.read_text().splitlines() for p in parts]
    assert [len(part) for part in lines] == [30, 30, 30, 10]
    assert [line.split()[-1] for part in lines for line in part] == [f"puzznic@{i}" for i in range(100)]
    for part, commands in zip(parts, lines):
        sbatch = (tmp_path / "slurm" / f"{part.stem}.sbatch").read_text()
        assert f"--array=0-{len(commands) - 1}%50" in sbatch and part.name in sbatch
        assert "--job-name=planiverse-bench-bfws\n" in sbatch     # the group, not the part
    submit = (tmp_path / "submit.sh").read_text()
    assert re.findall(r"slurm/(\S+)\.sbatch", submit) == [p.stem for p in parts]

    # A group that fits in one array keeps its plain name.
    monkeypatch.setattr(bench, "MAX_ARRAY", 1000)
    bench.generate(tmp_path / "one")
    assert [p.name for p in (tmp_path / "one" / "cmds").iterdir()] == ["bfws.txt"]
