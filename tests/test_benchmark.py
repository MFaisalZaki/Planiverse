"""The benchmark's own job: one file per run whatever happened, in the format pyPMTEvalToolkit
reads, and a report from the toolkit that adds up."""
import json

import pytest

# The toolkit's runner imports `resource`, which Windows does not have.
pytest.importorskip("pypmt_eval_toolkit.runner")

from planiverse.benchmark import report, solve  # noqa: E402

def test_a_run_is_written_out_whatever_happens(tmp_path):
    record = solve(tmp_path, "bfws", "puzznic@1")
    assert record["status"] == "SOLVED" and record["metrics"]["plan-length"] == 10
    assert record["metrics"]["validated"] is True
    written = json.loads((tmp_path / "results/bfws/planiverse_puzznic_1.json").read_text())
    assert written["metrics"]["search-status"] == "solved" and written["plan"] == record["plan"]
    assert not list(tmp_path.glob("results/bfws/*.running")), "the marker goes once written"
    assert solve(tmp_path, "iw", "puzznic@9999")["status"] == "UNSUPPORTED"


def test_a_seeded_planner_writes_one_file_per_seed(tmp_path):
    record = solve(tmp_path, "fsx", "puzznic@9999", seed=3)
    assert record["planner"]["tag"] == "fsx-s3"
    assert (tmp_path / "results/fsx-s3/planiverse_puzznic_9999.json").is_file()
    assert solve(tmp_path, "fsx", "puzznic@9999")["planner"]["seed"] == 0   # by hand: seed 0


def test_no_registered_planner_takes_a_reward():
    """A planning library: every configuration is driven by the goal test and, at most, a
    progress heuristic."""
    import inspect
    from planiverse.benchmark import PLANNERS, REFERENCE
    assert set(REFERENCE) == {"bfws", "iw", "siw", "fsx"}
    for cls, _ in PLANNERS.values():
        assert "reward" not in inspect.signature(cls).parameters, cls.__name__


def test_every_planner_in_the_library_is_in_the_benchmark():
    """The benchmark runs the whole library: every planner class the planner packages
    export, and every documented configuration of one, is registered under a tag that
    `solve` builds the way it builds any other, so `generate` with no flag covers them
    all."""
    import inspect
    from planiverse.benchmark import PLANNERS, REFERENCE
    from planiverse.planners import blind, heuristic, macros, sampling, width

    registered = {cls for cls, _ in PLANNERS.values()}
    registered |= {base for cls in registered for base in cls.__mro__[1:]}
    exported = {getattr(module, name) for module in (width, heuristic, sampling)
                for name in module.__all__} | {
        blind.BreadthFirstSearch, blind.UniformCostSearch, blind.IterativeDeepening,
        macros.MacroPlanner}
    planners = {cls for cls in exported if inspect.isclass(cls) and hasattr(cls, "solve")}
    assert planners <= registered, {cls.__name__ for cls in planners - registered}
    assert list(PLANNERS)[:len(REFERENCE)] == list(REFERENCE)
    # The documented variants of one class, each under its own tag.
    assert {"astar", "wastar", "gbfslw", "kpiece", "ils", "bee", "multi"} <= set(PLANNERS)


def test_every_registered_planner_builds_and_runs_as_solve_runs_it():
    """Each tag's class takes its parameters, plus `progress` and `seed` where `solve` adds
    them, and runs on one instance under a small budget without returning a plan that does
    not replay."""
    import inspect
    from planiverse.benchmark import PLANNERS, _seeds
    from planiverse.benchmark.measures import MEASURES
    from planiverse.environments import get_spec
    from planiverse.planners.width import Budget, SearchResult

    for tag, (cls, params) in PLANNERS.items():
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


def test_the_toolkit_reads_the_runs_and_counts_the_missing_ones(tmp_path, monkeypatch):
    """`generate` writes the toolkit's `tasks.json`, so a run that never happened is MISSING in
    the toolkit's rows, and a solved one is solved there too."""
    from pypmt_eval_toolkit.analyzer import is_solved, load_rows

    import planiverse.benchmark as bench
    from planiverse.environments import get_spec

    monkeypatch.setattr(bench, "REGISTRY", [get_spec("puzznic")])      # 100 instances
    monkeypatch.setattr(bench, "PLANNERS", {"bfws": bench.REFERENCE["bfws"]})
    bench.generate(tmp_path)
    solve(tmp_path, "bfws", "puzznic@0")
    solve(tmp_path, "bfws", "puzznic@1")
    rows = load_rows(str(tmp_path))
    assert len(rows) == 100
    assert sum(is_solved(row) for row in rows) == 2
    assert sum(row["status"] == "MISSING" for row in rows) == 98
    report(str(tmp_path), plots=False)
    assert "2/100" in (tmp_path / "report/results.txt").read_text()


def test_a_memout_is_written_even_when_the_write_itself_runs_out(tmp_path, monkeypatch):
    """A write must still leave a file when `open` has truncated it and `json.dump` raises
    MemoryError under the address-space cap. The write lifts the cap and tries once more."""
    import planiverse.benchmark as bench
    real_dump, calls = bench.json.dump, []

    def dump_once_out_of_memory(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise MemoryError
        return real_dump(*args, **kwargs)

    monkeypatch.setattr(bench.json, "dump", dump_once_out_of_memory)
    record = {"task": {"task-id": "planiverse:puzznic:0", "domain": "puzznic", "instance": "0"},
              "planner": {"tag": "fsx-s0"},
              "run": {"started": 0.0}, "timings": {}, "metrics": {}, "logs": []}
    bench._write(tmp_path, record, "MEMOUT")
    written = json.loads((tmp_path / "results/fsx-s0/planiverse_puzznic_0.json").read_text())
    assert written["status"] == "MEMOUT" and len(calls) == 2


def test_generate_cuts_a_group_into_arrays_of_at_most_max_array(tmp_path, monkeypatch):
    """SLURM caps an array at the site's MaxArraySize, so a group over more instances than
    `MAX_ARRAY` is written as several arrays over slices of its one command file, and
    `submit_all.sh` submits every one."""
    import re
    import planiverse.benchmark as bench
    from planiverse.environments import get_spec

    monkeypatch.setattr(bench, "REGISTRY", [get_spec("puzznic")])      # 100 instances
    monkeypatch.setattr(bench, "PLANNERS", {"bfws": bench.REFERENCE["bfws"]})
    monkeypatch.setattr(bench, "MAX_ARRAY", 30)
    bench.generate(tmp_path)

    lines = (tmp_path / "cmds/bfws.txt").read_text().splitlines()
    assert [line.split()[-1] for line in lines] == [f"puzznic@{i}" for i in range(100)]
    arrays = sorted(tmp_path.glob("slurm/*.sbatch"))
    assert [p.name for p in arrays] == [f"pypmteval-bfws.{k}.sbatch" for k in (1, 2, 3, 4)]
    sizes = [int(re.search(r"--array=0-(\d+)%50", p.read_text()).group(1)) + 1 for p in arrays]
    assert sizes == [30, 30, 30, 10]
    assert all("--time=00:35:00" in p.read_text() and "--mem=9216M" in p.read_text()
               for p in arrays), "the run's own limits, plus the headroom to record them"
    submit = (tmp_path / "slurm/submit_all.sh").read_text()
    assert re.findall(r"slurm/(\S+)\.sbatch", submit) == [p.stem for p in arrays]

    # A group that fits in one array is one array.
    monkeypatch.setattr(bench, "MAX_ARRAY", 1000)
    bench.generate(tmp_path / "one")
    assert [p.name for p in (tmp_path / "one" / "slurm").glob("*.sbatch")] == \
        ["pypmteval-bfws.sbatch"]
