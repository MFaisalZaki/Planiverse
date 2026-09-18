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
    assert solve(tmp_path, "mcts", "puzznic@9999")["seed"] == 0   # run by hand: the first seed


def test_the_rollout_planners_run_under_the_protocol_with_a_seed(tmp_path):
    """Both take a seed, so the benchmark gives them the five like MCTS and FSX, and the
    progress measure like SIW and BFWS."""
    from planiverse.benchmark import _seeds
    assert _seeds("riw") == [0, 1, 2, 3, 4] and _seeds("piiw") == [0, 1, 2, 3, 4]
    for tag in ("riw", "piiw"):
        record = solve(tmp_path, tag, "puzznic@1", seed=2)
        assert record["status"] == "SOLVED", record
        assert record["seed"] == 2 and record["params"]["width"] == 1
        assert record["statistics"]["rollouts"] > 0 and record["statistics"]["episodes"] == 1
        assert (tmp_path / f"results/{tag}/puzznic__1__s2.json").is_file()


def test_the_report_expects_every_run_and_averages_over_seeds(tmp_path):
    (tmp_path / "tasks.json").write_text(
        json.dumps({"environments": [{"environment": "puzznic", "instances": 2}]}))
    solve(tmp_path, "bfws", "puzznic@0")
    solve(tmp_path, "bfws", "puzznic@1")
    report(tmp_path)
    statuses = (tmp_path / "report/statuses.tex").read_text()
    facts = (tmp_path / "report/facts.txt").read_text()
    assert "BFWS & \\textbf{2} & 0" in statuses and "Missing" not in statuses
    # Ten missing MCTS runs are two per seed, so the row still sums to the two instances.
    assert "IW & 0 & 2" in statuses and "MCTS & 0.0 (0.0) & 2" in statuses
    missing = facts.split("missing")[1].split("\n")[0]
    assert "iw on puzznic 2" in missing and "mcts on puzznic 10" in missing


@pytest.mark.skipif(not (SANDBOX / "tasks.json").is_file(),
                    reason="the paper's sandbox is not unpacked beside the repository")
def test_the_paper_comes_out_of_its_sandbox():
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
    record = {"task": "puzznic@0", "environment": "puzznic", "index": 0, "planner": "riw",
              "seed": 0, "started": 0.0}
    bench._write(tmp_path, record, "MEMOUT")
    written = json.loads((tmp_path / "results/riw/puzznic__0__s0.json").read_text())
    assert written["status"] == "MEMOUT" and len(calls) == 2
