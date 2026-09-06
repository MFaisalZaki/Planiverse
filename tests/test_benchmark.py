"""The harness's own job: one file per run whatever happened, and a report that adds up."""
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
    assert facts.startswith("solved per seed: bfws 435, iw 325, siw 193")
    statuses = (SANDBOX / "report/statuses.tex").read_text()
    assert "BFWS & \\textbf{435} & 149 & 248 & 100 & 6 &" in statuses
