"""`planiverse-bench` — the staged command line.

The stages mirror [pyPMTEvalToolkit](https://github.com/pyPMT/pyPMTEvalToolkit), because the
separation is the useful part: each one writes files the next one reads, so a benchmark can be
prepared on a laptop, run on a cluster, and analysed somewhere else again, without any stage
needing the others to be running.

    planiverse-bench init      --exp-dir experiment      # write a default experiment
    planiverse-bench environments                        # what can be benchmarked here
    planiverse-bench planners                            # what can be run
    planiverse-bench discover  --exp-dir experiment --sandbox-dir sandbox
    planiverse-bench generate  --exp-dir experiment --sandbox-dir sandbox
    bash sandbox/slurm/submit_all.sh                     # or: bash sandbox/run_local.sh 8
    planiverse-bench analyze   --sandbox-dir sandbox
    planiverse-bench report    --sandbox-dir sandbox

`solve` is the one stage you do not normally type: it is what each array element runs.
"""
import argparse
import json
import os
import sys

from planiverse.benchmark import analysis, catalogue, discovery, report, slurm
from planiverse.benchmark.config import (
    ExperimentConfig, Limits, PlannerSpec, SlurmConfig, TaskSelection,
)
from planiverse.benchmark.measures import has_measure
from planiverse.benchmark.runner import solve as solve_one
from planiverse.environments import REGISTRY

#: How high `iw` is allowed to climb. A bound, not a plan: `IteratedWidth` stops as soon as a
#: width solves the problem, the budget runs out, or a width covers the whole reachable space
#: without discarding anything for novelty — so in practice it stops at 1 or 2 and this
#: number is never reached. Fixing the width instead would report IW at a configuration we
#: chose rather than at the one the algorithm defines.
MAX_WIDTH = 1000

#: What `init` writes. A spread rather than a single planner: the point of the harness is
#: comparison, and one of each family makes the first report say something.
#:
#: `iw` and `siw` both iterate their width, for the same reason: novelty is a *filter* in
#: both, so a width too low loses states outright and which width is enough is a property of
#: the problem. `bfws` is pinned at 1 and 2 deliberately — see the note below.
DEFAULT_PLANNERS = (
    PlannerSpec(tag="iw", planner="iterated_width",
                params={"max_width": MAX_WIDTH, "strict": False}),
    PlannerSpec(tag="siw", planner="siw",
                params={"width": 1, "max_width": MAX_WIDTH, "strict": False}),
    # BFWS is *not* iterated, and that is not an oversight. It uses novelty as a sort key
    # rather than as a filter, so nothing is ever discarded and BFWS(k) is complete at every
    # width: a BFWS(1) that fails ran out of budget, it did not run out of width. Restarting
    # at a wider one would re-expand everything and split the same budget across the attempts,
    # which is strictly worse. What the width changes is the *ordering*, so 1 and 2 are two
    # different search strategies worth comparing side by side — not a weaker and a stronger.
    PlannerSpec(tag="bfws-1", planner="bfws", params={"width": 1}),
    PlannerSpec(tag="bfws-2", planner="bfws", params={"width": 2}),
    PlannerSpec(tag="fsx", planner="fsx",
                params={"horizon": 6, "walkers": 8, "seed": 0}),
    PlannerSpec(tag="mcts", planner="mcts",
                params={"iterations": 2000, "seed": 0}),
)


#: A friendlier spelling for a cartridge whose variable is an abbreviation. `--rom-sml`
#: follows PLANIVERSE_SML_ROM and stays for that reason, but nobody types "sml" first.
ROM_ALIASES = {"sml": ("--rom-mario",)}


def rom_environments():
    """The registry entries that need a cartridge, with the flag each one answers to."""
    return tuple((spec, spec.rom_flag()) for spec in REGISTRY if spec.needs_rom)


def add_rom_arguments(parser):
    """One `--rom-<name>` per cartridge environment, generated from the registry.

    Generated rather than listed so a new Game Boy environment gets a flag by existing, and
    so the flag and the environment variable stay the same name in two spellings.
    """
    for spec, flag in rom_environments():
        parser.add_argument(f"--rom-{flag}", *ROM_ALIASES.get(flag, ()),
                            dest=f"rom_{flag.replace('-', '_')}", metavar="PATH",
                            help=f"{spec.summary.split(',')[0]} "
                                 f"(or set {spec.rom_variable})")


def collect_roms(arguments):
    """`{environment: absolute path}` from the named flags and from `--rom ENV=PATH`.

    A path that does not exist is refused here rather than at discovery: a typo you find
    while typing costs a second, and one you find after submitting four thousand jobs costs
    rather more.
    """
    roms = {}
    for spec, flag in rom_environments():
        path = getattr(arguments, f"rom_{flag.replace('-', '_')}", None)
        if path:
            roms[spec.name] = _cartridge(path, f"--rom-{flag}")
    for entry in getattr(arguments, "rom", []) or []:
        name, _, path = entry.partition("=")
        if not name or not path:
            raise ValueError(f"--rom wants ENV=PATH, got {entry!r}")
        roms[name] = _cartridge(path, "--rom")
    return roms


def _cartridge(path, flag):
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(resolved):
        raise ValueError(f"{flag}: no file at {resolved}")
    return resolved


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="planiverse-bench",
        description="Benchmark Planiverse planners, on a cluster or locally.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("The stages mirror")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="write a default experiment directory")
    init.add_argument("--exp-dir", default="experiment")
    init.add_argument("--name", default="planiverse-bench")
    init.add_argument("--time", default="30m", help="per-run wall-clock limit")
    init.add_argument("--memory", default="8GB", help="per-run memory limit")
    init.add_argument("--max-expansions", type=int, default=100000)
    init.add_argument("--max-instances", type=int, default=0,
                      help="instances per environment; 0 means every one")
    init.add_argument("--no-roms", action="store_true",
                      help="leave out the environments that need a cartridge")
    add_rom_arguments(init)
    init.add_argument("--rom", action="append", default=[], metavar="ENV=PATH",
                      help="the same thing keyed by environment name, e.g. "
                           "--rom puzznic_gb=/path/to.gb. Repeatable; useful in a loop where "
                           "the environment is a variable.")
    init.add_argument("--partition", default=None, help="SLURM partition")
    init.add_argument("--account", default=None, help="SLURM account")
    init.add_argument("--qos", default=None, help="SLURM quality of service")
    init.add_argument("--setup-command", action="append", default=[], metavar="CMD",
                      help="run at the top of every job — `module load python`, "
                           "`source .../activate`. Repeatable, order preserved.")
    init.add_argument("--force", action="store_true", help="overwrite an existing experiment")
    init.set_defaults(handler=_init)

    environments = subparsers.add_parser(
        "environments", help="what can be benchmarked on this machine")
    environments.add_argument("--json", action="store_true")
    environments.set_defaults(handler=_environments)

    planners = subparsers.add_parser("planners", help="the planners and their parameters")
    planners.add_argument("--json", action="store_true")
    planners.set_defaults(handler=_planners)

    discover = subparsers.add_parser("discover", help="resolve the task list")
    discover.add_argument("--exp-dir", default="experiment")
    discover.add_argument("--sandbox-dir", default="sandbox")
    discover.set_defaults(handler=_discover)

    generate = subparsers.add_parser("generate", help="write commands and SLURM jobs")
    generate.add_argument("--exp-dir", default="experiment")
    generate.add_argument("--sandbox-dir", default="sandbox")
    generate.add_argument("--entry-point", default=slurm.DEFAULT_ENTRY_POINT,
                          help="how the jobs invoke this CLI")
    generate.add_argument("--seed", type=int, default=None,
                          help="seed for the randomised planners")
    generate.add_argument("--per-task-scripts", action="store_true",
                          help="one sbatch file per run, for sites without job arrays")
    generate.add_argument("--rediscover", action="store_true",
                          help="re-resolve the task list first")
    generate.set_defaults(handler=_generate)

    solve = subparsers.add_parser("solve", help="run one (planner, task) pair")
    solve.add_argument("--exp-dir", default="experiment")
    solve.add_argument("--sandbox-dir", default="sandbox")
    solve.add_argument("--planner", required=True, help="a planner tag")
    solve.add_argument("--task", required=True, help="environment@index")
    solve.add_argument("--seed", type=int, default=None)
    solve.set_defaults(handler=_solve)

    analyze = subparsers.add_parser("analyze", help="collect results into tables and CSV")
    analyze.add_argument("--sandbox-dir", default="sandbox")
    analyze.add_argument("--json", action="store_true")
    analyze.set_defaults(handler=_analyze)

    report_parser = subparsers.add_parser("report", help="tables, LaTeX and plots")
    report_parser.add_argument("--sandbox-dir", default="sandbox")
    report_parser.set_defaults(handler=_report)

    arguments = parser.parse_args(argv)
    return arguments.handler(arguments) or 0


# ------------------------------------------------------------------------------- stages

def _init(arguments):
    details = os.path.join(arguments.exp_dir, "exp-details.json")
    if os.path.exists(details) and not arguments.force:
        print(f"{details} already exists. Pass --force to overwrite it.", file=sys.stderr)
        return 1
    try:
        roms = collect_roms(arguments)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    experiment = ExperimentConfig(
        name=arguments.name,
        limits=Limits(time=arguments.time, memory=arguments.memory,
                      max_expansions=arguments.max_expansions),
        tasks=TaskSelection(max_instances_per_environment=arguments.max_instances,
                            include_rom_environments=not arguments.no_roms),
        slurm=SlurmConfig(partition=arguments.partition, account=arguments.account,
                          qos=arguments.qos,
                          setup_commands=tuple(arguments.setup_command)),
        planners=DEFAULT_PLANNERS,
        roms=roms,
    )
    experiment.save(arguments.exp_dir)
    print(f"wrote {details}")
    for spec in experiment.planners:
        print(f"  planners/{spec.tag}.json  ({spec.planner})")
    missing = [(spec.name, flag) for spec, flag in rom_environments()
               if not experiment.rom_for(spec)]
    if missing and experiment.tasks.include_rom_environments:
        names = ", ".join(name for name, _ in missing)
        flags = " ".join(f"--rom-{flag} /path/to/rom.gb" for _, flag in missing)
        print(f"\nNo cartridge for {names}. They will be skipped until you add one:\n"
              f"  planiverse-bench init --exp-dir {arguments.exp_dir} --force {flags}\n"
              f"or run ./setup_benchmark.sh, which asks for them.")
    print(f"\nNext: planiverse-bench discover --exp-dir {arguments.exp_dir}")
    return 0


def _environments(arguments):
    rows = []
    for spec in REGISTRY:
        ok, reason = discovery.eligible(spec, TaskSelection(include_rom_environments=True),
                                        rom=spec.rom_path())
        rows.append({"environment": spec.name, "available": ok, "reason": reason,
                     "needs_rom": spec.needs_rom, "instances": spec.instances,
                     "state_identity": spec.state_identity,
                     "has_progress_measure": has_measure(spec.name),
                     "tags": sorted(spec.tags)})
    if arguments.json:
        print(json.dumps(rows, indent=2))
        return 0

    width = max(len(row["environment"]) for row in rows)
    print(f"{'environment':{width}}  {'ready':6}  {'rom':4}  {'measure':8}  instances")
    print("-" * (width + 40))
    for row in rows:
        print(f"{row['environment']:{width}}  "
              f"{'yes' if row['available'] else 'no':6}  "
              f"{'yes' if row['needs_rom'] else '-':4}  "
              f"{'yes' if row['has_progress_measure'] else 'no':8}  "
              f"{row['instances']}"
              + (f"   ({row['reason']})" if not row["available"] else ""))
    print("\n'measure' is whether a progress measure exists for SIW and BFWS; without one "
          "they\nrun without their main input. See planiverse/benchmark/measures.py.")
    return 0


def _planners(arguments):
    rows = [{"planner": name, "class": catalogue.PLANNERS[name][1],
             "params": list(catalogue.PLANNERS[name][2]),
             "takes_progress": catalogue.takes_progress(name),
             "randomised": catalogue.is_randomised(name),
             "complete": catalogue.is_complete(name),
             "complete_when_exhausted": name in catalogue.COMPLETE_WHEN_EXHAUSTED}
            for name in catalogue.names()]
    if arguments.json:
        print(json.dumps(rows, indent=2))
        return 0
    for row in rows:
        flags = [name for name, on in
                 (("complete", row["complete"]),
                  ("complete when it reports exhausted", row["complete_when_exhausted"]),
                  ("randomised", row["randomised"]),
                  ("uses progress measure", row["takes_progress"])) if on]
        print(f"{row['planner']:16}  {row['class']:16}  {', '.join(row['params'])}")
        if flags:
            print(f"{'':16}  {'':16}  [{'; '.join(flags)}]")
    return 0


def _discover(arguments):
    experiment = ExperimentConfig.load(arguments.exp_dir)
    discovered = discovery.discover(experiment.tasks, roms=experiment.roms)
    pairs = discovery.pair_up(discovered["tasks"], experiment.active_planners())
    path = discovery.write_tasks(arguments.sandbox_dir, discovered, pairs, arguments.exp_dir)

    for row in discovered["environments"]:
        note = "" if row["has_progress_measure"] else "   (no progress measure)"
        print(f"  {row['environment']:20} {row['selected']:>3} of "
              f"{row['instances'] if row['instances'] is not None else '?'}{note}")
    for row in discovered["skipped"]:
        print(f"  {row['environment']:20}   skipped — {row['reason']}")
    print(f"\n{len(discovered['tasks'])} tasks x "
          f"{len(experiment.active_planners())} planners = {len(pairs)} runs")
    print(f"wrote {path}")
    return 0


def _generate(arguments):
    experiment = ExperimentConfig.load(arguments.exp_dir)
    if arguments.rediscover or not os.path.isfile(
            os.path.join(arguments.sandbox_dir, "tasks.json")):
        discovered = discovery.discover(experiment.tasks, roms=experiment.roms)
        pairs = discovery.pair_up(discovered["tasks"], experiment.active_planners())
        discovery.write_tasks(arguments.sandbox_dir, discovered, pairs, arguments.exp_dir)
    else:
        pairs = discovery.read_tasks(arguments.sandbox_dir).get("pairs", [])

    if not pairs:
        print("nothing to generate: the task list is empty. Check `discover` output.",
              file=sys.stderr)
        return 1

    written = slurm.generate(arguments.sandbox_dir, pairs, experiment, arguments.exp_dir,
                             entry_point=arguments.entry_point, seed=arguments.seed,
                             per_task_scripts=arguments.per_task_scripts)
    for tag, entry in sorted(written["commands"].items()):
        print(f"  {tag:16} {entry['count']:>5} runs  {entry['path']}")
    print(f"\n{written['runs']} runs across {len(written['scripts'])} job scripts")
    print(f"  submit:  bash {written['submit_all']}")
    print(f"  or here: bash {written['run_local']} 8")
    return 0


def _solve(arguments):
    experiment = ExperimentConfig.load(arguments.exp_dir)
    matching = [spec for spec in experiment.planners if spec.tag == arguments.planner]
    if not matching:
        print(f"no planner tagged {arguments.planner!r} in {arguments.exp_dir}",
              file=sys.stderr)
        return 2
    record = solve_one(matching[0], arguments.task, experiment.limits,
                       sandbox_dir=arguments.sandbox_dir, seed=arguments.seed,
                       roms=experiment.roms)
    print(f"{record['task']:24} {record['planner']:16} {record['status']:12} "
          f"{record['seconds']:.2f}s "
          f"exp={record.get('statistics', {}).get('expansions', 0)}"
          + (f"  plan={record['plan_length']}" if record.get("plan_length") is not None
             else "")
          + (f"  {record['note']}" if record.get("note") else ""))
    # Zero even on a failed run: the failure is the result, and a non-zero exit would make
    # SLURM mark the array element failed and hide it among genuine infrastructure errors.
    return 0


def _analyze(arguments):
    records = analysis.load_results(arguments.sandbox_dir)
    summary = analysis.summarise(arguments.sandbox_dir, records)
    csv_path = analysis.write_csv(arguments.sandbox_dir, records)
    summary_path = analysis.write_summary(arguments.sandbox_dir, summary)
    if arguments.json:
        print(json.dumps(summary, indent=2))
    else:
        print(report.text_tables(summary, records))
    print(f"wrote {csv_path}\nwrote {summary_path}")
    return 0


def _report(arguments):
    records = analysis.load_results(arguments.sandbox_dir)
    summary = analysis.summarise(arguments.sandbox_dir, records)
    written = report.write_report(arguments.sandbox_dir, summary, records)
    for kind, path in written.items():
        print(f"  {kind:8} {path}")
    if not report.PLOTTING:
        print("\nmatplotlib did not import, so the plots were skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
