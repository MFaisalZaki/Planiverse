"""`planiverse-bench`: the tool paper's evaluation protocol, as code.

    planiverse-bench generate [--sandbox-dir sandbox] [--partition P] [--qos Q] [--account A]
    bash sandbox/submit.sh                       # or: bash sandbox/run_local.sh 8
    planiverse-bench report   [--sandbox-dir sandbox]

`generate` asks every registered environment how many instances it has and writes one command
per (planner, instance, seed), plus a SLURM job array for each planner, or for each of a seeded
planner's seeds, that runs them. `solve` is what one array element runs: one planner on one
instance under the limits, written out as one JSON file whatever happens. `report` reads those
files back and writes the paper's tables, its figures, and the numbers its prose quotes.

The protocol is the constants below. There is no configuration file: a run that changed a limit
would not be the paper's experiment, and one that did not has nothing to configure.
"""
import argparse
import inspect
import json
import os
import pathlib
import platform
import resource
import shlex
import signal
import statistics
import sys
import time
import traceback

from planiverse.benchmark.measures import MEASURES
from planiverse.environments import REGISTRY, get_spec
from planiverse.planners.fsx import FSXPlanner
from planiverse.planners.mcts import MCTSPlanner
from planiverse.planners.width import Budget, IteratedBFWS, IteratedWidth, SIWSearch

#: Per run: 30 minutes of wall clock, 8 GB of address space, 500,000 expansions.
LIMITS = {"seconds": 1800, "bytes": 8 * 1024 ** 3, "expansions": 500_000}

#: The five configurations, in the paper's order. Anything not named is the class's own
#: default, which is where MCTS's exploration constant, rollout depth, backup rule and length
#: penalty come from, and FSX's measure, temperature and step cap.
PLANNERS = {
    "bfws": (IteratedBFWS, {"max_width": 1000}),
    "iw": (IteratedWidth, {"max_width": 1000, "strict": False}),
    "siw": (SIWSearch, {"width": 1, "max_width": 1000, "strict": False}),
    "mcts": (MCTSPlanner, {"iterations": 2000}),
    "fsx": (FSXPlanner, {"horizon": 6, "walkers": 8}),
}

#: The seeds a planner whose constructor takes one runs under. Every (instance, seed) is a
#: full run under the same limits, and the report averages over them; the environments are
#: deterministic, so the seed is the only source of variance. The width planners have no seed
#: and run once.
SEEDS = range(5)

#: The width family: deterministic, and what the overlap and runtime figures compare.
WIDTH = ("bfws", "iw", "siw")

#: The paper's environment names, in its table order. A `_gb` twin takes the same name under
#: "Game (cartridge)"; the family itself comes from the registry's tags.
NAMES = {"water_network": "Water distribution", "power_grid": "Power grid",
         "crop_management": "Crop management", "network_attack": "Network attack",
         "puzznic": "Puzznic", "flipull": "Flipull", "lolo": "Adventures of Lolo",
         "amazing_tater": "Amazing Tater", "super_mario_land": "Super Mario Land"}

#: How a run can end. Everything but MISSING is written by `solve`; MISSING is what `report`
#: calls an expected run that left no file, so a job that never ran cannot pass for coverage.
STATUSES = ("SOLVED", "INVALID", "UNSOLVED", "TIMEOUT", "NODEOUT", "MEMOUT", "ERROR",
            "UNSUPPORTED", "MISSING")

SBATCH = """#!/bin/bash
#SBATCH --job-name=planiverse-bench-{group}
#SBATCH --array=0-{last}%{parallel}
#SBATCH --time=00:35:00
#SBATCH --mem=9216M
#SBATCH --cpus-per-task=1
#SBATCH --output={sandbox}/logs/{group}/%A_%a.out
#SBATCH --error={sandbox}/logs/{group}/%A_%a.err
{extra}{exports}
# Line n of the command file is array element n. The five minutes and the gigabyte above the
# harness's own limits let it record its TIMEOUT or MEMOUT before SLURM steps in.
eval "$(sed -n "$((${{SLURM_ARRAY_TASK_ID:-0}} + 1))p" {cmds})"
"""


def _seeds(tag):
    """The seeds a planner runs under: SEEDS if its constructor takes one, else a single None."""
    return list(SEEDS) if "seed" in inspect.signature(PLANNERS[tag][0]).parameters else [None]


def _filename(environment, index, seed):
    return f"{environment}__{index}" + ("" if seed is None else f"__s{seed}") + ".json"


def generate(sandbox, partition=None, qos=None, account=None, parallel=50):
    """Count every environment's instances, then write the commands and the arrays to run them."""
    sandbox = os.path.abspath(sandbox)
    counts, roms = {}, {}
    for spec in REGISTRY:
        try:
            env = spec.build()
        except Exception as exc:
            print(f"  {spec.name:22} skipped: {exc}", file=sys.stderr)
            continue
        # The registry says how many instances there are in words; the environment says so
        # in code, by refusing the first index it does not have.
        for count in range(10_000):
            try:
                env.set_index(count)
            except Exception:
                break
        getattr(env, "close", lambda: None)()
        counts[spec.name] = count
        if spec.needs_rom:
            roms[spec.rom_variable] = spec.rom_path()
        print(f"  {spec.name:22} {count:>4} instances")

    # One job array per planner, or per seed of a seeded planner: each is one instance long,
    # which keeps every array under a site's MaxArraySize and finishes seed 0 first.
    groups = {tag if seed is None else f"{tag}-s{seed}": (tag, seed)
              for tag in PLANNERS for seed in _seeds(tag)}
    for name in ("cmds", "slurm", *(f"logs/{group}" for group in groups)):
        os.makedirs(f"{sandbox}/{name}", exist_ok=True)
    with open(f"{sandbox}/tasks.json", "w") as handle:
        json.dump({"environments": [{"environment": env, "instances": n}
                                    for env, n in counts.items()]}, handle, indent=1)
    exports = "".join(f"export {var}={shlex.quote(path)}\n" for var, path in roms.items())
    extra = "".join(f"#SBATCH --{key}={value}\n" for key, value in
                    (("partition", partition), ("qos", qos), ("account", account)) if value)
    tasks = [f"{env}@{index}" for env, n in counts.items() for index in range(n)]
    for group, (tag, seed) in groups.items():
        with open(f"{sandbox}/cmds/{group}.txt", "w") as handle:
            handle.writelines(
                f"{shlex.quote(sys.executable)} -m planiverse.benchmark solve "
                f"--sandbox-dir {shlex.quote(sandbox)} {tag} {task}"
                + ("" if seed is None else f" --seed {seed}") + "\n" for task in tasks)
        with open(f"{sandbox}/slurm/{group}.sbatch", "w") as handle:
            handle.write(SBATCH.format(group=group, last=len(tasks) - 1, parallel=parallel,
                                       sandbox=sandbox, extra=extra, exports=exports,
                                       cmds=shlex.quote(f"{sandbox}/cmds/{group}.txt")))
    scripts = {
        "submit.sh": "#!/bin/bash\n" + "".join(f"sbatch {sandbox}/slurm/{group}.sbatch\n"
                                               for group in groups),
        # `-L 1` hands each line to xargs as words, quotes honoured; `-I` would cap the line
        # at 255 bytes on BSD xargs, which a long sandbox path exceeds.
        "run_local.sh": "#!/bin/bash\n# bash run_local.sh [jobs-at-a-time]\n" + exports
                        + f"cat {shlex.quote(sandbox)}/cmds/*.txt | "
                          f"xargs -P \"${{1:-4}}\" -L 1 sh -c 'exec \"$@\"' _\n",
    }
    for name, body in scripts.items():
        with open(f"{sandbox}/{name}", "w") as handle:
            handle.write(body)
        os.chmod(f"{sandbox}/{name}", 0o755)
    print(f"{len(tasks)} instances x {len(groups)} arrays "
          f"({', '.join(f'{tag} x{len(_seeds(tag))}' for tag in PLANNERS)}) "
          f"= {len(tasks) * len(groups)} runs\n"
          f"  submit:  bash {sandbox}/submit.sh\n  or here: bash {sandbox}/run_local.sh 8")


def _alarm(*_):
    raise TimeoutError


def solve(sandbox, tag, task, seed=None):
    """Run one planner on one instance under the limits and write down what happened.

    The result is written even when the run fails, which is the point: a benchmark that only
    records its successes cannot say that a planner crashed on a third of the set.
    """
    name, index = task.rsplit("@", 1)
    seed = _seeds(tag)[0] if seed is None else seed   # a seeded planner run by hand gets its first
    record = {"task": task, "environment": name, "index": int(index), "planner": tag,
              "seed": seed, "params": PLANNERS[tag][1], "limits": LIMITS,
              "host": platform.node(), "started": time.time()}
    # An address-space cap turns an overrun into a MemoryError the run can record, instead of
    # an OOM kill that leaves no file. macOS refuses the call; the cap is for the Linux cluster.
    try:
        resource.setrlimit(resource.RLIMIT_AS,
                           (LIMITS["bytes"], resource.getrlimit(resource.RLIMIT_AS)[1]))
    except (ValueError, OSError):
        pass
    env = None
    try:
        try:
            env = get_spec(name).build()
            env.set_index(int(index))
        except Exception as exc:
            return _write(sandbox, record, "UNSUPPORTED", note=f"{type(exc).__name__}: {exc}")
        cls, params = PLANNERS[tag]
        if seed is not None:
            params = {**params, "seed": seed}
        if "progress" in inspect.signature(cls).parameters:
            params = {**params, "progress": MEASURES.get(name)}
        planner = cls(**params)
        # The Budget is checked between expansions; the alarm catches the one expansion that
        # itself overruns (a power-grid step can take twenty seconds), so the run records a
        # TIMEOUT rather than being killed by SLURM and leaving no file.
        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, 1.02 * LIMITS["seconds"])
        started = time.perf_counter()
        try:
            out = planner.solve(env, Budget(max_expansions=LIMITS["expansions"],
                                            max_seconds=LIMITS["seconds"]))
        except TimeoutError:
            return _write(sandbox, record, "TIMEOUT", time.perf_counter() - started)
        except MemoryError:
            return _write(sandbox, record, "MEMOUT", time.perf_counter() - started)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        elapsed = time.perf_counter() - started
        record.update(
            search_status=out.status, width=out.width,
            plan=None if out.plan is None else [str(action) for action in out.plan],
            plan_length=None if out.plan is None else len(out.plan),
            statistics={"expansions": out.statistics.expansions,
                        "generated": out.statistics.generated,
                        "search_seconds": out.statistics.elapsed,
                        "widths_tried": list(out.statistics.widths_tried)})
        if out.solved:
            try:
                status = "SOLVED" if env.validate(out.plan) else "INVALID"
            except Exception:
                status = "INVALID"
        elif out.status in ("failed", "exhausted"):
            status = "UNSOLVED"
        elif out.statistics.expansions >= LIMITS["expansions"]:
            status = "NODEOUT"
        elif elapsed >= 0.95 * LIMITS["seconds"]:
            status = "TIMEOUT"
        else:
            # Out of budget with neither limit reached: an iterated search whose per-width
            # allowances ran out, or FSX at its step cap or a dead end.
            status = "NODEOUT"
        return _write(sandbox, record, status, elapsed)
    except MemoryError:
        return _write(sandbox, record, "MEMOUT")
    except Exception as exc:
        record["traceback"] = traceback.format_exc()
        return _write(sandbox, record, "ERROR", note=f"{type(exc).__name__}: {exc}")
    finally:
        getattr(env, "close", lambda: None)()


def _write(sandbox, record, status, seconds=None, note=None):
    record.update(status=status,
                  seconds=time.time() - record["started"] if seconds is None else seconds)
    if note:
        record["note"] = note
    path = os.path.join(str(sandbox), "results", record["planner"],
                        _filename(record["environment"], record["index"], record["seed"]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(record, handle, indent=1, default=str)
    print(f"{record['task']:24} {record['planner']:6} {status:12} {record['seconds']:.2f}s")
    return record


def report(sandbox):
    """Read every expected result back and write the paper's tables, figures and numbers."""
    import matplotlib
    import pandas as pd
    matplotlib.use("Agg")

    manifest = json.loads(pathlib.Path(sandbox, "tasks.json").read_text())
    counts = {entry["environment"]: entry["instances"] for entry in manifest["environments"]}
    rows = []
    for tag in PLANNERS:
        for seed in _seeds(tag):
            for env, n in counts.items():
                for index in range(n):
                    try:
                        record = json.loads(pathlib.Path(
                            sandbox, "results", tag, _filename(env, index, seed)).read_text())
                    except FileNotFoundError:
                        record = {"status": "MISSING"}
                    except ValueError:  # a job killed mid-write leaves a truncated file
                        record = {"status": "ERROR"}
                    # An unseeded planner's runs sit under seed -1, so the column stays a
                    # number the tables can group and average on.
                    rows.append({"planner": tag, "seed": -1 if seed is None else seed,
                                 "environment": env, "task": f"{env}@{index}",
                                 "status": record.get("status", "ERROR"),
                                 "seconds": record.get("seconds"),
                                 "width": record.get("width"),
                                 "plan_length": record.get("plan_length")})
    df = pd.DataFrame(rows)
    solved = df[df.status == "SOLVED"]

    out = pathlib.Path(sandbox, "report")
    out.mkdir(exist_ok=True)
    (out / "coverage.tex").write_text(_coverage_tex(df, counts))
    (out / "statuses.tex").write_text(_statuses_tex(df, solved))
    (out / "facts.txt").write_text(_facts(df, counts))
    _cactus(df, out / "cactus.pdf")
    _overlap(df, counts, out / "overlap_bfws_iw_siw.pdf")
    _runtime(df, out / "runtime_bfws_iw_siw.pdf")
    print((out / "facts.txt").read_text())
    print(f"wrote {out}/{{coverage,statuses}}.tex, facts.txt, and the three figures")


def _families(counts):
    """The paper's table rows: (family, environments) in its order, cartridge twins last."""
    groups = {}
    for name in NAMES:
        for env in (name, name + "_gb"):
            if env in counts:
                family = next(tag for tag in ("operational", "security", "game")
                              if tag in get_spec(env).tags).capitalize()
                groups.setdefault(family + (" (cartridge)" if env.endswith("_gb") else ""),
                                  []).append(env)
    return list(groups.items())


def _fmt(values):
    """A count, or a seeded planner's mean over seeds with the standard deviation in brackets."""
    values = [float(value) for value in values]
    if len(values) == 1:
        return f"{values[0]:g}"
    return f"{statistics.mean(values):.1f} ({statistics.stdev(values):.1f})"


def _solved_per_seed(df):
    """Instances solved per (planner, seed), zeros included: the unit the tables average over."""
    return df.assign(ok=df.status == "SOLVED").groupby(["planner", "seed"]).ok.sum()


def _coverage_tex(df, counts):
    """Table 2: instances solved per environment and planner, in the paper's families."""
    solved = (df.assign(ok=df.status == "SOLVED")
              .groupby(["planner", "seed", "environment"]).ok.sum()
              .unstack("environment").reindex(columns=list(counts), fill_value=0))
    totals = solved.sum(axis=1)
    means = totals.groupby(level="planner").mean()
    lines = ["\\begin{tabular}{ll" + "r" * (len(PLANNERS) + 1) + "}", "\\toprule",
             "Family & Environment & Inst. & " + " & ".join(p.upper() for p in PLANNERS)
             + " \\\\", "\\midrule"]
    for family, envs in _families(counts):
        if len(envs) > 1:
            lines.append(f"\\multirow{{{len(envs)}}}{{*}}{{{family}}}")
        for env in envs:
            lines.append(f"{family if len(envs) == 1 else ''} & {NAMES[env.removesuffix('_gb')]}"
                         f" & {counts[env]} & "
                         + " & ".join(_fmt(solved.loc[p][env]) for p in PLANNERS) + " \\\\")
        lines.append("\\midrule")
    lines += [f"& Total & {sum(counts.values())} & " + " & ".join(
        f"\\textbf{{{_fmt(totals.loc[p])}}}" if means[p] == means.max() else _fmt(totals.loc[p])
        for p in PLANNERS) + " \\\\", "\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def _statuses_tex(df, solved):
    """Table 3: how every run ended, one row per planner, every status that occurred.

    The columns come from the data, so a status cannot be left out without the row totals
    showing it. A seeded planner's counts are means per seed, so its row still sums to the
    instance count, and its Solved cell carries the standard deviation. A run that never
    happened is counted as unsolved, as the paper does; it never credits a planner, and
    `facts.txt` still lists it.
    """
    import pandas as pd
    # Divided before reindexing: aligning against the seed counts sorts the planners, and the
    # reindex is what puts them back in the paper's order.
    n = (df.groupby(["planner", "status"]).size().unstack(fill_value=0)
         .div(df.groupby("planner").seed.nunique(), axis=0)
         .reindex(index=list(PLANNERS), columns=STATUSES, fill_value=0))
    n["UNSOLVED"] += n.pop("MISSING")
    n = n.loc[:, n.any()].round(1)
    n["Median (s)"] = solved.groupby("planner").seconds.median().round(1)
    per_seed = _solved_per_seed(df)
    heads = {"SOLVED": "Solved", "INVALID": "Invalid", "UNSOLVED": "Unsolved", "TIMEOUT": "Time",
             "NODEOUT": "Exp.", "MEMOUT": "Mem.", "ERROR": "Error", "UNSUPPORTED": "Unsup."}
    lines = ["\\begin{tabular}{l" + "r" * len(n.columns) + "}", "\\toprule"]
    budget = [c for c in n.columns if c in ("TIMEOUT", "NODEOUT", "MEMOUT")]
    if budget:
        first = list(n.columns).index(budget[0]) + 2
        lines += ["& " * (first - 1) + f"\\multicolumn{{{len(budget)}}}{{c}}{{Out of budget}}"
                  + " &" * (len(n.columns) + 2 - first - len(budget)) + " \\\\",
                  f"\\cmidrule(lr){{{first}-{first + len(budget) - 1}}}"]
    lines += ["Planner & " + " & ".join(heads.get(c, c) for c in n.columns) + " \\\\",
              "\\midrule"]
    for tag, row in n.iterrows():
        cells = []
        for column, value in row.items():
            best = (column == "SOLVED" and value == n.SOLVED.max()) or \
                   (column == "Median (s)" and value == n["Median (s)"].min())
            text = (_fmt(per_seed.loc[tag]) if column == "SOLVED" else
                    "--" if pd.isna(value) else f"{value:g}")
            cells.append(f"\\textbf{{{text}}}" if best else text)
        lines.append(f"{tag.upper()} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}", ""])


def _facts(df, counts):
    """The numbers the paper's prose quotes, read off here rather than worked out by hand."""
    import pandas as pd
    solved = df[df.status == "SOLVED"]
    per_seed = _solved_per_seed(df)
    seeds = {p: sorted(df.seed[df.planner == p].unique()) for p in PLANNERS}
    by_seed = solved.groupby(["planner", "seed"]).task.agg(set).to_dict()
    sets = {p: [by_seed.get((p, s), set()) for s in seeds[p]] for p in PLANNERS}
    union = {p: set.union(*sets[p]) for p in PLANNERS}
    every = {p: set.intersection(*sets[p]) for p in PLANNERS}
    env = df.drop_duplicates("task").set_index("task").environment
    family = {e: f.split(" (")[0] for f, envs in _families(counts) for e in envs}
    medians = solved.groupby("planner").seconds.median()
    lines = ["solved per seed: " + ", ".join(f"{p} {_fmt(per_seed.loc[p])}" for p in PLANNERS),
             "solved in some seed / in every seed: " + ", ".join(
                 f"{p} {len(union[p])} / {len(every[p])}" for p in PLANNERS
                 if len(seeds[p]) > 1),
             "solved in some seed but never by bfws: " + ", ".join(
                 f"{p} {len(union[p] - union['bfws'])}" for p in PLANNERS if p != "bfws"),
             "solved by bfws and by no seed of: " + ", ".join(
                 f"{p} {len(union['bfws'] - union[p])}" for p in PLANNERS if p != "bfws"),
             "median solve time over all solved runs (s): " + ", ".join(
                 f"{p} {medians.get(p, float('nan')):.1f}" for p in PLANNERS)]
    times = (solved[solved.planner.isin(WIDTH)]
             .pivot(index="task", columns="planner", values="seconds")
             .reindex(columns=list(WIDTH)))
    for p in ("iw", "siw"):
        both = times[["bfws", p]].dropna()
        ratio = both[p] / both.bfws
        lolo = ratio[env[both.index] == "lolo"]
        lines.append(f"{p} against bfws on the {len(both)} instances both solved: slower on "
                     f"{(ratio > 1).sum()}, median ratio {ratio.median():.2f} overall and "
                     f"{lolo.median():.1f} on the Python lolo rooms")
    for status in ("ERROR", "MISSING"):
        where = df[df.status == status].groupby(["planner", "environment"]).size()
        lines.append(f"{status.lower()} over all runs: " + (", ".join(
            f"{p} on {e} {k}" for (p, e), k in where.items()) or "none"))
    iw = solved[solved.planner == "iw"]
    widths = lambda runs: ", ".join(  # noqa: E731
        f"w{w:.0f} {k}" for w, k in runs.width.value_counts().sort_index().items())
    lines.append(f"iw widths: {widths(iw)}")
    lines += [f"iw widths on {e}: {widths(runs)}" for e, runs in iw.groupby("environment")]
    lines.append(f"bfws mean plan length: "
                 f"{solved.plan_length[solved.planner == 'bfws'].mean():.1f}")
    per_family = pd.Series(counts).groupby(pd.Series(counts).index.map(family)).sum()
    for p in ("fsx", "mcts"):
        runs = df[df.planner == p]
        by_family = (runs.assign(ok=runs.status == "SOLVED")
                     .groupby([runs.environment.map(family), "seed"]).ok.sum())
        lines.append(f"{p} solved per seed, per family: " + ", ".join(
            f"{f} {_fmt(by_family.loc[f])}/{k}" for f, k in per_family.items()))
    return "\n".join(lines) + "\n"


_LINES = ("-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1)))
_MARKERS = ("o", "s", "^", "v", "D", "P", "X")


def _cactus(df, path):
    """Each planner's sorted solve times, then its time-outs and memory-outs charged the limit.

    A seeded planner's runs are pooled and the count divided by its number of seeds, which is
    exactly the mean over seeds of instances solved within each time.
    """
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (tag, runs) in enumerate(df.groupby("planner", sort=False)):
        seeds = runs.seed.nunique()
        times = sorted(runs.seconds[runs.status == "SOLVED"].clip(lower=1e-3)) \
            + [LIMITS["seconds"]] * int(runs.status.isin(["TIMEOUT", "MEMOUT"]).sum())
        # The marker interval scales with the seeds, so it stays the same distance along x.
        ax.plot([k / seeds for k in range(1, len(times) + 1)], times, color="black",
                linestyle=_LINES[i % 5], linewidth=1.3, marker=_MARKERS[i % 7],
                markevery=(i * 9, 45 * seeds), markersize=5, markerfacecolor="white",
                markeredgewidth=0.9, label=tag.upper())
    ax.axhline(LIMITS["seconds"], color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set(xlabel="instances solved", ylabel="time (s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _overlap(df, counts, path):
    """One bar per environment, split by which of BFWS, IW and SIW solved each instance."""
    import pandas as pd
    from matplotlib import pyplot as plt
    solvers = (df[df.planner.isin(WIDTH) & (df.status == "SOLVED")]
               .groupby("task").planner.agg(frozenset))
    env = df.drop_duplicates("task").set_index("task").environment
    segments = {"all three": [frozenset(WIDTH)],
                "BFWS + IW": [frozenset({"bfws", "iw"})],
                "BFWS + SIW": [frozenset({"bfws", "siw"})],
                "BFWS only": [frozenset({"bfws"})],
                "IW or SIW, not BFWS": [frozenset({"iw"}), frozenset({"siw"}),
                                        frozenset({"iw", "siw"})]}
    table = (pd.DataFrame({label: solvers[solvers.isin(sets)].groupby(env).size()
                           for label, sets in segments.items()})
             .reindex(list(counts)).fillna(0).div(pd.Series(counts), axis=0)
             .sort_values("all three"))
    ax = table.plot.barh(stacked=True, color=["black", "0.35", "0.55", "0.75", "0.88"],
                         edgecolor="black", linewidth=0.6,
                         figsize=(9, 0.42 * len(table) + 1.8))
    done = table.sum(axis=1)
    ax.barh(range(len(table)), 1 - done, left=done, facecolor="white", edgecolor="black",
            linewidth=0.6, linestyle=(0, (2, 2)), label="none of them")
    ax.set_yticklabels([f"{e} (n={counts[e]})" for e in table.index], fontsize=8)
    ax.set(xlim=(0, 1), xlabel="fraction of the environment's instances")
    ax.legend(fontsize="x-small", loc="lower right", framealpha=1.0)
    ax.figure.tight_layout()
    ax.figure.savefig(path)
    plt.close(ax.figure)


def _runtime(df, path):
    """BFWS's time per instance against IW (filled, left axis) and SIW (hollow, right axis)."""
    from matplotlib import pyplot as plt
    width = df[df.planner.isin(WIDTH)]
    penalised = width.seconds.where(width.status == "SOLVED", LIMITS["seconds"]).clip(lower=1e-3)
    t = width.assign(t=penalised).pivot(index="task", columns="planner", values="t")
    env = width.drop_duplicates("task").set_index("task").environment.reindex(t.index)
    fig, left = plt.subplots(figsize=(11, 6))
    right = left.twinx()
    for i, name in enumerate(sorted(env.unique())):
        rows, marker, shade = env == name, _MARKERS[i % 7], ("black", "0.55")[i // 7 % 2]
        left.scatter(t.bfws[rows], t.iw[rows], marker=marker, color=shade, s=22, alpha=0.8,
                     label=name)
        right.scatter(t.bfws[rows], t.siw[rows], marker=marker, facecolors="none",
                      edgecolors=shade, s=22, alpha=0.8, linewidths=0.9)
    span = (1e-3 / 1.5, LIMITS["seconds"] * 1.5)
    for ax in (left, right):
        ax.set(xscale="log", yscale="log", xlim=span, ylim=span)
    left.plot(span, span, "k--", linewidth=0.8, alpha=0.6)
    left.axhline(LIMITS["seconds"], color="grey", linestyle=":", linewidth=0.8)
    left.axvline(LIMITS["seconds"], color="grey", linestyle=":", linewidth=0.8)
    left.set(xlabel="BFWS (s)", ylabel="IW (s), filled markers")
    right.set_ylabel("SIW (s), hollow markers")
    left.grid(alpha=0.3)
    left.legend(fontsize="x-small", loc="center left", bbox_to_anchor=(1.12, 0.5),
                title="environment", title_fontsize="small")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--sandbox-dir", default="sandbox", help="where the runs go")
    parser = argparse.ArgumentParser(prog="planiverse-bench",
                                     description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    generate_ = commands.add_parser("generate", parents=[common],
                                    help="write the commands and the SLURM arrays")
    for option in ("partition", "qos", "account"):
        generate_.add_argument(f"--{option}", help=f"SLURM {option}")
    generate_.add_argument("--parallel", type=int, default=50,
                           help="array elements running at once (default: 50)")
    solve_ = commands.add_parser("solve", parents=[common],
                                 help="run one planner on one instance")
    solve_.add_argument("planner", choices=list(PLANNERS))
    solve_.add_argument("task", help="environment@index")
    solve_.add_argument("--seed", type=int, help="for mcts and fsx; the generated commands set it")
    commands.add_parser("report", parents=[common],
                        help="the paper's tables, figures and numbers")
    args = parser.parse_args(argv)
    if args.command == "generate":
        generate(args.sandbox_dir, args.partition, args.qos, args.account, args.parallel)
    elif args.command == "solve":
        solve(args.sandbox_dir, args.planner, args.task, args.seed)
    else:
        report(args.sandbox_dir)
    return 0
