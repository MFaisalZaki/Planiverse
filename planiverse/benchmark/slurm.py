"""Turning a list of (planner, task) pairs into jobs a cluster will accept.

The shape follows [pyPMTEvalToolkit](https://github.com/pyPMT/pyPMTEvalToolkit): one command
per line in `cmds/<planner>.txt`, one **job array** per planner reading its line by
`$SLURM_ARRAY_TASK_ID`, and a `submit_all.sh` that fires them off. One array per planner rather
than one job per run, because a benchmark is thousands of short runs and a scheduler handling
them as thousands of jobs spends longer scheduling than computing.

Three things a generated array has to get right, none of which are obvious the first time:

**Array size.** `MaxArraySize` is a site limit, commonly 1001, and an array over it is
rejected at submission with a message that does not name the cause. So a long command list is
split across several `sbatch` files with an offset baked in, and each file covers a slice.

**Throttling.** `--array=0-999%50` runs fifty at a time. Without the `%`, a benchmark
submitted on a shared cluster takes every free node on the partition, which is how you become
unpopular.

**Headroom.** SLURM's `--time` and `--mem` are set *above* the harness's own limits. The
harness wants to notice its own timeout and write a `TIMEOUT` result; if SLURM kills it at the
same instant, the row is missing instead — and a missing row looks like an infrastructure
problem rather than a slow planner.
"""
import os
import stat

from planiverse.benchmark.config import format_slurm_time, parse_duration, parse_size

#: The console script the jobs call. Overridable for a checkout that is not pip-installed.
DEFAULT_ENTRY_POINT = "planiverse-bench"


def command_for(entry_point, planner_tag, task, experiment_dir, sandbox_dir, seed=None):
    """The shell command that runs one pair."""
    parts = [entry_point, "solve",
             "--exp-dir", _quote(experiment_dir),
             "--sandbox-dir", _quote(sandbox_dir),
             "--planner", _quote(planner_tag),
             "--task", _quote(task)]
    if seed is not None:
        parts += ["--seed", str(seed)]
    return " ".join(parts)


def _quote(text):
    text = str(text)
    return text if all(c.isalnum() or c in "./_-@=+:" for c in text) else "'%s'" % \
        text.replace("'", "'\\''")


def write_commands(sandbox_dir, pairs, experiment_dir, entry_point=DEFAULT_ENTRY_POINT,
                   seed=None):
    """One `cmds/<planner>.txt` per planner. Returns `{tag: (path, count)}`.

    Line *n* of the file is array index *n*, and nothing downstream re-derives the ordering —
    it reads the file. That is what lets a failed element be re-run by hand: line 412 of the
    file is exactly what array index 412 ran.
    """
    directory = os.path.join(sandbox_dir, "cmds")
    os.makedirs(directory, exist_ok=True)

    grouped = {}
    for pair in pairs:
        grouped.setdefault(pair["planner"], []).append(pair)

    written = {}
    for tag, entries in sorted(grouped.items()):
        path = os.path.join(directory, f"{tag}.txt")
        with open(path, "w") as handle:
            for pair in entries:
                handle.write(command_for(entry_point, tag, pair["task"], experiment_dir,
                                         sandbox_dir, seed=seed) + "\n")
        written[tag] = (path, len(entries))
    return written


def array_chunks(count, max_array_size):
    """Split `count` jobs into `(offset, length)` slices no longer than the site limit."""
    if count <= 0:
        return ()
    if not max_array_size or max_array_size <= 0 or count <= max_array_size:
        return ((0, count),)
    chunks = []
    offset = 0
    while offset < count:
        chunks.append((offset, min(max_array_size, count - offset)))
        offset += max_array_size
    return tuple(chunks)


def sbatch_script(name, planner_tag, commands_path, count, offset, limits, slurm,
                  sandbox_dir):
    """One `sbatch` file for one slice of one planner's array."""
    time_limit = parse_duration(limits.time) + parse_duration(slurm.time_headroom)
    memory_mb = (parse_size(limits.memory) + parse_size(slurm.memory_headroom)) // (1024 ** 2)
    logs = os.path.join(sandbox_dir, "logs", planner_tag)

    array = f"0-{count - 1}"
    if slurm.max_parallel_jobs and slurm.max_parallel_jobs > 0:
        array += f"%{slurm.max_parallel_jobs}"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={name}-{planner_tag}",
        f"#SBATCH --array={array}",
        f"#SBATCH --time={format_slurm_time(time_limit)}",
        f"#SBATCH --mem={memory_mb}M",
        f"#SBATCH --cpus-per-task={slurm.cpus_per_task}",
        f"#SBATCH --output={logs}/%A_%a.out",
        f"#SBATCH --error={logs}/%A_%a.err",
    ]
    if slurm.partition:
        lines.append(f"#SBATCH --partition={slurm.partition}")
    if slurm.account:
        lines.append(f"#SBATCH --account={slurm.account}")
    if slurm.qos:
        lines.append(f"#SBATCH --qos={slurm.qos}")
    lines += [f"#SBATCH {directive}" for directive in slurm.extra_directives]

    lines += [
        "",
        "# `set -u` is deliberately absent: SLURM_ARRAY_TASK_ID is unset when this file is",
        "# run directly, and the fallback below is how a single element gets re-run by hand.",
        "set -eo pipefail",
        "",
        f"COMMANDS={_quote(commands_path)}",
        f"OFFSET={offset}",
        'INDEX=$(( ${SLURM_ARRAY_TASK_ID:-0} + OFFSET ))',
        "",
        "# sed is 1-indexed, the array is 0-indexed.",
        'COMMAND=$(sed -n "$(( INDEX + 1 ))p" "$COMMANDS")',
        'if [ -z "$COMMAND" ]; then',
        '    echo "no command at line $(( INDEX + 1 )) of $COMMANDS" >&2',
        "    exit 1",
        "fi",
        "",
    ]
    lines += list(slurm.setup_commands)
    lines += [
        "",
        'echo "[$(date -Is)] $COMMAND"',
        "# The harness records its own failures as results, so a non-zero exit here means the",
        "# run died in a way it could not catch. Let it through rather than masking it.",
        'eval "$COMMAND"',
        "",
    ]
    return "\n".join(lines)


def generate(sandbox_dir, pairs, experiment, experiment_dir,
             entry_point=DEFAULT_ENTRY_POINT, seed=None, per_task_scripts=False):
    """Write the whole job set. Returns a summary dictionary.

    `per_task_scripts` writes one `sbatch` file per run instead of arrays — far more files and
    far more scheduler load, but some sites disable job arrays entirely, and one file per run
    is the fallback that always works.

    Every path written into a job is absolute. `sbatch` inherits the submitting shell's
    working directory, which holds right up until someone submits from elsewhere or the
    scheduler starts the job on a node that mounts the tree at a different point — and then a
    relative path fails a thousand array elements at once, each with an empty log.
    """
    sandbox_dir = os.path.abspath(sandbox_dir)
    experiment_dir = os.path.abspath(experiment_dir)
    commands = write_commands(sandbox_dir, pairs, experiment_dir, entry_point, seed)
    slurm_dir = os.path.join(sandbox_dir, "slurm")
    os.makedirs(slurm_dir, exist_ok=True)

    scripts, total = [], 0
    for tag, (path, count) in sorted(commands.items()):
        os.makedirs(os.path.join(sandbox_dir, "logs", tag), exist_ok=True)
        total += count
        if per_task_scripts:
            scripts += _per_task(slurm_dir, tag, path, count, experiment, sandbox_dir)
            continue
        chunks = array_chunks(count, experiment.slurm.max_array_size)
        for number, (offset, length) in enumerate(chunks):
            suffix = "" if len(chunks) == 1 else f".part{number + 1}"
            script = os.path.join(slurm_dir, f"{experiment.name}-{tag}{suffix}.sbatch")
            _write_executable(script, sbatch_script(
                experiment.name, tag, path, length, offset, experiment.limits,
                experiment.slurm, sandbox_dir))
            scripts.append(script)

    submit = os.path.join(slurm_dir, "submit_all.sh")
    _write_executable(submit, _submit_all(scripts))
    local = os.path.join(sandbox_dir, "run_local.sh")
    _write_executable(local, _run_local(sorted(path for path, _ in commands.values()),
                                        experiment.slurm.setup_commands))

    return {"commands": {tag: {"path": path, "count": count}
                         for tag, (path, count) in commands.items()},
            "scripts": scripts, "submit_all": submit, "run_local": local,
            "runs": total}


def _per_task(slurm_dir, tag, commands_path, count, experiment, sandbox_dir):
    directory = os.path.join(slurm_dir, tag)
    os.makedirs(directory, exist_ok=True)
    written = []
    for index in range(count):
        script = os.path.join(directory, f"{index:05d}.sbatch")
        body = sbatch_script(experiment.name, tag, commands_path, 1, index,
                             experiment.limits, experiment.slurm, sandbox_dir)
        # A single job, not an array of one: `--array=0-0` still allocates an array id.
        body = "\n".join(line for line in body.split("\n")
                         if not line.startswith("#SBATCH --array="))
        body = body.replace("%A_%a", f"{tag}-{index:05d}-%j")
        _write_executable(script, body)
        written.append(script)
    return written


def _submit_all(scripts):
    lines = [
        "#!/bin/bash",
        "# Submit every generated job. Each sbatch prints the job id it was given.",
        "set -euo pipefail",
        "",
    ]
    lines += [f"sbatch {_quote(script)}" for script in scripts]
    if not scripts:
        lines.append('echo "nothing to submit" >&2')
    lines.append("")
    return "\n".join(lines)


def _run_local(command_files, setup_commands=()):
    """The no-cluster path.

    Uses GNU parallel when it is there and `xargs -P` when it is not, because `xargs` is
    everywhere and the difference does not matter for running a command per line.

    It runs the same `setup_commands` an sbatch job does. Skipping them here was a real hole:
    `setup_benchmark.sh` builds a virtualenv and puts its activation in those commands, and a
    local run that ignored them used whichever interpreter happened to be on PATH — which is
    exactly the kind of difference that makes two runs of "the same" benchmark disagree.
    """
    lines = [
        "#!/bin/bash",
        "# Run the whole benchmark here instead of on a cluster.",
        "#   bash run_local.sh [parallel-jobs]",
        "# The harness applies the same limits either way, so results are comparable — but",
        "# wall-clock timings from a loaded laptop are not comparable with a cluster's.",
        "set -euo pipefail",
        "",
    ]
    if setup_commands:
        lines += ["# The same setup the cluster jobs run, so both use the same interpreter.",
                  *setup_commands, ""]
    return "\n".join(lines + [
        'JOBS="${1:-4}"',
        f"FILES=({' '.join(_quote(path) for path in command_files)})",
        "",
        'for FILE in "${FILES[@]}"; do',
        '    echo "== $FILE ($(wc -l < "$FILE") runs, $JOBS at a time)"',
        "    if command -v parallel > /dev/null 2>&1; then",
        '        parallel --will-cite -j "$JOBS" < "$FILE"',
        "    else",
        '        xargs -a "$FILE" -d "\\n" -P "$JOBS" -I {} bash -c "{}"',
        "    fi",
        "done",
        "",
    ])


def _write_executable(path, body):
    with open(path, "w") as handle:
        handle.write(body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return path
