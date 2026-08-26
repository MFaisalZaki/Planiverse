"""Working out what there is to run.

A task is one `(environment, index)` pair, written `puzznic@7`. Discovery turns the registry
plus a `TaskSelection` into a concrete list of them, and writes it to `tasks.json` so that
every later stage agrees on what the experiment covers. Re-probing at each stage would be both
slow and unsound: an environment whose dependencies vanish between `generate` and `analyze`
would silently shrink the benchmark rather than showing up as missing results.
"""
import json
import os

from planiverse.benchmark.measures import has_measure
from planiverse.environments import REGISTRY, get_spec

#: How far `count_instances` will probe before deciding an environment is unbounded.
PROBE_CEILING = 512


def task_id(environment, index):
    return f"{environment}@{index}"


def parse_task_id(identifier):
    environment, _, index = str(identifier).rpartition("@")
    if not environment or not index.isdigit():
        raise ValueError(f"{identifier!r} is not a task id: expected 'environment@index'")
    return environment, int(index)


def task_filename(identifier):
    """A task id as a filename. `@` is legal on POSIX but awkward in shell and in URLs."""
    environment, index = parse_task_id(identifier)
    return f"{environment}__{index}"


def count_instances(spec, ceiling=PROBE_CEILING, rom=None):
    """How many instances an environment offers, by asking it.

    Probed rather than declared. The registry's `instances` field is prose written for a
    human — "50 levels", "9 contingencies" — and a second, machine-readable copy would be one
    more thing that can drift out of step with the code. So this constructs the environment
    once and walks `fix_index` upwards until it refuses.

    Environments differ in how they refuse — `IndexError`, `AssertionError`, a bare
    `ValueError` — so anything raised counts as the end of the range.
    """
    environment = spec.build(**({spec.rom_argument: rom} if rom else {}))
    try:
        count = 0
        while count < ceiling:
            try:
                environment.fix_index(count)
            except Exception:
                break
            count += 1
        return count
    finally:
        close = getattr(environment, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def choose(count, wanted, strategy="even"):
    """Pick `wanted` indices out of `count`.

    `"even"` spreads them across the range, `"first"` takes a prefix. Even is the default
    because instance 0 of most of these environments is a tutorial, and a benchmark made of
    tutorials measures very little.
    """
    if count <= 0:
        return ()
    if wanted is None or wanted <= 0 or wanted >= count:
        return tuple(range(count))
    if strategy == "first":
        return tuple(range(wanted))
    if strategy != "even":
        raise ValueError(f"selection must be 'even' or 'first', got {strategy!r}")
    if wanted == 1:
        return (0,)
    step = (count - 1) / (wanted - 1)
    return tuple(sorted({int(round(i * step)) for i in range(wanted)}))


def eligible(spec, selection, rom=None):
    """Is this environment in scope, and if not, why not?

    Returns `(True, "")` or `(False, reason)`. The reason is kept because "skipped" and
    "skipped because PyBoy is not installed" are different things to read in a report.

    `rom` is the cartridge the experiment has for this environment, if any. Missing
    dependencies and a missing cartridge are reported separately: one is fixed by installing
    something, the other by supplying a file only you have.
    """
    if selection.include_environments and spec.name not in selection.include_environments:
        return False, "not in include-environments"
    if spec.name in selection.exclude_environments:
        return False, "in exclude-environments"
    if selection.tags and not (spec.tags & set(selection.tags)):
        return False, f"carries none of the tags {sorted(selection.tags)}"
    if spec.needs_rom and not selection.include_rom_environments:
        return False, "excluded by include-rom-environments"
    missing = [name for name in spec.requires if not _importable(name)]
    if missing:
        return False, f"missing {', '.join(missing)}"
    if spec.needs_rom and rom is None:
        return False, ('no cartridge — add one to the experiment\'s "roms", set '
                       f"{spec.rom_variable}, or run ./setup_benchmark.sh")
    return True, ""


def _importable(module_name):
    import importlib
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def _rom_for(spec, roms):
    """The cartridge for an environment: the experiment's, else its variable, else None.

    The experiment's copy wins so that a run is reproducible from its config alone, but the
    path is still checked — one recorded on the machine that wrote the config is a promise
    about a different filesystem until it is.
    """
    if not spec.needs_rom:
        return None
    path = (roms or {}).get(spec.name)
    if path and os.path.isfile(path):
        return path
    return spec.rom_path()


def discover(selection, registry=REGISTRY, roms=None):
    """The tasks an experiment covers, plus what was left out and why.

    `roms` maps an environment name to a cartridge path, for the Game Boy environments.
    Returns `{"tasks": [...], "environments": [...], "skipped": [...]}`.
    """
    roms = roms or {}
    if selection.selected_tasks:
        return _explicit(selection, registry)

    tasks, environments, skipped = [], [], []
    for spec in registry:
        rom = _rom_for(spec, roms)
        ok, reason = eligible(spec, selection, rom)
        if not ok:
            skipped.append({"environment": spec.name, "reason": reason})
            continue
        try:
            count = count_instances(spec, rom=rom)
        except Exception as exc:
            skipped.append({"environment": spec.name,
                            "reason": f"could not be built: {type(exc).__name__}: {exc}"})
            continue
        if not count:
            skipped.append({"environment": spec.name, "reason": "offers no instances"})
            continue

        indices = choose(count, selection.max_instances_per_environment, selection.selection)
        environments.append({
            "environment": spec.name,
            "instances": count,
            "selected": len(indices),
            "state_identity": spec.state_identity,
            "has_progress_measure": has_measure(spec.name),
            "tags": sorted(spec.tags),
        })
        tasks += [{"id": task_id(spec.name, index), "environment": spec.name, "index": index}
                  for index in indices]

    return {"tasks": tasks, "environments": environments, "skipped": skipped}


def _explicit(selection, registry):
    """`selected-tasks` given by hand. Everything else in the selection is ignored."""
    known = {spec.name for spec in registry}
    tasks, skipped, counted = [], [], {}
    for identifier in selection.selected_tasks:
        environment, index = parse_task_id(identifier)
        if environment not in known:
            skipped.append({"environment": environment, "reason": "not in the registry"})
            continue
        counted[environment] = counted.get(environment, 0) + 1
        tasks.append({"id": task_id(environment, index), "environment": environment,
                      "index": index})
    environments = [{"environment": name, "instances": None, "selected": count,
                     "state_identity": get_spec(name).state_identity,
                     "has_progress_measure": has_measure(name),
                     "tags": sorted(get_spec(name).tags)}
                    for name, count in sorted(counted.items())]
    return {"tasks": tasks, "environments": environments, "skipped": skipped}


def pair_up(tasks, planners, registry=REGISTRY):
    """Every (planner, task) the experiment should run.

    A planner's own `tags` and `exclude_environments` narrow it further, so one experiment can
    run a cheap planner over everything and an expensive one over a subset.
    """
    specs = {spec.name: spec for spec in registry}
    pairs = []
    for planner in planners:
        for task in tasks:
            spec = specs.get(task["environment"])
            if spec is None:
                continue
            if planner.tags and not (spec.tags & set(planner.tags)):
                continue
            if task["environment"] in planner.exclude_environments:
                continue
            pairs.append({"planner": planner.tag, "task": task["id"],
                          "environment": task["environment"], "index": task["index"]})
    return pairs


def write_tasks(sandbox_dir, discovered, pairs, experiment_dir=None):
    os.makedirs(sandbox_dir, exist_ok=True)
    payload = dict(discovered)
    payload["pairs"] = pairs
    if experiment_dir:
        payload["experiment"] = os.path.abspath(experiment_dir)
    path = os.path.join(sandbox_dir, "tasks.json")
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def read_tasks(sandbox_dir):
    path = os.path.join(sandbox_dir, "tasks.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} does not exist. Run `planiverse-bench discover` first.")
    with open(path) as handle:
        return json.load(handle)
