"""What an experiment is, as JSON on disk.

Modelled on [pyPMTEvalToolkit](https://github.com/pyPMT/pyPMTEvalToolkit): an experiment is a
directory holding one `exp-details.json` and one file per planner under `planners/`, and every
later stage reads that directory rather than command-line flags. Keeping the definition on disk
is what makes a run reproducible — the sandbox records which experiment produced it, so a
result can always be traced back to the limits it was obtained under.
"""
import json
import os
import re
from dataclasses import asdict, dataclass, field

#: Where an experiment keeps its planner definitions.
PLANNERS_DIRNAME = "planners"

#: The experiment file itself.
DETAILS_FILENAME = "exp-details.json"

_DURATION = re.compile(r"^(?:(\d+):)?(\d+):(\d+)$|^(\d+(?:\.\d+)?)\s*([smhd])$", re.I)
_SIZE = re.compile(r"^(\d+(?:\.\d+)?)\s*([kmgt]?)i?b?$", re.I)
_SIZE_UNITS = {"": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3, "t": 1024 ** 4}
_TIME_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(text):
    """`"30m"`, `"00:30:00"` or `1800` into seconds.

    Both spellings are accepted because both turn up: SLURM writes `HH:MM:SS` and humans
    write `30m`.
    """
    if isinstance(text, (int, float)):
        return float(text)
    match = _DURATION.match(str(text).strip())
    if not match:
        raise ValueError(f"cannot read {text!r} as a duration: try '30m' or '00:30:00'")
    if match.group(3) is not None:
        hours, minutes, seconds = match.group(1) or 0, match.group(2), match.group(3)
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    return float(match.group(4)) * _TIME_UNITS[match.group(5).lower()]


def parse_size(text):
    """`"8GB"`, `"8G"` or a byte count into bytes."""
    if isinstance(text, (int, float)):
        return int(text)
    match = _SIZE.match(str(text).strip())
    if not match:
        raise ValueError(f"cannot read {text!r} as a size: try '8GB'")
    return int(float(match.group(1)) * _SIZE_UNITS[match.group(2).lower()])


def format_slurm_time(seconds):
    """Seconds as SLURM's `HH:MM:SS`, rounded up so the limit is never shortened."""
    seconds = int(seconds + 0.999)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


@dataclass
class SlurmConfig:
    """The directives that go at the top of a generated `sbatch` file.

    `partition` and `account` are deliberately `None` by default: they are site-specific, and
    a wrong guess produces a job that is rejected at submission rather than one that runs
    somewhere unexpected.
    """

    cpus_per_task: int = 1
    partition: str = None
    account: str = None
    qos: str = None
    #: `--array=...%N`. Without it a 4000-task array can swamp a shared cluster.
    max_parallel_jobs: int = 50
    #: Arrays longer than this are split into several `sbatch` files. 1000 is a common
    #: `MaxArraySize`; check yours with `scontrol show config | grep MaxArraySize`.
    max_array_size: int = 1000
    #: Added to the time and memory limits so SLURM kills the job *after* the harness has had
    #: its own chance to record a TIMEOUT or MEMOUT. Without headroom every timeout looks
    #: like a cancelled job and the result file is never written.
    time_headroom: str = "00:05:00"
    memory_headroom: str = "1GB"
    extra_directives: tuple = ()
    #: Prepended to every job body — `module load`, `conda activate`, and so on.
    setup_commands: tuple = ()


@dataclass
class TaskSelection:
    """Which (environment, index) pairs the experiment runs on."""

    #: Registry tags to include; empty means every environment.
    tags: tuple = ()
    include_environments: tuple = ()
    exclude_environments: tuple = ()
    #: 0 means every instance. Capping is for a quick look; a benchmark that reports
    #: coverage over a tenth of each environment is reporting on a sample it chose.
    max_instances_per_environment: int = 0
    #: `"even"` spreads the chosen indices across the range, `"first"` takes a prefix. Even
    #: is the default because instance 0 of most of these environments is a tutorial.
    selection: str = "even"
    #: Environments whose dependencies are missing are skipped rather than recorded as
    #: failures — running half a catalogue and saying so beats refusing to run at all.
    skip_unavailable: bool = True
    #: Environments needing a ROM the user supplies. On, so that a cartridge-backed
    #: environment and its pure-Python twin are benchmarked side by side — which is most of
    #: the point of having both. Without a cartridge they are skipped with a reason, so
    #: leaving this on costs nothing when the files are absent.
    include_rom_environments: bool = True
    #: Explicit `env@index` strings. When set, everything above is ignored.
    selected_tasks: tuple = ()


@dataclass
class Limits:
    """What one (planner, task) run is allowed to consume."""

    time: str = "30m"
    memory: str = "8GB"
    #: The node allowance. Against a simulator this is usually the binding limit, because
    #: wall-clock is just however many expansions you allowed times the cost of one.
    max_expansions: int = 1000000
    #: Replay every plan and check it reaches a goal. Cheap, and it catches a planner that
    #: reports success on a trace it cannot reproduce.
    validate_plans: bool = True

    def seconds(self):
        return parse_duration(self.time)

    def bytes(self):
        return parse_size(self.memory)


@dataclass
class PlannerSpec:
    """One planner entry, as `planners/<tag>.json`.

    `tag` names it everywhere afterwards — in filenames, in the sbatch job name, and in every
    table — so it has to be filesystem-safe and stable.
    """

    tag: str
    planner: str                      #: a name from `catalogue.PLANNERS`
    params: dict = field(default_factory=dict)
    #: Restrict this planner to environments carrying one of these tags. Empty means all.
    tags: tuple = ()
    exclude_environments: tuple = ()
    enabled: bool = True

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9._-]+", self.tag or ""):
            raise ValueError(
                f"planner tag {self.tag!r} must be filesystem-safe: letters, digits, dot, "
                f"dash and underscore only")


@dataclass
class ExperimentConfig:
    """A whole experiment: limits, task selection, SLURM settings and the planners."""

    name: str = "planiverse-bench"
    limits: Limits = field(default_factory=Limits)
    tasks: TaskSelection = field(default_factory=TaskSelection)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    planners: tuple = ()
    #: `environment name -> cartridge path`, for the Game Boy environments. Held here rather
    #: than left to environment variables so the experiment is self-contained: a variable
    #: exported in the shell that ran `generate` is not there on the compute node, and the
    #: whole array would come back UNSUPPORTED. A variable is still honoured as a fallback.
    roms: dict = field(default_factory=dict)

    # ------------------------------------------------------------------------ on disk

    @classmethod
    def load(cls, experiment_dir):
        """Read `<experiment_dir>/exp-details.json` and every `planners/*.json`."""
        details_path = os.path.join(experiment_dir, DETAILS_FILENAME)
        if not os.path.isfile(details_path):
            raise FileNotFoundError(
                f"{details_path} does not exist. Run `planiverse-bench init "
                f"--exp-dir {experiment_dir}` first.")
        with open(details_path) as handle:
            details = json.load(handle)

        planners = []
        planners_dir = os.path.join(experiment_dir, PLANNERS_DIRNAME)
        for filename in sorted(os.listdir(planners_dir)) if os.path.isdir(planners_dir) else []:
            if not filename.endswith(".json"):
                continue
            with open(os.path.join(planners_dir, filename)) as handle:
                entry = json.load(handle)
            entry.setdefault("tag", os.path.splitext(filename)[0])
            planners.append(PlannerSpec(**_snake(entry, PlannerSpec)))

        duplicates = {spec.tag for spec in planners
                      if sum(1 for other in planners if other.tag == spec.tag) > 1}
        if duplicates:
            raise ValueError(f"two planner files share a tag: {sorted(duplicates)}")

        return cls(
            name=details.get("name", "planiverse-bench"),
            limits=Limits(**_snake(details.get("limits", {}), Limits)),
            tasks=TaskSelection(**_snake(details.get("tasks", {}), TaskSelection)),
            slurm=SlurmConfig(**_snake(details.get("slurm", {}), SlurmConfig)),
            planners=tuple(planners),
            roms=dict(details.get("roms", {})),
        )

    def save(self, experiment_dir):
        """Write the experiment out, one file for the details and one per planner."""
        os.makedirs(os.path.join(experiment_dir, PLANNERS_DIRNAME), exist_ok=True)
        details = {
            "name": self.name,
            "limits": _kebab(asdict(self.limits)),
            "tasks": _kebab(asdict(self.tasks)),
            "slurm": _kebab(asdict(self.slurm)),
            "roms": dict(self.roms),
        }
        with open(os.path.join(experiment_dir, DETAILS_FILENAME), "w") as handle:
            json.dump(details, handle, indent=4)
            handle.write("\n")
        for spec in self.planners:
            path = os.path.join(experiment_dir, PLANNERS_DIRNAME, f"{spec.tag}.json")
            with open(path, "w") as handle:
                json.dump(_kebab(asdict(spec)), handle, indent=4)
                handle.write("\n")
        return experiment_dir

    def active_planners(self):
        return tuple(spec for spec in self.planners if spec.enabled)

    def rom_for(self, spec):
        """The cartridge for an environment: this experiment's, else its variable, else None.

        Checked for existence, because a path recorded on the machine that wrote the config
        is a promise about a different filesystem until it is.
        """
        path = self.roms.get(spec.name)
        if path and os.path.isfile(path):
            return path
        return spec.rom_path()


def _snake(mapping, cls):
    """`{"max-array-size": 1000}` into `{"max_array_size": 1000}`, dropping unknown keys.

    Unknown keys are dropped rather than raising so that a config written for a later version
    still loads. Tuple fields are rebuilt from the JSON lists.
    """
    fields = {f.name: f for f in cls.__dataclass_fields__.values()}
    converted = {}
    for key, value in mapping.items():
        name = str(key).replace("-", "_")
        if name not in fields:
            continue
        if isinstance(value, list):
            value = tuple(value)
        converted[name] = value
    return converted


def _kebab(mapping):
    return {key.replace("_", "-"): (list(value) if isinstance(value, tuple) else value)
            for key, value in mapping.items()}
