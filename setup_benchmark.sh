#!/bin/bash
# Set up a Planiverse benchmark, interactively.
#
# Builds a virtualenv, installs planiverse into it, walks you through the preparation —
# limits, cartridge paths, SLURM settings — and runs init, discover and generate, leaving you
# a directory of jobs to submit. Every answer has a default, so holding Enter through it gives
# a complete experiment.
#
# The generated jobs call the venv's own planiverse-bench by absolute path and activate the
# venv before running, so they do not depend on whatever happens to be on PATH on a node.
#
#   ./setup_benchmark.sh                    # ask about everything
#   ./setup_benchmark.sh --yes              # take every default, ask nothing
#   ./setup_benchmark.sh --rom-puzznic ~/roms/Puzznic.gb --rom-flipull ~/roms/Flipull.gb
#   ./setup_benchmark.sh --venv /shared/planiverse-venv     # put the venv somewhere else
#   ./setup_benchmark.sh --exp-dir e --sandbox-dir s
#
# The reason this exists rather than a line in the README is the cartridges. Puzznic, Flipull
# and Super Mario Land are copyrighted and cannot ship here, so their paths can only come
# from you — and an experiment that silently skipped them would quietly be benchmarking half
# of what it claims to. This asks.
set -euo pipefail

EXP_DIR="experiment"
SANDBOX_DIR="sandbox"
NAME="planiverse-bench"
TIME_LIMIT="30m"
MEMORY_LIMIT="8GB"
MAX_EXPANSIONS="100000"
MAX_INSTANCES="0"          # 0 = every instance of every environment
ENVIRONMENTS=""            # empty = every environment; else comma-separated registry names
PARTITION=""
ACCOUNT=""
QOS=""
SEED="0"
ASSUME_YES=0
ENTRY_POINT=""
VENV_DIR=""
USE_VENV=1
PYTHON_BIN="python3"

usage() {
    sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'USAGE'

Options:
  --venv DIR            where the virtualenv goes            (default: <repo>/.venv)
                        On a cluster this must be on a filesystem the compute nodes can
                        see, or every job will fail identically.
  --no-venv             do not build one; use whatever planiverse is already importable
  --python BIN          interpreter to build the venv with          (default: python3)
  --rom-puzznic PATH    Puzznic cartridge; skips the question for it
  --rom-flipull PATH    Flipull cartridge
  --rom-boxxle2 PATH    Boxxle II cartridge
  --rom-lolo PATH       Adventures of Lolo cartridge
  --rom-amazing-tater PATH  Amazing Tater cartridge
  --rom-super-mario-land PATH  Super Mario Land cartridge (--rom-mario and --rom-sml work too)
  --exp-dir DIR         where to write the experiment  (default: experiment)
  --sandbox-dir DIR     where results will go          (default: sandbox)
  --name NAME           experiment name, used in job names
  --time DURATION       per-run wall clock             (default: 30m)
  --memory SIZE         per-run memory                 (default: 8GB)
  --max-expansions N    per-run node budget            (default: 100000)
  --max-instances N     instances per environment, 0 for all  (default: 0)
  --environments LIST   comma-separated registry names to run; skips the question
                        (default: every environment)
  --partition NAME      SLURM partition
  --account NAME        SLURM account
  --qos NAME            SLURM quality of service
  --seed N              seed for the randomised planners       (default: 0)
  --entry-point CMD     how jobs invoke the CLI; defaults to planiverse-bench when it is
                        installed, otherwise python -m planiverse.benchmark.cli
  --yes, -y             accept every default without asking
  --help, -h            this
USAGE
}

ROM_ARGS=()
SUPPLIED_ROMS=()

while [ $# -gt 0 ]; do
    case "$1" in
        # --rom-puzznic, --rom-flipull, --rom-super-mario-land (--rom-mario), and whatever a future Game
        # Boy environment adds. Matched by shape and handed straight to `init`, which
        # generates the real flags from the registry — so this script never holds a second
        # list of them that could fall out of step.
        --rom-*)
            ROM_ARGS+=("$1" "$2")
            SUPPLIED_ROMS+=("${1#--rom-}")
            shift 2 ;;
        --exp-dir) EXP_DIR="$2"; shift 2 ;;
        --sandbox-dir) SANDBOX_DIR="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --time) TIME_LIMIT="$2"; shift 2 ;;
        --memory) MEMORY_LIMIT="$2"; shift 2 ;;
        --max-expansions) MAX_EXPANSIONS="$2"; shift 2 ;;
        --max-instances) MAX_INSTANCES="$2"; shift 2 ;;
        --environments) ENVIRONMENTS="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --qos) QOS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --entry-point) ENTRY_POINT="$2"; shift 2 ;;
        --venv) VENV_DIR="$2"; shift 2 ;;
        --no-venv) USE_VENV=0; shift ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --yes|-y) ASSUME_YES=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_COMMANDS=()

if [ "$USE_VENV" = "1" ]; then
    : "${VENV_DIR:=$REPO/.venv}"

    if [ ! -d "$VENV_DIR" ]; then
        echo "== creating virtualenv at $VENV_DIR"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    else
        echo "== reusing virtualenv at $VENV_DIR"
    fi
    VENV_DIR="$(cd "$VENV_DIR" && pwd)"

    # shellcheck disable=SC1091
    . "$VENV_DIR/bin/activate"
    echo "== installing planiverse from $REPO"
    python -m pip install --quiet --upgrade pip setuptools wheel
    python -m pip install --quiet -e "$REPO"
    deactivate

    # The console script inside the venv, by absolute path. That is what the generated jobs
    # call, and it is deliberately not "activate, then run planiverse-bench": a job runs in a
    # shell that never saw this activation, and one that silently falls back to some other
    # interpreter on PATH is worse than one that fails. An absolute path cannot do either.
    CLI="$VENV_DIR/bin/planiverse-bench"
    if [ ! -x "$CLI" ]; then
        echo "planiverse-bench was not installed into $VENV_DIR" >&2
        exit 1
    fi
    ENTRY_POINT="$CLI"
    # Activated as well, so anything a job runs after the CLI — and anything you run by hand
    # in one of these directories — gets the same interpreter.
    SETUP_COMMANDS+=(--setup-command ". $VENV_DIR/bin/activate")
else
    if [ -z "$ENTRY_POINT" ]; then
        if command -v planiverse-bench > /dev/null 2>&1; then
            ENTRY_POINT="planiverse-bench"
        else
            ENTRY_POINT="python -m planiverse.benchmark.cli"
        fi
    fi
fi

BENCH=($ENTRY_POINT)

# Fail here rather than three stages in. Without this the script picks an entry point, runs
# `init`, and dies on an ImportError that says nothing about what to do next.
if ! "${BENCH[@]}" --help > /dev/null 2>&1; then
    cat >&2 <<INSTALL

planiverse is not usable through '$ENTRY_POINT'.

Drop --no-venv and this script will build one and install into it, or install
it yourself with:

  pip install -e "$REPO"
INSTALL
    exit 1
fi

ask() {
    # ask <prompt> <default> -> echoes the answer
    local prompt="$1" default="$2" answer
    if [ "$ASSUME_YES" = "1" ] || [ ! -t 0 ]; then
        echo "$default"
        return
    fi
    read -r -p "$prompt [$default]: " answer < /dev/tty || answer=""
    echo "${answer:-$default}"
}

ask_rom() {
    # ask_rom <environment> <variable> <label> -> echoes "env=path" or nothing
    local environment="$1" variable="$2" label="$3"
    local default="${!variable:-}" answer

    # Every exit is an explicit `return 0`. A bare `return` carries the status of the last
    # command, so a skipped cartridge left a failing `[ -n ... ]` as the result — and
    # `rom=$(ask_rom ...)` takes its status, which under `set -e` ended the script on the
    # first environment the user does not have a ROM for.
    if [ "$ASSUME_YES" = "1" ] || [ ! -t 0 ]; then
        if [ -n "$default" ] && [ -f "$default" ]; then
            echo "$environment=$default"
        fi
        return 0
    fi

    while true; do
        read -r -p "  $label ROM (blank to skip) [${default:-none}]: " answer < /dev/tty \
            || answer=""
        answer="${answer:-$default}"
        if [ -z "$answer" ]; then
            return 0
        fi
        # Expand a leading ~ and strip quotes a drag-and-drop leaves behind.
        answer="${answer%\"}"; answer="${answer#\"}"
        answer="${answer%\'}"; answer="${answer#\'}"
        answer="${answer/#\~/$HOME}"
        if [ -f "$answer" ]; then
            echo "$environment=$(cd "$(dirname "$answer")" && pwd)/$(basename "$answer")"
            return 0
        fi
        echo "    no file at $answer — try again, or press Enter to skip." >&2
        default=""
    done
}

echo "Planiverse benchmark setup"
echo "=========================="
echo

if [ "$ASSUME_YES" != "1" ] && [ -t 0 ]; then
    echo "Limits for a single (planner, task) run."
    TIME_LIMIT=$(ask "  wall-clock limit" "$TIME_LIMIT")
    MEMORY_LIMIT=$(ask "  memory limit" "$MEMORY_LIMIT")
    MAX_EXPANSIONS=$(ask "  expansion budget" "$MAX_EXPANSIONS")
    echo
    echo "Task selection."
    MAX_INSTANCES=$(ask "  instances per environment (0 = all)" "$MAX_INSTANCES")
    if [ -z "$ENVIRONMENTS" ]; then
        echo "  What can be benchmarked here:"
        "${BENCH[@]}" environments 2> /dev/null | sed 's/^/    /' || true
        ENVIRONMENTS=$(ask "  environments to run, comma-separated" "all")
        if [ "$ENVIRONMENTS" = "all" ]; then ENVIRONMENTS=""; fi
    fi
    echo
    echo "SLURM. Leave blank if your site does not need them, or if you are running locally."
    PARTITION=$(ask "  partition" "$PARTITION")
    ACCOUNT=$(ask "  account" "$ACCOUNT")
    QOS=$(ask "  qos" "$QOS")
    echo
fi

echo "Game Boy cartridges."
echo "  Puzznic, Flipull, Boxxle II, Adventures of Lolo, Amazing Tater and Super Mario Land are"
echo "  copyrighted and are not in this"
echo "  repo, so their paths have to come from you. Each one you give is benchmarked alongside"
echo "  its pure-Python twin; each one you skip is reported as skipped rather than silently"
echo "  dropped. Paths are recorded in the experiment, so cluster jobs get them too."
echo

# Environment, variable, flag and label, four fields per entry — kept as a flat list and read
# four at a time, because a label contains a space and word-splitting one string would tear
# "Super Mario Land" into three.
ROM_ENTRIES=(
    puzznic_gb        PLANIVERSE_PUZZNIC_ROM  puzznic  "Puzznic"
    flipull_gb        PLANIVERSE_FLIPULL_ROM  flipull  "Flipull"
    boxxle2_gb        PLANIVERSE_BOXXLE2_ROM  boxxle2  "Boxxle II"
    lolo_gb           PLANIVERSE_LOLO_ROM     lolo     "Adventures of Lolo"
    amazing_tater_gb  PLANIVERSE_AMAZING_TATER_ROM  amazing-tater  "Amazing Tater"
    super_mario_land_gb  PLANIVERSE_SUPER_MARIO_LAND_ROM  super-mario-land  "Super Mario Land"
)

supplied() {
    # Was this cartridge already given as a --rom-<flag> on the command line?
    local flag="$1" given
    for given in ${SUPPLIED_ROMS[@]+"${SUPPLIED_ROMS[@]}"}; do
        # --rom-mario and --rom-sml are aliases for --rom-super-mario-land; accept all three.
        if [ "$given" = "$flag" ] || { [ "$flag" = "super-mario-land" ] && { [ "$given" = "mario" ] || [ "$given" = "sml" ]; }; }; then
            return 0
        fi
    done
    return 1
}

# Spaces after commas are forgiven — "puzznic, flipull" means what it says.
ENVIRONMENTS="${ENVIRONMENTS// /}"

for ((i = 0; i < ${#ROM_ENTRIES[@]}; i += 4)); do
    environment="${ROM_ENTRIES[i]}"
    variable="${ROM_ENTRIES[i + 1]}"
    flag="${ROM_ENTRIES[i + 2]}"
    label="${ROM_ENTRIES[i + 3]}"

    # No point asking for a cartridge the experiment will not run.
    if [ -n "$ENVIRONMENTS" ] && [[ ",$ENVIRONMENTS," != *",$environment,"* ]]; then
        echo "  $label: not among the selected environments."
        continue
    fi

    if supplied "$flag"; then
        echo "  $label: given on the command line."
        continue
    fi

    rom=$(ask_rom "$environment" "$variable" "$label")
    # An `if`, not `[ -n "$rom" ] && ROM_ARGS+=(...)`: under `set -e` that compound returns
    # non-zero for every skipped cartridge and takes the whole script down with it.
    if [ -n "$rom" ]; then
        ROM_ARGS+=("--rom-$flag" "${rom#*=}")
        echo "    using ${rom#*=}"
    fi
done
if [ ${#ROM_ARGS[@]} -eq 0 ]; then
    echo "  No cartridges. The three Game Boy environments will be skipped."
fi
echo

INIT_ARGS=(init --exp-dir "$EXP_DIR" --name "$NAME"
           --time "$TIME_LIMIT" --memory "$MEMORY_LIMIT"
           --max-expansions "$MAX_EXPANSIONS" --max-instances "$MAX_INSTANCES" --force)
if [ -n "$ENVIRONMENTS" ]; then INIT_ARGS+=(--environments "$ENVIRONMENTS"); fi
if [ -n "$PARTITION" ]; then INIT_ARGS+=(--partition "$PARTITION"); fi
if [ -n "$ACCOUNT" ]; then INIT_ARGS+=(--account "$ACCOUNT"); fi
if [ -n "$QOS" ]; then INIT_ARGS+=(--qos "$QOS"); fi
if [ ${#SETUP_COMMANDS[@]} -gt 0 ]; then INIT_ARGS+=("${SETUP_COMMANDS[@]}"); fi
if [ ${#ROM_ARGS[@]} -gt 0 ]; then INIT_ARGS+=("${ROM_ARGS[@]}"); fi

echo "== writing the experiment"
"${BENCH[@]}" "${INIT_ARGS[@]}"
echo
echo "== resolving the task list"
"${BENCH[@]}" discover --exp-dir "$EXP_DIR" --sandbox-dir "$SANDBOX_DIR"
echo
echo "== generating jobs"
"${BENCH[@]}" generate --exp-dir "$EXP_DIR" --sandbox-dir "$SANDBOX_DIR" \
    --entry-point "$ENTRY_POINT" --seed "$SEED"

cat <<DONE

Ready.

  submit to SLURM     bash $SANDBOX_DIR/slurm/submit_all.sh
  or run it here      bash $SANDBOX_DIR/run_local.sh 8

then

  planiverse-bench analyze --sandbox-dir $SANDBOX_DIR
  planiverse-bench report  --sandbox-dir $SANDBOX_DIR

Edit $EXP_DIR/exp-details.json or $EXP_DIR/planners/*.json and re-run
'planiverse-bench generate' to change anything without starting over.
DONE
