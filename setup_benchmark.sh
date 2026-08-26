#!/bin/bash
# Set up a Planiverse benchmark, interactively.
#
# Walks you through the whole preparation — limits, cartridge paths, SLURM settings — and
# then runs init, discover and generate, leaving you a directory of jobs to submit. Every
# answer has a default, so holding Enter through it gives a complete experiment.
#
#   ./setup_benchmark.sh                    # ask about everything
#   ./setup_benchmark.sh --yes              # take every default, ask nothing
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
PARTITION=""
ACCOUNT=""
SEED="0"
ASSUME_YES=0
ENTRY_POINT=""

usage() {
    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'USAGE'

Options:
  --exp-dir DIR         where to write the experiment  (default: experiment)
  --sandbox-dir DIR     where results will go          (default: sandbox)
  --name NAME           experiment name, used in job names
  --time DURATION       per-run wall clock             (default: 30m)
  --memory SIZE         per-run memory                 (default: 8GB)
  --max-expansions N    per-run node budget            (default: 100000)
  --max-instances N     instances per environment, 0 for all  (default: 0)
  --partition NAME      SLURM partition
  --account NAME        SLURM account
  --seed N              seed for the randomised planners       (default: 0)
  --entry-point CMD     how jobs invoke the CLI; defaults to planiverse-bench when it is
                        installed, otherwise python -m planiverse.benchmark.cli
  --yes, -y             accept every default without asking
  --help, -h            this
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --exp-dir) EXP_DIR="$2"; shift 2 ;;
        --sandbox-dir) SANDBOX_DIR="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --time) TIME_LIMIT="$2"; shift 2 ;;
        --memory) MEMORY_LIMIT="$2"; shift 2 ;;
        --max-expansions) MAX_EXPANSIONS="$2"; shift 2 ;;
        --max-instances) MAX_INSTANCES="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --entry-point) ENTRY_POINT="$2"; shift 2 ;;
        --yes|-y) ASSUME_YES=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ -z "$ENTRY_POINT" ]; then
    if command -v planiverse-bench > /dev/null 2>&1; then
        ENTRY_POINT="planiverse-bench"
    else
        # A source checkout that has not been pip-installed. The generated jobs need
        # something that works on the compute node, and `python -m` does.
        ENTRY_POINT="python -m planiverse.benchmark.cli"
    fi
fi
BENCH=($ENTRY_POINT)

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
    echo
    echo "SLURM. Leave blank if your site does not need them, or if you are running locally."
    PARTITION=$(ask "  partition" "$PARTITION")
    ACCOUNT=$(ask "  account" "$ACCOUNT")
    echo
fi

echo "Game Boy cartridges."
echo "  Puzznic, Flipull and Super Mario Land are copyrighted and are not in this repo, so"
echo "  their paths have to come from you. Each one you give is benchmarked alongside its"
echo "  pure-Python twin; each one you skip is reported as skipped rather than silently"
echo "  dropped. Paths are recorded in the experiment, so cluster jobs get them too."
echo

# Environment, variable and label, three fields per entry — kept as a flat list and read
# three at a time, because a label contains a space and word-splitting one string would tear
# "Super Mario Land" into three.
ROM_ENTRIES=(
    puzznic_gb        PLANIVERSE_PUZZNIC_ROM  "Puzznic"
    flipull_gb        PLANIVERSE_FLIPULL_ROM  "Flipull"
    super_mario_land  PLANIVERSE_SML_ROM      "Super Mario Land"
)

ROM_ARGS=()
for ((i = 0; i < ${#ROM_ENTRIES[@]}; i += 3)); do
    rom=$(ask_rom "${ROM_ENTRIES[i]}" "${ROM_ENTRIES[i + 1]}" "${ROM_ENTRIES[i + 2]}")
    # An `if`, not `[ -n "$rom" ] && ROM_ARGS+=(...)`: under `set -e` that compound returns
    # non-zero for every skipped cartridge and takes the whole script down with it.
    if [ -n "$rom" ]; then
        ROM_ARGS+=(--rom "$rom")
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
if [ -n "$PARTITION" ]; then INIT_ARGS+=(--partition "$PARTITION"); fi
if [ -n "$ACCOUNT" ]; then INIT_ARGS+=(--account "$ACCOUNT"); fi
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
