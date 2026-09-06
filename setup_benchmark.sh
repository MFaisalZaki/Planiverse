#!/bin/bash
# Build a virtualenv, install planiverse into it, and generate the benchmark's jobs.
#
#   ./setup_benchmark.sh [--venv DIR] [--python BIN] [generate options...]
#   ./setup_benchmark.sh --partition gpu --qos long --parallel 100
#
# The Game Boy environments need their cartridges, which are copyrighted and cannot ship here.
# Export PLANIVERSE_PUZZNIC_ROM, PLANIVERSE_FLIPULL_ROM, PLANIVERSE_LOLO_ROM,
# PLANIVERSE_AMAZING_TATER_ROM and PLANIVERSE_SUPER_MARIO_LAND_ROM first; an environment
# without one is skipped and says so. Then: bash sandbox/submit.sh, or bash sandbox/run_local.sh 8.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO/.venv"
PYTHON="python3.13"
while [ $# -gt 0 ]; do
    case "$1" in
        --venv) VENV="$2"; shift 2 ;;
        --python) PYTHON="$2"; shift 2 ;;
        *) break ;;
    esac
done

[ -d "$VENV" ] || "$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet -e "$REPO"
# The venv's interpreter by absolute path: the generated jobs call the same one, so they need
# no activation and cannot pick up a different install off PATH.
"$VENV/bin/python" -m planiverse.benchmark generate "$@"
