#!/bin/bash
# Build factory-sim and install it, for `planiverse.environments.factory`.
#
# Fetches factory-sim from https://github.com/divagr18/factory-sim (MIT; see THIRD-PARTY-NOTICES.md)
# at the commit the environment's patches were drawn on, compiles its C core into the cffi
# extension `fsim._fsim` with the system C compiler, and copies the `fsim` package into the
# running Python's site-packages. Needs git, a C compiler, the Python headers, cffi and numpy
# (`pip install cffi numpy`). Nothing of Factorio is involved: the simulator was written from
# measurements of the game and contains none of its code or data.
set -euo pipefail
commit="${FACTORY_SIM_COMMIT:-859eab23353caac5132d484c43b7e0f2061d73ad}"
work="${1:-$(mktemp -d)}"
echo "building in $work"
cd "$work"
if [ ! -d factory-sim ]; then
    git clone https://github.com/divagr18/factory-sim factory-sim
fi
cd factory-sim
git checkout --quiet "$commit"
python3 -c 'import cffi, numpy' 2>/dev/null || python3 -m pip install cffi numpy
python3 build.py
site="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
rm -rf "$site/fsim"
cp -r fsim "$site/fsim"
rm -f "$site"/fsim/_fsim.c "$site"/fsim/_fsim.o
python3 -c 'import fsim; s = fsim.Sim(water=[]); print("fsim installed:", fsim.lib.IT_COUNT, "items")'
