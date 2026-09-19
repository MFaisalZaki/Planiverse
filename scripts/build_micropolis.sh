#!/bin/bash
# Build the Micropolis engine's Python binding and install it, for `planiverse.environments.micropolis`.
#
# Fetches MicropolisCore from https://github.com/SimHacker/micropolis (GPL-3.0 with Electronic
# Arts' additional terms; see THIRD-PARTY-NOTICES.md), patches two Python 2 names in its SWIG
# callback hook, makes the engine's `seedRandom` public, builds the extension with swig and a C++
# compiler, and copies the module into
# the running Python's site-packages. Needs git, a C++ compiler, the Python headers and swig
# (`pip install swig` provides one).
set -euo pipefail
work="${1:-$(mktemp -d)}"
echo "building in $work"
cd "$work"
if [ ! -d micropolis ]; then
    git clone --depth 1 --filter=blob:none --sparse https://github.com/SimHacker/micropolis micropolis
    (cd micropolis && git sparse-checkout set MicropolisCore)
fi
cd micropolis/MicropolisCore/src/MicropolisEngine
sed -i 's/PyString_FromString/PyUnicode_FromString/g; s/PyInt_FromLong/PyLong_FromLong/g' swig/micropolisengine-swig-python.i
# The engine reseeds its random numbers from the clock after generating a map; the environment
# reseeds them from the instance's seed instead, which needs `seedRandom` reachable from Python.
sed -i 's/^    void seedRandom(int seed);/public:\n    void seedRandom(int seed);\nprivate:/' src/micropolis.h
mkdir -p objs
swig -python -c++ -Isrc -Wall -outdir objs -o objs/micropolisengine_wrap.cpp swig/micropolisengine.i
python3 setup.py build --build-base=objs
site="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
cp objs/micropolisengine.py "$site/"
cp objs/lib.*/_micropolisengine*.so "$site/"
python3 -c 'import micropolisengine; e = micropolisengine.Micropolis(); e.initGame(); print("micropolisengine installed")'
