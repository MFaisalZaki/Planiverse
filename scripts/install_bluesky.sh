#!/bin/bash
# Install BlueSky and what the airspace environment needs of it, for `planiverse.environments.operational.airspace`.
#
# BlueSky (https://github.com/TUDelft-CNS-ATM/bluesky, MIT) declares a dependency on `zmq`, a
# placeholder package that no longer builds; the binding it wants is `pyzmq`. So the simulator is
# installed without its dependency list and the real ones beside it: pyzmq, msgpack, the OpenAP
# performance model (LGPL-3.0) and BlueSky's navigation data (GPL-3.0). See THIRD-PARTY-NOTICES.md.
set -euo pipefail
python3 -m pip install pyzmq msgpack openap bluesky-navdata numpy scipy pandas matplotlib
python3 -m pip install --no-deps "bluesky-simulator>=1.1"
python3 -c 'import bluesky; bluesky.init(mode="sim"); print("bluesky installed:", bluesky.__file__)'
