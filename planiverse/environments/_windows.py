"""What Windows needs before a simulator's extension module can load.

libsumo's Windows wheel does not import on its own: `import libsumo` fails with "DLL load failed
while importing _libsumo: The specified procedure could not be found". The extension depends on
the libraries in the eclipse-sumo wheel's `sumo/bin`, and libsumo adds that folder to the DLL
search path, but Windows searches the application and system directories first, so a library
of the same name there (an older runtime or a tool's copy) is taken instead and lacks a
procedure SUMO's build expects. Loading the wheel's own libraries by full path first settles
the question: a library already loaded under that name is reused rather than searched for.

`planiverse.environments` calls `preload_sumo_libraries` on import, on Windows only. It is a
no-op where eclipse-sumo is not installed, and costs a fraction of a second where it is.
"""
import ctypes
import os
import sys


def preload_sumo_libraries():
    """Load the eclipse-sumo wheel's libraries by path, so libsumo's extension finds them."""
    if sys.platform != "win32":
        return
    try:
        import sumo
    except ImportError:
        return
    folder = os.path.join(sumo.SUMO_HOME, "bin")
    if not os.path.isdir(folder):
        return
    os.add_dll_directory(folder)
    for name in sorted(os.listdir(folder)):
        # The OpenSceneGraph plugins are for the GUI, and some of them want libraries the wheel
        # does not carry.
        if not name.endswith(".dll") or name.startswith("osgdb_"):
            continue
        try:
            ctypes.WinDLL(os.path.join(folder, name))
        except OSError:
            pass
