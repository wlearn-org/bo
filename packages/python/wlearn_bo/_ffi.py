"""
ctypes bindings to libbo -- Bayesian optimization C11 core.

Library is loaded lazily on first call to get_lib().
"""

import ctypes
import ctypes.util
import os
import platform
import sys

_lib = None
_ptr = ctypes.c_void_p


def _find_lib():
    """Find libbo shared library. Returns path or None."""
    # 1. BO_LIB_PATH env var (explicit override)
    env_path = os.environ.get('BO_LIB_PATH')
    if env_path and os.path.isfile(env_path):
        return env_path

    # 2. Installed _native extension module (pip install)
    import importlib.machinery
    this_dir = os.path.dirname(os.path.abspath(__file__))
    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    for suffix in suffixes:
        candidate = os.path.join(this_dir, '_native' + suffix)
        if os.path.isfile(candidate):
            return candidate

    # 3. Development layout: packages/python/../../build/libbo.so
    dev_build = os.path.normpath(
        os.path.join(this_dir, '..', '..', '..', 'build', 'libbo.so')
    )
    if os.path.isfile(dev_build):
        return dev_build

    # 4. System library search
    name = 'bo'
    if platform.system() == 'Darwin':
        name = 'bo'  # ctypes.util handles .dylib
    found = ctypes.util.find_library(name)
    if found:
        return found

    return None


def _load_lib():
    """Load and configure ctypes bindings."""
    path = _find_lib()
    if path is None:
        raise RuntimeError(
            'Cannot find libbo shared library. Options:\n'
            '  1. Set BO_LIB_PATH=/path/to/libbo.so\n'
            '  2. pip install wlearn-bo (builds from source)\n'
            '  3. cd bo && mkdir build && cd build && cmake .. && make'
        )
    lib = ctypes.CDLL(path)
    _declare_signatures(lib)
    return lib


def _declare_signatures(lib):
    """Declare argtypes and restype for all wl_bo_* functions."""
    d = ctypes.c_double
    i = ctypes.c_int
    dp = ctypes.POINTER(ctypes.c_double)

    # Error
    lib.wl_bo_get_last_error.argtypes = []
    lib.wl_bo_get_last_error.restype = ctypes.c_char_p

    # Space construction
    lib.wl_bo_space_create.argtypes = [i]
    lib.wl_bo_space_create.restype = _ptr

    lib.wl_bo_space_add_continuous.argtypes = [_ptr, i, d, d, i]
    lib.wl_bo_space_add_continuous.restype = i

    lib.wl_bo_space_add_integer.argtypes = [_ptr, i, d, d, i]
    lib.wl_bo_space_add_integer.restype = i

    lib.wl_bo_space_add_categorical.argtypes = [_ptr, i, i]
    lib.wl_bo_space_add_categorical.restype = i

    lib.wl_bo_space_add_condition.argtypes = [_ptr, i, i, i]
    lib.wl_bo_space_add_condition.restype = i

    lib.wl_bo_space_finalize.argtypes = [_ptr]
    lib.wl_bo_space_finalize.restype = None

    lib.wl_bo_space_free.argtypes = [_ptr]
    lib.wl_bo_space_free.restype = None

    # Optimizer
    lib.wl_bo_create.argtypes = [_ptr, i, i, d, d, i, i, i, i, i]
    lib.wl_bo_create.restype = _ptr

    lib.wl_bo_observe.argtypes = [_ptr, dp, d]
    lib.wl_bo_observe.restype = i

    lib.wl_bo_suggest.argtypes = [_ptr, dp]
    lib.wl_bo_suggest.restype = i

    lib.wl_bo_suggest_liar.argtypes = [_ptr, dp, d]
    lib.wl_bo_suggest_liar.restype = i

    # Queries
    lib.wl_bo_get_n_obs.argtypes = [_ptr]
    lib.wl_bo_get_n_obs.restype = i

    lib.wl_bo_get_best_score.argtypes = [_ptr]
    lib.wl_bo_get_best_score.restype = d

    lib.wl_bo_get_n_contexts.argtypes = [_ptr]
    lib.wl_bo_get_n_contexts.restype = i

    lib.wl_bo_free.argtypes = [_ptr]
    lib.wl_bo_free.restype = None


def get_lib():
    """Return the loaded ctypes library (lazy-init)."""
    global _lib
    if _lib is None:
        _lib = _load_lib()
    return _lib
