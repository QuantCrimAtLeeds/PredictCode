"""
locator
~~~~~~~

Ah, for a Dependency Injection framework...
"""

from open_cp.gui.tk import threads
from threading import RLock

def _make_pool(root=None):
    if root is None:
        raise ValueError("Must be initialised with a root window.")
    _CACHE["pool"] = threads.Pool(root)

_LOOKUP = {
    "pool": _make_pool
    }

_CACHE = dict()
_LOCK = RLock()

def get(object_name):
    with _LOCK:
        if object_name in _CACHE:
            return _CACHE[object_name]
        if object_name not in _LOOKUP:
            raise ValueError("Unknown object '{}' to locate".format(object_name))
        _LOOKUP[object_name]()
        return _CACHE[object_name]
