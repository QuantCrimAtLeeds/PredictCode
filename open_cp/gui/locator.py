"""
locator
~~~~~~~

Ah, for a Dependency Injection framework...
"""

from open_cp.gui.tk import threads
from threading import RLock
import open_cp.gui.settings as settings

def _make_pool(root=None):
    if root is None:
        raise ValueError("Must be initialised with a root window.")
    _CACHE["pool"] = threads.Pool(root)

def _make_settings():
    _CACHE["settings"] = settings.Settings()

_LOOKUP = {
    "pool": _make_pool,
    "settings" : _make_settings
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


class GuiThreadTask():
    """Mixin class which adds ability to run a task on the gui thread (which
    is a common pattern in this code base).
    """
    _lock = RLock()
    _pool = None
    
    @staticmethod
    def submit_gui_task(task):
        if GuiThreadTask._pool is None:
            with GuiThreadTask._lock:
                if GuiThreadTask._pool is None:
                    GuiThreadTask._pool = get("pool")
        
        GuiThreadTask._pool.submit_gui_task(task)
