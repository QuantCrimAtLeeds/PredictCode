import inspect as _inspect
import sys as _sys
import logging as _logging
import logging.handlers as _handlers

from . import predictor
from . import comparitor

# Add new prediction modules here ###########################################

from . import naive
from . import grid
from . import lonlat
from . import retro
from . import prohotspot
from . import prohotspotcts
from . import stscan
from . import kde
from . import sepp
from . import sepp2

# End #######################################################################

# Add new comparator modules here ###########################################

from . import pred_type
from . import geo_clip
from . import hit_rate

# End #######################################################################


class _Find():
    def __init__(self, start, base_class=predictor.Predictor):
        self._predictors = set()
        self._checked = set()
        self._max_depth = 2
        self._base_class = base_class

        self._scan_module(start)
        delattr(self, "_checked")
        
    @property
    def predictors(self):
        """Set of classes which (properly) extend predictor.Predictor"""
        return self._predictors

    def optionally_remove(self, clazz):
        if clazz in self._predictors:
            self._predictors.remove(clazz)
        
    def _scan_module(self, mod, depth=0):
        if depth >= self._max_depth:
            return
        if mod in self._checked:
            return
        self._checked.add(mod)
        for name, value in _inspect.getmembers(mod):
            if not name.startswith("_"):
                if _inspect.isclass(value):
                    self._scan_class(value)
                elif _inspect.ismodule(value):
                    self._scan_module(value, depth + 1)
                    
    def _scan_class(self, cla):
        if self._base_class in _inspect.getmro(cla):
            self._predictors.add(cla)


_fp = _Find(_sys.modules[__name__], predictor.Predictor)
_fp.optionally_remove(predictor.Predictor)
all_predictors = list(_fp.predictors)

_fp = _Find(_sys.modules[__name__], comparitor.Comparitor)
_fp.optionally_remove(comparitor.Comparitor)
all_comparitors = list(_fp.predictors)


_LOGGER_NAME = "__interactive_warning_logger__"
_stdout_handler = None
_current_handler = None

def _get_stdout_handler():
    global _stdout_handler
    if _stdout_handler is None:
        _stdout_handler = _logging.StreamHandler(_sys.__stdout__)
        fmt = _logging.Formatter("{asctime} {levelname} {name} - {message}", style="{")
        _stdout_handler.setFormatter(fmt)
    return _stdout_handler

def _set_handler(handler):
    logger = get_logger()
    global _current_handler
    if _current_handler is not None:
        logger.removeHandler(_current_handler)
    _current_handler = handler
    logger.addHandler(handler)

def set_edit_logging():
    """Only log errors, in the usual style."""
    get_logger().setLevel(_logging.ERROR)
    _set_handler(_get_stdout_handler())

def set_queue_logging(queue):
    """Log all to the passed queue."""
    get_logger().setLevel(_logging.DEBUG)
    handler = _handlers.QueueHandler(queue)
    _set_handler(handler)

def get_logger():
    """Get the message logger"""
    return _logging.getLogger(_LOGGER_NAME)

set_edit_logging()