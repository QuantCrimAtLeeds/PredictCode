import inspect as _inspect
from . import predictor
from . import comparitor

# Add new prediction modules here ###########################################

from . import naive
from . import grid
from . import lonlat

# End #######################################################################

# Add new comparator modules here ###########################################

from . import pred_type

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


import sys as _sys

_fp = _Find(_sys.modules[__name__], predictor.Predictor)
_fp.optionally_remove(predictor.Predictor)
all_predictors = list(_fp.predictors)

_fp = _Find(_sys.modules[__name__], comparitor.Comparitor)
_fp.optionally_remove(comparitor.Comparitor)
all_comparitors = list(_fp.predictors)
