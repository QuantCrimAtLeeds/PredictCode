from . import predictors
from . import data

import abc as _abc
import numpy as _np

class KernelRiskPredictor(predictors.ContinuousPrediction):
    def __init__(self, kernel):
        self._kernel = kernel;
    
    def risk(self, x, y):
        """Assume x and y can be 1D arrays"""
        return self._kernel(x,y)


class Weight(metaclass=_abc.ABCMeta):
    @_abc.abstractmethod
    def __call__(self, x, y):
        pass


class Quartic(Weight):
    def __init__(self, bandwidth = 200):
        self.space_bandwidth = bandwidth

    def __call__(self, x, y):
        cutoff =  self.space_bandwidth ** 2
        distance_sq = x*x + y*y
        weight = (1 - distance_sq / cutoff) ** 2
        return weight * ( distance_sq <= cutoff)

        
class RetroHotSpot(predictors.Predictor):
    def __init__(self):
        self.weight = Quartic()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if not isinstance(value, data.TimedPoints):
            raise TypeError("data should be of class TimedPoints")
        self._data = value

    def predict(self, start_time=None, end_time=None):
        mask = None
        if start_time is not None:
            mask = self.data.timestamps >= start_time
        if end_time is not None:
            end_mask = self.data.timestamps <= end_time
            mask = end_mask if (mask is None) else (mask & end_mask)
        coords = self.data.coords
        if mask is not None:
            coords = coords[:,mask]
        
        if coords.shape[1] == 0:
            def kernel(x, y):
                return 0
        else:
            def kernel(x_loc, y_loc):
                x = _np.stack([x_loc]*coords.shape[1], axis=-1)
                y = _np.stack([y_loc]*coords.shape[1], axis=-1)
                return _np.sum(self.weight(x - coords[0], y - coords[1]), axis=-1)
                #return _np.sum(self.weight(coords[0] - x_loc, coords[1] - y_loc))
        
        return KernelRiskPredictor(kernel)
