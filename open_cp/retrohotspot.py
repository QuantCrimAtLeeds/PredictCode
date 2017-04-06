"""
retrohotspot
~~~~~~~~~~~~

This is a traditional hotspotting technique.  A window of past data (values
around two months seem to be common) is used; the timestamps of the data are
then ignored.  Around each point we lay down a kernel: typically this is
localised in space, e.g. a "quartic" kernel with a certain bandwidth.  These
are then summed to arrive at an overall relative risk.

Traditionally, a grid-based risk is produced, instead of a continuous kernel.
(It seems likely this is due to limitations of historic technology, and not due
to any belief in intrinsic superiority of this method).  A grid is laid down,
and in computing the weight assigned to each grid cell, the distance from the
mid-point of that cell to each event is used.

To provide your work kernel / weight, subclass the abstract base class Weight.
"""

from . import predictors
from . import data

import abc as _abc
import numpy as _np

class Weight(metaclass=_abc.ABCMeta):
    """Base class for kernels / weights for the retrospective hotspotting
    algorithm.
    """
    @_abc.abstractmethod
    def __call__(self, x, y):
        """Evaluate the weight.  Should always return a non-negative number
        for any input.  If the input is out of the support of the kernel,
        return 0.

        :param x: A scalar and one-dimensional array of x coordinates.
        :param y: A scalar and one-dimensional array of y coordinates.

        :return: A scalar or one-dimensional array as appropriate.
        """
        pass


class Quartic(Weight):
    """The classic "quartic" weight, which is the function :math (1-d^2)^2:
    for :math |d| \leq 1:.  In general, we compute the distance from the
    origin and then divide by a bandwidth to create the variable :math d:.

    :param bandwidth: The maximum extend of the kernel.
    """
    def __init__(self, bandwidth = 200):
        self._cutoff = bandwidth ** 2

    def __call__(self, x, y):
        distance_sq = x*x + y*y
        weight = (1 - distance_sq / self._cutoff) ** 2
        return weight * ( distance_sq <= self._cutoff)


def _clip_data(data, start_time, end_time):
    mask = None
    if start_time is not None:
        mask = data.timestamps >= start_time
    if end_time is not None:
        end_mask = data.timestamps <= end_time
        mask = end_mask if (mask is None) else (mask & end_mask)
    return data.coords if mask is None else data.coords[:,mask]

        
class RetroHotSpot(predictors.DataTrainer):
    """Implements the retro-spective hotspotting algorithm.  To change the
    weight/kernel used, set the :attribute weight: attribute.
    """
    def __init__(self):
        self.weight = Quartic()

    def predict(self, start_time=None, end_time=None):
        """Produce a continuous risk prediction over the optional time range.

        :param start_time: If given, only use the data with a timestamp after
        this time.
        :param end_time: If given, only use the data with a timestamp before
        this time.
        """
        coords = _clip_data(self.data, start_time, end_time)
        if coords.shape[1] == 0:
            def kernel(points):
                return 0
        else:
            def kernel(points):
                x, y = points[0], points[1]
                xc, yc = coords[0], coords[1]
                return _np.sum(self.weight(x[:,None] - xc[None,:], y[:,None] - yc[None,:]), axis=1)
        
        return predictors.KernelRiskPredictor(kernel)


class RetroHotSpotGrid(predictors.DataTrainer):
    """Applies the grid-based retro-spective hotspotting algorithm.
    To change the weight/kernel used, set the :attribute weight: attribute.

    This applies a grid at the start of the algorithm, and so differs from
    using :class RetroHotSpot: and then gridding the resulting continuous risk 
    estimate.

    :param region: The rectangular region the grid should cover.
    :param grid_size: The size of grid to use.
    """
    def __init__(self, region, grid_size=150):
        self.grid_size = grid_size
        self.region = region
        self.weight = Quartic()

    def predict(self, start_time=None, end_time=None):
        coords = _clip_data(self.data, start_time, end_time)
        xsize, ysize = self.region.grid_size(self.grid_size)
        matrix = _np.empty((ysize, xsize))
        for gridx in range(xsize):
            x = gridx * self.grid_size + self.region.xmin + self.grid_size / 2
            for gridy in range(ysize):
                y = gridy * self.grid_size + self.region.ymin + self.grid_size / 2
                matrix[gridy][gridx] = _np.sum(self.weight(
                        x - self.data.xcoords, y - self.data.ycoords))
        return predictors.GridPredictionArray(self.grid_size, self.grid_size,
            matrix, self.region.xmin, self.region.ymin)