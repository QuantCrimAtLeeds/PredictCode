"""
kde
~~~

Implements some general "kernel density estimation" methods which, while not
drawn directly from the literature, can be thought of as generalisation of
the `prohotpot` and `retrohotspot` methods.
"""

from . import predictors as _predictors

import numpy as _np
from . import kernels as _kernels

class ConstantTimeKernel():
    def __call__(self, x):
        x = _np.asarray(x)
        return _np.zeros_like(x) + 1
    

class KernelProvider():
    def __call__(self, data):
        raise NotImplementedError()
        

class GaussianBaseProvider(KernelProvider):
    def __call__(self, data):
        return _kernels.GaussianBase(data)


class KDE(_predictors.DataTrainer):
    """Implements a kernel density estimation, grid based prediction.
    
    TODO...
    
    :param region: The rectangular region to use to grid the data, or `None`
      to auto compute
    :param grid_size: The size of the grid to use
    :param grid: If not `None` that take the `region` and `grid_size` settings
      from this grid.
    """
    def __init__(self, region=None, grid_size=50, grid=None):
        if grid is None:
            self.grid = grid_size
            self.region = region
        else:
            self.region = grid.region()
            self.grid = grid.xsize
            if grid.xsize != grid.ysize:
                raise ValueError("Only supports *square* grid cells.")
        self.time_kernel = ConstantTimeKernel()
        self.time_unit = _np.timedelta64(1, "D")
        self.space_kernel = GaussianBaseProvider()

    @property
    def time_unit(self):
        """The "unit" of time to divide the time differences by to obtain
        a scalar, prior to passing to the time kernel."""
        return self._time_unit
    
    @time_unit.setter
    def time_unit(self, v):
        self._time_unit = _np.timedelta64(v)

    @property
    def time_kernel(self):
        """The weighting to apply to timestamps.  Should be a callable object
        correponds to a one-dimensional "kernel"."""
        return self._time_kernel
    
    @time_kernel.setter
    def time_kernel(self, v):
        self._time_kernel = v
        
    @property
    def space_kernel(self):
        """The kernel _estimator provider_ for the space coordinates.  Needs to
        have the interface of :class:`KernelProvider`."""
        return self._space_kernel
    
    @space_kernel.setter
    def space_kernel(self, v):
        self._space_kernel = v

    def predict(self, start_time=None, end_time=None):
        """Calculate a grid based prediction.

        :param start_time: Only use data after (and including) this time.  If
          `None` then use from the start of the data.
        :param end_time: Only use data before this time.  If `None` then use
          to the end of the data.

        :return: An instance of :class:`GridPredictionArray`
        """
        data = self.data
        if start_time is not None:
            data = data[data.timestamps >= start_time]
        if end_time is not None:
            data = data[data.timestamps < end_time]

        kernel = self.space_kernel(data.coords)
        time_deltas = (data.timestamps[-1] - data.timestamps) / self.time_unit
        kernel.weights = self.time_kernel(time_deltas)
        return _predictors.grid_prediction_from_kernel(kernel, self.region, self.grid)
