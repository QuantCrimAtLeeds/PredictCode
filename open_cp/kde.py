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
    """A "time kernel" which is constantly 1."""
    def __call__(self, x):
        x = _np.asarray(x)
        return _np.zeros_like(x) + 1


class ExponentialTimeKernel():
    """An exponentially decaying kernel, :math:`f(x) = \exp(-x/\beta)`
    where :math:`beta` is the "scale".
    """
    def __init__(self, scale):
        self._scale = scale
        
    @property
    def scale(self):
        return self._scale
    
    @scale.setter
    def scale(self, v):
        self._scale = v
    
    def __call__(self, x):
        return _np.exp( - _np.asarray(x) / self._scale ) / self._scale

    def __repr__(self):
        return "ExponentialTimeKernel(Scale={})".format(self._scale)

    @property
    def args(self):
        return "E{}".format(self._scale)


class QuadDecayTimeKernel():
    """A quadratically decaying kernel, :math:`f(x) = (1 + (x/\beta)^2)^{-1]}`
    where :math:`beta` is the "scale".
    """
    def __init__(self, scale):
        self.scale = scale
    
    @property
    def scale(self):
        return self._scale
    
    @scale.setter
    def scale(self, v):
        self._scale = v
        self._norm = 2 / (self._scale * _np.pi)

    def __call__(self, x):
        x = _np.asarray(x)
        return self._norm / (1 + (x / self._scale)**2)

    def __repr__(self):
        return "QuadDecayTimeKernel(Scale={})".format(self._scale)

    @property
    def args(self):
        return "Q{}".format(self._scale)
    

class KernelProvider():
    """Abstract base class for a "factory" which produces kernels, based
    on data.
    
    :param data: Array of coordinates in shape `(n,N)` for `n` dimensions
      and `N` data points.  Typically `n==2`.
    """
    def __call__(self, data):
        raise NotImplementedError()
        

class GaussianBaseProvider(KernelProvider):
    """Use the :class:`kernels.GaussianBase` to estimate a kernel.
    This emulates the `scipy.kde` Gaussian kernel."""
    def __call__(self, data):
        return _kernels.GaussianBase(data)

    def __repr__(self):
        return "GaussianBaseProvider"

    @property
    def args(self):
        return "G".format(self._scale)


class GaussianFixedBandwidthProvider(KernelProvider):
    """Use the :class:`kernels.GaussianBase` to estimate a kernel.
    Has a fixed bandwidth (and identity covariance matrix).
    """
    def __init__(self, bandwidth):
        self._h = bandwidth
    
    def __call__(self, data):
        ker = _kernels.GaussianBase(data)
        ker.bandwidth = self._h
        ker.covariance_matrix = _np.eye(ker.dimension)
        return ker

    def __repr__(self):
        return "GaussianFixedBandwidthProvider(bandwidth={})".format(self._h)

    @property
    def args(self):
        return "GF{}".format(self._h)


class GaussianNearestNeighbourProvider(KernelProvider):
    """Use the :class:`kernels.GaussianNearestNeighbour` to estimate
    a kernel."""
    def __init__(self, k):
        self._k = k
        
    @property
    def k(self):
        """The nearest neighbour to look at for local bandwidth estimation."""
        return self._k
    
    @k.setter
    def k(self, v):
        self._k = v

    def __call__(self, data):
        return _kernels.GaussianNearestNeighbour(data, self._k)

    def __repr__(self):
        return "GaussianNearestNeighbourProvider(k={})".format(self._k)

    @property
    def args(self):
        return "GNN{}".format(self._k)


class KDE(_predictors.DataTrainer):
    """Implements a kernel density estimation, grid based prediction.  We
    implement a hybrid approach which, while now exactly common in the
    statistics literature, seems to capture the essential features of all of
    the standard "out of the box" kernel estimators, and the "Prohotspot" type
    estimators.

    The predictor itself is simple.  We select an interval time (or all time)
    and use just the data from that time range.  The distance in time from each
    event to the end time is calculated, and optionally a "time kernel" is
    calculated: typically this kernel falls off in time, so that events in the
    past are waited less.
    
    The space locations are events are then passed to a kernel density
    estimator.  Finally (in a slightly non-standard way) the space kernel is
    weighted by the time kernel to produce a "risk surface".
    
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
        
    def _kernel(self, start_time=None, end_time=None):
        data = self.data
        if start_time is not None:
            start_time = _np.datetime64(start_time)
            data = data[data.timestamps >= start_time]
        if end_time is not None:
            end_time = _np.datetime64(end_time)
            data = data[data.timestamps < end_time]
        if end_time is None:
            end_time = data.timestamps[-1]

        kernel = self.space_kernel(data.coords)
        time_deltas = (end_time - data.timestamps) / self.time_unit
        kernel.weights = self.time_kernel(time_deltas)
        return kernel        

    def cts_predict(self, start_time=None, end_time=None):
        """Calculate a "continuous" prediction.

        :param start_time: Only use data after (and including) this time.  If
          `None` then use from the start of the data.
        :param end_time: Only use data before this time, and treat this as the
          time point to calculate the time kernel relative to.  If `None` then use
          to the end of the data, and use the final timestamp as the "end time".
          
        :return: 
        """
        kernel = self._kernel(start_time, end_time)
        return _predictors.KernelRiskPredictor(kernel)
    
    def predict(self, start_time=None, end_time=None, samples=None):
        """Calculate a grid based prediction.

        :param start_time: Only use data after (and including) this time.  If
          `None` then use from the start of the data.
        :param end_time: Only use data before this time, and treat this as the
          time point to calculate the time kernel relative to.  If `None` then use
          to the end of the data, and use the final timestamp as the "end time".
        :samples: As for :class:`ContinuousPrediction`.

        :return: An instance of :class:`GridPredictionArray`
        """
        kernel = self._kernel(start_time, end_time)
        return _predictors.grid_prediction_from_kernel(kernel, self.region,
                                self.grid, samples)
