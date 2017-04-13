"""
sources.random
==============

Produces synthetic data based upon simple random models.

Currently overlaps a bit with the `Sampler` classes from the `sources.sepp` module.
"""

from ..data import TimedPoints

import numpy as _np
import numpy.random as _npr
from numpy import timedelta64
    
def random_spatial(space_sampler, start_time, end_time, expected_number,
        time_rate_unit = timedelta64(1, "s")):
    """Simulate a homogeneous Poisson process in time with independent,
    identically distributed space locations.

    :param space_sampler: The callable object to return the space coordinates.
      Expects to be called as `space_sampler(N)` and returns an array of
      shape (2,N) of (x,y) coordinates.
    :param start_time: The start time of the simulation.
    :param end_time: The end time of the simulation.
    :param expected_number: The expected number of events to simulate.
    :param time_rate_unit: The :class:`numpy.timedelta64` unit to use: this
      becomes the *smallest* interval of time we can simulate.  By default,
      one second.

    :returns: A :class:`open_cp.data.TimedPoints` instance giving the
      simulation.
    """
    num_events = _npr.poisson(lam = expected_number)
    time_length = timedelta64(end_time - start_time) / time_rate_unit
    times = ( _npr.random(size = num_events) * time_length ) * time_rate_unit 
    times = _np.sort(times + _np.datetime64(start_time))
    
    coords = space_sampler(num_events)
    return TimedPoints.from_coords(times, coords[0], coords[1])

def random_uniform(region, start_time, end_time, expected_number,
        time_rate_unit = timedelta64(1, "s")):
    """Simulate a homogeneous Poisson process in time with space locations
    chosen uniformly at random in a region.

    :param region: A :class:`open_cp.data.RectangularRegion` instance giving
      the region to sample space locations in.
    :param start_time: The start time of the simulation.
    :param end_time: The end time of the simulation.
    :param expected_number: The expected number of events to simulate.
    :param time_rate_unit: The :class:`numpy.timedelta64` unit to use: this
      becomes the *smallest* interval of time we can simulate.  By default,
      one second.

    :returns: A :class:`TimedPoints` instance giving the simulation.
    """
    def uniform_sampler(size=1):
        x = _npr.random(size = size) * (region.xmax - region.xmin)
        y = _npr.random(size = size) * (region.ymax - region.ymin)
        return _np.stack([x + region.xmin, y + region.ymin])
    return random_spatial(uniform_sampler, start_time, end_time, expected_number, time_rate_unit)

def _rejection_sample_2d_single(kernel, k_max):
    while True:
        p = _npr.random(size = 2)
        if _npr.random() * k_max <= kernel(p):
            return p

def rejection_sample_2d(kernel, k_max, samples=1, oversample=2):
    """A simple two-dimensional rejection sampler.  The kernel is assumed to be
    defined on [0,1] times [0,1].

    :param kernel: A callable object giving the kernel.  Should be able to
      accept an array of shape (2, #points) and return an array of shape (#points).
    :param k_max: The maximum value the kernel takes (or an upper bound).
    :param samples: The number of samples to return.
    :param oversample: Change this to improve performance.  At each iteration,
      we test this many more samples than we need.  Make this parameter too
      large, and we "waste" random numbers.  Make it too small, and we don't
      utilise the parallel nature of numpy enough.  Defaults to 2.0

    :return: If one sample required, an array [x,y] of the point sampled.
      Otherwise an array of shape (2,N) where N is the number of samples.
    """
    
    if samples == 1:
        return _rejection_sample_2d_single(kernel, k_max)
    points = _np.empty((2,samples))
    num_samples = 0
    while num_samples < samples:
        x = _npr.random(size = samples * oversample)
        y = _npr.random(size = samples * oversample)
        k = kernel(_np.stack([x,y]))
        mask = _npr.random(size = samples * oversample) * k_max <= k
        xx, yy = x[mask], y[mask]
        if samples - num_samples < len(xx):
            xx, yy = xx[:samples - num_samples], yy[:samples - num_samples]
        points[0, num_samples:num_samples + len(xx)] = xx
        points[1, num_samples:num_samples + len(xx)] = yy
        num_samples += len(xx)
    return points

class KernelSampler():
    """A simple "sampler" class which can sample from a kernel defined on a
    rectangular region.  Call as `kernel(N)` to make N samples, returning an
    array of shape (2,N).

    See also :class:`open_cp.sources.sepp.SpaceSampler`

    :param region: A :class:`open_cp.data.RectangularRegion` instance
      describing the region the kernel is defined on.
    :param kernel: The kernel, callable with an array of shape (2,k).
    :param k_max: The maximum value the kernel takes (or an upper bound).
    """
    def __init__(self, region, kernel, k_max):
        """The kernel should be defined on all of the region"""
        self.x = region.xmin
        self.xscale = region.xmax - region.xmin
        self.y = region.ymin
        self.yscale = region.ymax - region.ymin
        def rescaled_kernel(pts):
            npts = _np.empty_like(pts)
            npts[0] = pts[0] * self.xscale + self.x
            npts[1] = pts[1] * self.yscale + self.y
            return kernel(npts)
        self.sampler = lambda num : rejection_sample_2d(rescaled_kernel, k_max, num)

    def __call__(self, size=1):
        points = self.sampler(size)
        points[0] = points[0] * self.xscale + self.x
        points[1] = points[1] * self.yscale + self.y
        return points