from ..data import TimedPoints  #, RectangularRegion

import numpy as _np
import numpy.random as _npr
from numpy import timedelta64
    
def random_uniform(region, start_time, end_time, expected_number,
        time_rate_unit = timedelta64(1, "s")):
    """TODO
    
    Note that `time_rate_unit` becomes the smallest gap we will observe
    in the output timestamps."""
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
    """Stuff

    kernel : Assumed signature array -> array where input array is of shape (2, #points)
    and output array is of shape (#points)
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
    def __init__(self, region, kernel, k_max):
        self.sampler = lambda num : rejection_sample_2d(kernel, k_max, num)
        self.x = region.xmin
        self.xscale = region.xmax - region.xmin
        self.y = region.ymin
        self.yscale = region.ymax - region.ymin

    def __call__(self, size=1):
        points = self.sampler(size)
        points[0] = points[0] * self.xscale + self.x
        points[1] = points[1] * self.yscale + self.y
        return points

def random_spatial(space_sampler, start_time, end_time, expected_number,
        time_rate_unit = timedelta64(1, "s")):
    num_events = _npr.poisson(lam = expected_number)
    time_length = timedelta64(end_time - start_time) / time_rate_unit
    times = ( _npr.random(size = num_events) * time_length ) * time_rate_unit 
    times = _np.sort(times + _np.datetime64(start_time))
    
    coords = space_sampler(size = num_events)
    return TimedPoints.from_coords(times, coords[0], coords[1])
