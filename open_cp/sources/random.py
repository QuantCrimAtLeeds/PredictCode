from ..data import TimedPoints  #, RectangularRegion

import numpy as _np
import numpy.random as _npr
from numpy import timedelta64

def uniform_random(region, start_time, end_time, expected_number,
                   time_rate_unit = timedelta64(1, "s")):
    """TODO
    
    Note that `time_rate_unit` becomes the smallest gap we will observe
    in the output timestamps."""
    num_events = _npr.poisson(lam = expected_number)
    time_length = timedelta64(end_time - start_time) / time_rate_unit
    times = ( _npr.random(size = num_events) * time_length ) * time_rate_unit 
    times = _np.sort(times + _np.datetime64(start_time))
    
    xcoords = _npr.random(size = num_events) * (region.xmax - region.xmin)
    ycoords = _npr.random(size = num_events) * (region.ymax - region.ymin)
    
    return TimedPoints.from_coords(times, xcoords + region.xmin,
                                   ycoords + region.ymin)
    
def rejection_sample_2d(kernel, k_max, samples=1, oversample=2):
    """Stuff

    kernel : Assumed signature array -> array where input array is of shape (2, #points)
    and output array is of shape (#points)
    """
    
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