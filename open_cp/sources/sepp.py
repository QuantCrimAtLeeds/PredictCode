from .. import data
from . import random

import abc as _abc
import numpy as _np
from numpy import timedelta64

## TODO: I want to standise the notion of a "kernel" across the project
# I think it should just be (using duck typing) anything which can be
# called with an array of shape (k,n) where k is the dimension of space,
# and n is the number of samples.  It should return an array of length n.
# Should be normalised or not.
#
# But maybe we do want to keep the ABC which also have a property returning
# the maximimum of kernel is a time region?

class SpaceTimeKernel(metaclass=_abc.ABCMeta):
    """To produce a kernel as required by the samplers in this package,
    either extend this abstract class implementing `intensity(t, x, y)`
    or provide your own class which has the same signature as `__call__`
    and the property `kernel_max`"""
    
    @_abc.abstractmethod
    def intensity(self, t, x, y):
        """t, x and y will be one-dimensional numpy arrays of the same length.
        
        Return should be a numpy array of the same length as the input"""
        pass
    
    def __call__(self, points):
        return self.intensity(points[0], points[1], points[2])
    
    @_abc.abstractmethod
    def kernel_max(self, time_start, time_end):
        pass


class Sampler(metaclass=_abc.ABCMeta):
    @_abc.abstractmethod
    def sample(self, start_time, end_time):
        pass

    
class PoissonTimeGaussianSpace(SpaceTimeKernel):
    def __init__(self, time_rate, mus, variances, correlation):
        self.time_rate = time_rate
        self.mus = mus
        self.variances = variances
        self.correlation = correlation
    
    def _normalisation(self):
        c = (1 - self.correlation**2)
        return 1.0 / (2 * _np.pi * _np.sqrt(self.variances[0] * self.variances[1] * c) )
    
    def intensity(self, t, x, y):
        xf = (x - self.mus[0]) ** 2 / self.variances[0]
        yf = (y - self.mus[1]) ** 2 / self.variances[1]
        jf = ( 2 * self.correlation * (x - self.mus[0]) * (y - self.mus[1])
            / _np.sqrt(self.variances[0] * self.variances[1]) )
        c = (1 - self.correlation**2)
        k = _np.exp( - (xf + yf - jf) / (2 * c) )
        return self.time_rate * k * self._normalisation()
    
    def kernel_max(self, time_start, time_end):
        return self._normalisation() * self.time_rate


class InhomogeneousPoisson(Sampler):
    """A simple rejection (aka Otago thining) sampler.  You need to supply
    an upper bound on the kernel.  If you have a special kernel, then a custom
    made sampler is likely to be faster."""
    
    def __init__(self, region, kernel):
        """region is the spatial extent of the simulation"""
        self._region = region
        if not isinstance(kernel, SpaceTimeKernel):
            raise ValueError("kernel should be of type SpaceTimeKernel")
        self._kernel = kernel

    def _uniform_sample_region(self, start_time, end_time, num_points):
        pts = _np.random.random((3,num_points))
        pts *= _np.array([end_time - start_time, self._region.xmax - self._region.xmin,
            self._region.ymax - self._region.ymin])[:,None]
        pts += _np.array([start_time, self._region.xmin, self._region.ymin])[:,None]
        return pts

    def sample(self, start_time, end_time):
        area = (self._region.xmax - self._region.xmin) * (self._region.ymax - self._region.ymin)
        kmax = self._kernel.kernel_max(start_time, end_time)
        total_points = kmax * area * (end_time - start_time)
        num_points = _np.random.poisson(lam = total_points)
        pts = self._uniform_sample_region(start_time, end_time, num_points)
        accept_prob = _np.random.random(num_points) * kmax
        accept = (self._kernel(pts) >= accept_prob)
        return pts[:,accept]


class TimeKernel(metaclass=_abc.ABCMeta):
    @_abc.abstractmethod
    def __call__(self, times):
        """times is a one-dimensional numpy array; return is of the same type"""
        pass
    
    @_abc.abstractmethod
    def kernel_max(self, time_start, time_end):
        pass


class HomogeneousPoisson(TimeKernel):
    def __init__(self, rate=1):
        self._rate = rate

    def __call__(self, times):
        return _np.zeros_like(times) + self._rate
    
    def kernel_max(self, time_start, time_end):
        return self._rate


class Exponential(TimeKernel):
    def __init__(self, exp_rate=1, total_rate=1):
        self._rate = exp_rate
        self._total = total_rate

    def __call__(self, times):
        return _np.exp( -self._rate * times) * self._rate * self._total
    
    def kernel_max(self, time_start, time_end):
        return self._rate * self._total


class SpaceSampler(metaclass=_abc.ABCMeta):
    @_abc.abstractmethod
    def __call__(self, length):
        """Return an array of shape (2,length)"""
        pass


class GaussianSpaceSampler(SpaceSampler):
    def __init__(self, mus, variances, correlation):
        self.mus = mus
        self.stds = _np.sqrt(_np.array(variances))
        self.correlation = correlation
    
    def __call__(self, length):
        xy = _np.random.standard_normal(size = length * 2).reshape((2,length))
        theta = _np.arcsin(self.correlation) / 2
        sin, cos = _np.sin(theta), _np.cos(theta)
        x = xy[0] * sin + xy[1] * cos
        y = xy[0] * cos + xy[1] * sin
        x = x * self.stds[0] + self.mus[0]
        y = y * self.stds[1] + self.mus[1]
        return _np.vstack([x,y])


class InhomogeneousPoissonFactors(Sampler):
    def __init__(self, time_kernel, space_sampler):
        if not isinstance(time_kernel, TimeKernel):
            raise ValueError("time_kernel should be of type TimeKernel")
        self._time_kernel = time_kernel
        self._space_sampler = space_sampler
    
    def sample(self, start_time, end_time):
        kmax = self._time_kernel.kernel_max(start_time, end_time)
        number_samples = _np.random.poisson(lam=kmax * (end_time - start_time))
        times = _np.random.random(size=number_samples) * (end_time - start_time) + start_time
        accept_prob = _np.random.random(size=number_samples) * kmax
        accept = (self._time_kernel(times) >= accept_prob)
        times = times[accept]
        points = self._space_sampler(len(times))
        return _np.vstack([times, points])


class SelfExcitingPointProcess(Sampler):
    def __init__(self, region=None, background_sampler=None, trigger_sampler=None):
        """region is the spatial extent of the simulation"""
        self.region = region
        self.background_sampler = background_sampler
        self.trigger_sampler = trigger_sampler
        
    def sample(self, start_time, end_time):
        output = []
        background_points = self.background_sampler.sample(start_time, end_time)
        to_process = [ pt for pt in background_points.T ]
        output.extend(to_process)
        while len(to_process) > 0:
            trigger_point = to_process.pop()
            time, x, y = trigger_point
            new_points = self.trigger_sampler.sample(0, end_time - time)
            new_points += trigger_point[:,None]
            output.extend(new_points.T)
            to_process.extend(new_points.T)
        output.sort(key = lambda triple : triple[0])
        return _np.array(output).T

def scale_to_real_time(points, start_time, time_unit=timedelta64(1, "m")):
    times = [_np.datetime64(start_time) + time_unit * t for t in points[0]]
    return data.TimedPoints.from_coords(times, points[1], points[2])