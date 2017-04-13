"""
sources.sepp
============

Produces synthetic data based upon a "self-exciting" or "Hawkes model" point
process.  These are point processes where the conditional intensity function
depends upon a background intensity (i.e. a homogeneous or possibly
inhomogeneous Poisson process) and when each event in the past contributes
a further (linearly additive) terms governed by a trigger / aftershock kernel.

Such models, with specific forms for the trigger kernel, are known as
"epidemic type aftershock models" in the Earthquake modelling literature.

Rather than rely upon external libraries (excepting numpy which we do use) we
produce a number of base classes which define kernels and samplers, and provide
some common kernels and samplers for backgrounds and triggers.
"""

from .. import data
from .. import kernels
from . import random

import abc as _abc
import numpy as _np
from numpy import timedelta64
import itertools as _itertools

class SpaceTimeKernel(kernels.Kernel):
    """To produce a kernel as required by the samplers in this package,
    either extend this abstract class implementing `intensity(t, x, y)`
    or provide your own class which has the same signature as `__call__`
    and the property `kernel_max`"""
    
    @_abc.abstractmethod
    def intensity(self, t, x, y):
        """t, x and y will be one-dimensional numpy arrays of the same length.
        
        :return: A numpy array of the same length as the input"""
        pass
    
    def __call__(self, points):
        return self.intensity(points[0], points[1], points[2])
    
    def set_scale(self):
        raise NotImplementedError()
    
    @_abc.abstractmethod
    def kernel_max(self, time_start, time_end):
        """Return a value which is greater than or equal to the maximum
        intensity of the kernel over the time range (and for any space input).
        """
        pass


class PoissonTimeGaussianSpace(SpaceTimeKernel):
    """A kernel which is a constant rate Poisson process in time, and a two
    dimensional Gaussian kernel in space (see
    https://en.wikipedia.org/wiki/Multivariate_normal_distribution).

    :param time_rate: The rate of the Poisson process in time.
    :param mus: A pair of the mean values of the Gaussian in each variable.
    :param variances: A pair of the variances of the Gaussian in each variable.
    :param correlation: The correlation between the two Gaussians.
    """
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


class TimeKernel(kernels.Kernel):
    """A one dimensional kernel which can estimate its upper bound, for use
    with rejection sampling.
    """
    
    @_abc.abstractmethod
    def kernel_max(self, time_start, time_end):
        """Return a value which is greater than or equal to the maximum
        intensity of the kernel over the time range.
        """
        pass

    def set_scale(self):
        raise NotImplementedError()


class HomogeneousPoisson(TimeKernel):
    """A constant kernel, representing a homogeneous poisson process.
    
    :param rate: The rate of the process: the expected number of events per
      time unit.
    """
    def __init__(self, rate=1):
        self._rate = rate

    def __call__(self, times):
        return _np.zeros_like(times) + self._rate
    
    def kernel_max(self, time_start, time_end):
        return self._rate


class Exponential(TimeKernel):
    """An exponentially decaying kernel.

    :param exp_rate: The "rate" parameter of the exponential.
    :param total_rate: The overall scaling of the kernel.  If this kernel is
      used to simulate a point process, then this is the expected number of
      events.
    """
    def __init__(self, exp_rate=1, total_rate=1):
        self._rate = exp_rate
        self._total = total_rate

    def __call__(self, times):
        return _np.exp( -self._rate * times) * self._rate * self._total
    
    def kernel_max(self, time_start, time_end):
        return self._rate * self._total


class Sampler(metaclass=_abc.ABCMeta):
    """Sample from a point process."""
    @_abc.abstractmethod
    def sample(self, start_time, end_time):
        """Find a sample from a point process.

        :param start_time: The start of the time window to sample from.
        :param end_time: The end of the time window to sample from.

        :return: An array of shape (3,n) of space/time coordinates.
          The data should always be _sorted_ in time.
        """
        pass

    @staticmethod
    def _order_by_time(points):
        """Utility method to sort by time.

        :param points: Usual time/space array of points.

        :return: The same data, with each triple (t,x,y) preserved, but now
          ordered so that points[0] is increasing.
        """
        a = _np.argsort(points[0])
        return points[:,a]

    
class InhomogeneousPoisson(Sampler):
    """A simple rejection (aka Otago thining) sampler.

    :param region: the spatial extent of the simulation.
    :param kernel: should follow the interface of :class SpaceTimeKernel:
    """
    def __init__(self, region, kernel):
        self._region = region
        self._kernel = kernel

    def _uniform_sample_region(self, start_time, end_time, num_points):
        scale = _np.array([end_time - start_time,
                        self._region.xmax - self._region.xmin,
                        self._region.ymax - self._region.ymin])
        offset = _np.array([start_time, self._region.xmin, self._region.ymin])
        return _np.random.random((3,num_points)) * scale[:,None] + offset[:,None]

    def sample(self, start_time, end_time):
        area = (self._region.xmax - self._region.xmin) * (self._region.ymax - self._region.ymin)
        kmax = self._kernel.kernel_max(start_time, end_time)
        total_points = kmax * area * (end_time - start_time)
        num_points = _np.random.poisson(lam = total_points)
        pts = self._uniform_sample_region(start_time, end_time, num_points)
        accept_prob = _np.random.random(num_points) * kmax
        accept = (self._kernel(pts) >= accept_prob)
        return self._order_by_time(pts[:,accept])


class SpaceSampler(metaclass=_abc.ABCMeta):
    """Base class for classes which can return samples from a space (two
    dimensional) distribution.
    """
    @_abc.abstractmethod
    def __call__(self, length):
        """Return an array of shape (2,length)"""
        pass


class GaussianSpaceSampler(SpaceSampler):
    """Returns samples from a Multivariate normal distribution.

    :param mus: A pair of the mean values of the Gaussian in each variable.
    :param variances: A pair of the variances of the Gaussian in each variable.
    :param correlation: The correlation between the two Gaussians.
    """
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


class UniformRegionSampler(SpaceSampler):
    """Returns space samples chosen uniformly from a rectangular region.
    
    :param region: An instance of :class RectangularRegion: giving the region.
    """
    def __init__(self, region):
        self.region = region
        
    def __call__(self, length):
        x = _np.random.random(length) * self.region.width + self.region.xmin
        y = _np.random.random(length) * self.region.height + self.region.ymin
        return _np.vstack([x,y])


class InhomogeneousPoissonFactors(Sampler):
    """A time/space sampler where the kernel factorises into a time kernel and
    a space kernel.  For efficiency, we use a space sampler.

    :param time_kernel: Should follow the interface of :class:`TimeKernel`
    :param space_sampler: Should follow the interface of :class:`SpaceSampler`
    """
    def __init__(self, time_kernel, space_sampler):
        self._time_kernel = time_kernel
        self._space_sampler = space_sampler
    
    def sample(self, start_time, end_time):
        kmax = self._time_kernel.kernel_max(start_time, end_time)
        number_samples = _np.random.poisson(kmax * (end_time - start_time))
        times = _np.random.random(size=number_samples) * (end_time - start_time) + start_time
        accept_prob = _np.random.random(size=number_samples) * kmax
        accept = (self._time_kernel(times) >= accept_prob)
        times = times[accept]
        times.sort()
        points = self._space_sampler(len(times))
        return _np.vstack([times, points])


class HomogeneousPoissonSampler(Sampler):
    """A one-dimensional time sampler, sampling from a homogeneous Poisson
    process.

    :param rate: The rate of the process: the expected number of events per
      time unit.
    """
    def __init__(self, rate):
        self.rate = rate

    def sample(self, start_time, end_time):
        time_length = end_time - start_time
        number_points = _np.random.poisson(time_length * self.rate)
        times = _np.random.random(number_points) * time_length + start_time
        return _np.sort(times)


class ExponentialDecaySampler(Sampler):
    """A one-dimensional time sampler, sampling from an exponentially decaying
    kernel.

    :param exp_rate: The "rate" parameter of the exponential.
    :param intensity: The expected number of events.
    """
    def __init__(self, intensity, exp_rate):
        self.intensity = intensity
        self.exp_rate = exp_rate

    def sample(self, start_time, end_time):
        number_points = _np.random.poisson(self.intensity)
        unit_rate_poisson = _np.random.random(number_points)
        times = _np.log( 1 / unit_rate_poisson ) / self.exp_rate
        mask = (times >= start_time) & (times < end_time)
        return _np.sort( times[mask] )



class SelfExcitingPointProcess(Sampler):
    """Sample from a self-exciting point process model.  Can sample in
    arbitrary dimensions: if the samplers return one-dimensional points then
    we simulate a time-only process.  If the samplers return multi-dimensional
    points, then we use the first coordinate as time, and the remaining
    coordinates as space.
    
    :param background_sampler: Should follow the interface of :class:`Sampler`
    :param trigger_sampler: Should follow the interface of :class:`Sampler`
    """
    def __init__(self, background_sampler=None, trigger_sampler=None):
        self.background_sampler = background_sampler
        self.trigger_sampler = trigger_sampler

    def sample(self, start_time, end_time):
        return self.sample_with_details(start_time, end_time).points

    class Sample():
        """Contains details of the sample as returned by
        :class:`SelfExcitingPointProcess`.  This can be useful when, for example,
        checking the correctness of the simulation.

        :param points: All points from the sampled process.
        :param backgrounds: All the background events.
        :param trigger_deltas: The "deltas" between trigger and triggered (aka
          parent and child) points.
        :param trigger_points: With the same ordering as `trigger_deltas`, the
          position of the trigger (aka parent) point.
        """
        def __init__(self, points, backgrounds, trigger_deltas, trigger_points):
            self.points = points
            self.backgrounds = backgrounds
            self.trigger_deltas = trigger_deltas
            self.trigger_points = trigger_points

    def sample_with_details(self, start_time, end_time):
        """Takes a sample from the process, but returns details"""
        background_points = self.background_sampler.sample(start_time, end_time)
        to_process = [ pt for pt in background_points.T ]
        output = list(to_process)
        trigger_deltas, trigger_points = [], []
        while len(to_process) > 0:
            trigger_point = _np.asarray(to_process.pop())
            trigger_point_time = trigger_point[0] if trigger_point.shape else trigger_point
            new_points = self.trigger_sampler.sample(0, end_time - trigger_point_time)
            trigger_deltas.extend(new_points.T)
            trigger_points.extend([trigger_point] * new_points.shape[-1])
            if trigger_point.shape:
                shifted_points = new_points + trigger_point[:,None]
            else:
                shifted_points = new_points + trigger_point
            output.extend(shifted_points.T)
            to_process.extend(shifted_points.T)
        if len(output) > 0:
            if _np.asarray(output[0]).shape:
                output.sort(key = lambda triple : triple[0])
            else:
                output.sort()
        return SelfExcitingPointProcess.Sample(_np.asarray(output).T, _np.asarray(background_points),
            _np.asarray(trigger_deltas).T, _np.asarray(trigger_points).T)

def make_time_unit(length_of_time, minimal_time_unit=timedelta64(1,"ms")):
    """Utility method to create a `time_unit`.
    
    :param length_of_time: A time delta object, representing the length of time
      "one unit" should represent: e.g. an hour, a day, etc.
    :param minimal_time_unit: The minimal time length the resulting data
      represents.  Defaults to milli-seconds.
    """
    return (timedelta64(length_of_time) / minimal_time_unit) * minimal_time_unit

def scale_to_real_time(points, start_time, time_unit=timedelta64(60, "s")):
    """Transform abstract time/space data to real timestamps.

    :param points: Array of shape (3,n) representing time/space coordinates.
    :param start_time: The time to map 0.0 to
    :param time_unit: The duration of unit time, by default 60 seconds
      (so one minute, but giving the resulting data a resolution of seconds).
      See :func:`make_time_unit`.

    :return: An instance of :class:`open_cp.data.TimedPoints`
    """
    times = [_np.datetime64(start_time) + time_unit * t for t in points[0]]
    return data.TimedPoints.from_coords(times, points[1], points[2])


class GridHawkesProcess(Sampler):
    """Sample from a grid-based, Hawkes type (expoential decay self-excitation
    kernel) model, as used by Mohler et al, "Randomized Controlled Field Trials
    of Predictive Policing", 2015.
    
    :param background_rates: An array of arbitrary shape, giving the background
      rate in each "cell".
    :param theta: The overall "intensity" of trigger / aftershock events.
      Should be less than 1.
    :param omega: The rate (or inverse scale) of the exponential kernel.
      Increase to make aftershock events more localised in time.
    """
    def __init__(self, background_rates, theta, omega):
        self.mus = _np.asarray(background_rates)
        self.theta = theta
        self.omega = omega

    def _sample_one_cell(self, mu, start_time, end_time):
        background_sampler = HomogeneousPoissonSampler(rate=mu)
        trigger_sampler = ExponentialDecaySampler(intensity=self.theta, exp_rate=self.omega)
        process = SelfExcitingPointProcess(background_sampler, trigger_sampler)
        return process.sample(start_time, end_time)

    def sample(self, start_time, end_time):
        """Will return an array of the same shape as that used by the
        background event, each entry of which is an array of zero or
        more times of events.
        """
        out = _np.empty_like(self.mus, dtype=_np.object)        
        for index in _itertools.product(*[list(range(i)) for i in self.mus.shape]):
            out[index] = self._sample_one_cell(self.mus[index], start_time, end_time)
        return out
    
    def sample_to_randomised_grid(self, start_time, end_time, grid_size):
        """Asuming that the background rate is a two-dimensional array,
        generate (uniformly at random) event locations so when confinded to
        a grid, the time-stamps agree with simulated data for that grid cell.
        We treat the input background rate as a matrix, so it has entries
        [row, col] or [y, x].
        
        :return: An array of shape (3,N) of N sampled points
        """
        cells = self.sample(start_time, end_time)
        points = []
        for row in range(cells.shape[0]):
            for col in range(cells.shape[1]):
                times = cells[row, col]
                if len(times) == 0:
                    continue
                xcs = _np.random.random(len(times)) + col
                ycs = _np.random.random(len(times)) + row
                for t,x,y in zip(times, xcs, ycs):
                    points.append((t, x * grid_size, y * grid_size))
        points.sort(key = lambda triple : triple[0])
        return _np.asarray(points).T