"""
prohotspot
~~~~~~~~~~

Implements the "prospective hotspotting" technique from:

1. Bowers, Johnson, Pease,
   "Prospective hot-spotting: The future of crime mapping?",
   Brit. J. Criminol. (2004) 44 641--658.  doi:10.1093/bjc/azh036
2. Johnson et al.,
   "Prospective crime mapping in operational context",
   Home Office Online Report 19/07
   `Police online library <http://library.college.police.uk/docs/hordsolr/rdsolr1907.pdf>`_

The underlying idea is to start with a kernel / weight defined in space and
positive time.  This typically has finite extent, and might be related to
discretised space and/or time.  Weights used in the literature tend to be
of the form :math:`1/(1+d)`.

The classical algorithm assigns all events to cells in a gridding of space,
and a "grid" of time (typically the number of whole weeks before the current
time).  Only events which are close enough in space and time to the grid cell
of interest are used.  For these, the weight is evaluated on each one, and then
the sum taken.

It is important to note the coupling between the grid size used and the weight,
because it is the distance between grid cells which is used.  Exactly what
"distance" here means is unclear, and we have provided a number of options.

Alternatively, we can just use the weight / kernel in a continuous kernel
density estimate scheme.
"""

from . import predictors as _predictors

import abc as _abc
import numpy as _np

class Weight(metaclass=_abc.ABCMeta):
    """Base class for weights / kernels.  Classes implementing this algorithm
    are responsible purely for providing weights.  We leave the details of
    possibly discretising data to other classes.
    """

    @_abc.abstractmethod
    def __call__(self, dt, dd):
        """Evaluate the weight given the potentially discretised input.

        :param dt: The time distance from 0.  May be a scalar or a numpy array;
          should be of a number type, not `timedelta` or similar.
        :param dd: Spatial distance.  May be a scalar or a one-dimensional
          numpy array.

        :return: A scalar or one-dimensional numpy array as appropriate.
        """
        pass


class ClassicWeight(Weight):
    """The classical weight, :math:`(1/(1+d))(1/(1+t))` where :math:`d` is
    distance and :math:`t` is time.  Default units are "grid cells" and "weeks",
    respectively.

    :param space_bandwidth: Distances greater than or equal to this set the
      weight to 0.
    :param time_bandwidth: Times greater than or equal to this set the weight
      to 0.
    """
    def __init__(self, space_bandwidth=8, time_bandwidth=8):
        self.space_bandwidth = space_bandwidth
        self.time_bandwidth = time_bandwidth

    def __call__(self, dt, dd):
        mask = (dt < self.time_bandwidth) & (dd < self.space_bandwidth)
        return 1 / ( (1 + dd) * ( 1 + dt) ) * mask

    def __repr__(self):
        return "Classic(sb={}, tb={})".format(self.space_bandwidth, self.time_bandwidth)

    @property
    def args(self):
        return "C{},{}".format(self.space_bandwidth, self.time_bandwidth)


class GridDistance(metaclass=_abc.ABCMeta):
    """Abstract base class to calculate the distance between grid cells"""
    @_abc.abstractmethod
    def __call__(self, x1, y1, x2, y2):
        pass


class DistanceDiagonalsSame(GridDistance):
    """Distance in the grid.  Diagonal distances are one, so (1,1) and
    (2,2) are adjacent points.  This equates to using an :math:`\ell^\infty`
    norm.
    """
    def __call__(self, x1, y1, x2, y2):
        xx = _np.abs(x1 - x2)
        yy = _np.abs(y1 - y2)
        return _np.max(_np.vstack((xx, yy)), axis=0)
    
    def __repr__(self):
        return "DiagsSame"


class DistanceDiagonalsDifferent(GridDistance):
    """Distance in the grid.  Now diagonal distances are two, so (1,1) and
    (2,2) are two grid cells apart.  This equates to using an :math:`\ell^1`
    norm.
    """
    def __call__(self, x1, y1, x2, y2):
        return _np.abs(x1 - x2) + _np.abs(y1 - y2)

    def __repr__(self):
        return "DiagsDiff"


class DistanceCircle(GridDistance):
    """Distance in the grid using the usual Euclidean distance, i.e. the
    :math:`\ell^2` norm.  This will work better with the continuous version
    of the predictor.
    """
    def __call__(self, x1, y1, x2, y2):
        return _np.sqrt((x1-x2)**2 + (y1-y2)**2)

    def __repr__(self):
        return "DiagsCircle"


class ProspectiveHotSpot(_predictors.DataTrainer):
    """Implements the classical, grid based algorithm.  To calculate distances,
    we consider the grid cell we are computing the risk intensity for, the grid
    cell the event falls into, and then delegate to an instance of :class
    GridDistance: to compute the distance.  To compute time, we look at the
    time difference between the prediction time and the timestamp of the event
    and then divide by the :attr:`time_unit`, then round down to the
    nearest whole number.  So 6 days divided by 1 week is 0 whole units.

    Set :attr:`distance` to change the computation of distance between
    grid cells.  Set :attr:`weight` to change the weight used.

    :param region: The :class:`RectangularRegion` the data is in.
    :param grid_size: The size of the grid to place the data into.
    :param grid: Alternative to specifying the region and grid_size is to pass
      a :class:`BoundedGrid` instance.
    :param time_unit: A :class:`numpy.timedelta64` instance giving the time
      unit.
    """
    def __init__(self, region=None, grid_size=50, time_unit=_np.timedelta64(1, "W"), grid=None):
        if grid is None:
            self.grid = grid_size
            self.region = region
        else:
            self.region = grid.region()
            self.grid = grid.xsize
            if grid.xsize != grid.ysize:
                raise ValueError("Only supports *square* grid cells.")
        self.time_unit = time_unit
        self.weight = ClassicWeight()
        self.distance = DistanceDiagonalsSame()

    def _cell(self, x, y):
        gridx = _np.floor((x - self.region.xmin) / self.grid)
        gridy = _np.floor((y - self.region.ymin) / self.grid)
        return gridx, gridy

    def _total_weight(self, time_deltas, coords, cellx, celly):
        gridx, gridy = self._cell(coords[0], coords[1])
        distances = self.distance(gridx, gridy, cellx, celly)
        return _np.sum(self.weight(time_deltas, distances))

    def predict(self, cutoff_time, predict_time):
        """Calculate a grid based prediction.

        :param cutoff_time: Ignore data with a timestamp after this time.
        :param predict_time: Timestamp of the prediction.  Used to calculate
          the time difference between events and "now".  Typically the same as
          `cutoff_time`.

        :return: An instance of :class:`GridPredictionArray`
        """
        if not cutoff_time <= predict_time:
            raise ValueError("Data cutoff point should be before prediction time")
        events = self.data.events_before(cutoff_time)
        time_deltas = _np.datetime64(predict_time) - events.timestamps
        time_deltas = _np.floor(time_deltas / self.time_unit)

        width = int(_np.rint((self.region.xmax - self.region.xmin) / self.grid))
        height = int(_np.rint((self.region.ymax - self.region.ymin) / self.grid))
        matrix = _np.empty((height, width))
        for x in range(width):
            for y in range(height):
                matrix[y][x] = self._total_weight(time_deltas, events.coords, x, y)
        return _predictors.GridPredictionArray(self.grid, self.grid, matrix,
                                              self.region.xmin, self.region.ymin)


class ProspectiveHotSpotContinuous(_predictors.DataTrainer):
    """Implements the prospective hotspot algorithm as a kernel density
    estimation.  A copy of the space/time kernel / weight is laid down over
    each event and the result is summed.  To allow compatibility with the grid
    based method, we set a time unit and a grid size, but these are purely used
    to scale the data appropriately.
    """
    def __init__(self, grid_size=50, time_unit=_np.timedelta64(1, "W")):
        self.grid = grid_size
        self.time_unit = time_unit
        self.weight = ClassicWeight()

    def predict(self, cutoff_time, predict_time):
        """Calculate a continuous prediction.

        :param cutoff_time: Ignore data with a timestamp after this time.
        :param predict_time: Timestamp of the prediction.  Used to calculate
          the time difference between events and "now".  Typically the same as
          `cutoff_time`.

        :return: An instance of :class:`ContinuousPrediction`
        """
        if not cutoff_time <= predict_time:
            raise ValueError("Data cutoff point should be before prediction time")
        events = self.data.events_before(cutoff_time)
        time_deltas = (_np.datetime64(predict_time) - events.timestamps) / self.time_unit

        def kernel(points):
            points = _np.asarray(points)
            xdeltas = (points[0][:,None] - events.coords[0][None,:]) / self.grid
            ydeltas = (points[1][:,None] - events.coords[1][None,:]) / self.grid
            distances = _np.sqrt(xdeltas**2 + ydeltas**2)
            times = time_deltas[None,:]
            r = _np.sum(self.weight(times, distances), axis=-1)
            # Return a scalar if input as scalar
            return r[0] if len(r)==1 else r

        return _predictors.KernelRiskPredictor(kernel, cell_width=self.grid,
                cell_height=self.grid)
        
    def grid_predict(self, cutoff_time, start, end, grid, samples=None):
        """Directly calculate a grid prediction, by taking the mean value over
        both time and space.  We also normalise the resulting grid prediction.
        (But be aware that if you subsequently "mask" the grid, you will then
        need to re-normalise).

        :param cutoff_time: Ignore data with a timestamp after this time.
        :param start: The start of the prediction time window.  Typically the
          same as `cutoff_time`.
        :param end: The end of the prediction window.  We will average the
          kernel between `start` and `end`.
        :param grid: An instance of :class:`data.BoundedGrid` to use as a basis
          for the prediction.
        :param samples: Number of samples to use, or `None` for auto-compute

        :return: An instance of :class:`GridPredictionArray`.
        """
        if not cutoff_time <= start:
            raise ValueError("Data cutoff point should be before prediction time")
        events = self.data.events_before(cutoff_time)
        start, end = _np.datetime64(start), _np.datetime64(end)
        
        # Rather than copy'n'paste a lot of code, we do this...
        def kernel(points):
            points = _np.asarray(points)
            xdeltas = (points[0][:,None] - events.coords[0][None,:]) / self.grid
            ydeltas = (points[1][:,None] - events.coords[1][None,:]) / self.grid
            distances = _np.sqrt(xdeltas**2 + ydeltas**2)
            num_points = points.shape[1] if len(points.shape) > 1 else 1
            time_deltas = (end - start) * _np.random.random(num_points) + start
            times = (time_deltas[:,None] - events.timestamps[None,:]) / self.time_unit
            r = _np.sum(self.weight(times, distances), axis=-1)
            # Return a scalar if input as scalar
            return r[0] if len(r)==1 else r
        
        krp = _predictors.KernelRiskPredictor(kernel, cell_width=self.grid,
                cell_height=self.grid, samples=samples)
        
        return _predictors.GridPredictionArray.from_continuous_prediction_grid(krp, grid)        
