"""
seppexp
~~~~~~~

Implements the ETAS (Epidemic Type Aftershock-Sequences) model intensity
estimation scheme outlined in Mohler et al. (2015).  This model is somewhat
different, and simplified, from that used in the :mod:`open_cp.sepp` module:

- This is an explicitly grid based model.  All events are assigned to the grid
  cell in which the occur, and we make no more use of their location.
- For each cell, we produce an independent estimate of the background rate of
  events.
- We model "self-excitation" only in time, as a simple exponential decay (much
  like the classical Hawkes model in Financial mathematics).  We assume the decay
  parameters are the same across all grid cells.

References
~~~~~~~~~~
1. Mohler et al, "Randomized Controlled Field Trials of Predictive Policing",
   Journal of the American Statistical Association (2015)
   DOI:10.1080/01621459.2015.1077710

2. Lewis, Mohler, "A Nonparametric EM Algorithm for Multiscale Hawkes Processes"
   in Proceedings of the 2011 Joint Statistical Meetings, pp. 1–16
   http://math.scu.edu/~gmohler/EM_paper.pdf
"""

from . import predictors
import numpy as _np
import itertools as _itertools
import logging as _logging

def _normalise_matrix(p):
    column_sums = _np.sum(p, axis=0)
    return p / column_sums[None,:]

def _iter_array(a):
    """Returns an iterable which yields tuples of indexes into the array `a`."""
    return _itertools.product(*[list(range(i)) for i in a.shape])

def p_matrix(points, omega, theta, mu):
    """Computes the probability matrix.  Diagonal entries are the background
    rate, and entry [i,j] is `g(points[j] - points[i])` for `i<j`, where
    :math:`g(t) = \theta \omega e^{-\omega t}`.  Finally we normalise the matrix
    to have columns which sum to 1.

    :param points: A one-dimensional array of the times of events, in
      increasing order.
    :param omega: The scale of the "triggering" exponential distribution
    :param theta: The rate of the "triggering" intensity
    :param mu: The background Poisson process rate.

    :return: The normalised probability matrix.
    """
    number_data_points = len(points)
    p = _np.zeros((number_data_points, number_data_points))
    for j in range(1, number_data_points):
        d = points[j] - points[0:j]
        p[0:j, j] = theta * omega * _np.exp(-omega * d)
    for j in range(number_data_points):
        p[j, j] = mu
    return _normalise_matrix(p)

def _slow_maximisation(cells, omega, theta, mu, time_duration):
    """Pure Python implementation of :func:`maximisation` for testing."""
    cells, mu = _np.asarray(cells), _np.asarray(mu)
    new_mu = []
    omega_1, omega_2, count = 0, 0, 0
    for cell, m in zip(cells.ravel(), mu.ravel()):
        p = p_matrix(cell, omega, theta, m)
        diag = 0
        for j in range(len(cell)):
            for i in range(j):
                omega_1 += p[i, j]
                omega_2 += p[i, j] * (cell[j] - cell[i])
            count += 1
            diag += p[j, j]
        new_mu.append(diag / time_duration)

    omega = omega_1 / omega_2
    theta = omega_1 / count
    mu = _np.asarray(new_mu).reshape(mu.shape)
    return (omega, theta, mu)

def maximisation(cells, omega, theta, mu, time_duration):
    """Perform an iteration of the EM algorithm.

    :param cells: An array (of any shape) each entry of which is an array of
      times of events, in increasing order.
    :param mu: An array, of the same shape as `cells`, giving the background
      rate in each cell.
    :param time_duration: The total time range of the data.

    :return: Triple `(omega, theta, mu)` of new estimates.
    """
    cells, mu = _np.asarray(cells), _np.asarray(mu)
    upper_trianglar_sums = _np.zeros_like(mu)
    weighted_upper_trianglar_sums = _np.zeros_like(mu)
    diagonal_sums = _np.zeros_like(mu)
    event_counts = _np.zeros_like(mu)
    
    for index in _iter_array(cells):
        times = cells[index]
        p = p_matrix(times, omega, theta, mu[index])
        diag_sum = _np.sum(_np.diag(p))
        diagonal_sums[index] = diag_sum
        upper_trianglar_sums[index] = _np.sum(p) - diag_sum
        weighted_p = p * (times[None, :] - times[:, None])
        weighted_upper_trianglar_sums[index] = _np.sum(weighted_p)
        event_counts[index] = len(times)
    
    omega = _np.sum(upper_trianglar_sums) / _np.sum(weighted_upper_trianglar_sums)
    theta = _np.sum(upper_trianglar_sums) / _np.sum(event_counts)
    mu = diagonal_sums / time_duration

    return (omega, theta, mu)

def _slow_maximisation_corrected(cells, omega, theta, mu, time_duration):
    """Pure Python implementation of :func:`maximisation_corrected`
    for testing.
    """
    cells, mu = _np.asarray(cells), _np.asarray(mu)
    new_mu = []
    omega_1, omega_2, count = 0, 0, 0
    for cell, m in zip(cells.ravel(), mu.ravel()):
        p = p_matrix(cell, omega, theta, m)
        diag = 0
        for j in range(len(cell)):
            for i in range(j):
                omega_1 += p[i, j]
                omega_2 += p[i, j] * (cell[j] - cell[i])
            count += 1
            diag += p[j, j]
        ctheta, comega = 0, 0
        for t in cell:
            c = time_duration - t
            ce = _np.exp(-omega * c)
            ctheta += ce
            comega += c * ce
        count -= ctheta
        omega_2 += comega * theta
        new_mu.append(diag / time_duration)

    omega = omega_1 / omega_2
    theta = omega_1 / count
    mu = _np.asarray(new_mu).reshape(mu.shape)
    return (omega, theta, mu)

def maximisation_corrected(cells, omega, theta, mu, time_duration):
    """Perform an iteration of the EM algorithm.  This version applies "edge
    corrections" (see Lewis, Mohler) which take account of the fact that by
    looking at a finite time window, we ignore aftershocks which occur after
    the end of the time window.  This leads to better parameter estimation
    when `omega` is small.

    :param cells: An array (of any shape) each entry of which is an array of
      times of events, in increasing order.
    :param mu: An array, of the same shape as `cells`, giving the background
      rate in each cell.
    :param time_duration: The total time range of the data.

    :return: Triple `(omega, theta, mu)` of new estimates.
    """
    cells, mu = _np.asarray(cells), _np.asarray(mu)
    upper_trianglar_sums = _np.zeros_like(mu)
    weighted_upper_trianglar_sums = _np.zeros_like(mu)
    diagonal_sums = _np.zeros_like(mu)
    event_counts = _np.zeros_like(mu)
    
    for index in _itertools.product(*[list(range(i)) for i in cells.shape]):
        times = cells[index]
        p = p_matrix(times, omega, theta, mu[index])
        diag_sum = _np.sum(_np.diag(p))
        diagonal_sums[index] = diag_sum
        upper_trianglar_sums[index] = _np.sum(p) - diag_sum
        weighted_p = p * (times[None, :] - times[:, None])
        dt = time_duration - times
        dtt = _np.exp(-omega * dt)
        weighted_upper_trianglar_sums[index] = (_np.sum(weighted_p) + 
            theta * _np.sum(dt * dtt))
        event_counts[index] = len(times) - _np.sum(dtt)
    
    omega = _np.sum(upper_trianglar_sums) / _np.sum(weighted_upper_trianglar_sums)
    theta = _np.sum(upper_trianglar_sums) / _np.sum(event_counts)
    mu = diagonal_sums / time_duration

    return (omega, theta, mu)

def _make_cells(region, grid_size, events, times):
    xsize, ysize = region.grid_size(grid_size)
    cells = _np.empty((ysize, xsize), dtype=_np.object)
    for x in range(xsize):
        for y in range(ysize):
            cells[y,x] = []
    xcs = _np.floor((events.xcoords - region.xmin) / grid_size)
    ycs = _np.floor((events.ycoords - region.ymin) / grid_size)
    xcs = xcs.astype(_np.int)
    ycs = ycs.astype(_np.int)
    for i, time in enumerate(times):
        cells[ycs[i], xcs[i]].append(time)
    for x in range(xsize):
        for y in range(ysize):
            cells[y,x] = _np.asarray(cells[y,x])
    return cells


class SEPPPredictor(predictors.DataTrainer):
    """Returned by :class:`SEPPTrainer` encapsulated computed background rates
    and triggering parameters.  This class allows these to be evaluated on
    potentially different data to produce predictions.
    """
    def __init__(self, region, grid_size, omega, theta, mu):
        self.omega = omega
        self.theta = theta
        self.mu = mu
        self.region = region
        self.grid_size = grid_size

    def background_rate(self, x, y):
        """Return the background rate in grid cell `(x,y)`."""
        return self.mu[y, x]

    def background_prediction(self):
        """Make a "prediction" just using the background rate.  Useful as it
        allows a direct comparison with the output of :meth:`predict`.
        
        :return: Instance of :class:`open_cp.predictors.GridPredictionArray`
        """
        matrix = _np.array(self.mu, dtype=_np.float)
        return predictors.GridPredictionArray(self.grid_size, self.grid_size,
            matrix, self.region.xmin, self.region.ymin)

    def predict(self, predict_time, cutoff_time=None):
        """Make a prediction at a time, using the data held by this instance.
        That is, evaluate the background rate plus the trigger kernel at
        events before the prediction time.  Optionally you can limit the data
        used, though this is against the underlying statistical model.

        :param predict_time: Time point to make a prediction at.
        :param cutoff_time: Optionally, limit the input data to only be from
          before this time.

        :return: Instance of :class:`open_cp.predictors.GridPredictionArray`
        """
        events = self.data.events_before(cutoff_time)
        times = (_np.datetime64(predict_time) - events.timestamps) / _np.timedelta64(1, "m")
        cells = _make_cells(self.region, self.grid_size, events, times)
        if cells.shape != self.mu.shape:
            raise ValueError("Background rate on grid sized {} but this region"
                "gives grid of size {}".format(self.mu.shape, cells.shape))

        matrix = _np.empty_like(cells, dtype=_np.float)
        for index in _iter_array(cells):
            dt = cells[index]
            dt = dt[dt>0]
            matrix[index] = _np.sum(self.theta * self.omega * _np.exp(-self.omega * dt))
        matrix += self.mu
        return predictors.GridPredictionArray(self.grid_size, self.grid_size,
            matrix, self.region.xmin, self.region.ymin)


class SEPPTrainer(predictors.DataTrainer):
    """Use the algorithm described in Mohler et al. 2015.  The input data is
    placed into grid cells, and background rates estimated for each cell.  The
    parameters for the exponential decay model of self-excitation are also
    estimated.  The returned object can be used to make predictions of risk
    from other data.

    For interpretting the parameters, we note that a time unit of "minutes" is
    used internally.  The initial conditions are chosen in a way which fits
    "real" data (e.g. an expected trigger time of the order of days).  This can
    lead to problems with data which is distributed in a very different way.

    :param region: The rectangular region the grid should cover.
    :param grid_size: The size of grid to use.
    :param grid: Alternative to specifying the region and grid_size is to pass
      a :class:`BoundedGrid` instance.
    """
    def __init__(self, region=None, grid_size=50, grid=None):
        self._logger = _logging.getLogger(__name__)
        if grid is None:
            self.grid_size = grid_size
            self.region = region
        else:
            self.region = grid.region()
            self.grid_size = grid.xsize
            if grid.xsize != grid.ysize:
                raise ValueError("Only supports *square* grid cells.")

    def _make_cells(self, events):
        times = events.time_deltas(time_unit = _np.timedelta64(1, "m"))
        cells = _make_cells(self.region, self.grid_size, events, times)
        return cells, times[-1]

    def train(self, cutoff_time=None, iterations=20, use_corrected=False):
        """Perform the (slow) training step on historical data.  This estimates
        kernels, and returns an object which can make predictions.

        :param cutoff_time: If specified, then limit the historical data to
          before this time.
        
        :return: A :class:`SEPPPredictor` instance.
        """
        events = self.data.events_before(cutoff_time)
        cells, time_duration = self._make_cells(events)
        theta = 0.5
        # time unit of minutes, want mean to be a day
        omega = 1 / (60 * 24)
        mu = _np.vectorize(len)(cells) / time_duration
        if use_corrected:
            for _ in range(iterations):
                self._logger.debug("Iterating with omega=%s, theta=%s, mu=%s", omega, theta, mu)
                omega, theta, mu = maximisation_corrected(cells, omega, theta, mu, time_duration)
                if not _np.all(_np.isfinite([omega, theta])) or not _np.all(_np.isfinite(mu)):
                    raise Exception("Convergence failed!")
            self._logger.debug("Using edge-corrected algorithm, estimated omega=%s, theta=%s, mu=%s",
                               omega, theta, mu)
        else:
            for _ in range(iterations):
                self._logger.debug("Iterating with omega=%s, theta=%s, mu=%s", omega, theta, mu)
                omega, theta, mu = maximisation(cells, omega, theta, mu, time_duration)
                if not _np.all(_np.isfinite([omega, theta])) or not _np.all(_np.isfinite(mu)):
                    raise Exception("Convergence failed!")
            self._logger.debug("Using quicker algorithm, estimated omega=%s, theta=%s, mu=%s",
                               omega, theta, mu)

        return SEPPPredictor(self.region, self.grid_size, omega, theta, mu)
