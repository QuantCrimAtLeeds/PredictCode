"""
knox
~~~~

Implements a monte carlo algorithm to compute knox statistics and p-values.

ADD REFERENCES

"""

from . import data
from . import predictors as _predictors
import numpy as _np
import scipy.spatial.distance as _distance

def distances(points, start=None, end=None):
    """Computes the distances between pairs of points.  In simple operation,
    computes the distances between all pairs of points and returns a
    "compressed distance matrix", exactly like `scipy.spatial.distance`.
    
    The `(i,j)` entry of the distance matrix is the distance from point `i` to
    point `j`.  The distance matrix has identically 0 diagonal, and is
    symmetric.  The compressed distance matrix lists just the entries above the
    diagonal.  That is, the compressed distance matrix lists, in a
    one-dimensional array, the first row of the distance matrix except the
    first entry (which is always 0), then the second row, excepting the first
    2 entries (which we already) know, and so forth.

    Optionally pass `start:end`, to be treated like a slice.  We will then only
    calculate the distances from points `i` to points `j` where `i<j` and
    `start <= i < end`.  By repeatedly calling with different intervals, the
    whole distance matrix can be computed, but in a memory-efficient way.

    :param points: Array of scalars, or array of shape `(n,k)` of `n` points in
      `k` dimensional space.
    :param start: 
    :param end: Optionally, only look at a slice `[start:end]` for the `i`
      entries (as above).
    """
    if start is not None or end is not None:
        raise NotImplementedError()
    if len(points.shape) == 1:
        points = points[:,None]
    numpts = points.shape[0]
    out = _np.empty(numpts * (numpts - 1) // 2)
    index = 0
    for i in range(numpts - 1):
        out[index : index + numpts - i - 1] = _np.sqrt(_np.sum((points[i+1:] - points[i,:])**2, axis=1))
        index += numpts - i - 1
    return out

class Knox(_predictors.DataTrainer):
    def __init__(self, space_bins = None, time_bins = None):
        self.space_bins = space_bins
        self.time_bins = time_bins

    @property
    def space_bins(self):
        """List of pairs `(min, max)`.  When computing, we'll look only at
        pairs of points which are between `min` and `max` spatial distance
        apart (inclusive).
        """
        return list(self._space_bins)

    @space_bins.setter
    def space_bins(self, bins):
        if bins is None:
            self._space_bins = None
        else:
            self._space_bins = list(bins)

    @property
    def time_bins(self):
        """List of pairs `(start, end)`.  When computing, we'll look only at
        pairs of points which are between `start` and `end` time apart
        (inclusive).

        When setting, pass a list (or other iterable) of pairs of objects
        which can either be converted to :module:`numpy`.:class:`timedelta64`
        objects, or use :method:`set_time_bins`.
        """
        return self._time_bins

    @time_bins.setter
    def time_bins(self, bins):
        if bins is None:
            self._time_bins = None
        else:
            self._time_bins = [
                (_np.timedelta64(s), _np.timedelta64(e))
                for s, e in bins ]

    _UNIT_LOOKUP = {"millis":1, "seconds":1000, "minutes":60000, "hours":3600000, "days":3600*1000*24}

    def set_time_bins(self, bins, unit="hours"):
        """Set the time bins from a list of numbers are a unit.

        :param bins: A list of pairs of (start, end) times.
        :param unit: One of "days", "hours", "minutes", "seconds", "millis"
          defining the time unit.
        """
        if unit not in self._UNIT_LOOKUP:
            raise ValueError("Unknown unit '{}'".format(unit))
        scale = self._UNIT_LOOKUP[unit]
        self._time_bins = [
            (_np.timedelta64(int(s*scale),"ms"),_np.timedelta64(int(e*scale),"ms"))
            for s, e in bins ]
    
    def _stat(self, distances, time_distances):
        cells = _np.empty((len(self.space_bins), len(self.time_bins)))
        for j, time_bin in enumerate(self.time_bins):
            start = time_bin[0] / _np.timedelta64(1, "ms")
            end = time_bin[1] / _np.timedelta64(1, "ms")
            for i, space_bin in enumerate(self.space_bins):
                cells[i][j] = _np.sum( (distances >= space_bin[0]) & (distances <= space_bin[1])
                        & (time_distances >= start) & (time_distances <= end) )
        return cells

    def calculate(self, iterations=999):
        """Calculates the knox statistic for each cell, and the monte carlo
        derived p-value.  For each space bin and time bin, we create a cell
        and perform the calculation for that cell with that space/time cutoff.

        :param iterations: The number of iterations to perform when estimating
          the p-value.
        
        :return: An array whose `[i,j]` entry corresponds to space bin [i] and
          time bin [j].  Each entry is a pair `(statistic, p-value)`.
        """
        points = self.data.coords.T
        dists = _distance.pdist(points)
        times = (self.data.timestamps - self.data.timestamps[0]) / _np.timedelta64(1, "ms")
        time_distances = distances(times)
        
        cells = self._stat(dists, time_distances)
        monte_carlo_cells = []
        for _ in range(iterations):
            _np.random.shuffle(times)
            time_distances = distances(times)
            monte_carlo_cells.append(self._stat(dists, time_distances))
        
        out = _np.empty(cells.shape + (2,))
        for i in range(cells.shape[0]):
            for j in range(cells.shape[1]):
                statistic = cells[i][j]
                out[i][j][0] = statistic
                stats = _np.asarray([c[i][j] for c in monte_carlo_cells])
                pvalue = _np.sum( statistic <= stats ) / (1 + iterations)
                out[i][j][1] = pvalue
        return out