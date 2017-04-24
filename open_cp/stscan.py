"""
stscan
~~~~~~

Implements the "prospective" space-time permutation scan statistic algorithm.
This was originally described in (1) in reference to disease outbreak
detection.  The algorithm is implemented in the software package (2).  We
apply it to crime predication as in (3).

We look at events which have occurred in the past, and try to detect "clusters"
which are existing up to the current time.  To do this, a simple statistic
which measures deviation was expected randomness is computed for every
possible space/time "cylinder": events which occur is a circular disk in space,
in an interval of time (always ending at the point of prediction).  The space/
time cylinder with the largest statistic is deemed the most likely "cluster".
Further clusters are computed by finding the next most likely cluster which
does not intersect (in space only) the existing cluster.

As detailed in (1) and (2) it is possible to use monte-carlo methods to
estimate the p-value of the primary cluster, but for prediction purposes this
is not necessary.  As adapted from (3), we use the clusters in order to find
a relative "risk" of crime.

References
~~~~~~~~~~
1. Kulldorff et al, "A Space–Time Permutation Scan Statistic for Disease
  Outbreak Detection", PLoS Med 2(3): e59, DOI:10.1371/journal.pmed.0020059
2. Kulldorff M. and Information Management Services, Inc. SaTScanTM v8.0:
  Software for the spatial and space-time scan statistics.
  http://www.satscan.org/, 2016.
3. Adepeju, Rosser, Cheng, "Novel evaluation metrics for sparse spatiotemporal
  point process hotspot predictions - a crime case study", International
  Journal of Geographical Information Science, 30:11, 2133-2154,
  DOI:10.1080/13658816.2016.1159684
"""

from . import predictors
from . import data
import numpy as _np
import collections as _collections

Cluster = _collections.namedtuple("Cluster", ["centre", "radius"])


def _possible_start_times(timestamps, max_interval_length, end_time):
    times = end_time - timestamps
    zerotime = _np.timedelta64(0,"s")
    times = timestamps[(zerotime <= times) & (times <= max_interval_length)]
    if len(times) <= 1:
        return times
    deltas = times[1:] - times[:-1]
    return _np.hstack(([times[0]],times[1:][deltas > zerotime]))

def _possible_space_clusters(points):
    discs = []
    for pt in points.T:
        distances = pt[:,None] - points
        distances = _np.sqrt(_np.sum(distances**2, axis=0))
        distances.sort()
        discs.extend(Cluster(pt, r) for r in distances)
    # Reduce number
    allmasks = [_np.sum((points - cluster.centre[:,None])**2, axis=0) <= cluster.radius**2
             for cluster in discs]
    masks = []
    for i,m in enumerate(allmasks):
        if any( _np.all(m==allmasks[index]) for index in masks ):
            continue
        masks.append(i)
    return [discs[i] for i in masks]


class STSTrainer(predictors.DataTrainer):
    """From past events, produce an instance of :class:`STSResult` which
    stores details of the found clusters.  Contains a variety of properties
    which may be changed to affect the prediction behaviour.
    """
    def __init__(self):
        self.geographic_population_limit = 0.5
        self.geographic_radius_limit = 3000
        self.time_population_limit = 0.5
        self.time_max_interval = _np.timedelta64(12, "W")
        pass
    
    @property
    def geographic_population_limit(self):
        """No space disc can contain more than this fraction of the total
        number of events.
        """
        return self._geo_pop_limit
    
    @geographic_population_limit.setter
    def geographic_population_limit(self, value):
        if value < 0 or value > 1:
            raise ValueError("Should be fraction of total population, so value between 0 and 1")
        self._geo_pop_limit = value

    @property
    def geographic_radius_limit(self):
        """The maximum radius of the space discs."""
        return self._geo_max_radius
    
    @geographic_radius_limit.setter
    def geographic_radius_limit(self, value):
        self._geo_max_radius = value
        
    @property
    def time_population_limit(self):
        """No time interval can contain more than this fraction of the total
        number of events.start_times
        """
        return self._time_pop_limit
    
    @time_population_limit.setter
    def time_population_limit(self, value):
        if value < 0 or value > 1:
            raise ValueError("Should be fraction of total population, so value between 0 and 1")
        self._time_pop_limit = value
        
    @property
    def time_max_interval(self):
        """The maximum length of a time interval."""
        return self._time_max_len
    
    @time_max_interval.setter
    def time_max_interval(self, value):
        self._time_max_len = _np.timedelta64(value)
        
    def clone(self):
        """Return a new instance which has all the underlying settings
        but with no data.
        """
        new = STSTrainer()
        new.geographic_population_limit = self.geographic_population_limit
        new.geographic_radius_limit = self.geographic_radius_limit
        new.time_population_limit = self.time_population_limit
        new.time_max_interval = self.time_max_interval
        return new
        
    def bin_timestamps(self, offset, bin_length):
        """Returns a new instance with the underlying timestamped data
        adjusted.  Any timestamp between `offset` and `offset + bin_length`
        is mapped to `offset`; timestamps between `offset + bin_length`
        and `offset + 2 * bin_length` are mapped to `offset + bin_length`,
        and so forth.
        
        :param offset: A datetime-like object which is the start of the
          binning.
        :param bin_length: A timedelta-like object which is the length of
          each bin.
        """
        offset = _np.datetime64(offset)
        bin_length = _np.timedelta64(bin_length)
        new_times = _np.floor((self.data.timestamps - offset) / bin_length)
        new_times = offset + new_times * bin_length
        new = self.clone()
        new.data = data.TimedPoints(new_times, self.data.coords)
        return new
    
    def grid_coords(self, region, grid_size):
        """Returns a new instance with the underlying coordinate data
        adjusted to always be the centre point of grid cells.
        
        :param region: A `data.RectangularRegion` instance giving the
          region to grid to.  Only the x,y offset is used.
        :param grid_size: The width and height of each grid cell.
        """
        offset = _np.array([region.xmin, region.ymin])
        newcoords = _np.floor((self.data.coords - offset[:,None]) / grid_size) + 0.5
        newcoords = newcoords * grid_size + offset[:,None]
        new = self.clone()
        new.data = data.TimedPoints(self.data.timestamps, newcoords)
        return new
    
    def predict(self, time=None):
        """Make a prediction.
        
        :param time: Timestamp of the prediction point.  Only data up to
          and including this time is used when computing clusters.  If `None`
          then use the last timestamp of the data.
        
        :return: A instance of :class:`STSResult` giving the found clusters.
        """
        events = self.data.events_before(time)
        if time is None:
            time = self.data.timestamps[-1]
            
        start_times = _possible_start_times(events.timestamps,
                                            self.time_max_interval, time)
        time_counts = _np.empty(len(start_times), dtype=_np.object)
        for i, st in enumerate(start_times):
            time_counts[i] = (events.timestamps >= st) & (events.timestamps <= time)

        discs = _possible_space_clusters(events.coords)
        space_counts = _np.empty(len(discs), dtype=_np.object)
        for i, disc in enumerate(discs):
            space_counts[i] = ( _np.sum((events.coords - disc.centre[:,None])**2, axis=0)
                                <= disc.radius ** 2 )

        for i, disc in enumerate(discs):
            space_count = space_counts[i] # So don't need to cache??!
            norm_sc = _np.sum(space_count) / events.number_data_points
            for j, start in enumerate(start_times):
                time_count = time_counts[j]
                expected = _np.sum(time_count) * norm_sc
                actual = _np.sum(time_count & space_count)
                if actual > expected:
                    pass

        raise NotImplementedError()


class STSContinuousPrediction(predictors.ContinuousPrediction):
    """A :class:`predictors.ContinuousPrediction` which uses the computed
    clusters and a user-defined weight to generate a continuous "risk"
    prediction.  Set the :attr:`weight` to change weight.
    
    :param clusters: List of computed clusters.
    """
    def __init__(self, clusters):
        self.weight = self.quatric_weight
        self.clusters = clusters
        pass
    
    @staticmethod
    def quatric_weight(t):
        return (1 - t * t) ** 2
    
    @property
    def weight(self):
        """A function-like object which when called with a float between 0 and
        1 (interpreted as the distance to the edge of a unit disc) returns a
        float between 0 and 1, the "intensity".  Default is the quatric
        function :math:`t \mapsto (1-t^2)^2`.
        """
        return self._weight
    
    @weight.setter
    def weight(self, value):
        self._weight = value
    
    def risk(self, x, y):
        """The relative "risk", varying between 0 and `n`, the number of
        clusters detected.
        """
        pt = _np.array([x,y])
        risk = 0.0
        for n, cluster in enumerate(self.clusters):
            dist = ( _np.sqrt(_np.sum((_np.asarray(cluster.centre) - pt)**2))
                    / cluster.radius )
            if dist < 1.0:
                risk += len(self.clusters) - n - 1 + self.weight(dist)
        return risk                


class STSResult():
    """Stores the computed clusters from :class:`STSTrainer`.  These can be
    used to produce gridded or continuous "risk" predictions.
    """
    def __init__(self, region, clusters):
        self.region = region
        self.clusters = clusters
        pass
    
    def _add_cluster(self, cluster, risk_matrix, grid_size, base_risk):
        """Adds risk in base_risk + (0,1]"""
        cells = []
        for y in range(risk_matrix.shape[0]):
            for x in range(risk_matrix.shape[1]):
                xcoord = (x + 0.5) * grid_size + self.region.xmin
                ycoord = (y + 0.5) * grid_size + self.region.ymin
                distance = _np.sqrt((xcoord - cluster.centre[0]) ** 2 +
                                    (ycoord - cluster.centre[1]) ** 2)
                if distance <= cluster.radius:
                    cells.append((x,y,distance))
        cells.sort(key = lambda triple : triple[2], reverse=True)
        for i, (x,y,d) in enumerate(cells):
            risk_matrix[y][x] = base_risk + (i+1) / len(cells)
    
    def grid_prediction(self, grid_size):
        """Using the grid size, construct a grid from the region and 
        produce an instance of :class:`predictors.GridPredictionArray` which
        contains the relative "risk".
        
        We treat each cluster in order, so that the primary cluster has higher
        risk than the secondary cluster, and so on.  Within each cluster,
        cells near the centre have a higher risk than cells near the boundary.
        
        It is probably more "accurate" to produce a continuous prediction
        and then convert that to a gridded prediction in the standard way.
        
        :param grid_size: The size of resulting grid.
        """
        xs, ys = self.region.grid_size(grid_size)
        risk_matrix = _np.zeros((ys, xs))
        print(risk_matrix.shape, ys, xs)
        for n, cluster in enumerate(self.clusters):
            self._add_cluster(cluster, risk_matrix, grid_size,
                              len(self.clusters) - n - 1)
        return predictors.GridPredictionArray(xs, ys, risk_matrix,
            xoffset=self.region.xmin, yoffset=self.region.ymin)

    def continuous_prediction(self):
        return STSContinuousPrediction(self.clusters)
