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
import datetime as _datetime

Cluster = _collections.namedtuple("Cluster", ["centre", "radius"])

def _possible_start_times(timestamps, max_interval_length, end_time):
    times = _np.datetime64(end_time) - timestamps
    zerotime = _np.timedelta64(0,"s")
    times = timestamps[(zerotime <= times) & (times <= max_interval_length)]
    if len(times) <= 1:
        return times
    deltas = times[1:] - times[:-1]
    return _np.hstack(([times[0]],times[1:][deltas > zerotime]))

def _possible_space_clusters(points, max_radius=_np.inf):
    discs = []
    for pt in points.T:
        distances = pt[:,None] - points
        distances = _np.sqrt(_np.sum(distances**2, axis=0))
        distances.sort()
        discs.extend(Cluster(pt, r*1.00001) for r in distances if r <= max_radius)
    # Reduce number
    # Use a tuple here so we can use a set; this is _much_ faster
    allmasks = [tuple(_np.sum((points - cluster.centre[:,None])**2, axis=0) <= cluster.radius**2)
             for cluster in discs]
    masks = []
    set_masks = set()
    for i,m in enumerate(allmasks):
        if m not in set_masks:
            masks.append(i)
            set_masks.add(m)
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
        self.data = None
        self.region = None
        pass
    
    @property
    def region(self):
        """The :class:`data.RectangularRegion` which contains the data; used
        by the output to generate grids etc.  If set to `None` then will
        automatically be the bounding-box of the input data.
        """
        if self._region is None:
            self.region = None
        return self._region
    
    @region.setter
    def region(self, value):
        if value is None and self.data is not None:
            value = self.data.bounding_box
        self._region = value

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
    
    def _possible_start_times(self, end_time, timestamps):
        """A generator returing all possible start times"""
        N = len(timestamps)
        times = _np.unique(timestamps)
        for st in times:
#        for st in _possible_start_times(timestamps,
#                                        self.time_max_interval, end_time):
            events_in_time = (timestamps >= st) & (timestamps <= end_time)
            count = _np.sum(events_in_time)
            if count <= self.time_population_limit * N:
                yield st, count, events_in_time
                
    def _disc_generator(self, discs, events):
        """A generator which yields triples `(disc, count, mask)` where `disc`
        is a :class:`Cluster` giving the space disk, `count` is the number of
        events in this disc, and `mask` is the boolean mask of which events are
        in the disc.
        
        :param discs: An iterable giving the discs
        """
        for disc in discs:
            space_counts = ( _np.sum((events.coords - disc.centre[:,None])**2, axis=0)
                    <= disc.radius ** 2 )
            count = _np.sum(space_counts)
            yield disc, count, space_counts
    
    def _possible_discs(self, events):
        """Yield all possible discs which satisfy our limits"""
        all_discs = _possible_space_clusters(events.coords, self.geographic_radius_limit)
        N = events.number_data_points
        for disc, count, space_counts in self._disc_generator(all_discs, events):
            if count <= N * self.geographic_population_limit:
                yield disc, count, space_counts
    
    def _statistic(self, actual, expected, total):
        """Calculate the log likelihood"""
        stat = actual * (_np.log(actual) - _np.log(expected))
        stat += (total - actual) * (_np.log(total - actual) - _np.log(total - expected))
        return stat
    
    def _thing(self, disc_times, events, end_time, N, times_lookup):
        times = _np.unique(disc_times)
        for start_time in times:
            if end_time - start_time > self.time_max_interval:
                continue
            total_count = times_lookup.get(start_time)
            if total_count is None or total_count > self.time_population_limit * N:
                continue
            count = _np.sum(disc_times >= start_time)
            yield start_time, total_count, count
    
    def _scan_all(self, end_time, events, discs_generator, disc_output=None, timestamps=None):
        if timestamps is None:
            timestamps = events.timestamps
        best = (None, -_np.inf, None)
        N = events.number_data_points
        times_lookup = { time:count for time, count, _ in
                        self._possible_start_times(end_time, timestamps) }

        for disc, space_count, space_mask in discs_generator:
            if disc_output is not None:
                disc_output.append(disc)
            for start, time_count, actual in self._thing(events.timestamps[space_mask], events, end_time, N, times_lookup):
                expected = time_count * space_count / N
                if actual > expected and actual > 1:
                    stat = self._statistic(actual, expected, N)
                    if stat > best[1]:
                        best = (disc, stat, start)
        return best

    def _scan_all_old(self, end_time, events, discs_generator, disc_output=None, timestamps=None):
        if timestamps is None:
            timestamps = events.timestamps
        best = (None, -_np.inf, None)
        N = events.number_data_points
        possible_times = list(self._possible_start_times(end_time, timestamps))

        for disc, space_count, space_mask in discs_generator:
            if disc_output is not None:
                disc_output.append(disc)
            for start, time_count, time_mask in possible_times:
                expected = time_count * space_count / N
                actual = _np.sum(time_mask & space_mask)
                if actual > expected and actual > 1:
                    stat = self._statistic(actual, expected, N)
                    if stat > best[1]:
                        best = (disc, stat, start)
        return best

    def _remove_intersecting(self, all_discs, disc):
        return [ d for d in all_discs
            if _np.sum((d.centre - disc.centre)**2) > (d.radius + disc.radius)**2
            ]

    def _events_time(self, time=None):
        events = self.data.events_before(time)
        if time is None:
            time = self.data.timestamps[-1]
        return events, time

    def predict(self, time=None, max_clusters=None):
        """Make a prediction.
        
        :param time: Timestamp of the prediction point.  Only data up to
          and including this time is used when computing clusters.  If `None`
          then use the last timestamp of the data.
        :param max_clusters: If not `None` then return at most this many
          clusters.
        
        :return: A instance of :class:`STSResult` giving the found clusters.
        """
        events, time = self._events_time(time)
        all_discs = []
        clusters = []
        best_disc, stat, start_time = self._scan_all(time, events,
            self._possible_discs(events), all_discs)
        
        while best_disc is not None:
            clusters.append((best_disc, stat, start_time))
            all_discs = self._remove_intersecting(all_discs, best_disc)
            if len(all_discs) == 0:
                break
            if max_clusters is not None and len(clusters) >= max_clusters:
                break
            best_disc, stat, start_time = self._scan_all(time, events,
                self._disc_generator(all_discs, events))

        clusters, stats, start_times = zip(*clusters)
        time_regions = [(s,time) for s in start_times]
        max_clusters = self.maximise_clusters(clusters, time)
        return STSResult(self.region, clusters, max_clusters,
                         time_ranges=time_regions, statistics=stats)

    def monte_carlo_simulate(self, time=None, runs=999):
        """Perform a monte carlo simulation for the purposes of estimating 
        p-values.  We repeatedly shuffle the timestamps of the data and then
        find the most likely cluster for each new dataset.  This method is
        more efficient than calling :method:`predict` repeatedly with
        shuffled data.

        :param time: Optionally restrict the data to before this time, as for
          :method:`predict`
        :param runs: The number of samples to take, by default 999

        :return: An ordered list of statistics.
        """
        events, time = self._events_time(time)
        all_discs = []
        best_disc, stat, start_time = self._scan_all(time, events,
            self._possible_discs(events), all_discs)
        timestamps = _np.array(events.timestamps)
        stats = []
        for _ in range(runs):
            _np.random.shuffle(timestamps)
            _,stat,_ = self._scan_all(time, events,
                self._disc_generator(all_discs, events), timestamps = timestamps)
            stats.append(stat)
        stats = _np.asarray(stats)
        stats.sort()
        return stats

    def maximise_clusters(self, clusters, time=None):
        """The prediction method will return the smallest clusters (subject
        to each cluster being centred on the coordinates of an event).  This
        method will enlarge each cluster to the maxmimum radius it can be
        without including further events.
        
        :param clusters: List-like object of :class:`Cluster` instances.
        :param time: Only data up to and including this time is used when
          computing clusters.  If `None` then use the last timestamp of the
          data.
        
        :return: Array of clusters with larger radii.
        """
        events, time = self._events_time(time)
        out = []
        for disc in clusters:
            distances = _np.sum((events.coords - disc.centre[:,None])**2, axis=0)
            rr = disc.radius ** 2
            new_radius = _np.sqrt(min( dd for dd in distances if dd > rr ))
            out.append(Cluster(disc.centre, new_radius))
        return out
    
    def to_satscan(self, filename):
        """Writes the training data to two SaTScan compatible files.  Does
        *not* currently write settings, so these will need to be entered
        manually.
        
        :param filename: Saves files "filename.geo" and "filename.cas"
          containing the geometry and "cases" repsectively.
        """
        def timeformatter(t):
            t = _np.datetime64(t, "s")
            return str(t)

        unique_coords = list(set( (x,y) for x,y in self.data.coords.T ))
        with open(filename + ".geo", "w") as geofile:
            for i, (x,y) in enumerate(unique_coords):
                print("{}\t{}\t{}".format(i+1, x, y), file=geofile)

        unique_times = list(set( t for t in self.data.timestamps ))
        with open(filename + ".cas", "w") as casefile:
            for i, (t) in enumerate(unique_times):
                pts = self.data.coords.T[self.data.timestamps == t]
                pts = [ (x,y) for x,y in pts ]
                import collections
                c = collections.Counter(pts)
                for pt in c:
                    index = unique_coords.index(pt)
                    print("{}\t{}\t{}".format(index+1, c[pt], timeformatter(t)), file=casefile)


class STSContinuousPrediction(predictors.ContinuousPrediction):
    """A :class:`predictors.ContinuousPrediction` which uses the computed
    clusters and a user-defined weight to generate a continuous "risk"
    prediction.  Set the :attr:`weight` to change weight.
    
    It is not clear that the generated "risk" has much to do with reality!
    We, by default, use enlarged cluster sizes (with removes the problem of
    clusters with zero radius!) which can lead to overlapping clusters.
    
    :param clusters: List of computed clusters.
    """
    def __init__(self, clusters):
        super().__init__()
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
    
    def _vectorised_weight(self, values):
        """Allows values to be a one-dimensional array.  Returns 0 is the
        value is not in the interval [0,1).
        """
        values = _np.asarray(values)
        allowed = (values >= 0) & (values < 1)
        if len(values.shape) > 0:
            return _np.asarray([self.weight(x) if a else 0.0 for x, a in zip(values,allowed)])
        return self.weight(values) if allowed else 0.0
    
    def risk(self, x, y):
        """The relative "risk", varying between 0 and `n`, the number of
        clusters detected.
        """
        pt = _np.array([x,y])
        if len(pt.shape) == 1:
            pt = pt[:,None]
        risk = _np.zeros(pt.shape[1])
        for n, cluster in enumerate(self.clusters):
            dist = ( _np.sqrt(_np.sum((pt - _np.asarray(cluster.centre)[:,None])**2, axis=0))
                    / cluster.radius )
            weights = self._vectorised_weight(dist)
            risk += (len(self.clusters) - n - 1 + weights) * (weights > 0)
        return risk                


class STSResult():
    """Stores the computed clusters from :class:`STSTrainer`.  These can be
    used to produce gridded or continuous "risk" predictions.
    
    :param region: The rectangular region enclosing the data.
    :param clusters: A list of :class:`Cluster` instances describing the found
      clusters.
    :param max_clusters: A list of :class:`Cluster` instances describing the
      clusters with radii enlarged to the maximal extent.
    :param time_ranges: The time range associated with each cluster.
    :param statistics: The value of the log likelihood for each cluster.
    :param pvalues: (Optionally) the estimated p-values.
    """
    def __init__(self, region, clusters, max_clusters=None, time_ranges=None,
                 statistics=None, pvalues=None):
        self.region = region
        self.clusters = clusters
        if max_clusters is None:
            max_clusters = clusters
        self.max_clusters = max_clusters
        self.time_ranges = time_ranges
        self.statistics = statistics
        self.pvalues = pvalues
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
        
        :param grid_size: The size of resulting grid.
        """
        xs, ys = self.region.grid_size(grid_size)
        risk_matrix = _np.zeros((ys, xs))
        for n, cluster in enumerate(self.clusters):
            self._add_cluster(cluster, risk_matrix, grid_size,
                              len(self.clusters) - n - 1)
        return predictors.GridPredictionArray(grid_size, grid_size, risk_matrix,
            xoffset=self.region.xmin, yoffset=self.region.ymin)

    def continuous_prediction(self, use_maximal_clusters=True):
        """Make a continuous prediction based upon the found clusters.
        
        :param use_maximal_clusters: If `True` then use the largest possible
          radii for each cluster.
        
        :return: An instance of :class:`STSContinuousPrediction` which allows
          further options to be set.
        """
        clusters = self.max_clusters if use_maximal_clusters else self.clusters
        return STSContinuousPrediction(clusters)
