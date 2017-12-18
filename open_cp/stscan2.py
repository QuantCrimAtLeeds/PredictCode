"""
stscan2
~~~~~~~

Two further attempts at the algorithm.  The first tries harder to conform
_exactly_ to what SaTScan does (but fails).  The second uses `numpy` to
accelerate the (original) algorithm to speeds whereby this is a useful method.

- A "cluster" never consists of just one event.
- Discs are also centred on actual events.
- If the boundary of a disc contains more than one event, then we test *all*
  possibilities of including / excluding events on the boundary
    - So, for example, if we have a disc centred at (1,1) and events at
      (0,0), (1,1) and (2,2) then we would consider the discs containing events
      {0,1}, {1,2} and {0,1,2}.  More naively, we'd only look at {0,1,2}.
  This _still_ doesn't exactly reproduce what SaTScan does.
     
  
The classes here are useful for testing and verification.  The module
:mod:`stscan` should still be used for making actual predictions (it uses
:class:`STScanNumpy` below internally).
"""

import numpy as _np
from collections import namedtuple as _nt
import itertools as _itertools

class AbstractSTScan():
    """For testing and verification.  Coordinates are as usual, but timestamps
    are just float values, with 0 being the end time, and e.g. 10 being 10
    units into the past.
    """
    def __init__(self, coords, timestamps):
        self.coords = _np.asarray(coords)
        self.timestamps = _np.asarray(timestamps)
        if len(self.timestamps) != self.coords.shape[1]:
            raise ValueError("Timestamps and Coordinates must be of same length.")
        self._sort_times_increasing()
        self._unique_points = self._make_unique_points()
        self.geographic_radius_limit = 100
        self.geographic_population_limit = 0.5
        self.time_max_interval = 28
        self.time_population_limit = 0.5
        self.only_full_disks = False

    def _sort_times_increasing(self):
        self.arg_sort = _np.argsort(self.timestamps)
        self.timestamps = self.timestamps[self.arg_sort]
        self.coords = self.coords[:,self.arg_sort]

    def allowed_times_into_past(self):
        """Find the times into the past which satisfy the constraints of
        maximum time interval, and maximum time population.
        
        :return: Array of times into the past, in increasing order.
        """
        mask = self.timestamps <= self.time_max_interval
        if not _np.any(mask):
            return []
        times = _np.unique(self.timestamps[mask])
        times.sort()
        index = len(times) - 1
        cutoff = times[index]
        maxsize = int(self.time_population_limit * self.timestamps.shape[0])
        while _np.sum(self.timestamps <= cutoff) > maxsize:
            index -= 1
            if index == -1:
                return []
            cutoff = times[index]
        return times[:index+1]

    Disc = _nt("Disc", ["centre", "radius_sq", "mask"])

    def _make_unique_points(self):
        """Return an array of the unique coordinates."""
        return _np.array(list(set((x,y) for x,y in self.coords.T))).T
    
    @staticmethod
    def _product(s):
        if len(s) == 1:
            yield s
        else:
            for i in range(1, len(s)+1):
                yield from _itertools.combinations(s, i)
    
    def all_discs_around(self, centre):
        """Find all discs around the centre.  Applies the rules above: no disc
        contains a single point, and the rule about boundary cases.
        
        Is a generator, yields pairs (radius**2, mask)
        """
        centre = _np.asarray(centre)
        limit = self.timestamps.shape[0] * self.geographic_population_limit

        if self.only_full_disks:
            distsqun = _np.sum((self.coords - centre[:,None])**2, axis=0)
            uniques = _np.unique(distsqun)
            uniques = uniques[uniques <= self.geographic_radius_limit**2]
            uniques.sort()
            for d in uniques:
                mask = distsqun <= d
                if _np.sum(mask) > limit:
                    return
                yield (d, mask)
            return
        
        distsqun = _np.sum((self._unique_points - centre[:,None])**2, axis=0)
        index_array = _np.arange(len(distsqun))
        uniques = _np.unique(distsqun)
        uniques = uniques[uniques <= self.geographic_radius_limit**2]
        uniques.sort()
        
        # Zero case
        mask = (self.coords[0] == centre[0]) & (self.coords[1] == centre[1])
        count = _np.sum(mask)
        if count > 1:
            if count > limit:
                return
            yield (0, mask)
        
        current_mask = mask
        for d in uniques[1:]:
            new_indices = index_array[distsqun == d]
            seen_too_large = False
            new_mask = current_mask.copy()
            for to_add in self._product(new_indices):
                mask = current_mask.copy()
                for i in to_add:
                    mask |= ( (self.coords[0] == self._unique_points[0][i]) &
                        (self.coords[1] == self._unique_points[1][i]) )
                new_mask |= mask
                if _np.sum(mask) > limit:
                    seen_too_large = True
                else:
                    yield (d, mask)
            if seen_too_large:
                return
            current_mask = new_mask
    
    def all_discs(self):
        """Generate all discs according to the rules.
        
        Is a generator, yielding Disc objects.
        """
        all_masks = set()
        for centre in self._unique_points.T:
            for rr, mask in self.all_discs_around(centre):
                m = tuple(mask)
                if m not in all_masks:
                    yield self.Disc(centre, rr, mask)
                    all_masks.add(m)

    Result = _nt("Result", ["centre", "radius", "mask", "time", "statistic"])
    
    def build_times_cutoff(self):
        """Returns pair (times, cutoff) where `times` is an array of all valid
        times into the past to test, in increasing order, and `cutoff[i]` is
        the greatest index, plus one, into `self.timestamps` whose value is
        `<= times[i]`.
        """
        times = self.allowed_times_into_past()
        cutoff = []
        i = 0
        for t in times:
            while i < self.timestamps.shape[0] and self.timestamps[i] <= t:
                i += 1
            if i == self.timestamps.shape[0]:
                cutoff.append(self.timestamps.shape[0])
            else:
                cutoff.append(i)
        return times, cutoff

    def score_clusters(self):
        """A generator returning triples `(disc, time, statistic)` describing
        each cluster of relevance.
        """
        N = self.timestamps.shape[0]
        times, cutoff = self.build_times_cutoff()
        for disc in self.all_discs():
            space_count = _np.sum(disc.mask) / N
            for c in cutoff:
                actual = _np.sum(disc.mask[:c])
                expected = space_count * c
                if actual > 1 and actual > expected:
                    yield (disc, self.timestamps[c-1],
                                      self._statistic(actual, expected, N))

    @staticmethod
    def _not_intersecting(all_clusters, cluster):
        return [cc for cc in all_clusters if 
                _np.sum((cc.centre - cluster.centre)**2) >
                (cluster.radius + cc.radius)**2 ]
    
    def find_all_clusters(self):
        """Find all the disjoint clusters from most to least significant."""
        all_clusters = [self.Result(centre = c[0].centre,
            radius = _np.sqrt(c[0].radius_sq),
            mask = c[0].mask,
            time = c[1],
            statistic = c[2]) for c in self.score_clusters()]
        all_clusters.sort(key = lambda r : -r.statistic)
        while len(all_clusters) > 0:
            c = all_clusters[0]
            yield c
            all_clusters = self._not_intersecting(all_clusters, c)
    
    @staticmethod
    def _statistic(actual, expected, total):
        """Calculate the log likelihood"""
        stat = actual * (_np.log(actual) - _np.log(expected))
        stat += (total - actual) * (_np.log(total - actual) - _np.log(total - expected))
        return stat

    def to_satscan(self, filename, offset):
        """Writes the training data to two SaTScan compatible files.  Does
        *not* currently write settings, so these will need to be entered
        manually.  The timestamps are rounded down to an integer.
        
        :param filename: Saves files "filename.geo" and "filename.cas"
          containing the geometry and "cases" repsectively.
        :param offset: The "end time" in generic units, from which the
          `timestamps` are subtracted.
        """
        self.write_to_satscan(filename, offset, self.coords, self.timestamps)

    @staticmethod
    def write_to_satscan(filename, offset, coords, timestamps):
        unique_coords = list(set( (x,y) for x,y in coords.T ))
        with open(filename + ".geo", "w") as geofile:
            for i, (x,y) in enumerate(unique_coords):
                print("{}\t{}\t{}".format(i+1, x, y), file=geofile)

        unique_times = list(set( t for t in timestamps ))
        with open(filename + ".cas", "w") as casefile:
            for i, (t) in enumerate(unique_times):
                pts = coords.T[timestamps == t]
                pts = [ (x,y) for x,y in pts ]
                import collections
                c = collections.Counter(pts)
                for pt in c:
                    index = unique_coords.index(pt)
                    print("{}\t{}\t{}".format(index+1, c[pt], int(offset - t)), file=casefile)


class SaTScanData():
    """Load and manipulate data in SaTScan format.  Currently assumes "generic
    time", i.e. time in integers.
    """
    def __init__(self, filename, time_end):
        self.time_end = time_end
        self.geo = { i : (x,y) for i,x,y in self._geo(filename)}
        self.cases = list(self._cases(filename))
    
    def to_coords_time(self):
        """Convert to the same format as for :class:`AbstractSTScan`"""
        times = []
        coords = []
        for i, c, t in self.cases:
            for _ in range(c):
                times.append(self.time_end - t)
                coords.append(self.geo[i])
        return _np.asarray(coords).T, _np.asarray(times)

    def _geo(self, filename):
        with open(filename + ".geo") as geofile:
            for row in geofile:
                i, x, y = row.split()
                yield int(i), float(x), float(y)

    def _cases(self, filename):
        with open(filename + ".cas") as casfile:
            for row in casfile:
                i, count, t = row.split()
                yield int(i), int(count), int(t)


class STScanNumpy():
    """For testing and verification; numpy accelerated.
    Coordinates are as usual, but timestamps
    are just float values, with 0 being the end time, and e.g. 10 being 10
    units into the past.
    """
    def __init__(self, coords, timestamps):
        self.coords = _np.asarray(coords)
        self.timestamps = _np.asarray(timestamps)
        if len(self.timestamps) != self.coords.shape[1]:
            raise ValueError("Timestamps and Coordinates must be of same length.")
        self._sort_times_increasing()
        self.geographic_radius_limit = 100
        self.geographic_population_limit = 0.5
        self.time_max_interval = 28
        self.time_population_limit = 0.5
        self._cache_N = 0
        
    def _sort_times_increasing(self):
        arg_sort = _np.argsort(self.timestamps)
        self.timestamps = self.timestamps[arg_sort]
        self.coords = self.coords[:,arg_sort]

    def make_time_ranges(self):
        """Compute the posssible time intervals.
        
        :return: Tuple of masks (of shape (N,k) where N is the number of data
          points), counts (of length k) and the cutoff used for each count (of
          length k).  Hence `masks[:,i]` corresponds to `count[i]` is given by
          looking at event `<= cutoff[i]` before the end of time.
        """
        unique_times = _np.unique(self.timestamps)
        unique_times = unique_times[unique_times <= self.time_max_interval]
        unique_times.sort()
        time_masks = self.timestamps[:,None] <= unique_times[None,:]
        
        limit = self.timestamps.shape[0] * self.time_population_limit
        time_counts = _np.sum(time_masks, axis=0)
        m = time_counts <= limit
        
        return time_masks[:,m], time_counts[m], unique_times[m]
    
    def find_discs(self, centre):
        """Compute the possible disks.
        
        :return: Tuple of masks (of shape (N,k) where N is the number of data
          points), counts (of length k) and the distances squared from the
        centre point (of length k).  Hence `masks[:,i]` corresponds to
        `count[i]` is given by looking at event `<= cutoff[i]` before the end
        of time.
        """
        centre = _np.asarray(centre)
        distsq = _np.sum( (self.coords - centre[:,None])**2, axis=0 )
        unique_dists = _np.unique(distsq)
        unique_dists = unique_dists[ unique_dists <= self.geographic_radius_limit**2 ]
        mask = distsq[:,None] <= unique_dists[None,:]
        
        limit = self.timestamps.shape[0] * self.geographic_population_limit
        space_counts = _np.sum(mask, axis=0)
        m = (space_counts > 1) & (space_counts <= limit)
        
        return mask[:,m], space_counts[m], unique_dists[m]

    @staticmethod
    def _calc_actual(space_masks, time_masks, time_counts):
        # Does this, but >9 times quicker:
        # uber_mask = space_masks[:,:,None] & time_masks[:,None,:]
        # actual = _np.sum(uber_mask, axis=0)
        x = _np.empty((space_masks.shape[1], time_masks.shape[1]))
        # This is better, but still >20 times slower...
        #for i, c in enumerate(time_counts):
        #    x[:,i] = _np.sum(space_masks[:c,:], axis=0)
        current_sum = _np.zeros(space_masks.shape[1])
        current_column = 0
        for i, c in enumerate(time_counts):
            while current_column < c:
                current_sum += space_masks[current_column,:]
                current_column += 1
            x[:,i] = current_sum
        return x

    def faster_score_all_new(self):
        """As :method:`score_all` but yields tuples (centre, distance_array,
        time_array, statistic_array)."""
        time_masks, time_counts, times = self.make_time_ranges()
        N = self.timestamps.shape[0]
        for centre in self.coords.T:
            space_masks, space_counts, dists = self.find_discs(centre)

            actual = self._calc_actual(space_masks, time_masks, time_counts)
            expected = space_counts[:,None] * time_counts[None,:] / N
            _mask = (actual > 1) & (actual > expected)
            actual = _np.ma.array(actual, mask=~_mask)
            expected = _np.ma.array(expected, mask=~_mask)
            stats = self._ma_statistic(actual, expected, N)
            _mask1 = _np.any(_mask, axis=1)
            if not _np.any(_mask1):
                continue
            m = _np.ma.argmax(stats, axis=1)[_mask1]
            stats = stats[_mask1,:]
            stats = stats[range(stats.shape[0]),m].data
            used_dists = dists[_mask1]
            used_times = times[m]

            yield centre, used_dists, used_times, stats

    @staticmethod
    def _ma_statistic(actual, expected, total):
        """Calculate the log likelihood"""
        stat = actual * (_np.ma.log(actual) - _np.ma.log(expected))
        stat += (total - actual) * (_np.ma.log(total - actual) - _np.ma.log(total - expected))
        return stat

    @staticmethod
    def _calc_actual1(space_masks, time_masks, time_counts):
        x = _np.empty((space_masks.shape[1], time_masks.shape[1]), dtype=_np.int)
        current_sum = _np.zeros(space_masks.shape[1], dtype=_np.int)
        current_column = 0
        for i, c in enumerate(time_counts):
            while current_column < c:
                current_sum += space_masks[current_column,:]
                current_column += 1
            x[:,i] = current_sum
        return x

    def faster_score_all(self):
        """As :method:`score_all` but yields tuples (centre, distance_array,
        time_array, statistic_array)."""
        time_masks, time_counts, times = self.make_time_ranges()
        N = self.timestamps.shape[0]
        for centre in self.coords.T:
            space_masks, space_counts, dists = self.find_discs(centre)

            actual = self._calc_actual1(space_masks, time_masks, time_counts)
            
            stcounts = space_counts[:,None] * time_counts[None,:]
            _mask = (actual > 1) & (N * actual > stcounts)
            stats = self._ma_statistics_lookup(space_counts, time_counts, stcounts, actual, _mask, N)
            _mask1 = _np.any(_mask, axis=1)
            if not _np.any(_mask1):
                continue
            m = _np.ma.argmax(stats, axis=1)[_mask1]
            stats = stats[_mask1,:]
            stats = stats[range(stats.shape[0]),m].data
            used_dists = dists[_mask1]
            used_times = times[m]

            yield centre, used_dists, used_times, stats

    @staticmethod
    def _build_log_lookup(N):
        lookup = _np.empty(N+1, dtype=_np.float64)
        lookup[0] = 1
        for i in range(1, N+1):
            lookup[i] = i
        return _np.log(lookup)

    def _ma_statistics_lookup(self, space_counts, time_counts, stcounts, actual, _mask, N):
        # Faster version which uses lookup tables
        if self._cache_N != N:
            self._cache_N = N
            self._log_lookup = self._build_log_lookup(N)
            if N > 2000:
                self._log_lookup2 = None
            else:
                self._log_lookup2 = self._build_log_lookup(N*N)
        sl = self._log_lookup[space_counts]
        tl = self._log_lookup[time_counts]
        y = actual * (self._log_lookup[actual] - sl[:,None] - tl[None,:])
        if self._log_lookup2 is None:
            yy = (N-actual) * (self._log_lookup[N-actual] - _np.log(N*N-stcounts))
        else:
            yy = (N-actual) * (self._log_lookup[N-actual] - self._log_lookup2[N*N-stcounts])
        return _np.ma.array(y + yy + N*_np.log(N), mask=~_mask)

    def faster_score_all_old(self):
        """As :method:`score_all` but yields tuples (centre, distance_array,
        time_array, statistic_array)."""
        time_masks, time_counts, times = self.make_time_ranges()
        N = self.timestamps.shape[0]
        for centre in self.coords.T:
            space_masks, space_counts, dists = self.find_discs(centre)

            uber_mask = space_masks[:,:,None] & time_masks[:,None,:]
        
            actual = _np.sum(uber_mask, axis=0)
            expected = space_counts[:,None] * time_counts[None,:] / N
            _mask = (actual > 1) & (actual > expected)

            used_dists = _np.broadcast_to(dists[:,None], _mask.shape)[_mask]
            used_times = _np.broadcast_to(times[None,:], _mask.shape)[_mask]
            actual = actual[_mask]
            expected = expected[_mask]
            stats = AbstractSTScan._statistic(actual, expected, N)

            if len(stats) > 0:
                yield centre, used_dists, used_times, stats

    def score_all(self):
        """Consider all possible space and time regions (which may include many
        essentially repeated disks) and yield tuples of the centre of disk, the
        radius squared of the disk, the time span of the region, and the 
        statistic.
        """
        for centre, dists, times, stats in self.faster_score_all():
            for d,t,s in zip(dists, times, stats):
                yield centre, d, t, s

    @staticmethod
    def _not_intersecting(scores, centre, radius):
        return [cc for cc in scores if 
                (cc[0] - centre[0])**2 + (cc[1] - centre[1])**2
                > (radius + cc[2])**2]

    Result = _nt("Result", ["centre", "radius", "time", "statistic"])
        
    def find_all_clusters(self):
        scores = []
        count = 0
        for centre, dists, times, stats in self.faster_score_all():
            dists = _np.sqrt(dists)
            scores.extend(zip(_itertools.repeat(centre[0]),
                            _itertools.repeat(centre[1]), dists, times, stats))
            count += 1
        if len(scores) == 0:
            return
        scores = _np.asarray(scores)
        if len(scores.shape) == 1:
            scores = scores[None,:]
        scores = scores[_np.argsort(-scores[:,4]), :]

        while scores.shape[0] > 0:
            best = scores[0]
            centre = _np.asarray([best[0],best[1]])
            radius = best[2]
            yield self.Result(centre = centre, radius = radius,
                              time = best[3], statistic = best[4])
            distances = (scores[:,0] - best[0])**2 + (scores[:,1] - best[1])**2
            mask = distances > (radius + scores[:,2]) ** 2
            scores = scores[mask,:]
