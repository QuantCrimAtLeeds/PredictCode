"""
stscan2
~~~~~~~

Work in progress to try to make the SaTScan code faster, and conform more
exactly to what SaTScan does.

- A "cluster" never consists of just one event.
- Discs are also centred on actual events.
- If the boundary of a disc contains more than one event, then we test *all*
  (to be verified...) possibilities of including / excluding events on the
  boundary
    - So, for example, if we have a disc centred at (1,1) and events at
      (0,0), (1,1) and (2,2) then we would consider the discs containing events
      {0,1}, {1,2} and {0,1,2}.  More naively, we'd only look at {0,1,2}.
     
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
        
    def _sort_times_increasing(self):
        args = _np.argsort(self.timestamps)
        self.timestamps = self.timestamps[args]
        self.coords = self.coords[:,args]
        
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
        distsqun = _np.sum((self._unique_points - centre[:,None])**2, axis=0)
        index_array = _np.arange(len(distsqun))
        uniques = _np.unique(distsqun)
        uniques = uniques[uniques <= self.geographic_radius_limit**2]
        uniques.sort()
        
        # Zero case
        mask = (self.coords[0] == centre[0]) & (self.coords[1] == centre[1])
        count = _np.sum(mask)
        limit = self.timestamps.shape[0] * self.geographic_population_limit
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
        manually.  The timestamps
        
        :param filename: Saves files "filename.geo" and "filename.cas"
          containing the geometry and "cases" repsectively.
        :param offset: The "end time" in generic units, from which the
          `timestamps` are subtracted.
        """
        unique_coords = list(set( (x,y) for x,y in self.coords.T ))
        with open(filename + ".geo", "w") as geofile:
            for i, (x,y) in enumerate(unique_coords):
                print("{}\t{}\t{}".format(i+1, x, y), file=geofile)

        unique_times = list(set( t for t in self.timestamps ))
        with open(filename + ".cas", "w") as casefile:
            for i, (t) in enumerate(unique_times):
                pts = self.coords.T[self.timestamps == t]
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
        