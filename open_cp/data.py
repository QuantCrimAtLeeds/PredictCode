"""Encapsulates input data."""

import numpy as _np

class Point():
    def __init__(self, x=0, y=0):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __repr__(self):
        return "Point({},{})".format(self.x, self.y)


class RectangularRegion():
    def __init__(self, xmin=0, xmax=1, ymin=0, ymax=1):
        self._min = Point(xmin, ymin)
        self._max = Point(xmax, ymax)

    @property
    def xmin(self):
        return self._min.x

    @property
    def xmax(self):
        return self._max.x

    @property
    def ymin(self):
        return self._min.y

    @property
    def ymax(self):
        return self._max.y

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def aspect_ratio(self):
        if self.xmax == self.xmin:
            return _np.nan
        return (self.ymax - self.ymin) / (self.xmax - self.xmin)

    def __add__(self, other):
        return RectangularRegion(xmin = self.xmin + other.x,
                                 xmax = self.xmax + other.x,
                                 ymin = self.ymin + other.y,
                                 ymax = self.ymax + other.y)

    def __repr__(self):
        return "RectangularRegion( ({},{}) -> ({},{}) )".format(self.xmin,
                                 self.ymin, self.xmax, self.ymax)


class TimedPoints:
    """Stores a list of timestamped x-y coordinates of events"""

    @staticmethod
    def _is_time_ordered(timestamps):
        it = iter(timestamps)
        prev = next(it)
        for time in it  :
            if prev > time:
                return False
            prev = time
        return True

    def _assert_times_ordered(self, timestamps):
        if not self._is_time_ordered(timestamps):
            raise ValueError("Input must be time ordered")

    def __init__(self, timestamps, coords):
        self._assert_times_ordered(timestamps)
        self.timestamps = _np.array(timestamps, dtype="datetime64[ms]")
        self.coords = _np.array(coords).astype(_np.float64)
        if len(self.coords.shape) != 2 or self.coords.shape[0] != 2:
            raise Exception("Coordinates should be of shape (2,#)")
        if len(self.timestamps) != self.coords.shape[1]:
            raise Exception("Input data should all be of the same length")

    @property
    def xcoords(self):
        return self.coords[0]

    @property
    def ycoords(self):
        return self.coords[1]
        
    def __getitem__(self, index):
        if isinstance(index, int):
            return [self.timestamps[index], *self.coords[:, index]]
        # Assume slice like object
        new_times = self.timestamps[index]
        new_coords = self.coords[:,index]
        if self._is_time_ordered(new_times):
            return TimedPoints(new_times, new_coords)
        data = [(t,x,y) for t, (x,y) in zip(new_times, new_coords.T)]
        data.sort(key = lambda triple : triple[0])
        for i, (t,x,y) in enumerate(data):
            new_times[i] = t
            new_coords[0,i] = x
            new_coords[1,i] = y
        return TimedPoints(new_times, new_coords)

    def events_before(self, cutoff_time):
        mask = self.timestamps <= cutoff_time
        return TimedPoints(self.timestamps[mask], self.coords[:,mask])

    def bounding_box(self):
        return RectangularRegion(xmin = _np.min(self.xcoords),
            xmax = _np.max(self.xcoords), ymin = _np.min(self.ycoords),
            ymax = _np.max(self.ycoords))

    def time_range(self):
        return ( self.timestamps[0], self.timestamps[-1] )

    @classmethod
    def from_coords(cls, timestamps, xcoords, ycoords):
        lengths = { len(timestamps), len(xcoords), len(ycoords) }
        if len(lengths) != 1:
            raise Exception("Input data should all be of the same length")
        return cls(timestamps,_np.stack([xcoords, ycoords]))


try:
    import pyproj as _proj
except ModuleNotFoundError:
    import sys
    print("Package 'pyproj' not found: projection methods will not be supported.", file=sys.stderr)
    _proj = None

# http://spatialreference.org/ref/epsg/
# 7405 is suitable for UK
def points_from_lon_lat(points, proj=None, epsg=None):
    if not _proj:
        return points
    if not proj:
        if not epsg:
            raise Exception("Need to provide one of 'proj' object or 'epsg' code")
        proj = _proj.Proj({"init": "epsg:"+str(epsg)})
    
    transformed = _np.empty(points.coords.shape)
    for i in range(len(points.timestamps)):
        transformed[0][i], transformed[1][i] = proj(points.xcoords[i], points.ycoords[i])
    
    return TimedPoints(points.timestamps, transformed)
