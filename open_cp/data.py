"""Encapsulates input data."""

import numpy as _np

class TimedPoints:
    """Stores a list of timestamped x-y coordinates of events"""

    def _assert_times_ordered(self, timestamps):
        it = iter(timestamps)
        prev = next(it)
        for time in it  :
            if prev > time:
                raise ValueError("Input must be time ordered")
            prev = time

    def __init__(self, timestamps, coords):
        if len(timestamps) != len(coords):
            raise Exception("Input data should all be of the same length")
        self._assert_times_ordered(timestamps)
        self.timestamps = _np.array(timestamps, dtype="datetime64[ms]")
        self.coords = _np.array(coords).astype(_np.float64)

    def __getitem__(self, index):
        return [self.timestamps[index], self.coords[index][0], self.coords[index][1]]

    def events_before(self, cutoff_time):
        mask = self.timestamps <= cutoff_time
        return TimedPoints(self.timestamps[mask], self.coords[mask])

    @classmethod
    def from_coords(cls, timestamps, xcoords, ycoords):
        lengths = { len(timestamps), len(xcoords), len(ycoords) }
        if len(lengths) != 1:
            raise Exception("Input data should all be of the same length")
        return cls(timestamps,_np.stack([xcoords, ycoords], axis=1))


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

    def __add__(self, other):
        return RectangularRegion(xmin = self.xmin + other.x,
                                 xmax = self.xmax + other.x,
                                 ymin = self.ymin + other.y,
                                 ymax = self.ymax + other.y)

    def __repr__(self):
        return "RectangularRegion( ({},{}) -> ({},{}) )".format(self.xmin,
                                 self.ymin, self.xmax, self.ymax)


import pyproj as _proj

def points_from_lon_lat():
    pass