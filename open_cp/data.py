"""Encapsulates input data."""

import numpy as _np
import datetime as _datetime

class Point():
    """A simple 2 dimensional point class.
    
    Is "iterable" and returns (x,y).  Similarly supports indexing."""
    def __init__(self, x=0, y=0):
        self._x = x
        self._y = y

    @property
    def x(self):
        """The x coordinate."""
        return self._x

    @property
    def y(self):
        """The y coordinate."""
        return self._y

    def __iter__(self):
        yield self.x
        yield self.y
        
    def __getitem__(self, i):
        if i == 0:
            return self.x
        if i == 1:
            return self.y
        raise ValueError("Index must be 0 or 1.")

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __repr__(self):
        return "Point({},{})".format(self.x, self.y)

    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return tuple(self) == tuple(other)


class RectangularRegion():
    """Stores a rectangular region."""
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
    def xrange(self):
        """The pair (xmin, xmax)"""
        return (self.xmin, self.xmax)

    @property
    def yrange(self):
        """The pair (ymin, ymax)"""
        return (self.ymin, self.ymax)

    @property
    def min(self):
        """The pair (xmin, ymin)"""
        return self._min

    @property
    def max(self):
        """The pair (xmax, ymax)"""
        return self._max

    @property
    def width(self):
        """The width of the region: xmax - xmin"""
        return self.xmax - self.xmin

    @property
    def height(self):
        """The height of the region: ymax - ymin"""
        return self.ymax - self.ymin

    @property
    def aspect_ratio(self):
        """Height divided by width"""
        if self.width == 0:
            return _np.nan
        return self.height / self.width

    def __add__(self, other):
        return RectangularRegion(xmin = self.xmin + other.x,
                                 xmax = self.xmax + other.x,
                                 ymin = self.ymin + other.y,
                                 ymax = self.ymax + other.y)

    def grid_size(self, cell_width, cell_height = None):
        """Return the size of grid defined by this region.

        :param cell_width: The width of each cell in the grid.
        :param cell_height: Optional.  The height of each cell in the grid;
         defaults to a square grid where the height is the same as the width.

        :return: (xsize, ysize) of the grid.
        """
        if cell_height is None:
            cell_height = cell_width
        xsize = int(_np.ceil((self.xmax - self.xmin) / cell_width))
        ysize = int(_np.ceil((self.ymax - self.ymin) / cell_height))
        return xsize, ysize

    def __eq__(self, other):
        return self.min == other.min and self.max == other.max

    def __iter__(self):
        return iter((self.xmin, self.ymin, self.xmax, self.ymax))

    def __repr__(self):
        return "RectangularRegion( ({},{}) -> ({},{}) )".format(self.xmin,
                                 self.ymin, self.xmax, self.ymax)


class Grid():
    """Stores details of a rectangular grid.

    :param xsize: Width of each grid cell.
    :param ysize: Height of each grid cell.
    :param xoffset: The x coordinate of the right side of grid cell (0,0).
    :param yoffset: The y coordinate of the bottom side of grid cell (0,0).
    """
    def __init__(self, xsize, ysize, xoffset, yoffset):
        self._xoffset = xoffset
        self._yoffset = yoffset
        self._xsize = xsize
        self._ysize = ysize

    @property
    def xsize(self):
        """The width of each cell"""
        return self._xsize

    @property
    def ysize(self):
        """The height of each cell"""
        return self._ysize
    
    @property
    def xoffset(self):
        """The x coordinate of the left side of the grid."""
        return self._xoffset
    
    @property
    def yoffset(self):
        """The y coordinate of the bottom side of the grid."""
        return self._yoffset

    def __repr__(self):
        return "Grid(offset=({},{}), size={}x{})".format(self.xoffset,
                self.yoffset, self.xsize, self.ysize)

    def grid_coord(self, x, y):
        """Where does the point fall in the grid.
        
        :param x: x coordinate
        :param y: y coordinate
        
        :return: `(gridx, gridy)` coordinates in the grid where this point
          falls.
        """
        xx = _np.asarray(x) - self.xoffset
        yy = _np.asarray(y) - self.yoffset
        return (_np.floor(xx / self.xsize).astype(_np.int), _np.floor(yy / self.ysize).astype(_np.int))

    def bounding_box_of_cell(self, gx, gy):
        """Return the bounding box of the cell.

        :param gx: x coordinate of the cell
        :param gy: y coordinate of the cell

        :return: A :class:`RectangularRegion` giving the (xmin,ymin) and
          (xmax,ymax) coordinates of the cell.
        """
        return RectangularRegion(xmin = self.xoffset + gx * self.xsize,
            xmax = self.xoffset + (gx + 1) * self.xsize,
            ymin = self.yoffset + gy * self.ysize,
            ymax = self.yoffset + (gy + 1) * self.ysize)


class BoundedGrid(Grid):
    """Abstract base class for a :class:`Grid` object which has an "extent":
    only cells in rectangle based at `(0,0)` have meaning.
    """
    def __init__(self, xsize, ysize, xoffset, yoffset):
        super().__init__(xsize, ysize, xoffset, yoffset)

    @property
    def xextent(self):        
        """The width of the grid area."""
        raise NotImplementedError()

    @property
    def yextent(self):        
        """The height of the grid area."""
        raise NotImplementedError()

    def region(self):
        """Returns the :class:`RectangularRegion` defined by the grid and its
        extent.
        """
        return RectangularRegion(xmin = self.xoffset, ymin = self.yoffset,
            xmax = self.xoffset + self.xextent * self.xsize,
            ymax = self.yoffset + self.yextent * self.ysize)


class MaskedGrid(BoundedGrid):
    """A rectangular grid of finite extent where some cells may be "masked" or
    "invalid".  Valid cells are always in a range from `(0,0)` to
    `(xextent - 1, yextent - 1)` inclusive.

    :param xsize: Width of each grid cell.
    :param ysize: Height of each grid cell.
    :param xoffset: The x coordinate of the right side of grid cell (0,0).
    :param yoffset: The y coordinate of the bottom side of grid cell (0,0).
    :param mask: An array-like object of shape (yextent, xextent) which can be
      converted to booleans.  We follow the numpy masking convention, and if a
      cell is "masked" then it is "invalid".
    """
    def __init__(self, xsize, ysize, xoffset, yoffset, mask):
        super().__init__(xsize, ysize, xoffset, yoffset)
        self._mask = _np.asarray(mask).astype(_np.bool)

    def __repr__(self):
        return "MaskedGrid(offset=({},{}), size={}x{}, mask region={}x{})".format(
                self.xoffset, self.yoffset, self.xsize, self.ysize, self.xextent,
                self.yextent)

    @property
    def mask(self):
        """The mask"""
        return self._mask

    @property
    def xextent(self):
        """The width of the masked grid area."""
        return self.mask.shape[1]

    @property
    def yextent(self):
        """The height of the masked grid area."""
        return self.mask.shape[0]

    def is_valid(self, gx, gy):
        """Is the grid cell `(gx, gy)` valid?"""
        if gx < 0 or gy < 0 or gx >= self.mask.shape[1] or gy >= self.mask.shape[0]:
            raise ValueError("Coordinates ({},{}) out of range for mask.", gx, gy)
        return not self.mask[gy][gx]

    @staticmethod
    def from_grid(grid, mask):
        """Static constructor from a :class:`Grid` instance."""
        return MaskedGrid(grid.xsize, grid.ysize, grid.xoffset, grid.yoffset, mask)

    def mask_matrix(self, matrix):
        """Return a `numpy` "masked array" from the matrix, and this class's
        mask.

        :param matrix: An array like object of the same shape as the mask, i.e.
          (yextent, xextent).
        """
        return _np.ma.masked_array(matrix, self.mask)


def order_by_time(timestamps, xcoords, ycoords):
    """Reorder the timestamps so they are increasing, and reorder the coords in
    the same way (so the timestamps and coordinates continue to be associated
    in the same way).
    
    :param timestamps: Array-like object of timestamps
    :param xcoords: Array-like object of x coordinates.
    :param ycoords: Array-like object of y coordinates.
    
    :return: Triple of `(timestamps, xcoords, ycoords)`.
    """
    timestamps = _np.asarray(timestamps)
    xcoords, ycoords = _np.asarray(xcoords), _np.asarray(ycoords)
    args = _np.argsort(timestamps)
    return timestamps[args], xcoords[args], ycoords[args]


class TimeStamps():
    """Base class for e.g. :class:`TimedPoints` which stores timestamps only.

    :param timestamps: An array of timestamps (must be convertible to
      :class:`numpy.datetime64`).
    """
    def __init__(self, timestamps):
        self._assert_times_ordered(timestamps)
        self._timestamps = _np.array(timestamps, dtype="datetime64[ms]")

    def _assert_times_ordered(self, timestamps):
        if not self._is_time_ordered(timestamps):
            raise ValueError("Input must be time ordered")

    @staticmethod
    def _is_time_ordered(timestamps):
        if len(timestamps) == 0:
            return True
        it = iter(timestamps)
        prev = next(it)
        for time in it  :
            if prev > time:
                return False
            prev = time
        return True

    @property
    def timestamps(self):
        """Array of timestamps, as :class:`numpy.datetime64` objects."""
        return self._timestamps

    @property
    def time_range(self):
        """Find the time range.

        :return: A pair (start, end) of timestamps.
        """
        return ( self.timestamps[0], self.timestamps[-1] )

    def time_deltas(self, time_unit = _np.timedelta64(1, "m")):
        """Returns a numpy array of floats, converted from the timestamps,
        starting from 0, and with the optional unit.

        :param time_unit: The unit to measure time by.  Defaults to 1 minute,
          so timestamps an hour apart will be converted to floats 60.0 apart.
          No rounding occurs, so there is no loss in accuracy by passing a
          different time unit.
        """
        return ( self.timestamps - self.timestamps[0] ) / time_unit

    def times_datetime(self):
        """Return an array of timestamps using the :class:`datetime.datetime`
        standard library class.  Useful for plotting with matplotlib, for
        example.
        """
        return self.timestamps.astype(_datetime.datetime)

    def bin_timestamps(self, offset, bin_length):
        """Return a new instance of :class:`TimeStamps` where each timestamp
        is adjusted.  Any timestamp between `offset` and `offset + bin_length`
        is mapped to `offset`; timestamps between `offset + bin_length` and
        `offset + 2 * bin_length` are mapped to `offset + bin_length`, and so
        forth.
        
        :param offset: A datetime-like object which is the start of the
          binning.
        :param bin_length: A timedelta-like object which is the length of each
          bin.
          
        :return: New instance of :class:`TimeStamps`.
        """
        offset = _np.datetime64(offset)
        bin_length = _np.timedelta64(bin_length)
        new_times = _np.floor((self._timestamps - offset) / bin_length)
        new_times = offset + new_times * bin_length
        return TimeStamps(new_times)


class TimedPoints(TimeStamps):
    """Stores a list of timestamped x-y coordinates of events.
    
    :param timestamps: An array of timestamps (must be convertible to
      :class:`numpy.datetime64`).
    :param coords: An array of shape (2,n) where `n` must match the number of
      timestamps.
    """
    def __init__(self, timestamps, coords):
        super().__init__(timestamps)
        self.coords = _np.array(coords).astype(_np.float64)
        if len(self.coords.shape) != 2 or self.coords.shape[0] != 2:
            raise Exception("Coordinates should be of shape (2,#)")
        if len(self.timestamps) != self.coords.shape[1]:
            raise Exception("Input data should all be of the same length")

    @property
    def xcoords(self):
        """A one dimensional array representing the x coordinates of events."""
        return self.coords[0]

    @property
    def ycoords(self):
        """A one dimensional array representing the y coordinates of events."""
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

    def events_before(self, cutoff_time=None):
        """Returns a new instance with just the events with timestamps before
        (or equal to) the cutoff.

        :param cutoff_time: End of the time period we're interested in.
          Default is `None` which means return all the data.
        """
        if cutoff_time is None:
            return self
        mask = self.timestamps <= _np.datetime64(cutoff_time)
        return TimedPoints(self.timestamps[mask], self.coords[:,mask])

    @property
    def empty(self):
        """True or False, do we have any events"""
        return len(self.timestamps) == 0
    
    @property
    def number_data_points(self):
        """The number of events"""
        return len(self.timestamps)

    @property
    def bounding_box(self):
        """The smallest (space) box containing all the data points.

        :return: A :class:`RectangularRegion` instance.
        """
        return RectangularRegion(xmin = _np.min(self.xcoords),
            xmax = _np.max(self.xcoords), ymin = _np.min(self.ycoords),
            ymax = _np.max(self.ycoords))

    def to_time_space_coords(self, time_unit = _np.timedelta64(1, "m")):
        """Returns a single numpy array `[t,x,y]` where the time stamps are
        converted to floats, starting from 0, and with the optional unit.

        :param time_unit: The unit to measure time by.  Defaults to 1 minute,
          so timestamps an hour apart will be converted to floats 60.0 apart.
          No rounding occurs, so there is no loss in accuracy by passing a
          different time unit.
        """
        times = self.time_deltas(time_unit)
        return _np.vstack([times, self.xcoords, self.ycoords])

    @staticmethod
    def from_coords(timestamps, xcoords, ycoords):
        """Static constructor allowing you to pass separate arrays of x and y
        coordinates.  Also allows `timestamps` to be unorderd: all data will
        be sorted first.
        """
        lengths = { len(timestamps), len(xcoords), len(ycoords) }
        if len(lengths) != 1:
            raise Exception("Input data should all be of the same length")
        timestamps = _np.asarray(timestamps)
        indices = _np.argsort(timestamps)
        timestamps = timestamps[indices]
        xcoords = _np.asarray(xcoords)[indices]
        ycoords = _np.asarray(ycoords)[indices]
        return TimedPoints(timestamps, _np.stack([xcoords, ycoords]))

    def bin_timestamps(self, offset, bin_length):
        """Return a new instance of :class:`TimedPoints` where each timestamp
        is adjusted.  Any timestamp between `offset` and `offset + bin_length`
        is mapped to `offset`; timestamps between `offset + bin_length` and
        `offset + 2 * bin_length` are mapped to `offset + bin_length`, and so
        forth.
        
        :param offset: A datetime-like object which is the start of the
          binning.
        :param bin_length: A timedelta-like object which is the length of each
          bin.
          
        :return: New instance of :class:`TimedPoints`.
        """
        new_times = super().bin_timestamps(offset, bin_length).timestamps
        return TimedPoints(new_times, self.coords)


try:
    import pyproj as _proj
except ImportError:
    import sys
    print("Package 'pyproj' not found: projection methods will not be supported.", file=sys.stderr)
    _proj = None

def points_from_lon_lat(points, proj=None, epsg=None):
    """Converts longitude / latitude data into x,y coordinates using a
    projection.  The module `pyproj` must be loaded, otherwise this does
    nothing.

    :param points: A :class TimedPoints: instance of lon/lat data.
    :param proj: Optionally, a :class:`pyproj.Proj` instance describing the
      projection.
    :param epsg: If no `proj` is given, this must be supplied.  A valid EPSG
      projection reference.  For example, 7405 is suitable for UK data. See
      http://spatialreference.org/ref/epsg/

    :return: A :class:`TimedPoints` instance of projected data with the same timestamps.
    """
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
