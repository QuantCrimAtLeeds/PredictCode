"""
geometry
~~~~~~~~

Methods to help with geometry work.  Uses `shapely`.
"""

import numpy as _np
from . import data as _data
import logging as _logging
import numpy as _np

_logger = _logging.getLogger(__name__)


try:
    import shapely.geometry as _geometry
except Exception:
    _logger.error("Failed to import `shapely`.")
    _geometry = None

def configure_gdal():
    """On windows, I have found that by default, the GDAL_DATA environment
    variable is not set.  One solution is to always use the (for example)
    Anaconda Prompt instead of the usual Command Prompt.  Another is to
    correctly set the variable programmatically, which is what this function
    does.  You can tell if this is a problem by noticing the message:

      > ERROR 4: Unable to open EPSG support file gcs.csv.
      > Try setting the GDAL_DATA environment variable to point to the
      > directory containing EPSG csv files.

    Appearing on stderr when you use e.g. geopandas.
    """
    import os, sys
    if "GDAL_DATA" in os.environ:
        _logger.debug("GDAL_DATA already set so nothing to do.")
        return
    _logger.info("GDAL_DATA not set, so searching...")
    if sys.platform.startswith("linux"):
        _logger.info("However, platform is linux, so assuming we'll be okay...")
        return
    choices = _find_gdal_choices()
    if len(choices) == 1:
        _logger.info("Set GDAL_DATA = '%s'", choices[0])
        os.environ["GDAL_DATA"] = choices[0]
    else:
        _logger.error("Found too many choices for setting GDAL_DATA: %s", str(choices))

def _find_gdal_choices():
    import os, sys
    choices = []
    for path, _, _ in os.walk(sys.exec_prefix):
        if path.endswith("gdal"):
            choices.append(path)

    library_choices = [x for x in choices if x.lower().find("library") > -1
        and x.lower().find("pkgs") == -1 and _contains_csv(x)]
    if len(library_choices) == 1:
        return library_choices
    return choices

def _contains_csv(path):
    import os
    csvs = [x for x in os.listdir(path) if x.endswith(".csv")]
    return len(csvs) > 1

def grid_intersection(geometry, grid):
    """Find the collection of grid cells which intersect with the geometry.
    Here "intersect" means "intersects with non-zero area", so grid cells just
    touching the geometry will not be returned.

    :param geometry: Geometry object to intersect with.
    :param grid: Instance of :class:`Grid` describing the grid.

    :return: List of pairs (x,y) of grid cells which intersect.
    """
    minx, miny, maxx, maxy = geometry.bounds
    xstart = int(_np.floor((minx - grid.xoffset) / grid.xsize))
    xend = int(_np.floor((maxx - grid.xoffset) / grid.xsize))
    ystart = int(_np.floor((miny - grid.yoffset) / grid.ysize))
    yend = int(_np.floor((maxy - grid.yoffset) / grid.ysize))

    intersections = []
    for y in range(ystart, yend + 1):
        yy = grid.yoffset + y * grid.ysize
        for x in range(xstart, xend + 1):
            xx = grid.xoffset + x * grid.xsize
            poly = _geometry.Polygon([[xx, yy], [xx + grid.xsize, yy],
                    [xx + grid.xsize, yy + grid.ysize], [xx, yy + grid.ysize]])
            poly = poly.intersection(geometry)
            if not poly.is_empty and poly.area > 0:
                intersections.append((x, y))
    return intersections

def mask_grid_by_intersection(geometry, grid):
    """Generate a :class:`MaskedGrid` by intersecting the grid with the
    geometry.  The returned grid may have a different x/y offset, so that it
    can contain all grid cells which intersect with the geometry.  However,
    the "relative offset" will be unchanged (so that the difference between the
    x offsets will be a multiple of the grid width, and the same for y).

    :param geometry: Geometry object to intersect with.
    :param grid: The :class:`Grid` instance describing the grid.
    """
    minx, miny, maxx, maxy = geometry.bounds
    xstart = int(_np.floor((minx - grid.xoffset) / grid.xsize))
    xend = int(_np.floor((maxx - grid.xoffset) / grid.xsize))
    ystart = int(_np.floor((miny - grid.yoffset) / grid.ysize))
    yend = int(_np.floor((maxy - grid.yoffset) / grid.ysize))
    width = xend - xstart + 1
    height = yend - ystart + 1

    mask = _np.empty((height, width), dtype=_np.bool)
    xo = grid.xoffset + xstart * grid.xsize
    yo = grid.yoffset + ystart * grid.ysize
    import shapely.prepared
    geo = shapely.prepared.prep(geometry)
    for y in range(height):
        yy = yo + y * grid.ysize
        polys = [_geometry.Polygon([[xo + x * grid.xsize, yy],
                    [xo + x * grid.xsize + grid.xsize, yy],
                    [xo + x * grid.xsize + grid.xsize, yy + grid.ysize],
                    [xo + x * grid.xsize, yy + grid.ysize]])
                for x in range(width)]
        mask[y] = _np.asarray([not geo.intersects(poly) for poly in polys])
    
    return _data.MaskedGrid(grid.xsize, grid.ysize, xo, yo, mask)

def mask_grid_by_points_intersection(timed_points, grid, bbox=False):
    """Generate a :class:`MaskedGrid` by intersecting the grid with collection
    of points.

    :param timed_points: Instance of :class:`TimedPoints` (or other object with
      `xcoords` and `ycoords` attributes).
    :param grid: The :class:`Grid` instance describing the grid.
    :param bbox: If `True` then return the smallest rectangle containing the
      points.  If `False` then just return the grid cells which contain at
      least once point.
    """
    xcs = _np.asarray(timed_points.xcoords)
    ycs = _np.asarray(timed_points.ycoords)
    minx, maxx = _np.min(xcs), _np.max(xcs)
    miny, maxy = _np.min(ycs), _np.max(ycs)
    xstart = int(_np.floor((minx - grid.xoffset) / grid.xsize))
    xend = int(_np.floor((maxx - grid.xoffset) / grid.xsize))
    ystart = int(_np.floor((miny - grid.yoffset) / grid.ysize))
    yend = int(_np.floor((maxy - grid.yoffset) / grid.ysize))
    width = xend - xstart + 1
    height = yend - ystart + 1

    mask = _np.zeros((height, width), dtype=_np.bool)
    xo = grid.xoffset + xstart * grid.xsize
    yo = grid.yoffset + ystart * grid.ysize
    if not bbox:
        def intersect(xx, yy):
            mask = ( (xcs >= xx) & (ycs >= yy)
                & (xcs <= (xx+grid.xsize)) & (ycs <= (yy+grid.ysize)) )
            return _np.any(mask)
        for y in range(height):
            yy = yo + y * grid.ysize
            for x in range(width):
                xx = xo + x * grid.xsize
                if not intersect(xx, yy):
                    mask[y][x] = True
    
    return _data.MaskedGrid(grid.xsize, grid.ysize, xo, yo, mask)

def intersect_timed_points(timed_points, geo):
    """Intersect the :class:`TimedPoints` data with the geometry, using
    `shapely`.
    
    :param timed_points: Instance of :class:`TimedPoints`
    :param geo: A geometry object
    
    :return: Instance of :class:`TimedPoints`
    """
    mask = []
    for (x,y) in timed_points.coords.T:
        pt = _geometry.Point((x,y))
        mask.append( geo.intersects(pt) )
    mask = _np.array(mask, dtype=_np.bool)
    return _data.TimedPoints(timed_points.timestamps[mask],
                       timed_points.coords[:,mask])
    
    
    
##### Point and line geometry #####

def _project_point_to_line(point, line):
    """Assumes line is only 2 points
    """
    v = line[1] - line[0]
    x = point - line[0]
    t = _np.dot(x, v) / _np.dot(v, v)
    if t <= 0:
        return line[0]
    if t >= 1:
        return line[1]
    return line[0] + t * v

def project_point_to_line(point, line):
    """Find the closest point on the line segment to the point.
    
    :param point: Pair `(x,y)`(
    :param line: A single linear segment, `[ [x_1,y_1], [x_2,y_2], ...,
      [x_n,y_n] ]`.  This ordering is compatible with `shapely` (and not
      compatible with our own code!)
    """
    point = _np.asarray(point)
    if len(point.shape) == 2:
        if point.shape[0] != 1:
            raise ValueError("Need a single point")
        point = point[0]
    if point.shape != (2,):
        raise ValueError("Point should be (x,y)")
    line = _np.asarray(line)
    if len(line.shape) != 2 or line.shape[0] < 2 or line.shape[1] != 2:
        raise ValueError("Line should be ((x_1,y_1), ..., (x_n,y_n))")    
    options = [ _project_point_to_line(point, line[i:i+2,:]) 
        for i in range(line.shape[0] - 1) ]
    if line.shape[0] == 2:
        return options[0]
    distsq = [_np.sum((point - opt)**2) for opt in options]
    return options[_np.argmin(distsq)]

def project_point_to_lines(point, lines):
    """Find the closest point on one of the line segments to the point.
    
    :param point: Pair `(x,y)`(
    :param line: A list of linear segments (see :func:`project_point_to_line`).
    """
    point = _np.asarray(point)
    options = [project_point_to_line(point, line) for line in lines]
    distsq = [_np.sum((point - opt)**2) for opt in options]
    return options[_np.argmin(distsq)]
    
def project_point_to_lines_shapely(point, lines):
    """As :func:`project_point_to_lines` but uses `shapely` at a first pass.
    
    :param point: Pair `(x,y)`
    :param lines: A list of :class:`shapely.geometry.LineString` objects.
    """
    pt = _geometry.Point(point)
    dists = _np.asarray([line.distance(pt) for line in lines])
    line = lines[dists.argmin()]
    return project_point_to_line(point, line.coords)


try:
    import rtree as _rtree
except:
    _logger.error("Failed to import `rtree`.")
    _rtree = None
 
class ProjectPointLinesRTree():
    """Stuff...
    
    :param lines: A list of linear segments (see
      :func:`project_point_to_line`).
    """
    def __init__(self, lines):
        self._lines = list(lines)
        def gen():
            for i, line in enumerate(self._lines):
                bds = self._bounds(line)
                yield i, bds, None
        self._idx = _rtree.index.Index(gen())

    @staticmethod
    def _bounds(line):
        it = iter(line)
        x, y = next(it)
        xmin, xmax = x, x
        ymin, ymax = y, y
        for (x, y) in it:
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)
        return [xmin, ymin, xmax, ymax]

    def project_point(self, point):
        """As :func:`project_point_to_lines` but uses `rtree` at a first pass.
    
        :param point: Pair `(x,y)`
        """
        point = _np.asarray(point)
        h = 1
        while True:
            xmin, xmax = point[0] - h, point[0] + h
            ymin, ymax = point[1] - h, point[1] + h
            indices = list(self._idx.intersection((xmin,ymin,xmax,ymax)))
            if len(indices) > 0:
                choices = [self._lines[i] for i in indices]
                best = project_point_to_lines(point, choices)        
                distsq = _np.sum((best - point)**2)
                if distsq <= h*h:
                    return best
            h += h
