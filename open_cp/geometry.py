"""
geometry
~~~~~~~~

Methods to help with geometry work.  Uses `shapely`.
"""

import numpy as _np
import math as _math
from . import data as _data
import logging as _logging
# For what we use this for, we could use e.g binary search; but why re-invent
# the wheel?
import scipy.optimize as _optimize

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
    points = [ (x,y) for x,y in zip(timed_points.xcoords, timed_points.ycoords) ]
    mp = _geometry.MultiPoint(points)
    mp = mp.intersection(geo)
    points_we_want = set(tuple(pt) for pt in _np.asarray(mp))
    mask = [pt in points_we_want for pt in points]
    mask = _np.array(mask, dtype=_np.bool)
    return timed_points[mask]


    
#############################################################################    
# Point and line geometry
#############################################################################    

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

def intersect_line_box(start, end, box_bounds):
    """Intersect a line with a rectangular box.  The box is "half-open", so
    only the top and left boundary edges are considered part of the box.  If
    the line only intersects the box in a point, we consider this a no
    intersection.

    :param start: Pair `(x,y)` of the start of the line segment
    :param end: Pair `(x,y)` of the end of the line segment
    :param box_bounds: `(xmin, ymin, xmax, ymax)` of the box.  Formally, the
      box is those `(x,y)` with `xmin <= x < xmax` and `ymin <= y < ymax`.

    :return: `None` or `(t1, t2)` where `start * (1-t) + end * t` is
      in the box for `t1 < t < t2`.
    """
    dx, dy = end[0] - start[0], end[1] - start[1]
    xmin, ymin, xmax, ymax = tuple(box_bounds)
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Not a valid box")
    if _np.abs(dx) < 1e-10:
        # Vertical line
        if not ( xmin <= start[0] and start[0] < xmax ):
            return None
        if _np.abs(dy) < 1e-10:
            # Must be point
            if not ( ymin <= start[1] and start[1] < ymax ):
                return None
            return (0, 1)
        else:
            c, d = ymin - start[1], ymax - start[1]
            if dy > 0:
                c, d = c / dy, d / dy
            else:
                c, d = d / dy, c / dy
            return max(0, c), min(1, d)
    elif _np.abs(dy) < 1e-10:
        # (Proper) Horizontal line
        if not ( ymin <= start[1] and start[1] < ymax ):
            return None
        a, b = xmin - start[0], xmax - start[0]
        if dx > 0:
            a, b = a / dx, b / dx
        else:
            a, b = b / dx, a / dx
        return max(0, a), min(1, b)
    else:
        # Line in general position
        a, b = xmin - start[0], xmax - start[0]
        if dx > 0:
            a, b = a / dx, b / dx
        else:
            a, b = b / dx, a / dx
        c, d = ymin - start[1], ymax - start[1]
        if dy > 0:
            c, d = c / dy, d / dy
        else: 
            c, d = d / dy, c / dy
        tmin = max(a, c, 0)
        tmax = min(b, d, 1)
        if tmin < tmax:
            return (tmin, tmax)
        return None

def line_meets_geometry(geo, line):
    """Does the line intersect the geometry?
    
    :param geo: `shapely` object
    :param line: A line in the usual format, an iterable of points `(x,y)`
    
    :return: True or False
    """
    line = _geometry.LineString(list(line))
    return geo.intersects(line)

def lines_which_meet_geometry(geo, lines):
    """Which of the lines intersect the geometry?
    
    :param geo: `shapely` object
    :param lines: An iterable of lines in the usual format: each an iterable of
      points `(x,y)`
    
    :return: List of True or False
    """
    return [line_meets_geometry(geo, line) for line in lines]

def intersect_line_grid_most(line, grid):
    """Intersect a line with a grid.  Finds the grid cell which contains the
    largest fraction of the line (which might be an _arbitrary_ choice between
    more than one grid cell).
    
    :param line: `((x1,y1), (x2,y2))`
    :param grid: Instance of :class:`data.Grid` or same interface.
    
    :return: The grid cell `(gx, gy)` which contains most of the line.
    """
    _, intervals = full_intersect_line_grid(line, grid)
    best, length = None, None
    for (gx, gy, t1, t2) in intervals:
        t = t2 - t1
        if length is None or t > length:
            best, length = (gx, gy), t
    return best

def intersect_line_grid(line, grid):
    """Intersect a line with a grid, returning the smallest set of new lines
    which cover the original line and such that each new line segment lies
    entirely within one grid cell.
    
    :param line: `((x1,y1), (x2,y2))`
    :param grid: Instance of :class:`data.Grid` or same interface.
    
    :return: List of line segments.
    """
    segments, _ = full_intersect_line_grid(line, grid)
    return segments

def full_intersect_line_grid(line, grid):
    """Intersect a line with a grid, returning the smallest set of new lines
    which cover the original line and such that each new line segment lies
    entirely within one grid cell.
    
    :param line: `((x1,y1), (x2,y2))`
    :param grid: Instance of :class:`data.Grid` or same interface.
    
    :return: `(segments, intervals)` where `segments` is as
      :meth:`intersect_line_grid_most` and `intervals` is a list of tuples
      `(gx, gy, t1, t2)` telling that the line segment from (line coordinates)
      `t1` to `t2` is in grid cell `gx, gy`.  The ordering is the same as
      `segments`.
    """
    gx, gy = grid.grid_coord(*line[0])
    if grid.grid_coord(*line[1]) == (gx, gy):
        return [line], [(gx, gy, 0, 1)]
    
    segments, intervals = [], []
    
    start = (line[0][0] - grid.xoffset, line[0][1] - grid.yoffset)
    end = (line[1][0] - grid.xoffset, line[1][1] - grid.yoffset)
    search = start
    delta = 1e-8

    while True:
        gx, gy = _math.floor(search[0] / grid.xsize), _math.floor(search[1] / grid.ysize)
        bbox = (gx * grid.xsize, gy * grid.ysize, (gx+1) * grid.xsize, (gy+1) * grid.ysize)
        intersects = intersect_line_box(start, end, bbox)
        if intersects is None:
            t2 = 0
        else:
            t1, t2 = intersects
            segments.append((
                    (start[0]*(1-t1) + end[0]*t1 + grid.xoffset, start[1]*(1-t1) + end[1]*t1 + grid.yoffset),
                    (start[0]*(1-t2) + end[0]*t2 + grid.xoffset, start[1]*(1-t2) + end[1]*t2 + grid.yoffset)
                    ))
            intervals.append((gx, gy, t1, t2))
        t2 += delta
        if t2 >= 1:
            break
        search = (start[0]*(1-t2) + end[0]*t2, start[1]*(1-t2) + end[1]*t2)
    
    return segments, intervals
    

try:
    import rtree as _rtree
except:
    _logger.error("Failed to import `rtree`.")
    _rtree = None
 
class ProjectPointLinesRTree():
    """Accelerated projection code using `rtree`.
    
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



#############################################################################
# Voroni cell stuff
#############################################################################

try:
    import scipy.spatial as _spatial
except Exception as ex:
    _logger.error("Failed to import `scipy.spatial` because {}".format(ex))
    _spatial = None

class Voroni():
    """A wrapper around the `scipy.spatial` voroni diagram finding routine.
    
    :param points: Array of shape `(N,n)` of `N` points in `n`-dimensional
      space.
    """
    def __init__(self, points):
        points = _np.asarray(points)
        if len(points.shape) != 2 or points.shape[1] != 2:
            raise ValueError("Need array of shape (N,2)")
        self._v = _spatial.Voronoi(points)
        self._infinity_directions = dict()
        centre = _np.mean(self._v.points, axis=0)
        for ((a,b),(aa,bb)) in zip(self._v.ridge_vertices, self._v.ridge_points):
            if a == -1:
                x, y = self.perp_direction(self._v.points, aa, bb, centre) 
                self._infinity_directions[b] = x, y
        
    @property
    def voroni(self):
        """The `scipy.spatial.Voroni` class"""
        return self._v
    
    def polygons(self, inf_dist=1):
        """Return a list of polygons, one for each "region" of the voroni
        diagram.
        
        :param inf_dist: The distance to make each line towards the "point at
          infinity".
        
        :return: Iterator of "polygons".  Each "polygon" is a list of `(x,y)`
          points specifying the vertices.
        """
        done = set()    
        for point_index in range(self._v.points.shape[0]):
            region_index = self._v.point_region[point_index]
            if region_index in done:
                continue
            done.add(region_index)
            yield self._region_as_polygon(region_index, point_index, inf_dist)
    
    def polygon_for(self, point_index, inf_dist=1):
        """Return the polygon from the diagram which contains the given point.

        :param point_index: Index into `self.points`
        :param inf_dist: The distance to make each line towards the "point at
          infinity".
        
        :return: A "polygon", which is a list of `(x,y)` points specifying the
          vertices.
        """
        region_index = self._v.point_region[point_index]
        return self._region_as_polygon(region_index, point_index, inf_dist)

    def polygon_for_by_distance(self, point_index, distance):
        """Return the polygon from the diagram which contains the given point.
        Scale the size so that the containing point is `distance` away from
        "infinity".
        """
        region_index = self._v.point_region[point_index]
        poly, extra = self._region_datum(region_index, point_index)
        if extra is not None:
            inf_index, (first, second) = extra
            x1 = _np.asarray([first[0], first[1]])
            dx1 = _np.asarray([first[2], first[3]])
            x2 = _np.asarray([second[0], second[1]])
            dx2 = _np.asarray([second[2], second[3]])
            pt = self.points[point_index]
            def dist(t):
                return self._distance_line_to_point(x1 + t * dx1, x2 + t * dx2, pt)

            res = _optimize.minimize(dist, [0], bounds=[[0,_np.inf]])
            tzero = res.x
            if dist(tzero) > distance:
                t0 = 1
            else:
                t_up = tzero * 2
                while dist(t_up) < 1.1 * distance:
                    t_up += t_up + 1
                t0 = _optimize.brentq(lambda x : dist(x) - distance, tzero, t_up)
            
            poly[inf_index] = x1 + t0 * dx1
            poly.insert(inf_index, x2 + t0 * dx2)
        return poly
    
    def _region_datum(self, region_index, point_index):
        region = self._v.regions[region_index]    
        containing_points = {point_index}
        poly = [self._v.vertices[k] for k in region]
        if -1 in region:
            inf_index = region.index(-1)
            
            after_vertex = region[(inf_index + 1) % len(region)]
            choices = self._find_perp_line_to_infinity(after_vertex, containing_points)
            a, b = choices[0]
            dx, dy = self.perp_direction(self._v.points, a, b)
            x, y = self._v.vertices[after_vertex]
            extras = [(x, y, dx, dy)]
            
            before_vertex = region[(inf_index - 1) % len(region)]
            if before_vertex == after_vertex:
                a, b = choices[1]
            else:
                a, b = self._find_perp_line_to_infinity(before_vertex, containing_points)[0]
            dx, dy = self.perp_direction(self._v.points, a, b)
            x, y = self._v.vertices[before_vertex]
            extras.append((x, y, dx, dy))
            return poly, (inf_index, extras)
        else:
            return poly, None

    def _region_as_polygon(self, region_index, point_index, inf_dist):
        poly, extra = self._region_datum(region_index, point_index)
        if extra is not None:
            inf_index, (first, second) = extra
            x, y, dx, dy = first
            poly[inf_index] = x + dx * inf_dist, y + dy * inf_dist
            x, y, dx, dy = second
            poly.insert(inf_index, (x + dx * inf_dist, y + dy * inf_dist))
        return poly
    
    @staticmethod
    def _distance_line_to_point(line_start, line_end, point):
        a = _np.asarray(line_start)
        b = _np.asarray(line_end)
        v = b - a
        vnormsq = _np.sum(v * v)
        x = _np.asarray(point) - a
        if vnormsq < 1e-12:
            return _np.sqrt(_np.sum(x * x))
        t = _np.sum(x * v) / vnormsq
        u = x - t * v
        return _np.sqrt(_np.sum(u * u))
    
    def _find_perp_line_to_infinity(self, vertex, containing_points):
        out = []
        for verts, between in zip(self._v.ridge_vertices, self._v.ridge_points):
            if set(verts) == {-1, vertex}:
                if len(set(between).intersection(containing_points)) > 0:
                    out.append(between)
        return out
        
    @property
    def points(self):
        """The input points"""
        return self._v.points
    
    @property
    def vertices(self):
        """The voroni diagram vertices.  An array of shape `(M,2)`.
        """
        return self._v.vertices
    
    @property
    def regions(self):
        """A list of the regions of the diagram.  Each region is a list of
        indicies into `vertices`, where `-1` means the point at infinity."""
        return self._v.regions
    
    @property
    def point_region(self):
        """A list, ordered as `points`, giving which "region" each input
        point is in."""
        return self._v.point_region
    
    @property
    def ridge_vertices(self):
        """The "ridges" of the diagram are the lines forming the boundaries
        between regions.  This gives a list of pairs of indicies into
        `vertices`, where `-1` means the point at infinity."""
        return self._v.ridge_vertices
    
    @property
    def ridge_points(self):
        """Each "ridge" is perpendicular to a line between two points in the
        input data.  For each entry of `ridge_vertices` the perpendicular line
        is given by the indicies of the corresponding entry in this list
        """
        return self._v.ridge_points
    
    @staticmethod
    def perp_direction(points, a, b, centre=None):
        """Find a vector perpendicular to the line specified, oriented away
        from `centre`.
        
        :param points: Array of shape `(N,n)` of `N` points in `n`-dimensional
          space.
        :param a: Index into `points` of start of line.
        :param b: Index into `points` of end of line.
        :param centre: The location to orient from; if `None` then compute
          as centroid of the `points`.
        
        :return: Tuple of size `n` giving a vector orthogonal to the line,
          and oriented away from `centre`.
        """
        diff = points[b] - points[a]
        norm = _np.sqrt(_np.sum(diff*diff))
        diff = _np.asarray([diff[1]/norm, -diff[0]/norm])
        if centre is None:
            centre = _np.mean(points, axis=0)
        else:
            centre = _np.asarray(centre)
        midpoint = (points[a] + points[b]) / 2
        
        if _np.dot(centre - midpoint, diff) <= 0:
            return diff
        else:
            return -diff
    