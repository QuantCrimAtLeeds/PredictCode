"""
plot
~~~~

Utility methods for interacting with matplotlib
"""

import matplotlib.patches
from . import data as _data
try:
    import descartes
except:
    import sys
    print("Failed to load 'descartes' package.", file=sys.stderr)
    descartes = None
try:
    import shapely.geometry as _sgeometry
except:
    import sys
    print("Failed to load 'shapely' package.", file=sys.stderr)
    _sgeometry = None


def outline_of_grid(grid):
    """Returns a `shapely` (multi-)polygon object given by merging all the
    grid cells from the passed :class:`MaskedGrid` object.

    :param grid: A :class:`MaskedGrid` instace.

    :return: A `shapely` geometry object.
    """
    height, width = grid.mask.shape
    polygon = None
    for y in range(height):
        yy = y * grid.ysize + grid.yoffset
        for x in range(width):
            if grid.is_valid(x, y):
                xx = x * grid.xsize + grid.xoffset
                poly = _sgeometry.Polygon([[xx,yy], [xx+grid.xsize, yy],
                        [xx+grid.xsize, yy+grid.ysize], [xx, yy+grid.ysize]])
                if polygon is None:
                    polygon = poly
                else:
                    polygon = polygon.union(poly)
    return polygon
    

def patches_from_grid(grid):
    """Returns a list of `matplotlib` `patches` from the passed
    :class:`MaskedGrid` object.  Typical usage:

        pc = matplotlib.collections.PatchCollection(patches_from_grid(grid))
        fig, ax = plt.subplots()
        ax.add_collection(pc)

    This will be slow if there are a large number of grid cells.

    :param grid: A :class:`MaskedGrid` instace.

    :return: A list of patches.
    """
    height, width = grid.mask.shape
    patches = []
    for y in range(height):
        yy = y * grid.ysize + grid.yoffset
        for x in range(width):
            if grid.is_valid(x, y):
                xx = x * grid.xsize + grid.xoffset
                patches.append(matplotlib.patches.Rectangle((xx, yy), grid.xsize, grid.ysize))
    return patches


def _add_line(lines, x1, y1, x2, y2):
    lines.append([(x1,y1), (x2,y2)])

def lines_from_grid(grid):
    """Returns a list of line segments which when drawn will form the cells of
    the passed :class:`MaskedGrid` object.  Typical usage:

        lc = matplotlib.collections.LineCollection(lines_from_grid(grid))
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        
    This is somewhat quicker than using :func:`patches_from_grid` but still
    slow.

    :param grid: A :class:`MaskedGrid` instace.

    :return: A list of "line"s.  Each line is a list with two entries, each
      entry being a tuple `(x,y)` of coordinates.
    This is somewhat quicker than using 
    """
    height, width = grid.mask.shape
    lines = []
    for y in range(height):
        yy = y * grid.ysize + grid.yoffset
        for x in range(width):
            if grid.is_valid(x, y):
                xx = x * grid.xsize + grid.xoffset
                xx1 = xx + grid.xsize
                yy1 = yy + grid.ysize
                _add_line(lines, xx, yy, xx1, yy)
                _add_line(lines, xx, yy, xx, yy1)
                _add_line(lines, xx1, yy, xx1, yy1)
                _add_line(lines, xx, yy1, xx1, yy1)
    return lines

def lines_from_regular_grid(grid):
    """As :func:`lines_from_grid` but the passed grid is assumed to be a whole
    rectangle, not a more complicated masked object.  Hugely faster.
    """
    height, width = grid.mask.shape
    lines = []
    for y in range(height+1):
        yy0 = y * grid.ysize + grid.yoffset
        xx0 = grid.xoffset
        xx1 = xx0 + width * grid.xsize
        _add_line(lines, xx0,yy0, xx1,yy0)
    for x in range(width+1):
        xx0 = x * grid.xsize + grid.xoffset
        yy0 = grid.yoffset
        yy1 = yy0 + height * grid.ysize
        _add_line(lines, xx0,yy0, xx0,yy1)
    return lines

def patches_from_geometry(geo, **kwargs):
    """Convert an iterable of geometry to `matplotlib` patches.
    
    :param geo: An iterable of geometry items.  If `descartes` cannot parse
      an item, it is ignored.
    :param **kwargs: Any key-word arguments to forward on to the `patch`
      constructor.
      
    :return: A list of `matplotlib.patches.Patch` objects.
    """
    patches = []
    for x in geo:
        try:
            patches.append(descartes.PolygonPatch(x, **kwargs))
        except:
            pass
    return patches

def lines_from_geometry(geo):
    """Convert an iterable of geometry to lines.  Suitable for passing 
    directly to `matplotlib.collections.LineCollection`.
    
    :param geo: An iterable of geometry items.  If cannot be coverted to a
      line, then ignored.
      
    :return: A list of coordinates.
    """
    lines = []
    for x in geo:
        try:
            lines.append( list(x.coords) )
        except:
            pass
    return lines
    