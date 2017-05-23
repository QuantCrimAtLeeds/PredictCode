"""
geometry
~~~~~~~~

Methods to help with geometry work.  Uses `shapely`.
"""

import numpy as _np
from . import data as _data

try:
    import shapely.geometry as _geometry
except Exception:
    import sys
    print("Failed to import shapely.", file=sys.stderr)
    _geometry = None

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
    can contain all grid cells which intersect with the geometry.

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

    mask = _np.zeros((height, width), dtype=_np.bool)
    xo = grid.xoffset + xstart * grid.xsize
    yo = grid.yoffset + ystart * grid.ysize
    for y in range(height):
        yy = yo + y * grid.ysize
        for x in range(width):
            xx = xo + x * grid.xsize
            poly = _geometry.Polygon([[xx, yy], [xx + grid.xsize, yy],
                    [xx + grid.xsize, yy + grid.ysize], [xx, yy + grid.ysize]])
            poly = poly.intersection(geometry)
            if poly.is_empty or poly.area == 0:
                mask[y][x] = True
    
    return _data.MaskedGrid(grid.xsize, grid.ysize, xo, yo, mask)