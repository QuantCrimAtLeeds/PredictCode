import pytest

import open_cp.geometry as geometry
import shapely.geometry
import open_cp.predictors
import open_cp.data
import numpy as np

def test_grid_intersection():
    geo = shapely.geometry.Polygon([[0,0],[10,0],[10,10],[0,10]])
    grid = data.Grid(xoffset=2, yoffset=2, xsize=7, ysize=4)
    out = geometry.grid_intersection(geo, grid)

    expected = []
    for x in range(-1,2):
        for y in range(-1,2):
            expected.append((x,y))
    assert(set(expected) == set(out))

def test_grid_intersection():
    geo = shapely.geometry.Polygon([[0,0],[10,0],[10,10],[0,10]])
    gp = open_cp.predictors.GridPrediction(xsize=11, ysize=3, xoffset=2, yoffset=-1)
    out = geometry.grid_intersection(geo, gp)

    expected = []
    for x in [-1,0]:
        for y in [0,1,2,3]:
            expected.append((x,y))
    assert(set(expected) == set(out))

def test_mask_grid_by_intersection():
    geo = shapely.geometry.Polygon([[0,0],[10,0],[10,10],[0,10]])
    gp = open_cp.predictors.GridPrediction(xsize=11, ysize=3, xoffset=2, yoffset=-1)
    mg = geometry.mask_grid_by_intersection(geo, gp)

    assert mg.xsize == 11
    assert mg.ysize == 3
    assert mg.xoffset == -9
    assert mg.yoffset == -1

    assert not mg.mask.any()
    #np.testing.assert_array_equal(expected, mg.mask)

