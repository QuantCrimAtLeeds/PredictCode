import pytest

import open_cp.plot as plot
import open_cp.data
import numpy as np
import shapely.geometry

@pytest.fixture
def grid():
    mask = np.asarray([[False]*5]*7)
    assert mask.shape == (7,5)
    return open_cp.data.MaskedGrid(xsize=10, ysize=20, xoffset=5, yoffset=7, mask=mask)

def test_outline_of_grid(grid):
    poly = plot.outline_of_grid(grid)
    assert poly.geom_type == "Polygon"
    expected = shapely.geometry.Polygon([[5,7], [55,7], [55,147], [5,147], [5,7]])
    assert expected.difference(poly).is_empty

@pytest.fixture
def grid1():
    mask = np.asarray([[False]*5]*7)
    mask[0,0] = True
    return open_cp.data.MaskedGrid(xsize=10, ysize=20, xoffset=5, yoffset=7, mask=mask)

def test_outline_of_grid_with_outer_gap(grid1):
    poly = plot.outline_of_grid(grid1)
    assert poly.geom_type == "Polygon"
    expected = shapely.geometry.Polygon([[15,7], [55,7], [55,147], [5,147], [5,27], [15,27]])
    assert expected.difference(poly).is_empty
