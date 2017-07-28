import pytest

import open_cp.geometry as geometry
import shapely.geometry
import open_cp.predictors
import open_cp.data
import numpy as np
import datetime

def test_grid_intersection():
    geo = shapely.geometry.Polygon([[0,0],[10,0],[10,10],[0,10]])
    grid = open_cp.data.Grid(xoffset=2, yoffset=2, xsize=7, ysize=4)
    out = geometry.grid_intersection(geo, grid)

    expected = []
    for x in range(-1,2):
        for y in range(-1,2):
            expected.append((x,y))
    assert(set(expected) == set(out))

def test_grid_intersection1():
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

@pytest.fixture
def points1():
    t = [datetime.datetime.now() for _ in range(5)]
    x = [0, 7, 12, 20, 22]
    y = [0, -2, 7, 8, 25]
    return open_cp.data.TimedPoints.from_coords(t,x,y)

def test_mask_grid_by_points_intersection_bbox(points1):
    grid = open_cp.data.Grid(10, 10, 5, 3)
    mg = geometry.mask_grid_by_points_intersection(points1, grid, True)
    assert mg.xoffset == -5
    assert mg.yoffset == -7
    assert mg.xsize == 10
    assert mg.ysize == 10
    assert mg.xextent == 3
    assert mg.yextent == 4
    for x in range(3):
        for y in range(4):
            assert mg.is_valid(x, y)

def test_mask_grid_by_points_intersection(points1):
    grid = open_cp.data.Grid(10, 10, 5, 3)
    mg = geometry.mask_grid_by_points_intersection(points1, grid)
    assert mg.xoffset == -5
    assert mg.yoffset == -7
    assert mg.xsize == 10
    assert mg.ysize == 10
    assert mg.xextent == 3
    assert mg.yextent == 4
    expected = [(0,0), (1,0), (1,1), (2,1), (2,3)]
    for x in range(3):
        for y in range(4):
            assert mg.is_valid(x, y) == ((x,y) in expected)

def test_configure_gdal():
    geometry.configure_gdal()
    import os
    print("This seems to fail on linux when run with `pytest tests\geometry_test.py")
    print("But is fine if run with `pytest`")
    print("Sadly I do not understand...")
    assert "GDAL_DATA" in os.environ
    
def test_shapely_line_format():
    # Paranoid, and annoyingly different convention to us
    line = shapely.geometry.LineString(((1,2), (3,1)))
    np.testing.assert_allclose(line.xy[0], [1,3])
    np.testing.assert_allclose(line.xy[1], [2,1])
    x = np.asarray(line.coords)
    np.testing.assert_allclose(x[0], [1,2])
    np.testing.assert_allclose(x[1], [3,1])
    
def test_project_point_to_line():
    line = [(0,0), (10,0)]
    np.testing.assert_allclose(geometry.project_point_to_line((3,1), line), (3,0))
    np.testing.assert_allclose(geometry.project_point_to_line((3,-1), line), (3,0))
    np.testing.assert_allclose(geometry.project_point_to_line((-1,1), line), (0,0))
    np.testing.assert_allclose(geometry.project_point_to_line((11,-1), line), (10,0))
    
    line = shapely.geometry.LineString(((5,2), (5,12)))
    np.testing.assert_allclose(geometry.project_point_to_line((3,5), line.coords), (5,5))
    np.testing.assert_allclose(geometry.project_point_to_line((-3,6), line.coords), (5,6))
    np.testing.assert_allclose(geometry.project_point_to_line((4,1), line.coords), (5,2))
    np.testing.assert_allclose(geometry.project_point_to_line([(5,14)], line.coords), (5,12))
    
    with pytest.raises(ValueError):
        geometry.project_point_to_line((3,5,6), line.coords)
    with pytest.raises(ValueError):
        geometry.project_point_to_line((3), line.coords)
                                       
    with pytest.raises(ValueError):
        geometry.project_point_to_line((3,5), [1,2])
    with pytest.raises(ValueError):
        geometry.project_point_to_line((3), [[1,2],[3,4],[5,6]])
    with pytest.raises(ValueError):
        geometry.project_point_to_line((3), [[1,2,3],[3,4,7]])

    line = [(0,0), (10,0), (10,6)]
    np.testing.assert_allclose(geometry.project_point_to_line((1,1), line), (1,0))
    np.testing.assert_allclose(geometry.project_point_to_line((8,1), line), (8,0))
    np.testing.assert_allclose(geometry.project_point_to_line((9.5,1), line), (10,1))
    np.testing.assert_allclose(geometry.project_point_to_line((9,7), line), (10,6))
    np.testing.assert_allclose(geometry.project_point_to_line((11,-1), line), (10,0))

@pytest.fixture
def lines():
    return [  [(0,0), (10,0)],
               [(0,1), (5,5), (9,1)]
           ]
    
def test_project_point_to_lines(lines):
    np.testing.assert_allclose(geometry.project_point_to_lines((5,1), lines), (5,0))
    np.testing.assert_allclose(geometry.project_point_to_lines((-0.5, -0.5), lines), (0,0))
    np.testing.assert_allclose(geometry.project_point_to_lines((-0.1,1), lines), (0,1))
    np.testing.assert_allclose(geometry.project_point_to_lines((5,4.8), lines), (5.1,4.9))
    np.testing.assert_allclose(geometry.project_point_to_lines((9,.4), lines), (9,0))
    np.testing.assert_allclose(geometry.project_point_to_lines((9,.6), lines), (9,1))
    
def test_project_point_to_lines_shapely(lines):
    lines = [ shapely.geometry.LineString(line) for line in lines ]
    np.testing.assert_allclose(geometry.project_point_to_lines_shapely((5,1), lines), (5,0))
    np.testing.assert_allclose(geometry.project_point_to_lines_shapely((-0.5, -0.5), lines), (0,0))
    np.testing.assert_allclose(geometry.project_point_to_lines_shapely((-0.1,1), lines), (0,1))
    np.testing.assert_allclose(geometry.project_point_to_lines_shapely((5,4.8), lines), (5.1,4.9))
    np.testing.assert_allclose(geometry.project_point_to_lines_shapely((9,.4), lines), (9,0))
    np.testing.assert_allclose(geometry.project_point_to_lines_shapely((9,.6), lines), (9,1))
    
def test_ProjectPointLinesRTree(lines):
    pp = geometry.ProjectPointLinesRTree(lines)
    np.testing.assert_allclose(pp.project_point((5,1)), (5,0))
    
def test_project_point_to_lines_compare(lines):
    lines_shapely = [ shapely.geometry.LineString(line) for line in lines ]
    pp = geometry.ProjectPointLinesRTree(lines)
    for _ in range(100):
        pt = np.random.random(2) * 10
        a = geometry.project_point_to_lines(pt, lines)
        b = geometry.project_point_to_lines_shapely(pt, lines_shapely)
        np.testing.assert_allclose(a, b)
        c = pp.project_point(pt)
        np.testing.assert_allclose(a, c)
