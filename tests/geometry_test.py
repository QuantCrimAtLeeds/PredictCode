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

def test_intersect_line_box():
    assert geometry.intersect_line_box((0.5, -1), (0.5, 2), (0,0,2,1)) == pytest.approx((1/3, 2/3))
    assert geometry.intersect_line_box((0, -1), (0, 2), (0,0,2,1)) == pytest.approx((1/3, 2/3))
    assert geometry.intersect_line_box((2, -1), (2, 2), (0,0,2,1)) is None
    assert geometry.intersect_line_box((-.1, -1), (-.1, 2), (0,0,2,1)) is None
    assert geometry.intersect_line_box((2.1, -1), (2.1, 2), (0,0,2,1)) is None
    assert geometry.intersect_line_box((1,0.2), (1,0.8), (0,0,2,1)) == pytest.approx((0, 1))
    
    assert geometry.intersect_line_box((-1,0.5), (3,0.5), (0,0,2,1)) == pytest.approx((0.25, 0.75))
    assert geometry.intersect_line_box((-1,0), (3,0), (0,0,2,1)) == pytest.approx((0.25, 0.75))
    assert geometry.intersect_line_box((-1,1), (3,1), (0,0,2,1)) is None
    assert geometry.intersect_line_box((-1,-0.1), (3,-0.1), (0,0,2,1)) is None
    assert geometry.intersect_line_box((-1,1.1), (3,1.1), (0,0,2,1)) is None
    assert geometry.intersect_line_box((0.2,0.5), (1,0.5), (0,0,2,1)) == pytest.approx((0,1))
    
    assert geometry.intersect_line_box((0.2,0.5), (0.2,0.5), (0,0,2,1)) == pytest.approx((0,1))
    
    assert geometry.intersect_line_box((0.2,0.5), (1,0.7), (0,0,2,1)) == pytest.approx((0,1))
    assert geometry.intersect_line_box((1,0.7), (0.2,0.5), (0,0,2,1)) == pytest.approx((0,1))
    
    assert geometry.intersect_line_box((-1,-1), (1,1), (0,0,2,1)) == pytest.approx((0.5, 1))
    assert geometry.intersect_line_box((-2,-1), (2,1), (0,0,2,1)) == pytest.approx((0.5, 1))
    assert geometry.intersect_line_box((-2,-1), (4,2), (0,0,2,1)) == pytest.approx((1/3, 2/3))
    assert geometry.intersect_line_box((0,1), (2,3), (1,2,3,3)) == pytest.approx((0.5, 1))
    
    assert geometry.intersect_line_box((-1,-1), (-0.2,-0.2), (0,0,2,1)) is None
    assert geometry.intersect_line_box((-1,-1), (0,0), (0,0,2,1)) is None
    assert geometry.intersect_line_box((-1,1), (1,-1), (0,0,2,1)) is None
    
def test_intersect_line_grid():
    grid = open_cp.data.Grid(xsize=10, ysize=10, xoffset=0, yoffset=0)
    line = ( (2, 2), (7, 7) )
    out = geometry.intersect_line_grid(line, grid)
    assert out == [ line ]

    grid = open_cp.data.Grid(xsize=10, ysize=10, xoffset=3, yoffset=3)
    line = ( (2, 2), (7, 7) )
    out = geometry.intersect_line_grid(line, grid)
    assert out == [ ((2,2), (3,3)), ((3,3), (7,7)) ]
    
    grid = open_cp.data.Grid(xsize=10, ysize=10, xoffset=3, yoffset=4)
    line = ( (2, 3), (7, 8) )
    out = geometry.intersect_line_grid(line, grid)
    assert out == [ ((2,3), (3,4)), ((3,4), (7,8)) ]
    
    grid = open_cp.data.Grid(xsize=10, ysize=1, xoffset=0, yoffset=0)
    line = ( (0,0.5), (20, 0.5) )
    out = geometry.intersect_line_grid(line, grid)
    assert out == [ ((0,0.5), (10,0.5)), ((10,0.5), (20,0.5)) ]

    grid = open_cp.data.Grid(xsize=10, ysize=1, xoffset=0, yoffset=0.5)
    line = ( (0,0.5), (20, 0.5) )
    out = geometry.intersect_line_grid(line, grid)
    assert out == [ ((0,0.5), (10,0.5)), ((10,0.5), (20,0.5)) ]

    grid = open_cp.data.Grid(xsize=10, ysize=4, xoffset=1, yoffset=1.5)
    for _ in range(100):
        a,b,c,d = np.random.random(size=4) * 100
        line = ( (a,b), (c,d) )
        out = geometry.intersect_line_grid(line, grid)
        assert out[0][0] == pytest.approx((a, b))
        assert out[-1][1] == pytest.approx((c, d))
        for i in range(0, len(out)-1):
            assert out[i][1] == pytest.approx(out[i+1][0])
        for (s, e) in out:
            t = 0.001
            ss = s[0] * (1-t) + e[0] * t, s[1] * (1-t) + e[1] * t
            ee = s[0] * t + e[0] * (1-t), s[1] * t + e[1] * (1-t)
            gx, gy = grid.grid_coord(*ss)
            bbox = grid.bounding_box_of_cell(gx, gy)
            tt = geometry.intersect_line_box(ss, ee, bbox)
            assert tt == pytest.approx((0, 1))

def test_intersect_line_grid_most():
    grid = open_cp.data.Grid(xsize=10, ysize=10, xoffset=0, yoffset=0)
    line = ( (2, 2), (7, 7) )
    assert geometry.intersect_line_grid_most(line, grid) == (0,0)
    line = ( (2, 2), (11, 11) )
    assert geometry.intersect_line_grid_most(line, grid) == (0,0)
    line = ( (2, 2), (18, 18) )
    assert geometry.intersect_line_grid_most(line, grid) == (0,0)
    line = ( (2, 2), (19, 19) )
    assert geometry.intersect_line_grid_most(line, grid) == (1,1)
    line = ( (2, 2), (30, 30) )
    assert geometry.intersect_line_grid_most(line, grid) == (1,1)
    
def test_voroni_perp():
    points = np.asarray([[1,2], [2,3], [3,4]])
    x, y = geometry.Voroni.perp_direction(points, 0, 1, [0,0])
    assert (x,y) == pytest.approx((-1/np.sqrt(2), 1/np.sqrt(2)))
    x, y = geometry.Voroni.perp_direction(points, 0, 1, [0,3])
    assert (x,y) == pytest.approx((1/np.sqrt(2), -1/np.sqrt(2)))
    
def test_voroni1():
    points = np.asarray([[0,0], [1,0], [0,1], [1,1]])
    voroni = geometry.Voroni(points)
    
    np.testing.assert_allclose(voroni.points, points)
    np.testing.assert_allclose(voroni.vertices, [[0.5,0.5]])
    x = list(voroni.polygons(1))
    print("This might not be robust, due to re-ording issues...")
    assert len(x) == 4
    np.testing.assert_allclose(x[0], [[0.5, -0.5], [-0.5,0.5], [0.5,0.5]])
    np.testing.assert_allclose(x[1], [[0.5, -0.5], [1.5,0.5], [0.5,0.5]])
    np.testing.assert_allclose(x[2], [[-0.5, 0.5], [0.5,1.5], [0.5,0.5]])
    np.testing.assert_allclose(x[3], [[1.5, 0.5], [0.5,1.5], [0.5,0.5]])

def test_voroni_poly():
    points = np.asarray([[0,0], [1,0], [0,1], [1,1]])
    voroni = geometry.Voroni(points)
    np.testing.assert_allclose(voroni.polygon_for(0, 2), [[0.5, -1.5], [-1.5,0.5], [0.5,0.5]])
    np.testing.assert_allclose(voroni.polygon_for(3, 2), [[2.5, 0.5], [0.5,2.5], [0.5,0.5]])
