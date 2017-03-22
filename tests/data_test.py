import pytest
from open_cp.data import Point, RectangularRegion, TimedPoints
import open_cp.data

import numpy as np

def test_Point_getters():
    p = Point(5, 8)
    assert p.x == 5
    assert p.y == 8
    assert p.__repr__() == "Point(5,8)"

def test_Point_add():
    p = Point(2, 3)
    pp = p + Point(7, 10)
    assert pp.x == 9
    assert pp.y == 13

def test_Point_immutable():
    p = Point(1, 2)
    with pytest.raises(AttributeError):
        p.x = 5
    with pytest.raises(AttributeError):
        p.y = 5


def test_RectangluarRegion_getters():
    r = RectangularRegion(5, 8, 10, 14)
    assert r.xmin == 5
    assert r.xmax == 8
    assert r.ymin == 10
    assert r.ymax == 14
    assert r.min.x == 5
    assert r.min.y == 10
    assert r.max.x == 8
    assert r.max.y == 14
    assert r.__repr__() == "RectangularRegion( (5,10) -> (8,14) )"

def test_RectangluarRegion_add():
    r = RectangularRegion(5, 8, 10, 14)
    addition = Point(3, 7)
    rr = r + addition
    assert rr.xmin == 8
    assert rr.xmax == 11
    assert rr.ymin == 17
    assert rr.ymax == 21

def test_RectangluarRegion_aspect():
    assert( RectangularRegion(xmin=5, xmax=5, ymin=1, ymax=10).aspect_ratio is np.nan )
    assert( RectangularRegion(xmin=5, xmax=10, ymin=1, ymax=10).aspect_ratio == pytest.approx(9/5) )

from datetime import datetime as dt
import numpy.testing as npt

# Test builder object pattern doesn't quite work in Python, but nevermind...
def a_valid_TimedPoints():
    timestamps = [dt(2017,3,20,12,30), dt(2017,3,20,14,30), dt(2017,3,21,0)]
    coords = [ [1, 5, 8], [7, 10, 3] ]
    return TimedPoints(timestamps, coords)

def test_TimedPoints_builds():
    tp = a_valid_TimedPoints()
    assert tp.timestamps[0] == np.datetime64("2017-03-20T12:30")
    assert tp.timestamps[1] == np.datetime64("2017-03-20T14:30")
    assert tp.timestamps[2] == np.datetime64("2017-03-21")
    npt.assert_array_almost_equal(tp.coords[:,0], [1, 7])
    npt.assert_array_almost_equal(tp.coords[:,1], [5, 10])
    npt.assert_array_almost_equal(tp.coords[:,2], [8, 3])
    assert len(tp.timestamps) == 3
    assert tp.coords.shape == (2, 3)

def test_coords_properties():
    tp = a_valid_TimedPoints()
    npt.assert_allclose( tp.xcoords, [1, 5, 8] )
    npt.assert_allclose( tp.ycoords, [7, 10, 3] )

def test_accessor():
    tp = a_valid_TimedPoints()
    assert( tp[0][0] == np.datetime64("2017-03-20T12:30") )
    assert( tp[0][1] == pytest.approx(1) )
    assert( tp[0][2] == pytest.approx(7) )
    assert( tp[2][0] == np.datetime64("2017-03-21") )
    assert( tp[2][1] == pytest.approx(8) )
    assert( tp[2][2] == pytest.approx(3) )

def test_accessor_index():
    tp = a_valid_TimedPoints()
    tp1 = tp[ [2,1] ]
    np.testing.assert_equal( tp1.timestamps, [np.datetime64("2017-03-20T14:30"), np.datetime64("2017-03-21")])
    np.testing.assert_allclose( tp1.xcoords, [5, 8])
    np.testing.assert_allclose( tp1.ycoords, [10, 3])

    tp2 = tp[ tp.xcoords < 5 ]
    np.testing.assert_equal( tp2.timestamps, [np.datetime64("2017-03-20T12:30")] )
    np.testing.assert_allclose( tp2.coords, [[1], [7]])

def test_bounding_box():
    tp = a_valid_TimedPoints()
    box = tp.bounding_box()
    assert( box.xmin == pytest.approx(1) )
    assert( box.xmax == pytest.approx(8) )
    assert( box.ymin == pytest.approx(3) )
    assert( box.ymax == pytest.approx(10) )
    assert( tp.time_range() == (np.datetime64("2017-03-20T12:30"), np.datetime64("2017-03-21")) )

def test_TimedPoints_must_be_time_ordered():
    timestamps = [dt(2017,3,20,14,30), dt(2017,3,20,12,30)]
    coords = [ [1, 5], [7, 10] ]
    with pytest.raises(ValueError):
        TimedPoints(timestamps, coords)

def test_TimedPoints_from_coords():
    tp2 = a_valid_TimedPoints()
    tp = TimedPoints.from_coords(tp2.timestamps, [3, 1, 4], [1, 5, 9])
    assert tp.timestamps[0] == np.datetime64("2017-03-20T12:30")
    assert tp.timestamps[1] == np.datetime64("2017-03-20T14:30")
    npt.assert_array_almost_equal(tp.coords[0], [3, 1, 4])
    npt.assert_array_almost_equal(tp.coords[1], [1, 5, 9])
    assert len(tp.timestamps) == 3
    assert tp.coords.shape == (2, 3)

def test_TimedPoints_events_before():
    tp2 = a_valid_TimedPoints()
    tp = tp2.events_before(dt(2017,3,20,14,0))
    assert tp.timestamps[0] == np.datetime64("2017-03-20T12:30")
    assert len(tp.timestamps) == 1
    assert( tp.coords[0] == pytest.approx(1) )
    assert( tp.coords[1] == pytest.approx(7) )
    assert tp.coords.shape == (2, 1)

def test_project_from_lon_lat():
    tp = TimedPoints([np.datetime64("2016-12")], [[-1.5],[50]])
    tp1 = open_cp.data.points_from_lon_lat(tp, epsg=7405)
    npt.assert_array_equal(tp.timestamps, tp1.timestamps)
    assert( tp1.xcoords[0] == pytest.approx(435830.0305552669) )
    assert( tp1.ycoords[0] == pytest.approx(11285.188608689801) )

    import pyproj, math
    proj = pyproj.Proj({"init": "epsg:4326"})
    tp2 = open_cp.data.points_from_lon_lat(tp, proj=proj)
    npt.assert_allclose( tp.coords / 180 * math.pi, tp2.coords )