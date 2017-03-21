import pytest
from open_cp.data import Point, RectangularRegion, TimedPoints

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


from datetime import datetime as dt
import numpy as np
import numpy.testing as npt

# Test builder object pattern doesn't quite work in Python, but nevermind...
def a_valid_TimedPoints():
    timestamps = [dt(2017,3,20,12,30), dt(2017,3,20,14,30)]
    coords = [ [1, 5], [7, 10] ]
    return TimedPoints(timestamps, coords)

def test_TimedPoints_builds():
    tp = a_valid_TimedPoints()
    assert tp.timestamps[0] == np.datetime64("2017-03-20T12:30")
    assert tp.timestamps[1] == np.datetime64("2017-03-20T14:30")
    npt.assert_array_almost_equal(tp.coords[0], [1, 5])
    npt.assert_array_almost_equal(tp.coords[1], [7, 10])
    assert len(tp.timestamps) == 2
    assert tp.coords.shape == (2, 2)

def test_TimedPoints_must_be_time_ordered():
    timestamps = [dt(2017,3,20,14,30), dt(2017,3,20,12,30)]
    coords = [ [1, 5], [7, 10] ]
    with pytest.raises(ValueError):
        TimedPoints(timestamps, coords)

def test_TimedPoints_from_coords():
    tp2 = a_valid_TimedPoints()
    tp = TimedPoints.from_coords(tp2.timestamps, tp2.coords[:,0], tp2.coords[:,1])
    assert tp.timestamps[0] == np.datetime64("2017-03-20T12:30")
    assert tp.timestamps[1] == np.datetime64("2017-03-20T14:30")
    npt.assert_array_almost_equal(tp.coords[0], [1, 5])
    npt.assert_array_almost_equal(tp.coords[1], [7, 10])
    assert len(tp.timestamps) == 2
    assert tp.coords.shape == (2, 2)

def test_TimedPoints_events_before():
    tp2 = a_valid_TimedPoints()
    tp = tp2.events_before(dt(2017,3,20,14,0))
    assert tp.timestamps[0] == np.datetime64("2017-03-20T12:30")
    assert len(tp.timestamps) == 1
    npt.assert_array_almost_equal(tp.coords[0], [1, 5])
    assert tp.coords.shape == (1, 2)
