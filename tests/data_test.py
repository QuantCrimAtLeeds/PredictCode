import pytest
from open_cp.data import Point, RectangularRegion, TimedPoints
import open_cp.data
import datetime

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
        
def test_Point_iterable():
    def func(x, y):
        assert x == 5
        assert y == 7
    p = Point(5, 7)
    func(*p)
    
def test_Point_indexing():
    p = Point(3,7)
    assert p[0] == 3
    assert p[1] == 7
    with pytest.raises(ValueError):
        p[2]


def test_MaskedGrid():
    mg = open_cp.data.MaskedGrid(10, 10, 0, 0, [[True, False], [False, False]])
    assert not mg.is_valid(0,0)
    assert mg.is_valid(1,0)
    assert mg.is_valid(0,1)
    assert mg.is_valid(1,1)

    with pytest.raises(ValueError):
        mg.is_valid(-1, 0)
    with pytest.raises(ValueError):
        mg.is_valid(0, -1)
    with pytest.raises(ValueError):
        mg.is_valid(2, 0)
    with pytest.raises(ValueError):
        mg.is_valid(0, 2)

def test_MaskedGrid_from_Grid():
    grid = open_cp.data.Grid(10, 15, 5, 7)
    mg = open_cp.data.MaskedGrid.from_grid(grid, [[False, True], [False, False]])

    assert mg.is_valid(0,0)
    assert not mg.is_valid(1,0)
    assert mg.is_valid(0,1)
    assert mg.is_valid(1,1)
    assert mg.xextent == 2
    assert mg.yextent == 2

    assert mg.region().xmin == 5
    assert mg.region().xmax == 25
    assert mg.region().ymin == 7
    assert mg.region().ymax == 37

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

def test_RectangluarRegion_to_tuple():
    r = RectangularRegion(xmin=5, ymin=8, xmax=10, ymax=12)
    assert tuple(r) == (5, 8, 10, 12)

def test_RectangluarRegion_aspect():
    assert( RectangularRegion(xmin=5, xmax=5, ymin=1, ymax=10).aspect_ratio is np.nan )
    assert( RectangularRegion(xmin=5, xmax=10, ymin=1, ymax=10).aspect_ratio == pytest.approx(9/5) )

def test_RectangluarRegion_grid_size():
    region = RectangularRegion(xmin=4, xmax=50, ymin=1, ymax=10)
    assert( region.grid_size(10) == (5, 1) )
    assert( region.grid_size(10, 5) == (5, 2) )

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

def test_time_deltas():
    tp = a_valid_TimedPoints()
    npt.assert_allclose(tp.time_deltas(), [0, 120, 12*60-30])

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
    box = tp.bounding_box
    assert( box.xmin == pytest.approx(1) )
    assert( box.xmax == pytest.approx(8) )
    assert( box.ymin == pytest.approx(3) )
    assert( box.ymax == pytest.approx(10) )
    assert( tp.time_range == (np.datetime64("2017-03-20T12:30"), np.datetime64("2017-03-21")) )

def test_TimedPoints_must_be_time_ordered():
    timestamps = [dt(2017,3,20,14,30), dt(2017,3,20,12,30)]
    coords = [ [1, 5], [7, 10] ]
    with pytest.raises(ValueError):
        TimedPoints(timestamps, coords)

def test_TimedPoints_from_coords():
    times = [np.datetime64("2016-10-11T12:30"), np.datetime64("2016-05-07T00:00"),
             np.datetime64("2017-01-02T14:00")]
    tp = TimedPoints.from_coords(times, [3, 1, 4], [1, 5, 9])
    assert tp.timestamps[0] == np.datetime64("2016-05-07T00:00")
    assert tp.timestamps[1] == np.datetime64("2016-10-11T12:30")
    assert tp.timestamps[2] == np.datetime64("2017-01-02T14:00")
    npt.assert_array_almost_equal(tp.coords[0], [1, 3, 4])
    npt.assert_array_almost_equal(tp.coords[1], [5, 1, 9])
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
    assert tp.number_data_points == 1

def test_TimedPoints_events_before_empty():
    tp2 = a_valid_TimedPoints()
    tp = tp2.events_before(dt(2017,3,20,12,0))
    assert( tp.empty )
    
def test_TimedPoints_to_time_space_coords():
    tp = a_valid_TimedPoints()
    points = tp.to_time_space_coords()
    np.testing.assert_allclose(points[1], tp.xcoords)
    np.testing.assert_allclose(points[2], tp.ycoords)
    np.testing.assert_allclose(points[0], [0, 120.0, 11.5*60])

def test_TimedPoints_times_datetime():
    tp = a_valid_TimedPoints()
    dates = tp.times_datetime()
    assert( dates[0] == dt(2017,3,20,12,30) )
    assert( dates[1] == dt(2017,3,20,14,30) )
    assert( dates[2] == dt(2017,3,21,0) )

def test_order_by_times():
    ts = np.floor(np.random.random(size=100) * 350)
    ts = [datetime.datetime(2017,1,1) + datetime.timedelta(days=t) for t in ts]
    xcs = list(range(100))
    ycs = [100-x for x in range(100)]
    timestamps, xcoords, ycoords = open_cp.data.order_by_time(ts, xcs, ycs)
    for t1, t2 in zip(timestamps[1:], timestamps[:-1]):
        assert(t1>=t2)
    for t, x, y in zip(timestamps, xcoords, ycoords):
        assert(x + y == 100)
        assert(t == ts[x])


@pytest.fixture
def timestamps():
    return open_cp.data.TimeStamps([
        datetime.datetime(2017,1,2,12,30),
        datetime.datetime(2017,1,2,14,23),
        datetime.datetime(2017,1,3,0,0),
        datetime.datetime(2017,1,3,4,5),
        datetime.datetime(2017,1,4,1,2)
        ])

def test_TimeStamps_bin(timestamps):
    ts = timestamps.bin_timestamps(datetime.datetime(2017,1,1), datetime.timedelta(days=1))
    np.testing.assert_allclose(ts.time_deltas(np.timedelta64(1,"h")) + 24, [24,24,48,48,72])
    ts = timestamps.bin_timestamps(datetime.datetime(2017,1,1), datetime.timedelta(hours=12))
    np.testing.assert_allclose(ts.time_deltas(np.timedelta64(1,"h")), [0,0,12,12,36])
    ts = timestamps.bin_timestamps(datetime.datetime(2017,1,1,12,0), datetime.timedelta(hours=24))
    np.testing.assert_allclose(ts.time_deltas(np.timedelta64(1,"h")), [0,0,0,0,24])
    
@pytest.fixture
def timedpoints():
    ts = [
        datetime.datetime(2017,1,2,12,30),
        datetime.datetime(2017,1,2,14,23),
        datetime.datetime(2017,1,3,0,0),
        datetime.datetime(2017,1,3,4,5),
        datetime.datetime(2017,1,4,1,2)
        ]
    x = np.random.random(len(ts))
    y = np.random.random(len(ts))
    return TimedPoints.from_coords(ts, x, y)

def test_TimedPoints_bin(timedpoints):
    ts = timedpoints.bin_timestamps(datetime.datetime(2017,1,1), datetime.timedelta(days=1))
    np.testing.assert_allclose(ts.time_deltas(np.timedelta64(1,"h")) + 24, [24,24,48,48,72])
    np.testing.assert_allclose(ts.coords, timedpoints.coords)
    ts = timedpoints.bin_timestamps(datetime.datetime(2017,1,1), datetime.timedelta(hours=12))
    np.testing.assert_allclose(ts.time_deltas(np.timedelta64(1,"h")), [0,0,12,12,36])
    np.testing.assert_allclose(ts.coords, timedpoints.coords)
    ts = timedpoints.bin_timestamps(datetime.datetime(2017,1,1,12,0), datetime.timedelta(hours=24))
    np.testing.assert_allclose(ts.time_deltas(np.timedelta64(1,"h")), [0,0,0,0,24])
    np.testing.assert_allclose(ts.coords, timedpoints.coords)


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