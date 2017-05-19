import pytest

import open_cp.knox as knox
import open_cp.data as data

import numpy as np
import scipy.spatial.distance as distance
import datetime

def test_distances():
    pts = np.random.random(size=(10, 2))
    np.testing.assert_allclose(distance.pdist(pts), knox.distances(pts))

def test_distances_1D():
    pts = np.random.random(size=10)
    expected = []
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            expected.append(abs(pts[i] - pts[j]))
    np.testing.assert_allclose(knox.distances(pts), expected)

def test_Knox_space_bins():
    k = knox.Knox()
    k.space_bins = [[1,2], (4,5), [-1,5]]
    np.testing.assert_allclose(k.space_bins, [[1,2],[4,5],[-1,5]])

def test_Knox_time_bins():
    k = knox.Knox()
    td = datetime.timedelta
    k.time_bins = [[td(days=1), td(days=2)],
                    [td(hours=1), td(hours=5)]]
    assert(len(k.time_bins) == 2)
    assert(k.time_bins[0][0] / np.timedelta64(1,"D") == 1.0)
    assert(k.time_bins[0][1] / np.timedelta64(1,"D") == 2.0)
    assert(k.time_bins[1][0] / np.timedelta64(1,"m") == 60.0)
    assert(k.time_bins[1][1] / np.timedelta64(1,"m") == 300.0)

def test_Knox_time_bins_set():
    k = knox.Knox()
    k.set_time_bins([[1,2], [4,6]], unit="minutes")
    assert(len(k.time_bins) == 2)
    assert(k.time_bins[0][0] / np.timedelta64(1,"s") == 60.0)
    assert(k.time_bins[0][1] / np.timedelta64(1,"s") == 120.0)
    assert(k.time_bins[1][0] / np.timedelta64(1,"m") == 4.0)
    assert(k.time_bins[1][1] / np.timedelta64(1,"m") == 6.0)

@pytest.fixture
def data1():
    k = knox.Knox()
    times = [datetime.datetime(2016,1,1), datetime.datetime(2016,1,3)]
    k.data = data.TimedPoints.from_coords(times, [1,2], [5,5])
    return k

def test_calculate(data1):
    data1.space_bins = [[0, 2]]
    data1.set_time_bins([[0, 50]], "hours")
    cells = data1.calculate()
    assert( cells.shape == (1,1,2) )
    assert( cells[0][0][0] == 1 )
    assert( cells[0][0][1] == 999/1000 )

@pytest.fixture
def data2():
    # Distances between points is    1, sqrt(2), 1
    # Time between points is (days)  1, 2,       1
    k = knox.Knox()
    times = [datetime.datetime(2016,1,1), datetime.datetime(2016,1,2),
        datetime.datetime(2016,1,3)]
    k.data = data.TimedPoints.from_coords(times, [1,2,2], [5,5,6])
    return k

def test_calculate_two(data2):
    data2.space_bins = [[0.9, 1.1]]
    data2.set_time_bins([[0, 25]], "hours")
    cells = data2.calculate()
    assert( cells.shape == (1,1,2) )
    assert( cells[0][0][0] == 2 )
    # Shuffle to 112 112 121 121 211 211
    # 1/3 of time we get 2, otherwise get 1
    # Evil non-deterministic test
    assert( abs(cells[0][0][1] - 1/3) < 0.2 )

@pytest.fixture
def data3():
    # Distances between points is     2, 1, 3
    # Time between points is (hours)  1, 3, 2
    k = knox.Knox()
    times = [datetime.datetime(2016,1,1,1), datetime.datetime(2016,1,1,2),
        datetime.datetime(2016,1,1,4)]
    k.data = data.TimedPoints.from_coords(times, [1,1,1], [5, 7, 4])
    return k

def test_calculate_two(data3):
    data3.space_bins = [[0.9, 1.1], [0.9, 2.1], [1.5, 2.5]]
    data3.set_time_bins([(0,3), (2,3)], "hours")
    cells = data3.calculate()
    assert( cells.shape == (3,2,2) )
    assert( cells[0][0][0] == 1 )
    assert( cells[1][0][0] == 2 )
    assert( cells[2][0][0] == 1 )
    assert( cells[0][1][0] == 1 )
    assert( cells[1][1][0] == 1 )
    assert( cells[2][1][0] == 0 )
