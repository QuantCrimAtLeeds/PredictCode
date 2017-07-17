import numpy as np
import pytest
import unittest.mock as mock

import open_cp
import open_cp.seppexp as testmod

def test__iter_array():
    a = np.array([1,2,3,4])
    assert(list(testmod._iter_array(a)) == [(0,),(1,),(2,),(3,)])

    a = np.empty((2,3))
    assert(list(testmod._iter_array(a)) == [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)])

def test_p_matrix():
    size = 10
    p = np.sort(np.random.random(size))
    for i in range(len(p)-1):
        assert(p[i] <= p[i+1])
    omega, theta, mu = np.random.random(3)
    def g(t):
        return omega * theta * np.exp(-omega * t)

    expected = np.zeros((size, size))
    for j in range(size):
        expected[j][j] = mu
        for i in range(j):
            expected[i][j] = g(p[j] - p[i])
    for j in range(size):
        expected[:,j] /= np.sum(expected[:,j])

    np.testing.assert_allclose(expected, testmod.p_matrix(p, omega, theta, mu))

def some_cells(rows, cols, rate):
    cells = [np.sort(np.random.random(np.random.poisson(rate))) for _ in range(rows*cols)]
    cells = np.asarray(cells).reshape((rows, cols))
    return cells

# Test two implementations: not perfect, but I guess I am mostly concerned that my
# clever numpy implementation does do what I expect it to...
def test_maximisation():
    cells = some_cells(4, 7, 100)
    omega, theta = np.random.random(2)
    mu = np.random.random((4, 7))
    got = testmod.maximisation(cells, omega, theta, mu, 100)
    want = testmod._slow_maximisation(cells, omega, theta, mu, 100)
    assert(got[0] == pytest.approx(want[0]))
    assert(got[1] == pytest.approx(want[1]))
    np.testing.assert_allclose(want[2], got[2])

def test_maximisation_corrected():
    cells = some_cells(4, 7, 100)
    omega, theta = np.random.random(2)
    mu = np.random.random((4, 7))
    got = testmod.maximisation_corrected(cells, omega, theta, mu, 100)
    want = testmod._slow_maximisation_corrected(cells, omega, theta, mu, 100)
    assert(got[0] == pytest.approx(want[0]))
    assert(got[1] == pytest.approx(want[1]))
    np.testing.assert_allclose(want[2], got[2])

def test__make_cells():
    region = open_cp.RectangularRegion(0, 100, 0, 100)
    events = mock.Mock()
    events.xcoords = np.asarray([0, 25, 26, 90, 90])
    events.ycoords = np.asarray([0, 0, 1, 90, 90])
    times = [5, 6, 7, 1, 2]
    cells = testmod._make_cells(region, 20, events, times)
    assert[cells[0,0] == [5]]
    assert[cells[0,1] == [6, 7]]
    assert[cells[4,4] == [1, 2]]
    assert[cells[3,3] == []]

def a_predictor():
    region = open_cp.RectangularRegion(0, 100, 0, 200)
    mu = np.random.random((10,5))
    predictor = testmod.SEPPPredictor(region=region, grid_size=20, omega=1.2, theta=0.7, mu=mu)
    return mu, predictor

def test_SEPP_Predictor_background_rate():
    mu, predictor = a_predictor()
    assert(predictor.background_rate(2, 3) == mu[3,2])

def test_SEPP_Predictor_background_prediction():
    mu, predictor = a_predictor()
    prediction = predictor.background_prediction()
    np.testing.assert_allclose(prediction.intensity_matrix, mu)

def test_SEPP_Predictor_predict_wrong_region_size():
    region = open_cp.RectangularRegion(0, 100, 0, 150)
    mu = np.random.random((10,5))
    predictor = testmod.SEPPPredictor(region=region, grid_size=20, omega=1.2, theta=0.7, mu=mu)
    predictor.data = open_cp.TimedPoints.from_coords([], [], [])
    with pytest.raises(ValueError):
        predictor.predict(np.datetime64("2018-01-01"))

def test_SEPP_Predictor_predict_wrong_region_size2():
    mu, predictor = a_predictor()
    times = [np.datetime64("2017-12-31"), np.datetime64("2017-12-31T19:00"), np.datetime64("2017-12-31T19:30"), np.datetime64("2018-01-01")]
    xcoords = [10, 90, 92, 0]
    ycoords = [25, 190, 194, 0]
    predictor.data = open_cp.TimedPoints.from_coords(times, xcoords, ycoords)
    prediction = predictor.predict(np.datetime64("2018-01-01"))
    expected = np.array(mu)
    expected[1, 0] += 0.7 * 1.2 * np.exp(-1.2 * 24 * 60)
    expected[9, 4] += 0.7 * 1.2 * np.exp(-1.2 * 5 * 60)
    expected[9, 4] += 0.7 * 1.2 * np.exp(-1.2 * 4.5 * 60)
    for y in range(10):
        for x in range(5):
            assert(prediction.grid_risk(x,y) == pytest.approx(expected[y,x]))
