import pytest
import open_cp.prohotspot as testmod

import numpy as np
from datetime import datetime, timedelta
from unittest import mock
import open_cp

# --- Distance tests ---

def test_DistanceDiagonalsSame():
    distance = testmod.DistanceDiagonalsSame()
    assert( distance(1, 1, 2, 2) == 1 )
    assert( distance(1, 1, 1, 2) == 1 )
    assert( distance(1, 1, 1, 1) == 0 )
    assert( distance(2, 1, 1, 2) == 1 )
    assert( distance(2, 1, 4, 3) == 2 )
    
def test_DistanceDiagonalsDifferent():
    distance = testmod.DistanceDiagonalsDifferent()
    assert( distance(1, 1, 2, 2) == 2 )
    assert( distance(1, 1, 1, 2) == 1 )
    assert( distance(1, 1, 2, 1) == 1 )
    assert( distance(1, 1, 4, 5) == 7 )
    
def test_ClassicWeight():
    weight = testmod.ClassicWeight()
    weight.time_bandwidth = 5
    
    assert( weight(5, 0) == 0 )
    assert( weight(0, 8) == 0 )
    for i in range(5):
        assert( weight(i, 0) == pytest.approx(1/(i+1)) )
    for i in range(8):
        assert( weight(0, i) == pytest.approx(1/(i+1)) )
    assert( weight(1, 1) == pytest.approx(1/4) )
    assert( weight(2, 1) == pytest.approx(1/6) )
    assert( weight(1, 2) == pytest.approx(1/6) )
    
def __test_ClassicWeight_vectorised():
    weight = testmod.ClassicWeight()
    weight.space_bandwidth = 5
    dt = np.asarray([8,0,0,1,2,3,4,5,6,7,1,2])
    dd = np.asarray([0,5,0,0,0,0,0,0,0,0,1,1])
    result = weight(dt, dd)
    np.testing.assert_allclose(result, [0,0,1,1/2,1/3,1/4,1/5,1/6,1/7,1/8,1/4,1/6])

def test_predict_wrong_times():
    pred = testmod.ProspectiveHotSpot(None)
    with pytest.raises(ValueError):
        pred.predict(datetime(2017, 3, 10, 12, 30), datetime(2017, 3, 10, 0))

def a_valid_predictor():
    region = open_cp.RectangularRegion(0,150,0,150)
    predictor = testmod.ProspectiveHotSpot(region)
    timestamps = [datetime(2017,3,1)]
    xcoords = [50]
    ycoords = [50]
    predictor.data = open_cp.TimedPoints.from_coords(timestamps, xcoords, ycoords)
    return predictor

def test_ProspectiveHotSpot_correct_return():
    p = a_valid_predictor()
    prediction = p.predict(datetime(2017,3,2), datetime(2017,3,2))
    assert( prediction.xsize == 50 )
    assert( prediction.ysize == 50 )
    expected = np.zeros((3,3)) + 1/2
    expected[1][1] = 1
    for i in range(3):
        for j in range(3):
            assert( prediction.grid_risk(i, j) == expected[j][i] )
    assert( prediction.grid_risk(-1,0) == 0 )
    assert( prediction.grid_risk(3,0) == 0 )
    assert( prediction.grid_risk(0,3) == 0 )

def test_ProspectiveHotSpot_filters_by_time():
    p = a_valid_predictor()
    prediction = p.predict(datetime(2017,2,1), datetime(2017,3,10))
    assert( prediction.grid_risk(0,0) == 0 )

def test_ProspectiveHotSpot_uses_predict_time():
    p = a_valid_predictor()
    expected = np.zeros((3,3)) + 1/2
    expected[1][1] = 1

    prediction = p.predict(datetime(2017,3,2), datetime(2017,3,7))
    np.testing.assert_allclose(prediction.intensity_matrix, expected)
    
    prediction = p.predict(datetime(2017,3,2), datetime(2017,3,8))
    np.testing.assert_allclose(prediction.intensity_matrix, expected * 0.5)

    prediction = p.predict(datetime(2017,3,2), datetime(2017,3,14))
    np.testing.assert_allclose(prediction.intensity_matrix, expected * 0.5)

    prediction = p.predict(datetime(2017,3,2), datetime(2017,3,15))
    np.testing.assert_allclose(prediction.intensity_matrix, expected * (1/3))

def test_ProspectiveHotSpot_additive():
    p = a_valid_predictor()
    timestamps = [datetime(2017,3,1), datetime(2017,3,1)]
    xcoords = [49, 101]
    ycoords = [49, 101]
    p.data = open_cp.TimedPoints.from_coords(timestamps, xcoords, ycoords)

    expected = np.asarray([[1,1/2,1/3], [1/2,1/2,1/3], [1/3,1/3,1/3]])
    expected += np.asarray([[1/3,1/3,1/3], [1/3,1/2,1/2], [1/3,1/2,1]])
    prediction = p.predict(datetime(2017,3,2), datetime(2017,3,2))
    np.testing.assert_allclose(prediction.intensity_matrix, expected)
