import pytest
from pytest import approx
import open_cp.prohotspot.predictor as testmod

import numpy as _np
from datetime import datetime, timedelta
import open_cp

Cell = open_cp.RectangularRegion

def test_weight_diags_same_distance():
    weight = testmod.ClassicDiagonalsSame()
    assert( weight.distance(1, 1, 2, 2) == 1 )
    assert( weight.distance(1, 1, 1, 2) == 1 )
    assert( weight.distance(1, 1, 1, 1) == 0 )
    assert( weight.distance(2, 1, 1, 2) == 1 )
    assert( weight.distance(2, 1, 4, 3) == 2 )
    

def test_weight_diags_same_weight():
    weight = testmod.ClassicDiagonalsSame()
    cell = Cell(10, 20, 50, 60)
    
    # Too far in past
    assert( weight(cell, _np.timedelta64(8, "W"), 0, 0) == 0 )
    
    # Too distant
    assert( weight(cell, _np.timedelta64(1, "W"), 1000, 0) == 0 )
    
    # Same cell and time
    assert( weight(cell, _np.timedelta64(0), 15, 55) == 1 )
    assert( weight(cell, _np.timedelta64(0), 10, 50) == 1 )
    assert( weight(cell, _np.timedelta64(0), 20 - 0.1, 60 - 0.1) == 1 )
    
    # Time tests
    assert( weight(cell, _np.timedelta64(1, "W"), 15, 55) == approx(0.5) )
    assert( weight(cell, _np.timedelta64(9, "D"), 15, 55) == approx(0.5) )
    assert( weight(cell, _np.timedelta64(13, "D"), 15, 55) == approx(0.5) )
    assert( weight(cell, _np.timedelta64(14, "D"), 15, 55) == approx(1 / 3) )
    assert( weight(cell, _np.timedelta64(20, "D"), 15, 55) == approx(1 / 3) )
    assert( weight(cell, _np.timedelta64(21, "D"), 15, 55) == approx(1 / 4) )
    assert( weight(cell, _np.timedelta64(7 * 7 - 1, "D"), 15, 55) == approx(1 / 7) )
    assert( weight(cell, _np.timedelta64(7 * 7, "D"), 15, 55) == 0 )
    
    # Space tests
    assert( weight(cell, _np.timedelta64(0), 20, 55) == approx(1 / 2) )
    assert( weight(cell, _np.timedelta64(0), 30, 55) == approx(1 / 3) )
    assert( weight(cell, _np.timedelta64(0), 40, 55) == approx(1 / 4) )
    assert( weight(cell, _np.timedelta64(0), 399, 55) == approx(1 / 39) )
    assert( weight(cell, _np.timedelta64(0), 400, 55) == 0 )
    
    # Combined
    assert( weight(cell, _np.timedelta64(1, "W"), 20, 65) == approx(1 / 4) )
    assert( weight(cell, _np.timedelta64(3, "W"), 30, 65) == approx(1 / (4 * 3)) )
    
    
def test_weight_diags_different_distance():
    weight = testmod.ClassicDiagonalsDifferent()
    assert( weight.distance(1, 1, 2, 2) == 2 )
    assert( weight.distance(1, 1, 1, 2) == 1 )
    assert( weight.distance(1, 1, 2, 1) == 1 )
    assert( weight.distance(1, 1, 4, 5) == 7 )
    
    
def test_predict_wrong_times():
    pred = testmod.ProspectiveHotSpot(Cell(0, 0, 0, 0))
    with pytest.raises(ValueError):
        pred.predict(datetime(2017, 3, 10, 12, 30), datetime(2017, 3, 10, 0))

def test_predict_single_event():
    region = open_cp.RectangularRegion(0, 150, 0, 150)
    cutoff = datetime(2017, 3, 10, 12, 30)
    predict = cutoff
    times = [cutoff]
    coords = [ [75, 75] ]
    data = open_cp.TimedPoints(times, coords)
    
    pred = testmod.ProspectiveHotSpot(region)
    pred.data = data
    prediction = pred.predict(cutoff, predict)
    
    wanted = [ [0.5, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 0.5] ]
    for i in range(3):
        for j in range(3):
            assert( prediction.grid_risk(i, j) == wanted[j][i] )
            
    assert( prediction.grid_risk(3, 0) == 0 )
    assert( prediction.grid_risk(0, 3) == 0 )
    
def test_predict_filter_by_time():
    region = open_cp.RectangularRegion(0, 100, 0, 100)
    cutoff = datetime(2017, 3, 10, 12, 30)
    predict = datetime(2017, 3, 17, 12, 30)
    times = [cutoff, cutoff + timedelta(hours = 5)]
    coords = [ [25, 25], [75, 75] ]
    data = open_cp.TimedPoints(times, coords)
    
    pred = testmod.ProspectiveHotSpot(region)
    pred.data = data
    prediction = pred.predict(cutoff, predict)
    
    assert( prediction.grid_risk(0, 0) == approx(1 / 2) )
    assert( prediction.grid_risk(0, 1) == approx(1 / 4) )
    assert( prediction.grid_risk(1, 0) == approx(1 / 4) )
    assert( prediction.grid_risk(1, 1) == approx(1 / 4) )

def test_predict_multiple_events():
    region = open_cp.RectangularRegion(0, 100, 0, 100)
    cutoff = datetime(2017, 3, 15, 12, 30)
    predict = datetime(2017, 3, 17, 12, 30)
    times = [cutoff - timedelta(days = 5), cutoff]
    coords = [ [25, 25], [75, 75] ]
    data = open_cp.TimedPoints(times, coords)
    
    pred = testmod.ProspectiveHotSpot(region)
    pred.data = data
    prediction = pred.predict(cutoff, predict)
    
    assert( prediction.grid_risk(0, 0) == approx(1 / 2 + 1 / 2) )
    assert( prediction.grid_risk(0, 1) == approx(1 / 4 + 1 / 2) )
    assert( prediction.grid_risk(1, 0) == approx(1 / 4 + 1 / 2) )
    assert( prediction.grid_risk(1, 1) == approx(1 / 4 + 1) )
