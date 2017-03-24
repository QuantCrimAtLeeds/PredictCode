#import pytest
from pytest import approx, raises
import open_cp.retrohotspot as testmod

import numpy as np
import datetime
#from unittest import mock
import open_cp

def test_Quartic():
    w = testmod.Quartic()
    assert( w(0, 0) == approx(1) )
    assert( w(50, 0) == approx((15/16)**2) )
    assert( w(0, 50) == approx((15/16)**2) )
    assert( w(0, 200) == 0 )
    assert( w(100, 100) == approx( 0.25 ) )
    assert( w(150, 150) == 0 )
    
def test_Quartic_can_pass_array():
    w = testmod.Quartic()
    xcoords = np.array([0,50,0,0,100,150])
    ycoords = np.array([0,0,50,200,100,150])
    expected = np.array([1,(15/16)**2,(15/16)**2,0,0.25,0])
    np.testing.assert_allclose(w(xcoords, ycoords), expected)

def test_RetroHotSpot_data_set():
    r = testmod.RetroHotSpot()
    with raises(TypeError):
        r.data = "bib"
        
# Again, having tested the real weight, we'll use a test weight

class TestWeight(testmod.Weight):
    def __call__(self, x, y):
        return ((abs(x) <= 50) & (abs(y) <= 50)).astype(np.float)
        
def test_RetroHotSpot_single_event():
    r = testmod.RetroHotSpot()
    r.weight = TestWeight()
    r.data = open_cp.TimedPoints.from_coords([datetime.datetime(2017,3,1)], [50], [60])
    prediction = r.predict()
    assert( prediction.risk(50, 60) == 1 )
    assert( prediction.risk(0, 10) == 1 )
    assert( prediction.risk(0, 9) == 0 )
    
def a_valid_RetroHotSpot():
    r = testmod.RetroHotSpot()
    r.weight = TestWeight()
    r.data = open_cp.TimedPoints.from_coords([datetime.datetime(2017,3,1),
        datetime.datetime(2017,3,2), datetime.datetime(2017,3,3)],
        [50, 100, 125], [50, 100, 25])
    return r    
    
def test_RetroHotSpot_multiple_events():
    r = a_valid_RetroHotSpot()
    prediction = r.predict()
    assert( prediction.risk(40, 40) == 1 )
    assert( prediction.risk(140, 130) == 1 )
    assert( prediction.risk(160, 20) == 1 )
    assert( prediction.risk(80, 60) == 3 )
    assert( prediction.risk(60, 90) == 2 )
    

def test_RetroHostSpot_filter_events_after_time():
    r = a_valid_RetroHotSpot()
    prediction = r.predict(start_time = datetime.datetime(2017,3,2))
    assert( prediction.risk(40, 40) == 0 )
    assert( prediction.risk(80, 60) == 2 )

def test_RetroHostSpot_filter_events_before_time():
    r = a_valid_RetroHotSpot()
    prediction = r.predict(end_time = datetime.datetime(2017,3,2))
    assert( prediction.risk(40, 40) == 1 )
    assert( prediction.risk(80, 60) == 2 )
    assert( prediction.risk(125, 25) == 0 )

def test_RetroHostSpot_filter_events_by_time():
    r = a_valid_RetroHotSpot()
    prediction = r.predict(start_time = datetime.datetime(2017,3,3),
            end_time = datetime.datetime(2017,3,2))
    assert( prediction.risk(80, 60) == 0 )
