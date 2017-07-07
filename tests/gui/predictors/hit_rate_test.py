import pytest
import unittest.mock as mock

import numpy as np
import datetime

import open_cp.gui.predictors.hit_rate as hit_rate
from . import helper
import open_cp.data
import open_cp.predictors

@pytest.fixture
def model():
    import logging
    logging.basicConfig(level=logging.DEBUG)
    model_mock = mock.MagicMock()
    #model_mock.analysis_tools_model = mock.MagicMock()
    #model_mock.analysis_tools_model.projected_coords.return_value = (
    #    [0,1,2], [1,0,-1] )
    #model_mock.analysis_tools_model.coordinate_projector = mock.MagicMock()
    #model_mock.analysis_tools_model.coordinate_projector.return_value = None
    return model_mock

@pytest.fixture
def hrm(model):
    return hit_rate.HitRate(model)

def test_model(hrm):
    assert hrm.settings_string == "Coverage: 5%"

    hrm.coverage = 13
    assert hrm.settings_string == "Coverage: 13%"
    helper.serialise(hrm)

def test_serialisation(hrm):
    helper.serialise(hrm)

@pytest.fixture
def prediction():
    matrix = np.array([[1,2,3,4],[5,6,7,8]])
    mask = np.array([[True,False,False,True],[False,True,False,False]])
    matrix = np.ma.array(matrix, mask=mask)
    return open_cp.predictors.GridPredictionArray(10, 10, matrix, 0, 0)

@pytest.fixture
def timed_points():
    timestamps = [np.datetime64("2017-04-01") + np.timedelta64(1, "D") * i for i in range(8)]
    xcoords = 5 + 10 * np.array([0,1,2,3,0,1,2,3])
    ycoords = 5 + 10 * np.array([0,0,0,0,1,1,1,1])
    return open_cp.data.TimedPoints(timestamps, [xcoords, ycoords])

def test_make_tasks(hrm, prediction, timed_points):
    tasks = hrm.make_tasks()
    assert len(tasks) == 1
    task = tasks[0]
    
    # Should be 5% coverage
    assert task(prediction, timed_points, datetime.datetime(2017,4,1),
                datetime.timedelta(days=8)) == 0
    
    
    hrm.coverage = 59
    task = hrm.make_tasks()[0]
    assert task(prediction, timed_points, datetime.datetime(2017,4,1),
                datetime.timedelta(days=8)) == pytest.approx(2/8)
    
    hrm.coverage = 60
    task = hrm.make_tasks()[0]
    assert task(prediction, timed_points, datetime.datetime(2017,4,1),
                datetime.timedelta(days=8)) == pytest.approx(3/8)
    

    hrm.coverage = 60
    task = hrm.make_tasks()[0]
    assert task(prediction, timed_points, datetime.datetime(2017,4,2),
                datetime.timedelta(days=7)) == pytest.approx(3/7)
