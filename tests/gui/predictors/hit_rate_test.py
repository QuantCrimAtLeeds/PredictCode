import pytest
import unittest.mock as mock

#import numpy as np

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
    # TODO: Should be a masked matrix...
    matrix = None
    return open_cp.predictors.GridPredictionArray(10, 10, matrix, 2, 4)

def test_make_tasks(hrm):
    tasks = hrm.make_tasks()
    assert len(tasks) == 1
    task = tasks[0]

