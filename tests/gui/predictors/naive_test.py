import pytest
import unittest.mock as mock
import datetime
import numpy as np

import open_cp.gui.predictors.naive as naive
import open_cp.data
import open_cp.gui.predictors

@pytest.fixture
def project_task():
    task = mock.MagicMock()
    xcoords = [0, 10, 20]
    ycoords = [10, 20, 0]
    task.return_value = (xcoords, ycoords)
    yield task

@pytest.fixture
def analysis_model():
    model = mock.MagicMock()
    times = [datetime.datetime(2017,5,21,12,30), datetime.datetime(2017,5,21,13,0),
            datetime.datetime(2017,5,21,13,30)]
    xcoords = [1,2,3]
    ycoords = [4,5,6]
    model.selected_by_crime_type_data.return_value = (times, xcoords, ycoords)
    model.time_range = (datetime.datetime(2017,3,4,5,6), None, None, None)
    yield model

@pytest.fixture
def model():
    times = [datetime.datetime(2017,5,21,12,40), datetime.datetime(2017,5,21,13,10),
            datetime.datetime(2017,5,21,13,33)]
    model = mock.MagicMock()
    model.times = times
    yield model

@pytest.fixture
def grid_task():
    task = mock.MagicMock()
    mask = np.zeros((10, 15))
    task.return_value = open_cp.data.MaskedGrid(50, 50, 25, 35, mask)
    yield task

def standard_calls(provider, project_task, analysis_model, grid_task):
    """Checks that:
      - the provider produces exactly one task
      - we projected the coordinates in the standard way
    """
    tasks = provider.make_tasks()
    assert len(tasks) == 1
    
    subtask = tasks[0](analysis_model, grid_task, project_task)
    project_task.assert_called_once_with([1,2,3], [4,5,6])
    tp = grid_task.call_args[0][0]
    np.testing.assert_array_equal(tp.timestamps, [np.datetime64("2017-05-21T12:30"),
        np.datetime64("2017-05-21T13:00"), np.datetime64("2017-05-21T13:30")])
    np.testing.assert_array_equal(tp.xcoords, [0,10,20])
    np.testing.assert_array_equal(tp.ycoords, [10,20,0])
    return subtask

def test_CountingGrid_make_task(model, project_task, analysis_model, grid_task):
    standard_calls(naive.CountingGrid(model), project_task, analysis_model, grid_task)

def test_CountingGrid_make_task_no_date(model, project_task, analysis_model, grid_task):
    analysis_model.time_range = datetime.datetime(2017,5,21,13,31), None, None, None
    with pytest.raises(open_cp.gui.predictors.predictor.PredictionError):
        standard_calls(naive.CountingGrid(model), project_task, analysis_model, grid_task)
    
@mock.patch("open_cp.naive.CountingGridKernel")
def test_CountingGrid_subtask(model, project_task, analysis_model, grid_task):
    mock_cgk = open_cp.naive.CountingGridKernel
    mock_cgk.return_value = mock.MagicMock()

    subtask = standard_calls(naive.CountingGrid(model), project_task, analysis_model, grid_task)
    region = open_cp.data.RectangularRegion(xmin=25, ymin=35, xmax=25 + 15 * 50, ymax=535)
    mock_cgk.assert_called_once_with(50, 50, region)
    
    mock_cgk.return_value.predict.return_value = mock.MagicMock()
    prediction = subtask(datetime.datetime(2017,5,21,13,1), None)
    tp = mock_cgk.return_value.data
    np.testing.assert_array_equal(tp.timestamps, [np.datetime64("2017-05-21T12:30"),
        np.datetime64("2017-05-21T13:00")])
    np.testing.assert_array_equal(tp.xcoords, [0,10])
    np.testing.assert_array_equal(tp.ycoords, [10,20])

    mock_cgk.return_value.predict.assert_called_once_with()
    assert prediction is mock_cgk.return_value.predict.return_value


def test_ScipyKDE_make_task(model, project_task, analysis_model, grid_task):
    standard_calls(naive.ScipyKDE(model), project_task, analysis_model, grid_task)

def test_ScipyKDE_make_task_no_date(model, project_task, analysis_model, grid_task):
    analysis_model.time_range = datetime.datetime(2017,5,21,13,31), None, None, None
    with pytest.raises(open_cp.gui.predictors.predictor.PredictionError):
        standard_calls(naive.ScipyKDE(model), project_task, analysis_model, grid_task)

class OurContinuousPrediction(open_cp.predictors.ContinuousPrediction):
    def risk(self, x, y):
        return 1.0


@mock.patch("open_cp.naive.ScipyKDE")
def test_ScipyKDE_subtask(model, project_task, analysis_model, grid_task):
    mock_cgk = open_cp.naive.ScipyKDE
    mock_cgk.return_value = mock.MagicMock()

    subtask = standard_calls(naive.ScipyKDE(model), project_task, analysis_model, grid_task)
    mock_cgk.assert_called_once_with()
    
    mock_cgk.return_value.predict.return_value = OurContinuousPrediction(20, 30, 10, 20)
    prediction = subtask(datetime.datetime(2017,5,21,13,1), None)
    tp = mock_cgk.return_value.data
    np.testing.assert_array_equal(tp.timestamps, [np.datetime64("2017-05-21T12:30"),
        np.datetime64("2017-05-21T13:00")])
    np.testing.assert_array_equal(tp.xcoords, [0,10])
    np.testing.assert_array_equal(tp.ycoords, [10,20])

    mock_cgk.return_value.predict.assert_called_once_with()
    assert isinstance(prediction, open_cp.predictors.GridPredictionArray)
    assert prediction.xsize == 50
    assert prediction.ysize == 50
    assert prediction.xoffset == 25
    assert prediction.yoffset == 35
    assert prediction.xextent == 15
    assert prediction.yextent == 10

    np.testing.assert_allclose(prediction.intensity_matrix, np.zeros((10,15)) + 1)
