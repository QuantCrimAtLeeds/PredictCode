import pytest
import unittest.mock as mock
import datetime
import numpy as np

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
