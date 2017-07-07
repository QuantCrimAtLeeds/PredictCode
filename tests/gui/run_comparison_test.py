import pytest
import unittest.mock as mock
import numpy as np
import datetime

import open_cp.gui.run_comparison as run_comparison
import open_cp.gui.run_analysis as run_analysis
import open_cp.gui.predictors as predictors
import open_cp.gui.predictors.comparitor as comparitor

@pytest.fixture
def main_model():
    main_model = mock.MagicMock()
    t = np.array([np.datetime64("2017-04-01")])
    x = np.array([5])
    y = np.array([15])
    main_model.selected_by_crime_type_data.return_value = (t, x, y)
    return main_model

@pytest.fixture
def controller(main_model):
    controller = mock.MagicMock()
    controller.model = main_model
    return controller
    
@pytest.fixture
def result():
    key = run_analysis.TaskKey("proj_type", "grid_type", "pred_type",
        datetime.datetime(2017,5,1), datetime.timedelta(days=2))
    grid_pred = mock.MagicMock()
    pred_res_1 = run_analysis.PredictionResult(key, grid_pred)
    results = [pred_res_1]
    return run_analysis.RunAnalysisResult(results)

@mock.patch("open_cp.gui.run_comparison.run_analysis_view.RunAnalysisView")
def test_builds(controller, result):
    run_comparison.RunComparison(None, controller, result)

@pytest.fixture
def log_queue():
    import queue
    q = queue.Queue()
    predictors.set_queue_logging(q)
    return q


class LocatorMock():
    def __init__(self):
        self.pool_mock = mock.MagicMock()
    
    def get(self, value):
        assert value == "pool"
        return self.pool_mock
    
    def last_call(self):
        call = self.pool_mock.submit.call_args
        return call[0]


@pytest.fixture
def locator_mock():
    #with mock.patch("open_cp.gui.run_comparison.locator") as locator_mock:
    with mock.patch("open_cp.gui.run_comparison.locator", new=LocatorMock()) as locator_mock:
        yield locator_mock

def print_log(queue):
    if not queue.empty():
        import logging
        formatter = logging.Formatter("{asctime} {levelname} : {message}", style="{")
        while not queue.empty():
            record = queue.get()
            msg = formatter.format(record)
            print(msg)
            if msg.find("ERROR") > -1:
                raise Exception()

@pytest.fixture
@mock.patch("open_cp.gui.run_comparison.run_analysis_view.RunAnalysisView")
def run_com(view, controller, result, log_queue, locator_mock):
    return run_comparison.RunComparison(None, controller, result)

def test_runs(run_com, log_queue, locator_mock):
    run_com.run()
    print_log(log_queue)
    print(locator_mock.last_call())
    
def test_projection_tasks_none_found(run_com, log_queue):
    run_com.run()
    lookup = run_com.construct_projection_tasks()
    assert set(lookup.keys()) == {"proj_type"}
    assert lookup["proj_type"] is None
    print_log(log_queue)

@pytest.fixture
def projector_task():
    with mock.patch("open_cp.gui.run_comparison.run_analysis.RunAnalysisModel") as run_analysis_model:
        def task(x, y):
            return x, y
        run_analysis_model.return_value.projectors = {"proj_type": [task]}
        yield task

def test_projection_tasks_with_task(run_com, log_queue, projector_task):
    run_com.run()
    lookup = run_com.construct_projection_tasks()
    assert set(lookup.keys()) == {"proj_type"}
    assert lookup["proj_type"] is projector_task
    print_log(log_queue)

@pytest.fixture
def new_prediction():
    return mock.MagicMock()
    
@pytest.fixture
def adjust_task(new_prediction):
    at = mock.MagicMock()
    at.return_value = [new_prediction]
    return at

@pytest.fixture
def adjuster(adjust_task):
    ad = mock.MagicMock()
    ad.make_tasks.return_value = [adjust_task]
    ad.pprint.return_value = "adjuster_type"
    return ad

@pytest.fixture
def compare_task():
    return mock.MagicMock()

@pytest.fixture
def comparer(compare_task):
    com = mock.MagicMock()
    com.pprint.return_value = "comparer_type"
    com.make_tasks.return_value = [compare_task]
    return com

@pytest.fixture
def comparator_types(main_model, adjuster, comparer):
    def comparators_of_type(value):
        if value == comparitor.TYPE_ADJUST:
            return [adjuster]
        elif value == comparitor.TYPE_COMPARE_TO_REAL:
            return [comparer]
        elif value == comparitor.TYPE_TOP_LEVEL:
            return []
        else:
            raise ValueError()
    main_model.comparison_model.comparators_of_type = comparators_of_type
    return comparators_of_type    

def test_run_adjust_tasks(run_com, log_queue, locator_mock, comparator_types,
                          adjust_task, result, new_prediction):
    run_com.run()
    task, at_end = locator_mock.last_call()
    out = task()
    assert len(out) == 1
    assert out[0][0] == "adjuster_type"
    assert out[0][1].key == result.results[0].key
    assert out[0][1].prediction is new_prediction
    adjust_task.assert_called_with(None, [result.results[0].prediction])
    print_log(log_queue)

def test_stage2_exception_rethrown(run_com, comparator_types, locator_mock, log_queue):
    run_com.run()
    task, at_end = locator_mock.last_call()
    at_end(ValueError())
    with pytest.raises(Exception):
        print_log(log_queue)
        
def test_stage2(run_com, comparator_types, locator_mock, log_queue, compare_task,
                result, main_model, new_prediction):
    run_com.run()
    task, at_end = locator_mock.last_call()
    at_end(task())
    print_log(log_queue)
    
    task, at_end = locator_mock.last_call()
    out = task()
    print_log(log_queue)
    assert len(out) == 1
    assert out[0][0] is compare_task
    assert out[0][1].prediction == new_prediction
    t, x, y = main_model.selected_by_crime_type_data()
    assert np.all(out[0][2].timestamps == t)
    np.testing.assert_allclose(out[0][2].xcoords, x)
    np.testing.assert_allclose(out[0][2].ycoords, y)
    assert out[0][3] == "adjuster_type"
    assert out[0][4] == "comparer_type"
    
def test_ComparisonTaskWrapper():
    com_task = mock.MagicMock()
    result = mock.MagicMock()
    timed_points = mock.MagicMock()
    adjust_name = "akjdgjd"
    com_name = "hglnh"
    arg_list = [(com_task, result, timed_points, adjust_name, com_name)]
    wrapper = run_comparison.ComparisonTaskWrapper(arg_list)
    out = wrapper()
    com_task.assert_called_with(result.prediction, timed_points,
        result.key.prediction_date, result.key.prediction_length)
    assert len(out) == 1
    assert out[0].prediction_key == result.key
    assert repr(out[0].comparison_key) == repr(run_comparison.TaskKey(adjust_name, com_name))
    assert out[0].score == com_task.return_value
    
