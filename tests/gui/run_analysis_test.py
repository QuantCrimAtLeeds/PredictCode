import pytest
import unittest.mock as mock

import open_cp.gui.run_analysis as run_analysis
import open_cp.gui.predictors as predictors
from open_cp.gui.common import CoordType
import open_cp.gui.analysis as analysis
import open_cp.predictors
import datetime

@pytest.fixture
def log_queue():
    import queue
    return queue.Queue()

@pytest.fixture
def runAnalysis(log_queue):
    predictors.set_queue_logging(log_queue)
    class Model():
        def selected_by_crime_type_data(self):
            return self.times, self.xcoords, self.ycoords
    model = Model()
    model.analysis_tools_model = analysis.AnalysisToolsModel(model)
    model.comparison_model = analysis.ComparisonModel(model)
    model.coord_type = CoordType.XY
    model.times = [datetime.datetime(2017,5,10,12,30)]
    model.xcoords = [3]
    model.ycoords = [17]
    model.time_range = (datetime.datetime(2017,5,4,0,0), None,
            datetime.datetime(2017,5,10,11,30),
            datetime.datetime(2017,5,11,13,30))
    model.analysis_tools_model.add(predictors.grid.GridProvider)
    model.comparison_model.add(predictors.pred_type.PredType)
    model.analysis_tools_model.add(predictors.naive.CountingGrid)
    controller = mock.MagicMock()
    controller.model = model
    with mock.patch("open_cp.gui.tk.run_analysis_view.RunAnalysisView") as mock_view:
        yield run_analysis.RunAnalysis(None, controller)

@pytest.fixture
def locator_mock():
    with mock.patch("open_cp.gui.run_analysis.locator") as locator_mock:
        yield locator_mock

def print_log(queue):
    if not queue.empty():
        import logging
        formatter = logging.Formatter("{asctime} {levelname} : {message}", style="{")
        while not queue.empty():
            record = queue.get()
            print(formatter.format(record))

def get_thread(locator_mock):
    pool = locator_mock.get("pool")
    assert len(pool.method_calls) == 1
    name, args, kwargs = pool.method_calls[0]
    assert name == "submit"
    off_thread = args[0]
    return off_thread

def test_controller_runs(runAnalysis, log_queue, locator_mock):
    runAnalysis.run()
    get_thread(locator_mock)
    print_log(log_queue)

def test_model(runAnalysis):
    model = run_analysis.RunAnalysisModel(runAnalysis, runAnalysis.main_model)

    assert len(model.projectors) == 1
    assert len(model.projectors['Coordinates already projected']) == 1
    import open_cp.gui.predictors.lonlat
    assert isinstance(model.projectors['Coordinates already projected'][0],
        open_cp.gui.predictors.lonlat.PassThrough.Task)

    assert len(model.grids) == 1
    assert len(model.grids['Grid 100x100m @ (0m, 0m)']) == 1
    import open_cp.gui.predictors.grid
    assert isinstance(model.grids['Grid 100x100m @ (0m, 0m)'][0],
        open_cp.gui.predictors.grid.GridProvider.Task)

    assert len(model.grid_prediction_tasks) == 1
    assert len(model.grid_prediction_tasks['Counting Grid naive predictor']) == 1
    import open_cp.gui.predictors.naive
    assert isinstance(model.grid_prediction_tasks['Counting Grid naive predictor'][0],
        open_cp.gui.predictors.naive.CountingGrid.Task)

@pytest.fixture
def pool():
    with mock.patch("open_cp.gui.run_analysis.pool") as pool_mock:
        yield pool_mock

def test_controller_tasks(runAnalysis, log_queue, locator_mock):
    runAnalysis.run()
    off_thread = get_thread(locator_mock)
    off_thread()

    assert str(off_thread.results[0][0]) == "projection: Coordinates already projected, grid: Grid 100x100m @ (0m, 0m), prediction_type: Counting Grid naive predictor, prediction_date: 2017-05-10 00:00:00, prediction_length: 1 day, 0:00:00"
    assert isinstance(off_thread.results[0][1], open_cp.predictors.GridPredictionArray)
