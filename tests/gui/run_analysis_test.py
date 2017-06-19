import pytest
import unittest.mock as mock

import open_cp.gui.run_analysis as run_analysis
import open_cp.gui.predictors as predictors
from open_cp.gui.common import CoordType

@pytest.fixture
def log_queue():
    import queue
    return queue.Queue()

@pytest.fixture
def runAnalysis(log_queue):
    predictors.set_queue_logging(log_queue)
    class Model():
        pass
    model = Model()
    with mock.patch("open_cp.gui.tk.run_analysis_view.RunAnalysisView") as mock_view:
        yield run_analysis.RunAnalysis(None, model)

def log_errors(queue):
    if not queue.empty():
        import logging
        formatter = logging.Formatter("{asctime} {levelname} : {message}", style="{")
        while not queue.empty():
            record = queue.get()
            print(formatter.format(record))
        #raise Exception()


def test_controller(runAnalysis, log_queue):
    model = runAnalysis.main_model
    model.coord_type = CoordType.XY
    runAnalysis.run()
    log_errors(log_queue)

