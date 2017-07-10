"""
kde
~~~

A variety of "other" KDE methods, not drawn directly from the literature.
"""

from . import predictor
import open_cp.predictors
import tkinter as tk
import tkinter.ttk as ttk
import open_cp.gui.tk.util as util
import open_cp.gui.tk.richtext as richtext

_text = {
    "main" : ("Other Kernel Density Estimation methods\n\n"
        + "A selection of other KDE (kernel density estimation) methods; to be compared with "
        + "the 'Scipy KDE naive estimator' (we here offer more options) or the 'Retrospective' and "
        + "'Prospective' hotspotting algorithms (which are explicitly explained in the scientific "
        + "literature, whereas the methods here are not).\n"
        + "We offer two ways to customise the algorithm: the choice of the KDE algorithm to be "
        + "applied to the spacial coordinates (estimating a continuous risk profile from the location "
        + "of events) and how (or not) to weight time.\n\n"
        + "Training Data usage: Can either be ignored, or directly used."
        ),
    "no_data" : "No data points found in time range: have enough crime types been selected?",
    
}

class KDE(predictor.Predictor):
    def __init__(self, model):
        super().__init__(model)
        pass
    
    @staticmethod
    def describe():
        return "Kernel density estimation predictor"

    @staticmethod
    def order():
        return predictor._TYPE_GRID_PREDICTOR

    def make_view(self, parent):
        return KDEView(parent)

    @property
    def name(self):
        # TODO: Other settings?
        return "KDE predictor"
        
    @property
    def settings_string(self):
        # TODO
        return None
        
    def to_dict(self):
        return {}
    
    def from_dict(self, data):
        pass
    
    def make_tasks(self):
        return [self.Task()]
        
    class Task(predictor.GridPredictorTask):
        def __init__(self):
            super().__init__()

        def __call__(self, analysis_model, grid_task, project_task):
            timed_points = self.projected_data(analysis_model, project_task)
            # TODO: Adjust this...
            training_start, _, _, _ = analysis_model.time_range
            timed_points = timed_points[timed_points.timestamps >= training_start]
            if timed_points.number_data_points == 0:
                raise predictor.PredictionError(_text["no_data"])
            grid = grid_task(timed_points)
            # TODO: Return...?

    class SubTask(predictor.SingleGridPredictor):
        def __init__(self, timed_points, grid):
            super().__init__()
            self._timed_points = timed_points
            # TODO

        def __call__(self, predict_time, length):
            pass


class KDEView(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        util.stretchy_rows_cols(self, [0], [0])
        self._text = richtext.RichText(self, height=12, scroll="v")
        self._text.grid(sticky=tk.NSEW)
        self._text.add_text(_text["main"])


def test(root):
    ll = KDE(predictor.test_model())
    predictor.test_harness(ll, root)
