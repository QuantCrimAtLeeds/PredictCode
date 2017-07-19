"""
sepp2
~~~~~

The grid-based SEPP method.
"""

from . import predictor
import open_cp.predictors
import tkinter as tk
import tkinter.ttk as ttk
import open_cp.seppexp
import open_cp.gui.tk.util as util
import open_cp.gui.tk.richtext as richtext
#import numpy as np

_text = {
    "main" : ("Grid based Self-Excited Point Process predictor.\n\n"
        + "This is an explicitly grid-based prediction method.  We assume that crime events occur at random, "
        + "with a varying 'intensity'.  The intensity has two components: a 'background' rate which is constant "
        + "in each grid cell, but varies between cells; and a 'triggered' component, meaning that a crime "
        + "event causes a time-localised increase in intensity.  Here the 'trigger' is modelled as an exponential "
        + "decay which has parameters which are constant acrsss all grid cells.  We use past data to estimate these "
        + "parameters and the background rate in each cell.\n"
        + "Training Data usage: The training data is used to estimate the parameters of the model.  These parameters "
        + "are then held fixed for each prediction.\n\n"
        + "WARNING: We have found that this model often performs poorly on real data.  If you obtain a "
        + "'convergence failed' error, try increasing the size of the grid."
        ),
    "no_data" : "No data points found in training time range: have enough crime types been selected?",

}

class SEPP(predictor.Predictor):
    def __init__(self, model):
        super().__init__(model)
    
    @staticmethod
    def describe():
        return "Grid based self-exciting point process model"

    @staticmethod
    def order():
        return predictor._TYPE_GRID_PREDICTOR

    def make_view(self, parent):
        return SEPPView(parent)

    @property
    def name(self):
        return "Grid based SEPP"
        
    @property
    def settings_string(self):
        return None
        
    def to_dict(self):
        return {}
    
    def from_dict(self, data):
        pass
    
    def make_tasks(self):
        return [self.Task()]
        
    class Task(predictor.GridPredictorTask):
        def __init__(self):
            super().__init__(True)

        def __call__(self, analysis_model, grid_task, project_task):
            timed_points = self.projected_data(analysis_model, project_task)
            training_start, training_end, _, _ = analysis_model.time_range
            mask = (timed_points.timestamps >= training_start) & (timed_points.timestamps < training_end)
            training_points = timed_points[mask]
            if training_points.number_data_points == 0:
                raise predictor.PredictionError(_text["no_data"])
            grid = grid_task(timed_points)
            
            trainer = open_cp.seppexp.SEPPTrainer(grid=grid)
            trainer.data = training_points
            pred = trainer.train(iterations=40, use_corrected=True)
            pred.data = timed_points
            
            return SEPP.SubTask(pred)        

    class SubTask(predictor.SingleGridPredictor):
        def __init__(self, pred):
            super().__init__(True)
            self._pred = pred

        def __call__(self, predict_time, length=None):
            return self._pred.predict(predict_time)


class SEPPView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        util.stretchy_rows_cols(self, [0], [0])
        self._text = richtext.RichText(self, height=14, scroll="v")
        self._text.grid(sticky=tk.NSEW)
        self._text.add_text(_text["main"])
        

def test(root):
    ll = SEPP(predictor.test_model())
    predictor.test_harness(ll, root)
