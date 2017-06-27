"""
retro
~~~~~

Uses the "retro-hotspotting" technique.
"""

from . import predictor
import open_cp.retrohotspot
import tkinter as tk
import tkinter.ttk as ttk
import open_cp.gui.tk.util as util
import open_cp.gui.tk.richtext as richtext
import numpy as np

_text = {
    "main" : ("Retro-Hotspotting Grid Predictor.\n\n"
            + "All data in a 'window' before the prediction point is used; other data is ignored.  Only the location (and not the times) of"
            + "the events are used.  Around each a point a 'kernel' is laid down, and then these are summed to produce a risk estimate.  "
            + "As such, this can be considered to be a form of Kernel Density Estimation.\n"
            + "This version applies the grid at the very start, assigning points to their grid cells.  This is truer to the original literature,"
            + "but with modern computer systems it is unnecessary, and is likely to be too much of an approximation.\n"
            + "Training Data usage: Ignored.  For each prediction point, all data in a 'window' before the prediction point is used."
        ),
}

class RetroHotspot(predictor.Predictor):
    def __init__(self, model):
        super().__init__(model)
        pass
    
    @staticmethod
    def describe():
        return "RetroHotspot grid predictor"

    @staticmethod
    def order():
        return predictor._TYPE_GRID_PREDICTOR

    def make_view(self, parent):
        return RetroHotspotView(parent)

    @property
    def name(self):
        return "RetroHotspot grid predictor"
        
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
        def __call__(self, analysis_model, grid_task, project_task):
            timed_points = self.projected_data(analysis_model, project_task)
            if timed_points.number_data_points == 0:
                raise predictor.PredictionError(_text["no_data"])
            grid = grid_task(timed_points)
            return RetroHotspot.SubTask(timed_points, grid)

    class SubTask(predictor.SingleGridPredictor):
        def __init__(self, timed_points, grid):
            self._timed_points = timed_points
            # TODO: Do we allow this to be changed?  NO: probably make a new predictor class
            self._predictor = open_cp.retrohotspot.RetroHotSpotGrid(grid=grid)
            # TODO: Set `predictor.weight`
            # TODO: Vary this
            self._window_length = np.timedelta64(1, "m") * 60 * 24 * 60

        def __call__(self, predict_time, length=None):
            return self._predictor.predict(start_time=predict_time - self._window_length,
                    end_time=predict_time)


class RetroHotspotView(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        util.stretchy_rows_cols(self, [0], [0])
        self._text = richtext.RichText(self, height=12, scroll="v")
        self._text.grid(sticky=tk.NSEW)
        self._text.add_text(_text["main"])
