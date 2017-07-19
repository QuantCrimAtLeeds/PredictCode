"""
sepp
~~~~

The fully non-parametric SEPP model from Mohler et al
"""

from . import predictor
import open_cp.predictors
import open_cp.sepp
import tkinter as tk
import tkinter.ttk as ttk
import open_cp.gui.tk.util as util
import open_cp.gui.tk.richtext as richtext
import open_cp.gui.tk.tooltips as tooltips
import datetime
#import numpy as np

_text = {
    "main" : ("KDE based Self-Excited Point Process predictor.\n\n"
        + "Uses the 'stocastic declustering' algorithm to attempt to split the data into "
        + "'background' events and 'triggered' events (motivated by the (near) repeat crime hypothesis). "
        + "A kernel density estimation is used to predict the density of background and triggering events, "
        + "and these are then used to inform predictions.\n\n"
        + "Training Data usage: The training data is used to estimate the kernels.  These kernels "
        + "are then held fixed for each prediction.\n\n"
        + "WARNING: We have found that this model often performs poorly on real data.  The use of KDE also "
        + "requires accurate timestamps: if the timestamps are only to the nearest day, you will likely get "
        + "errors."
        ),
    "no_data" : "No data points found in training time range: have enough crime types been selected?",
    "settings" : "Settings",
    "iters" : "Training iterations:",
    "iters_tt" : "The number of iterations of the main iterative algorithm to run.  A value of 40 is likely okay, but higher values could give between fitting.",
    "ktime" : "Time nearest neighbours:",
    "ktime_tt" : "The nearest neighbour to use for bandwidth estimation of the time kernel.  Values between 10 and 100 are reasonable, 100 the default.",
    "kspace" : "Space nearest neighbours:",
    "kspace_tt" : "The nearest neighbour to use for bandwidth estimation of the space kernel.  Values between 10 and 100 are reasonable, 15 the default.",
    "ibtime" : "Initial time bandwidth:",
    "ibtime2" : "days",
    "ibtime_tt" : "Used to form the initial 'guess' at a kernel.",
    "ibspace" : "Initial space bandwidth:",
    "ibspace_tt" : "Used to form the initial 'guess' at a kernel.",
    "ibspace2" : "meters",
    "cttime" : "Time cutoff:",
    "cttime2" : "days",
    "cttime_tt" : "For computing the trigger kernel, ignore events with a time gap greater than this.  This improves training speed, at the cost of a small amount of accuracy.",
    "ctspace" : "Space cutoff:",
    "ctspace2" : "meters",
    "ctspace_tt" : "For computing the trigger kernel, ignore events with a distance greater than this.  This improves training speed, at the cost of a small amount of accuracy.",

}

class SEPP(predictor.Predictor):
    def __init__(self, model):
        super().__init__(model)
        self.iterations = 40
        self.ktime = 100
        self.kspace = 15
        self.ibtime = 0.1
        self.ibspace = 50.0
        self.cttime = 120
        self.ctspace = 500
    
    @staticmethod
    def describe():
        return "KDE based self-exciting point process model"

    @staticmethod
    def order():
        return predictor._TYPE_GRID_PREDICTOR

    def make_view(self, parent):
        return SEPPView(parent, self)

    @property
    def name(self):
        return "KDE and SEPP (kt={}, ks={})".format(self.ktime, self.kspace)
        
    @property
    def settings_string(self):
        return "iters={} initial={:.2f}/{:.1f} cutoff={:.0f}/{:.0f}".format(
                self.iterations, self.ibtime, self.ibspace, self.cttime, self.ctspace)
        
    def to_dict(self):
        return {"iterations": self.iterations,
            "ktime" : self.ktime,
            "kspace" : self.kspace,
            "ibtime" : self.ibtime,
            "ibspace" : self.ibspace,
            "cttime" : self.cttime,
            "ctspace" : self.ctspace
            }
    
    def from_dict(self, data):
        self.iterations = data["iterations"]
        self.ktime = data["ktime"]
        self.kspace = data["kspace"]
        self.ibtime = data["ibtime"]
        self.ibspace = data["ibspace"]
        self.cttime = data["cttime"]
        self.ctspace = data["ctspace"]
    
    def make_tasks(self):
        return [self.Task(self)]
        
    class Task(predictor.GridPredictorTask):
        def __init__(self, parent):
            super().__init__(True)
            self._iters = parent.iterations
            self._ktime = parent.ktime
            self._kspace = parent.kspace
            self._ibtime = parent.ibtime * datetime.timedelta(days=1)
            self._ibspace = parent.ibspace
            self._cttime = parent.cttime * datetime.timedelta(days=1)
            self._ctspace = parent.ctspace

        def __call__(self, analysis_model, grid_task, project_task):
            timed_points = self.projected_data(analysis_model, project_task)
            training_start, training_end, _, _ = analysis_model.time_range
            mask = (timed_points.timestamps >= training_start) & (timed_points.timestamps < training_end)
            training_points = timed_points[mask]
            if training_points.number_data_points == 0:
                raise predictor.PredictionError(_text["no_data"])
            grid = grid_task(timed_points)
            
            trainer = open_cp.sepp.SEPPTrainer(k_time=self._ktime, k_space=self._kspace)
            trainer.data = training_points
            trainer.initial_time_bandwidth = self._ibtime
            trainer.initial_space_bandwidth = self._ibspace
            trainer.time_cutoff = self._cttime
            trainer.space_cutoff = self._ctspace
            result = trainer.train(iterations=self._iters)
            result.data = timed_points
            
            return SEPP.SubTask(result, grid)

    class SubTask(predictor.SingleGridPredictor):
        def __init__(self, pred, grid):
            super().__init__(True)
            self._pred = pred
            self._grid = grid

        def __call__(self, predict_time, length=None):
            cts_pred = self._pred.predict(predict_time)
            return open_cp.predictors.GridPredictionArray.from_continuous_prediction_grid(cts_pred, self._grid)

    @property
    def iterations(self):
        return self._iterations
    
    @iterations.setter
    def iterations(self, v):
        self._iterations = int(v)

    @property
    def ktime(self):
        return self._ktime
    
    @ktime.setter
    def ktime(self, v):
        self._ktime = int(v)

    @property
    def kspace(self):
        return self._kspace
    
    @kspace.setter
    def kspace(self, v):
        self._kspace = int(v)
        
    @property
    def ibtime(self):
        return self._ibtime / datetime.timedelta(days=1)
    
    @ibtime.setter
    def ibtime(self, v):
        try:
            value = float(v)
            self._ibtime = datetime.timedelta(days=1) * value
            return
        except:
            pass
        
        try:
            days = v / datetime.timedelta(days=1)
            if isinstance(days, float):
                self.ibtime = days
                return
        except:
            pass
        
        raise ValueError("Cannot set with {} / {}".format(repr(v), type(v)))

    @property
    def ibspace(self):
        return self._ibspace
    
    @ibspace.setter
    def ibspace(self, v):
        self._ibspace = float(v)
        
    @property
    def cttime(self):
        return self._cttime / datetime.timedelta(days=1)
    
    @cttime.setter
    def cttime(self, v):
        self._cttime = float(v) * datetime.timedelta(days=1)
        
    @property
    def ctspace(self):
        return self._ctspace
    
    @ctspace.setter
    def ctspace(self, v):
        self._ctspace = float(v)
    

class SEPPView(ttk.Frame):
    def __init__(self, parent, model):
        super().__init__(parent)
        self._model = model
        util.stretchy_rows_cols(self, [0], [0])
        self._text = richtext.RichText(self, height=14, scroll="v")
        self._text.grid(sticky=tk.NSEW)
        self._text.add_text(_text["main"])
        frame = ttk.LabelFrame(self, text=_text["settings"])
        frame.grid(row=1, column=0, sticky=tk.NSEW)
        self._add_widgets(frame)
        self._update()
    
    def _add_label_entry(self, frame, label_text, callback, second_label=None, validator=util.IntValidator):
        f = ttk.Frame(frame)
        label = ttk.Label(f, text=label_text)
        label.grid(row=0, column=0, padx=1, pady=1)
        var = tk.StringVar()
        entry = ttk.Entry(f, textvariable=var, width=6)
        validator(entry, var, callback=callback)
        entry.grid(row=0, column=1, padx=1, pady=1)
        if second_label is not None:
            ttk.Label(f, text=second_label).grid(row=0, column=2, padx=1, pady=1)
        return var, f, label
        
    def _add_widgets(self, frame):
        self._ktime_var, f, la = self._add_label_entry(frame, _text["ktime"], self._change)
        f.grid(row=0, column=0, sticky=tk.W, padx=1, pady=1)
        tooltips.ToolTipYellow(la, _text["ktime_tt"])
        self._kspace_var, f, la = self._add_label_entry(frame, _text["kspace"], self._change)
        f.grid(row=0, column=1, sticky=tk.W, padx=1, pady=1)
        tooltips.ToolTipYellow(la, _text["kspace_tt"])

        self._ibtime_var, f, la = self._add_label_entry(frame, _text["ibtime"], self._change, _text["ibtime2"], util.FloatValidator)
        f.grid(row=1, column=0, sticky=tk.W, padx=1, pady=1)
        tooltips.ToolTipYellow(la, _text["ibtime_tt"])
        self._ibspace_var, f, la = self._add_label_entry(frame, _text["ibspace"], self._change, _text["ibspace2"], util.FloatValidator)
        f.grid(row=1, column=1, sticky=tk.W, padx=1, pady=1)
        tooltips.ToolTipYellow(la, _text["ibspace_tt"])

        self._cttime_var, f, la = self._add_label_entry(frame, _text["cttime"], self._change, _text["cttime2"], util.FloatValidator)
        f.grid(row=2, column=0, sticky=tk.W, padx=1, pady=1)
        tooltips.ToolTipYellow(la, _text["cttime_tt"])
        self._ctspace_var, f, la = self._add_label_entry(frame, _text["ctspace"], self._change, _text["ctspace2"], util.FloatValidator)
        f.grid(row=2, column=1, sticky=tk.W, padx=1, pady=1)
        tooltips.ToolTipYellow(la, _text["ctspace_tt"])

        self._iters_var, f, la = self._add_label_entry(frame, _text["iters"], self._change)
        f.grid(row=100, column=0, sticky=tk.W, padx=1, pady=1)
        tooltips.ToolTipYellow(la, _text["iters_tt"])
        
    def _change(self, entry=None):
        self._model.iterations = self._iters_var.get()
        self._model.ktime = self._ktime_var.get()
        self._model.kspace = self._kspace_var.get()
        self._model.ibtime = self._ibtime_var.get()
        self._model.ibspace = self._ibspace_var.get()
        self._model.cttime = self._cttime_var.get()
        self._model.ctspace = self._ctspace_var.get()
    
    def _update(self):
        self._iters_var.set(self._model.iterations)
        self._ktime_var.set(self._model.ktime)
        self._kspace_var.set(self._model.kspace)
        self._ibtime_var.set(self._model.ibtime)
        self._ibspace_var.set(self._model.ibspace)
        self._cttime_var.set(self._model.cttime)
        self._ctspace_var.set(self._model.ctspace)


def test(root):
    ll = SEPP(predictor.test_model())
    predictor.test_harness(ll, root)
