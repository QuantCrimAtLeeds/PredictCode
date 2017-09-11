"""
hit_rate
~~~~~~~~

Compares the "hit rate" of a prediction against the reality.

Remember / TODO: Running the test on a range of "coverages" is (much) more
efficient than running a load of tests.
"""

from . import comparitor
import tkinter as tk
import tkinter.ttk as ttk
import open_cp.gui.tk.util as util
import open_cp.gui.tk.tooltips as tooltips
import open_cp.gui.tk.richtext as richtext
import open_cp.evaluation
import numpy as np

_text = {
    "main" : ("Hit Rate Calculator\n\n"
        + "For each prediction, selects a certain 'coverage' (which, after taking account of possibly "
        + "limiting the grid to a geographical region) finds the selected percentage of grid cells by "
        + "taking the highest 'risk' cells first.  We then compare the prediction against what actually "
        + "happened, by deeming that we 'predicted' an event if it falls in a cell which has been selected."
        ),
    "coverage" : "Coverage selection",
    "top1" : "Top 1%",
    "top1_tt" : "Show just the top 1% of grid cells by risk",
    "top5" : "Top 5%",
    "top5_tt" : "Show just the top 5% of grid cells by risk",
    "top10" : "Top 10%",
    "top10_tt" : "Show just the top 10% of grid cells by risk",
    "topcus" : "Top",
    "topcus_tt" : "Show just the top % of grid cells by risk",

}


class HitRate(comparitor.Comparitor):
    def __init__(self, model):
        super().__init__(model)
        self._coverage = 5
    
    @staticmethod
    def describe():
        return "Hit rate"

    @staticmethod
    def order():
        return comparitor.TYPE_COMPARE_TO_REAL

    def make_view(self, parent):
        return HitRateView(parent, self)

    @property
    def name(self):
        return "Hit rate"
        
    @property
    def settings_string(self):
        return "Coverage: {}%".format(self.coverage)

    def to_dict(self):
        return {"coverage" : self.coverage}

    def from_dict(self, data):
        self.coverage = data["coverage"]

    def make_tasks(self):
        return [self.Task(self.coverage)]
        
    class Task(comparitor.CompareRealTask):
        def __init__(self, coverage):
            self._coverage = coverage

        def __call__(self, grid_prediction, timed_points, predict_date, predict_length):
            start = np.datetime64(predict_date)
            end = start + np.timedelta64(predict_length)
            mask = ( (timed_points.timestamps >= start) & (timed_points.timestamps < end) )
            points = timed_points[mask]
            rates = open_cp.evaluation.hit_rates(grid_prediction, points, [self._coverage])
            # Return is a _dictionary!_
            return rates[self._coverage]

    @property
    def coverage(self):
        """The percentage coverage level selected."""
        return self._coverage

    @coverage.setter
    def coverage(self, value):
        self._coverage = int(value)


class HitRateView(tk.Frame):
    def __init__(self, parent, model):
        super().__init__(parent)
        self._model = model
        #util.stretchy_rows_cols(self, [3], [0])
        self._text = richtext.RichText(self, height=12, scroll="v")
        self._text.grid(sticky=tk.NSEW, row=0, column=0)
        self._text.add_text(_text["main"])

        subframe = ttk.LabelFrame(self, text=_text["coverage"])
        subframe.grid(row=1, column=0, sticky=tk.W)
        self._coverage_choice = tk.IntVar()
        self._add_rb(subframe, "top1", 1).grid(row=0, column=0, padx=2, sticky=tk.W)
        self._add_rb(subframe, "top5", 2).grid(row=1, column=0, padx=2, sticky=tk.W)
        self._add_rb(subframe, "top10", 3).grid(row=2, column=0, padx=2, sticky=tk.W)
        frame = ttk.Frame(subframe)
        self._add_rb(frame, "topcus", 4).grid(row=0, column=0, padx=2, sticky=tk.W)
        self._risk_level = tk.StringVar()
        self._risk_level.set(20)
        self._risk_level_entry = ttk.Entry(frame, textvariable=self._risk_level)
        self._risk_level_entry.grid(row=0, column=1, padx=2, sticky=tk.W)
        self._risk_level_entry["state"] = tk.DISABLED
        frame.grid(row=3, column=0, sticky=tk.W)
        util.PercentageValidator(self._risk_level_entry, self._risk_level,
            callback=self._coverage_choice_change)

        self._update()

    def _add_rb(self, frame, text_name, value):
        rb = ttk.Radiobutton(frame, text=_text[text_name], value=value,
            variable=self._coverage_choice, command=self._coverage_choice_change)
        tooltips.ToolTipYellow(rb, _text[text_name + "_tt"])
        return rb

    def _update(self):
        if self._model.coverage == 1:
            self._coverage_choice.set(1)
            self._risk_level_entry["state"] = tk.DISABLED
        elif self._model.coverage == 5:
            self._coverage_choice.set(2)
            self._risk_level_entry["state"] = tk.DISABLED
        elif self._model.coverage == 10:
            self._coverage_choice.set(3)
            self._risk_level_entry["state"] = tk.DISABLED
        else:
            self._coverage_choice.set(4)
            self._risk_level.set(self._model.coverage)
            self._risk_level_entry["state"] = tk.ACTIVE

    def _coverage_choice_change(self):
        choice = int(self._coverage_choice.get())
        if choice == 1:
            self._model.coverage = 1
        elif choice == 2:
            self._model.coverage = 5
        elif choice == 3:
            self._model.coverage = 10
        else:
            self._model.coverage = self._risk_level.get()
        self._update()
        

def test(root):
    from . import predictor
    ll = HitRate(predictor.test_model())
    predictor.test_harness(ll, root)
    