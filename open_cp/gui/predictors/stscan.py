"""
stscan
~~~~~~

Uses the "space/time scan statistic" method of prediction.

Things we can vary:
  - How far back in time to look
  - The bandwidth/cutoff rules for clusters
  - Quantise space and/or time?

Pending further work, we shall not worry about:
  - Expanding clusters to their maximum size
  - Pass via a "continuous" prediction first
"""

from . import predictor
import open_cp.stscan
import tkinter as tk
import tkinter.ttk as ttk
import open_cp.gui.tk.util as util
import open_cp.gui.tk.richtext as richtext
import open_cp.gui.tk.tooltips as tooltips
import numpy as np
import datetime
import enum
import open_cp.gui.tk.mtp as mtp

_text = {
    "main" : ("Space/Time scan Predictor.\n\n"
        + "A statistical test, as implemented in the SaTScan software package, is used to identify possible "
        + "'clusters' of events in space and time.  We use the algorithm in 'prediction' mode whereby only "
        + "clusters extending to the current time are considered.  We then form a grid based prediction by "
        + "marking the first cluster as being at 'risk', then marking the second cluster as being of slightly "
        + "lower risk, and so on.\n"
        + "Each cluster is a 'space/time cylinder'.  It encompases a spatial region which is a disk, and time region which is an interval.\n"
        + "Training Data usage: Ignored.  For each prediction point, all data in a 'window' before the prediction point is used."
        ),
    "no_data" : "No data points found in time range: have enough crime types been selected?",
    "tw" : "Time Window",
    "tw1" : "Use data from start of training range",
    "tw1tt" : "For each prediction, all the data from the start of the 'training' time range up to the prediction date is used",
    "tw2" : "Window from prediction date of",
    "tw2tt" : "For each prediction, only data in this length of time 'window' before the prediction date is used",
    "tw3" : "Days",
    "qu" : "Quantise data",
    "quu" : "Time bin length",
    "quutt" : "The length of each 'time bin' to put timestamps into.  For example, if '24 hours' then each timestamp is assigned to the day it falls in.",
    "qu1" : "Use data as is",
    "qu1tt" : "Do not do anything to the input data",
    "qu2" : "Grid data",
    "qu2tt" : "Move the location of each event to the centre of the grid cell it falls in",
    "qu3" : "Bin time",
    "qu3tt" : "Convert timestamp to be the same for each 'bin'",
    "qu4" : "Both",
    "qu4tt" : "Grid locations and bin time",
    "qu5" : "Length",
    "qu6" : "Hours",
    "opt1" : "Maximum cluster size",
    "opt2" : "Space population limit:",
    "opt2tt" : "No cluster can contain more than this percentage of the total data (here just considering the spatial disk, ignoring all timestamps)",
    "optper" : "%",
    "opt3" : "Space radius limit:",
    "opt3tt" : "No cluster can have a radius greater than this",
    "optm" : "Meters",
    "opt4" : "Time population limit:",
    "opt4tt" : "No cluster can contain more than this percentage of the total data (here just considering the time intrval, ignoraing all event coordinates)",
    "opt5" : "Time length limit:",
    "opt5tt" : "No cluster can have a time length greater than this",
    "optdays" : "Days"

}

class STScan(predictor.Predictor):
    def __init__(self, model):
        super().__init__(model)
        self.time_window_choice = 1
        self.time_window_length = datetime.timedelta(days=60)
        self.quantisation_choice = 1
        self.time_bin_length = datetime.timedelta(days=1)
        self.geographic_population_limit = 50
        self.time_population_limit = 50
        self.geographic_radius_limit = 3000
        self.time_max_interval = datetime.timedelta(days=60)

    @staticmethod
    def describe():
        return "Space Time Scan Predictor"

    @staticmethod
    def order():
        return predictor._TYPE_GRID_PREDICTOR

    def config(self):
        return {}

    def make_view(self, parent):
        return STScanView(parent, self)

    @property
    def name(self):
        return "Space Time Scan Predictor"
        
    @property
    def settings_string(self):
        out = ""
        if self.time_window_choice == self.TimeWindowChoice.window:
            days = int(self.time_window_length / datetime.timedelta(days=1))
            out += "<={} ".format(days)
        if (self.quantisation_choice == self.QuantiseDataChoice.space or
            self.quantisation_choice == self.QuantiseDataChoice.both):
            out += "grid "
        elif (self.quantisation_choice == self.QuantiseDataChoice.time or
            self.quantisation_choice == self.QuantiseDataChoice.both):
            hours = int(self.time_bin_length / datetime.timedelta(hours=1))
            out += "bins({}hours)".format(hours)
        out += "geo({}%/{}m) ".format(int(self.geographic_population_limit),
            self.geographic_radius_limit)
        days = int(self.time_max_interval / datetime.timedelta(days=1))
        out += "time({}%/{}days)".format(int(self.time_population_limit),days)
        return out

    def make_tasks(self):
        return [self.Task(self)]

    def to_dict(self):
        return {
            "time_window_choice" : self.time_window_choice,
            "time_window_length" : self.time_window_length.total_seconds(),
            "quantisation_choice" : self.quantisation_choice,
            "time_bin_length" : self.time_bin_length.total_seconds(),
            "geographic_population_limit" : self.geographic_population_limit,
            "time_population_limit" : self.time_population_limit,
            "geographic_radius_limit" : self.geographic_radius_limit,
            "time_max_interval" : self.time_max_interval.total_seconds()
            }

    def from_dict(self, data):
        self.time_window_choice = int(data["time_window_choice"])
        self.time_window_length = datetime.timedelta(seconds = int(data["time_window_length"]))
        self.quantisation_choice = int(data["quantisation_choice"])
        self.time_bin_length = datetime.timedelta(seconds = int(data["time_bin_length"]))
        self.geographic_population_limit = int(data["geographic_population_limit"])
        self.time_population_limit = int(data["time_population_limit"])
        self.geographic_radius_limit = int(data["geographic_radius_limit"])
        self.time_max_interval = datetime.timedelta(seconds = int(data["time_max_interval"]))

    class TimeWindowChoice(enum.Enum):
        from_training = 1
        window = 2

    @property
    def time_window_choice(self):
        return self._time_window_choice.value

    @time_window_choice.setter
    def time_window_choice(self, value):
        self._time_window_choice = self.TimeWindowChoice(value)

    @property
    def time_window_length(self):
        return self._time_window_length

    @time_window_length.setter
    def time_window_length(self, value):
        self._time_window_length = value

    class QuantiseDataChoice(enum.Enum):
        none = 1
        space = 2
        time = 3
        both = 4

    @property
    def quantisation_choice(self):
        return self._quan_choice.value

    @quantisation_choice.setter
    def quantisation_choice(self, value):
        self._quan_choice = self.QuantiseDataChoice(value)

    @property
    def time_bin_length(self):
        return self._time_bin

    @time_bin_length.setter
    def time_bin_length(self, value):
        self._time_bin = value

    @property
    def geographic_population_limit(self):
        return self._geo_pop_limit * 100

    @geographic_population_limit.setter
    def geographic_population_limit(self, value):
        if value < 0 or value > 100:
            raise ValueError()
        self._geo_pop_limit = value / 100

    @property
    def geographic_radius_limit(self):
        return self._geo_radius

    @geographic_radius_limit.setter
    def geographic_radius_limit(self, value):
        self._geo_radius = value

    @property
    def time_population_limit(self):
        return self._time_pop_limit * 100

    @time_population_limit.setter
    def time_population_limit(self, value):
        if value < 0 or value > 100:
            raise ValueError()
        self._time_pop_limit = value / 100
    
    @property
    def time_max_interval(self):
        return self._time_max_interval

    @time_max_interval.setter
    def time_max_interval(self, value):
        self._time_max_interval = value

    class Task(predictor.GridPredictorTask):
        def __init__(self, model):
            self._geo_pop_limit_perc = model.geographic_population_limit
            self._time_pop_limit_perc = model.time_population_limit
            self._geo_radius = model.geographic_radius_limit
            self._time_max = model.time_max_interval

        def __call__(self, analysis_model, grid_task, project_task):
            timed_points = self.projected_data(analysis_model, project_task)
            if timed_points.number_data_points == 0:
                raise predictor.PredictionError(_text["no_data"])
            grid = grid_task(timed_points)
            return STScan.SubTask(timed_points, grid, self)

    class SubTask(predictor.SingleGridPredictor):
        def __init__(self, timed_points, grid, task):
            self.grid_size = grid.xsize
            self.predictor = open_cp.stscan.STSTrainer()
            self.predictor.data = timed_points
            self.predictor.region = grid.region()
            self.predictor.geographic_population_limit = task._geo_pop_limit_perc / 100
            self.predictor.geographic_radius_limit = task._geo_radius
            self.time_population_limit = task._time_pop_limit_perc / 100
            self.time_max_interval = np.timedelta64(task._time_max)

        def __call__(self, predict_time, length=None):
            predict_time = np.datetime64(predict_time)
            result = self.predictor.predict(time = predict_time)
            return result.grid_prediction(self.grid_size)


class STScanView(tk.Frame):
    def __init__(self, parent, model):
        self._model = model
        super().__init__(parent)
        self._text = richtext.RichText(self, height=12, scroll="v")
        self._text.grid(sticky=tk.NSEW, row=0, column=0)
        self._text.add_text(_text["main"])
        frame = tk.Frame(self)
        frame.grid(sticky=tk.NSEW, row=1, column=0)
        self.add_widgets(frame)
        self.update()

    def add_widgets(self, frame):
        subframe = ttk.LabelFrame(frame, text=_text["tw"])
        subframe.grid(row=0, column=0, padx=2, pady=2, sticky=tk.NSEW)
        self._time_window_option = tk.IntVar()
        rb = ttk.Radiobutton(subframe, text=_text["tw1"], variable=self._time_window_option,
            value=STScan.TimeWindowChoice.from_training.value, command=self._time_window_option_changed)
        rb.grid(row=0, column=0, padx=2, pady=2, sticky=tk.W)
        tooltips.ToolTipYellow(rb, _text["tw1tt"])
        subsubframe = ttk.Frame(subframe)
        subsubframe.grid(row=1, column=0, sticky=tk.W)
        rb = ttk.Radiobutton(subsubframe, text=_text["tw2"], variable=self._time_window_option,
            value=STScan.TimeWindowChoice.window.value, command=self._time_window_option_changed)
        rb.grid(row=0, column=0, padx=2, pady=2)
        tooltips.ToolTipYellow(rb, _text["tw2tt"])
        self._time_window_length_var = tk.StringVar()
        self._time_window_length = ttk.Entry(subsubframe, textvariable=self._time_window_length_var, width=7)
        self._time_window_length.grid(row=0, column=1, padx=2)
        util.IntValidator(self._time_window_length, self._time_window_length_var, callback=self._time_window_changed)
        ttk.Label(subsubframe, text=_text["tw3"]).grid(row=0, column=2, padx=2)

        subframe = ttk.LabelFrame(frame, text=_text["qu"])
        subframe.grid(row=0, column=1, padx=2, pady=2, sticky=tk.NSEW)
        self._quantise_option = tk.IntVar()
        for n in range(1,5):
            label = _text["qu{}".format(n)]
            rb = ttk.Radiobutton(subframe, text=label, variable=self._quantise_option,
                value=n, command=self._quantise_option_changed)
            rb.grid(row=n, column=0, padx=2, pady=2, sticky=tk.W)
            tooltips.ToolTipYellow(rb, _text["qu{}tt".format(n)])
        subframe = ttk.LabelFrame(subframe, text=_text["quu"])
        tooltips.ToolTipYellow(subframe, _text["quutt"])
        subframe.grid(row=0, column=1, rowspan=4, padx=2, sticky=tk.NSEW)
        ttk.Label(subframe, text=_text["qu5"]).grid(row=0, column=0, padx=2, pady=2, sticky=tk.W)
        subsubframe = ttk.Frame(subframe)
        subsubframe.grid(row=1, column=0)
        self._time_bin_length_var = tk.StringVar()
        self._time_bin_length = ttk.Entry(subsubframe, textvariable=self._time_bin_length_var, width=7)
        util.IntValidator(self._time_bin_length, self._time_bin_length_var, callback=self._time_bin_changed)
        self._time_bin_length.grid(row=0, column=0, padx=2, pady=2)
        ttk.Label(subsubframe, text=_text["qu6"]).grid(row=0, column=1, padx=2, pady=2)

        subframe = ttk.LabelFrame(frame, text=_text["opt1"])
        subframe.grid(row=1, column=0, padx=2, pady=2, sticky=tk.NSEW)
        subsubframe = ttk.Frame(subframe)
        subsubframe.grid(row=0, column=0, sticky=tk.W)
        self._geo_pop_limit_var = tk.StringVar()
        entry = self._add_entry_row(subsubframe, text=_text["opt2"], tttext=_text["opt2tt"],
            entry_var=self._geo_pop_limit_var, last_text=_text["optper"])
        util.PercentageValidator(entry, self._geo_pop_limit_var, callback=self._option_changed)
        subsubframe = ttk.Frame(subframe)
        subsubframe.grid(row=1, column=0, sticky=tk.W)
        self._geo_radius_limit_var = tk.StringVar()
        entry = self._add_entry_row(subsubframe, text=_text["opt3"], tttext=_text["opt3tt"],
            entry_var=self._geo_radius_limit_var, last_text=_text["optm"])
        util.IntValidator(entry, self._geo_radius_limit_var, callback=self._option_changed)
        subsubframe = ttk.Frame(subframe)
        subsubframe.grid(row=2, column=0, sticky=tk.W)
        self._time_pop_limit_var = tk.StringVar()
        entry = self._add_entry_row(subsubframe, text=_text["opt4"], tttext=_text["opt4tt"],
            entry_var=self._time_pop_limit_var, last_text=_text["optper"])
        util.PercentageValidator(entry, self._time_pop_limit_var, callback=self._option_changed)
        subsubframe = ttk.Frame(subframe)
        subsubframe.grid(row=3, column=0, sticky=tk.W)
        self._time_radius_limit_var = tk.StringVar()
        entry = self._add_entry_row(subsubframe, text=_text["opt5"], tttext=_text["opt5tt"],
            entry_var=self._time_radius_limit_var, last_text=_text["optdays"])
        util.IntValidator(entry, self._time_radius_limit_var, callback=self._option_changed)

    def _add_entry_row(self, subsubframe, text, tttext, entry_var, last_text):
        label = ttk.Label(subsubframe, text=text)
        label.grid(row=0, column=0, padx=2, pady=2)
        tooltips.ToolTipYellow(label, tttext)
        entry = ttk.Entry(subsubframe, textvariable=entry_var, width=7)
        entry.grid(row=0, column=1, padx=2, pady=2)
        ttk.Label(subsubframe, text=last_text).grid(row=0, column=2)
        return entry

    def _option_changed(self, event=None):
        self._model.geographic_population_limit = int(self._geo_pop_limit_var.get())
        self._model.geographic_radius_limit = int(self._geo_radius_limit_var.get())
        self._model.time_population_limit = int(self._time_pop_limit_var.get())
        self._model.time_max_interval = datetime.timedelta(days=int(self._time_radius_limit_var.get()))

    def _time_bin_changed(self, event=None):
        hours = int(self._time_bin_length_var.get())
        self._model.time_bin_length = datetime.timedelta(hours=hours)

    def _quantise_option_changed(self, event=None):
        self._model.quantisation_choice = int(self._quantise_option.get())
        self.update()

    def _time_window_option_changed(self, event=None):
        self._model.time_window_choice = int(self._time_window_option.get())
        self.update()

    def _time_window_changed(self, event=None):
        self._model.time_window_length = datetime.timedelta(days=int(self._time_window_length_var.get()))

    def update(self):
        self._time_window_option.set(self._model.time_window_choice)
        if self._model.time_window_choice == STScan.TimeWindowChoice.from_training.value:
            self._time_window_length.state(["disabled"])
        else:
            self._time_window_length.state(["!disabled"])
        window = int(self._model.time_window_length / datetime.timedelta(days=1))
        self._time_window_length_var.set(window)

        hours = int(self._model.time_bin_length / datetime.timedelta(hours=1))
        self._time_bin_length_var.set(hours)
        self._quantise_option.set(self._model.quantisation_choice)
        if self._model.quantisation_choice in {STScan.QuantiseDataChoice.none.value,
            STScan.QuantiseDataChoice.space.value}:
            self._time_bin_length.state(["disabled"])
        else:
            self._time_bin_length.state(["!disabled"])

        self._geo_pop_limit_var.set(int(self._model.geographic_population_limit))
        self._geo_radius_limit_var.set(self._model.geographic_radius_limit)
        self._time_pop_limit_var.set(int(self._model.time_population_limit))
        max_length = int(self._model.time_max_interval / datetime.timedelta(days=1))
        self._time_radius_limit_var.set(max_length)


def test(root):
    ll = STScan(predictor.test_model())
    predictor.test_harness(ll, root)
