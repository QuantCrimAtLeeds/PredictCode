"""
prohotspot
~~~~~~~~~~

Uses the "prospective-hotspotting" technique.

The `prohotspot` method takes parameters:

  - The resolution to "bin" the times at (weeks, days, ?)
  - The weight to use (which takes the space distance, and time distance)
  - A way to calculate the space distance

Then want to visualise all this!
"""

from . import predictor
import open_cp.prohotspot
import tkinter as tk
import tkinter.ttk as ttk
import open_cp.gui.tk.util as util
import open_cp.gui.tk.richtext as richtext
import open_cp.gui.tk.tooltips as tooltips
import numpy as np
import datetime
import open_cp.gui.tk.mtp as mtp

_text = {
    "main" : ("Prospective-Hotspotting Grid Predictor.\n\n"
            + "Both space and time are made discrete (by placing a grid over space, and 'binning' the times to the nearest week, or "
            + "day, etc.) and then a kernel is placed around each event.  Compared to the more traditional 'retrospective' hot-spot "
            + "technique(s) this algorithm takes account of time, and gives more weight to more recent events.\n"
            + "As we follow the original literature, the 'kernel' or 'weight' used operates with 'areal' units, namely grid "
            + "cells and the time window selected.  This means that if you change the grid cell size, then the real shape of the kernel "
            + "will change, and the ratio between time and space will change.\n"
            + "Training Data usage: Ignored.  For each prediction point, all data in a 'window' before the prediction point is used."
        ),
    "time_bin" : "Time resolution:",
    "time_bin_tt" : ("How large the 'bins' to place timestamps into.  For example, if '1 Week' then all timestamps less than "
            + "7 days from the prediction point count as 'this week'; those between 7 and 14 days count at 'one week in the past' "
            + "and so forth."),
    "dist_tt" : "How distance from the current grid cell is calculated.",
    "time_bin_choices" : ["Week(s)", "Day(s)", "Hour(s)"],
    "kernel" : "Kernel choice",
    "kernels" : ["Classic"],
    "dist" : "Distance calculator",
    "dists" : ["Diagonals Same", "Diagonals Different", "Distance as a circle"],
    "dgx" : "Horizontal grid distance",
    "dgy" : "Vertical grid distance",
    "spbw" : "Spacial bandwidth:",
    "spbw1" : "grid cells",
    "spbw_tt" : "The bandwidth in space, here in terms of grid cells.  So changing the grid size will change the bandwidth!",
    "tbw" : "Time bandwidth:",
    "tbw_tt" : "The time bandwidth, in terms of the time window selected above.",
    "kgx" : "Distance in grid cells",
    "no_data" : "No data points found in time range: have enough crime types been selected?",

}

class ProHotspot(predictor.Predictor):
    def __init__(self, model):
        super().__init__(model)
        self._timelength = datetime.timedelta(days=7)
        self._diags = [DiagonalsSame(), DiagonalsDiff(), DiagonalsCircle()]
        self._diags_choice = 0
        self._weights = [Classical(self)]
        self._weight_choice = 0

    @staticmethod
    def describe():
        return "ProspectiveHotspot grid predictor"

    @staticmethod
    def order():
        return predictor._TYPE_GRID_PREDICTOR

    def config(self):
        return {"resize" : True}

    def make_view(self, parent):
        return ProHotspotView(parent, self)

    @property
    def name(self):
        return "ProspectiveHotspot grid predictor"
        
    @property
    def settings_string(self):
        num, unit = ProHotspotView.timeunit(self.time_window_length)
        return "{} / {} @ {} {}".format(str(self.distance_model),
            str(self.weight_model), num, _text["time_bin_choices"][unit])
        
    def make_tasks(self):
        weight = self.weight_model.weight()
        dist = self.distance_model.grid_distance()
        return [self.Task(weight, dist, self.time_window_length)]

    def to_dict(self):
        data =  {"distance" : self.distance_type,
            "weight" : self.weight_type,
            "time_window" : self.time_window_length.total_seconds()}
        data["distance_settings"] = [d.to_dict() for d in self._diags]
        data["weight_settings"] = [w.to_dict() for w in self._weights]
        return data

    def from_dict(self, data):
        self.distance_type = data["distance"]
        self.weight_type = data["weight"]
        self.time_window_length = datetime.timedelta(seconds=data["time_window"])
        for dist, di in zip(self._diags, data["distance_settings"]):
            dist.from_dict(di)
        for kernel, di in zip(self._weights, data["weight_settings"]):
            kernel.from_dict(di)

    @property
    def time_window_length(self):
        """The length of each time bin, a :class:`datetime.timedelta` instance."""
        return self._timelength

    @time_window_length.setter
    def time_window_length(self, value):
        self._timelength = value

    @property
    def distance_type(self):
        """The type of distance computation used."""
        return self._diags_choice

    @distance_type.setter
    def distance_type(self, value):
        self._diags_choice = value

    @property
    def distance_model(self):
        return self._diags[self._diags_choice]

    @property
    def weight_type(self):
        """The type of weight used."""
        return self._weight_choice

    @weight_type.setter
    def weight_type(self, value):
        self._weight_choice = value

    @property
    def weight_model(self):
        return self._weights[self._weight_choice]

    class Task(predictor.GridPredictorTask):
        def __init__(self, weight, distance, timeunit):
            super().__init__()
            self.weight = weight
            self.distance = distance
            self.timeunit = np.timedelta64(timeunit)
        
        def __call__(self, analysis_model, grid_task, project_task):
            timed_points = self.projected_data(analysis_model, project_task)
            if timed_points.number_data_points == 0:
                raise predictor.PredictionError(_text["no_data"])
            grid = grid_task(timed_points)
            return ProHotspot.SubTask(timed_points, grid, self.weight,
                self.distance, self.timeunit)

    class SubTask(predictor.SingleGridPredictor):
        def __init__(self, timed_points, grid, weight, distance, timeunit):
            super().__init__()
            self._predictor = open_cp.prohotspot.ProspectiveHotSpot(
                grid=grid, time_unit=timeunit)
            self._predictor.data = timed_points
            self._predictor.weight = weight
            self._predictor.distance = distance

        def __call__(self, predict_time, length=None):
            predict_time = np.datetime64(predict_time)
            return self._predictor.predict(predict_time, predict_time)


class DiagonalsSame():
    def __init__(self):
        pass

    def to_dict(self):
        return {}

    def from_dict(self, data):
        pass

    def __str__(self):
        return "DiagsSame"

    def grid_distance(self):
        return open_cp.prohotspot.DistanceDiagonalsSame()

    def view(self, parent):
        return DiagonalsSameView(parent, self)


class DiagonalsSameView(ttk.Frame):
    def __init__(self, parent, model):
        self._model = model
        super().__init__(parent)
        self._plot = mtp.CanvasFigure(self)
        self._plot.grid(padx=2, pady=2, sticky=tk.NSEW, row=0, column=0)
        self._plot.set_figure_task(self._plot_task)
        util.stretchy_rows_cols(self, [0], [0])

    def _plot_task(self):
        fig = mtp.new_figure((5,5))
        ax = fig.add_subplot(1,1,1)
        dist_obj = self._model.grid_distance()
        xcs = np.array([-3, -2, -1, 0, 1, 2, 3, 4])
        ycs = np.array([-3, -2, -1, 0, 1, 2, 3, 4])
        dists = np.empty((len(ycs)-1,len(xcs)-1))
        for ix, x in enumerate(xcs[:-1]):
            for iy, y in enumerate(ycs[:-1]):
                dists[iy][ix] = dist_obj(0,0,x,y)
        pcm = ax.pcolormesh(xcs, ycs, dists, cmap="rainbow")
        fig.colorbar(pcm)
        ax.set(xlabel=_text["dgx"], ylabel=_text["dgy"])
        fig.set_tight_layout(True)
        return fig


class DiagonalsDiff():
    def __init__(self):
        pass

    def to_dict(self):
        return {}

    def from_dict(self, data):
        pass

    def __str__(self):
        return "DiagsDiff"

    def grid_distance(self):
        return open_cp.prohotspot.DistanceDiagonalsDifferent()

    def view(self, parent):
        return DiagonalsSameView(parent, self)


class DiagonalsCircle():
    def __init__(self):
        pass

    def to_dict(self):
        return {}

    def from_dict(self, data):
        pass

    def __str__(self):
        return "DiagsCircle"

    def grid_distance(self):
        return open_cp.prohotspot.DistanceCircle()

    def view(self, parent):
        return DiagonalsSameView(parent, self)


class Classical():
    def __init__(self, parent_model):
        self._space_bandwidth = 8
        self._time_bandwidth = 8
        self._model = parent_model

    def to_dict(self):
        return {"space" : self.space_bandwidth,
            "time" : self.time_bandwidth}

    def from_dict(self, data):
        self.space_bandwidth = data["space"]
        self.time_bandwidth = data["time"]

    def __str__(self):
        return "Classic({}x{})".format(self.space_bandwidth, self.time_bandwidth)

    def weight(self):
        weight = open_cp.prohotspot.ClassicWeight()
        weight.space_bandwidth = self.space_bandwidth
        weight.time_bandwidth = self.time_bandwidth
        return weight

    def view(self, parent):
        return ClassicalView(parent, self, self._model)

    @property
    def space_bandwidth(self):
        """The bandwidth in terms of *grid* cells"""
        return self._space_bandwidth

    @space_bandwidth.setter
    def space_bandwidth(self, value):
        self._space_bandwidth = int(value)

    @property
    def time_bandwidth(self):
        """The bandwidth in terms of the selected time interval"""
        return self._time_bandwidth

    @time_bandwidth.setter
    def time_bandwidth(self, value):
        self._time_bandwidth = int(value)


class ClassicalView(ttk.Frame):
    def __init__(self, parent, model, parent_model):
        self._model = model
        self._parent_model = parent_model
        super().__init__(parent)

        subframe = ttk.Frame(self)
        subframe.grid(row=0, column=0, sticky=tk.W)
        label = ttk.Label(subframe, text=_text["spbw"])
        label.grid(row=0, column=0, padx=2, pady=2)
        tooltips.ToolTipYellow(label, _text["spbw_tt"])
        self._space_bandwidth = tk.StringVar()
        entry = ttk.Entry(subframe, textvariable=self._space_bandwidth, width=5)
        entry.grid(row=0, column=1, padx=2)
        util.IntValidator(entry, self._space_bandwidth, callback=self._space_bw_changed)
        ttk.Label(subframe, text=_text["spbw1"]).grid(row=0, column=2, padx=2)

        subframe = ttk.Frame(self)
        subframe.grid(row=1, column=0, sticky=tk.W)
        label = ttk.Label(subframe, text=_text["tbw"])
        label.grid(row=0, column=0, padx=2, pady=2)
        tooltips.ToolTipYellow(label, _text["tbw_tt"])
        self._time_bandwidth = tk.StringVar()
        entry = ttk.Entry(subframe, textvariable=self._time_bandwidth, width=5)
        entry.grid(row=0, column=1, padx=2)
        util.IntValidator(entry, self._time_bandwidth, callback=self._time_bw_changed)

        self._plot = mtp.CanvasFigure(self)
        self._plot.grid(padx=2, pady=2, sticky=tk.NSEW, row=2, column=0)
        util.stretchy_rows_cols(self, [2], [0])

        self._update()

    def _update(self):
        self._space_bandwidth.set(self._model.space_bandwidth)
        self._time_bandwidth.set(self._model.time_bandwidth)
        self._plot.set_figure_task(self._plot_task)

    def _plot_task(self):
        fig = mtp.new_figure((5,5))
        ax = fig.add_subplot(1,1,1)
        weight = self._model.weight()
        xcs = np.arange(0, self._model.space_bandwidth + 1)
        ycs = np.arange(0, self._model.time_bandwidth + 1)
        dists = np.empty((len(ycs)-1,len(xcs)-1))
        for ix, d_space in enumerate(xcs[:-1]):
            for iy, d_time in enumerate(ycs[:-1]):
                dists[iy][ix] = weight(d_time,d_space)
        num, unit = ProHotspotView.timeunit(self._parent_model.time_window_length)
        pcm = ax.pcolormesh(xcs, ycs * num, dists, cmap="rainbow")
        fig.colorbar(pcm)
        yl = _text["time_bin_choices"][unit]
        ax.set(xlabel=_text["kgx"], ylabel=yl)
        fig.set_tight_layout(True)
        return fig

    def _space_bw_changed(self, event=None):
        self._model.space_bandwidth = int(self._space_bandwidth.get())
        self._update()

    def _time_bw_changed(self, event=None):
        self._model.time_bandwidth = int(self._time_bandwidth.get())
        self._update()


class ProHotspotView(tk.Frame):
    def __init__(self, parent, model):
        self._model = model
        super().__init__(parent)
        util.stretchy_rows_cols(self, [1], [0])
        self._text = richtext.RichText(self, height=12, scroll="v")
        self._text.grid(sticky=tk.NSEW, row=0, column=0)
        self._text.add_text(_text["main"])
        frame = tk.Frame(self)
        frame.grid(sticky=tk.NSEW, row=1, column=0)
        self._add_controls(frame)
        self._update()

    def _update(self):
        num, choice = self.timeunit(self._model.time_window_length)
        self._time_resolution.set(num)
        self._time_res_cbox.current(choice)
        self._update_dist()
        self._update_weight()

    def _update_dist(self):
        self._dist_cbox.current(self._model.distance_type)
        for w in self._dist_frame.winfo_children():
            w.destroy()
        self._model.distance_model.view(self._dist_frame).grid(sticky=tk.NSEW)
        util.stretchy_rows_cols(self._dist_frame, [0], [0])

    def _update_weight(self):
        self._kernel_cbox.current(self._model.weight_type)
        for w in self._kernel_frame.winfo_children():
            w.destroy()
        self._model.weight_model.view(self._kernel_frame).grid(sticky=tk.NSEW)
        util.stretchy_rows_cols(self._kernel_frame, [0], [0])

    @staticmethod
    def timeunit(length):
        """Convert `length` an instance of :class:`datetime.timedelta` to a
        pair `(num, unit)` where:
          - unit == 0 : Weeks
          - 1 : Days
          - 2 : Hours
        """
        weeks = length / datetime.timedelta(days=7)
        days = length / datetime.timedelta(days=1)
        hours = length / datetime.timedelta(hours=1)
        if abs(weeks - int(weeks)) < 1e-7:
            return int(weeks), 0
        elif abs(days - int(days)) < 1e-7:
            return int(days), 1
        elif abs(hours - int(hours)) < 1e-7:
            return int(hours), 2
        else:
            raise ValueError()

    def _add_controls(self, frame):
        subframe = ttk.Frame(frame)
        subframe.grid(row=0, column=0, columnspan=2, sticky=tk.NW)
        label = ttk.Label(subframe, text=_text["time_bin"])
        label.grid(row=0, column=0, padx=2, pady=2)
        tooltips.ToolTipYellow(label, _text["time_bin_tt"])
        self._time_resolution = tk.StringVar()
        entry = ttk.Entry(subframe, textvariable=self._time_resolution)
        entry.grid(row=0, column=1, padx=2, pady=2)
        util.IntValidator(entry, self._time_resolution, callback=self._time_res_changed)
        self._time_res_cbox = ttk.Combobox(subframe, height=5, state="readonly", width=10)
        self._time_res_cbox["values"] = _text["time_bin_choices"]
        self._time_res_cbox.bind("<<ComboboxSelected>>", self._time_res_changed)
        self._time_res_cbox.current(0)
        self._time_res_cbox.grid(row=0, column=2, padx=2, pady=2)

        subframe = ttk.LabelFrame(frame, text=_text["kernel"])
        subframe.grid(row=1, column=0, sticky=tk.NSEW, padx=2, pady=2)
        self._kernel_cbox = ttk.Combobox(subframe, height=5, state="readonly", width=20)
        self._kernel_cbox["values"] = _text["kernels"]
        self._kernel_cbox.bind("<<ComboboxSelected>>", self._kernel_changed)
        self._kernel_cbox.current(0)
        self._kernel_cbox.grid(row=0, column=0, padx=2, pady=2, sticky=tk.W)
        self._kernel_frame = ttk.Frame(subframe)
        self._kernel_frame.grid(row=1, column=0, sticky=tk.NSEW)
        util.stretchy_rows_cols(subframe, [1], [0])

        subframe = ttk.LabelFrame(frame, text=_text["dist"])
        subframe.grid(row=1, column=1, sticky=tk.NSEW, padx=2, pady=2)
        self._dist_cbox = ttk.Combobox(subframe, height=5, state="readonly", width=30)
        self._dist_cbox["values"] = _text["dists"]
        self._dist_cbox.bind("<<ComboboxSelected>>", self._dist_changed)
        self._dist_cbox.current(0)
        self._dist_cbox.grid(row=0, column=0, padx=2, pady=2, sticky=tk.W)
        self._dist_frame = ttk.Frame(subframe)
        self._dist_frame.grid(row=1, column=0, sticky=tk.NSEW)
        tooltips.ToolTipYellow(self._dist_frame, _text["dist_tt"])
        util.stretchy_rows_cols(subframe, [1], [0])

        util.stretchy_rows_cols(frame, [1], [0,1])

    def _time_res_changed(self, event=None):
        num = int(self._time_resolution.get())
        choice = self._time_res_cbox.current()
        if choice == 0:
            self._model.time_window_length = datetime.timedelta(days=7*num)
        elif choice == 1:
            self._model.time_window_length = datetime.timedelta(days=num)
        elif choice == 2:
            self._model.time_window_length = datetime.timedelta(hours=num)
        else:
            raise ValueError()
        self._update_weight()

    def _kernel_changed(self, event=None):
        self._model.weight_type = self._kernel_cbox.current()
        self._update_weight()

    def _dist_changed(self, event=None):
        self._model.distance_type = self._dist_cbox.current()
        self._update_dist()


def test(root):
    ll = ProHotspot(predictor.test_model())
    predictor.test_harness(ll, root)
