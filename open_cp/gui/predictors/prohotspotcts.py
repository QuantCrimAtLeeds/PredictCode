"""
prohotspotcts
~~~~~~~~~~~~~

Uses the "prospective-hotspotting" technique, but now as a "continuous"
prediction which is converted to a grid at the last stage.  This is less
consistent with (my interpretation) of the literature, but is more "accurate"
and is easier to use, as it removes the annoying coupling between grid size and
"bandwidth".
"""

from . import predictor
from . import prohotspot
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
    "main" : ("Prospective-Hotspotting Continuous Grid Predictor.\n\n"
            + "A Kernel Density estimator which takes account of time.  Around each point is space/time we place a 'kernel' or "
            + "'weight' which decays in space/time up to certain 'bandwidth's.  These kernels are then summed to produce an overall "
            + "risk profile which is converted to a grid and used for prediction.\n"
            + "Compared to the non-'Continuous' version of the Prospective-Hotspotting method, this version applies a grid at the "
            + "very end of the process.  This is less true to the original litertare, and computationally slowly, but is more accurate, "
            + "should reduce spurious effects due to the use of a grid, and makes setting bandwidths a lot easier (as they no longer "
            + "depend on the, separately chosen, grid size).\n"
            + "Training Data usage: Ignored.  For each prediction point, all data in a 'window' before the prediction point is used."
        ),
    "dgx" : "Horizontal distance (meters)",
    "dgy" : "Vertical distance (meters)",
    "time_bin" : "Time length:",
    "time_bin_tt" : ("How much one 'unit' of time represents.  This changes the speed at which "
            + "the weight falls off in time.  Remember that one day is 24 hours, and one week "
            + "is 168 hours."),
    "time_unit" : "Hours",
    "space_bin" : "Space length:",
    "space_bin_tt" : ("How much one 'unit' of space represents.  This changes the speed at which "
            + "the weight falls off in space, and is used together with the distance calculator method."),
    "space_unit" : "Meters",
    "kernel" : "Kernel choice",
    "kernels" : ["Classic"],
    "dist_tt" : "How distance from the current grid cell is calculated.",
    "dist" : "Distance calculator",
    "dists" : ["Diagonals Same", "Diagonals Different", "Distance as a circle"],
    "no_data" : "No data points found in time range: have enough crime types been selected?",
    "spbw" : "Spacial bandwidth:",
    "spbw_tt" : "The bandwidth in space, in terms of areal `units` which are scaled by the `space length` chosen.",
    "tbw" : "Time bandwidth:",
    "tbw_tt" : "The time bandwidth, in terms of areal `units` which are scaled by the `time lentgh` chosen.",
    "kgx" : "Distance (meters)",
    "kgy" : "Time (hours)",

}

class ProHotspotCts(predictor.Predictor):
    def __init__(self, model):
        super().__init__(model)
        self._timelength = datetime.timedelta(days=7)
        self._spacelength = 50
        self._diags = [DiagonalsSame(self), DiagonalsDiff(self), DiagonalsCircle(self)]
        self._diags_choice = 0
        self._weights = [Classical(self)]
        self._weight_choice = 0

    @staticmethod
    def describe():
        return "ProspectiveHotspot continuous grid predictor"

    @staticmethod
    def order():
        return predictor._TYPE_GRID_PREDICTOR

    def make_view(self, parent):
        return ProHotspotCtsView(parent, self)

    def config(self):
        return {"resize" : True}

    @property
    def name(self):
        return "ProspectiveHotspot cts grid predictor"
        
    @property
    def settings_string(self):
        hours = int(self.time_window_length / datetime.timedelta(hours=1))
        return "{} / {} @ {}m, {}h".format(str(self.distance_model),
            str(self.weight_model), self.space_length, hours)
        
    def make_tasks(self):
        weight = self.weight_model.weight()
        dist = self.distance_model.grid_distance()
        return [self.Task(weight, dist, self.time_window_length, self.space_length)]

    def to_dict(self):
        data =  {"distance" : self.distance_type,
            "weight" : self.weight_type,
            "time_window" : self.time_window_length.total_seconds(),
            "space_length" : self.space_length}
        data["distance_settings"] = [d.to_dict() for d in self._diags]
        data["weight_settings"] = [w.to_dict() for w in self._weights]
        return data

    def from_dict(self, data):
        self.distance_type = data["distance"]
        self.weight_type = data["weight"]
        self.time_window_length = datetime.timedelta(seconds=data["time_window"])
        self.space_length = data["space_length"]
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
    def space_length(self):
        """The spatial "bandwidth".  Equivalent to the grid size in the
        non-continuous version of this predictor."""
        return self._spacelength

    @space_length.setter
    def space_length(self, value):
        self._spacelength = value

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
        def __init__(self, weight, distance, timeunit, space_bandwidth):
            super().__init__()
            self.weight = weight
            self.distance = distance
            self.timeunit = np.timedelta64(timeunit)
            self.grid = space_bandwidth
        
        def __call__(self, analysis_model, grid_task, project_task):
            timed_points = self.projected_data(analysis_model, project_task)
            if timed_points.number_data_points == 0:
                raise predictor.PredictionError(_text["no_data"])
            grid = grid_task(timed_points)
            return ProHotspotCts.SubTask(timed_points, grid, self.weight,
                self.distance, self.timeunit, self.grid)

    class SubTask(predictor.SingleGridPredictor):
        def __init__(self, timed_points, grid, weight, distance, timeunit, grid_size):
            super().__init__()
            self._predictor = open_cp.prohotspot.ProspectiveHotSpotContinuous(
                grid_size=grid_size, time_unit=timeunit)
            self._predictor.data = timed_points
            self._predictor.weight = weight
            self._predictor.distance = distance
            self._grid = grid

        def __call__(self, predict_time, length=None):
            predict_time = np.datetime64(predict_time)
            cts_pred = self._predictor.predict(predict_time, predict_time)
            return open_cp.predictors.GridPredictionArray.from_continuous_prediction_region(
                cts_pred, self._grid.region(), self._grid.xsize, self._grid.ysize)


class DiagonalsSame(prohotspot.DiagonalsSame):
    def __init__(self, model):
        super().__init__()
        self.model = model

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
        bandwidth = self._model.model.space_length
        xcs = np.linspace(-5, 5, 100)
        ycs = np.linspace(-5, 5, 100)
        dists = np.empty((len(ycs)-1,len(xcs)-1))
        for ix, x in enumerate(xcs[:-1]):
            for iy, y in enumerate(ycs[:-1]):
                dists[iy][ix] = dist_obj(0,0,x,y)
        pcm = ax.pcolormesh(xcs * bandwidth, ycs * bandwidth, dists, cmap="rainbow")
        fig.colorbar(pcm)
        ax.set(xlabel=_text["dgx"], ylabel=_text["dgy"])
        fig.set_tight_layout(True)
        return fig


class DiagonalsDiff(prohotspot.DiagonalsDiff):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def view(self, parent):
        return DiagonalsSameView(parent, self)


class DiagonalsCircle(prohotspot.DiagonalsCircle):
    def __init__(self, model):
        super().__init__()
        self.model = model

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
        xcs = np.linspace(0, self._model.space_bandwidth, 100)
        ycs = np.linspace(0, self._model.time_bandwidth, 100)
        dists = np.empty((len(ycs)-1,len(xcs)-1))
        for ix, d_space in enumerate(xcs[:-1]):
            for iy, d_time in enumerate(ycs[:-1]):
                dists[iy][ix] = weight(d_time, d_space)
        xscale = self._parent_model.space_length
        yscale = self._parent_model.time_window_length / datetime.timedelta(hours=1)
        pcm = ax.pcolormesh(xcs * xscale, ycs * yscale, dists, cmap="rainbow")
        fig.colorbar(pcm)
        ax.set(xlabel=_text["kgx"], ylabel=_text["kgy"])
        fig.set_tight_layout(True)
        return fig

    def _space_bw_changed(self, event=None):
        self._model.space_bandwidth = int(self._space_bandwidth.get())
        self._update()

    def _time_bw_changed(self, event=None):
        self._model.time_bandwidth = int(self._time_bandwidth.get())
        self._update()


class ProHotspotCtsView(tk.Frame):
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
        num = int(self._model.time_window_length / datetime.timedelta(hours=1))
        self._time_resolution.set(num)
        self._space_length_var.set(self._model.space_length)
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

    def _add_controls(self, frame):
        subframe = ttk.Frame(frame)
        subframe.grid(row=0, column=0, columnspan=2, sticky=tk.NW)
        label = ttk.Label(subframe, text=_text["time_bin"])
        label.grid(row=0, column=0, padx=2, pady=2)
        tooltips.ToolTipYellow(label, _text["time_bin_tt"])
        self._time_resolution = tk.StringVar()
        entry = ttk.Entry(subframe, textvariable=self._time_resolution, width=8)
        entry.grid(row=0, column=1, padx=2, pady=2)
        util.IntValidator(entry, self._time_resolution, callback=self._time_res_changed)
        ttk.Label(subframe, text=_text["time_unit"]).grid(row=0, column=2, padx=2)

        ttk.Frame(subframe).grid(row=0, column=3, ipadx=20)

        label = ttk.Label(subframe, text=_text["space_bin"])
        label.grid(row=0, column=4, padx=2, pady=2)
        tooltips.ToolTipYellow(label, _text["space_bin_tt"])
        self._space_length_var = tk.StringVar()
        entry = ttk.Entry(subframe, textvariable=self._space_length_var, width=8)
        entry.grid(row=0, column=5, padx=2, pady=2)
        util.IntValidator(entry, self._space_length_var, callback=self._space_length_change)
        ttk.Label(subframe, text=_text["space_unit"]).grid(row=0, column=6, padx=2)

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

    def _space_length_change(self, event=None):
        num = int(self._space_length_var.get())
        self._model.space_length = num
        self._update_weight()

    def _time_res_changed(self, event=None):
        num = int(self._time_resolution.get())
        self._model.time_window_length = datetime.timedelta(hours=num)
        self._update_weight()

    def _kernel_changed(self, event=None):
        self._model.weight_type = self._kernel_cbox.current()
        self._update_weight()

    def _dist_changed(self, event=None):
        self._model.distance_type = self._dist_cbox.current()
        self._update_dist()


def test(root):
    ll = ProHotspotCts(predictor.test_model())
    predictor.test_harness(ll, root)
