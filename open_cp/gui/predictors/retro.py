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
import open_cp.gui.tk.tooltips as tooltips
import numpy as np
import open_cp.gui.tk.mtp as mtp

_text = {
    "main" : ("Retro-Hotspotting Grid Predictor.\n\n"
            + "All data in a 'window' before the prediction point is used; other data is ignored.  Only the location (and not the times) of"
            + "the events are used.  Around each a point a 'kernel' is laid down, and then these are summed to produce a risk estimate.  "
            + "As such, this can be considered to be a form of Kernel Density Estimation.\n"
            + "This version applies the grid at the very start, assigning points to their grid cells.  This is truer to the original literature,"
            + "but with modern computer systems it is unnecessary, and is likely to be too much of an approximation.\n"
            + "Training Data usage: Ignored.  For each prediction point, all data in a 'window' before the prediction point is used."
        ),
    "main1" : ("Retro-Hotspotting Continuous to Grid Predictor.\n\n"
            + "All data in a 'window' before the prediction point is used; other data is ignored.  Only the location (and not the times) of"
            + "the events are used.  Around each a point a 'kernel' is laid down, and then these are summed to produce a risk estimate.  "
            + "As such, this can be considered to be a form of Kernel Density Estimation.\n"
            + "This version works with continuous Kernels until the point of producing a prediction, when a grid is applied.  As such, it is more "
            + "'accurate', though less true to the literature.\n"
            + "Training Data usage: Ignored.  For each prediction point, all data in a 'window' before the prediction point is used."
        ),
    "win" : "Time window:",
    "wintt" : "For each prediction, events in a window of this size before the prediction point are considered.",
    "win2" : "Days",
    "kernelframe" : "Kernel selection",
    "kernel_choices" : ["Quartic", "Gaussian"],
    "q_bandwidth" : "Bandwidth:",
    "q_bandwidth_tt" : "The maximal spatial extent of the kernel in metres",
    "q_bandwidth1" : "Metres",
    "g_bandwidth" : "Bandwidth:",
    "g_bandwidth_tt" : "The maximal spatial extent of the kernel in metres",
    "g_bandwidth1" : "Metres",
    "g_std_devs" : "Standard Deviations:",
    "g_std_devs_tt" : "The number of standard deviations the kernel should extend for",
    "no_data" : "No data points found in time range: have enough crime types been selected?",

}

class RetroHotspot(predictor.Predictor):
    def __init__(self, model):
        super().__init__(model)
        self._window_length = 60
        self._kernel = 0
        self._kernels = [QuarticModel(), GaussianModel()]

    @staticmethod
    def describe():
        return "RetroHotspot grid predictor"

    @staticmethod
    def order():
        return predictor._TYPE_GRID_PREDICTOR

    def make_view(self, parent):
        return RetroHotspotView(parent, self)

    @property
    def name(self):
        return "RetroHotspot grid predictor"
        
    @property
    def settings_string(self):
        return "{} Days, {}".format(self.window_length, str(self._kernels[self.kernel]))
        
    def make_tasks(self):
        weight = self._kernels[self.kernel].make_weight()
        return [self.Task(self.window_length, weight)]
        
    def to_dict(self):
        data = {"window_length" : self.window_length,
                "kernel" : self.kernel }
        for key, model in zip(["weight_quartic"], self._kernels):
            data[key] = model.to_dict()
        return data
    
    def from_dict(self, data):
        self.window_length = data["window_length"]
        self.kernel = data["kernel"]
        for key, model in zip(["weight_quartic"], self._kernels):
            model.from_dict(data[key])

    @property
    def window_length(self):
        """Length of window to use, in days"""
        return self._window_length

    @window_length.setter
    def window_length(self, value):
        self._window_length = int(value)
        
    @property
    def kernel(self):
        """The kernel choosen:
            - 0 == Quartic
            - 1 == Gaussian
        """
        return self._kernel
    
    @kernel.setter
    def kernel(self, value):
        if value < 0 or value > 2:
            raise ValueError()
        self._kernel = value
    
    def kernel_model_view(self, parent):
        """Return the model and view for the currently selected kernel.
        
        :param parent: The parent `tk` widget
        
        :return: Pair `(model, view)`
        """
        model = self.kernel_model()
        if self.kernel == 0:
            view = QuarticView(parent, model)
        elif self.kernel == 1:
            view = GaussianView(parent, model)
        else:
            raise ValueError()
        return (model, view)
    
    def kernel_model(self):
        return self._kernels[self.kernel]
    
    class Task(predictor.GridPredictorTask):
        def __init__(self, window, weight):
            super().__init__()
            self.window = np.timedelta64(1, "D") * window
            self.weight = weight
        
        def __call__(self, analysis_model, grid_task, project_task):
            timed_points = self.projected_data(analysis_model, project_task)
            if timed_points.number_data_points == 0:
                raise predictor.PredictionError(_text["no_data"])
            grid = grid_task(timed_points)
            return RetroHotspot.SubTask(timed_points, grid, self.window, self.weight)

    class SubTask(predictor.SingleGridPredictor):
        def __init__(self, timed_points, grid, window, weight):
            super().__init__()
            self._predictor = open_cp.retrohotspot.RetroHotSpotGrid(grid=grid)
            self._predictor.weight = weight
            self._predictor.data = timed_points
            self._window_length = window

        def __call__(self, predict_time, length=None):
            predict_time = np.datetime64(predict_time)
            return self._predictor.predict(start_time=predict_time - self._window_length,
                    end_time=predict_time)


class RetroHotspotCtsGrid(RetroHotspot):
    """Uses the continuous version of the retro hotspot prediction, but then
    grids the result at the end."""
    def __init__(self, model):
        super().__init__(model)

    @staticmethod
    def describe():
        return "RetroHotspot continuous to grid predictor"

    def make_view(self, parent):
        return RetroHotspotView(parent, self, mode=1)

    @property
    def name(self):
        return "RetroHotspot cts/grid predictor"
        
    def make_tasks(self):
        weight = self._kernels[self.kernel].make_weight()
        return [self.Task(self.window_length, weight)]

    class Task(predictor.GridPredictorTask):
        def __init__(self, window, weight):
            super().__init__()
            self.window = np.timedelta64(1, "D") * window
            self.weight = weight
        
        def __call__(self, analysis_model, grid_task, project_task):
            timed_points = self.projected_data(analysis_model, project_task)
            if timed_points.number_data_points == 0:
                raise predictor.PredictionError(_text["no_data"])
            grid = grid_task(timed_points)
            return RetroHotspotCtsGrid.SubTask(timed_points, grid, self.window, self.weight)

    class SubTask(predictor.SingleGridPredictor):
        def __init__(self, timed_points, grid, window, weight):
            super().__init__()
            self._predictor = open_cp.retrohotspot.RetroHotSpot()
            self._predictor.weight = weight
            self._predictor.data = timed_points
            self._window_length = window
            self._grid = grid

        def __call__(self, predict_time, length=None):
            predict_time = np.datetime64(predict_time)
            cts_pred = self._predictor.predict(start_time=predict_time - self._window_length,
                    end_time=predict_time)
            return open_cp.predictors.GridPredictionArray.from_continuous_prediction_region(
                    cts_pred, self._grid.region(), self._grid.xsize, self._grid.ysize)


class GaussianModel():
    def __init__(self):
        self.bandwidth = 200
        self.std_devs = 3.0
        
    def from_dict(self, data):
        self.bandwidth = data["bandwidth"]
        self.std_devs = data["standard_deviations"]
        
    def to_dict(self):
        return {"bandwidth" : self.bandwidth,
                "standard_deviations" : self.std_devs}
        
    def make_weight(self):
        return open_cp.retrohotspot.TruncatedGaussian(self.bandwidth, self.std_devs)
    
    @property
    def bandwidth(self):
        """The maximum extend of the kernel, in meters"""
        return self._bandwidth
    
    @bandwidth.setter
    def bandwidth(self, value):
        self._bandwidth = value
        
    @property
    def std_devs(self):
        """The number of standard deviations which the kernel extends for."""
        return self._std_devs
    
    @std_devs.setter
    def std_devs(self, value):
        self._std_devs = value

    def __str__(self):
        return "Gaussian({}m, {}sds)".format(self.bandwidth, self.std_devs)


class GaussianView(ttk.Frame):
    def __init__(self, parent, model):
        self._model = model
        super().__init__(parent)
        self._add_widgets()
        self._update()

    def _update(self):
        self._bandwidth_var.set(self._model.bandwidth)
        self._std_var.set(self._model.std_devs)
        def make_fig():
            return _plot_weight(self._model.make_weight(), self._model.bandwidth)
        self._plot.set_figure_task(make_fig, dpi=100)

    def _change(self, event=None):
        self._model.bandwidth = int(self._bandwidth_var.get())
        self._model.std_devs = float(self._std_var.get())
        self._update()

    def _add_widgets(self):
        subframe = ttk.Frame(self)
        subframe.grid(row=0, column=0, sticky=tk.W)

        label = ttk.Label(subframe, text=_text["g_bandwidth"])
        label.grid(row=0, column=0, padx=2, pady=2)
        tooltips.ToolTipYellow(label, _text["g_bandwidth_tt"])
        self._bandwidth_var = tk.StringVar()
        entry = ttk.Entry(subframe, textvariable=self._bandwidth_var, width=7)
        entry.grid(row=0, column=1, padx=2)
        util.IntValidator(entry, self._bandwidth_var, callback=self._change)
        ttk.Label(subframe, text=_text["g_bandwidth1"]).grid(row=0, column=2)

        label = ttk.Label(subframe, text=_text["g_std_devs"])
        label.grid(row=1, column=0, padx=2, pady=2)
        tooltips.ToolTipYellow(label, _text["g_std_devs_tt"])
        self._std_var = tk.StringVar()
        entry = ttk.Entry(subframe, textvariable=self._std_var, width=7)
        entry.grid(row=1, column=1, padx=2)
        util.FloatValidator(entry, self._std_var, callback=self._change)

        self._plot = mtp.CanvasFigure(self)
        self._plot.grid(padx=2, pady=2, sticky=tk.NSEW, row=1, column=0)
        util.stretchy_rows_cols(self, [1], [0])


class QuarticModel():
    def __init__(self):
        self.bandwidth = 200
        
    def from_dict(self, data):
        self.bandwidth = data["bandwidth"]
        
    def to_dict(self):
        return {"bandwidth" : self.bandwidth}
        
    def make_weight(self):
        return open_cp.retrohotspot.Quartic(self.bandwidth)
    
    @property
    def bandwidth(self):
        """The maximum extend of the kernel, in meters"""
        return self._bandwidth
    
    @bandwidth.setter
    def bandwidth(self, value):
        self._bandwidth = value
        
    def __str__(self):
        return "Quartic({}m)".format(self.bandwidth)


class QuarticView(ttk.Frame):
    def __init__(self, parent, model):
        self._model = model
        super().__init__(parent)
        self._add_widgets()
        self._update()

    def _update(self):
        self._quartic_bandwidth_var.set(self._model.bandwidth)
        def make_fig():
            fig = _plot_weight(self._model.make_weight(), self._model.bandwidth)
            return fig
        self._quartic_plot.set_figure_task(make_fig, dpi=100)

    def _quartic_bandwidth_changed(self, event=None):
        self._model.bandwidth = int(self._quartic_bandwidth_var.get())
        self._update()

    def _add_widgets(self):
        subframe = ttk.Frame(self)
        subframe.grid(row=0, column=0, sticky=tk.W)
        label = ttk.Label(subframe, text=_text["q_bandwidth"])
        label.grid(row=0, column=0, padx=2, pady=2)
        tooltips.ToolTipYellow(label, _text["q_bandwidth_tt"])
        self._quartic_bandwidth_var = tk.StringVar()
        entry = ttk.Entry(subframe, textvariable=self._quartic_bandwidth_var, width=7)
        entry.grid(row=0, column=1, padx=2)
        util.IntValidator(entry, self._quartic_bandwidth_var, callback=self._quartic_bandwidth_changed)
        ttk.Label(subframe, text=_text["q_bandwidth1"]).grid(row=0, column=2)
        self._quartic_plot = mtp.CanvasFigure(self)
        self._quartic_plot.grid(padx=2, pady=2, sticky=tk.NSEW, row=1, column=0)
        util.stretchy_rows_cols(self, [1], [0])


def _plot_weight(weight, bandwidth):
    fig = mtp.new_figure((5,5))
    ax = fig.add_subplot(1,1,1)
    xc = np.linspace(-bandwidth, bandwidth, 50)
    yc = weight(xc, [0]*len(xc))
    ax.plot(xc, yc, color="black")
    ax.set(xlabel="Distance in metres")
    ax.set(ylabel="Relative risk")
    fig.set_tight_layout(True)
    return fig
    

class RetroHotspotView(tk.Frame):
    def __init__(self, parent, model, mode = 0):
        self._model = model
        super().__init__(parent)
        util.stretchy_rows_cols(self, [1], [0])
        self._text = richtext.RichText(self, height=12, scroll="v")
        self._text.grid(sticky=tk.NSEW, row=0, column=0)
        if mode == 0:
            self._text.add_text(_text["main"])
        elif mode == 1:
            self._text.add_text(_text["main1"])
        else:
            raise ValueError()
        frame = tk.Frame(self)
        frame.grid(sticky=tk.NSEW, row=1, column=0)
        self._add_controls(frame)
        self._update()

    def _update(self):
        self._window_var.set(self._model.window_length)
        self._cbox.current(self._model.kernel)
        self._kernel_options(self._model.kernel)
        
    def _kernel_options(self, kernel):
        for w in self._kernel_options_frame.winfo_children():
            w.destroy()
        _, view = self._model.kernel_model_view(self._kernel_options_frame)
        view.grid(sticky=tk.NSEW, row=0, column=0)
        util.stretchy_rows_cols(self._kernel_options_frame, [0], [0])

    def _add_controls(self, frame):
        subframe = tk.Frame(frame)
        subframe.grid(row=0, column=0, sticky=tk.W)
        label = ttk.Label(subframe, text=_text["win"])
        label.grid(row=0, column=0, padx=5, pady=2)
        tooltips.ToolTipYellow(label, _text["wintt"])
        self._window_var = tk.StringVar()
        entry = ttk.Entry(subframe, textvariable=self._window_var, width=5)
        entry.grid(row=0, column=1)
        util.IntValidator(entry, self._window_var, callback=self._window_changed)
        ttk.Label(subframe, text=_text["win2"]).grid(row=0, column=2, padx=2)
        
        subframe = ttk.LabelFrame(frame, text=_text["kernelframe"])
        subframe.grid(row=1, column=0, sticky=tk.NSEW)
        self._cbox = ttk.Combobox(subframe, height=5, state="readonly")
        self._cbox["values"] = _text["kernel_choices"]
        self._cbox.bind("<<ComboboxSelected>>", self._kernel_selected)
        self._cbox.current(0)
        self._cbox["width"] = 10
        self._cbox.grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
        self._kernel_options_frame = ttk.Frame(subframe)
        self._kernel_options_frame.grid(row=1, column=0, sticky=tk.NSEW)
        util.stretchy_rows_cols(subframe, [1], [0])

        util.stretchy_rows_cols(frame, [1], [0])

    def _window_changed(self, event=None):
        self._model.window_length = int(self._window_var.get())
        
    def _kernel_selected(self, event=None):
        self._model.kernel = event.widget.current()
        self._update()
        
        
def test(root):
    ll = RetroHotspot(predictor.test_model())
    predictor.test_harness(ll, root)
