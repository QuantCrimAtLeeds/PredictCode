"""
browse_analysis_view
~~~~~~~~~~~~~~~~~~~~

"""

import tkinter as tk
import tkinter.ttk as ttk
import open_cp.gui.tk.util as util
import open_cp.gui.tk.tooltips as tooltips
import open_cp.gui.tk.mtp as mtp
import open_cp.gui.tk.hierarchical_view as hierarchical_view

_text = {
    "title" : "Previous Analysis results.  Run @ {}",
    "dtfmt" : "%d %b %Y %H:%M",
    "cp" : "Coordinate projection:",
    "grid" : "Grid system used:",
    "pred" : "Prediction type:",
    "date" : "Prediction date:",
    "len" : "Prediction length:",
    "graph_name" : "Estimated (relative) risk",
    "risktype" : "Plot type",
    "rrisk" : "Relative risk",
    "rrisk_tt" : "Plot of raw (relative) risk given by the predictor",
    "top1" : "Top 1%",
    "top1_tt" : "Show just the top 1% of grid cells by risk",
    "top5" : "Top 5%",
    "top5_tt" : "Show just the top 5% of grid cells by risk",
    "top10" : "Top 10%",
    "top10_tt" : "Show just the top 10% of grid cells by risk",
    "topcus" : "Top",
    "topcus_tt" : "Show just the top % of grid cells by risk",
    "adj" : "Adjusters",
    "none" : "None",
    "exit" : "Exit",

}

class BrowseAnalysisView(util.ModalWindow):
    def __init__(self, parent, controller):
        self.controller = controller
        title = _text["title"].format(self.controller.model.result.run_time.strftime(_text["dtfmt"]))
        super().__init__(parent, title, resize="wh")
        self.set_size_percentage(50, 60)

    def add_risk_choice_widgets(self):
        frame = ttk.LabelFrame(self, text=_text["risktype"])
        frame.grid(row=0, column=1)
        self._risk_choice = tk.IntVar()
        for row, (text, tt) in enumerate([ (_text["rrisk"], _text["rrisk_tt"]),
            (_text["top1"], _text["top1_tt"]),
            (_text["top5"], _text["top5_tt"]),
            (_text["top10"], _text["top10_tt"]) ]):
            self._make_rb(frame, row, text, tt).grid(row=row, column=0, padx=2, sticky=tk.W)

        subframe = ttk.Frame(frame)
        subframe.grid(row=4, column=0, sticky=tk.W)
        self._make_rb(subframe, 4, _text["topcus"], _text["topcus_tt"]).grid(row=0, column=0, padx=2, sticky=tk.W)
        self._risk_level = tk.StringVar()
        self._risk_level.set("5")
        self._risk_level_entry = ttk.Entry(subframe, textvariable=self._risk_level)
        self._risk_level_entry.grid(row=0, column=1, padx=2, sticky=tk.W)
        self._risk_level_entry["state"] = tk.DISABLED
        util.PercentageValidator(self._risk_level_entry, self._risk_level, callback=self._risk_choice_change)

    def _make_rb(self, frame, value, text, tt):
        rb = ttk.Radiobutton(frame, text=text, value=value,
            variable=self._risk_choice, command=self._risk_choice_change)
        tooltips.ToolTipYellow(rb, tt)
        return rb

    def add_choice_widgets(self):
        self._selection_frame = ttk.Frame(self)
        self._selection_frame.grid(row=0, column=0)
        for row, text in enumerate([_text["cp"], _text["grid"], _text["pred"],
                _text["date"], _text["len"]]):
            ttk.Label(self._selection_frame, text=text).grid(row=row, column=0, padx=2, sticky=tk.E)
        self._hview = hierarchical_view.HierarchicalView(self.model.prediction_hierarchy, None, self._selection_frame)
        for row, frame in zip(range(5), self._hview.frames):
            frame.grid(row=row, column=1, padx=2, pady=2, sticky=tk.W)

    def add_adjust_widgets(self):
        self._adjust_frame = ttk.LabelFrame(self, text=_text["adj"])
        self._adjust_frame.grid(row=1, column=0, sticky=tk.EW)
        util.stretchy_rows_cols(self._adjust_frame, [0], [0])
        choices = [key for (key,_) in self.controller.model.adjust_tasks]
        self._adjust_choice = ttk.Combobox(self._adjust_frame, height=5,
            state="readonly", values=choices)#, width=50)
        self._adjust_choice.bind("<<ComboboxSelected>>", self._adjust_changed)
        self._adjust_choice.current(0)
        self._adjust_choice.grid(padx=2, pady=2, sticky=tk.EW)

    def add_widgets(self):
        self.add_choice_widgets()
        self.add_risk_choice_widgets()
        self.add_adjust_widgets()
        ttk.Button(self, text=_text["exit"], command=self.destroy).grid(row=1,column=1,sticky=tk.NSEW, padx=10, pady=5)

    def _adjust_changed(self, event):
        self.controller.notify_adjust_choice(event.widget.current())

    def _risk_choice_change(self):
        if self._risk_choice.get() < 4:
            self._risk_level_entry["state"] = tk.DISABLED
        if self._risk_choice.get() == 0:
            self.controller.notify_plot_type_risk()
        elif self._risk_choice.get() == 1:
            self.controller.notify_plot_type_risk_level(1)
        elif self._risk_choice.get() == 2:
            self.controller.notify_plot_type_risk_level(5)
        elif self._risk_choice.get() == 3:
            self.controller.notify_plot_type_risk_level(10)
        elif self._risk_choice.get() == 4:
            self._risk_level_entry["state"] = tk.ACTIVE
            level = int(self._risk_level.get())
            self.controller.notify_plot_type_risk_level(level)
        else:
            raise ValueError()

    @property
    def model(self):
        return self.controller.model

    @property
    def hierarchical_view(self):
        return self._hview

    def update_prediction(self, level=-1, adjust_task=None):
        """Change the plotted prediction.

        :param level: The % of "coverage" to display, or -1 to mean plot all the
          relative risk.
        """
        def make_fig():
            fig = mtp.new_figure((20,20))
            ax = fig.add_subplot(1,1,1)
            prediction = self.model.current_prediction
            if level == -1:
                plot_risk(prediction, ax, adjust_task)
            else:
                plot_coverage(prediction, level, ax, adjust_task)
            ax.set_aspect(1)
            fig.set_tight_layout(True)
            return fig
        if not hasattr(self, "_plot") or self._plot is None:
            frame = ttk.LabelFrame(self, text=_text["graph_name"])
            frame.grid(row=10, column=0, columnspan=2, padx=2, pady=2, sticky=tk.NSEW)
            util.stretchy_rows_cols(frame, [0], [0])
            util.stretchy_rows(self, [10])
            util.stretchy_columns(self, [0,1])
            self._plot = mtp.CanvasFigure(frame)
            self._plot.grid(padx=2, pady=2, sticky=tk.NSEW)
        self._plot.set_figure_task(make_fig, dpi=50)



### Plotting grid predictions etc. #######################################

import matplotlib.colors
import open_cp.evaluation
import numpy as _np

yellow_to_red = matplotlib.colors.LinearSegmentedColormap("yellow_to_red",
    {'red':   [(0.0,  1.0, 1.0),
              (1.0,  1.0, 1.0)],
    'green': [(0.0,  1.0, 1.0),
              (1.0,  0.0, 0.0)],
    'blue':  [(0.0,  0.2, 0.2),
              (1.0,  0.2, 0.2)]}
    )

def plot_risk(prediction, axis, adjust_task=None):
    """Plot the risk defined by the grid prediction to the axis.

    :param prediction: Instance of :class:`GridPrediction`
    :param axis: `matplotlib` axis object.
    """
    if adjust_task is not None:
        prediction = adjust_task(prediction)
    axis.pcolormesh(*prediction.mesh_data(), prediction.intensity_matrix, cmap="Blues")

def plot_coverage(prediction, level, axis, adjust_task=None):
    """Plot the risk defined by the grid prediction to the axis, constraining
    to a set % level of coverage.

    :param prediction: Instance of :class:`GridPrediction`
    :param level: The level of coverage to plot.
    :param axis: `matplotlib` axis object.
    """
    if adjust_task is not None:
        prediction = adjust_task(prediction)
    current_mask = _np.ma.getmaskarray(prediction.intensity_matrix)
    in_coverage = open_cp.evaluation.top_slice(prediction.intensity_matrix, level / 100)
    current_mask = current_mask | (~in_coverage)
    matrix = _np.ma.masked_array(prediction.intensity_matrix, current_mask)
    axis.pcolormesh(*prediction.mesh_data(), matrix, cmap=yellow_to_red)
    