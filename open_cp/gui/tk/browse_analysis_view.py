"""
browse_analysis_view
~~~~~~~~~~~~~~~~~~~~

"""

import tkinter as tk
import tkinter.ttk as ttk
import open_cp.gui.tk.util as util
import open_cp.gui.tk.tooltips as tooltips
import open_cp.gui.tk.mtp as mtp

_text = {
    "title" : "Previous Analysis results.  Run @ {}",
    "dtfmt" : "%d %b %Y %H:%M",
    "cp" : "Coordinate projection:",
    "grid" : "Grid system used:",
    "pred" : "Prediction type:",
    "date" : "Prediction date:",
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

}

class BrowseAnalysisView(util.ModalWindow):
    def __init__(self, parent, controller):
        self.controller = controller
        title = _text["title"].format(self.controller.model.result.run_time.strftime(_text["dtfmt"]))
        super().__init__(parent, title, resize="wh")

    def add_widgets(self):
        ttk.Label(self, text=_text["cp"]).grid(row=0, column=0, padx=2, sticky=tk.E)
        ttk.Label(self, text=_text["grid"]).grid(row=1, column=0, padx=2, sticky=tk.E)
        ttk.Label(self, text=_text["pred"]).grid(row=2, column=0, padx=2, sticky=tk.E)
        ttk.Label(self, text=_text["date"]).grid(row=3, column=0, padx=2, sticky=tk.E)
        
        frame = ttk.LabelFrame(self, text=_text["risktype"])
        frame.grid(row=0, column=2, rowspan=5)
        self._risk_choice = tk.IntVar()
        rb = ttk.Radiobutton(frame, text=_text["rrisk"], value=0, variable=self._risk_choice, command=self._risk_choice_change)
        rb.grid(row=0, column=0, padx=2, sticky=tk.W)
        tooltips.ToolTipYellow(rb, _text["rrisk_tt"])
        rb = ttk.Radiobutton(frame, text=_text["top1"], value=1, variable=self._risk_choice, command=self._risk_choice_change)
        rb.grid(row=1, column=0, padx=2, sticky=tk.W)
        tooltips.ToolTipYellow(rb, _text["top1_tt"])
        rb = ttk.Radiobutton(frame, text=_text["top5"], value=2, variable=self._risk_choice, command=self._risk_choice_change)
        rb.grid(row=2, column=0, padx=2, sticky=tk.W)
        tooltips.ToolTipYellow(rb, _text["top5_tt"])
        rb = ttk.Radiobutton(frame, text=_text["top10"], value=3, variable=self._risk_choice, command=self._risk_choice_change)
        rb.grid(row=3, column=0, padx=2, sticky=tk.W)
        tooltips.ToolTipYellow(rb, _text["top10_tt"])
        subframe = ttk.Frame(frame)
        subframe.grid(row=4, column=0, sticky=tk.W)
        rb = ttk.Radiobutton(subframe, text=_text["topcus"], value=4, variable=self._risk_choice, command=self._risk_choice_change)
        rb.grid(row=0, column=0, padx=2, sticky=tk.W)
        tooltips.ToolTipYellow(rb, _text["topcus_tt"])
        self._risk_level = tk.StringVar()
        self._risk_level.set("5")
        self._risk_level_entry = ttk.Entry(subframe, textvariable=self._risk_level)
        self._risk_level_entry.grid(row=0, column=1, padx=2, sticky=tk.W)
        self._risk_level_entry["state"] = tk.DISABLED
        util.PercentageValidator(self._risk_level_entry, self._risk_level, callback=self._risk_choice_change)

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

    def _cbox_or_label(self, choices, command=None):
        """Produces a :class:`ttk.Combobox` unless `choices` is of length 1,
        in which case just produces a label.

        :return: Pair of `(widget, flag)` where `flag` is True if and only if
          we produced a box.
        """
        if len(choices) == 1:
            p = choices[0]
            label = ttk.Label(self, text=str(p))
            return label, False
        else:
            cbox = ttk.Combobox(self, height=5, state="readonly")
            cbox["values"] = choices
            cbox.bind("<<ComboboxSelected>>", command)
            cbox.current(0)
            cbox["width"] = max(len(str(t)) for t in choices)
            return cbox, True

    def update_projections(self):
        w, flag = self._cbox_or_label(self.controller.model.projections, command=self._proj_chosen)
        w.grid(row=0, column=1, padx=2, pady=2, sticky=tk.W)
        if flag:
            self._proj_cbox = w
        else:
            self._proj_cbox = None
        self._proj_choice = 0
        self.controller.notify_projection_choice(0)

    def _proj_chosen(self, event):
        self._proj_choice = event.widget.current()
        self.controller.notify_projection_choice(self._proj_choice)

    @property
    def projection_choice(self):
        """Pair of (index, string_value)"""
        if self._proj_cbox is None:
            return 0, self.controller.model.projections[0]
        return self._proj_choice, self._proj_cbox["values"][self._proj_choice]

    def update_grids(self, choices):
        self._grid_choices = list(choices)
        w, flag = self._cbox_or_label(self._grid_choices, command=self._grid_chosen)
        w.grid(row=1, column=1, padx=2, pady=2, sticky=tk.W)
        if flag:
            self._grid_cbox = w
        else:
            self._grid_cbox = None
        self._grid_choice = 0
        self.controller.notify_grid_choice(0)
        
    def _grid_chosen(self, event):
        self._grid_choice = event.widget.current()
        self.controller.notify_grid_choice(self._grid_choice)

    @property
    def grid_choice(self):
        """Pair of (index, string_value)"""
        if self._grid_cbox is None:
            return 0, self._grid_choices[0]
        return self._grid_choice, self._grid_cbox["values"][self._grid_choice]
        
    def update_predictions(self, choices):
        self._pred_choices = list(choices)
        w, flag = self._cbox_or_label(self._pred_choices, command=self._pred_chosen)
        w.grid(row=2, column=1, padx=2, pady=2, sticky=tk.W)
        if flag:
            self._pred_cbox = w
        else:
            self._pred_cbox = None
        self._pred_choice = 0
        self.controller.notify_pred_choice(0)
        
    def _pred_chosen(self, event):
        self._pred_choice = event.widget.current()
        self.controller.notify_pred_choice(self._pred_choice)

    @property
    def prediction_choice(self):
        """Pair of (index, string_value)"""
        if self._pred_cbox is None:
            return 0, self._pred_choices[0]
        return self._pred_choice, self._pred_cbox["values"][self._pred_choice]

    def update_dates(self, choices):
        self._date_choices = list(choices)
        w, flag = self._cbox_or_label(self._date_choices, command=self._date_chosen)
        w.grid(row=3, column=1, padx=2, pady=2, sticky=tk.W)
        if flag:
            self._date_cbox = w
        else:
            self._date_cbox = None
        self._date_choice = 0
        self.controller.notify_date_choice(0)
        
    def _date_chosen(self, event):
        self._date_choice = event.widget.current()
        self.controller.notify_date_choice(self._date_choice)

    @property
    def date_choice(self):
        """Pair of (index, string_value)"""
        if self._date_cbox is None:
            return 0, self._date_choices[0]
        return self._date_choice, self._date_cbox["values"][self._date_choice]

    @property
    def model(self):
        return self.controller.model

    def update_prediction(self, level=-1):
        """Change the plotted prediction.

        :param level: The % of "coverage" to display, or -1 to mean plot all the
          relative risk.
        """
        def make_fig():
            fig = mtp.new_figure((20,20))
            ax = fig.add_subplot(1,1,1)
            prediction = self.model.current_prediction.prediction
            if level == -1:
                plot_risk(prediction, ax)
            else:
                plot_coverage(prediction, level, ax)
            ax.set_aspect(1)
            fig.set_tight_layout(True)
            return fig
        if not hasattr(self, "_plot") or self._plot is None:
            frame = ttk.LabelFrame(self, text=_text["graph_name"])
            frame.grid(row=10, column=0, columnspan=3, padx=2, pady=2, sticky=tk.NSEW)
            util.stretchy_rows_cols(frame, [0], [0])
            util.stretchy_rows(self, [10])
            util.stretchy_columns(self, [0,1,2])
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

def plot_risk(prediction, axis):
    """Plot the risk defined by the grid prediction to the axis.

    :param prediction: Instance of :class:`GridPrediction`
    :param axis: `matplotlib` axis object.
    """
    axis.pcolormesh(*prediction.mesh_data(), prediction.intensity_matrix, cmap="Blues")#yellow_to_red)

def plot_coverage(prediction, level, axis):
    """Plot the risk defined by the grid prediction to the axis, constraining
    to a set % level of coverage.

    :param prediction: Instance of :class:`GridPrediction`
    :param level: The level of coverage to plot.
    :param axis: `matplotlib` axis object.
    """
    current_mask = _np.ma.getmaskarray(prediction.intensity_matrix)
    in_coverage = open_cp.evaluation.top_slice(prediction.intensity_matrix, level / 100)
    current_mask = current_mask | (~in_coverage)
    matrix = _np.ma.masked_array(prediction.intensity_matrix, current_mask)
    axis.pcolormesh(*prediction.mesh_data(), matrix, cmap=yellow_to_red)
    