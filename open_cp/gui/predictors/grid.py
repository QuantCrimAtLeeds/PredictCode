"""
grid
~~~~

Produce a grid over the input data
"""

from . import predictor
import open_cp.data
import open_cp.geometry
import numpy as _np

## The Predictor #########################################################

class GridProvider(predictor.Predictor):
    def __init__(self, model):
        super().__init__(model)
        self._grid_size = 100
        self._xoffset = 0
        self._yoffset = 0

    @staticmethod
    def describe():
        return "Produce a grid over the data"

    @staticmethod
    def order():
        return predictor._TYPE_GRID

    @property
    def name(self):
        return "Grid {}x{}m @ ({}m, {}m)".format(self._grid_size, self._grid_size, self._xoffset, self._yoffset)

    @property
    def settings_string(self):
        return None

    def make_view(self, parent, inline=False):
        self._view = GridView(parent, self, inline)
        return self._view

    def to_dict(self):
        return { "size" : self._grid_size,
                "xoffset" : self._xoffset,
                "yoffset" : self._yoffset
            }

    def from_dict(self, data):
        self._grid_size = data["size"]
        self._xoffset = data["xoffset"]
        self._yoffset = data["yoffset"]

    def get_grid(self):
        """Make a grid for the current settings.  Clipped to contain all data
        points."""
        grid = open_cp.data.Grid(xsize=self._grid_size, ysize=self._grid_size,
            xoffset=self._xoffset, yoffset=self._yoffset)
        return open_cp.geometry.mask_grid_by_points_intersection(
                self._as_coords(), grid, bbox=True)

    class Task(predictor.GridTask):
        def __init__(self, size, xo, yo):
            super().__init__()
            self._grid_size = size
            self._xoffset = xo
            self._yoffset = yo
            
        def __call__(self, timed_points):
            grid = open_cp.data.Grid(xsize=self._grid_size, ysize=self._grid_size,
                                     xoffset=self._xoffset, yoffset=self._yoffset)
            return open_cp.geometry.mask_grid_by_points_intersection(
                timed_points, grid, bbox=True)

    def make_tasks(self):
        return [self.Task(self.size, self.xoffset, self.yoffset)]

    @property
    def size(self):
        return self._grid_size

    @size.setter
    def size(self, value):
        self._grid_size = int(value)

    @property
    def xoffset(self):
        return self._xoffset

    @xoffset.setter
    def xoffset(self, value):
        value = float(value)
        if value < 0 or value >= self.size:
            raise ValueError("Should be between 0 and the grid size")
        if int(value) == value:
            value = int(value)
        self._xoffset = value

    @property
    def yoffset(self):
        return self._yoffset

    @yoffset.setter
    def yoffset(self, value):
        value = float(value)
        if value < 0 or value >= self.size:
            raise ValueError("Should be between 0 and the grid size")
        if int(value) == value:
            value = int(value)
        self._yoffset = value

    def config(self):
        return {"resize":True}


## GUI Stuff #############################################################

import tkinter as tk
import tkinter.ttk as ttk
import open_cp.gui.tk.tooltips as tooltips
import open_cp.gui.tk.util as util
import open_cp.gui.tk.mtp as mtp
import open_cp.plot

_text = {
    "size" : "Grid size",
    "sizett" : "The width and height of each grid cell.",
    "xo" : "X Offset",
    "yo" : "X Offset",
    "xyott" : "A value between 0 and the size of the grid, giving the relative offset of the grid.  Changing shifts the grid around with respect to the data points.",
    "no_proj" : "Data in Longitude/Latitude format and no projection is selected, so cannot visualise.",
    "no_proj_tt" : "You must select a projection method before analysis.",
    "vp" : "Visual preview",
}

class GridView(tk.Frame):
    def __init__(self, parent, controller, inline=False):
        super().__init__(parent)
        self._controller = controller
        self._inline = inline
        util.stretchy_rows_cols(self, [3], [2])
        self._add_widgets()

    def _add_widgets(self):
        label = ttk.Label(self, text=_text["size"])
        label.grid(row=0, column=0, padx=2, pady=2)
        tooltips.ToolTipYellow(label, _text["sizett"])
        self._size = tk.StringVar()
        self._size_entry = ttk.Entry(self, textvariable=self._size)
        self._size_entry.grid(row=0, column=1, padx=2, pady=2)
        util.IntValidator(self._size_entry, self._size, self._change, False)

        label = ttk.Label(self, text=_text["xo"])
        label.grid(row=1, column=0, padx=2, pady=2)
        tooltips.ToolTipYellow(label, _text["xyott"])
        self._xo = tk.StringVar()
        self._xo_entry = ttk.Entry(self, textvariable=self._xo)
        self._xo_entry.grid(row=1, column=1, padx=2, pady=2)
        util.FloatValidator(self._xo_entry, self._xo, self._change, False)

        label = ttk.Label(self, text=_text["yo"])
        label.grid(row=2, column=0, padx=2, pady=2)
        tooltips.ToolTipYellow(label, _text["xyott"])
        self._yo = tk.StringVar()
        self._yo_entry = ttk.Entry(self, textvariable=self._yo)
        self._yo_entry.grid(row=2, column=1, padx=2, pady=2)
        util.FloatValidator(self._yo_entry, self._yo, self._change, False)

        self._plot = None
        self._plot_grid()
        self._init()

    def _init(self):
        self._size.set(self._controller.size)
        self._xo.set(self._controller.xoffset)
        self._yo.set(self._controller.yoffset)

    def _no_proj(self):
        if self._plot is None:
            self._plot = ttk.Label(self, text=_text["no_proj"], wraplength=200)
            self._plot.grid(sticky=tk.W, padx=2, pady=2, row=3, column=0, columnspan=2)
            tooltips.ToolTipYellow(self._plot, _text["no_proj_tt"])

    def _plot_grid(self):
        coords = self._controller._projected_coords()
        if coords is None:
            return self._no_proj()
        if self._inline:
            return

        def make_fig():
            xmin, xmax = _np.min(coords[0]), _np.max(coords[0])
            xd = (xmax - xmin) / 100 * 3
            xmin, xmax = xmin - xd, xmax + xd
            ymin, ymax = _np.min(coords[1]), _np.max(coords[1])
            yd = (ymax - ymin) / 100 * 3
            ymin, ymax = ymin - yd, ymax + yd

            width = xmax - xmin
            height = ymax - ymin
            if width == 0 or height == 0:
                size=(10,10)
            else:
                height = height / width * 10.0
                width = 10.0
                if height > 10.0:
                    width = 100.0 / height
                    height = 10.0
                size = (width, height)
                
            fig = mtp.new_figure(size)
            ax = fig.add_subplot(1,1,1)
            ax.scatter(coords[0], coords[1], marker="x", color="black", alpha=0.5)
            lc = open_cp.plot.lines_from_regular_grid(self._controller.get_grid())
            ax.add_collection(mtp.matplotlib.collections.LineCollection(lc, color="black", linewidth=0.5))
            ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax])
            ax.set_aspect(1.0)
            fig.set_tight_layout(True)
            return fig
        
        if self._plot is None:
            frame = ttk.LabelFrame(self, text=_text["vp"])
            frame.grid(row=0, column=2, rowspan=4, padx=2, pady=2, sticky=tk.NSEW)
            util.stretchy_rows_cols(frame, [0], [0])
            self._plot = mtp.CanvasFigure(frame)
            self._plot.grid(padx=2, pady=2, sticky=tk.NSEW)
        self._plot.set_figure_task(make_fig, dpi=150)

    def _change(self, event=None):
        old = (self._controller.size, self._controller.xoffset, self._controller.yoffset)
        try:
            self._controller.size = self._size.get()
        except:
            pass
        try:
            self._controller.xoffset = self._xo.get()
        except:
            pass
        try:
            self._controller.yoffset = self._yo.get()
        except:
            pass
        self._init()
        if old != (self._controller.size, self._controller.xoffset, self._controller.yoffset):
            self._plot_grid()

def test(root):
    ll = GridProvider(predictor.test_model())
    predictor.test_harness(ll, root)
