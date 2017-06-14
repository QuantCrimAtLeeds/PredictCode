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
        """Make a grid for the current settings."""
        grid = open_cp.data.Grid(xsize=self._grid_size, ysize=self._grid_size,
            xoffset=self._xoffset, yoffset=self._yoffset)
        return open_cp.geometry.mask_grid_by_points_intersection(self._as_coords(), grid, bbox=True)

    def make_tasks(self):
        raise NotImplementedError()

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
}

class GridView(tk.Frame):
    def __init__(self, parent, controller, inline=False):
        super().__init__(parent)
        self._controller = controller
        self._inline = inline
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

    def _plot_grid(self):
        if not self._inline:
            # This works, but is (a) incredibly slow,
            # and (b) steals the focus...
            fig, ax = mtp.plt.subplots()
            ax.scatter(self._controller._xcoords, self._controller._ycoords, marker="x", color="black", alpha=0.5)
            pc = open_cp.plot.patches_from_grid(self._controller.get_grid())
            ax.add_collection(mtp.matplotlib.collections.PatchCollection(pc, facecolor="None", edgecolor="black"))
            coords = self._controller._as_coords()
            xmin, xmax = _np.min(coords.xcoords), _np.max(coords.xcoords)
            xd = (xmax - xmin) / 100 * 3
            ax.set(xlim=[xmin-xd, xmax+xd])
            ymin, ymax = _np.min(coords.ycoords), _np.max(coords.ycoords)
            yd = (ymax - ymin) / 100 * 3
            ax.set(ylim=[ymin-yd, ymax+yd])
            
            if self._plot is not None:
                self._plot.destroy()
            self._plot = mtp.figure_to_canvas(fig, self)
            self._plot.grid(row=0, column=2, rowspan=4, padx=2, pady=2)

    def _change(self, event=None):
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
        self._plot_grid()

def test(root):
    ll = GridProvider(predictor.test_model())
    predictor.test_harness(ll, root)
