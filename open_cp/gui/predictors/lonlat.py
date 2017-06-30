"""
lonlat
~~~~~~

Transform lon/lat coords into meters
"""

from . import predictor
import open_cp.gui.import_file_model as import_file_model
import logging
import numpy as _np
from open_cp.gui.common import CoordType

_text = {
    "bi" : "Builtin",
    "bi_tt" : "Approximation; works, but UTM or an EPSG should be better.",
    "utm" : "Use best match UTM",
    "utm_tt" : "Uses the best match Universal Transverse Mercator coordinate system.  This is likely the best choice unless you know an EPSG projection for your data.",
    "uk" : "British national grid",
    "uk_tt" : "The standard British National Grid projection",
    "epsg" : "EPSG code",
    "epsg_tt" : "Manually enter an EPSG code",
    "epsg_entry_tt" : "Enter a valid EPSG code",
    "epsg_url" : "http://spatialreference.org/ref/epsg/",
    "epsg_url_text" : "(Click for a list)",
    "no_pyproj" : "Python package `pyproj` could not be loaded, so no more options.",
    "alpr" : "Coordinates are already projected, so no need to project from Lon/Lat!",
}

## Actual work classes ###################################################

class Builtin():
    """An approximation; see
    https://en.wikipedia.org/wiki/Geographic_coordinate_system#Expressing_latitude_and_longitude_as_linear_units
    """
    def __init__(self, ycoords):
        average_lat = _np.average(_np.asarray(ycoords))
        phi = _np.pi * average_lat / 180
        self._y = 111132.92 - 559.82 * _np.cos(2 * phi) + 1.175 * _np.cos(4 * phi)
        self._x = 111412.84 * _np.cos(phi) - 93.5 * _np.cos(3 * phi)

    def __call__(self, lon, lat):
        return self._x * _np.asarray(lon), self._y * _np.asarray(lat)


try:
    import pyproj
except:
    pyproj = None
    logging.getLogger(__name__).error("Failed to load `pyproj`.")


class ViaUTM():
    """Use the suitable UTM for the input data.

    https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system

    Uses `pyproj` to do the actual projecting.
    """
    def __init__(self, xcoords):
        average_longitude = _np.average(_np.asarray(xcoords))
        utm_zone = int(_np.floor((average_longitude + 180) / 6) + 1)
        self._proj = pyproj.Proj(datum="NAD83", ellps="GRS80", proj="utm", units="m", zone=utm_zone)

    def __call__(self, lon, lat):
        return self._proj(lon, lat)


class EPSG():
    """Use an epsg setting"""
    def __init__(self, epsg):
        self._proj = pyproj.Proj(init="epsg:"+str(epsg))

    def __call__(self, lon, lat):
        return self._proj(lon, lat)


class BritishNationalGrid(EPSG):
    """Use EPSG:27700 which is "British National Grid"."""
    def __init__(self):
        super().__init__(27700)


## The Predictor class ###################################################

class PassThrough():
    """For use when the data is already projected.  Selected automatically by
    the model."""
    def __init__(self, model):
        if model.coord_type != import_file_model.CoordType.XY:
            raise ValueError("Can only be used on data already projected.")

    class Task(predictor.ProjectTask):
        def __init__(self):
            super().__init__(predictor._TYPE_COORD_PROJ)

        def __call__(self, xcoords, ycoords):
            return xcoords, ycoords

    def make_tasks(self):
        return [self.Task()]

    def pprint(self):
        return "Coordinates already projected"


class LonLatConverter(predictor.Predictor):
    def __init__(self, model):
        super().__init__(model)
        if model.coord_type != CoordType.LonLat:
            raise ValueError(_text["alpr"])
        self.selected = 0
        self._espg = 27700
        self._view = None

    @staticmethod
    def describe():
        return "Project LonLat coordinates to meters"

    @staticmethod
    def order():
        return predictor._TYPE_COORD_PROJ

    def make_view(self, parent, inline=False):
        self._view = LonLatConverterView(parent, self)
        return self._view

    @property
    def name(self):
        return "Project longitude/latitude coordinates to meters."

    @property
    def settings_string(self):
        return self._settings

    def to_dict(self):
        return { "selected" : self.selected,
            "epsg" : self._espg }

    def from_dict(self, data):
        self.selected = data["selected"]
        self._epsg = data["epsg"]

    @property
    def selected(self):
        return self._selected
    
    @selected.setter
    def selected(self, value):
        if value == 0:
            self._projector = Builtin(self._ycoords)
            self._settings = "<Builtin best guess>"
        elif value == 1:
            self._projector = ViaUTM(self._xcoords)
            self._settings = "<UTM auto-detect>"
        elif value == 2:
            self._projector = BritishNationalGrid()
            self._settings = "<British national grid>"
        else:
            raise ValueError()
        self._selected = value

    def set_epsg(self, code):
        self._selected = 3
        try:
            self._projector = EPSG(code)
        except RuntimeError:
            self._view.set_epsg_code(self._espg)
        self._espg = code
        self._settings = "<EPSG {}>".format(code)

    def get_epsg(self):
        return self._epsg

    class Task(predictor.ProjectTask):
        def __init__(self, delegate):
            super().__init__()
            self._delegate = delegate

        def __call__(self, xcoords, ycoords):
            return self._delegate(_np.asarray(xcoords), _np.asarray(ycoords))

    def make_tasks(self):
        return [self.Task(self._projector)]


## GUI related ###########################################################

import tkinter as tk
import tkinter.ttk as ttk
import open_cp.gui.tk.util as util
import open_cp.gui.tk.tooltips as tooltips

class LonLatConverterView(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self._controller = controller
        self._add_widgets()

    def _add_widgets(self):
        self._selected = tk.StringVar()
        rb = ttk.Radiobutton(self, text=_text["bi"], value=0, variable=self._selected, command=self._sel_changed)
        rb.grid(row=0, column=0, sticky=tk.W)
        tooltips.ToolTipYellow(rb, _text["bi_tt"])
        if pyproj is not None:
            choices = ["utm", "uk", "epsg"]
            for index, name in enumerate(choices):
                rb = ttk.Radiobutton(self, text=_text[name], value=index+1, variable=self._selected, command=self._sel_changed)
                rb.grid(row=index+1, column=0, sticky=tk.W)
                tooltips.ToolTipYellow(rb, _text[name+"_tt"])
            self._espg_value = tk.StringVar()
            self._espg_entry = ttk.Entry(self, textvariable=self._espg_value)
            self._espg_entry.grid(row=3, column=1, sticky=tk.W)
            util.IntValidator(widget=self._espg_entry, variable=self._espg_value, callback=self._sel_changed, allow_empty=False)
            href = util.HREF(self, text=_text["epsg_url_text"], url=_text["epsg_url"])
            href.grid(row=3, column=2)
            tooltips.ToolTipYellow(href, _text["epsg_url"])
        else:
            ttk.Label(self, text=_text["no_pyproj"], wraplength=200).grid(row=1, column=0, sticky=tk.W)
        self._init()

    def _init(self):
        if self._controller.selected < 3:
            self._selected.set(self._controller.selected)
            self._espg_entry["state"] = tk.DISABLED
        elif self._controller.selected == 3:
            self._espg_entry["state"] = tk.ACTIVE
            self._selected.set(3)
            self._espg_value.set(self._controller.get_epsg())

    def _sel_changed(self):
        value = int(self._selected.get())
        if value < 3:
            self._controller.selected = value
            self._espg_entry["state"] = tk.DISABLED
        elif value == 3:
            self._espg_entry["state"] = tk.ACTIVE
            code = self._espg_value.get()
            if code is None or code == "":
                code = 27700
                self._espg_value.set(code)
            self._controller.set_epsg(code)

    def set_epsg_code(self, code):
        self._espg_value.set(code)

def test(root):
    ll = LonLatConverter(predictor.test_model())
    predictor.test_harness(ll, root)
    