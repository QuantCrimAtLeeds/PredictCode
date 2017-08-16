"""
projectors_view
~~~~~~~~~~~~~~~

`tkinter` code for `projectors.py`


"""

import tkinter as tk
import tkinter.ttk as ttk
import open_cp.gui.tk.util as util
import open_cp.gui.tk.tooltips as tooltips

_text = {
    "icrs" : "Input crs: {}",
    "icrstt" : ("The input Coordinate Reference System (CRS) detected for the geometry.  Specifies "
            + "how the geometry related to the real world.  The code 'epsg:4326' indicates that "
            + "the data is in Longitude/Latitude format; this is also often the default choice."),
    "icrstt1" : "No input projection information found, so we assumed the coordinates are longitude / latitude.",
    "newcrs" : "Transform to epsg:",
    "epsg_url" : "http://spatialreference.org/ref/epsg/",
    "epsg_url_text" : "(Click for a list)",
    "newcrstt" : ("The EPSG code to transform the geometry to.  "
            + "This should be the same as the projection used for the input data.  "
            + "If left blank, we make a best guess, which is likely the best choice if the events are given as Longitude/Latitude."),
    "guesscrs" : "Assumed Longitude/Latitude",
    
}


class GeoFrameProjectorWidget(tk.Frame):
    """A :class:`tkinter.Frame` which contains details of the input CRS,
    and an entry box to specify a new EPSG code, together with a hyperlink to
    a list of codes.
    
    :param parent: The `tkinter` parent widget.
    :param model: Instance of :class:`projectors.GeoFrameProjector`
    :param callback: Optional callable object to be invoked when the user
      changes the EPSG code.
    """
    def __init__(self, parent, model, callback=None):
        super().__init__(parent)
        self._model = model
        self.callback = callback
        
        subframe = ttk.Frame(self)
        subframe.grid(row=2, column=0, padx=2, pady=3, sticky=tk.NW)
        self._input_crs_label = ttk.Label(subframe, text=_text["icrs"].format(""))
        self._input_crs_label.grid(row=0, column=0, padx=2)
        self._input_crs_lavel_tt=tooltips.ToolTipYellow(self._input_crs_label, _text["icrstt"])
        label = ttk.Label(subframe, text=_text["newcrs"])
        label.grid(row=0, column=1, padx=2)
        self._output_crs_entry_var = tk.StringVar()
        self._output_crs_entry = ttk.Entry(subframe, textvariable=self._output_crs_entry_var)
        self._output_crs_entry.grid(row=0, column=2, padx=2)
        util.IntValidator(self._output_crs_entry, self._output_crs_entry_var, callback=self._new_epsg, allow_empty=True)
        tooltips.ToolTipYellow(label, _text["newcrstt"])
        href = util.HREF(subframe, text=_text["epsg_url_text"], url=_text["epsg_url"])
        href.grid(row=0, column=3, padx=2)
        tooltips.ToolTipYellow(href, _text["epsg_url"])

    def update(self):
        """Update the view from the model."""
        self._input_crs_lavel_tt.text = _text["icrstt"]
        crs = self._model.crs
        if self._model.guessed_crs:
            crs = _text["guesscrs"]
            self._input_crs_lavel_tt.text = _text["icrstt1"]
        self._input_crs_label["text"] = _text["icrs"].format(crs)

        if self._model.epsg is None:
            self._output_crs_entry_var.set("")
        else:
            self._output_crs_entry_var.set(self._model.epsg)

    def _new_epsg(self):
        epsg = self._output_crs_entry_var.get()
        if epsg is None or epsg == "":
            self._model.epsg = None
        else:
            self._model.epsg = int(epsg)
        self.update()
        if self.callback is not None:
            self.callback()

    @property
    def callback(self):
        return self._callback
    
    @callback.setter
    def callback(self, v):
        self._callback = v
