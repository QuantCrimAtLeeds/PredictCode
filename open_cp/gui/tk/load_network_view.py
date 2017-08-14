"""
load_network_view
~~~~~~~~~~~~~~~~~

"""

import tkinter as tk
import tkinter.ttk as ttk
from . import util
from .. import funcs
#from . import mtp
from . import tooltips
#from . import date_picker
from .. import load_network_model
NetworkType = load_network_model.NetworkModel.NetworkType

_text = {
    "title" : "Load Network Geometry",
    "what" : "Load a geometry file which specifies the road/street network.",
    "okay" : "Accept",
    "cancel" : "Cancel",
    "net" : "Network",
    "load_file" : "Load geometry file",
    "current_file" : "Current file: {}",
    "none" :  "None loaded",
    "preview" : "Network preview",
    "os_type" : "Ordnance Survey style",
    "os_type_tt" : ("Assume the network geometry is similar to the UK Ordnance Survey OpenRoads data.  "
            + "This assumes that paths intersect only at their end-points.  This allows for accurate "
            + "representation of over/under passes, for example."),
    "tl_type" : "TIGER/Lines (tm) style",
    "tl_type_tt" : ("Assume the network geometry is similar to the USA TIGER/Lines (tm) data.  "
            + "This assumes that every path is composed of line segements.  These line segments are "
            + "considered individually.  If the coordinates of different start/end points of the line "
            + "segment are very close, they are merged.  This might lead us to identify over/under "
            + "passes, for example, but it correctly detects road junctions etc. in this dataset." ),
    
}

class LoadNetworkView(util.ModalWindow):
    def __init__(self, parent, controller):
        super().__init__(parent, title=_text["title"], resize="hw")
        self.controller = controller
        util.centre_window_percentage(self, 75, 66)
        self._refresh_network_view()
        
    @property
    def model(self):
        return self.controller.model
        
    def add_widgets(self):
        ttk.Label(self, text=_text["what"]).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        frame = ttk.LabelFrame(self, text=_text["net"])
        frame.grid(row=10, column=0, sticky=tk.NSEW, padx=2, pady=2)
        subframe = ttk.Frame(frame)
        subframe.grid(row=0, column=0, sticky=tk.W)
        ttk.Button(subframe, text=_text["load_file"], command=self._load_file).grid(row=0, column=0, padx=2, pady=2)
        self._filename_label = ttk.Label(subframe)
        self._filename_label.grid(row=0, column=1, padx=2, pady=2)
        
        subframe = ttk.Frame(frame)
        subframe.grid(row=1, column=0, sticky=tk.NSEW)
        
        subsubframe = ttk.Frame(subframe)
        subsubframe.grid(row=0, column=0, sticky=tk.NSEW)
        self._network_type = tk.IntVar()
        rb = ttk.Radiobutton(subsubframe, text=_text["os_type"], value=NetworkType.ORDNANCE.value, variable=self._network_type, command=self._new_network_type)
        rb.grid(row=0, column=0, sticky=tk.W)
        tooltips.ToolTipYellow(rb, _text["os_type_tt"])
        rb = ttk.Radiobutton(subsubframe, text=_text["tl_type"], value=NetworkType.TIGER_LINES.value, variable=self._network_type, command=self._new_network_type)
        rb.grid(row=1, column=0, sticky=tk.W)
        tooltips.ToolTipYellow(rb, _text["tl_type_tt"])
        
        self._preview_frame = ttk.LabelFrame(subframe, text=_text["preview"])
        self._preview_frame.grid(row=0, column=1, sticky=tk.NSEW)
        
        frame = ttk.Frame(self)
        frame.grid(row=100, column=0, sticky=tk.EW, pady=2)
        ttk.Button(frame, text=_text["okay"], command=self.okay_pressed).grid(row=0, column=0, padx=5)
        ttk.Button(frame, text=_text["cancel"], command=self.cancel).grid(row=0, column=1, padx=5)
        
    def _refresh_network_view(self):
        if self.model.filename is not None:
            name = funcs.string_ellipse(self.model.filename, 80)
        else:
            name = _text["none"]
        self._filename_label["text"] = _text["current_file"].format(name)
        
        self._network_type = self.model.network_type.value
        
    def _load_file(self):
        # TODO
        pass
        
    def _new_network_type(self):
        # TODO
        pass
    
    def okay_pressed(self):
        self.okay = True
        super().cancel()
        
    def cancel(self):
        self.okay = False
        super().cancel()
        