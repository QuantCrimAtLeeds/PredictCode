"""
config_view
~~~~~~~~~~~
"""

import tkinter as tk
import tkinter.ttk as ttk
import open_cp.gui.tk.util as util
import open_cp.gui.tk.richtext as richtext
import open_cp.gui.tk.tooltips as tooltips
import open_cp.gui.funcs as funcs
import numpy as np
import scipy as scipy
try:
    import geopandas as gpd
except:
    gpd = None
try:
    import pyproj
except:
    pyproj = None
import tilemapbase
import sys

_text = {
    "config" : "Configuration",
    "ttk" : "ttk 'theme':",
    "ttk_tt" : "Select the ttk theme to use.  On Windows / OS X leaving this alone is strongly recommended",
    "info" : "System information",
    "info_tt" : "Information about the Python system you are running",
    "plat" : "Platform: {}\n",
    "tcl" :  "TCL version in use: {}\n",
    "pyplat" : "Python platform: {}\n",
    "pyroot" : "Python root path: {}\n",
    "np" : "Numpy version: {}\n",
    "scipy" : "Scipy version: {}\n",
    "gpd" : "GeoPandas version: {}\n",
    "gpd_none" : "GeoPandas could not be loaded\n",
    "pyproj" : "Pyproj version: {}\n",
    "tilemapbase" : "TileMapBase version: {}\n",
    "tilemapbase_none" : "TileMapBase could not be loaded\n",
    "pyproj_none" : "Pyproj could not be loaded\n",
    "okay": "Okay",
    "cancel": "Cancel",
    "sf" : "Settings filename: {}",
    
}

class ConfigView(tk.Frame):
    def __init__(self, parent, model, controller):
        super().__init__(parent)
        self._parent = parent
        self.controller = controller
        self.model = model
        self.master.protocol("WM_DELETE_WINDOW", self._cancel)
        self.grid(sticky=util.NSEW)
        util.stretchy_rows_cols(self, [11], [0])
        self._add_widgets()
        self.resize()        

    def resize(self, final=False):
        self.update_idletasks()
        util.centre_window(self._parent, self._parent.winfo_reqwidth(), self._parent.winfo_reqheight())
        if not final:
            self.after_idle(lambda : self.resize(True))

    def _add_widgets(self):
        frame = ttk.Frame(self)
        frame.grid(row=0, column=0, padx=2, pady=2, sticky=tk.W)
        la = ttk.Label(frame, text=_text["sf"].format(funcs.string_ellipse(self.model.settings_filename, 80)))
        la.grid(row=0, column=0)
        frame = ttk.LabelFrame(self, text=_text["config"])
        frame.grid(row=10, column=0, padx=2, pady=2, sticky=tk.NSEW)
        self._add_config_box(frame)
        frame = ttk.LabelFrame(self, text=_text["info"])
        frame.grid(row=11, column=0, padx=2, pady=2, sticky=tk.NSEW)
        tooltips.ToolTipYellow(frame, _text["info_tt"])
        util.stretchy_rows(frame, [0])
        self._add_info_box(frame)
        frame = ttk.Frame(self)
        frame.grid(row=20, column=0, padx=2, pady=2, sticky=tk.EW)
        util.stretchy_columns(frame, [0,1])
        b = ttk.Button(frame, text=_text["okay"], command=self._okay)
        b.grid(row=0, column=0, padx=2, pady=2, sticky=tk.NSEW)
        b = ttk.Button(frame, text=_text["cancel"], command=self._cancel)
        b.grid(row=0, column=1, padx=2, pady=2, sticky=tk.NSEW)
        
    def _add_info_box(self, frame):
        self._text = richtext.RichText(frame, height=10, scroll="v")
        self._text.grid(row=0, column=0, padx=1, pady=1, sticky=tk.NSEW)
        self._text.add_text(_text["plat"].format(sys.platform))
        self._text.add_text(_text["tcl"].format(tk.Tcl().eval('info patchlevel')))
        self._text.add_text(_text["pyplat"].format(sys.implementation))
        self._text.add_text(_text["pyroot"].format(sys.base_prefix))
        self._text.add_text(_text["np"].format(np.__version__))
        self._text.add_text(_text["scipy"].format(scipy.__version__))
        if gpd is not None:
            self._text.add_text(_text["gpd"].format(gpd.__version__))
        else:
            self._text.add_text(_text["gpd_none"])
        if pyproj is not None:
            self._text.add_text(_text["pyproj"].format(pyproj.__version__))
        else:
            self._text.add_text(_text["pyproj_none"])
        if tilemapbase is not None:
            self._text.add_text(_text["tilemapbase"].format(tilemapbase.__version__))
        else:
            self._text.add_text(_text["tilemapbase_none"])
        
    def _add_config_box(self, frame):
        label = ttk.Label(frame, text=_text["ttk"])
        label.grid(row=0, column=0, padx=2, pady=2)
        tooltips.ToolTipYellow(label, _text["ttk_tt"])
        self.theme_cbox = ttk.Combobox(frame, height=5, state="readonly")
        self.theme_cbox.bind("<<ComboboxSelected>>", self._theme_selected)
        self.theme_cbox.grid(row=0, column=1, padx=2, pady=2)
        self.theme_cbox["values"] = list(self.model.themes)

    def set_theme_selected(self, choice):
        self.theme_cbox.current(choice)

    def _theme_selected(self, event):
        index = int(self.theme_cbox.current())
        self.controller.selected_theme(index)

    def _okay(self):
        self.controller.okay()
        self.destroy()
        
    def _cancel(self):
        self.controller.cancel()
        self.destroy()
