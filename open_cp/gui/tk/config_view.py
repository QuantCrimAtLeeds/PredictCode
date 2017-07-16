"""
config_view
~~~~~~~~~~~
"""

import tkinter as tk
import tkinter.ttk as ttk
import open_cp.gui.tk.util as util

_text = {
    "config" : "Configuration",
    "ttk" : "ttk 'theme':",

}

class ConfigView(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self._parent = parent
        self.controller = controller
        #util.centre_window_percentage(self.master, 80, 60)
        #self.master.minsize(400, 300)
        self.master.protocol("WM_DELETE_WINDOW", self._cancel)
        self.grid(sticky=util.NSEW)
        #util.stretchy_columns(self, [0])
        self._add_widgets()

    def _add_widgets(self):
        frame = ttk.LabelFrame(self, text=_text["config"])
        frame.grid(row=0, column=0, sticky=tk.NSEW)
        ttk.Label(frame, text=_text["ttk"]).grid(row=0, column=0)
        self.theme_cbox = ttk.Combobox(frame, height=5, state="readonly")
        self.theme_cbox.bind("<<ComboboxSelected>>", self._theme_selected)
        self.theme_cbox.grid(row=0, column=1)

    def set_themes(self, values):
        self.theme_cbox["values"] = list(values)

    def set_theme_selected(self, choice):
        self.theme_cbox.current(choice)

    def set_theme(self, name):
        self._parent.style.theme_use(name)

    def _theme_selected(self, event):
        index = int(self.theme_cbox.current())
        self.controller.selected_theme(index)

    def theme_names(self):
        """List of theme names"""
        return list(self._parent.style.theme_names())

    def current_theme(self):
        return self._parent.style.theme_use()

    def _cancel(self):
        self.destroy()
