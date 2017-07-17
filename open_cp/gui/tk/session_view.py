"""
session_view
~~~~~~~~~~~~
"""


import tkinter as tk
import tkinter.ttk as ttk
import open_cp.gui.tk.util as util

_text = {
    "none" : "No recent sessions found",
    "cancel" : "Cancel",
    
}

class SessionView(ttk.Frame):
    def __init__(self, parent, controller, model):
        super().__init__(parent)
        self._parent = parent
        self.controller = controller
        self.model = model
        
        self.master.protocol("WM_DELETE_WINDOW", self.cancel)
        self.grid(sticky=util.NSEW)
        util.stretchy_rows_cols(self, range(101), [0])
        self._add_widgets()
        self.resize()
        
    def resize(self, final=False):
        self.update_idletasks()
        util.centre_window(self._parent, self._parent.winfo_reqwidth(), self._parent.winfo_reqheight())
        
    def _add_widgets(self):
        if len(self.model.recent_sessions) == 0:
            la = ttk.Label(self, text=_text["none"], anchor=tk.CENTER)
            la.grid(row=0, column=0, padx=2, pady=2, sticky=tk.EW)
        
        for index, name in enumerate(self.model.recent_sessions):
            b = ttk.Button(self, text=name, command=lambda i=index : self._pressed(i))
            b.grid(row=index, column=0, padx=2, pady=2, sticky=tk.NSEW)
            
        b = ttk.Button(self, text=_text["cancel"], command=self.cancel)
        b.grid(row=100, column=0, padx=2, pady=2, sticky=tk.NSEW)

    def _pressed(self, index):
        self.controller.selected(index)
        
    def cancel(self, event=None):
        self.destroy()
