"""
import_file_view
~~~~~~~~~~~~~~~~
"""

import tkinter as tk
import tkinter.filedialog as tk_fd
import tkinter.ttk as ttk
from . import util
from . import simplesheet

def get_file_name():
    return tk_fd.askopenfilename(defaultextension=".csv",
                                 title="Please select a CSV file to open")


class LoadFileProgress(tk.Frame):
    def __init__(self):
        super().__init__()
        self.grid()
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        bar = ttk.Progressbar(self, mode="indeterminate")
        bar.grid(sticky=util.NSEW)
        bar.start()


class ImportFileView(tk.Frame):
    def __init__(self, model):
        super().__init__()
        self.model = model
        util.centre_window_percentage(self.master, 80, 60)
        #self["width"] = 300
        #self["height"] = 200
        #self.grid_propagate(0)
        self.grid()
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.add_widgets()
                

    def add_widgets(self):
        self.unprocessed = simplesheet.SimpleSheet(self)
        self.unprocessed.grid(row=0, column=0)#, sticky=util.NSEW)
        self.unprocessed.set_columns(self.model.header)
        for c, _ in enumerate(self.model.header):
            # TODO: More sensible?
            self.unprocessed.set_column_width(c, 50)
        for r, row in enumerate(self.model.firstrows):
            self.unprocessed.add_row()
            for c, entry in enumerate(row):
                self.unprocessed.set_entry(r, c, entry)
        self.unprocessed.height = len(self.model.firstrows)
        sx = self.unprocessed.xscrollbar(self)
        sx.grid(row=1, column=0, sticky=(tk.E, tk.W))
            
