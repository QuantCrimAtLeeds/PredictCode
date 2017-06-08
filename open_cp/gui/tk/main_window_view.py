"""
main_window_view
~~~~~~~~~~~~~~~~
"""

import tkinter as tk
import tkinter.ttk as ttk
from . import util
import traceback
import logging

_text = {
    "name" : "OpenCP",
    "load" : "Load CSV File",
    "exit" : "Exit",
    "ss" : "Load a saved session",
    "exp" : "Export settings",
    "about" : "About"
}

class TopWindow(tk.Tk):
    """A single top-level window.  Should be a singleton, but we'll enforce
    this via creating once and then injecting into clients.  Having a constant
    top-level window makes life a lot easier with `tkinter`.
    """
    def __init__(self):
        super().__init__()
        self.grid()
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.title(_text["name"])
        
    def resize(self, width_per, height_per):
        util.centre_window_percentage(self, width_per, height_per)

    def end(self):
        self.destroy()
        self.quit()

    def report_callback_exception(self, *args):
        """Make the application exit with a log and dump of stack trace."""
        logger = logging.getLogger(__name__)
        err = traceback.format_exception(*args)
        logger.error("Exception: %s", err)
        print("".join(err))
        import os
        os._exit(1)


class MainWindowView(tk.Frame):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.grid(sticky=util.NSEW)
        util.stretchy_columns(self, [0])
        util.stretchy_rows(self, range(5))
        self.createWidgets()
        self.master.protocol("WM_DELETE_WINDOW", self.end)
        util.centre_window_percentage(self.master, 30, 30)
        
    def createWidgets(self):
        for name, cmd in [(_text["load"], self.controller.load_csv),
                        (_text["ss"], None),
                        (_text["exp"], None),
                        (_text["about"], None),
                        (_text["exit"], self.end)
                        ]:
            ttk.Button(self, text=name, command=cmd).grid(sticky=util.NSEW, padx=10, pady=3)

    def end(self):
        self.destroy()
        self.quit()
