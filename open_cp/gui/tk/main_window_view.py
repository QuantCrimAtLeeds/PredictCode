"""
main_window_view
~~~~~~~~~~~~~~~~
"""

import tkinter as tk
import tkinter.ttk as ttk
from . import util
import traceback
import tkinter.messagebox
import logging

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
        self.protocol("WM_DELETE_WINDOW", self.end)
        self.title("OpenCP")
        import uuid
        self._uuid = uuid.uuid4()
        
    def resize(self, width_per, height_per):
        util.centre_window_percentage(self, width_per, height_per)

    def end(self):
        self.destroy()
        self.quit()

    def __repr__(self):
        return "TopWindow(uuid={})".format(self._uuid)

    def __str__(self):
        return self.__repr__()

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
        self.columnconfigure(0, weight=1)
        for i in range(2):
            self.rowconfigure(i, weight=1)
        self.createWidgets()
        
    def createWidgets(self):
        self.load_csv_button = ttk.Button(self, text="Load CSV File",
                                          command = self.controller.load_csv)
        self.load_csv_button.grid(sticky=(tk.N, tk.S, tk.E, tk.W), padx=10, pady=10)
        
        self.quit_button = ttk.Button(self, text="Exit", command = self.end)
        self.quit_button.grid(sticky=(tk.N, tk.S, tk.E, tk.W), padx=10, pady=10)

    def end(self):
        self.destroy()
        self.quit()