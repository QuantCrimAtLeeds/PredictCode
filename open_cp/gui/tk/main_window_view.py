"""
main_window_view
~~~~~~~~~~~~~~~~
"""

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox
from . import util
import traceback
import logging
import sys

_text = {
    "name" : "OpenCP",
    "load" : "Load CSV File",
    "exit" : "Exit",
    "ss" : "Load a saved session",
    "about" : "About",
    "recent" : "Reload a recent session",
    "config" : "Configuration",

}

class TopWindow(tk.Tk):
    """A single top-level window.  Should be a singleton, but we'll enforce
    this via creating once and then injecting into clients.  Having a constant
    top-level window makes life a lot easier with `tkinter`.
    """
    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self.grid()
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.title(_text["name"])
        self._window_icon()
        self._theme = ttk.Style()
        
    def resize(self, width_per, height_per):
        util.centre_window_percentage(self, width_per, height_per)

    def end(self):
        self.destroy()
        self.quit()

    @property
    def style(self):
        """The TTK `style`"""
        return self._theme

    def _window_icon(self):
        try:
            import open_cp.gui.resources
            import PIL.ImageTk
            bitmap = PIL.ImageTk.PhotoImage(open_cp.gui.resources.app_icon)
            self.tk.call("wm", "iconphoto", self._w, bitmap)
        except:
            self._logger.exception()

    def report_callback_exception(self, *args):
        """Make the application exit with a log and dump of stack trace."""
        err = traceback.format_exception(*args)
        self._logger.error("Exception: %s", err)
        print("".join(err))
        import os
        os._exit(1)


class MainWindowView(tk.Frame):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.grid(sticky=util.NSEW)
        util.stretchy_columns(self, range(3))
        util.stretchy_rows(self, range(4))
        self.createWidgets()
        self.master.protocol("WM_DELETE_WINDOW", self.end)
        util.centre_window_percentage(self.master, 30, 30)
        
    def createWidgets(self):
        b = ttk.Button(self, text=_text["load"], command=self.controller.load_csv)
        b.grid(sticky=util.NSEW, padx=10, pady=3, row=0, column=0, columnspan=3)
        b = ttk.Button(self, text=_text["ss"], command=self.controller.load_session)
        b.grid(sticky=util.NSEW, padx=10, pady=3, row=1, column=0, columnspan=2)
        frame = ttk.Frame(self)
        frame.grid(sticky=util.NSEW, padx=10, pady=3, row=1, column=2)
        util.stretchy_rows_cols(frame, [0,1], [0])
        b = ttk.Button(frame, text=_text["recent"], command=self.controller.recent)
        b.grid(row=0, column=0, sticky=tk.NSEW, pady=1)
        b = ttk.Button(frame, text=_text["config"], command=self.controller.config)
        b.grid(row=1, column=0, sticky=tk.NSEW, pady=1)
        b = ttk.Button(self, text=_text["about"], command=self.controller.about)
        b.grid(sticky=util.NSEW, padx=10, pady=3, row=2, column=0, columnspan=3)
        b = ttk.Button(self, text=_text["exit"], command=self.end)
        b.grid(sticky=util.NSEW, padx=10, pady=3, row=3, column=0, columnspan=3)

    def end(self):
        self.destroy()
        self.quit()

    def alert(self, message):
        tkinter.messagebox.showerror("Error", message)
