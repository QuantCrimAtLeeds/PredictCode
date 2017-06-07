"""
main_window
~~~~~~~~~~~

The main menu which is displayed at the start.
"""

from .tk import main_window_view as main_window_view
import tkinter.filedialog
from . import import_file
import logging
import tkinter as _tk

class MainWindow():
    def __init__(self, root):
        self._logger = logging.getLogger(__name__)
        self._logger.debug("tkinter version in use: %s", _tk.Tcl().eval('info patchlevel'))
        self._user_quit = False
        self._root = root
        self.init()
        
    def run(self):
        self.view.mainloop()
        
    @property
    def user_quit(self):
        """Has the user quit?"""
        return self._user_quit

    def init(self):
        self._root.resize(30, 20)
        self.view = main_window_view.MainWindowView(self)
        
    def load_csv(self):
        filename = tkinter.filedialog.askopenfilename(defaultextension=".csv",
            filetypes = [("csv", "*.csv")],
            title="Please select a CSV file to open")
        if filename is None or filename == "" or len(filename) == 0:
            return
        self.view.destroy()
        import_file.ImportFile(filename).run()
        self.init()