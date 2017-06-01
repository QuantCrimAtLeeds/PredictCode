"""
main_window
~~~~~~~~~~~

The main menu which is displayed at the start.
"""

from .tk import main_window_view as main_window_view
import tkinter.filedialog
from . import import_file

class MainWindow():
    def __init__(self, root):
        self._user_quit = False
        self._root = root
        root.resize(30, 20)
        self.init()
        
    def run(self):
        self.view.mainloop()
        
    @property
    def user_quit(self):
        """Has the user quit?"""
        return self._user_quit

    def init(self):
        self.view = main_window_view.MainWindowView(self)
        
    def load_csv(self):
        filename = tkinter.filedialog.askopenfilename(defaultextension=".csv",
            filetypes = [("csv", "*.csv")],
            title="Please select a CSV file to open")
        #filename = "../Open data sets/2017-01-cumbria-street.csv"
        if filename == "":
            return
        self.view.destroy()
        import_file.ImportFile(filename).run()
        print("Done with processing file; back to main menu...")
        self.init()