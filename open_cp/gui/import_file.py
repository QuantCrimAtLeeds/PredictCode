"""
import_file
~~~~~~~~~~~

The model and controller of the "import data" dialog.
"""

import open_cp.gui.tk.import_file_view as import_file_view
from open_cp.gui import locator

import csv
import collections

InitialData = collections.namedtuple("InitialData", ["header", "firstrows", "rowcount", "filename"])

class Data():
    def __init__(self):
        pass
    


class ImportFile():
    def __init__(self, filename):
        self._filename = filename
        pass
    
    @property
    def data(self):
        """Get the data after the load and process sequence has completed."""
        pass
    
    def run(self):
        self._load_file()
        self.view = import_file_view.ImportFileView(self.initial_data)
        self.view.wait_window(self.view)
        
    def _load_file(self):
        self.view = import_file_view.LoadFileProgress()
        pool = locator.get("pool")
        pool.submit(self._process_file, self._done_process_file)
        self.view.wait_window(self.view)
        
    def _process_file(self):
        with open(self._filename, encoding="UTF8", mode="rt") as f:
            reader = csv.reader(f)
            header = next(reader)
            row_count = 0
            rows = []
            for i, row in zip(range(5), reader):
                rows.append(row)
                row_count += 1
            for row in reader:
                row_count += 1
        return InitialData(header, rows, row_count, self._filename)
        
    def _done_process_file(self, value):
        self.initial_data = value
        self.view.destroy()
