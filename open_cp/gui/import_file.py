"""
import_file
~~~~~~~~~~~

The model and controller of the "import data" dialog.
"""

from . import import_file_model
import open_cp.gui.tk.import_file_view as import_file_view
from open_cp.gui import locator
import csv


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
        self.view = import_file_view.ImportFileView(self.initial_data, self)
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
        return import_file_model.InitialData(header, rows, row_count, self._filename)
        
    def _done_process_file(self, value):
        self.initial_data = import_file_model.Model(value)
        self.view.destroy()

    def notify_time_format(self, format_string):
        print("New time format", format_string)
        self._try_parse()

    def notify_coord_format(self, coord_format):
        print("Coord format", coord_format)
        self._try_parse()

    def notify_meters_conversion(self, to_meters):
        print("Meters conversion", to_meters)
        self._try_parse()

    def notify_datetime_field(self, field_number):
        print("Timestamp field is", field_number)
        self._try_parse()

    def notify_xcoord_field(self, field_number):
        print("X coord field is", field_number)
        self._try_parse()

    def notify_ycoord_field(self, field_number):
        print("Y coord field is", field_number)
        self._try_parse()

    def _try_parse(self):
        error = self.initial_data.try_parse(self.view.time_format, self.view.datetime_field,
            self.view.xcoord_field, self.view.ycoord_field)
        if error is None:
            self.view.allow_continue(True)
            error = ""
        else:
            self.view.allow_continue(False)
        self.view.set_error(error)
        self.view.new_parse_data()
