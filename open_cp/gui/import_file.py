"""
import_file
~~~~~~~~~~~

The controller of the "import data" dialog.
"""

from . import import_file_model
from . import process_file
import open_cp.gui.tk.import_file_view as import_file_view
import open_cp.gui.analysis as analysis
from open_cp.gui import locator
import csv
import array
import logging


class ImportFile():
    def __init__(self, root, filename):
        self._filename = filename
        self.parse_settings = import_file_model.ParseSettings()
        self._process_model = None
        self._root = root
        self._logger = logging.getLogger(__name__)
        self.model = None
    
    def run(self):
        self._load_file()
        if self.model is None:
            return
        self.view = import_file_view.ImportFileView(self._root, self.model, self)
        self.view.wait_window(self.view)
        if self._process_model is not None:
            model = analysis.Model.init_from_process_file_model(self._filename,
                    self._process_model)
            analysis.Analysis(model, self._root).run()
        else:
            # Return control to main window...
            pass
        
    def _load_file(self):
        self.view = import_file_view.LoadFileProgress()
        pool = locator.get("pool")
        pool.submit(self._process_file, self._done_process_file)
        self.view.wait_window(self.view)
        
    def _yield_rows(self):
        with open(self._filename, encoding="UTF8", mode="rt") as f:
            reader = csv.reader(f)
            yield from reader
        
    def _process_file(self):
        reader = self._yield_rows()
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
        if isinstance(value, Exception):
            import_file_view.display_error("{} / {}".format(type(value), value))
        else:
            self.model = import_file_model.Model(value)
        self.view.destroy()

    def notify_time_format(self, format_string, initial=False):
        self.parse_settings.timestamp_format = format_string
        if not initial:
            self._try_parse()

    def notify_coord_format(self, coord_format, initial=False):
        self.parse_settings.coord_type = coord_format
        if not initial:
            self._try_parse()

    def notify_crime_field(self, order, field):
        current = self.parse_settings.crime_type_fields
        while len(current) < order + 1:
            current.append(-1)
        current[order] = field
        self.parse_settings.crime_type_fields = current
        self._try_parse()

    def notify_meters_conversion(self, to_meters, initial=False):
        self.parse_settings.meters_conversion = to_meters
        if not initial:
            self._try_parse()

    def notify_datetime_field(self, field_number, initial=False):
        self.parse_settings.timestamp_field = field_number
        if not initial:
            self._try_parse()

    def notify_xcoord_field(self, field_number, initial=False):
        self.parse_settings.xcoord_field = field_number
        if not initial:
            self._try_parse()

    def notify_ycoord_field(self, field_number, initial=False):
        self.parse_settings.ycoord_field = field_number
        if not initial:
            self._try_parse()

    def cancel(self):
        self.view.destroy()
        self.okay = False
        
    def contin(self):
        process = process_file.ProcessFile(self._filename, self.model.rowcount,
                self.parse_settings, parent_view = self.view)
        code = process.run()
        if code is None:
            return
        if code:
            self._process_model = process.model
        self.view.destroy()

    def _try_parse(self):
        error = self.model.try_parse(self.parse_settings)
        if error is None:
            error = ""
        if error == "":
            self.view.allow_continue(True)
        else:
            self.view.allow_continue(False)
        self.view.set_error(error)
        self.view.new_parse_data()
