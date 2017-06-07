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


class ImportFile():
    def __init__(self, filename):
        self._filename = filename
        self.datetime_field = None
        self.xcoord_field = None
        self.ycoord_field = None
        self._process_model = None
        pass
    
    def run(self):
        self._load_file()
        self.view = import_file_view.ImportFileView(self.model, self)
        self.view.wait_window(self.view)
        if self._process_model is not None:
            model = analysis.Model.init_from_process_file_model(self._filename,
                self._process_model, self.model.coord_type)
            analysis.Analysis(model, None).run()
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
        self.model = import_file_model.Model(value)
        self.view.destroy()

    def notify_time_format(self, format_string, initial=False):
        self.time_format = format_string
        if not initial:
            self._try_parse()

    def notify_coord_format(self, coord_format, initial=False):
        self.model.coord_type = coord_format
        if not initial:
            self._try_parse()

    def notify_meters_conversion(self, to_meters, initial=False):
        self.model.meters_conversion = to_meters
        if not initial:
            self._try_parse()

    def notify_datetime_field(self, field_number, initial=False):
        self.datetime_field = field_number
        if not initial:
            self._try_parse()

    def notify_xcoord_field(self, field_number, initial=False):
        self.xcoord_field = field_number
        if not initial:
            self._try_parse()

    def notify_ycoord_field(self, field_number, initial=False):
        self.ycoord_field = field_number
        if not initial:
            self._try_parse()

    def cancel(self):
        self.view.destroy()
        self.okay = False
        
    def contin(self):
        processor = import_file_model.Model.load_full_dataset(self.time_format,
                self.datetime_field, self.xcoord_field, self.ycoord_field,
                self.model.coordinate_scaling)
        process = process_file.ProcessFile(self._filename, self.model.rowcount, processor, self.view)
        code = process.run()
        if code is None:
            return
        if code:
            self._process_model = process.model
        self.view.destroy()

    def _try_parse(self):
        error = self.model.try_parse(self.time_format, self.datetime_field,
            self.xcoord_field, self.ycoord_field)
        if error is None:
            self.view.allow_continue(True)
            error = ""
        else:
            self.view.allow_continue(False)
        self.view.set_error(error)
        self.view.new_parse_data()
