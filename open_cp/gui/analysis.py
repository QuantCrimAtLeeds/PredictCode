"""
analysis
~~~~~~~~

The main window, once we have loaded data.
"""

import open_cp.gui.tk.analysis_view as analysis_view
from open_cp.gui.import_file_model import CoordType

class Analysis():
    def __init__(self, model, root):
        self.view = analysis_view.AnalysisView(model, root)
        self.model = model
        pass

    def run(self):
        self.view.wait_window(self.view)


class Model():
    """The model

    :param filename: Name of the file we loaded
    :param data: A triple `(timestamps, xcoords, ycoords, crime_types)`
    :param parse_settings: An instance of :class:`import_file_model.ParseSettings`
    """
    def __init__(self, filename, data, parse_settings):
        self.filename = filename
        self.times = data[0]
        self.xcoords = data[1]
        self.ycoords = data[2]
        self.crime_types = data[3]
        self._parse_settings = parse_settings

    @staticmethod
    def init_from_process_file_model(filename, model):
        new_model = Model(filename, model.data, model.settings)
        new_model.num_empty_rows = len(model.empties)
        new_model.num_error_rows = len(model.errors)
        return new_model

    @property
    def num_rows(self):
        return len(self.times)

    @property
    def num_empty_rows(self):
        return self._empty_rows

    @num_empty_rows.setter
    def num_empty_rows(self, value):
        self._empty_rows = value
    
    @property
    def num_error_rows(self):
        return self._error_rows

    @num_error_rows.setter
    def num_error_rows(self, value):
        self._error_rows = value

    @property
    def coord_type(self):
        return self._parse_settings.coord_type
    