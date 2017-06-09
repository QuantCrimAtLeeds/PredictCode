"""
analysis
~~~~~~~~

The main window, once we have loaded data.
"""

import datetime
import open_cp.gui.tk.analysis_view as analysis_view
from open_cp.gui.import_file_model import CoordType

class Analysis():
    def __init__(self, model, root):
        self.model = model
        self.view = analysis_view.AnalysisView(model, self, root)
        
        self.reset_times()

    def _round_dt(self, dt, how="past"):
        if how == "past":
            return datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)
        if how == "future":
            return datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, 59)
        raise ValueError()

    def _update_times(self):
        old_range = self.model.time_range
        self.model.time_range = (self.view.training_start, self.view.training_end,
                self.view.assess_start, self.view.assess_end)
        update = (old_range is None)
        if not update:
            update = (old_range != self.model.time_range)
        if update:
            self.view.update_time_counts(*self.model.counts_by_time())

    def run(self):
        self.view.wait_window(self.view)

    def notify_training_start(self):
        self._update_times()

    def notify_training_end(self):
        self._update_times()

    def notify_assess_start(self):
        self._update_times()

    def notify_assess_end(self):
        self._update_times()

    def reset_times(self):
        start = self._round_dt(min(self.model.times), "past")
        end = self._round_dt(max(self.model.times), "future")        
        mid = self._round_dt(start + (end - start) / 2, "past")
        if mid - start > datetime.timedelta(days=365):
            mid = start + datetime.timedelta(days=365)
        self.view.training_start = start
        self.view.training_end = mid
        self.view.assess_start = mid
        self.view.assess_end = end
        self._update_times()


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
        self._time_range = None

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
    
    @property
    def time_range(self):
        """(training_start, training_end, assessment_start, assessment_end)."""
        return self._time_range

    @time_range.setter
    def time_range(self, value):
        if len(value) != 4:
            raise ValueError()
        self._time_range = tuple(value)

    def counts_by_time(self):
        """:return: `(train_count, assess_count)`"""
        start, end = self.time_range[0], self.time_range[1]
        train_count = sum(t >= start and t <= end for t in self.times)
        start, end = self.time_range[2], self.time_range[3]
        assess_count = sum(t > start and t <= end for t in self.times)
        return train_count, assess_count
