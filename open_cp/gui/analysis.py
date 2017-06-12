"""
analysis
~~~~~~~~

The main window, once we have loaded data.
"""

import datetime, json, logging, array
import numpy as np
import open_cp.gui.tk.analysis_view as analysis_view
import open_cp.gui.funcs as funcs

class Analysis():
    def __init__(self, model, root):
        self.model = model
        self._root = root
        self._logger = logging.getLogger(__name__)
        self.view = analysis_view.AnalysisView(self.model, self, self._root)
        self._init()

    def _init(self):
        errors = self.model.consume_errors()
        if len(errors) > 0:
            self.view.show_errors(errors)
        self._crime_types_model_to_view()
        self.notify_crime_type_selection(None)
        self._repaint_times()
        self.view.refresh_plot()
        self.recalc_total_count()

    def _crime_types_model_to_view(self):
        if self.model.num_crime_type_levels == 0:
            return
        self.view.crime_type_selections = self.model.selected_crime_types

    def _update_times(self, force=False):
        """Update the model to the view, and if necessary or `force`d redraw."""
        old_range = self.model.time_range
        self.model.time_range = (self.view.training_start, self.view.training_end,
                self.view.assess_start, self.view.assess_end)
        update = (old_range is None)
        if not update:
            update = (old_range != self.model.time_range)
        if update or force:
            self.view.update_time_counts(*self.model.counts_by_time())
            self.recalc_total_count()

    def _repaint_times(self):
        """Sync view to model and redraw."""
        self.view.training_start = self.model.time_range[0]
        self.view.training_end = self.model.time_range[1]
        self.view.assess_start = self.model.time_range[2]
        self.view.assess_end = self.model.time_range[3]
        self._update_times(True)

    def run(self):
        self.view.wait_window(self.view)

    def new_input_file(self, filename):
        from . import process_file
        total_rows = process_file.rows_in_csv(filename)
        pf = process_file.ProcessFile(filename, total_rows, self.model._parse_settings, self._root, "new_data")
        code = pf.run()
        if code is None or not code:
            self.view.destroy()
            return
        
        self.model = Model.init_from_process_file_model(filename, pf.model)
        self.view.new_model(self.model)
        self._init()

    def notify_training_start(self):
        self._update_times()

    def notify_training_end(self):
        self._update_times()

    def notify_assess_start(self):
        self._update_times()

    def notify_assess_end(self):
        self._update_times()

    def reset_times(self):
        self.model.reset_times()
        self._repaint_times()

    def notify_crime_type_selection(self, selection):
        """Called by view when the crime type selection changes.
        Updates the model with the selections and recalcultes totals (and
        displays them).

        :param selection: The new selection.  If `None` then fetch from view.        
        """
        if self.model.num_crime_type_levels == 0:
            return
        if selection is None:
            selection = self.view.crime_type_selections
        self.model.selected_crime_types = selection
        count = self.model.counts_by_crime_type()
        self.view.update_crime_type_count(count)
        self.recalc_total_count()
        
    def recalc_total_count(self):
        train_count = len(self.model.training_data()[0])
        assess_count = len(self.model.assess_data()[0])
        self.view.update_total_count(train_count, assess_count)

    def save(self, filename):
        try:
            with open(filename, "wt") as f:
                json.dump(self.model.to_dict(), f, indent=2)
        except Exception as e:
            self._logger.exception("Failed to save")
            self.view.alert(analysis_view._text["fail_save"].format(type(e), e))


class Model():
    """The model

    :param filename: Name of the file we loaded
    :param data: `(timestamps, xcoords, ycoords, crime_types)`
    :param parse_settings: An instance of :class:`import_file_model.ParseSettings`
    """
    def __init__(self, filename, data, parse_settings):
        self._errors = []
        self.filename = filename
        self.times = np.asarray([np.datetime64(x) for x in data[0]])
        self.xcoords = np.asarray(data[1]).astype(np.float)
        self.ycoords = np.asarray(data[2]).astype(np.float)
        self.crime_types = data[3]
        if len(self.crime_types) > 0 and len(self.crime_types[0]) > 2:
            raise ValueError("Cannot handle more than 2 crime types.")
        self._make_unique_crime_types()
        self._parse_settings = parse_settings
        self._time_range = None
        self._selected_crime_types = set()
        self._logger = logging.getLogger(__name__)
        self.reset_times()

    @staticmethod
    def init_from_process_file_model(filename, model):
        new_model = Model(filename, model.data, model.settings)
        new_model.num_empty_rows = len(model.empties)
        new_model.num_error_rows = len(model.errors)
        return new_model

    def to_dict(self):
        """Convert all settings to a dictionary."""
        return {"filename" : self.filename,
                "parse_settings" : self._parse_settings.to_dict(),
                "training_time_range" : [funcs.datetime_to_string(self.time_range[0]),
                        funcs.datetime_to_string(self.time_range[1])],
                "assessment_time_range" : [funcs.datetime_to_string(self.time_range[2]),
                        funcs.datetime_to_string(self.time_range[3])],
                "selected_crime_types" : list(self._crime_types)
            }

    def settings_from_dict(self, data):
        """Over-write the current settings with the passed dictionary,
        excepting the filename and parse settings.
        
        Checks that the selected crime types are consistent with the current
        data and adds problems to the errors list.
        """
        t = data["training_time_range"]
        a = data["assessment_time_range"]
        self.time_range = [funcs.string_to_datetime(t[0]), funcs.string_to_datetime(t[1]),
            funcs.string_to_datetime(a[0]), funcs.string_to_datetime(a[1])]
        
        sel_ctypes = []
        for ctype in data["selected_crime_types"]:
            ctype = tuple(ctype)
            le = len(ctype)
            if le != self.num_crime_type_levels:
                self._errors.append(analysis_view._text["ctfail1"].format(ctype))
                continue
            try:
                index = self.unique_crime_types.index(ctype)
                sel_ctypes.append(index)
            except:
                self._errors.append(analysis_view._text["ctfail2"].format(ctype))
        if len(self._errors) > 0:
            self._logger.warn("Errors in loading saved settings: %s", self._errors)
        self.selected_crime_types = sel_ctypes

    def consume_errors(self):
        """Returns a list of error messages and resets the list to be empty."""
        errors = self._errors
        self._errors = []
        return errors

    @property
    def num_crime_type_levels(self):
        """Zero if no crime type field was selected, 1 if one field selected,
        etc."""
        if len(self.unique_crime_types) == 0:
            return 0
        return len(self.unique_crime_types[0])

    def _make_unique_crime_types(self):
        data = [tuple(x) for x in self.crime_types]
        self._unique_crime_types = list(set(data))
        if len(self._unique_crime_types) > 1000:
            self._errors.append(analysis_view._text["ctfail3"].format(len(self._unique_crime_types)))
            self.crime_types = [[] for _ in self.crime_types]
            self._make_unique_crime_types()
        else:
            self._unique_crime_types.sort()
            lookup = dict()
            self.crime_types = array.array("l")
            for x in data:
                if x not in lookup:
                    lookup[x] = self._unique_crime_types.index(x)
                self.crime_types.append(lookup[x])
        self.crime_types = np.asarray(self.crime_types).astype(np.int)

    @property
    def unique_crime_types(self):
        """Return a list of crime types.  This ordering should be used by the
        view to maintain coherence with the model.  Each crime type will be a
        tuple, the length of which agrees with :attr:`num_crime_type_levels`.
        """
        return self._unique_crime_types

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

    def _datetime64_to_datetime(self, dt):
        dt = np.datetime64(dt)
        ts = (dt - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        return datetime.datetime.utcfromtimestamp(ts)

    def _round_dt(self, dt, how="past"):
        dt = self._datetime64_to_datetime(dt)
        if how == "past":
            return datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)
        if how == "future":
            return datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, 59)
        raise ValueError()

    def reset_times(self):
        """Set the default time ranges."""
        start = self._round_dt(np.min(self.times), "past")
        end = self._round_dt(np.max(self.times), "future")        
        mid = self._round_dt(start + (end - start) / 2, "past")
        if mid - start > datetime.timedelta(days=365):
            mid = start + datetime.timedelta(days=365)
        self.time_range = (start, mid, mid, end)

    @property
    def time_range_of_data(self):
        """`(start, end)` times for the whole dataset."""
        return (self._datetime64_to_datetime(np.min(self.times)),
                self._datetime64_to_datetime(np.max(self.times)))

    @property
    def selected_crime_types(self):
        """The selected crime types.  A set of indexes into
        :attr:`unique_crime_types`."""
        return self._selected_crime_types

    @selected_crime_types.setter
    def selected_crime_types(self, value):
        if value is None or len(value) == 0:
            self._selected_crime_types = []
            return
        value = set(value)
        limit = len(self.unique_crime_types)
        for x in value:
            if x < 0 or x >= limit:
                raise ValueError()
        self._selected_crime_types = value

    def _crime_selected_mask(self):
        if self.num_crime_type_levels == 0:
            return (np.zeros_like(self.crime_types) + 1).astype(np.bool)
        allowed = list(self.selected_crime_types)
        if len(allowed) == 0:
            return np.zeros_like(self.crime_types).astype(np.bool)
        # Oddly, faster than doing massive single numpy operation
        mask = (self.crime_types == allowed[0])
        for x in allowed[1:]:
            mask |= (self.crime_types == x)
        return mask

    def counts_by_crime_type(self):
        """:return: The number of events with the selected crime type(s)."""
        return np.sum(self._crime_selected_mask())

    def training_data(self):
        """`(times, xcoords, ycoords)` for the selected time range and crime
        types.
        """
        start, end = self.time_range[0], self.time_range[1]
        start, end = np.datetime64(start), np.datetime64(end)
        mask = (self.times >= start) & (self.times <= end)
        mask &= self._crime_selected_mask()
        return self.times[mask], self.xcoords[mask], self.ycoords[mask]

    def assess_data(self):
        """`(times, xcoords, ycoords)` for the selected time range and crime
        types.
        """
        start, end = self.time_range[2], self.time_range[3]
        start, end = np.datetime64(start), np.datetime64(end)
        mask = (self.times > start) & (self.times <= end)
        mask &= self._crime_selected_mask()
        return self.times[mask], self.xcoords[mask], self.ycoords[mask]

    def counts_by_time(self):
        """:return: `(train_count, assess_count)`"""
        # TODO
        start, end = self.time_range[0], self.time_range[1]
        start, end = np.datetime64(start), np.datetime64(end)
        train_count = np.sum((self.times >= start) & (self.times <= end))
        start, end = self.time_range[2], self.time_range[3]
        start, end = np.datetime64(start), np.datetime64(end)
        assess_count = np.sum((self.times > start) & (self.times <= end))
        return train_count, assess_count
