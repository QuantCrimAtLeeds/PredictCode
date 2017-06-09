"""
analysis
~~~~~~~~

The main window, once we have loaded data.
"""

import datetime, itertools
import open_cp.gui.tk.analysis_view as analysis_view
from open_cp.gui.import_file_model import CoordType

class Analysis():
    def __init__(self, model, root):
        self.model = model
        self._root = root
        self.view = analysis_view.AnalysisView(self.model, self, self._root)
        self._init()
        
    def _init(self):
        self.view.refresh_plot()
        self.reset_times()
        self.view.set_crime_type_selections([], 0)
        self.notify_crime_type_selection([], 0)
        self.recalc_total_count()

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
            self.recalc_total_count()

    def run(self):
        self.view.wait_window(self.view)

    def new_input_file(self, filename):
        from . import process_file
        total_rows = process_file.rows_in_csv(filename)
        pf = process_file.ProcessFile(filename, total_rows, self.model._parse_settings, self._root)
        code = pf.run()
        if code is None or not code:
            self.view.destroy()
            return
        
        self.model = Model.init_from_process_file_model(filename, pf.model)
        self.view._model = self.model
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

    def notify_crime_type_selection(self, selection, level):
        if self.model.num_crime_type_levels == 1:
            if level != 0:
                raise ValueError()
            self.model.selected_crime_types = [
                (self.view.crime_type_selection_text(0, x),)
                for x in selection ]
        elif self.model.num_crime_type_levels == 2:
            if level == 0:
                level0 = [ (self.view.crime_type_selection_text(0, x),)
                    for x in self.view.crime_type_selections(0) ]
                level1 = list(self.model.unique_crime_types(level0))
                self.view.set_crime_type_options(1, level1)
        
            sel1, sel2 = self.view.crime_type_selections(0), self.view.crime_type_selections(1)
            self.model.selected_crime_types = [
                (self.view.crime_type_selection_text(0, x1), self.view.crime_type_selection_text(1, x2))
                for x1, x2 in itertools.product(sel1, sel2)]
        else:
            raise ValueError()
        # TODO: These can get slow-- push to thread?
        count = self.model.counts_by_crime_type()
        self.view.update_crime_type_count(count)
        self.recalc_total_count()
        

    def recalc_total_count(self):
        train_count = len(self.model.training_data())
        assess_count = len(self.model.assess_data())
        self.view.update_total_count(train_count, assess_count)


class Model():
    """The model

    :param filename: Name of the file we loaded
    :param data: `(timestamps, xcoords, ycoords, crime_types)`
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
        self._crime_types = None

    @staticmethod
    def init_from_process_file_model(filename, model):
        new_model = Model(filename, model.data, model.settings)
        new_model.num_empty_rows = len(model.empties)
        new_model.num_error_rows = len(model.errors)
        return new_model

    def __iter__(self):
        yield from zip(self.times, self.xcoords, self.ycoords, self.crime_types)

    @property
    def num_crime_type_levels(self):
        """Zero if no crime type field was selected, 1 is one field selected,
        etc."""
        return len(self.crime_types[0])

    def unique_crime_types(self, previous_level_selections):
        """Return a set of crime types.

        :param previous_level_selections: If `None` then return all crime types
          in level 0.  If iterable of lists/tuples of length 1, then return all
          crime types in level 1 which are paired with some crime in the set,
          and so forth.
        """
        if previous_level_selections is None:
            return set(x[0] for x in self.crime_types)
        previous_level_selections = list(previous_level_selections)
        if len(previous_level_selections) == 0:
            return set()
        sel = previous_level_selections[0]
        if isinstance(sel, list) or isinstance(sel, tuple):
            allowed = set(tuple(x) for x in previous_level_selections)
        else:
            allowed = set((x,) for x in previous_level_selections)
        index = len(next(iter(allowed)))
        return set(x[index] for x in self.crime_types
                if tuple(x[:index]) in allowed)

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

    @property
    def selected_crime_types(self):
        """The selected crime types.  If `None` then all.  Otherwise a list/set
        of tuples of selected crime types."""
        return self._crime_types

    @selected_crime_types.setter
    def selected_crime_types(self, value):
        if value is None:
            self._crime_types = None
            return
        new_sel = set()
        length = None
        for x in value:
            if not isinstance(x, tuple) and not isinstance(x, list):
                raise ValueError("Each type should be a list or tuple")
            if length is None:
                length = len(x)
            elif length != len(x):
                raise ValueError("Each type should be the same length")
            new_sel.add(tuple(x))
        self._crime_types = new_sel

    def counts_by_crime_type(self):
        """:return: The number of events with the selected crime type(s)."""
        filter = self.crime_type_filter()
        return sum(filter(ct) for ct in self)

    def crime_type_filter(self):
        """Returns a callable object which when called on an `entry` (as from
        iterating this class) returns True or False.  Is _not_ dynamically
        updated, so if `selected_crime_types` changes then this needs to be
        called again.
        """
        if self.selected_crime_types is None:
            return lambda e : True
        allowed = self.selected_crime_types
        if len(allowed) == 0:
            return lambda e : False
        length = len(next(iter(allowed)))
        def allow(ct):
            return tuple(ct[3][:length]) in allowed
        return allow
        
    def training_data(self):
        """`(times, xcoords, ycoords)` for the selected time range and crime
        types.
        """
        filter = self.crime_type_filter()
        start, end = self.time_range[0], self.time_range[1]
        def in_time_range(e):
            return e[0] >= start and e[0] <= end
        return [e[:3] for e in self if filter(e) and in_time_range(e)]

    def assess_data(self):
        """`(times, xcoords, ycoords)` for the selected time range and crime
        types.
        """
        filter = self.crime_type_filter()
        start, end = self.time_range[2], self.time_range[3]
        def in_time_range(e):
            return e[0] > start and e[0] <= end
        return [e[:3] for e in self if filter(e) and in_time_range(e)]

    def counts_by_time(self):
        """:return: `(train_count, assess_count)`"""
        start, end = self.time_range[0], self.time_range[1]
        train_count = sum(t >= start and t <= end for t in self.times)
        start, end = self.time_range[2], self.time_range[3]
        assess_count = sum(t > start and t <= end for t in self.times)
        return train_count, assess_count
