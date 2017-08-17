"""
analysis
~~~~~~~~

The main window, once we have loaded data.
"""

import array
import datetime
import json
import logging
import pickle
import lzma
import numpy as np

import open_cp.gui.funcs as funcs
import open_cp.gui.predictors as predictors
import open_cp.gui.tk.analysis_view as analysis_view
from open_cp.gui.import_file_model import CoordType
import open_cp.gui.run_analysis as run_analysis
import open_cp.gui.run_comparison as run_comparison
import open_cp.gui.browse_analysis as browse_analysis
import open_cp.gui.browse_comparison as browse_comparison
import open_cp.gui.locator as locator
import open_cp.gui.session as session
import open_cp.gui.load_network as load_network
import open_cp.gui.load_network_model as load_network_model

class Analysis():
    def __init__(self, model, root, settings_data=None):
        self.model = model
        self._root = root
        self._logger = logging.getLogger(__name__)
        self._tools = AnalysisToolsController(self.model, self)
        self._comparisons = ComparisonController(self.model, self)
        self.view = analysis_view.AnalysisView(self.model, self, self._root)
        self._tools.view = self.view
        self._comparisons.view = self.view
        self._init()
        self._finalise_loading(settings_data)

    def _finalise_loading(self, data):
        if data is None:
            return
        view = analysis_view.FurtherWait(self._root)
        def task():
            self.model.load_settings_slow(data)
        def done(input=None):
            self.model.reset_settings_dict()
            self._init()
            self.view.update_run_analysis_results()
            view.destroy()
        locator.get("pool").submit(task, done)
        self.view.wait_window(view)

    def _init(self):
        errors = self.model.consume_errors()
        if len(errors) > 0:
            self.view.show_errors(errors)
        self._crime_types_model_to_view()
        self.notify_crime_type_selection(None)
        self._repaint_times()
        self.view.refresh_plot()
        self.recalc_total_count()
        self._tools.update()
        self._comparisons.update()
        self.view.update_network_info()

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

    def load_network(self):
        load_network.LoadNetwork(self._root, self).run()
        self.view.update_network_info()
    
    def with_basemap(self):
        pass

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
            d = self.model.to_dict()
            json_string = json.dumps(d, indent=2)
            with open(filename, "wt") as f:
                f.write(json_string)
            self.model.reset_settings_dict()
            self.model.session_filename = filename
            self.view.update_session_name()
        except Exception as e:
            self._logger.exception("Failed to save")
            self.view.alert(analysis_view._text["fail_save"].format(type(e), e))

    @property
    def tools_controller(self):
        return self._tools

    @property
    def comparison_controller(self):
        return self._comparisons

    def run_analysis(self):
        run_analysis.RunAnalysis(self.view, self).run()

    def new_run_analysis_result(self, result, filename=None):
        self.model.new_analysis_run(result, filename)
        self.view.update_run_analysis_results()

    def update_run_messages(self, pred_msgs=None, comp_msgs=None):
        if pred_msgs is not None:
            self._pred_msgs = pred_msgs
        if comp_msgs is not None:
            self._comp_msgs = comp_msgs
        combine = []
        if hasattr(self, "_pred_msgs"):
            combine.extend(self._pred_msgs)
        if hasattr(self, "_comp_msgs"):
            combine.extend(self._comp_msgs)
        self.view.set_run_messages(combine)

    def view_past_run(self, run):
        result = self.model.analysis_runs[run]
        browse_analysis.BrowseAnalysis(self._root, result, self.model).run()

    def view_all_past_runs(self):
        result = run_analysis.merge_all_results(self.model.analysis_runs)
        browse_analysis.BrowseAnalysis(self._root, result, self.model).run()
        
    def view_comparison(self, analysis_index, comparison_index):
        result = self.model.analysis_run_comparisons(analysis_index)[comparison_index]
        browse_comparison.BrowseComparison(self._root, result).run()

    def view_all_comparisons(self):
        result = self.model.complete_comparison
        browse_comparison.BrowseComparison(self._root, result).run()

    def run_comparison_for(self, run):
        self._last_comprison_run_for = run
        result = self.model.analysis_runs[run]
        run_comparison.RunComparison(self.view, self, result).run()

    def run_comparison_for_all(self):
        self._last_comprison_run_for = -1
        result = run_analysis.merge_all_results(self.model.analysis_runs)
        run_comparison.RunComparison(self.view, self, result).run()

    def new_run_comparison_result(self, result):
        if self._last_comprison_run_for == -1:
            self.model.new_complete_comparison(result)
        else:
            self.model.new_comparison(self._last_comprison_run_for, result)
        self.view.update_run_analysis_results()
        
    def remove_comparison_run(self, run_index, com_index):
        self.model.remove_run_comparison(run_index, com_index)
        self.view.update_run_analysis_results()

    def save_comparison_run(self, run_index, comparison_index, filename):
        if run_index == -1:
            result = self.model.complete_comparison
        else:
            result = self.model.analysis_run_comparisons(run_index)[comparison_index]
        result.save_to_csv(filename)

    def remove_past_run(self, run):
        self.model.remove_analysis_run(run)
        self.view.update_run_analysis_results()

    def save_run(self, run_index, filename):
        view = analysis_view.Saving(self.view)
        def save():
            self.model.save_analysis_run(run_index, filename)
        def done(out=None):
            view.destroy()
            if out is not None and isinstance(out, Exception):
                self.view.alert(analysis_view._text["r_save_fail"].format(type(out), out))
            else:
                self.view.update_run_analysis_results()
        locator.get("pool").submit(save, done)
        view.wait_window(view)
        
    def load_saved_run(self, filename):
        view = analysis_view.Saving(self.view, loading=True)
        def load():
            self.model.load_analysis_run(filename)
        def done(result):
            view.destroy()
            if isinstance(result, Exception):
                self.view.alert(analysis_view._text["r_load_fail"].format(result))
            else:
                self.view.update_run_analysis_results()
        locator.get("pool").submit(load, done)
        view.wait_window(view)


## The model #############################################################

class DataModel():
    """Just stores the actual time/coordinates data, which is what we need to
    pass around to certain tasks.

    :param data: `(timestamps, xcoords, ycoords, crime_types)`
    :param parse_settings: An instance of :class:`import_file_model.ParseSettings`
    """
    def __init__(self, data, parse_settings):
        self._parse_settings = parse_settings
        self.times = np.asarray([np.datetime64(x) for x in data[0]])
        self.xcoords = np.asarray(data[1]).astype(np.float)
        self.ycoords = np.asarray(data[2]).astype(np.float)
        self.crime_types = data[3]
        if len(self.crime_types) > 0 and len(self.crime_types[0]) > 2:
            raise ValueError("Cannot handle more than 2 crime types.")
        self._make_unique_crime_types()
        
    def clone(self):
        """Make a copy; deliberately makes a "slice" of a sub-class, in C++
        speak."""
        data = [self.times, self.xcoords, self.ycoords, []]
        new_model = DataModel(data, self._parse_settings)
        new_model._unique_crime_types = self._unique_crime_types
        new_model.crime_types = self.crime_types
        new_model._selected_crime_types = self._selected_crime_types
        new_model._time_range = self._time_range
        return new_model
        
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

    def selected_by_crime_type_data(self):
        """`(times, xcoords, ycoords)` for the crime types (but with any
        timestamp).
        """
        mask = self._crime_selected_mask()
        return self.times[mask], self.xcoords[mask], self.ycoords[mask]

    def counts_by_time(self):
        """:return: `(train_count, assess_count)`"""
        start, end = self.time_range[0], self.time_range[1]
        start, end = np.datetime64(start), np.datetime64(end)
        train_count = np.sum((self.times >= start) & (self.times <= end))
        start, end = self.time_range[2], self.time_range[3]
        start, end = np.datetime64(start), np.datetime64(end)
        assess_count = np.sum((self.times > start) & (self.times <= end))
        return train_count, assess_count


class Model(DataModel):
    """The model 

    :param filename: Name of the file we loaded
    :param data: `(timestamps, xcoords, ycoords, crime_types)`
    :param parse_settings: An instance of :class:`import_file_model.ParseSettings`
    """
    def __init__(self, filename, data, parse_settings):
        super().__init__(data, parse_settings)
        self._errors = []
        self.filename = filename
        self.analysis_tools_model = AnalysisToolsModel(self)
        self.comparison_model = ComparisonModel(self)
        self._time_range = None
        self._selected_crime_types = set()
        self._meta_comparison = None
        self._logger = logging.getLogger(__name__)
        self.reset_times()
        self._analysis_runs = []
        self._loaded_from_dict = None
        self.session_filename = None
        self._network_model = load_network_model.NetworkModel(self)
        
    class AnalysisRunHolder():
        def __init__(self, result, filename=None):
            self._result = result
            self._filename = filename
            self._coms = []
            
        @property
        def result(self):
            return self._result
        
        @property
        def filename(self):
            return self._filename
        
        @filename.setter
        def filename(self, value):
            self._filename = value
            
        def add_comparison(self, result):
            self._coms.append(result)

        @property
        def comparisons(self):
            return list(self._coms)
        
        def remove_comparison(self, index):
            del self._coms[index]

    @property
    def analysis_runs(self):
        """List of results of previous runs."""
        return [x.result for x in self._analysis_runs]

    def new_analysis_run(self, result, filename=None):
        """Add a new analysis run with optional saved file name."""
        self._analysis_runs.append( self.AnalysisRunHolder(result, filename) )
        self._meta_comparison = None

    def new_comparison(self, analysis_run_index, result):
        self._analysis_runs[analysis_run_index].add_comparison(result)

    def analysis_run_filename(self, run_index):
        """`None` indicates not saved."""
        return self._analysis_runs[run_index].filename
    
    def analysis_run_comparisons(self, run_index):
        """List of comparison runs for this run"""
        return self._analysis_runs[run_index].comparisons

    def remove_run_comparison(self, run_index, comparison_index):
        self._analysis_runs[run_index].remove_comparison(comparison_index)

    def remove_analysis_run(self, run_index):
        del self._analysis_runs[run_index]
        self._meta_comparison = None

    def new_complete_comparison(self, result):
        self._meta_comparison = result

    @property
    def complete_comparison(self):
        return self._meta_comparison

    def save_analysis_run(self, run_index, filename):
        with lzma.open(filename, "wb") as file:
            pickle.dump(self._analysis_runs[run_index].result, file)
        self._analysis_runs[run_index].filename = filename

    def load_analysis_run(self, filename):
        self._logger.debug("Attempting to load past analysis run: '%s'", filename)
        with lzma.open(filename, "rb") as file:
            result = pickle.load(file)
        self.new_analysis_run(result, filename)

    @staticmethod
    def init_from_process_file_model(filename, model):
        new_model = Model(filename, model.data, model.settings)
        new_model.num_empty_rows = len(model.empties)
        new_model.num_error_rows = len(model.errors)
        return new_model

    def to_dict(self):
        """Convert all settings to a dictionary."""
        data = {"filename" : self.filename,
                "parse_settings" : self._parse_settings.to_dict(),
                "training_time_range" : [funcs.datetime_to_string(self.time_range[0]),
                        funcs.datetime_to_string(self.time_range[1])],
                "assessment_time_range" : [funcs.datetime_to_string(self.time_range[2]),
                        funcs.datetime_to_string(self.time_range[3])],
                "analysis_tools" : self.analysis_tools_model.to_dict(),
                "comparison_tools" : self.comparison_model.to_dict(),
            }
        data["selected_crime_types"] = [ self.unique_crime_types[index] for index in self.selected_crime_types]
        data["saved_analysis_runs"] = [ res.filename for res in self._analysis_runs if res.filename is not None ]
        data["network_model"] = self.network_model.to_dict()
        return data

    @property
    def session_filename(self):
        """Filename of the session, or `None` to indicate has never been saved."""
        return self._session_filename

    @session_filename.setter
    def session_filename(self, value):
        self._session_filename = value
        if value is not None and value != "":
            session.Session().new_session(value)

    def session_changed(self):
        """Has the session changed since it was loaded?"""
        return self.to_dict() == self._loaded_from_dict

    def reset_settings_dict(self):
        """Set the stored settings to be equal to the current settings."""
        self._loaded_from_dict = self.to_dict()

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

    def load_settings_slow(self, data):
        """Load further settings from a dictionary; should be run off-thread...
        """
        self._load_analysis_tools_from_dict(data)
        self._load_comparison_tools_from_dict(data)
        self._load_saved_runs_from_dict(data)
        self._load_network_model(data)
        
    def _load_network_model(self, data):
        if "network_model" in data:
            try:
                self._network_model.from_dict(data["network_model"])
            except ValueError as ex:
                self._errors.append(str(ex))
        else:
            self._logger.warn("Didn't find key 'network_model': Is this an old input file?")

    def _load_analysis_tools_from_dict(self, data):
        if "analysis_tools" in data:
            try:
                self.analysis_tools_model.settings_from_dict(data["analysis_tools"])
            except ValueError as ex:
                self._errors.append(str(ex))
        else:
            self._logger.warn("Didn't find key 'analysis_tools': Is this an old input file?")

    def _load_comparison_tools_from_dict(self, data):
        if "comparison_tools" in data:
            try:
                self.comparison_model.settings_from_dict(data["comparison_tools"])
            except ValueError as ex:
                self._errors.append(str(ex))
        else:
            self._logger.warn("Didn't find key 'comparison_tools': Is this an old input file?")

    def _load_saved_runs_from_dict(self, data):
        if "saved_analysis_runs" in data:
            for filename in data["saved_analysis_runs"]:
                try:
                    self.load_analysis_run(filename)
                except Exception as ex:
                    self._errors.append(str(ex))
        else:
            self._logger.warn("Didn't find key 'saved_analysis_runs': Is this an old input file?")

    def consume_errors(self):
        """Returns a list of error messages and resets the list to be empty."""
        errors = self._errors
        self._errors = []
        return errors

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
    def network_model(self):
        return self._network_model


## Bases classes for the predictors / comparitors ########################


class _ListController():
    """Base class for a "list" of objects.
    
    :param controller: The main controller, :class:`Analysis`
    :param basemodel: Instance of :class:`Model`
    :param model: The sub-model, sub-class of :class:`_ListModel`
    :param pick_model: Model for "picking" a new object
    """
    def __init__(self, controller, basemodel, model, pick_model):
        self.controller = controller
        self.main_model = basemodel
        self.model = model
        self.view = None
        self._pick_model = pick_model

    def update(self):
        raise NotImplementedError()

    def add(self):
        pred_class = Pick(self.view, self._pick_model).run()
        if pred_class is None:
            return
        try:
            obj = pred_class(self.main_model)
            if self._edit(obj):
                self.model.add(obj)
        except ValueError as ex:
            self.view.alert(str(ex))
        self.update()

    def remove(self, index):
        self.model.remove(index)
        self.update()

    def _edit(self, pred):
        resize = None
        if "resize" in pred.config():
            if pred.config()["resize"]:
                resize = "wh"
        view = analysis_view.PredictionEditView(self.view, pred.describe(), resize)
        edit_view = pred.make_view(view)
        data = pred.to_dict()
        view.run(edit_view)
        result = view.result
        if not result:
            pred.from_dict(data)
        self.update()
        return result

    def edit(self, index):
        pred = self.model.objects[index]
        self._edit(pred)


class _ListModel():
    """Model for :class:`_ListController`.  Assumed that each object has an
    interface similar to :class:`Predictor`.

    :param model: The main model, so we can access coord/times data.
    """
    def __init__(self, model):
        self._objects = []
        self._model = model
        self._logger = logging.getLogger(__name__)
    
    def to_dict(self):
        """This base class version just serialises the :attr:`objects` as
        a list of dictionaries."""
        objs = [ {"name" : p.describe(), "settings" : p.to_dict()}
            for p in self.objects ]
        return objs

    @property
    def objects(self):
        """An ordered list of predictors."""
        return self._objects

    @objects.setter
    def objects(self, value):
        v = list(value)
        v.sort(key = lambda p : p.order())
        self._objects = v

    def add(self, obj):
        if isinstance(obj, type):
            obj = obj(self._model)
        v = list(self.objects)
        v.append(obj)
        self.objects = v

    def remove(self, index):
        v = list(self.objects)
        del v[index]
        self.objects = v


## Analysis Tools (i.e. "predictors") ####################################


class AnalysisToolsController(_ListController):
    """Partner of :class:`AnalysisToolsModel`.
    
    :param model: Instance of :class:`Model`
    """
    def __init__(self, model, controller):
        super().__init__(controller, model, model.analysis_tools_model,
            PickPredictionModel())

    def update(self):
        self.view.update_predictors_list()
        self.make_msgs()

    def make_msgs(self):
        out = []
        if self.main_model.coord_type == CoordType.LonLat:
            projs = self.model.predictors_of_type(predictors.predictor._TYPE_COORD_PROJ)
            if len(projs) == 0:
                out.append(analysis_view._text["noproj"])
        # TODO: Support continuous predictions
        grids = self.model.predictors_of_type(predictors.predictor._TYPE_GRID)
        if len(grids) == 0:
            out.append(analysis_view._text["nogrid"])
        preds = self.model.predictors_of_type(predictors.predictor._TYPE_GRID_PREDICTOR)
        if len(preds) == 0:
            out.append(analysis_view._text["nopreds"])
        self.controller.update_run_messages(pred_msgs=out)


class AnalysisToolsModel(_ListModel):
    """Model for the prediction and analysis settings.
    Separated out just to make the classes easier to read.
    
    :param model: The main model, so we can access coord/times data.
    """
    def __init__(self, model):
        super().__init__(model)

    def to_dict(self):
        """Write settings to dictionary."""
        return { "predictors" : super().to_dict() }

    def settings_from_dict(self, data):
        """Import settings from a dictionary."""
        v, errors = [], []
        for pred_data in data["predictors"]:
            name = pred_data["name"]
            pred = [p for p in predictors.all_predictors if p.describe() == name]
            if len(pred) == 0:
                errors.append(analysis_view._text["pi_fail1"].format(name))
                continue
            if len(pred) > 1:
                errors.append(analysis_view._text["pi_fail2"].format(name))
                continue
            try:
                self._logger.debug("Loading settings for analysis type %s", name)
                pred = pred[0](self._model)
                pred.from_dict(pred_data["settings"])
                v.append(pred)
            except Exception as ex:
                errors.append(analysis_view._text["pi_fail3"].format(name, type(ex), ex))
        self.objects = v
        if len(errors) > 0:
            raise ValueError("\n".join(errors))

    def predictors_of_type(self, order):
        """Get all predictors of the given order/type."""
        return [p for p in self.objects if p.order() == order]

    def coordinate_projector(self):
        """Obtain the first chosen coordinate projection stage, if there is
        one, or `None` otherwise."""
        preds = self.predictors_of_type(predictors.predictor._TYPE_COORD_PROJ)
        if len(preds) == 0:
            return None
        return preds[0]

    def projected_coords(self):
        """Obtain, if possible using the current settings, the entire data-set
        of projected coordinates.  Returns `None` otherwise."""
        if self._model.coord_type == CoordType.XY:
            return self._model.xcoords, self._model.ycoords
        pred = self.coordinate_projector()
        if pred is None:
            return None
        task = pred.make_tasks()[0]
        return task(self._model.xcoords, self._model.ycoords)


class PickPredictionModel():
    def __init__(self):
        self._predictors = [
                (clazz.order(), clazz.describe(), clazz)
                for clazz in predictors.all_predictors
            ]
        self._predictors.sort()

    def names(self):
        return [pair[1] for pair in self._predictors]

    def orders(self):
        return [pair[0] for pair in self._predictors]

    def classes(self):
        return [pair[2] for pair in self._predictors]


class Pick():
    def __init__(self, parent, model):
        self._model = model
        self._root = parent

    def run(self):
        self._selected = None
        self._view = analysis_view.PickPredictionView(self._root, self._model, self)
        self._view.wait_window(self._view)
        
        if self._selected is None:
            return None
        return self._model.classes()[self._selected]

    def selected(self, index):
        self._selected = index
        self._view.cancel()


## Comparison tools ######################################################


class ComparisonController(_ListController):
    """Partner of :class:`ComparisonModel`.
    
    :param model: Instance of :class:`Model`
    """
    def __init__(self, model, controller):
        super().__init__(controller, model, model.comparison_model,
            PickComparitorModel())

    def update(self):
        self.view.update_comparitors_list()
        self.make_msgs()

    def make_msgs(self):
        out = []
        strategy = self.model.comparators_of_type(predictors.comparitor.TYPE_TOP_LEVEL)
        if len(strategy) == 0:
            out.append(analysis_view._text["nostrat"])
        self.controller.update_run_messages(comp_msgs=out)


class ComparisonModel(_ListModel):
    """Model for the prediction and analysis settings.
    Separated out just to make the classes easier to read.
    
    :param model: The main model, so we can access coord/times data.
    """
    def __init__(self, model):
        super().__init__(model)

    def to_dict(self):
        return {"comparitors" : super().to_dict()}
        
    def settings_from_dict(self, data):
        """Import settings from a dictionary."""
        v, errors = [], []
        for pred_data in data["comparitors"]:
            name = pred_data["name"]
            pred = [p for p in predictors.all_comparitors if p.describe() == name]
            if len(pred) == 0:
                errors.append(analysis_view._text["ci_fail1"].format(name))
                continue
            if len(pred) > 1:
                errors.append(analysis_view._text["ci_fail2"].format(name))
                continue
            try:
                self._logger.debug("Loading settings for comparitor type %s", name)
                pred = pred[0](self._model)
                pred.from_dict(pred_data["settings"])
                v.append(pred)
            except Exception as ex:
                errors.append(analysis_view._text["ci_fail3"].format(name, type(ex), ex))
        self.objects = v
        if len(errors) > 0:
            raise ValueError("\n".join(errors))

    def comparators_of_type(self, order):
        """Get all predictors of the given order/type."""
        return [p for p in self.objects if p.order() == order]


class PickComparitorModel():
    def __init__(self):
        self._comparitors = [
                (clazz.order(), clazz.describe(), clazz)
                for clazz in predictors.all_comparitors
            ]
        self._comparitors.sort()

    def names(self):
        return [pair[1] for pair in self._comparitors]

    def orders(self):
        return [pair[0] for pair in self._comparitors]

    def classes(self):
        return [pair[2] for pair in self._comparitors]
