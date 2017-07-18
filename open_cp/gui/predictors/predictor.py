"""
predictor
~~~~~~~~~

Base classes and utility methods for the GUI side of describing algorithms
which produce predictions.
"""

import collections as _collections
import open_cp.data

PREDICTOR_LOGGER_NAME = "__interactive_warning_logger__"

class Task():
    """Abstract base class for a computational task which needs to be carried
    out to make a prediction.

    May either be run in this process (e.g. generating a grid) or run off
    process to enable parallelisation.
    
    We should sub-class for specific types of task.
    
    :param allow_off_process: If true, then run in a separate process in
      parallel.  Default is False, with is optimal for quick tasks.
    """
    def __init__(self, allow_off_process=False):
        self._off_process = allow_off_process

    @property
    def off_process(self):
        """Should we run as a separate process?"""
        if not hasattr(self, "_off_process"):
            print("WTF?  {} / {}".format(self, type(self)))
        return self._off_process


_TYPE_COORD_PROJ = 0

class ProjectTask(Task):
    def __call__(self, xcoords, ycoords):
        """Project the coordinates, which may be one dimensional arrays, or
        scalars."""
        raise NotImplementedError()


_TYPE_GRID = 10

class GridTask(Task):
    def __call__(self, timed_points):
        """Return a :class:`open_cp.data.BoundedGrid` instance giving the grid
        for the passed dataset.  Predictors are free to estimate risk outside
        of this grid, but the grid should give an indication of the extent of
        the data."""
        raise NotImplementedError()
        

_TYPE_GRID_PREDICTOR = 100

class GridPredictorTask(Task):
    """Current does not support running off-process, due to pickling issues."""
    def __call__(self, analysis_model, grid_task, project_task):
        """For the given instance of :class:`analysis.Model` generate one or
        more instances of :class:`SingleGridPredictor` making actual
        predictions."""
        raise NotImplementedError()

    def projected_data(self, analysis_model, project_task):
        """Use the projector to return all data from the model which matches
        the selected crime types, projected in a suitable way.
        
        :return: Instance of :class:`open_cp.data.TimedPoints`.
        """
        times, xcoords, ycoords = analysis_model.selected_by_crime_type_data()
        xcoords, ycoords = project_task(xcoords, ycoords)
        return open_cp.data.TimedPoints.from_coords(times, xcoords, ycoords)
        
        
class SingleGridPredictor(Task):
    def __init__(self, off_thread=True):
        super().__init__(off_thread)

    def __call__(self, predict_time, length):
        """Perform a prediction for the given time and window of time (which
        may be ignored).

        :param predict_time: Instance of :class:`datetime.datetime`
        :param length: Instance of :class:`datetime.timedelta`

        :return: Instance of :class:`GridPrediction` (or most likely a
          subclass)
        """
        raise NotImplementedError()

#_TYPE_CTS_PREDICTOR = 200



class PredictionError(Exception):
    """For signally "expected" problems with running a prediction."""
    def __init__(self, message):
        super().__init__(message)
        

class Predictor():
    """Abstract base class which all prediction methods derive from.
    The actual mechanics of making a prediction should be in the main `open_cp`
    package.  Here we are concerned with GUI issues.

    A predictor may depend upon other predictors (e.g. the user need only
    provide a single algorithm to generate a grid, and all grid-based
    predictors can use it.  The :attr:`name` is used to find suitable partners.

    Sub-classes should not change the constructor.

    :param model: An instance of :class:`analysis.Model` from which we can
      obtain data.  We should directly access `times`, `xcoords`, `ycoords`
      but not worry about which events are actually in the time range etc.
    """
    def __init__(self, model):
        self._times = model.times
        self._xcoords = model.xcoords
        self._ycoords = model.ycoords
        self._model = model

    @staticmethod
    def describe():
        """Return human readable short description of this predictor."""
        raise NotImplementedError()

    @staticmethod
    def order():
        """An ordinal specifying the order, lowest is "first".  E.g. the generator
        of a grid would be before an actual predictor."""
        raise NotImplementedError()

    def make_view(self, parent, inline=False):
        """Construct and return a view object.  This object is the model, and
        the controller may either be another object constructed here, or the
        model.
        
        :param parent: The parent `tk` object (typically another view)
        :param inline: If True, then if applicable, produce a more minimal view.
        """
        raise NotImplementedError()

    def config(self):
        """Optionally return a non-empty dictionary to specify extra options.

        - {"resize": True}  allow the edit view window to be resized.
        """
        return dict()

    @property
    def name(self):
        """Human readable giving the prediction method and perhaps headline
        settings."""
        raise NotImplementedError()

    @property
    def settings_string(self):
        """Human readable giving further settings.  May be `None`."""
        raise NotImplementedError()

    def make_tasks(self):
        raise NotImplementedError()

    def to_dict(self):
        """Write state out to a dictionary for serialisation."""
        raise NotImplementedError()

    def from_dict(self, data):
        """Restore state from a dictionary."""
        raise NotImplementedError()

    def pprint(self):
        settings = self.settings_string
        if settings is not None:
            return self.name + " : " + settings
        return self.name

    _Coords = _collections.namedtuple("Coords", "xcoords ycoords")
    def _as_coords(self, xcoords=None, ycoords=None):
        """Return the coordinates of the data points in an object which has
        attributes `xcoords` and `ycoords`."""
        if xcoords is None:
            xcoords, ycoords = self._projected_coords()
        return self._Coords(xcoords, ycoords)

    def _projected_coords(self):
        """Returns a pair `(xcs, ycs)` of the coordinates of the whole data
        set, projected appropriately."""
        return self._model.analysis_tools_model.projected_coords()


def test_model():
    import collections, datetime
    Model = collections.namedtuple("Model", "times xcoords ycoords analysis_tools_model")
    xcs = [0, 0.1, 0.2, 0.3, 0.4]
    ycs = [50, 50.1, 49.9, 50.3, 50.2]
    times = [datetime.datetime(2017,6,13,12,30) for _ in range(4)]
    class TestAnalysisToolsModel():
        def __init__(self, xcs, ycs):
            self.coords = xcs, ycs
        def projected_coords(self):
            return self.coords
    analysis_tools_model = TestAnalysisToolsModel(xcs, ycs)
    return Model(times, xcs, ycs, analysis_tools_model)

def test_harness(pred, root=None):
    import tkinter as tk
    if root is None:
        root = tk.Tk()
    view = pred.make_view(root)
    view.grid(sticky=tk.NSEW)
    root.mainloop()
