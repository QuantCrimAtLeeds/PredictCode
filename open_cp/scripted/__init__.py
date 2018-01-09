"""
scripted
~~~~~~~~

The GUI mode was originally desgined as the interactive way to use the
prediction algorithms.  However, many of the algorithms have turned out to be
rather slow to run, and often to require a certain amount of "tweaking",
neither of which is well-suited to a truly interactive mode.

This module provides a simple framework for writing _scripts_, short bits of
python code, which can perform prediction tasks.  This should be easier to use
than writing a custom Jupyter Notebook, for example, but more reproducible (and
easier to run overnight from the command line) than the GUI mode.

TODO: Quick list of what we provide.
"""

from .preds import *
from .evaluators import *
from .processors import *

from .. import data
from .. import logger
import logging, lzma, pickle, collections, datetime
from .. import geometry

logger.log_to_true_stdout()
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


# TODO: It might be nice to make this full "lazy evaluation", so that the
# `__exit__` method does everything.  But this would be a big refactor, and I
# don't have time right now.
class Data():
    def __init__(self, point_provider, geometry_provider=None, grid=None, start=None, end=None):
        _logger.info("Loading timed points...")
        points = point_provider()
        _logger.info("Loaded %s crime events, time range: %s", points.number_data_points, points.time_range)
        if start is not None:
            points = points[points.timestamps >= start]
            _logger.info("Restricted to events not before %s, leading %s events", start, points.number_data_points)
        if end is not None:
            points = points[points.timestamps < end]
            _logger.info("Restricted to events before %s, leading %s events", end, points.number_data_points)
        if grid is None:
            grid = data.Grid(150, 150, 0, 0)
        self._grid = grid
        self._geometry = None
        if geometry_provider is not None:
            _logger.info("Loading geometry...")
            self._geometry = geometry_provider()
            _logger.info("Masking grid with geometry...")
            self._grid = geometry.mask_grid_by_intersection(self._geometry, self._grid)
            _logger.info("Grid is now: %s", self._grid)
            _logger.info("Intersecting points with geometry...")
            self._points = geometry.intersect_timed_points(points, self._geometry)
        else:
            self._points = points
        _logger.info("Using %s events from %s to %s", self._points.number_data_points, *self._points.time_range)

        self._predictors = []
        self._evaluators = []
        self._processors = []
        self._prediction_cache = PredictionCache()
        self._prediction_notifiers = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if args[0] is not None:
            _logger.info("Exception raised in context body, rethrowing...")
            return False
        
        _logger.info("Starting processing...")
        for evaluator in self._evaluators:
            _logger.info("  Using evaluator %s", evaluator)
        _logger.info("Have a total of %s prediction methods", len(self._predictors))

        try:
            all_scores = dict()
            pl = logger.ProgressLogger(len(self._predictors), datetime.timedelta(minutes=1), _logger, level=logging.INFO)
            pl.message = "Total prediction tasks; completed %s / %s, time left: %s"
            for predictor, time_range, class_type in self._predictors:
                all_scores[predictor] = self._run_predictor(predictor, time_range, class_type), time_range
                pl.increase_count()
            
            for processor in self._processors:
                _logger.info("Running processor %s", processor)
                processor.init()
                for predictor, (scores, time_range) in all_scores.items():
                    for evaluator, score_list in scores.items():
                        processor.process(predictor, evaluator, score_list, time_range)
                processor.done()
        finally:
            for watcher in self._prediction_notifiers:
                watcher.close()

    def _run_predictor(self, predictor, times, class_type):
        _logger.info("Making predictions using %s for %s", predictor, times)
        pl = logger.ProgressLogger(len(times), datetime.timedelta(minutes=1), _logger, level=logging.INFO)
        scores = {ev : list() for ev in self._evaluators}
        
        for start, end in times:
            if not self._prediction_cache.has(predictor, start):
                _logger.debug("Making prediction for time %s", start)
                if class_type == 0:
                    pred = predictor.predict(start)
                elif class_type == 1:
                    pred = predictor.predict(start, end)
                else:
                    raise ValueError("Unsupported class type {}".format(class_type))
                self._prediction_cache.put(predictor, start, pred)
                for watcher in self._prediction_notifiers:
                    watcher.notify(predictor, start, pred)
            pred = self._prediction_cache.get(predictor, start)
            for evaluator in self._evaluators:
                score = evaluator.evaluate(pred, start, end)
                scores[evaluator].append(score)
            pl.increase_count()

        return scores

    def add_prediction(self, prediction_provider, times):
        """Add a prediction method to be run
        
        :param prediction_provider: Class of type
          :class:`open_cp.evaluation.StandardPredictionProvider` (not an
          instance!)
        :param times: An iterable of time intervals `[start, end)`, for
          example, :class:`TimeRange`
        """
        predictor = prediction_provider(self._points, self._grid)
        self._predictors.append((predictor, times, 0))

    def add_prediction_range(self, prediction_provider, times):
        """Add a prediction method to be run.  For the newer interface, where
        having an `end_time` on the prediction method is supported.
        
        :param prediction_provider: Class of type
          :class:`open_cp.evaluation.StandardPredictionProvider` (not an
          instance!)
        :param times: An iterable of time intervals `[start, end)`, for
          example, :class:`TimeRange`
        """
        predictor = prediction_provider(self._points, self._grid)
        self._predictors.append((predictor, times, 1))

    def score(self, evaluator):
        """Add an evaluation step.
        
        :param evaluator: Class of type :class:`EvaluatorBase` (not an
          instance!)
        """
        ev = evaluator()
        ev.data = self._points
        self._evaluators.append(ev)

    def process(self, interpretor):
        """Add a "processor" step.
        
        :param interpretor: Instance (not class!) of :class:`ProcessorBase`
        """
        self._processors.append(interpretor)

    def save_predictions(self, filename):
        """Save the input data points, geometry (if applicable) and each
        prediction.
        
        :param filename: Name of a file to save to.  Will be a `lzma`
          compressed `pickle` file.
        """
        saver = Saver(filename, self._points, self._geometry, self._grid)        
        self._prediction_notifiers.append(saver)

    @property
    def grid(self):
        """The grid we'll use for predictions"""
        return self._grid

    @property
    def timed_points(self):
        """The :class:`open_cp.data.TimedPoints" instance containing the input
        data."""
        return self._points



class Saver():
    def __init__(self, filename, points, geometry, grid):
        self._file = lzma.open(filename, "wb")
        pickle.dump(points, self._file)
        pickle.dump(geometry, self._file)
        pickle.dump(grid, self._file)
        
    def close(self):
        self._file.close()
        
    def notify(self, predictor, time, prediction):
        pickle.dump((predictor, time, prediction), self._file)
        _logger.debug("Saving prediction for {}".format(time))


LoadedPrediction = collections.namedtuple("LoadedPrediction", "predictor_class time prediction")

class Loader():
    """Use to load saved predictions.
    
    :param filename: The `lzma` compressed file to load.
    """
    def __init__(self, filename):
        with lzma.open(filename, "rb") as f:
            self._points = pickle.load(f)
            self._geometry = pickle.load(f)
            self._grid = pickle.load(f)
            self._predictions = []
            while True:
                try:
                    triple = pickle.load(f)
                except EOFError:
                    break
                self._predictions.append(triple)
                
    @property
    def timed_points(self):
        """The loaded data"""
        return self._points
    
    @property
    def geometry(self):
        """The loaded geometry, or `None`"""
        return self._geometry
    
    @property
    def grid(self):
        """The grid object we used for making predictions."""
        return self._grid
    
    def __iter__(self):
        for row in self._predictions:
            yield LoadedPrediction(*row)


class PredictionCache():
    """Simple cache from `(predictor, time)` to prediction.
    
    Currently just a wrapper around a dictionary."""
    def __init__(self):
        self._cache = dict()
        
    def _key(self, predictor, time):
        return (id(predictor), time)

    def has(self, predictor, time):
        return self._key(predictor, time) in self._cache
    
    def get(self, predictor, time):
        key = self._key(predictor, time)
        if key not in self._cache:
            return KeyError()
        return self._cache[key]
    
    def put(self, predictor, time, prediction):
        self._cache[self._key(predictor, time)] = prediction

    