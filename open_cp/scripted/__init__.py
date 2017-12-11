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

from ..evaluation import *
import numpy as np
from .. import data
from .. import logger
import logging
from .. import geometry
from .. import predictors

logger.log_to_stdout()
_logger = logging.getLogger(__name__)

class Data():
    def __init__(self, point_provider, geometry_provider=None, start=None, end=None):
        _logger.info("Loading timed points...")
        points = point_provider()
        _logger.info("Loaded %s crime events", points.number_data_points)
        if start is not None:
            points = points[points.timestamps >= start]
            _logger.info("Restricted to events not before %s, leading %s events", start, points.number_data_points)
        if end is not None:
            points = points[points.timestamps < end]
            _logger.info("Restricted to events before %s, leading %s events", end, points.number_data_points)
        self._grid = data.Grid(150, 150, 0, 0)
        if geometry_provider is not None:
            _logger.info("Loading geometry...")
            self._geometry = geometry_provider()
            # TODO: Grid tuning???
            _logger.info("Masking grid with geometry...")
            self._grid = geometry.mask_grid_by_intersection(self._geometry, self._grid)
            _logger.info("Grid is now: %s", self._grid)
            _logger.info("Intersecting points with geometry...")
            self._points = geometry.intersect_timed_points(points, self._geometry)
        else:
            self._points = points
        _logger.info("Using %s events from %s to %s", points.number_data_points, *points.time_range)

        self._predictors = []
        self._evaluators = []
        self._processors = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        _logger.info("Starting processing...")
        for predictor in self._predictors:
            self._run_predictor(predictor)

    def _run_predictor(self, predictor):
        # For each evaluator, if we need to, make a new prediction for the time,
        # and then score it.
        # Once done, run through the processor

    def add_prediction(self, prediction_provider):
        predictor = prediction_provider(self._points, self._grid)
        self._predictors.append(predictor)

    def score(self, evaluator, times):
        ev = evaluator()
        ev.data = self._points
        self._evaluators.append((ev, times))

    def process(self, interpretor):
        self._processors.append(interpretor)



def time_range(first, stop_before, duration):
    """Returns a list `[first, first + duration, first + 2 * duration, ...]`
    until we get to `stop_before` (not inclusive)."""
    out = []
    t = first
    while t < stop_before:
        out.append(t)
        t = t + duration
    return out

class HitRateEvaluator(predictors.DataTrainer):
    def __init__(self):
        pass



class ProcessorBase():
    def accept(self, evaluator):
        raise NotImplementedError()

class HitRateSave():
    def __init__(self, csv_filename):
        self._filename = csv_filename

    def accept(self, evaluator):
        return isinstance(evaluator, HitRateEvaluator)
    


    