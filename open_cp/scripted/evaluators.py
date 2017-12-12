"""
evaluators.py
~~~~~~~~~~~~~

We use a more general notion of "evaluator" than that in
:mod:`open_cp.evaluation`.
"""

from .. import evaluation as _evaluation
from .. import predictors as _predictors

import logging as _logging
_logger = _logging.getLogger(__name__)


class EvaluatorBase(_predictors.DataTrainer):
    """Abstract base class for `evaluators`."""
    def __init__(self):
        super().__init__()
    
    def evaluate(self, prediction, start, end):
        """For the given prediction, look at the events which occur in
        [start, end) and return a "score"."""
        raise NotImplementedError()
    

class HitRateEvaluator(EvaluatorBase):
    """We don't use the "inverse hit rate" because of the "breaking ties"
    issue.  This is slower, but at least makes it reproducible."""
    def __init__(self):
        super().__init__()

    def evaluate(self, prediction, start, end):
        mask = (self.data.timestamps >= start) & (self.data.timestamps < end)
        points = self.data[mask]
        coverages = list(range(1,101))
        return _evaluation.hit_rates(prediction, points, coverages)


class HitCountEvaluator(EvaluatorBase):
    """Return counts `(capture_crime_count, total_crime_count)` instead of
    the rate."""
    def __init__(self):
        super().__init__()

    def evaluate(self, prediction, start, end):
        mask = (self.data.timestamps >= start) & (self.data.timestamps < end)
        points = self.data[mask]
        coverages = list(range(1,101))
        return _evaluation.hit_counts(prediction, points, coverages)
