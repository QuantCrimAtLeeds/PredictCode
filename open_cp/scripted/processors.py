"""
processors.py
~~~~~~~~~~~~~

Take the output from "evaluators" and process into a human-readable form.
"""

import csv as _csv
from . import evaluators

class ProcessorBase():
    """Abstract base class for `processors`.  The idea of a processor is to
    take the output of an "evaluator" and to produce some human readable
    output: a table, a graph, etc."""

    def init(self):
        """Perform any initialisation (e.g. open file, write header etc.)"""
        raise NotImplementedError()
        
    def process(self, predictor, evaluator, scores, time_range):
        """Perform the magic on this data.
        
        :param predictor: The predictor which produced the original prediction.
        :param evaluator: The evaluator which produced these "scores".  If this
          evaluator is not applicable for this processor, just silently ignore.
        :param scores: List of results from the evaluator.
        :param time_range: Iterable of `[start, end)` intervals, same length
            as the scores.
        """
        raise NotImplementedError()

    def done(self):
        """Notify that all information has been sent."""
        raise NotImplementedError()


class _TextFileOpenMixin():
    def _open(self, filename, callback):
        need_close = False
        if isinstance(filename, str):
            file = open(filename, "wt", newline="")
            need_close = True
        else:
            file = filename
        try:
            callback(file)
        finally:
            if need_close:
                file.close()


class HitRateSave(_TextFileOpenMixin):
    """Process the hit rates into a simple CSV file of time ranges
    against hit rates for various coverage levels.
    
    :param csv_filename: Filename or file-like object
    :param coverages: Iterable of coverage levels; or `None` to mean 1, 2, 3,
      ...,  100%.  Must be integers in range [1,100]
    """
    def __init__(self, csv_filename, coverages=None):
        self._filename = csv_filename
        if coverages is None:
            coverages = range(1, 101, 1)
        self._coverages = list(coverages)
        self._rows = []

    def init(self):
        pass
    
    def done(self):
        self._open(self._filename, self._done)
    
    def _done(self, file):
        writer = _csv.writer(file)
        writer.writerow(self._header())
        writer.writerows(self._rows)
    
    def _header(self):
        header = ["Predictor", "Start time", "End time"]
        for c in self._coverages:
            header.append("{}%".format(c))
        return header

    def process(self, predictor, evaluator, scores, time_range):
        if not isinstance(evaluator, evaluators.HitRateEvaluator):
            return

        for hit_rate, (start, end) in zip(scores, time_range):
            row = [str(predictor), str(start), str(end)]
            for cov in self._coverages:
                row.append(hit_rate[cov])
            self._rows.append(row)
            
    def __repr__(self):
        return "HitRateSave(filename='{}')".format(self._filename)


class HitCountSave(_TextFileOpenMixin):
    """Process the hit counts into a simple CSV file of time ranges
    against hit counts for various coverage levels.
    
    :param csv_filename: Filename or file-like object
    :param coverages: Iterable of coverage levels; or `None` to mean 1, 2, 3,
      ...,  100%.  Must be integers in range [1,100]
    """
    def __init__(self, csv_filename, coverages=None):
        self._filename = csv_filename
        if coverages is None:
            coverages = range(1, 101, 1)
        self._coverages = list(coverages)
        self._rows = []

    def init(self):
        pass
    
    def done(self):
        self._open(self._filename, self._done)
            
    def _done(self, file):
        writer = _csv.writer(file)
        writer.writerow(self._header())
        writer.writerows(self._rows)
    
    def _header(self):
        header = ["Predictor", "Start time", "End time", "Number events"]
        for c in self._coverages:
            header.append("{}%".format(c))
        return header

    def process(self, predictor, evaluator, scores, time_range):
        if not isinstance(evaluator, evaluators.HitCountEvaluator):
            return

        for hit_rate, (start, end) in zip(scores, time_range):
            row = [str(predictor), str(start), str(end)]
            row.append( next(iter(hit_rate.values()))[1] )
            for cov in self._coverages:
                row.append(hit_rate[cov][0])
            self._rows.append(row)
            
    def __repr__(self):
        return "HitCountSave(filename='{}')".format(self._filename)
