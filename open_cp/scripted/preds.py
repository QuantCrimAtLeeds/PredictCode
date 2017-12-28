"""
preds.py
~~~~~~~~

Wrappers around each prediction algorithm, and some utilities.

We import all of :mod:`open_cp.evaluation` and reuse the
:class:`StandardPredictionProvider` and subclasses thereof.
"""

from ..evaluation import *

import logging as _logging
import math as _math
_logger = _logging.getLogger(__name__)


class TimeRange():
    """A simple class to define an interface for time ranges.  This class just
    supports a series of equally sized, contiguous intervals, but the interface
    is general.
    
    Encapsulates a list of intervals
    - `[first, first + duration)`
    - `[first + duration, first + 2 * duration)`
    - ...
    Stops before `stop_before`.
        
    :param first: Start of the first inverval.
    :param stop_before: The final interval will not include this time point.
    :param duration: Length of each interval.
    """
    def __init__(self, first, stop_before, duration):
        self._first = first
        self._stop_before = stop_before
        self._duration = duration

    def __iter__(self):
        st = self._first
        en = st + self._duration
        while en <= self._stop_before:
            yield st, en
            st = en
            en = st + self._duration

    def __len__(self):
        return _math.floor((self._stop_before - self._first) / self._duration)

    def __repr__(self):
        return "TimeRange({} -> {}, length={})".format(self._first,
                          self._stop_before, self._duration)

    @property
    def first(self):
        """Start of the time range."""
        return self._first

    @property
    def stop_before(self):
        """End (not inclusive) of the time range."""
        return self._stop_before

    @property
    def duration(self):
        """Gap between each entry."""
        return self._duration
