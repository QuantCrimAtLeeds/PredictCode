"""
logger
~~~~~~

Provides some simple logging functionality.  A thin wrapper around the Python
standard library :mod:`logging` module.

Currently, few modules log.  Logging is always under the `__name__` of the module.
"""

import logging as _logging
import sys as _sys
import datetime as _datetime


class _OurHandler(_logging.StreamHandler):
    """Wrap :class:`logging.StreamHandler` with an attribute which we can
    recognise, to stop adding more than one handler."""
    def __init__(self, obj):
        super().__init__(obj)
        self.open_cp_marker = True


def _set_handler(handler, logger):
    existing = [ h for h in logger.handlers if hasattr(h, "open_cp_marker") ]
    for h in existing:
        logger.removeHandler(h)
    logger.addHandler(handler)

def standard_formatter():
    """Our standard logging formatter"""
    return _logging.Formatter("{asctime} {levelname} {name} - {message}", style="{")

def _log_to(file, name="open_cp"):
    logger = _logging.getLogger(name)
    logger.setLevel(_logging.DEBUG)
    ch = _OurHandler(file)
    ch.setFormatter(standard_formatter())
    _set_handler(ch, logger)

def log_to_stdout(name="open_cp"):
    """Start logging to `stdout`.  In a Jupyter notebook, this will print
    logging to the notebook.
    """
    _log_to(_sys.stdout, name)

def log_to_true_stdout(name="open_cp"):
    """Start logging to the "real" `stdout`.  In a Jupyter notebook, this will
    print logging to the console the server is running in (and not to the
    notebook) itself.
    """
    _log_to(_sys.__stdout__, name)


class ProgressLogger():
    """A simple way to report progress of a long-running task.
    
    :param target: The total number of "events" to count.
    :param report_interval: `timedelta` object giving the ideal report period.
    """
    def __init__(self, target, report_interval, logger=None):
        self._target = target
        self._report_interval = report_interval
        self._count = 0
        self._start = _datetime.datetime.now()
        self._last_log_time = None
        self.logger = logger
        
    def increase_count(self):
        """Increase the count, and if appropriate, log.  If the logger is set
        then debug log.
        
        :return: `None` if no log needed, or :attr:`logger` is not set.
          Otherwise `(count, target, time_left)`.
        """
        self._count += 1
        now = _datetime.datetime.now()
        out = None
        if self._last_log_time is None or now - self._last_log_time >= self._report_interval:
            expected_time = (now - self._start) / self._count * self._target
            expected_end = self._start + expected_time
            expected_time_left = expected_end - now
            out = (self._count, self._target, expected_time_left)
            self._last_log_time = now
        if out is not None and self.logger is not None:
            self.logger.debug("Completed %s out of %s, time left: %s", *out)
        else:
            return out
        
    @property
    def count(self):
        """The number of counts logged."""
        return self._count
        
    @property
    def logger(self):
        """The logger to output to, or `None`."""
        return self._logger
    
    @logger.setter
    def logger(self, v):
        self._logger = v
    