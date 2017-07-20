"""
logger
~~~~~~

Provides some simple logging functionality.  A thin wrapper around the Python
standard library :mod:`logging` module.

Currently, few modules log.  Logging is always under the `__name__` of the module.
"""

import logging as _logging
import sys as _sys


class _OurHandler(_logging.StreamHandler):
    """Wrap :class:`logging.StreamHandler` with an attribute which we can
    recognise, to stop adding more than one handler."""
    def __init__(self, obj):
        super().__init__(obj)
        self.open_cp_marker = True


def _set_handler(handler):
    logger = _logging.getLogger("open_cp")
    existing = [ h for h in logger.handlers if hasattr(h, "open_cp_marker") ]
    for h in existing:
        logger.removeHandler(h)
    logger.addHandler(handler)

def standard_formatter():
    """Our standard logging formatter"""
    return _logging.Formatter("{asctime} {levelname} {name} - {message}", style="{")

def _log_to(file):
    logger = _logging.getLogger("open_cp")
    logger.setLevel(_logging.DEBUG)
    ch = _OurHandler(file)
    ch.setFormatter(standard_formatter())
    _set_handler(ch)

def log_to_stdout():
    """Start logging to `stdout`.  In a Jupyter notebook, this will print
    logging to the notebook.
    """
    _log_to(_sys.stdout)

def log_to_true_stdout():
    """Start logging to the "real" `stdout`.  In a Jupyter notebook, this will
    print logging to the console the server is running in (and not to the
    notebook) itself.
    """
    _log_to(_sys.__stdout__)
