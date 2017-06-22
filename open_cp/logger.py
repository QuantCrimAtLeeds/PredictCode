"""
logger
~~~~~~

Provides some simple logging functionality.  A thin wrapper around the Python
standard library :mod:`logging` module.

Currently, few modules log.  Logging is always under the `__name__` of the module.
"""

import logging as _logging
import sys as _sys

_current_handler = None

def _set_handler(handler):
    logger = _logging.getLogger("open_cp")
    global _current_handler
    if _current_handler is not None:
        logger.removeHandler(_current_handler)
    _current_handler = handler
    logger.addHandler(handler)

def standard_formatter():
    """Our standard logging formatter"""
    return logging.Formatter("{asctime} {levelname} {name} - {message}", style="{")

def log_to_stdout():
    """Start logging to `stdout`.  In a Jupyter notebook, this will print
    logging to the notebook.
    """
    logger = logging.getLogger("open_cp")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(standard_formatter())
    _set_handler(ch)

def log_to_true_stdout():
    """Start logging to the "real" `stdout`.  In a Jupyter notebook, this will
    print logging to the console the server is running in (and not to the
    notebook) itself.
    """
    logger = logging.getLogger("open_cp")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.__stdout__)
    ch.setFormatter(standard_formatter())
    _set_handler(ch)
