"""
app
~~~

The main application for when running in GUI mode.
"""

import logging
import sys

#import open_cp.gui.settings
from open_cp.gui import settings

def start_logging():
    logger = logging.getLogger("open_cp")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.__stdout__)
    fmt = logging.Formatter("{asctime} {levelname} {name} - {message}", style="{")
    ch.setFormatter(fmt)
    logger.addHandler(ch)

def run():
    start_logging()
    sett = settings.Settings()
    