"""
app
~~~

The main application for when running in GUI mode.
"""

import logging
import sys

#import open_cp.gui.settings
from open_cp.gui import settings
from open_cp.gui import main_window

from open_cp.gui.tk import main_window_view
from open_cp.gui import locator

def start_logging():
    logger = logging.getLogger("open_cp")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.__stdout__)
    fmt = logging.Formatter("{asctime} {levelname} {name} - {message}", style="{")
    ch.setFormatter(fmt)
    logger.addHandler(ch)

def run():
    start_logging()
    # TODO: Move settings into the locator... Have it _yield_ it, perhaps?
    #   (or optionally)
    sett = settings.Settings()
    root = main_window_view.TopWindow()
    locator._make_pool(root)
    
    mw = main_window.MainWindow(root)
    mw.run()

    # Don't wait for threads...
    import os
    os._exit(0)