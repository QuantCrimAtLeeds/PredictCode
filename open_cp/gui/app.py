"""
app
~~~

The main application for when running in GUI mode.
"""

import logging
import sys
import os

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

def jump_to_analysis(root):
    import os.path
    #filename = os.path.join("..", "Open data sets", "2017-01-cumbria-street.csv")
    filename = "../../Crime Predict Project/Open data sets/2017-01-cumbria-street.csv"
    from . import import_file_model
    parse_settings = import_file_model.ParseSettings()
    parse_settings.timestamp_field = 1
    parse_settings.xcoord_field = 4
    parse_settings.ycoord_field = 5
    parse_settings.crime_type_fields = [9]

    from . import process_file
    pf = process_file.ProcessFile(filename, 100000, parse_settings, root)
    assert pf.run()

    from . import analysis
    model = analysis.Model.init_from_process_file_model(filename, pf.model)
    analysis.Analysis(model, root).run()

def run():
    start_logging()
    # TODO: Move settings into the locator... Have it _yield_ it, perhaps?
    #   (or optionally)
    sett = settings.Settings()
    root = main_window_view.TopWindow()
    locator._make_pool(root)

    jump_to_analysis(root)
    os._exit(0)

    mw = main_window.MainWindow(root)
    mw.run()

    # Don't wait for threads...
    os._exit(0)