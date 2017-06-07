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
    time_format = ""
    time_field = 1
    xcoord_field = 4
    ycoord_field = 5
    from . import import_file_model
    processor = import_file_model.Model.load_full_dataset(time_format, time_field, xcoord_field, ycoord_field)

    from . import process_file
    task = process_file.LoadTask(filename, 100000, processor, None)
    errors, empties, times, xcs, ycs = task()
    # TODO: More here in due course...
    from . import analysis
    model = analysis.Model(filename, (times, xcs, ycs))
    model.num_empty_rows = len(empties)
    model.num_error_rows = len(errors)
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