"""
app
~~~

The main application for when running in GUI mode.
"""

import logging, sys

def start_logging():
    logger = logging.getLogger("open_cp")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.__stdout__)
    fmt = logging.Formatter("{asctime} {levelname} {name} - {message}", style="{")
    ch.setFormatter(fmt)
    logger.addHandler(ch)

def run():
    start_logging()
    logger = logging.getLogger(__name__)
    logger.info("Started...")

    import os
    from open_cp.gui import main_window
    from open_cp.gui.tk import main_window_view
    from open_cp.gui import locator

    if "win" in sys.platform:
        logger.info("Platform is windows: '%s'", sys.platform)
        import ctypes
        myappid = "PredictCode.OpenCP"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    elif "linux" in sys.platform:
        logger.info("Platform is linux: '%s'", sys.platform)
    else:
        logger.warn("Unexpected flatform: %s.  So visuals might be wrong", sys.platform)


    root = main_window_view.TopWindow()
    locator._make_pool(root)

    #import open_cp.gui.predictors.retro as pred
    #pred.test(root)
    #return

    mw = main_window.MainWindow(root)
    # Quick start jump to analysis...    
    #mw.view.after(50, mw.load_session(os.path.join("..", "Open data sets", "session.json")))
    mw.run()

    # Don't wait for threads...
    os._exit(0)