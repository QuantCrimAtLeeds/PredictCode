"""
process_file
~~~~~~~~~~~~

Controller and model for loading the full dataset.
"""

import csv
import array
import open_cp.gui.tk.process_file_view as process_file_view
import open_cp.gui.locator as locator
from open_cp.gui.import_file_model import ParseErrorData
import open_cp.gui.import_file_model as import_file_model
import open_cp.gui.tk.threads as threads
from collections import namedtuple
from . import locator

Model = namedtuple("Model", "errors empties data settings")


class ProcessFile():
    """Load in the entire file, display progress while we do, then display
    errors and allow the user to continue or not.
    
    :param filename: Name of the file to load
    :param total_rows: The total number of rows in the file (used to display
      progress information)
    :param processor: A coroutine which can be sent rows from the CSV file and
      which will yield data back
    :param parent_view: The window from which to construct our modal dialog.
    """
    def __init__(self, filename, total_rows, parse_settings, parent_view, mode="initial"):
        self._filename = filename
        self._total_rows = total_rows
        self._parse_settings = parse_settings
        self._parent = parent_view
        self._mode = mode
    
    def run(self):
        """Loads, displays results.

        :return: `None` means "go back to import options.",
          `True` means continue,
          `False` means quit to main menu
        """
        task = LoadTask(self._filename, self._total_rows, self._parse_settings, self)
        self._view = process_file_view.LoadFullFile(self._parent, task)
        locator.get("pool").submit(task)
        self._view.wait_window(self._view)
        if self._view.cancelled:
            return None
        
        self._view = process_file_view.DisplayResult(self._parent, self.model, self._mode)
        self._view.wait_window(self._view)
        if self._view.result == "back":
            return None
        elif self._view.result == "go":
            return True
        elif self._view.result == "quit":
            return False
        else:
            raise ValueError()
        
    def done_process_whole_file(self, value):
        errors, empties, times, xcs, ycs, ctypes = value
        self.model = Model(errors=errors, empties=empties, data=(times, xcs, ycs, ctypes),
                settings = self._parse_settings)
        self._view.destroy()

    def error_in_process_file(self, ex):
        self._view.alert_error(ex)
        self._view.cancel()
    

def rows_in_csv(filename):
    """Helper function to quickly count the number of rows in the passed csv
    file."""
    with open(filename, encoding="UTF8", mode="rt") as f:
        reader = csv.reader(f)
        count = 0
        for _ in reader:
            count += 1
        return count - 1


class LoadTask(threads.OffThreadTask, locator.GuiThreadTask):
    """Actually load the data.  Parameters as for :class:`ProcessFile`

    :param parent: The instance of :class:`ProcessFile` to send the result
      back to.
    """
    def __init__(self, filename, total_rows, parse_settings, parent):
        super().__init__()
        self._filename = filename
        self._parse_settings = parse_settings
        self._controller = parent
        self._total_rows = total_rows
        self._view = None

    def _yield_rows(self):
        with open(self._filename, encoding="UTF8", mode="rt") as f:
            reader = csv.reader(f)
            yield from reader
    
    @staticmethod
    def _handle_exception(row, ex, errors, empties):
        if isinstance(ex, ParseErrorData):
            reason = "Row {}: ".format(row)
            if ex.reason == "time":
                reason += "timestamp "
            elif ex.reason == "X":
                reason += "X Coord "
            elif ex.reason == "Y":
                reason += "Y Coord "
            else:
                raise ValueError()
            if ex.data == "":
                reason += "is empty, so skipping."
                empties.append(reason)
            else:
                reason += "is '{}' and cannot be understood, so skipping.".format(ex.data)
                errors.append(reason)
        else:
            errors.append("Unexpected exception: {}/{}".format(type(ex), str(ex)))

    def _calc_total_rows(self):
        count = 0
        for row in self._yield_rows():
            count += 1
        return count - 1

    def __call__(self):
        if self._total_rows is None:
            self.submit_gui_task(lambda : self._view.start_indet_progress())
            self._total_rows = self._calc_total_rows()
            if self._view.cancelled:
                return
            self.submit_gui_task(lambda : self._view.start_det_progress())
        processor = import_file_model.Model.load_full_dataset(self._parse_settings)
        reader = self._yield_rows()
        header = next(reader)
        times, ctypes = [], []
        xcs, ycs = array.array("d"), array.array("d")
        errors, empties = [], []
        row_count = 0
        next(processor)
        try:
            for row in reader:
                if self.cancelled:
                    return
                row_count += 1
                if row_count % 1000 == 0 and self._view is not None:
                    self.submit_gui_task(lambda : self._view.notify(row_count, self._total_rows))
                data = processor.send(row)
                if isinstance(data[1], Exception):
                    self._handle_exception(*data, errors, empties)
                else:
                    
                    t,x,y,*ct = data
                    times.append(t)
                    xcs.append(x)
                    ycs.append(y)
                    ctypes.append(ct)
        finally:
            processor.close()
        return errors, empties, times, xcs, ycs, ctypes

    def set_view(self, view):
        self._view = view

    def on_gui_thread(self, value):
        if isinstance(value, Exception):
            self._controller.error_in_process_file(value)
            return
        elif value is None:
            return
        else:
            self._controller.done_process_whole_file(value)
