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
import open_cp.gui.tk.threads as threads
from collections import namedtuple
from . import locator

Model = namedtuple("Model", "errors empties data")


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
    def __init__(self, filename, total_rows, processor, parent_view):
        self._filename = filename
        self._total_rows = total_rows
        self._processor = processor
        self._parent = parent_view
    
    def run(self):
        """Loads, displays results.

        :return: `None` means "go back to import options.",
          `True` means continue,
          `False` means quit to main menu
        """
        task = LoadTask(self._filename, self._total_rows, self._processor, self)
        locator.get("pool").submit(task)
        self._view = process_file_view.LoadFullFile(self._parent, task)
        self._view.wait_window(self._view)
        if self._view.cancelled:
            return None
        
        self._view = process_file_view.DisplayResult(self._parent, self.model)
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
        errors, empties, times, xcs, ycs = value
        self.model = Model(errors=errors, empties=empties, data=(times, xcs, ycs))
        self._view.destroy()


class LoadTask(threads.OffThreadTask, locator.GuiThreadTask):
    """Actually load the data.  Parameters as for :class:`ProcessFile`

    :param parent: The instance of :class:`ProcessFile` to send the result
      back to.
    """
    def __init__(self, filename, total_rows, processor, parent):
        super().__init__()
        self._filename = filename
        self._processor = processor
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

    def __call__(self):
        reader = self._yield_rows()
        header = next(reader)
        next(self._processor)
        times = []
        xcs, ycs = array.array("d"), array.array("d")
        errors, empties = [], []
        row_count = 0
        try:
            for row in reader:
                if self.cancelled:
                    return
                row_count += 1
                if row_count % 1000 == 0 and self._view is not None:
                    self.submit_gui_task(lambda : self._view.notify(row_count, self._total_rows))
                data = self._processor.send(row)
                if isinstance(data[1], Exception):
                    self._handle_exception(*data, errors, empties)
                else:
                    t,x,y = data
                    times.append(t)
                    xcs.append(x)
                    ycs.append(y)
        finally:
            self._processor.close()
        return errors, empties, times, xcs, ycs

    def set_view(self, view):
        self._view = view

    def on_gui_thread(self, value):
        if value is None:
            return
        self._controller.done_process_whole_file(value)
