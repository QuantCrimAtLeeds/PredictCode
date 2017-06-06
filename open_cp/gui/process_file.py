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

class Model():
    def __init__(self):
        pass


class ProcessFile():
    def __init__(self, filename, processor, parent_view):
        self._filename = filename
        self._processor = processor
        self._parent = parent_view
    
    def run(self):
        self._view = process_file_view.LoadFullFile(self._parent)
        pool = locator.get("pool")
        pool.submit(self._process_whole_file, self._done_process_whole_file)
        self._view.wait_window(self._view)
        
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
        
    def _yield_rows(self):
        with open(self._filename, encoding="UTF8", mode="rt") as f:
            reader = csv.reader(f)
            yield from reader
        
    def _process_whole_file(self):
        try:
            reader = self._yield_rows()
            header = next(reader)
            next(self._processor)
            times = []
            xcs, ycs = array.array("d"), array.array("d")
            errors, empties = [], []
            try:
                for row in reader:
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
        except Exception as ex:
            return str(ex)
        return errors, empties, times, xcs, ycs

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

    def _done_process_whole_file(self, value):
        model = Model()
        model.errors, model.empties, times, xcs, ycs = value    
        model.data = (times, xcs, ycs)
        self._view.destroy()
        self.model = model
