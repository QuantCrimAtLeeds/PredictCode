"""
process_file_view
~~~~~~~~~~~~~~~~~

View for processing the full file.
"""

import tkinter as tk
import tkinter.ttk as ttk
from . import util

_text = {
    "loading" : "Loading...",
    "loading2" : "Attempting to load entire file using current settings",
    "cancel" : "Cancel",
    "processing" : "Processing result",
    "loaded" : "Loaded {} rows of data, out of a possible {} rows.",
    "empty" : "Empty rows",
    "error" : "Rows with errors",
    "fur_empties" : "And a further {} empty rows",
    "fur_errors" : "And a further {} error rows",
    "continue" : "Continue to analysis",
    "quit" : "Quit to main menu",
    "back" : "Go back to import options"
}

class LoadFullFile(util.ModalWindow):
    def __init__(self, parent, task):
        super().__init__(parent, _text["loading"])
        self.cancelled = False
        self._task = task
        self._task.set_view(self)
        
    def add_widgets(self):
        self.set_size_percentage(20, 10)
        label = ttk.Label(self, text=_text["loading2"],
                          wraplength = self.winfo_width() - 20)
        label.grid(padx=10, pady=5)
        self.bar = ttk.Progressbar(self, mode="determinate")
        self.bar_pos = 0
        self.bar.grid(pady=5, padx=10, sticky=tk.E+tk.W)
        button = ttk.Button(self, text=_text["cancel"], command=self.cancel)
        button.grid(pady=5)
        self.set_to_actual_size()
    
    def notify(self, current, maximum):
        if not self.cancelled:
            pos = current * 100 / maximum
            self.bar.step(pos - self.bar_pos)
            self.bar_pos = pos

    def cancel(self):
        self.cancelled = True
        self._task.cancel()
        self.destroy()


class DisplayResult(util.ModalWindow):
    def __init__(self, parent, model):
        self.model = model
        super().__init__(parent, _text["processing"])
        self.result = "back"
        
    def add_widgets(self):
        self.set_size_percentage(50, 30)
        
        num_rows = len(self.model.data[0])
        possible_rows = num_rows + len(self.model.errors) + len(self.model.empties)
        text = _text["loaded"].format(num_rows, possible_rows)
        ttk.Label(self, text=text).grid(row=0, column=0, padx=5, pady=5)

        frame = ttk.LabelFrame(self, text=_text["empty"])
        frame.grid(row=1, column=0, sticky=util.NSEW, padx=5, pady=5)
        text = [e for _, e in zip(range(5), self.model.empties)]
        if len(self.model.empties) > 5:
            text.append(_text["fur_empties"].format(len(self.model.empties)-5))
        text = "\n".join(text)
        ttk.Label(frame, text=text).grid(sticky=util.NSEW, padx=5, pady=5)
        
        frame = ttk.LabelFrame(self, text=_text["error"])
        frame.grid(row=2, column=0, sticky=util.NSEW, padx=5, pady=5)
        text = [e for _, e in zip(range(5), self.model.errors)]
        if len(self.model.errors) > 5:
            text.append(_text["fur_errors"].format(len(self.model.errors)-5))
        text = "\n".join(text)
        ttk.Label(frame, text=text).grid(sticky=util.NSEW, padx=5, pady=5)
        
        frame = ttk.Frame(self)
        frame.grid(row=3, column=0, sticky=util.NSEW, padx=5, pady=5)
        ttk.Button(frame, text=_text["continue"], command=self._go).grid(column=0, row=0, padx=5)
        ttk.Button(frame, text=_text["back"], command=self.cancel).grid(column=1, row=0, padx=5)
        ttk.Button(frame, text=_text["quit"], command=self._quit).grid(column=2, row=0, padx=5)
        
        self.set_to_actual_size()

    def cancel(self):
        self.destroy()
        
    def _go(self):
        self.result = "go"
        self.destroy()
        
    def _quit(self):
        self.result = "quit"
        self.destroy()
        
