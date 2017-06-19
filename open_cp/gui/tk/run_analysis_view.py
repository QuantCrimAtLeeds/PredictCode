"""
run_analysis_view
~~~~~~~~~~~~~~~~~
"""

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as messagebox
import open_cp.gui.tk.util as util
import open_cp.gui.tk.richtext as richtext
import open_cp.gui.predictors as predictors
import queue, logging

_text = {
    "title" : "Running analysis",
    "okay" : "Okay",
    "cancel" : "Cancel",
    "dtfmt" : "%a %d %b %Y",
    "genfail" : "Failed to build RunAnlaysisModel",
    "log1" : "Constructed %s prediction task(s)",
    "log2" : "Number of days to predict for and score: %s",
    "log3" : "Starting from %s",
    "log4" : "Ending with %s",
    "log5" : "Built %s way(s) to lay down a grid",
    "log6" : "Built %s way(s) to project the coordinates",
    "log7" : "A total of %s tasks to run"
}

class RunAnalysisView(util.ModalWindow):
    def __init__(self, parent):
        super().__init__(parent, _text["title"], resize="wh")
        self.set_size_percentage(70, 50)
        self._formatter = logging.Formatter("{asctime} {levelname} : {message}", style="{")
        self._poll_task()
        self.protocol("WM_DELETE_WINDOW", self._close)
        
    def add_widgets(self):
        self._text = richtext.RichText(self, height=12, scroll="v")
        util.stretchy_rows_cols(self, [0], [0])
        self._text.grid(sticky=tk.NSEW)
        self._queue = queue.Queue()
        predictors.set_queue_logging(self._queue)
        
        frame = ttk.Frame(self)
        frame.grid(row=1, column=0, sticky=tk.EW)
        self._okay_button = ttk.Button(frame, text=_text["okay"], command=self.cancel)
        self._okay_button.grid(row=0, column=0, padx=5, pady=3, ipadx=20)
        self._okay_button.state(["disabled"])
        self._cancel_button = ttk.Button(frame, text=_text["cancel"], command=self.cancel)
        self._cancel_button.grid(row=0, column=1, padx=5, pady=3, ipadx=20)
        self._done = False

    def done(self):
        """Set the okay button as pressable."""
        self._done = True
        self._okay_button.state(["!disabled"])
        self._cancel_button.state(["disabled"])

    def cancel(self):
        predictors.set_edit_logging()
        self._queue = None
        super().cancel()

    def emit(self, record):
        text = self._formatter.format(record) + "\n"
        if record.levelno >= logging.ERROR:
            self._text.add_coloured_text(text, "#bb6666")
        else:
            self._text.add_text(text)

    def _close(self):
        if not self._done:
            if not messagebox.askyesno("Cancel tasks", "Do you want to cancel the runnnig tasks?"):
                return
        self.cancel()

    def _poll_task(self):
        if self._queue is None:
            return
        while not self._queue.empty():
            logger_entry = self._queue.get()
            self.emit(logger_entry)
        self.after(100, self._poll_task)