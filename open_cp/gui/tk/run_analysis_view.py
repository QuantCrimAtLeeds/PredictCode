"""
run_analysis_view
~~~~~~~~~~~~~~~~~

View for both `run_analysis` and `run_comparison`.
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
    "title2" : "Running comparisons",
    "okay" : "Okay",
    "cancel" : "Cancel",
    "dtfmt" : "%a %d %b %Y",
    "genfail" : "Failed to build RunAnlaysisModel",
    "genfail1" : "Failed to build RunComparisonModel",
    "log1" : "Constructed %s prediction task(s)",
    "log2" : "Number of days to predict for and score: %s",
    "log3" : "Starting from %s",
    "log4" : "Ending with %s",
    "log5" : "Built %s way(s) to lay down a grid",
    "log6" : "Built %s way(s) to project the coordinates",
    "log7" : "A total of %s tasks to run",
    "log8" : "Completed task %s",
    "log9" : "Collecting tasks to run...",
    "log10" : "Cancelling run...",
    "log11" : "Unexpected error while running analysis: {}",
    "log12" : "Internal error, got: '{}'",
    "log13" : "All tasks collected, now running...",
    "log14" : "Adjusting predictions for %s with projection %s...",
    "log15" : "Failed to find current projector matching %s",
    "warning" : "Failed to complete analysis run",
    "warning1" : "Analysis run stopped early due to: {}",
    "cancel" : "Cancel tasks",
    "cancel1" : "Do you want to cancel the running tasks?",
    
}

class RunAnalysisView(util.ModalWindow):
    def __init__(self, parent, controller, title=_text["title"]):
        self._controller = controller
        super().__init__(parent, title, resize="wh")
        self.set_size_percentage(70, 50)
        self._formatter = logging.Formatter("{asctime} {levelname} : {message}", style="{")
        self.protocol("WM_DELETE_WINDOW", self._close)
        
    def add_widgets(self):
        self._text = richtext.RichText(self, height=12, scroll="v")
        util.stretchy_rows_cols(self, [0], [0])
        self._text.grid(sticky=tk.NSEW)
        self._queue = queue.Queue()
        predictors.set_queue_logging(self._queue)
        
        self._bottom_frame = ttk.Frame(self)
        self._bottom_frame.grid(row=1, column=0, sticky=tk.EW)
        util.stretchy_columns(self._bottom_frame, [2])
        self._okay_button = ttk.Button(self._bottom_frame, text=_text["okay"], command=self.cancel)
        self._okay_button.grid(row=0, column=0, padx=5, pady=3, ipadx=20)
        self._okay_button.state(["disabled"])
        self._cancel_button = ttk.Button(self._bottom_frame, text=_text["cancel"], command=self._cancel)
        self._cancel_button.grid(row=0, column=1, padx=5, pady=3, ipadx=20)
        self._done = False
        self._poll_task()

    def done(self):
        """Set the okay button as pressable."""
        self._done = True
        self._okay_button.state(["!disabled"])
        self._cancel_button.state(["disabled"])

    def _cancel(self):
        if self._done:
            self.cancel()
        else:
            self._controller.cancel()

    def cancel(self):
        predictors.set_edit_logging()
        self._queue = None
        super().cancel()

    def emit(self, record):
        """Log the `record` to the window."""
        text = self._formatter.format(record) + "\n"
        if record.levelno >= logging.ERROR:
            self._text.add_coloured_text(text, "#bb6666")
        else:
            self._text.add_text(text)
        self._text.widget.see(tk.END)

    def start_progress_bar(self):
        self._bar = ttk.Progressbar(self._bottom_frame, mode="determinate")
        self._bar_pos = 0
        self._bar.grid(row=0, column=2, padx=10, sticky=tk.EW)

    def stop_progress_bar(self):
        if self._bar is not None:
            self._bar.grid_forget()
            self._bar = None

    def set_progress(self, done, out_of):
        new_pos = done * 100 / out_of
        self._bar.step(new_pos - self._bar_pos)
        self._bar_pos = new_pos

    def alert(self, message):
        messagebox.showwarning(_text["warning"], message)

    def _close(self):
        if not self._done:
            if not messagebox.askyesno(_text["cancel"], _text["cancel1"]):
                return
        self._cancel()

    def _poll_task(self):
        if self._queue is None:
            return
        while not self._queue.empty():
            logger_entry = self._queue.get()
            self.emit(logger_entry)
        self.after(100, self._poll_task)