"""
pred_type
~~~~~~~~~

Describe the top level prediction required.  E.g.:

    - Produce a prediction for each day in the assessment time range, and score
      the prediction using the actual events which occurred that day.
    - Or... same, but on a weekly basis.
"""

from . import comparitor
import logging
import tkinter as tk
import tkinter.ttk as ttk
import open_cp.gui.tk.util as util
import open_cp.gui.tk.tooltips as tooltips
import open_cp.gui.tk.richtext as richtext
import datetime

_text = {
    "main" : ("Prediction Type\n\n"
              + "Using the selected 'assessment time range' as a base, how often do we want to create predictions, and what time range do we wish to 'score' the prediction on?\n\n"
              + "Currently supported is making predictions for a whole number of days, and then testing each prediction for a whole number of days."
              + "You may mix and match, but be aware of 'multiple testing' issues with setting the score interval to be longer than the repeat interval.\n"
              + "For example, the default setting is 1 day and 1 day.  This generates a prediction for every day in the selected assessment time range, and then scores each such prediction by comparing against the actual events for that day.\n"
              + "Setting to 7 days and 1 day would generate a prediction for every seven days (i.e. once a week) but would score for just the next day.  This could be used to test predictions for just one day of the week."),
    "every" : "Repeat interval:",
    "everytt" : "How often do we want to create a prediction?  Should be a whole number of days.",
    "score" : "Score time interval:",
    "scorett" : "How long a period of time should each prediction be compared with reality for?  Should be a whole number of days.",
    "whole_day_warning" : "Currently we only support making predictions for whole days.",
    "wdw_round" : "Rounded {} to {}",
    "pp" : "Preview of Prediction ranges",
    "pp1" : "Predict for {} and score for the next {} day(s)",
    "dtfmt" : "%a %d %b %Y",
    
}

class PredType(comparitor.Comparitor):
    def __init__(self, model):
        super().__init__(model)
        self._every = datetime.timedelta(days=1)
        self._score = datetime.timedelta(days=1)
    
    @staticmethod
    def describe():
        return "Prediction Type required"

    @staticmethod
    def order():
        return comparitor.TYPE_TOP_LEVEL

    def make_view(self, parent):
        self._view  = PredTypeView(parent, self)
        return self._view

    @property
    def name(self):
        return "Predict for every {} days, scoring the next {} days".format(
            self._every / datetime.timedelta(days=1),
            self._score / datetime.timedelta(days=1)
            )
        
    @property
    def settings_string(self):
        return None

    def config(self):
        return {"resize" : True}

    def to_dict(self):
        return { "every_interval" : self.every.total_seconds(),
            "score_interval" : self.score_length.total_seconds() }

    def from_dict(self, data):
        every_seconds = data["every_interval"]
        self._every = datetime.timedelta(seconds = every_seconds)
        score_seconds = data["score_interval"]
        self._score = datetime.timedelta(seconds = score_seconds)
        
    @property
    def every(self):
        """Period at which to generate predictions."""
        return self._every
    
    @every.setter
    def every(self, value):
        self._every = value
    
    @property
    def score_length(self):
        """Length of time to score the prediction on."""
        return self._score
    
    @score_length.setter
    def score_length(self, value):
        self._score = value

    @staticmethod
    def _just_date(dt):
        return datetime.datetime(year=dt.year, month=dt.month, day=dt.day)

    def run(self):
        """Returns a list of pairs `(start_date, score_duration)`"""
        _, _, _assess_start, _assess_end = self._model.time_range
        logger = logging.getLogger(comparitor.COMPARATOR_LOGGER_NAME)
        assess_start = self._just_date(_assess_start)
        assess_end = self._just_date(_assess_end)
        if assess_start != _assess_start or assess_end != _assess_end:
            logger.warn(_text["whole_day_warning"])
            if assess_start != _assess_start:
                logger.warn(_text["wdw_round"].format(_assess_start, assess_start))
            if assess_end != _assess_end:
                logger.warn(_text["wdw_round"].format(_assess_end, assess_end))

        out = []
        start = assess_start
        while True:
            end = start + self.score_length
            if end > assess_end:
                break
            out.append( (start, self.score_length) )
            start += self.every
        return out


class PredTypeView(tk.Frame):
    def __init__(self, parent, model):
        super().__init__(parent)
        self._model = model
        util.stretchy_rows_cols(self, [0], [0])
        self._text = richtext.RichText(self, height=12, scroll="v")
        self._text.grid(sticky=tk.NSEW, row=0, column=0)
        self._text.add_text(_text["main"])

        frame = ttk.Frame(parent)
        frame.grid(row=1, column=0, sticky=tk.NSEW)
        ttk.Label(frame, text=_text["every"]).grid(row=0, column=0, sticky=tk.E, pady=2)
        self._every_var = tk.StringVar()
        self._every = ttk.Entry(frame, width=5, textvariable=self._every_var)
        self._every.grid(row=0, column=1, sticky=tk.W, pady=2)
        util.IntValidator(self._every, self._every_var, self.change)
        tooltips.ToolTipYellow(self._every, _text["everytt"])

        ttk.Label(frame, text=_text["score"]).grid(row=1, column=0, sticky=tk.E, pady=2)
        self._score_var = tk.StringVar()
        self._score = ttk.Entry(frame, width=5, textvariable=self._score_var)
        self._score.grid(row=1, column=1, sticky=tk.W, pady=2)
        util.IntValidator(self._score, self._score_var, self.change)
        tooltips.ToolTipYellow(self._score, _text["scorett"])
        
        self._preview_frame = ttk.LabelFrame(frame, text=_text["pp"])
        self._preview_frame.grid(row=2, column=0, columnspan=2, sticky=tk.NSEW, padx=5, pady=5)
        self._preview_label = ttk.Label(self._preview_frame)
        self._preview_label.grid(sticky=tk.NSEW)
        
        self.update()
        
    @staticmethod
    def _add_time(text, start, length):
        text.append( _text["pp1"].format(start.strftime(_text["dtfmt"]), length.days) )
        
    def update(self):
        self._every_var.set( int(self._model.every / datetime.timedelta(days=1)) )
        self._score_var.set( int(self._model.score_length / datetime.timedelta(days=1)) )
        preds = self._model.run()
        text = []
        for (start, length), _ in zip(preds, range(2)):
            self._add_time(text, start, length)
        if len(preds) > 3:
            text.append("...")
        if len(preds) > 2:
            start, length = preds[-1]
            self._add_time(text, start, length)
        self._preview_label["text"] = "\n".join(text)
        
    def change(self):
        every = int(self._every_var.get())
        if every > 0:
            self._model.every = datetime.timedelta(days=every)
        score = int(self._score_var.get())
        if score > 0:
            self._model.score_length = datetime.timedelta(days=score)
        self.update()
