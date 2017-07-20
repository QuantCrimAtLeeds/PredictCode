"""
browse_comparison_view
~~~~~~~~~~~~~~~~~~~~~~

"""

import tkinter as tk
import tkinter.ttk as ttk
import open_cp.gui.tk.util as util
import open_cp.gui.tk.hierarchical_view as hierarchical_view
import open_cp.gui.tk.simplesheet as simplesheet
#import open_cp.gui.tk.tooltips as tooltips

_text = {
    "title" : "Previous Comparison results.  Run @ {}",
    "dtfmt" : "%d %b %Y %H:%M",
    "key1" : "Projection method:",
    "key2" : "Grid type:",
    "key3" : "Prediction length:",
    "key4" : "Adjustment method:",
    "key5" : "Comparison method:",
    "row1" : "Prediction Type",
    "row2" : "Prediction Date",
    "row3" : "Score",
    "exit" : "Exit",
    "sel" : "Selection",

}

class BrowseComparisonView(util.ModalWindow):
    def __init__(self, parent, controller):
        self.controller = controller
        title = _text["title"].format(self.model.result.run_time.strftime(_text["dtfmt"]))
        super().__init__(parent, title, resize="wh")
        self.set_size_percentage(50, 60)
        util.stretchy_rows_cols(self, [1], [0])

    @property
    def model(self):
        return self.controller.model

    @property
    def hierarchical_view(self):
        return self._hview

    def add_choice_widgets(self):
        meta_frame = ttk.Frame(self)
        meta_frame.grid(row=0, column=0)
        frame = ttk.LabelFrame(meta_frame, text=_text["sel"])
        frame.grid(row=0, column=0)
        for row, text in enumerate([_text["key1"], _text["key2"], _text["key3"],
                _text["key4"], _text["key5"]]):
            ttk.Label(frame, text=text).grid(row=row, column=0, padx=2, sticky=tk.E)
        self._hview = hierarchical_view.HierarchicalView(self.model.hierarchical_model, None, frame)
        for row, f in enumerate(self._hview.frames):
            f.grid(row=row, column=1, padx=2, pady=2, sticky=tk.W)
        b = ttk.Button(meta_frame, text=_text["exit"], command=self.destroy)
        b.grid(row=0, column=1, sticky=tk.NSEW, padx=5, pady=5)

    def add_sheet(self):
        frame = ttk.Frame(self)
        frame.grid(row=1, column=0, sticky=tk.NSEW)
        util.stretchy_rows_cols(frame, [0], [0])
        self._sheet = simplesheet.SimpleSheet(frame)
        self._sheet.grid(row=0, column=0, sticky=tk.NSEW)
        self._sheet.set_columns([_text["row{}".format(i)] for i in [1,2,3]])
        self._sheet.xscrollbar(frame).grid(row=1, column=0, sticky=tk.EW)
        self._sheet.yscrollbar(frame).grid(row=0, column=1, sticky=tk.NS)
        for i in range(3):
            self._sheet.callback_to_column_heading(i, lambda ii=i: self.sort_column(ii))

    def add_widgets(self):
        self.add_choice_widgets()
        self.add_sheet()

    def sort_column(self, column):
        self._rows.sort(key = lambda row : row[column])
        self._repopulate_sheet()

    def _repopulate_sheet(self):
        for ri, row in enumerate(self._rows):
            for ci, entry in enumerate(row):
                self._sheet.set_entry(ri, ci, str(entry))

    def new_data(self, rows):
        self._sheet.remove_rows()
        self._rows = []
        for row in rows:
            self._rows.append(tuple(row))
            self._sheet.add_row()
        self._repopulate_sheet()
