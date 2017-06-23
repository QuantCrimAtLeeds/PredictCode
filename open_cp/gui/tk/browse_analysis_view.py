"""
browse_analysis_view
~~~~~~~~~~~~~~~~~~~~

"""

import tkinter as tk
import tkinter.ttk as ttk
import open_cp.gui.tk.util as util

_text = {
    "title" : "Previous Analysis results.  Run @ {}",
    "dtfmt" : "%d %b %Y %H:%M",
    "cp" : "Coordinate projection:",
    "grid" : "Grid system used:",
    
}

class BrowseAnalysisView(util.ModalWindow):
    def __init__(self, parent, controller):
        self.controller = controller
        title = _text["title"].format(self.controller.model.result.run_time.strftime(_text["dtfmt"]))
        super().__init__(parent, title, resize="wh")

    def add_widgets(self):
        ttk.Label(self, text=_text["cp"]).grid(row=0, column=0, padx=2, pady=2)
        ttk.Label(self, text=_text["grid"]).grid(row=1, column=0, padx=2, pady=2)
        pass
    
    def _cbox_or_label(self, choices, command=None):
        """Produces a :class:`ttk.Combobox` unless `choices` is of length 1,
        in which case just produces a label.

        :return: Pair of `(widget, flag)` where `flag` is True if and only if
          we produced a box.
        """
        if len(choices) == 1:
            p = choices[0]
            label = ttk.Label(self, text=str(p))
            return label, False
        else:
            cbox = ttk.Combobox(self, height=5, state="readonly")
            cbox["values"] = choices
            cbox.bind("<<ComboboxSelected>>", command)
            cbox.current(0)
            cbox["width"] = max(len(t) for t in choices)
            return cbox, True

    def update_projections(self):
        w, flag = self._cbox_or_label(self.controller.model.projections, command=self._proj_chosen)
        w.grid(row=0, column=1, padx=2, pady=2)
        if flag:
            self._proj_cbox = w
        else:
            self._proj_cbox = None
        self._proj_choice = 0
        self.controller.notify_projection_choice(0)

    def _proj_chosen(self, event):
        self._proj_choice = event.widget.current()
        self.controller.notify_projection_choice(self._proj_choice)

    @property
    def projection_choice(self):
        """Pair of (index, string_value)"""
        if self._proj_cbox is None:
            return 0, self.controller.model.projections[0]
        return self._proj_choice, self._proj_cbox["values"][self._proj_choice]

    def update_grids(self, choices):
        self._grid_choices = choices
        w, flag = self._cbox_or_label(choices, command=self._grid_chosen)
        w.grid(row=1, column=1, padx=2, pady=2)
        if flag:
            self._grid_cbox = w
        else:
            self._grid_cbox = None
        
    def _grid_chosen(self, event):
        self._grid_choice = event.widget.current()
        self.controller.notify_grid_choice(self._grid_choice)

    @property
    def grid_choice(self):
        """Pair of (index, string_value)"""
        if self._grid_cbox is None:
            return 0, self._grid_choices[0]
        return self._grid_choice, self._grid_cbox["values"][self._grid_choice]
        

