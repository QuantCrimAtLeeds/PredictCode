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

}

class BrowseAnalysisView(util.ModalWindow):
    def __init__(self, parent, controller):
        self.controller = controller
        title = _text["title"].format(self.controller.model.result.run_time.strftime(_text["dtfmt"]))
        super().__init__(parent, title, resize="wh")

    def add_widgets(self):
        projections = self.controller.model.projections
        if len(projections) == 1:
            p = projections[0]
            self._proj_choice = 0
            ttk.Label(self, text=str(p)).grid(row=0, column=0, padx=2, pady=2)
            self._proj_cbox = None
        else:
            self._proj_cbox = ttk.Combobox(self, height=5, state="readonly")
            self._proj_cbox["values"] = projections
            self._proj_cbox.bind("<<ComboboxSelected>>", self._proj_chosen)
            self._proj_choice = 0
            self._proj_cbox.current(0)
            self._proj_cbox.grid(row=0, column=0, padx=2, pady=2)
            self._proj_cbox["width"] = max(len(t) for t in projections)
        self.controller.notify_projection_choice(0)

    def update_grids(self):
        self.controller.model.gri

    def _proj_chosen(self, event):
        self._proj_choice = event.widget.current()
        self.controller.notify_projection_choice(self._proj_choice)

    @property
    def projection_choice(self):
        """Pair of (index, string_value)"""
        if self._proj_cbox is None:
            return 0, self.controller.model.projections[0]
        return self._proj_choice, self._proj_cbox["values"][self._proj_choice]
