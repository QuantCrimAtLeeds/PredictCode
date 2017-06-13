"""
about
~~~~~

Display credits etc.
"""

import open_cp.gui.tk.about_view as about_view

class About():
    def __init__(self, parent):
        self.view = about_view.AboutView(self, parent)

    def run(self):
        self.view.wait_window(self.view)
        