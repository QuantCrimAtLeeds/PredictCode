"""
load_network
~~~~~~~~~~~~

Load network geometry for making network based predictions.
"""

import open_cp.gui.tk.load_network_view as load_network_view

class LoadNetwork():
    def __init__(self, parent, analysis_controller):
        self._parent = parent
        self._analysis_controller = analysis_controller
    
    def run(self):
        self.view = load_network_view.LoadNetworkView(self._parent, self)
        self.view.wait_window(self.view)
        print("Got:", self.view.okay)
        # TODO: Stuff
        
    @property
    def model(self):
        return self._analysis_controller.model.network_model
