"""
load_network
~~~~~~~~~~~~

Load network geometry for making network based predictions.
"""

import open_cp.gui.tk.load_network_view as load_network_view
import logging

_logger = logging.getLogger(__name__)

try:
    import geopandas as gpd
except Exception as ex:
    gpd = None
    _logger.error("geopandas not loaded because {}/{}".format(type(ex), ex))


class LoadNetwork():
    def __init__(self, parent, analysis_controller):
        self._parent = parent
        self._analysis_controller = analysis_controller
    
    def run(self):
        if gpd is None:
            load_network_view.alert("No geopandas loaded, so network support disabled.")
            return
        self.model.backup()
        self.view = load_network_view.LoadNetworkView(self._parent, self)
        self.view.wait_window(self.view)
        if not self.view.okay:
            self.model.restore()
        self.model.backup(False)
        
    @property
    def model(self):
        return self._analysis_controller.model.network_model

    def _display_error_if_needed(self):
        error = self.model.consume_recent_error()
        if error is not None:
            load_network_view.alert(error)

    def load(self, filename):
        def task():
            self.model.filename = filename
        view = load_network_view.FurtherWait(self._parent)
        view.run(task)
        self._display_error_if_needed()
        self.view.refresh()

    def reload(self):
        view = load_network_view.FurtherWait(self._parent)
        view.run(lambda : self.model.reload())
        self._display_error_if_needed()
        self.view.refresh()

    def remove(self):
        self.model.filename = None
        self.view.refresh()

    def new_epsg(self):
        """Use has entered a new epsg number."""
        view = load_network_view.FurtherWait(self._parent)
        view.run(lambda : self.model.reload())
        self._display_error_if_needed()
        self.view.refresh_projected()
