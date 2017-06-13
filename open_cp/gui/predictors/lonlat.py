"""
lonlat
~~~~~~

Transform lon/lat coords into meters
"""

from . import predictor

class LonLatConverter(predictor.Predictor):
    def __init__(self):
        pass

    @staticmethod
    def describe():
        return "Project LonLat coordinates to meters"

    @staticmethod
    def order():
        return 0

    @staticmethod
    def make_view(parent):
        """Construct and return a view object.  This object is the model, and
        the controller may either be another object constructed here, or the
        model."""
        raise NotImplementedError()

    @property
    def name(self):
        return "Project longitude/latitude coordinates to meters."

    @property
    def settings_string(self):
        return "<Builtin best guess>"

    def make_tasks(self):
        raise NotImplementedError()
