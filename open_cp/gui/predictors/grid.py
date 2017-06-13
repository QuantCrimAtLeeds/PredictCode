"""
grid
~~~~

Produce a grid over the input data
"""

from . import predictor

class GridProvider(predictor.Predictor):
    def __init__(self):
        pass

    @staticmethod
    def describe():
        return "Produce a grid over the data"

    @staticmethod
    def order():
        return 10

    @staticmethod
    def make_view(parent):
        """Construct and return a view object.  This object is the model, and
        the controller may either be another object constructed here, or the
        model."""
        raise NotImplementedError()

    @property
    def name(self):
        """Human readable giving the prediction method and perhaps headline
        settings."""
        raise NotImplementedError()

    @property
    def settings_string(self):
        """Human readable giving further settings.  May be `None`."""
        raise NotImplementedError()

    def make_tasks(self):
        raise NotImplementedError()
