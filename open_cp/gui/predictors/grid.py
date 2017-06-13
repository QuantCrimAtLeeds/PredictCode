"""
grid
~~~~

Produce a grid over the input data
"""

from . import predictor

class GridProvider(predictor.Predictor):
    def __init__(self, model):
        super().__init__(model)
        self._grid_size = 100
        self._xoffset = 0
        self._yoffset = 0

    @staticmethod
    def describe():
        return "Produce a grid over the data"

    @staticmethod
    def order():
        return 10

    @property
    def name(self):
        return "Grid {}x{}m @ ({}m, {}m)".format(self._grid_size, self._grid_size, self._xoffset, self._yoffset)

    @property
    def settings_string(self):
        return None

    def make_view(parent):
        """Construct and return a view object.  This object is the model, and
        the controller may either be another object constructed here, or the
        model."""
        raise NotImplementedError()

    def to_dict(self):
        """Write state out to a dictionary for serialisation."""
        raise NotImplementedError()

    def from_dict(self, data):
        """Restore state from a dictionary."""
        raise NotImplementedError()

    def make_tasks(self):
        raise NotImplementedError()
