"""
comparitor
~~~~~~~~~

Base classes and utility methods for the GUI side of describing algorithms
which compare predictors (or describe the high-level tasks to make predictors:
e.g. make a new prediction for every day and score it on that day's actual
crime events).
"""

import open_cp.data

COMPARATOR_LOGGER_NAME = "__interactive_warning_logger__"

# Provides actual tasks
TYPE_TOP_LEVEL = 0
# Adjust the prediction in some way
TYPE_ADJUST = 50

class Comparitor():
    """Still a work in progress as I think about what we need to support:
    currently the business end of how to get "output" is not specified in this
    base class.
    
    :param model: An instance of :class:`analysis.Model` from which we can
      obtain data and settings.
    """
    def __init__(self, model):
        self._model = model

    @staticmethod
    def describe():
        """Return human readable short description of this comparitor."""
        raise NotImplementedError()

    @staticmethod
    def order():
        """An ordinal specifying the order, lowest is "first"."""
        raise NotImplementedError()

    def make_view(self, parent, inline=False):
        """Construct and return a view object.  This object is the model, and
        the controller may either be another object constructed here, or the
        model.
        
        :param parent: The parent `tk` object (typically another view)
        :param inline: If True, then if applicable, produce a more minimal view.
        """
        raise NotImplementedError()

    def config(self):
        """Optionally return a non-empty dictionary to specify extra options.

        - {"resize": True}  allow the edit view window to be resized.
        """
        return dict()

    @property
    def name(self):
        """Human readable giving the comparison method and perhaps headline
        settings."""
        raise NotImplementedError()

    @property
    def settings_string(self):
        """Human readable giving further settings.  May be `None`."""
        raise NotImplementedError()

    def to_dict(self):
        """Write state out to a dictionary for serialisation."""
        raise NotImplementedError()

    def from_dict(self, data):
        """Restore state from a dictionary."""
        raise NotImplementedError()

