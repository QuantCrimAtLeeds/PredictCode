"""
comparitor
~~~~~~~~~~

Base classes and utility methods for the GUI side of describing algorithms
which compare predictors (or describe the high-level tasks to make predictors:
e.g. make a new prediction for every day and score it on that day's actual
crime events).
"""

COMPARATOR_LOGGER_NAME = "__interactive_warning_logger__"

# Provides actual tasks
# `run()` method returns list `(start_date, score_duration)`
TYPE_TOP_LEVEL = 0

# Adjust the prediction in some way
# `make_tasks()` method returns :class:`AdjustTask` object(s)
TYPE_ADJUST = 50

# Compares the prediction to actual events
# `make_tasks()` method returns :class:`CompareRealTask` object(s)
TYPE_COMPARE_TO_REAL = 100

class CompareRealTask():
    """Compares predictions against actual events."""

    def __call__(self, grid_prediction, timed_points, predict_date, predict_length):
        """Compare the prediction to what actually happened.

        :param grid_prediction: The predicted "risk"
        :param timed_points: The actual data
        :param predict_date: The date the prediction is for
        :param predict_length: The length of time the prediction is meant to be valid for
        """
        raise NotImplementedError()
        

class AdjustTask():
    """An `adjust` type of comparator should return one or more of these tasks."""
    
    def __call__(self, projector, grid_prediction):
        """Process the grid based prediction in some way.  For efficiency,
        should also allow passing of a list of predictions.
        
        :param projector: The projector task used in to make this prediction.
          May be `None`.  Parameter can be ignored if it is not relevant to
          this task.
        :param grid_prediction: The prediction to base the new prediction on
          (or an iterable).
        
        :return: A _new instance_ of a prediction, suitably modified.
        """
        raise NotImplementedError()


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

    def pprint(self):
        settings = self.settings_string
        if settings is not None:
            return self.name + " : " + settings
        return self.name

    def to_dict(self):
        """Write state out to a dictionary for serialisation."""
        raise NotImplementedError()

    def from_dict(self, data):
        """Restore state from a dictionary."""
        raise NotImplementedError()
        