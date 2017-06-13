"""
naive
~~~~~

Very simple-minded prediction techniques.  For testing and benchmarking.
"""

from . import predictor
import open_cp.naive


class CountingGrid(predictor.Predictor):
    def __init__(self):
        pass
    
    @staticmethod
    def describe():
        return "Counting Grid naive predictor"

    @staticmethod
    def order():
        return predictor._TYPE_PREDICTOR

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



class ScipyKDE(predictor.Predictor):
    def __init__(self):
        pass
    
    @staticmethod
    def describe():
        return "Scipy Kernel Density Estimator naive predictor"

    @staticmethod
    def order():
        return predictor._TYPE_PREDICTOR

    @staticmethod
    def make_view(self, parent):
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
    