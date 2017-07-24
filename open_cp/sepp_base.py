"""
sepp_base
~~~~~~~~~

A more abstract approach to SEPP type algorithms.

"""

#from . import predictors
#from . import kernels
import numpy as _np
#import logging as _logging


class ModelBase():
    """Interface for a "model".
    
    We use the convention that the data is always an array of shape `(3,N)`
    formed of `[times, x, y]` where `times` is an increasing sequence of
    numbers from 0.
    """
    
    def background(self, points):
        """Evaluate the background kernel at `points`.  If `points is of
        shape `(3,N)` then should return an array of shape `(N,)`"""
        raise NotImplementedError()
        
    def trigger(self, trigger_point, delta_points):
        """We allow quite general trigger risk kernels which can depend on the
        "trigger point" as well as the delta between the trigger and triggered
        events.
        
        :param trigger_point: Array of shape `(3,)` specifying the `(t, x, y)`
          coords of the (single) trigger event.
        :param delta_points: Array of shape `(3,?)` specifying the deltas to
          the triggered events.  Add to `trigger_point` to get the absolute
          location of the triggered events.
        """
        raise NotImplementedError()
        

def p_matrix(model, points):
    """Compute the normalised "p" matrix.
    
    :param model: Instance of :class:`ModelBase`
    :param points: Data
    """
    d = points.shape[1]
    p = _np.zeros((d,d))
    p[_np.diag_indices(d)] = model.background(points)
    for i in range(d):
        trigger_point = points[:,i]
        delta_points = trigger_point[:,None] - points[:, :i]
        p[:i, i] = model.trigger(trigger_point, delta_points)
    return p / _np.sum(p, axis=0)[None,:]


class Optimiser():
    """We cannot know all models and how to optimise them, but we provide some
    helper routines."""
    def __init__(self, model, points):
        self._model = model
        self._points = points
        self._p = _np.asarray( p_matrix(model, points) )
        
    @property
    def p(self):
        """The p matrix"""
        return self._p
    
    @property
    def model(self):
        return self._model
    
    @property
    def points(self):
        return self._points
    
    @property
    def num_points(self):
        return self._points.shape[1]
    
    @property
    def p_diag(self):
        """The diagonal of the p matrix."""
        d = self._points.shape[1]
        return self._p[_np.diag_indices(d)]
    
    @property
    def p_diag_sum(self):
        return _np.sum(self.p_diag)
    
    @property
    def p_upper_tri_sum(self):
        out = 0.0
        for i in range(1, self._p.shape[0]):
            out += _np.sum(self._p[:i, i])
        if abs(out) < 1e-10:
            raise ValueError()
        return out
    
    def upper_tri_col(self, col):
        return self._p[:col, col]
    
    def diff_col_times(self, col):
        """`times[col] - times[:col]`"""
        return self._points[0, col] - self._points[0, :col]
    
    def diff_col_points(self, col):
        """`xypoints[col] - xypoints[:col]`"""
        return self._points[1:, col][:,None] - self._points[1:, :col]

    def iterate(self):
        """Abstract method to be over-riden.  Should return a new `model`."""
        raise NotImplementedError()
