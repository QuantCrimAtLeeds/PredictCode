"""
sepp_base
~~~~~~~~~

A more abstract approach to SEPP type algorithms.

"""

from . import predictors
#from . import kernels
import numpy as _np
import logging as _logging

class ModelBase():
    """Interface for a "model".
    
    We use the convention that the data is always an array of shape `(3,N)`
    formed of `[times, x, y]` where `times` is an increasing sequence of
    numbers from 0.
    """
    
    def background(self, points):
        """Evaluate the background kernel at `points`.  If `points is of
        shape `(3,N)` then should return an array of shape `(N,)`.
        
        :return: Array of shape `(N,)`
        """
        raise NotImplementedError()
        
    def trigger(self, trigger_point, delta_points):
        """We allow quite general trigger risk kernels which can depend on the
        "trigger point" as well as the delta between the trigger and triggered
        events.
        
        :param trigger_point: Array of shape `(3,)` specifying the `(t, x, y)`
          coords of the (single) trigger event.
        :param delta_points: Array of shape `(3,m)` specifying the deltas to
          the triggered events.  Add to `trigger_point` to get the absolute
          location of the triggered events.
          
        :return: Array of shape `(m,)`
        """
        raise NotImplementedError()
        
    def log_likelihood_base(self, points):
        """Computes the non-normalised log likelihood,
        :math:`\sum_{i=1}^n \log \lambda^*(t_i,x_i,y_i)`.
        The normalisation requires integrating which is best left to a concrete
        subclass.
        """
        points = _np.asarray(points)
        out = 0.0
        for i in range(points.shape[1]):
            pt = points[:,i]
            ptt = pt[:,None]
            li = self.background(ptt)[0]
            deltas = ptt - points[:,:i]
            li += _np.sum(self.trigger(pt, deltas))
            out += _np.log(li)
        return out
        

class PredictorBase():
    """Base class which can perform "predictions".  Predictions are formed by
    evaluating the intensity (background and triggers) at one or more time
    points and averaging.
    
    :param model: The :class:`ModelBase` object to get the trigger and
      background from.
    :param points: Usual array of shape `(3,N)`
    """
    def __init__(self, model, points):
        self._model = model
        self._points = _np.asarray(points)
        
    @property
    def model(self):
        return self._model
    
    @property
    def points(self):
        return self._points
    
    def point_predict(self, time, space_points):
        """Find a point prediction at one time and one or more locations.
        The data the class holds will be clipped to be before `time` and
        the used as the trigger events.
        
        :param time: Time point to evaluate at
        :param space_points: Array of shape `(2,n)`
        
        :return: Array of shape `(n,)`
        """
        space_points = _np.asarray(space_points)
        if len(space_points.shape) == 1:
            space_points = space_points[:,None]
        eval_points = _np.asarray([[time] * space_points.shape[1],
                                   space_points[0], space_points[1]])
        out = self._model.background(eval_points)
        data = self._points[:,self._points[0] < time]
        for i, pt in enumerate(eval_points.T):
            out[i] += _np.sum(self._model.trigger(pt, pt[:,None] - data))
        return out
    
    def range_predict(self, time_start, time_end, space_points, samples=20):
        if not time_start < time_end:
            raise ValueError()
        out = self.point_predict(time_start, space_points)
        for i in range(1, samples):
            t = time_start + (time_end - time_start) * i / (samples - 1)
            print(t, out)
            n = self.point_predict(t, space_points)
            out = out + n
        return out / samples
    

def non_normalised_p_matrix(model, points):
    d = points.shape[1]
    p = _np.zeros((d,d))
    p[_np.diag_indices(d)] = model.background(points)
    for i in range(d):
        trigger_point = points[:,i]
        delta_points = trigger_point[:,None] - points[:, :i]
        m = delta_points[0] > 0
        p[:i, i][m] = model.trigger(trigger_point, delta_points[:,m])
        p[:i, i][~m] = 0
    return p

def p_matrix(model, points):
    """Compute the normalised "p" matrix.
    
    :param model: Instance of :class:`ModelBase`
    :param points: Data
    """
    p = non_normalised_p_matrix(model, points)
    return p / _np.sum(p, axis=0)[None,:]


class Optimiser():
    """We cannot know all models and how to optimise them, but we provide some
    helper routines."""
    def __init__(self, model, points, make_p=True):
        self._logger = _logging.getLogger(__name__)
        self._model = model
        self._points = points
        if make_p:
            self._p = _np.asarray( p_matrix(model, points) )
            if _np.any(self._p < 0):
                raise ValueError("p should ve +ve")
        
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


class _BaseTrainer(predictors.DataTrainer):
    def __init__(self):
        self.time_unit = _np.timedelta64(1, "D")
        self._logger = _logging.getLogger(__name__)
    
    @property
    def time_unit(self):
        """The unit of time to use to convert real timestamps into abstract
        timestamps."""
        return self._time_unit

    @time_unit.setter
    def time_unit(self, v):
        self._time_unit = _np.timedelta64(v)

    def make_data(self, predict_time=None):
        """Internal method, and for testing.  Returns the data in the format
        expected by the base classes.
        
        :param predict_time: Crop the data to before this time, and use this time
          as the end point.  If `None` then use the final timestamp in the
          data, rounded up by the currently in use time unit.
        
        :return: `predict_time, for_fixed, data`
        """
        if predict_time is None:
            offset = _np.datetime64("2000-01-01T00:00")
            x = self.data.timestamps[-1] - offset
            x = _np.ceil(x / self.time_unit) * self.time_unit
            predict_time = offset + x
        else:
            predict_time = _np.datetime64(predict_time)
        data = self.data[self.data.timestamps <= predict_time]
        times = (data.timestamps - data.timestamps[0]) / self.time_unit
        for_fixed = (predict_time - data.timestamps) / self.time_unit
        data = _np.asarray([times, data.xcoords, data.ycoords])
        return predict_time, for_fixed, data


class Trainer(_BaseTrainer):
    """Base class for a standard "trainer".  It is not assumed that this will
    always be used; but it may prove helpful often.
    
    """
    def __init__(self):
        super().__init__()

    def make_data(self, predict_time=None):
        """Internal method, and for testing.  Returns the data in the format
        expected by the base classes.
        
        :param predict_time: As in :meth:`train`.
        
        :return: `(fixed, data)` where `fixed` is a class describing any
          "fixed" parameters of the model (meta-parameters if you like) and
          `data` is an array of shape `(3,N)`.
        """
        predict_time, for_fixed, data = super().make_data(predict_time)
        return self.make_fixed(for_fixed), data

    def make_fixed(self, times):
        """Abstract method to return the "fixed" model.
        
        :param times: An array of the timestamps, converted to units of time
          before the "predict point".
        """
        raise NotImplementedError()

    def initial_model(self, fixed, data):
        """Abstract method to return the initial model from which optimisation
          is performed.  The pair `(fixed, data)` is as returned by
          :meth:`make_data`.
        """
        raise NotImplementedError()
        
    @property
    def _optimiser(self):
        """The class to be used as the optimiser"""
        raise NotImplementedError()
        
    def train(self, predict_time=None, iterations=1):
        """Optimise the model.
        
        :predict_time: Crop the data to before this time, and use this time
          as the end point.  If `None` then use the final timestamp in the
          data, rounded up by the currently in use time unit.
        
        :return: Instances of :class:`Model`.
        """
        fixed, data = self.make_data(predict_time)        
        model = self.initial_model(fixed, data)
        for _ in range(iterations):
            opt = self._optimiser(model, data)
            model = opt.iterate()
            self._logger.debug(model)
        return model
    
    
class Predictor(_BaseTrainer):
    """A :class:`DataTrainer` which uses a model to make predictions.
    
    :param grid: The Grid object to make predictions against.
    :param model: The model object to use.
    """
    def __init__(self, grid, model):
        super().__init__()
        self._grid = grid
        self._model = model
        
    def predict(self, predict_time, end_time=None, time_samples=20, space_samples=20):
        """Make a prediction at this time.
        
        :param predict_time: Limit to data before this time, and
          use this as the predict time.
        :param end_time: If not `None`, then approximately intergate
          over this time range.
         
        :return: A grid prediction, masked if possible with the grid, and
          normalised.
        """
        predict_time, for_fixed, data = self.make_data(predict_time)
        time = _np.max(for_fixed)
        pred = PredictorBase(self._model, data)

        if end_time is None:
            def kernel(pts):
                return pred.point_predict(time, pts)
        else:
            time_end = time + (end_time - predict_time) / self.time_unit
            def kernel(pts):
                return pred.range_predict(time, time_end, pts, samples=time_samples)
        
        cts_predictor = predictors.KernelRiskPredictor(kernel,
            xoffset=self._grid.xoffset, yoffset=self._grid.yoffset,
            cell_width=self._grid.xsize, cell_height=self._grid.ysize,
            samples=space_samples)

        grid_pred = predictors.GridPredictionArray.from_continuous_prediction_grid(
                    cts_predictor, self._grid)
        try:
            grid_pred.mask_with(self._grid)
        except:
            pass
        return grid_pred.renormalise()
