"""
sepp_base
~~~~~~~~~

A more abstract approach to SEPP type algorithms.

"""

from . import predictors
from . import logger as _ocp_logger
from . import data as _ocp_data
import numpy as _np
import datetime as _datetime
import logging as _logging
_logger = _logging.getLogger(__name__)

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


class FastModel():
    """An interface for a "fast" model."""
    def time_trigger(self, times):
        """Return the time kernel (and, by convention, the overall rate as
          well).

        :param times: Array of shape `(n,)` of times into the past.

        :return: Array of shape `(n,)` giving intensity at these times.
        """
        raise NotImplementedError()

    def space_trigger(self, space_points):
        """Return the space kernel (by convention, is a probability kernel).

        :param space_points: Array of shape `(2,n)` of space locations.

        :return: Array of shape `(n,)` giving intensity at these places.
        """
        raise NotImplementedError()

    def background_in_space(self, space_points):
        """Return the background risk, which is assumed not to vary in time.
        
        :param space_points: Array of shape `(2,n)` of space locations.

        :return: Array of shape `(n,)` giving intensity at these places.
        """
        raise NotImplementedError()


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
    
    def background_predict(self, time, space_points):
        """Find a point prediction at one time and one or more locations.
        Ignores triggers, and only uses the background intensity.
        
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
        return out        

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
            n = self.point_predict(t, space_points)
            out = out + n
        return out / samples

    def to_fast_split_predictor(self):
        """Return a new instance of a "predictor" which better performance if
        the model conforms to the interface :class:`FastModel`.
        """
        return FastPredictorBase(self._model)
        
    def to_fast_split_predictor_histogram(self, grid, time_bin_size=1, space_bin_size=25):
        """Return a new instance of a "predictor" which offers faster
        predictions by using approximations.

        Currently we assume the the trigger intensity does not vary with
        starting position, and that it "factors" into a product of a time
        kernel and a space kernel.  The model must conform to the
        :class:`FastModel` interface.  We also assume that the background
        intensity does not vary in time.

        :param time_bin_size: Size of bins for the histogram we use to
          approximate the time kernel.
        :param space_bin_size: Size of bins for the two dimensional histogram
          we use to approximate the space kernel.
        :param grid: The grid to base the background estimate on: for best
          results, this should be the same grid you will eventually make
          predictions for.
        """
        return FastPredictorHist(self._model,
            self._to_time_hist(time_bin_size), time_bin_size,
            self._to_space_grid(space_bin_size), self._to_background(grid))

    def _to_background(self, grid):
        cts_pred = predictors.KernelRiskPredictor(self._model.background_in_space)
        cts_pred.samples = -5
        return predictors.grid_prediction(cts_pred, grid)

    def _to_space_grid(self, space_bin_size):
        size = 5
        while True:
            d = size * space_bin_size
            region = _ocp_data.RectangularRegion(xmin=-d, ymin=-d, xmax=d, ymax=d)
            pred = predictors.grid_prediction_from_kernel(self._model.space_trigger,
                    region, space_bin_size, samples=-5)

            mat = pred.intensity_matrix
            sorted_mat = _np.sort(mat.flatten())
            cs = _np.cumsum(sorted_mat)
            if not _np.any(cs <= cs[-1]*.001):
                size += size
                continue
            sorted_index = _np.max(_np.where(cs <= cs[-1]*.001))
            cutoff = sorted_mat[sorted_index]

            mask = (pred.intensity_matrix <= cutoff)
            r = int(size*80/100)
            x = _np.broadcast_to(_np.arange(size*2)[:,None], (size*2, size*2))
            y = _np.broadcast_to(_np.arange(size*2)[None,:], (size*2, size*2))
            disc = _np.sqrt((x-size)**2 + (y-size)**2) >= r
            if _np.all(mask[disc]):
                return pred
            size += size

    def _to_time_hist(self, time_bin_size):
        size = 100
        while True:
            hist = self._model.time_trigger(_np.arange(size) * time_bin_size)
            sorted_hist = _np.sort(hist)
            cs = _np.cumsum(sorted_hist)
            if not _np.any(cs <= cs[-1]*.001):
                size += size
                continue
            sorted_index = _np.max(_np.where(cs <= cs[-1]*.001))
            cutoff = sorted_hist[sorted_index]
            mask = (hist <= cutoff)
            index_start = int(size * 80 / 100)
            if _np.all(mask[index_start:]):
                index_end = _np.max(_np.where(~mask))
                return hist[:index_end+1]
            size += size


class FastPredictorBase():
    """Base class which can perform fast "predictions" by assuming that the
    background rate does not vary in time, and that the trigger kernel factors.
    
    :param model: The :class:`FastModel` object to get the trigger and
      background from.
    """
    def __init__(self, model):
        self._model = model
        
    @property
    def model(self):
        """The model we base predictions on."""
        return self._model
    
    @property
    def points(self):
        """Points in the past we use as triggers."""
        return self._points

    @points.setter
    def points(self, v):
        self._points = v

    def time_kernel(self, times):
        return self._model.time_trigger(times)

    def space_kernel(self, pts):
        return self._model.space_trigger(pts)

    def background_kernel(self, pts):
        return self._model.background_in_space(pts)

    def range_predict(self, time_start, time_end, space_points, time_samples=5):
        space_points = _np.asarray(space_points)
        if len(space_points.shape) == 1:
            space_points = space_points[:,None]
        data = self._points[:,self._points[0] < time_start]

        tl = space_points.shape[-1] * data.shape[-1]
        pts = (space_points[:,:,None] - data[1:,None,:]).reshape((2,tl))
        space_triggers = self.space_kernel(pts).reshape(space_points.shape[-1], data.shape[-1])

        times = _np.linspace(time_start, time_end, time_samples)
        dtimes = (times[None,:] - data[0][:,None])
        time_triggers = self.time_kernel(dtimes.flatten()).reshape(dtimes.shape)
        time_triggers = _np.mean(time_triggers, axis=1)

        return self.background_kernel(space_points) + _np.sum(space_triggers * time_triggers[None,:], axis=1)


class FastPredictorHist(FastPredictorBase):
    """Base class which can perform fast "predictions", based on using
    histograms to approximate the kernels.
    
    :param model: The :class:`FastModel` object to get the trigger and
      background from.
    :param time_hist: Array of shape `(k,)` giving the time kernel.
    :param time_bandwidth: Width of each bin in the time histogram.
    :param space_grid: Instance of :class:`GridPredictionArray` to use as an
      approximation to the space kernel.
    :param background_grid: Instance of :class:`GridPredictionArray` to use as an
      approximation to the (time-invariant) background rate.
    """
    def __init__(self, model, time_hist, time_bandwidth, space_grid, background_grid):
        super().__init__(model)
        self._time = (time_hist, time_bandwidth)
        self._space_grid = space_grid
        self._background_grid = background_grid
        
    @property
    def time_histogram_width(self):
        """The width of each bar in the time histogram."""
        return self._time[1]

    @property
    def time_histogram(self):
        """An array giving the height of each bar in the time histogram."""
        return self._time[0]
    
    @property
    def space_grid(self):
        """The grid array we use for approximating the space kernel."""
        return self._space_grid

    def time_kernel(self, times):
        times = _np.atleast_1d(times)
        indices = _np.floor_divide(times, self._time[1]).astype(_np.int)
        m = indices < self._time[0].shape[0]
        out = _np.empty(times.shape)
        out[m] = self._time[0][indices[m]]
        out[~m] = 0
        return out

    def space_kernel(self, pts):
        return self._space_grid.risk(*pts)

    def background_kernel(self, pts):
        return self._background_grid.risk(*pts)


def non_normalised_p_matrix(model, points):
    d = points.shape[1]
    p = _np.zeros((d,d))
    progress = _ocp_logger.ProgressLogger(d * (d+1) / 2, _datetime.timedelta(seconds=10), _logger)
    p[_np.diag_indices(d)] = model.background(points)
    progress.add_to_count(d)
    for i in range(d):
        trigger_point = points[:,i]
        delta_points = trigger_point[:,None] - points[:, :i]
        m = delta_points[0] > 0
        p[:i, i][m] = model.trigger(trigger_point, delta_points[:,m])
        p[:i, i][~m] = 0
        progress.add_to_count(i)
    return p

def normalise_p(p):
    norm = _np.sum(p, axis=0)[None,:]
    if _np.any(norm==0):
        raise ValueError("Zero column in p matrix", p)
    return p / norm

def p_matrix(model, points):
    """Compute the normalised "p" matrix.
    
    :param model: Instance of :class:`ModelBase`
    :param points: Data
    """
    p = non_normalised_p_matrix(model, points)
    return normalise_p(p)

def clamp_p(p, cutoff = 99.9):
    """For each column, set entries beyond the `cutoff` percentile to 0.
    """
    pp = _np.array(p)
    for j in range(1, p.shape[1]):
        x = pp[:j+1,j]
        lookup = _np.argsort(x)
        s = x[lookup]
        c = _np.sum(_np.cumsum(s) < 1 - cutoff / 100)
        x[lookup[:c]] = 0
    return pp

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
            #raise ValueError()
            self._logger.warn("p-matrix has become diagonal-- no repeat behaviour!")
        return out
    
    def upper_tri_col(self, col):
        return self._p[:col, col]
    
    def diff_col_times(self, col):
        """`times[col] - times[:col]`"""
        return self._points[0, col] - self._points[0, :col]
    
    def diff_col_points(self, col):
        """`xypoints[col] - xypoints[:col]`"""
        return self._points[1:, col][:,None] - self._points[1:, :col]

    def sample(self):
        """Use the p-matrix to take a "sample", returning background events
        and triggered events.

        :return: Pair `(bk_indices, trigger_pairs)` where `bk_indices` are
          indices into :attr:`points` giving the sampled background events,
          and `trigger_pairs` is a list of pairs `(trigger, triggered)` where
          `trigger` is the trigger index, and `triggered` if the (later) index
          of the event which is triggered.
        """
        bk, tr = [], []
        for i in range(self.num_points):
            j = _np.random.choice(i+1, p=self.p[:i+1,i])
            if i==j:
                bk.append(i)
            else:
                tr.append((j,i))
        return bk, tr

    def sample_to_points(self):
        """Use the p-matrix to take a "sample", returning background events
        and triggered events.

        :return: Pair `(bk_points, trigger_deltas)` both arrays of points,
          `bk_points` being the background events, and `trigger_deltas` being
          the "jumps" from the triggering to the triggered events.
        """
        bk, tr = self.sample()
        bk = _np.array(bk, dtype=_np.int)
        bk_points = self._points[:, bk]
        trigger_deltas = [self._points[:,end] - self._points[:,start]
            for start, end in tr]
        return bk_points, _np.asarray(trigger_deltas).T

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
        
    def to_fast_split_predictor_histogram(self, time_bin_size=1, space_bin_size=25):
        """Return a new instance of a "predictor" which offers faster
        predictions by using approximations.

        Currently we assume the the trigger intensity does not vary with
        starting position, and that it "factors" into a product of a time
        kernel and a space kernel.  The model must conform to the
        :class:`FastModel` interface.  We also assume that the background
        intensity does not vary in time.

        :param time_bin_size: Size of bins for the histogram we use to
          approximate the time kernel.  In units of :attr:`time_unit`.
        :param space_bin_size: Size of bins for the two dimensional histogram
          we use to approximate the space kernel.
        """
        pred = PredictorBase(self._model, [])
        fsp = pred.to_fast_split_predictor_histogram(self._grid, time_bin_size, space_bin_size)
        return FastPredictor(self._grid, fsp)

    def to_fast_split_predictor(self):
        """Return a new instance of a "predictor" which offers faster
        predictions, assuming that the model conforms to the interface
        :class:`FastModel`.
        """
        pred = PredictorBase(self._model, [])
        return FastPredictor(self._grid, pred.to_fast_split_predictor())

    def background_continuous_predict(self, predict_time, space_samples=20):
        """Make a prediction at this time, returning a continuous prediction.
        
        :param predict_time: Limit to data before this time, and
          use this as the predict time.
        :param end_time: If not `None`, then approximately intergate
          over this time range.
         
        :return: A continuous prediction.
        """
        predict_time, for_fixed, data = self.make_data(predict_time)
        time = _np.max(for_fixed)
        pred = PredictorBase(self._model, data)

        def kernel(pts):
            return pred.background_predict(time, pts)
        
        return predictors.KernelRiskPredictor(kernel,
            xoffset=self._grid.xoffset, yoffset=self._grid.yoffset,
            cell_width=self._grid.xsize, cell_height=self._grid.ysize,
            samples=space_samples)

    def background_predict(self, predict_time, space_samples=20):
        """Make a prediction at this time.
        
        :param predict_time: Limit to data before this time, and
          use this as the predict time.
        :param end_time: If not `None`, then approximately intergate
          over this time range.
         
        :return: A grid prediction, masked if possible with the grid, and
          normalised.
        """
        cts_predictor = self.background_continuous_predict(predict_time, space_samples)
        return self._to_grid_pred(cts_predictor)

    def continuous_predict(self, predict_time, end_time=None, time_samples=20, space_samples=20):
        """Make a prediction at this time, returning a continuous prediction.
        
        :param predict_time: Limit to data before this time, and
          use this as the predict time.
        :param end_time: If not `None`, then approximately intergate
          over this time range.
         
        :return: A continuous prediction.
        """
        predict_time, for_fixed, data = self.make_data(predict_time)
        time = _np.max(for_fixed)
        pred = PredictorBase(self._model, data)

        if end_time is None:
            def kernel(pts):
                return pred.point_predict(time, pts)
        else:
            end_time = _np.datetime64(end_time)
            time_end = time + (end_time - predict_time) / self.time_unit
            def kernel(pts):
                return pred.range_predict(time, time_end, pts, samples=time_samples)
        
        return predictors.KernelRiskPredictor(kernel,
            xoffset=self._grid.xoffset, yoffset=self._grid.yoffset,
            cell_width=self._grid.xsize, cell_height=self._grid.ysize,
            samples=space_samples)

    def predict(self, predict_time, end_time=None, time_samples=20, space_samples=20):
        """Make a prediction at this time.
        
        :param predict_time: Limit to data before this time, and
          use this as the predict time.
        :param end_time: If not `None`, then approximately intergate
          over this time range.
         
        :return: A grid prediction, masked if possible with the grid, and
          normalised.
        """
        cts_predictor = self.continuous_predict(predict_time, end_time, time_samples, space_samples)
        return self._to_grid_pred(cts_predictor)

    def _to_grid_pred(self, cts_predictor):
        grid_pred = predictors.GridPredictionArray.from_continuous_prediction_grid(
                cts_predictor, self._grid)
        try:
            grid_pred.mask_with(self._grid)
        except:
            pass
        return grid_pred.renormalise()


class FastPredictor(_BaseTrainer):
    """A :class:`DataTrainer` which uses a model to make predictions.
    Is optimised for certain classes of models and can optionally also
    approximate kernels by histograms.

    Currently we assume the the trigger intensity does not vary with
    starting position, and that it "factors" into a product of a time
    kernel and a space kernel.  The model must conform to the
    :class:`FastModel` interface.

    :param grid: The Grid object to make predictions against.
    :param fast_pred_base: The instance of :class:`FastPredictorBase`
      we'll use internally.
    """
    def __init__(self, grid, fast_pred_base):
        super().__init__()
        self._grid = grid
        self._fast_pred_base = fast_pred_base

    @property
    def fast_predictor_base(self):
        """The underlying :class:`FastPredictorBase` which is used."""
        return self._fast_pred_base
    
    def background_predict(self, space_samples=-5):
        cts_predictor = predictors.KernelRiskPredictor(self._fast_pred_base.background_kernel,
            xoffset=self._grid.xoffset, yoffset=self._grid.yoffset,
            cell_width=self._grid.xsize, cell_height=self._grid.ysize,
            samples=space_samples)
        return self._to_grid_pred(cts_predictor)

    def continuous_predict(self, predict_time, end_time, time_samples=5, space_samples=-5):
        """Make a prediction at this time, returning a continuous prediction.
        
        :param predict_time: Limit to data before this time, and use this as
          the predict time.
        :param end_time: Approximately intergate over this time range.
        :param time_samples: The number of samples to use in approximating the
          integral over time.
        :param space_samples: The number of samples to use in the monte-carlo
          integration over space
         
        :return: A continuous prediction.
        """
        predict_time, for_fixed, data = self.make_data(predict_time)
        time = _np.max(for_fixed)
        self._fast_pred_base.points = data

        end_time = _np.datetime64(end_time)
        time_end = time + (end_time - predict_time) / self.time_unit
        def kernel(pts):
            return self._fast_pred_base.range_predict(time, time_end, pts, time_samples=time_samples)
        
        return predictors.KernelRiskPredictor(kernel,
            xoffset=self._grid.xoffset, yoffset=self._grid.yoffset,
            cell_width=self._grid.xsize, cell_height=self._grid.ysize,
            samples=space_samples)

    def predict(self, predict_time, end_time, time_samples=5, space_samples=-5):
        """Make a prediction at this time.
        
        :param predict_time: Limit to data before this time, and
          use this as the predict time.
        :param end_time: Approximately intergate over this time range.
        :param time_samples: The number of samples to use in approximating the
          integral over time.
        :param space_samples: The number of samples to use in the monte-carlo
          integration over space
         
        :return: A grid prediction, masked if possible with the grid, and
          normalised.
        """
        cts_predictor = self.continuous_predict(predict_time, end_time,
                                    time_samples, space_samples)
        return self._to_grid_pred(cts_predictor)

    def _to_grid_pred(self, cts_predictor):
        grid_pred = predictors.GridPredictionArray.from_continuous_prediction_grid(
                cts_predictor, self._grid)
        try:
            grid_pred.mask_with(self._grid)
        except:
            pass
        return grid_pred.renormalise()


