"""
sepp
~~~~

Implements the ETAS (Epidemic Type Aftershock-Sequences) model intensity
estimation scheme outlined in Mohler et al. (2011).

References
~~~~~~~~~~
Mohler et al, "Self-Exciting Point Process Modeling of Crime",
   Journal of the American Statistical Association, 2011
   DOI: 10.1198/jasa.2011.ap09546

Rosser, Cheng, "Improving the Robustness and Accuracy of Crime Prediction with
the Self-Exciting Point Process Through Isotropic Triggering"
   Appl. Spatial Analysis
   DOI 10.1007/s12061-016-9198-y
"""

from . import predictors
from . import kernels
import numpy as _np

def _normalise_matrix(p):
    column_sums = _np.sum(p, axis=0)
    return p / column_sums[None,:]

def p_matrix(points, background_kernel, trigger_kernel):
    """Computes the probability matrix.

    :param points: The (time, x, y) data
    :param background_kernel: The kernel giving the background event intensity.
    :param trigger_kernel: The kernel giving the triggered event intensity.

    :return: A matrix `p` such that `p[i][i]` is the probability event `i` is a
    background event, and `p[i][j]` is the probability event `j` is triggered
    by event `i`.
    """

    number_data_points = points.shape[-1]
    p = _np.zeros((number_data_points, number_data_points))
    for j in range(1, number_data_points):
        d = points[:, j][:,None] - points[:, :j]
        p[0:j, j] = trigger_kernel(d)
    p += _np.diag(background_kernel(points))
    return _normalise_matrix(p)

def p_matrix_fast(points, background_kernel, trigger_kernel, time_cutoff=150, space_cutoff=1):
    """Computes the probability matrix.  Offers faster execution speed than
    :function:`p_matrix` by, in the calculation of triggered event
    probabilities, ignoring events which are beyond a space or time cutoff.
    These parameters should be set so that the `trigger_kernel` evaluates to
    (very close to) zero outside the cutoff zone.

    :param points: The (time, x, y) data
    :param background_kernel: The kernel giving the background event intensity.
    :param trigger_kernel: The kernel giving the triggered event intensity.
    :param time_cutoff: The maximum time between two events which can be
    considered in the trigging calculation.
    :param space_cutoff: The maximum (two-dimensional Eucliean) distance
    between two events which can be considered in the trigging calculation.

    :return: A matrix `p` such that `p[i][i]` is the probability event `i` is a
    background event, and `p[i][j]` is the probability event `j` is triggered
    by event `i`.
    """

    number_data_points = points.shape[-1]
    p = _np.zeros((number_data_points, number_data_points))
    space_cutoff *= space_cutoff
    for j in range(1, number_data_points):
        d = points[:, j][:,None] - points[:, :j]
        dmask = (d[0] <= time_cutoff) & ((d[1]**2 + d[2]**2) < space_cutoff)
        d = d[:, dmask]
        if d.shape[-1] == 0:
            continue
        p[0:j, j][dmask] = trigger_kernel(d)
    p += _np.diag(background_kernel(points))
    return _normalise_matrix(p)

def initial_p_matrix(points, initial_time_bandwidth = 0.1,
        initial_space_bandwidth = 50.0):
    """Returns an initial estimate of the probability matrix.  Uses a Gaussian
    kernel in space, and an exponential kernel in    time, both non-normalised.
    Diagonal (i.e. background "probabilities") are set to 1.  Finally the
    matrix is normalised.

    :param points: The (time, x, y) data.
    :param initial_time_bandwidth: The "scale" of the exponential.
    :param initial_space_bandwidth: The standard deviation of the Gaussian.
    """

    def bkernel(pts):
        return _np.zeros(pts.shape[-1]) + 1
    def tkernel(pts):
        time = _np.exp( - pts[0] / initial_time_bandwidth )
        norm = 2 * initial_space_bandwidth ** 2
        space = _np.exp( - (pts[1]**2 + pts[2]**2) / norm )
        return time * space
    return p_matrix(points, bkernel, tkernel)

def sample_points(points, p):
    """Using the probability matrix, sample background and triggered points.

    :param points: The (time, x, y) data.
    :param p: The probability matrix.

    :return: A pair of `(backgrounds, triggered)` where `backgrounds` is the
    (time, x, y) data of the points classified as being background events,
    and `triggered` is the (time, x, y) *delta* of the triggered events.
    That is, `triggered` represents the difference in space and time between
    each triggered event and the event which triggered it, as sampled from the
    probability matrix.
    """

    number_data_points = points.shape[-1]
    choice = _np.array([ _np.random.choice(j+1, p=p[0:j+1, j])
        for j in range(number_data_points) ])
    mask = ( choice == _np.arange(number_data_points) )
    
    backgrounds = points[:,mask]
    triggered = (points - points[:,choice])[:,~mask]
    return backgrounds, triggered


class StocasticDecluster():
    """Implements the 'stocastic declustering algorithm' from Mohler et al
    (2011).  This allows estimation of two time-space kernels, one for the
    background events, and one the 'trigger' kernel which elevates risk
    according to past events.
    
    This class works with floating-point data, and exposes elements of the
    underlying optimisation algorithm.  It is designed for testing and
    experimentation.

    :param background_kernel_estimator: The kernel estimator to use for
    background events.
    :param trigger_kernel_estimator: The kernel estimator to use for triggered
    / aftershock events.
    :param initial_time_bandwidth: The bandwidth in time to use when making an
    initial classification of data into background or triggered events.  Default
    is 0.1 day**(-1) in units of minutes (so 0.1*24*60).
    :param initial_space_bandwidth: The bandwidth in space to use when making
    an initial classification of data into background or triggered events.
    Default is 50 units.
    :param space_cutoff: The maximum distance we believe the triggered kernel
    will extend to in space.  Decrease this to improve the speed of the
    estimation, at the cost of possibly missing data.  Default is 500 units.
    :param time_cutoff: The maximum distance we believe the triggered kernel
    will extend to in time.  Decrease this to improve the speed of the
    estimation, at the cost of possibly missing data.  Default is 120 days,
    in units of minutes (so 120*24*60).
    :param points: The three dimensional data.  `points[0]` is the times of
    events, and `points[1]` and `points[2]` are the x and y coordinates.
    """

    def __init__(self, background_kernel_estimator = None,
            trigger_kernel_estimator = None,
            initial_time_bandwidth = 0.1 * (_np.timedelta64(1, "D") / _np.timedelta64(1, "m")),
            initial_space_bandwidth = 50.0,
            space_cutoff = 500.0,
            time_cutoff = 120 * (_np.timedelta64(1, "D") / _np.timedelta64(1, "m")),
            points = None
            ):
        self.background_kernel_estimator = background_kernel_estimator
        self.trigger_kernel_estimator = trigger_kernel_estimator
        self.initial_time_bandwidth = initial_time_bandwidth
        self.initial_space_bandwidth = initial_space_bandwidth
        self.space_cutoff = space_cutoff
        self.time_cutoff = time_cutoff
        self.points = points

    def next_iteration(self, p):
        """Perform a single iteration of the optimisation algorithm:
        1. Samples background and triggered events using the p matrix.
        2. Estimates kernels from these samples.
        3. Normalises these kernels.
        4. Computes the new p matrix from these kernels.

        :param p: The matrix of probabilities to sample from.

        :return: A triple `(p, bkernel, tkernel)` where `p` is the new
        probability matrix, `bkernel` the kernel for background events used to
        compute `p`, and `tkernel` the kernel for triggered events.
        """
        
        backgrounds, triggered = sample_points(self.points, p)
        bkernel = self.background_kernel_estimator(backgrounds)
        tkernel = self.trigger_kernel_estimator(triggered)

        number_events = self.points.shape[-1]
        number_background_events = backgrounds.shape[-1]
        number_triggered_events = number_events - number_background_events
        norm_tkernel = lambda pts : ( tkernel(pts) * number_triggered_events / number_events )
        norm_bkernel = lambda pts : ( bkernel(pts) * number_background_events )
        pnew = p_matrix_fast(self.points, norm_bkernel, norm_tkernel,
            time_cutoff = self.time_cutoff, space_cutoff = self.space_cutoff)
        return pnew, norm_bkernel, norm_tkernel
    
    def _make_kernel(self, bkernel, tkernel):
        def kernel(pt):
            # TODO: Vectorise this!
            bdata = self.points[self.points[0] < pt[0]]
            return bkernel(pt) + _np.sum(tkernel(bdata))
        return kernel
    
    def run_optimisation(self, iterations=20):
        """Runs the optimisation algorithm, and returns information on the
        result.

        :return: :class:`OptimisationResult`
        """

        p = initial_p_matrix(self.points, self.initial_time_bandwidth, self.initial_space_bandwidth)
        errors = []
        for _ in range(iterations):
            pnew, bkernel, tkernel = self.next_iteration(p)
            errors.append(_np.sum((pnew - p) ** 2))
            p = pnew
        kernel = self._make_kernel(bkernel, tkernel)
        return OptimisationResult(kernel=kernel, p=p, background_kernel=bkernel,
            trigger_kernel=tkernel, ell2_error=_np.sqrt(_np.asarray(errors)))

    def predict_time_space_intensity(self, iterations=20):
        """Runs the optimisation algorithm by taking an initial estimation of
        the probability matrix, and then running the optimisation step.  If
        this step ever classifies most events as background, or as triggered,
        then optimisation will fail.  Tuning the initial bandwidth parameters
        may help.

        :param iterations: The number of optimisation steps to perform.

        :return: the estimated intensity kernel.
        
        Tuple of `(kernel, p, bkernel, tkenerl)` where `kernel` is , `p` is the estimated probability matrix,
        `bkernel` is the estimated background kernel, and `tkernel` the
        estimated triggering kernel.
        """

        result = run_optimisation(iterations)
        return result.kernel

class OptimisationResult():
    """Contains results of the optimisation process.

    :param kernel: the overall estimated intensity kernel.
    :param p: the estimated probability matrix.
    :param background_kernel: the estimated background event intensity kernel.
    :param trigger_kernel: the estimated triggered event intensity kernel.
    :param ell2_error: an array of the L^2 differences between successive
    estimates of the probability matrix.  That these decay is a good indication
    of convergence.
    """

    def __init__(self, kernel, p, background_kernel, trigger_kernel, ell2_error):
        self.kernel = kernel
        self.p = p
        self.background_kernel = background_kernel
        self.trigger_kernel = trigger_kernel
        self.ell2_error = ell2_error


def timed_points_to_space_time(timed_points, time_unit = _np.timedelta64(1, "m")):
    times = timed_points.timestamps - timed_points.timestamps[0]
    times /= time_unit
    return np.vstack([times, timed_points.xcoords, timed_points.ycoords])


class SEPPPredictor(predictors.DataPredictor):
    ## TODO: Some docs
    def __init__(self, k_time=100, k_space=15):
        self.k_time = k_time
        self.k_space = k_space
        pass

    # TODO: Somehow, this interface decision is now killing me, because I don't
    # want to force a predict_time as that's rather costly in terms of computation
    def predict(self, cutoff_time, predict_time):
        decluster = StocasticDecluster()
        decluster.background_kernel_estimator = kernels.KNNG1_NDFactors(self.k_time, self.k_space)
        decluster.trigger_kernel_estimator = kernels.KthNearestNeighbourGaussianKDE(self.k_space)
        events = self.data.events_before(cutoff_time)
        decluster.points = timed_points_to_space_time(event)
        kernel = decluster.predict_time_space_intensity(iterations=40)
        def timed_evaluated_kernel(x, y):
            # TODO: Test!
            x = _np.asarray(x)
            if len(x.shape) > 0:
                times = [cutoff_time] * len(x)
            else:
                times = cutoff_time
            return kernel([x,y,times])
        return predictors.KernelRiskPredictor(timed_evaluated_kernel)