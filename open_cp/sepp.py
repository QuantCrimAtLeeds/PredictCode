from . import predictors
#from . import data
#from . import kernels

#import abc as _abc
import numpy as _np

def _normalise_matrix(p):
    column_sums = _np.sum(p, axis=0)
    return p / column_sums[None,:]

def p_matrix(points, background_kernel, trigger_kernel):
    number_data_points = points.shape[-1]
    p = _np.zeros((number_data_points, number_data_points))
    for j in range(number_data_points):
        t = points[0][j] - points[0][:j]
        x = points[1][j] - points[1][:j]
        y = points[2][j] - points[2][:j]
        p[0:j, j] = trigger_kernel(_np.vstack([t,x,y]))
    for j, v in enumerate(background_kernel(points)):
        p[j][j] = v
    return _normalise_matrix(p)

def initial_p_matrix(points, initial_time_bandwidth = 0.1,
        initial_space_bandwidth = 50.0):
    def bkernel(pts):
        return _np.zeros(pts.shape[-1]) + 1
    def tkernel(pts):
        norm = 2 * initial_space_bandwidth ** 2
        return ( _np.exp( - initial_time_bandwidth * pts[0] ) *
                _np.exp( - (pts[1]**2 + pts[2]**2) / norm ) )
    return p_matrix(points, bkernel, tkernel)

def sample_points(points, p):
    number_data_points = points.shape[-1]
    choice = _np.array([ _np.random.choice(j+1, p=p[0:j+1, j])
        for j in range(number_data_points) ])
    mask = ( choice == _np.arange(number_data_points) )
    
    backgrounds = points[:,mask]
    triggered = (points - points[:,choice])[:,~mask]
    return backgrounds, triggered


class StocasticDecluster():
    def __init__(self):
        self.background_kernel_estimator = None
        self.trigger_kernel_estimator = None
        self.initial_time_bandwidth = 0.1 * (_np.timedelta64(1, "D") / _np.timedelta64(1, "m"))
        self.initial_space_bandwidth = 50.0
        self.points = _np.empty((3,0))

    def next_iteration(self, p):
        backgrounds, triggered = sample_points(self.points, p)

        bkernel = self.background_kernel_estimator(backgrounds)
        tkernel = self.trigger_kernel_estimator(triggered)

        number_events = self.points.shape[-1]
        number_background_events = backgrounds.shape[-1]
        number_triggered_events = number_events - number_background_events
        norm_tkernel = lambda pts : ( tkernel(pts) * number_triggered_events / number_events )
        total_time = self.points[0][-1] - self.points[0][0]
        norm_bkernel = lambda pts : ( bkernel(pts) * total_time *
                                     number_background_events / number_events )
        pnew = p_matrix(self.points, norm_bkernel, norm_tkernel)
        return pnew, norm_bkernel, norm_tkernel
    
    def _make_kernel(self, bkernel, tkernel):
        def kernel(pt):
            # TODO: Vectorise this!
            bdata = self.points[self.points[0] < pt[0]]
            return bkernel(pt) + _np.sum(tkernel(bdata))
        return kernel
    
    def predict_time_space_intensity(self):
        p = initial_p_matrix(self.points, self.initial_time_bandwidth, self.initial_space_bandwidth)
        for _ in range(20):
            p, bkernel, tkernel = self.next_iteration(p)
        return self._make_kernel(bkernel, tkernel)

