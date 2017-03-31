from . import predictors
import numpy as _np

def _normalise_matrix(p):
    column_sums = _np.sum(p, axis=0)
    return p / column_sums[None,:]

def p_matrix(points, background_kernel, trigger_kernel):
    number_data_points = points.shape[-1]
    p = _np.zeros((number_data_points, number_data_points))
    for j in range(1, number_data_points):
        d = points[:, j][:,None] - points[:, :j]
        p[0:j, j] = trigger_kernel(d)
    p += _np.diag(background_kernel(points))
    return _normalise_matrix(p)

def p_matrix_fast(points, background_kernel, trigger_kernel, time_cutoff=150, space_cutoff=1):
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
    def bkernel(pts):
        return _np.zeros(pts.shape[-1]) + 1
    def tkernel(pts):
        time = _np.exp( - pts[0] / initial_time_bandwidth )
        norm = 2 * initial_space_bandwidth ** 2
        space = _np.exp( - (pts[1]**2 + pts[2]**2) / norm )
        return time * space
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
        norm_bkernel = lambda pts : ( bkernel(pts) * number_background_events )
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

