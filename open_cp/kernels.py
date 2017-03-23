import scipy.spatial as _spatial
import numpy as _np

# TODO: Test this
def _gaussian_kernel(pts, mean, var):
    """pts is array of shape (N,k) where k is the dimension of space.

    mean is array of shape (M,k) and var an array of shape (M)

    For each point in `pts`: for each of i=1...M we compute the Gaussian
    kernel centred on mean[i] with variance var[i], and then sum the result.

    Returns an array of shape (N).
    """
    # TODO: Doesn't allow pts to be scalar... (do I care?)

    if len(mean.shape) == 1:
        mean = _np.broadcast_to(mean, (1, len(mean)))
        var = _np.array([var])
    if len(pts) == 1:
        pts = _np.broadcast_to(pts, (1, len(pts)))

    # x[i][j] = (pts[i] - mean[j])**2   (as a vector)
    x = _np.broadcast_to(pts, (mean.shape[0],) + pts.shape).swapaxes(0, 1)
    x = x - _np.broadcast_to(mean, x.shape)
    x = x ** 2

    var_broad = _np.broadcast_to(var, pts.shape + (len(var),)).swapaxes(1, 2) * 2.0
    x = _np.exp( - x / var_broad ) / _np.sqrt((_np.pi * var_broad))
    return _np.sum(_np.product(x, axis=2), axis=1)

def kth_nearest_neighbour_gaussian_kde(coords, k=15):
    points = coords.T
    stds = _np.std(points, axis=0, ddof=1)
    points = points / stds
    tree = _spatial.KDTree(points)
    
    distance_to_k = np.empty_like(stds)
    for i, p in enumerate(points):
        distances, indexes = tree.query(p, k=k)
        distance_to_k[i] = distances[-1]

    means = coords.T
    stds = stds * distance_to_k

    def kernel(point):
        # Allow the "kernel" convention in argument passing
        return _gaussian_kernel(point.T, means, stds)

    return kernel