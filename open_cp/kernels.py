import scipy.spatial as _spatial
import numpy as _np

def _gaussian_kernel(pts, mean, var):
    """pts is array of shape (N,k) where k is the dimension of space.

    mean is array of shape (M,k) and var an array of shape (M,k)

    For each point in `pts`: for each of i=1...M and each coord j=1...k
    we compute the Gaussian kernel centred on mean[i][j] with variance var[i][j],
    and then product over the j, sum over the i, and finally divide by M.

    Returns an array of shape (N).
    """
    # TODO: Doesn't allow pts to be scalar... (do I care?)

    if len(mean.shape) == 1:
        mean = _np.broadcast_to(mean, (1, len(mean))).T
        var = _np.broadcast_to(var, (1, len(var))).T
    if len(pts.shape) == 1:
        pts = _np.broadcast_to(pts, (1, len(pts))).T

    # x[i][j] = (pts[i] - mean[j])**2   (as a vector)
    x = _np.broadcast_to(pts, (mean.shape[0],) + pts.shape).swapaxes(0, 1)
    x = x - _np.broadcast_to(mean, x.shape)
    x = x ** 2

    var_broad = _np.broadcast_to(var, (pts.shape[0], ) + var.shape) * 2.0
    x = _np.exp( - x / var_broad ) / _np.sqrt((_np.pi * var_broad))
    return _np.mean(_np.product(x, axis=2), axis=1)

def kth_nearest_neighbour_gaussian_kde(coords, k=15):
    """Input should be N coordinates in k dimensional space, and when k>1,
    an array of shape (k,N) to be consistent with ourselves"""
    if len(coords.shape) == 1:
        stds = _np.std(coords, ddof=1)
        points = coords / stds
        points = _np.broadcast_to(points, (1, len(coords))).T
    else:
        points = coords.T
        stds = _np.std(points, axis=0, ddof=1)
        points = points / stds

    tree = _spatial.KDTree(points)
    
    distance_to_k = _np.empty(points.shape[0])
    for i, p in enumerate(points):
        distances, indexes = tree.query(p, k=k+1)
        distance_to_k[i] = distances[-1]

    means = coords.T
    var = _np.tensordot(distance_to_k, stds, axes=0) ** 2

    def kernel(point):
        # Allow the "kernel" convention in argument passing
        return _gaussian_kernel(point.T, means, var)

    return kernel