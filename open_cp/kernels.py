import scipy.spatial as _spatial
import numpy as _np
import abc as _abc


class KernelEstimator(metaclass=_abc.ABCMeta):
    @_abc.abstractmethod
    def __call__(self, coords):
        """Input should be N coordinates in n dimensional space, and when n>1,
        an array of shape (n,N) to be consistent with ourselves.
        
        Output is an instance of Kernel"""
        pass


class Kernel(metaclass=_abc.ABCMeta):
    @_abc.abstractmethod
    def __call__(self, points):
        """Input should be N coordinates in n dimensional space, and when n>1,
        an array of shape (n,N) to be consistent with ourselves.
        
        Output is an array of length N giving the kernel intensity at each point."""
        pass


def _gaussian_kernel(points, mean, var):
    """pts is array of shape (N,k) where k is the dimension of space.

    mean is array of shape (M,k) and var an array of shape (M,k)

    For each point in `pts`: for each of i=1...M and each coord j=1...k
    we compute the Gaussian kernel centred on mean[i][j] with variance var[i][j],
    and then product over the j, sum over the i, and finally divide by M.

    Returns an array of shape (N) unless N=1 when returns a scalar.
    """
    if len(mean.shape) == 1:
        # So k=1
        mean = mean[:, None]
        var = var[:, None]
        if len(points.shape) == 0:
            pts = _np.array([points])[:, None]
        else:
            pts = points[:, None]
    else:
        # k>1 so if points is 1D it's a single point
        if len(points.shape) == 1:
            pts = points[None, :]
        else:
            pts = points

    # x[i][j] = (pts[i] - mean[j])**2   (as a vector)
    x = (pts[:,None,:] - mean[None,:,:]) ** 2
    var_broad = var[None,:,:] * 2.0
    x = _np.exp( - x / var_broad ) / _np.sqrt((_np.pi * var_broad))
    return_array = _np.mean(_np.product(x, axis=2), axis=1)
    return return_array if pts.shape[0] > 1 else return_array[0]

def kth_nearest_neighbour_gaussian_kde(coords, k=15):
    """Input should be N coordinates in n dimensional space, and when n>1,
    an array of shape (n,N) to be consistent with ourselves"""
    coords = _np.asarray(coords)
    k = min(k, coords.shape[-1] - 1)
    
    if len(coords.shape) == 1:
        stds = _np.std(coords, ddof=1)
        points = coords / stds
        points = points[:, None]
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
        # TODO: If we have one dimensional data and point is just a number,
        #    this doesnt work.
        # Allow the "kernel" convention in argument passing
        return _gaussian_kernel(_np.asarray(point).T, means, var)

    return kernel


class KthNearestNeighbourGaussianKDE(KernelEstimator):
    def __init__(self, k=15):
        self.k = k
        
    def __call__(self, coords):
        return kth_nearest_neighbour_gaussian_kde(coords, self.k)


class KNNG1_NDFactors(KernelEstimator):
    """Applies the KthNearestNeighbourGaussianKDE to first coorindate
    with one value of k, and then to the remaining coordinates with another
    value of k, and combines the result."""
    
    def __init__(self, k_first=100, k_rest=15):
        self.k_first = k_first
        self.k_rest = k_rest
        
    def __call__(self, coords):
        first = kth_nearest_neighbour_gaussian_kde(coords[0], self.k_first)
        rest = kth_nearest_neighbour_gaussian_kde(coords[1:], self.k_rest)
        return lambda coords : first(coords[0]) * rest(coords[1:])