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

def compute_kth_distance(coords, k=15):
    """Input is an array of shape (n,N) of N points in n dimensional space;
    if n=1 then input if an array of shape (N)
    
    Returns an array of shape (N) where the i-th entry is the distance from
    the i-th point to its k-th nearest neighbour."""
    points = _np.asarray(coords)
    k = min(k, points.shape[-1] - 1)

    if len(points.shape) == 1:
        points = points[:, None]
    else:
        points = points.T

    tree = _spatial.KDTree(points)
    distance_to_k = _np.empty(points.shape[0])
    for i, p in enumerate(points):
        distances, indexes = tree.query(p, k=k+1)
        distance_to_k[i] = distances[-1]
    
    return distance_to_k

def compute_normalised_kth_distance(coords, k=15):
    """Input is an array of shape (n,N) of N points in n dimensional space;
    if n=1 then input if an array of shape (N)

    In each coordinate axis independently, the data is scaled to have a sample
    variance of size 1.

    Returns an array of shape (N) where the i-th entry is the distance from
    the i-th point to its k-th nearest neighbour."""
    coords = _np.asarray(coords)

    if len(coords.shape) == 1:
        points = coords / _np.std(coords, ddof=1)
    else:
        points = coords / _np.std(coords, axis=1, ddof=1)[:, None]
    return compute_kth_distance(points, k)

def marginal_knng(coords, coord_index=0, k=15):
    if len(coords.shape) == 1:
        raise ValueError("Input data is already one dimensional")
    distances = compute_normalised_kth_distance(coords, k)
    data = coords[coord_index]
    var = (_np.std(data, ddof=1) * distances) ** 2
    def kernel(x):
        return _gaussian_kernel(_np.asarray(x).T, data, var)
    return kernel

def kth_nearest_neighbour_gaussian_kde(coords, k=15):
    """Input should be N coordinates in n dimensional space, and when n>1,
    an array of shape (n,N) to be consistent with ourselves"""
    coords = _np.asarray(coords)
    means = coords.T

    if len(coords.shape) == 1:
        stds = _np.std(coords, ddof=1)
        points = coords / stds
    else:
        stds = _np.std(means, axis=0, ddof=1)
        points = coords / stds[:, None]
    distance_to_k = compute_kth_distance(points, k)

    var = _np.tensordot(distance_to_k, stds, axes=0) ** 2

    def kernel(point):
        # Allow the "kernel" convention in argument passing
        return _gaussian_kernel(_np.asarray(point).T, means, var)

    return kernel


class KthNearestNeighbourGaussianKDE(KernelEstimator):
    def __init__(self, k=15):
        self.k = k
        
    def __call__(self, coords):
        return kth_nearest_neighbour_gaussian_kde(coords, self.k)


class KNNG1_NDFactors(KernelEstimator):
    """Applies the KthNearestNeighbourGaussianKDE to first coordinate
    with one value of k, and then to the remaining coordinates with another
    value of k, and combines the result."""
    
    def __init__(self, k_first=100, k_rest=15):
        self.k_first = k_first
        self.k_rest = k_rest
        
    def __call__(self, coords):
        first = self.first(coords)
        rest = self.rest(coords)
        return lambda pts : first(pts[0]) * rest(pts[1:])

    def first(self, coords):
        return kth_nearest_neighbour_gaussian_kde(coords[0], self.k_first)

    def rest(self, coords):
        return kth_nearest_neighbour_gaussian_kde(coords[1:], self.k_rest)