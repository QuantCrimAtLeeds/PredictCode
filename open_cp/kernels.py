"""
kernels
~~~~~~~

For us, a "kernel" is simply a non-normalised probability density function.
We use kernels extensively to represent (conditional) intensity functions in
point processes.

More formally, a kernel is any python object which is callable (e.g. a
function, or an instance of a class implementing `__call__`).  We follow the
e.g. scipy convention:

- A kernel expecting a one dimensional input may take a scalar as input,
or a one-dimensional numpy array.  It should return, respectively, a scalar
or a one-dimensional array of the same size.  For example:

    def gaussian(p):
        return np.exp(-p * p)

Here we use `np.exp` to make sure that if `p` is an array, we handle it
correctly.

- A kernel expecting a `k` dimensional input may take an array of shape `(k)`
to represent a point, or an array of shape `(k,N)` to represent `N` points.
The return should be, respectively, a scalar or an array of shape `(N)`.
We follow this convention to allow e.g. the following:

   def x_y_sum(p):
       return p[0] + p[1]

In the single-point case, `p[0]` is a scalar representing the x coordinate and
`p[1]` a scalar representing the y coordinate.  In the multiple point case,
`p[0]` is an array of all the x coordinates.
"""


import scipy.spatial as _spatial
import numpy as _np
import abc as _abc


class Kernel(metaclass=_abc.ABCMeta):
    @_abc.abstractmethod
    def __call__(self, points):
        """:param points: N coordinates in n dimensional space.  When n>1, the
        input should be an array of shape (n,N).
        
        :return: An array of length N giving the kernel intensity at each point.
        """
        pass


class KernelEstimator(metaclass=_abc.ABCMeta):
    @_abc.abstractmethod
    def __call__(self, coords):
        """:param coords: N coordinates in n dimensional space.  When n>1, the
        input should be an array of shape (n,N).
        
        :return: A kernel, probably an instance of Kernel.
        """
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
    """Find the (Euclidean) distance to the `k`th nearest neighbour.

    :param coords: An array of shape (n,N) of N points in n dimensional space;
    if n=1 then input is an array of shape (N).
    :param k: The nearest neighbour to use, defaults to 15, if N is too small
    then uses N-1.
    
    :return: An array of shape (N) where the i-th entry is the distance from
    the i-th point to its k-th nearest neighbour."""
    points = _np.asarray(coords)
    k = min(k, points.shape[-1] - 1)

    # scipy.spatial uses the other convention; wants shape (N,n)
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
    """Find the (Euclidean) distance to the `k`th nearest neighbour.
    The input data is first scaled so that each coordinate (independently) has
    unit sample variance.

    :param coords: An array of shape (n,N) of N points in n dimensional space;
    if n=1 then input is an array of shape (N).
    :param k: The nearest neighbour to use, defaults to 15, if N is too small
    then uses N-1.
    
    :return: An array of shape (N) where the i-th entry is the distance from
    the i-th point to its k-th nearest neighbour."""
    coords = _np.asarray(coords)
    if len(coords.shape) == 1:
        points = coords / _np.std(coords, ddof=1)
    else:
        points = coords / _np.std(coords, axis=1, ddof=1)[:, None]
    return compute_kth_distance(points, k)

def kth_nearest_neighbour_gaussian_kde(coords, k=15):
    """Estimate a kernel using variable bandwidth with a Gaussian kernel.
    The input data is scaled (independently in each coordinate) to have unit
    variance in each coordinate, and then the distance to the `k`th nearest
    neighbour is found.  The returned kernel is normalised, and is the sum
    of Gaussians centred on each data point, where the standard deviation for
    each coordinate is the distance for that point, multiplied by the standard
    deviation for that coordinate.

    See the Appendix of:
    Mohler et al, "Self-Exciting Point Process Modeling of Crime",
       Journal of the American Statistical Association, 2011
       DOI: 10.1198/jasa.2011.ap09546

    :param coords: An array of shape (n,N) of N points in n dimensional space;
    if n=1 then input is an array of shape (N).
    :param k: The nearest neighbour to use, defaults to 15, if N is too small
    then uses N-1.
    
    :return: A kernel object.
    """
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
        # TODO change the call signature of _gaussian_kernel
        return _gaussian_kernel(_np.asarray(point).T, means, var)

    return kernel

def marginal_knng(coords, coord_index=0, k=15):
    """Computes a one-dimensional marginal for the kernel which would be
    returned by :function kth_nearest_neighbour_gaussian_kde: Equivalent to,
    but much faster, than (numerically) integerating out all but one variable.
    
    :param coords: An array of shape (n,N) of N points in n dimensional space;
    if n=1 then input is an array of shape (N).
    :param coord_index: Which coordinate to return the marginal for; defaults
    to 0 so giving the first coordinate.
    :param k: The nearest neighbour to use, defaults to 15, if N is too small
    then uses N-1.
    
    :return: A one-dimensional kernel.
    """
    if len(coords.shape) == 1:
        raise ValueError("Input data is already one dimensional")
    distances = compute_normalised_kth_distance(coords, k)
    data = coords[coord_index]
    var = (_np.std(data, ddof=1) * distances) ** 2
    def kernel(x):
        return _gaussian_kernel(_np.asarray(x).T, data, var)
    return kernel

class KthNearestNeighbourGaussianKDE(KernelEstimator):
    """A :class KernelEstimator: which applies the algorithm given by
    :function kth_nearest_neighbour_gaussian_kde:

    :param k: The nearest neighbour to use, defaults to 15, if N is too small
    then uses N-1.
    """
    def __init__(self, k=15):
        self.k = k
        
    def __call__(self, coords):
        return kth_nearest_neighbour_gaussian_kde(coords, self.k)


class KNNG1_NDFactors(KernelEstimator):
    """A :class KernelEstimator: which applies the
    :class KthNearestNeighbourGaussianKDE: to first coordinate with one value
    of k, and then to the remaining coordinates with another value of k, and
    combines the result.
    
    :param k_first: The nearest neighbour to use in the first coordinate,
    defaults to 100, if N is too small then uses N-1.
    :param k_rest: The nearest neighbour to use for the remaining coordinates,
    defaults to 15, if N is too small then uses N-1.
    """
    def __init__(self, k_first=100, k_rest=15):
        self.k_first = k_first
        self.k_rest = k_rest
        
    class KNNG1_NDFactors_Kernel(Kernel):
        def __init__(self, first, rest):
            self.first, self.rest = first, rest

        def __call__(self, points):
            return self.first(points[0]) * self.rest(points[1:])

    def __call__(self, coords):
        return self.KNNG1_NDFactors_Kernel(self.first(coords), self.rest(coords))

    def first(self, coords):
        """Find the kernel estimate for the first coordinate only.
        
        :param coords: All the coordinates; only the 1st coordinate will be
        used.
        
        :return: A one dimensional kernel.
        """
        return kth_nearest_neighbour_gaussian_kde(coords[0], self.k_first)

    def rest(self, coords):
        """Find the kernel estimate for the remaining (n-1) coordinates only.
        
        :param coords: All the coordinates; the 1st coordinate will be ignored.
        
        :return: A (n-1) dimensional kernel.
        """
        return kth_nearest_neighbour_gaussian_kde(coords[1:], self.k_rest)