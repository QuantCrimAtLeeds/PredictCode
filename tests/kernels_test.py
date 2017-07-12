import numpy as np
import scipy.stats as stats
import pytest
import open_cp.kernels as testmod
import unittest.mock as mock

def slow_gaussian_kernel_new(pts, mean, var):
    """Test case where `pts`, `mean`, `var` are all of shape 2."""
    assert(len(pts.shape) == 2 and len(mean.shape) == 2 and len(var.shape) == 2)
    space_dim = pts.shape[0]
    num_pts = pts.shape[1]
    num_samples = mean.shape[1]
    assert(space_dim == mean.shape[0])
    assert((space_dim, num_samples) == var.shape)

    out = np.empty(num_pts)
    for i in range(num_pts):
        total = np.empty(num_samples)
        for j in range(num_samples):
            prod = np.empty(space_dim)
            for k in range(space_dim):
                v = var[k][j] * 2
                prod[k] = np.exp(- (pts[k][i] - mean[k][j]) **
                                 2 / v) / np.sqrt(np.pi * v)
            total[j] = np.product(prod)
        out[i] = np.mean(total)

    return out

def test_slow_gaussian_kernel_single_new():
    pts = np.empty((1, 1))
    pts[0][0] = 1
    mean = np.empty((1, 1))
    mean[0][0] = 0.5
    var = np.empty((1, 1))
    var[0][0] = 3

    expected = np.exp(-0.25 / 6) / np.sqrt(6 * np.pi)
    got = slow_gaussian_kernel_new(pts, mean, var)
    np.testing.assert_allclose(expected, got)

def test_compare_GaussianKernel():
    for k in range(1, 6):
        for M in range(1, 6):
            mean = np.random.random(size=(k,M))
            var = 0.0001 + np.random.random(size=(k,M))**2
            kernel = testmod.GaussianKernel(mean, var)
            for N in range(1, 6):
                pts = np.random.random(size=(k,N))
                want = slow_gaussian_kernel_new(pts, mean, var)
                got = kernel(pts)
                print(k,M,N)
                np.testing.assert_allclose(got, want)
            # Single point case
            pts = np.random.random(size=k)
            want = slow_gaussian_kernel_new(pts[:,None], mean, var)[0]
            got = kernel(pts)
            print("Single point case k={}, M={}".format(k,M))
            assert want == pytest.approx(got)

def test_compare_GaussianKernel_k1_case():
    for M in range(1, 6):
        mean = np.random.random(size=M)
        var = 0.0001 + np.random.random(size=M)**2
        kernel = testmod.GaussianKernel(mean, var)
        for N in range(1, 6):
            pts = np.random.random(size=N)
            want = slow_gaussian_kernel_new(pts[None,:], mean[None,:], var[None,:])
            got = kernel(pts)
            print(M,N)
            np.testing.assert_allclose(got, want)
        # Single point case
        print("Single point case, M={}".format(M))
        pts = np.random.random()
        want = slow_gaussian_kernel_new(np.asarray(pts)[None,None], mean[None,:], var[None,:])[0]
        got = kernel(pts)
        assert want == pytest.approx(got)
        
def test_1D_kth_distance():
    coords = [0,1,2,3,6,7,9,15]
    distances = testmod.compute_kth_distance(coords, k=3)
    np.testing.assert_allclose(distances, [3,2,2,3,3,4,6,9])

def test_2D_kth_distance():
    coords = [[0,0,1,1],[0,1,0,2]]
    distances = testmod.compute_kth_distance(coords, k=2)
    np.testing.assert_allclose(distances, [1,np.sqrt(2),np.sqrt(2),2])

def slow_kth_nearest(points, index):
    """(k, N) input.  Returns ordered list [0,...] of distance to kth nearest point from index"""
    if len(points.shape) == 1:
        points = points[None, :]
    pt = points[:, index]
    distances = np.sqrt(np.sum((points - pt[:,None])**2, axis=0))
    distances.sort()
    return distances

def test_slow_kth_nearest():
    pts = np.array([1,2,4,5,7,8,9])
    got = slow_kth_nearest(pts, 0)
    np.testing.assert_array_equal(got, [0,1,3,4,6,7,8])
    got = slow_kth_nearest(pts, 3)
    np.testing.assert_array_equal(got, [0,1,2,3,3,4,4])
    got = slow_kth_nearest(pts, 4)
    np.testing.assert_array_equal(got, [0,1,2,2,3,5,6])

    pts = np.array([[0,0],[1,1],[0,1],[1,0],[2,3]]).T
    got = slow_kth_nearest(pts, 0)
    np.testing.assert_allclose(got, [0,1,1,np.sqrt(2),np.sqrt(13)])
    got = slow_kth_nearest(pts, 1)
    np.testing.assert_allclose(got, [0,1,1,np.sqrt(2),np.sqrt(5)])

def test_1d_kth_nearest():
    # In the 1D scale we don't need to rescale
    pts = np.random.random(size=20) * 20 - 10
    for k in [1,2,3,4,5]:
        distances = [slow_kth_nearest(pts, i)[k] for i in range(len(pts))]
        def expected_kernel(x):
            value = 0
            for i, p in enumerate(pts):
                value += stats.norm(loc=p, scale=distances[i]).pdf(x)
            return value / len(pts)
        kernel = testmod.kth_nearest_neighbour_gaussian_kde(pts, k=k)
        test_points = np.random.random(size=10) * 15
        np.testing.assert_allclose( kernel(test_points), expected_kernel(test_points) )

def test_2d_kth_nearest():
    for space_dim in range(2, 5):
        pts = np.random.random(size=(space_dim, 20))
        stds = np.std(pts, axis=1)
        rescaled = np.empty_like(pts)
        for i in range(space_dim):
            rescaled[i] = pts[i] / stds[i]
        for k in [1,2,3,4,5,6]:
            distances = [slow_kth_nearest(rescaled, i)[k] for i in range(pts.shape[1])]
            def expected_kernel(x):
                value = 0
                for i in range(pts.shape[1]):
                    prod = 1
                    for coord in range(space_dim):
                        p = pts[coord,i]
                        prod *= stats.norm(loc=p, scale=distances[i]*stds[coord]).pdf(x[coord])
                    value += prod
                return value / pts.shape[1]
            kernel = testmod.kth_nearest_neighbour_gaussian_kde(pts, k=k)
            test_points = np.random.random(size=(space_dim, 10))
            np.testing.assert_allclose( kernel(test_points), expected_kernel(test_points) )

def test_ReflectedKernel():
    kernel = lambda pt : np.abs(pt)
    testkernel = testmod.ReflectedKernel(kernel)
    assert( testkernel(5) == 10 )
    np.testing.assert_allclose(testkernel([1,2,3]), [2,4,6])
    
    # 2 (or 3 etc.) dim kernel only
    testkernel = testmod.ReflectedKernel(lambda pt : np.abs(pt[0]))
    np.testing.assert_allclose(testkernel([[1,2,3],[4,5,6]]), [2,4,6])
    testkernel = testmod.ReflectedKernel(lambda pt : pt[0] * (pt[0]>=0))
    np.testing.assert_allclose(testkernel([[1,2,3],[4,5,6]]), [1,2,3])
    testkernel = testmod.ReflectedKernel(lambda pt : pt[0] * (pt[0]>=0), reflected_axis=1)
    np.testing.assert_allclose(testkernel([[1,2,3],[4,5,6]]), [2,4,6])

def test_ReflectedKernelEstimator():
    estimator = mock.MagicMock()
    kernel_mock = mock.MagicMock()
    estimator.return_value = kernel_mock
    test = testmod.ReflectedKernelEstimator(estimator)
    kernel = test([1,2,3,4])
    estimator.assert_called_with([1,2,3,4])
    assert(kernel.reflected_axis == 0)
    assert(kernel.delegate is kernel_mock)

    test = testmod.ReflectedKernelEstimator(estimator, reflected_axis=2)
    kernel = test([1,2,3,4])
    assert(kernel.reflected_axis == 2)
    