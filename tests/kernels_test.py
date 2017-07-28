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
    
    
def test_GaussianBase_not_point():
    with pytest.raises(ValueError):
        testmod.GaussianBase(5.2)
        
def test_GaussianBase_set_covariance():
    gb = testmod.GaussianBase([1,2,3,4])
    with pytest.raises(ValueError):
        gb.covariance_matrix = [[1,2,3], [2,3,4]]
    with pytest.raises(ValueError):
        gb.covariance_matrix = [[1,2], [3,4]]
    gb.covariance_matrix = 1
    
    gb = testmod.GaussianBase([[1,2,3,4], [4,2,2,1]])
    with pytest.raises(ValueError):
        gb.covariance_matrix = [[1,2,3], [2,3,4]]
    gb.covariance_matrix = [[1,2], [3,4]]
    with pytest.raises(ValueError):
        gb.covariance_matrix = 1
    
def test_GaussianBase_set_band():
    gb = testmod.GaussianBase([1,2,3,4])
    assert gb.bandwidth == pytest.approx(4 ** (-1/5))
    gb.bandwidth = "scott"
    assert gb.bandwidth == pytest.approx(4 ** (-1/5))
    with pytest.raises(ValueError):
        gb.bandwidth = "matt"
    gb.bandwidth = "silverman"
    assert gb.bandwidth == pytest.approx(3 ** (-1/5))

    gb = testmod.GaussianBase([[1,2,3,4],[4,2,1,3]])
    assert gb.bandwidth == pytest.approx(4 ** (-1/6))
    gb.bandwidth = "scott"
    assert gb.bandwidth == pytest.approx(4 ** (-1/6))
    with pytest.raises(ValueError):
        gb.bandwidth = "matt"
    gb.bandwidth = "silverman"
    assert gb.bandwidth == pytest.approx(4 ** (-1/6))

def test_GaussianBase_set_weights():
    gb = testmod.GaussianBase([1,2,3,4])
    assert gb.weights is None
    
    gb.weights = [.2, 0, 5, 2]
    
    with pytest.raises(ValueError):
        gb.weights = [.2, 0, 5]
        
    with pytest.raises(ValueError):
        gb.weights = 2
        
    with pytest.raises(ValueError):
        gb.weights = [[1,2,3],[4,5,6]]

sqrt2pi = np.sqrt(2 * np.pi)

def test_GaussianBase_eval():
    gb = testmod.GaussianBase([1,2,3,4])
    assert gb.covariance_matrix[0,0] == pytest.approx(20/12)
    
    gb.covariance_matrix = 1.0
    gb.bandwidth = 1.0
    x5 = np.sum(np.exp([-16/2, -9/2, -4/2, -1/2])) / 4 / sqrt2pi
    assert gb(5) == pytest.approx(x5)
    x2 = np.sum(np.exp([-1/2, 0, -1/2, -4/2])) / 4 / sqrt2pi
    assert gb(2) == pytest.approx(x2)
    x0 = np.sum(np.exp([-1/2, -4/2, -9/2, -16/2])) / 4 / sqrt2pi
    assert gb(0) == pytest.approx(x0)
    np.testing.assert_allclose(gb([0]), [x0])
    np.testing.assert_allclose(gb([0,2,5,2,5,0]), [x0,x2,x5,x2,x5,x0])

def test_GaussianBase_eval_with_bandwidth():
    gb = testmod.GaussianBase([1,2,3,4])
    assert gb.covariance_matrix[0,0] == pytest.approx(20/12)
    
    gb.covariance_matrix = 1.0
    gb.bandwidth = 2.0
    x5 = np.sum(np.exp([-16/8, -9/8, -4/8, -1/8])) / 8 / sqrt2pi
    assert gb(5) == pytest.approx(x5)
    x2 = np.sum(np.exp([-1/8, 0, -1/8, -4/8])) / 8 / sqrt2pi
    assert gb(2) == pytest.approx(x2)
    x0 = np.sum(np.exp([-1/8, -4/8, -9/8, -16/8])) / 8 / sqrt2pi
    assert gb(0) == pytest.approx(x0)
    np.testing.assert_allclose(gb([0]), [x0])
    np.testing.assert_allclose(gb([0,2,5,2,5,0]), [x0,x2,x5,x2,x5,x0])
    
def test_GaussianBase_eval_with_cov():
    gb = testmod.GaussianBase([1,2,3,4])
    assert gb.covariance_matrix[0,0] == pytest.approx(20/12)
    
    gb.covariance_matrix = 0.5
    gb.bandwidth = 1.0
    x5 = np.sum(np.exp([-16, -9, -4, -1])) / 4 / np.sqrt(0.5) / sqrt2pi
    assert gb(5) == pytest.approx(x5)
    x2 = np.sum(np.exp([-1, 0, -1, -4])) / 4 / np.sqrt(0.5) / sqrt2pi
    assert gb(2) == pytest.approx(x2)
    x0 = np.sum(np.exp([-1, -4, -9, -16])) / 4 / np.sqrt(0.5) / sqrt2pi
    assert gb(0) == pytest.approx(x0)
    np.testing.assert_allclose(gb([0]), [x0])
    np.testing.assert_allclose(gb([0,2,5,2,5,0]), [x0,x2,x5,x2,x5,x0])

def test_GaussianBase_eval_with_weights():
    gb = testmod.GaussianBase([1,2,3,4])
    assert gb.covariance_matrix[0,0] == pytest.approx(20/12)
    
    gb.covariance_matrix = 1.0
    gb.bandwidth = 1.0
    gb.weights = [0,1,20,30]
    x5 = np.sum(np.exp([-16/2, -9/2, -4/2, -1/2]) * [0,1,20,30]) / 51 / sqrt2pi
    assert gb(5) == pytest.approx(x5)
    x2 = np.sum(np.exp([-1/2, 0, -1/2, -4/2]) * [0,1,20,30]) / 51 / sqrt2pi
    assert gb(2) == pytest.approx(x2)
    x0 = np.sum(np.exp([-1/2, -4/2, -9/2, -16/2]) * [0,1,20,30]) / 51 / sqrt2pi
    assert gb(0) == pytest.approx(x0)
    np.testing.assert_allclose(gb([0]), [x0])
    np.testing.assert_allclose(gb([0,2,5,2,5,0]), [x0,x2,x5,x2,x5,x0])

def test_GaussianBase_eval_with_bandwidths():
    gb = testmod.GaussianBase([1,2,3,4])
    assert gb.covariance_matrix[0,0] == pytest.approx(20/12)
    
    gb.covariance_matrix = 1.0
    gb.bandwidth = [0.5, 0.1, 0.7, 5]
    x5 = np.sum(np.exp([-16/2/(0.5**2), -9/2/(0.1**2), -4/2/(0.7**2), -1/2/(5**2)])
        / [0.5, 0.1, 0.7, 5] ) / 4 / sqrt2pi
    assert gb(5) == pytest.approx(x5)
    x3 = np.sum(np.exp([-4/2/(0.5**2), -1/2/(0.1**2), 0, -1/2/(5**2)])
        / [0.5, 0.1, 0.7, 5] ) / 4 / sqrt2pi
    assert gb(3) == pytest.approx(x3)
    np.testing.assert_allclose(gb([3,5,3]), [x3,x5,x3])
    
    with pytest.raises(ValueError):
        gb.bandwidth = [[0.5, 0.1], [0.7, 5]]

def test_GaussianBase_eval_2d():
    gb = testmod.GaussianBase([[1,2,3,4],[1,3,7,5]])
    gb.covariance_matrix = [[1,0],[0,1]]
    gb.bandwidth = 1.0
    
    with pytest.raises(ValueError):
        gb(5)
    with pytest.raises(ValueError):
        gb([1,2,3])
    
    x0 = np.sum(np.exp([-1/2, -2/2, -29/2, -18/2])) / 4 / sqrt2pi / sqrt2pi
    assert gb([1,2]) == pytest.approx(x0)
    
    gb.bandwidth = 2.0
    x0 = np.sum(np.exp([-1/2/4, -2/2/4, -29/2/4, -18/2/4])) / 4 / 4 / sqrt2pi / sqrt2pi
    assert gb([1,2]) == pytest.approx(x0)

def test_GaussianBase_agrees_with_scipy():
    data = np.random.random(size=100)
    gb = testmod.GaussianBase(data)
    kernel = stats.kde.gaussian_kde(data, bw_method="scott")
    
    pts = np.random.random(size=50)
    np.testing.assert_allclose(gb(pts), kernel(pts))

    gb = testmod.GaussianBase(data)
    gb.bandwidth = "silverman"
    kernel = stats.kde.gaussian_kde(data, bw_method="silverman")
    np.testing.assert_allclose(gb(pts), kernel(pts))

def test_GaussianBase_agrees_with_scipy_nd():
    for n in range(2,5):
        data = np.random.random(size=(n, 100))
        gb = testmod.GaussianBase(data)
        kernel = stats.kde.gaussian_kde(data, bw_method="scott")
        
        pts = np.random.random(size=(n, 50))
        np.testing.assert_allclose(gb(pts), kernel(pts))
    
        gb = testmod.GaussianBase(data)
        gb.bandwidth = "silverman"
        kernel = stats.kde.gaussian_kde(data, bw_method="silverman")
        np.testing.assert_allclose(gb(pts), kernel(pts))
    
def test_GaussianNearestNeighbour():
    data = np.random.random(size=20)
    gnn = testmod.GaussianNearestNeighbour(data)
    kernel = testmod.kth_nearest_neighbour_gaussian_kde(data)
    
    pts = np.random.random(size=50)
    np.testing.assert_allclose(gnn(pts), kernel(pts))

    for n in range(1,7):
        data = np.random.random(size=(n,100))
        gnn = testmod.GaussianNearestNeighbour(data)
        kernel = testmod.kth_nearest_neighbour_gaussian_kde(data)
        
        pts = np.random.random(size=(n,50))
        np.testing.assert_allclose(gnn(pts), kernel(pts))
    
