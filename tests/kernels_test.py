import numpy as np
import scipy.stats as stats
import scipy.linalg
import pytest
import open_cp.kernels as testmod
import open_cp.data
import unittest.mock as mock
import shapely.geometry

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
    gb.covariance_matrix = [[2,2], [3,4]]
    with pytest.raises(ValueError):
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

def test_GaussianBase_large_eval():
    n = 1000000
    pts = np.arange(n) / n
    gb = testmod.GaussianBase(pts)
    gb.covariance_matrix = 1.0
    gb.bandwidth = 1.0

    x5 = np.sum(np.exp(-(5 - pts)**2 / 2)) / n / sqrt2pi
    assert gb(5) == pytest.approx(x5)
    x3 = np.sum(np.exp(-(3 - pts)**2 / 2)) / n / sqrt2pi
    assert gb(3) == pytest.approx(x3)
    np.testing.assert_allclose(gb([5,3]), [x5,x3])

def test_GaussianBase_large_eval_3d():
    n = 1000000
    pts = np.random.random((3,n)) * 100
    gb = testmod.GaussianBase(pts)
    gb.covariance_matrix = np.eye(3)
    gb.bandwidth = 1.0

    pt = np.asarray([1,2,3])
    x = np.sum(np.exp(-np.sum((pts - pt[:,None])**2,axis=0) / 2)) / n / (sqrt2pi**3)
    assert gb([1,2,3]) == pytest.approx(x)
    pt = np.asarray([4,2,1])
    y = np.sum(np.exp(-np.sum((pts - pt[:,None])**2,axis=0) / 2)) / n / (sqrt2pi**3)
    assert gb([4,2,1]) == pytest.approx(y)

    np.testing.assert_allclose(gb([[1,4], [2,2], [3,1]]), [x,y])

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
    
def check_marginal_kernel(ker, axis=0):
    new_ker = testmod.marginalise_gaussian_kernel(ker, axis)
    
    import scipy.integrate
    def expect(x):
        def func(t):
            y = list(x)
            y.insert(axis, t)
            return ker(y)
        return scipy.integrate.quad(func, -10, 10)
    
    for _ in range(20):
        pt = np.random.random(2)
        val, error = expect(pt)
        assert np.abs(new_ker(pt) - val) <= val * 1e-5
    
def test_marginalise_gaussian_kernel():
    pts = np.random.random((3,20))
    ker = testmod.GaussianBase(pts)
    ker.covariance_matrix = np.diag([2,3,4])
    ker.bandwidth = 1.4
    check_marginal_kernel(ker, 0)

    ker = testmod.GaussianBase(pts)
    ker.covariance_matrix = np.diag([2,3,4])
    ker.bandwidth = 1.4
    ker.weights = np.random.random(20)
    check_marginal_kernel(ker, 0)

    ker = testmod.GaussianBase(pts)
    ker.covariance_matrix = np.diag([2,3,4])
    ker.bandwidth = np.random.random(20)
    ker.weights = np.random.random(20)
    check_marginal_kernel(ker, 0)

    ker = testmod.GaussianBase(pts)
    ker.covariance_matrix = np.diag([2,3,4])
    check_marginal_kernel(ker, 1)
    
@pytest.fixture
def geometry_square():
    return shapely.geometry.Polygon([[0,0],[10,0], [10,10], [0,10]])

@pytest.fixture
def gec1(geometry_square):
    data = [[1,9,9,1], [1,1,9,9]]
    data = np.asarray(data)
    assert data.shape == (2,4)
    return testmod.GaussianEdgeCorrect(data, geometry_square)

def test_GaussianEdgeCorrect_point_inside(gec1):
    assert gec1.point_inside(0, 0)
    assert gec1.point_inside(10, 0)
    assert gec1.point_inside(9, 9)
    assert not gec1.point_inside(-1, 0)
    assert not gec1.point_inside(0, -1)
    assert not gec1.point_inside(11, 11)

def test_GaussianEdgeCorrect_agrees_with_GaussianBase(gec1):
    gb = testmod.GaussianBase(gec1.data)
    pts = np.random.random((2,10)) * np.asarray([10,10])[:,None]
    np.testing.assert_allclose(gb(pts), gec1(pts))

def test_GaussianEdgeCorrect_halfS(gec1):
    gb = testmod.GaussianBase(gec1.data)
    hS = scipy.linalg.inv(gb.covariance_matrix)
    hS = scipy.linalg.fractional_matrix_power(hS, 0.5)
    np.testing.assert_allclose(gec1.half_S, hS)

def test_GaussianEdgeCorrect_transformed_geometry(gec1):
    hS = gec1.half_S
    pts = np.asarray([[0,10,10,0], [0,0,10,10]])
    pts = np.dot(hS, pts)

    got = np.asarray(gec1.transformed_geometry.exterior)
    assert got.shape == (5, 2)
    got = got[:4,:]
    np.testing.assert_allclose(pts.T, got)

def _make_sample_points(h, m=10, k=100):
    expected_points = []
    for i in range(1, m+1):
        r = np.sqrt(-2 * h * h * (np.log(m-i+0.5) - np.log(m)))
        for a in range(k):
            angle = a * 2 * np.pi / k
            x, y = r * np.cos(angle), r * np.sin(angle)
            expected_points.append([x, y])
    return np.asarray(expected_points)

def test_GaussianEdgeCorrect_edge_sample_points(gec1):
    expected_points = _make_sample_points(h=gec1.bandwidth)

    def expected_pts(x, y):
        pt = np.dot(gec1.half_S, np.asarray([x,y]))
        return expected_points + pt

    for _ in range(10):
        x, y = np.random.random(2) * [10, 10]
        print("Possibly we don't expect the _order_ to be the same...")
        np.testing.assert_allclose(gec1.edge_sample_points([x,y]), expected_pts(x, y))

def test_GaussianEdgeCorrect_number_intersecting_pts(gec1):
    for _ in range(10):
        pt = np.random.random(2) * [10, 10]
        got = gec1.number_intersecting_pts(pt)
        pts = shapely.geometry.MultiPoint(gec1.edge_sample_points(pt)).intersection(gec1.transformed_geometry)
        assert len(pts) == got

def test_GaussianEdgeCorrect_correction_factor(gec1):
    for _ in range(10):
        pt = np.random.random(2) * [10, 10]
        got = gec1.correction_factor(pt)
        expected = gec1.number_intersecting_pts(pt) / (gec1._m * gec1._k)
        assert expected == pytest.approx(got)
    
    pts = np.random.random((100,2)) * [10, 10]
    got = gec1.correction_factor(pts.T)
    expected = [gec1.correction_factor(pt) for pt in pts]
    np.testing.assert_allclose(got, expected)
        
@pytest.fixture
def masked_grid():
    mask = np.random.random((10,20)) <= 0.5
    return open_cp.data.MaskedGrid(10, 15, 5, 7, mask)

@pytest.fixture
def gecg1(masked_grid):
    data = [[10,90,90,10], [10,10,90,90]]
    data = np.asarray(data)
    assert data.shape == (2,4)
    return testmod.GaussianEdgeCorrectGrid(data, masked_grid)

def test_GaussianEdgeCorrectGrid_pts_to_grid_space(gecg1):
    expected_points = _make_sample_points(h=gecg1.bandwidth)
    pt = np.asarray([1,2])
    S = scipy.linalg.fractional_matrix_power(gecg1.covariance_matrix, 0.5)
    expected_points = np.dot(S, expected_points.T)
    expected_points = (expected_points.T + pt - [5,7] ) / [10,15]
    np.testing.assert_allclose(gecg1.points_to_grid_space(pt), np.floor(expected_points))
    assert expected_points.shape == (1000, 2)

def test_GaussianEdgeCorrectGrid_number_intersecting_pts(gecg1, masked_grid):
    pt = [1,2]
    got = gecg1.number_intersecting_pts([1,2])
    expected = 0
    for gx, gy in gecg1.points_to_grid_space(pt):
        if gx >= 0 and gy >= 0 and gx < 20 and gy < 10 and not masked_grid.mask[gy][gx]:
            expected += 1
    assert got == expected

def test_GaussianEdgeCorrectGrid_correction_factor(gecg1):
    pt = np.asarray([[1,2,3], [4,5,6]])
    assert pt.shape == (2,3)
    expected = []
    for x, y in pt.T:
        expected.append(gecg1.correction_factor((x,y)))
    np.testing.assert_allclose(expected, gecg1.correction_factor(pt))

def _masked_grid_to_poly(mg):
    poly = None
    for x in range(mg.xextent):
        for y in range(mg.yextent):
            if mg.is_valid(x, y):
                xx = x * mg.xsize + mg.xoffset
                yy = y * mg.ysize + mg.yoffset
                p = [[xx,yy], [xx+mg.xsize,yy], [xx+mg.xsize,yy+mg.ysize], [xx,yy+mg.ysize]]
                p = shapely.geometry.Polygon(p)
                if poly is None:
                    poly = p
                else:
                    poly = poly.union(p)
    return poly

def test_GaussianEdgeCorrectGrid_vs_GaussianEdgeCorrect(gecg1):
    geo = _masked_grid_to_poly(gecg1.masked_grid)
    gec = testmod.GaussianEdgeCorrect(gecg1.data, geo)

    for _ in range(100):
        pt = np.random.random(2) * 100
        gpt = pt - [gecg1.masked_grid.xoffset, gecg1.masked_grid.yoffset]
        gpt = np.floor_divide(gpt, [gecg1.masked_grid.xsize, gecg1.masked_grid.ysize]).astype(np.int)
        if gecg1.masked_grid.mask[gpt[1], gpt[0]]:
            continue
        assert gec.number_intersecting_pts(pt) == gecg1.number_intersecting_pts(pt)
        assert gec.correction_factor(pt) == gecg1.correction_factor(pt)


def test_Reflect1D():
    def kernel(pts):
        return np.exp(-pts*pts)

    k = testmod.Reflect1D(kernel)

    assert k.kernel is kernel
    assert k(5) == pytest.approx(2*np.exp(-25))
    np.testing.assert_allclose(k([5,7]), 2*np.exp([-25, -49]))
