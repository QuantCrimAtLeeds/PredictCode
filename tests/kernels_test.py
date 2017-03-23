import numpy as np
import scipy.stats as stats
import pytest
import open_cp.kernels as testmod


def slow_gaussian_kernel(pts, mean, var):
    assert(len(pts.shape) == 2 and len(
        mean.shape) == 2 and len(var.shape) == 2)
    space_dim = pts.shape[1]
    num_pts = pts.shape[0]
    num_samples = mean.shape[0]
    assert(space_dim == mean.shape[1])
    assert((num_samples, space_dim) == var.shape)

    out = np.empty(num_pts)
    for i in range(num_pts):
        total = np.empty(num_samples)
        for j in range(num_samples):
            prod = np.empty(space_dim)
            for k in range(space_dim):
                v = var[j][k] * 2
                prod[k] = np.exp(- (pts[i][k] - mean[j][k]) **
                                 2 / v) / np.sqrt(np.pi * v)
            total[j] = np.product(prod)
        out[i] = np.mean(total)

    return out


def test_slow_gaussian_kernel_single():
    pts = np.empty((1, 1))
    pts[0][0] = 1
    mean = np.empty((1, 1))
    mean[0][0] = 0.5
    var = np.empty((1, 1))
    var[0][0] = 3

    expected = np.array(np.exp(-0.25 / 6) / np.sqrt(6 * np.pi))
    np.testing.assert_allclose(expected, slow_gaussian_kernel(pts, mean, var))


def test_gaussian_kernel_single():
    pts = np.empty((1, 1))
    pts[0][0] = 1
    mean = np.empty((1, 1))
    mean[0][0] = 0.5
    var = np.empty((1, 1))
    var[0][0] = 3

    expected = np.array(np.exp(-0.25 / 6) / np.sqrt(6 * np.pi))
    np.testing.assert_allclose(
        expected, testmod._gaussian_kernel(pts, mean, var))


def test_gaussian_kernel_allows_simple_single():
    pts = np.array([1])
    mean = np.array([0.5])
    var = np.array([3])

    expected = np.array(np.exp(-0.25 / 6) / np.sqrt(6 * np.pi))
    np.testing.assert_allclose(
        expected, testmod._gaussian_kernel(pts, mean, var))


def test_gaussian_kernel():
    pts = np.random.rand(20, 2)
    mean = np.random.rand(5, 2)
    var = np.random.rand(5, 2)
    got = testmod._gaussian_kernel(pts, mean, var)
    expected = slow_gaussian_kernel(pts, mean, var)
    assert(got.shape == (20,))
    np.testing.assert_allclose(expected, got)


def slow_kth_nearest(points, index):
    """(k, N) input.  Returns ordered list [0,...] of distance to kth nearest point from index"""
    if len(points.shape) == 1:
        points = np.broadcast_to(points, (1, len(points)))
    pt = points.T[index]
    distances = np.empty(points.shape[1])
    for i in range(points.shape[1]):
        p = points.T[i]
        distances[i] = np.sqrt(np.sum((p-pt)**2))
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
    pts = np.random.random(size=(2,20))
    stds = np.std(pts, axis=1)
    rescaled = np.empty_like(pts)
    for i in range(2):
        rescaled[i] = pts[i] / stds[i]
    for k in [1,2,3,4,5,6]:
        distances = [slow_kth_nearest(rescaled, i)[k] for i in range(pts.shape[1])]
        def expected_kernel(x):
            value = 0
            for i in range(pts.shape[1]):
                prod = 1
                for coord in range(2):
                    p = pts[coord,i]
                    prod *= stats.norm(loc=p, scale=distances[i]*stds[coord]).pdf(x[coord])
                value += prod
            return value / pts.shape[1]
        kernel = testmod.kth_nearest_neighbour_gaussian_kde(pts, k=k)
        test_points = np.random.random(size=(2,10))
        np.testing.assert_allclose( kernel(test_points), expected_kernel(test_points) )
    