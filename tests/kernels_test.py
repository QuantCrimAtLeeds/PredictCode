import numpy as np
import pytest
import open_cp.kernels as testmod

def slow_gaussian_kernel(pts, mean, var):
    assert( len(pts.shape) == 2 and len(mean.shape) == 2 and len(var.shape) == 1 )
    space_dim = pts.shape[1]
    num_pts = pts.shape[0]
    num_samples = mean.shape[0]
    assert( space_dim == mean.shape[1] )
    assert( num_samples == len(var) )

    out = np.empty(num_pts)
    for i in range(num_pts):
        total = np.empty(num_samples)
        for j in range(num_samples):
            v = var[j] * 2
            prod = np.empty(space_dim)
            for k in range(space_dim):
                prod[k] = np.exp( - (pts[i][k] - mean[j][k]) ** 2 / v ) / np.sqrt(np.pi * v)
            total[j] = np.product(prod)
        out[i] = np.sum(total)
    
    return out

def test_slow_gaussian_kernel_single():
    pts = np.empty((1,1))
    pts[0][0] = 1
    mean = np.empty((1,1))
    mean[0][0] = 0.5
    var = np.empty(1)
    var[0] = 3

    expected = np.array( np.exp(-0.25 / 6) / np.sqrt(6 * np.pi) )
    np.testing.assert_allclose( expected, slow_gaussian_kernel(pts, mean, var) )

def test_gaussian_kernel_single():
    pts = np.empty((1,1))
    pts[0][0] = 1
    mean = np.empty((1,1))
    mean[0][0] = 0.5
    var = np.empty(1)
    var[0] = 3

    expected = np.array( np.exp(-0.25 / 6) / np.sqrt(6 * np.pi) )
    np.testing.assert_allclose( expected, testmod._gaussian_kernel(pts, mean, var) )

def test_gaussian_kernel_allows_simple_single():
    pts = np.array([1])
    mean = np.array([0.5])
    var = 3

    expected = np.array( np.exp(-0.25 / 6) / np.sqrt(6 * np.pi) )
    np.testing.assert_allclose( expected, testmod._gaussian_kernel(pts, mean, var) )

def test_gaussian_kernel():
    pts = np.random.rand(20, 2)
    mean = np.random.rand(5, 2)
    var = np.random.rand(5)
    got = testmod._gaussian_kernel(pts, mean, var)
    expected = slow_gaussian_kernel(pts, mean, var)
    assert( got.shape == (20,) )
    np.testing.assert_allclose(expected, got)