from pytest import approx
import open_cp.sources.random as testmod

import open_cp
from datetime import datetime
import numpy as np
import unittest.mock as mock

class RandomFromBuffer():
    def __init__(self, data):
        self.buffer = data
        self.index = -1
        
    def _get(self):
        self.index += 1
        if self.index == len(self.buffer):
            self.index = 0
        return self.buffer[self.index]

    def __call__(self, size=1):
        r = np.empty(size, dtype=np.float)
        for i in range(size):
            r[i] = self._get()
        return r

@mock.patch("numpy.random.random", RandomFromBuffer([0.5, 0.2, 0.8]))
def test_RandomFromBuffer():
    x = np.random.random(size = 5)
    np.testing.assert_allclose(x, [0.5, 0.2, 0.8, 0.5, 0.2])

@mock.patch("numpy.random.random", RandomFromBuffer([0.5, 0.2, 0.8]))
@mock.patch("numpy.random.poisson")
def test_uniform_random(poisson_mock):
    poisson_mock.return_value = 5
    region = open_cp.RectangularRegion(xmin=0, xmax=100, ymin=-20, ymax=-10)
    start = datetime(2017, 3, 10, 0)
    end = datetime(2017, 3, 20, 0)
    points = testmod.random_uniform(region, start, end, 100)
    poisson_mock.assert_called_with(lam=100)
    
    expected_times = [np.datetime64(s) for s in ["2017-03-15", "2017-03-12",
        "2017-03-18", "2017-03-15", "2017-03-12"]]
    expected_times = np.array(expected_times)
    expected_times.sort()
    assert( all(expected_times == points.timestamps) )

    assert( all(points.coords[0] == [80, 50, 20, 80, 50]) )
    assert( all(points.coords[1] == [-18, -12, -15, -18, -12]) )

@mock.patch("numpy.random.random", RandomFromBuffer([0.1, 0.2, 0.15]))
def test_rejection_sample_2d_single_sample_no_rejection():
    def kernel(p):
        return p[0] + p[1]
    pts = testmod.rejection_sample_2d(kernel, 2)
    assert( pts[0] == approx(0.1) )
    assert( pts[1] == approx(0.2) )

@mock.patch("numpy.random.random", RandomFromBuffer([0.1, 0.2, 0.2, 0.3, 0.4, 0.3]))
def test_rejection_sample_2d_single_sample_with_rejection():
    def kernel(p):
        return p[0] + p[1]
    pts = testmod.rejection_sample_2d(kernel, 2)
    assert( pts[0] == approx(0.3) )
    assert( pts[1] == approx(0.4) )

# Slightly brittle tests, as they assume over-sampling of size 2
@mock.patch("numpy.random.random", RandomFromBuffer([0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
    0.5, 0.4, 0.3, 0, 0, 0,
    0.3, 0.25, 0.2, 0, 0, 0]))
def test_rejection_sample_2d_no_rejection():
    def kernel(p):
        return p[0] + p[1]
    pts = testmod.rejection_sample_2d(kernel, 2, samples=3)
    np.testing.assert_allclose(pts[0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(pts[1], [0.5, 0.4, 0.3])

@mock.patch("numpy.random.random", RandomFromBuffer([0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
    0.5, 0.4, 0.3, 0.2, 0.1, 0,
    0.3, 0.31, 0.37, 0.2, 0.1, 0.15]))
def test_rejection_sample_2d_some_rejection():
    def kernel(p):
        return p[0] + p[1]
    pts = testmod.rejection_sample_2d(kernel, 2, samples=3)
    np.testing.assert_allclose(pts[0], [0.1, 0.4, 0.5])
    np.testing.assert_allclose(pts[1], [0.5, 0.2, 0.1])

@mock.patch("numpy.random.random", RandomFromBuffer([0.1, 0.2, 0.3, 0.4,
    0.5, 0.4, 0.3, 0.2,
    0.3, 0.31, 0.37, 0.4]))
def test_rejection_sample_2d_multiple_passes():
    def kernel(p):
        return p[0] + p[1]
    pts = testmod.rejection_sample_2d(kernel, 2, samples=2)
    np.testing.assert_allclose(pts[0], [0.1, 0.1])
    np.testing.assert_allclose(pts[1], [0.5, 0.5])

@mock.patch("open_cp.sources.random.rejection_sample_2d")
def test_KernelSampler(rejection_sample_mock):
    rejection_sample_mock.return_value = np.array([[0.1, 0.2, 0.3],[0.4, 0.5, 0.6]])
    region = open_cp.RectangularRegion(10, 30, 50, 100)
    def kernel(pts):
        return pts[0] + 2 * pts[1]
    sampler = testmod.KernelSampler(region, kernel, None)
    points = sampler(size = 3)
    np.testing.assert_allclose(points[0], [12, 14, 16])
    np.testing.assert_allclose(points[1], [70, 75, 80])

    kernel_in_use = rejection_sample_mock.call_args_list[0][0][0]
    assert( kernel_in_use([0, 0]) == 110 )
    assert( kernel_in_use([0, 1]) == 10 + 100 * 2 )
    assert( kernel_in_use([1, 0]) == 30 + 50 * 2 )
    assert( kernel_in_use([1, 1]) == 30 + 100 * 2 )
    np.testing.assert_allclose( kernel_in_use(np.array([[0, 0, 1, 1], [0, 1, 0, 1]])),
        [110, 210, 130, 230] )

@mock.patch("numpy.random.poisson")
def test_random_spatial(poisson_mock):
    poisson_mock.return_value = 3
    def sampler(size):
        return np.array([[1,2,3], [4,5,6]])
    start = datetime(2017, 3, 10, 0)
    end = datetime(2017, 3, 20, 0)
    points = testmod.random_spatial(sampler, start, end, 100)

    np.testing.assert_allclose(points.coords[0], [1,2,3])
    np.testing.assert_allclose(points.coords[1], [4,5,6])
