import pytest
import numpy as np
import unittest.mock as mock
import tests.helpers as helpers

import open_cp.data
import open_cp.sources.sepp as testmod
import datetime

def test_scale_to_real_time():
    points = np.arange(15).reshape(3,5)
    pts = testmod.scale_to_real_time(points, datetime.datetime(2017,3,20,12,30))
    np.testing.assert_array_equal(pts.timestamps, [np.datetime64("2017-03-20T12:30"),
        np.datetime64("2017-03-20T12:31"), np.datetime64("2017-03-20T12:32"),
        np.datetime64("2017-03-20T12:33"), np.datetime64("2017-03-20T12:34")])


def expected_ptgs_kernel(t, x, y):
    c = np.sqrt(1 - 0.4**2)
    norm = 1 / ( 2 * np.pi * c * np.sqrt(2) )
    xx = (x - 3)**2
    yy = (y - 4)**2 / 2
    xy = 0.8 * (x - 3) * (y - 4) / np.sqrt(2)
    return 5 * np.exp(-(xx + yy - xy) / (2*c*c)) * norm

def test_PoissonTimeGaussianSpace():
    ker = testmod.PoissonTimeGaussianSpace(5, [3, 4], [1, 2], 0.4)
    
    c = np.sqrt(1 - 0.4**2)
    assert( ker.kernel_max(0,1) == pytest.approx(5 / ( 2 * np.pi * c * np.sqrt(2))) )
    
    pts = []
    want = []
    for _ in range(100):
        pt = np.random.random(3) * np.array([2,4,4]) + np.array([0,-2,-2])
        pts.append(pt)
        want.append( expected_ptgs_kernel(*pt) )
        assert( ker(pt) == pytest.approx(want[-1]) )
    
    pts = np.array(pts).T
    assert( pts.shape == (3,100) )
    np.testing.assert_allclose(ker(pts), want)
    
@mock.patch("numpy.random.poisson")
@mock.patch("numpy.random.random")
def test_multi_dim_random_mock(random_mock, poisson_mock):
    poisson_mock.return_value = 5
    random_mock.return_value = (np.arange(1,10) * 0.1).reshape(3,3)
    
    x = np.random.random((3,10))
    np.testing.assert_allclose( x[0], np.array([1,2,3])/10 )
    np.testing.assert_allclose( x[1], np.array([4,5,6])/10 )
    np.testing.assert_allclose( x[2], np.array([7,8,9])/10 )
    random_mock.assert_called_with((3,10))
    
    assert( np.random.poisson(lam=3.2) == 5 )
    poisson_mock.assert_called_with(lam=3.2)


class TestKernel(testmod.SpaceTimeKernel):
    def intensity(self, t, x, y):
        return np.zeros_like(t) + 2
    
    def kernel_max(self, start_time, end_time):
        return 2.0
    
@mock.patch("numpy.random.random", helpers.RandomCyclicBuffer([.1,.2,.3]))
@mock.patch("numpy.random.poisson")
def test_InhomogeneousPoisson_zero_kernel(poisson_mock):
    region = open_cp.data.RectangularRegion(10, 20, 50, 100)
    sampler = testmod.InhomogeneousPoisson(region, TestKernel())
    
    poisson_mock.return_value = 5
    points = sampler.sample(10, 20)
    poisson_mock.assert_called_with(lam=10000)
    
    np.testing.assert_allclose( points[0], [11,11,12,12,13] )
    np.testing.assert_allclose( points[1], np.array([3,3,1,1,2]) + 10 )
    np.testing.assert_allclose( points[2], np.array([2,2,3,3,1]) * 5 + 50 )


class TestKernel1(testmod.SpaceTimeKernel):
    def intensity(self, t, x, y):
        #return ((x-15)**2 + (y-75)**2 <= 25).astype(np.float)
        return 2/5 - 0.01
    
    def kernel_max(self, start_time, end_time):
        return 2.0

@mock.patch("numpy.random.random", helpers.RandomCyclicBuffer([.1,.2,.3]))
@mock.patch("numpy.random.poisson")
def test_InhomogeneousPoisson_varying_kernel(poisson_mock):
    region = open_cp.data.RectangularRegion(10, 20, 50, 100)
    sampler = testmod.InhomogeneousPoisson(region, TestKernel1())
    
    poisson_mock.return_value = 5
    points = sampler.sample(0, 10)
    
    # accept_prob == [1,2,3,1,2] / 5
    np.testing.assert_allclose( points[0], [1,1] )
    np.testing.assert_allclose( points[1], np.array([13,13]) )
    np.testing.assert_allclose( points[2], np.array([60,60]) )


class TestSamplerMiddle2(testmod.Sampler):
    def sample(self, start_time, end_time):
        td = end_time - start_time
        return np.array([[start_time + td/3, start_time + td*2/3],[0,1],[1,2]]).astype(np.float)
    
class TestSamplerAdd(testmod.Sampler):
    def sample(self, start_time, end_time):
        t = start_time + 2
        if t > end_time:
            return np.array([[],[],[]])
        return np.array([[t],[1],[0]]).astype(np.float)
    
def test_SelfExcitingPointProcess():
    region = open_cp.data.RectangularRegion(0,100,0,100)
    sampler = testmod.SelfExcitingPointProcess(TestSamplerMiddle2(), TestSamplerAdd())
    pts = sampler.sample(0, 10)
    np.testing.assert_allclose(pts[0], [10/3, 16/3, 20/3, 22/3, 26/3, 28/3])
    np.testing.assert_allclose(pts[1], [0, 1, 1, 2, 2, 3])
    np.testing.assert_allclose(pts[2], [1, 1, 2, 1, 2, 1])