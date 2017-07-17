import numpy as np
import pytest
import unittest.mock as mock
import io, pickle

import open_cp.sepp as testmod
import open_cp.data

def uniform_data(length=10):
    times = np.arange(length) * 0.1
    xcoords = np.arange(length)
    ycoords = -np.arange(length)
    return np.vstack([times, xcoords, ycoords])

def test__normalise_matrix():
    p = np.random.random(size=(20,20))
    q = testmod._normalise_matrix(p)
    np.testing.assert_allclose( np.sum(q, axis=0), np.zeros(20)+1 )
    
def expected_initial_matrix(points):
    size= points.shape[-1]
    p = np.zeros((size, size))
    for j in range(size):
        p[j][j] = 1
        for i in range(j):
            td = points[0][j] - points[0][i]
            t = np.exp( - td / 0.1 )
            xd = points[1][j] - points[1][i]
            yd = points[2][j] - points[2][i]
            s = np.exp( -(xd**2 + yd**2) / (2*50*50))
            p[i][j] = t*s
    return testmod._normalise_matrix(p)
    
def test_initial_p_matrix():
    points = uniform_data()
    p = testmod.initial_p_matrix(points)
    assert( p.shape == (10,10) )
    np.testing.assert_allclose(p, expected_initial_matrix(points))
    
def test_sample_points_all_background():
    points = uniform_data()
    p = np.zeros((10,10))
    for i in range(10):
        p[i][i] = 1
    backs, trigs = testmod.sample_points(points, p)
    assert(trigs.shape[-1] == 0)
    np.testing.assert_allclose(backs, points)
    
def test_sample_points():
    points = uniform_data(4)
    p = np.zeros((4,4))
    for j, i in enumerate([0,1,0,0]):
        p[i, j] = 1
    backs, trigs = testmod.sample_points(points, p)
    assert(backs.shape == (3,2))
    assert(trigs.shape == (3,2))
    np.testing.assert_allclose(backs, points[:,:2])
    np.testing.assert_allclose(trigs[:,0], [0.2, 2, -2] )
    np.testing.assert_allclose(trigs[:,1], [0.3, 3, -3] )
    
def test_p_matrix():
    def bk(pts):
        return np.zeros_like(pts[0]) + 1
    def tk(pts):
        return np.zeros_like(pts[0]) + 0.5
    points = np.empty((3,2))
    p = testmod.p_matrix(points, bk, tk)
    expected = np.asarray([[1,0.5], [0,1]])
    np.testing.assert_allclose(p, testmod._normalise_matrix(expected))

def test_p_matrix_fast():
    def bk(pts):
        return pts[0]**2
    def tk(pts):
        return pts[1]**2
    points = np.random.random(size=(3,10))
    p = testmod.p_matrix(points, bk, tk)
    pf = testmod.p_matrix_fast(points, bk, tk, time_cutoff=2, space_cutoff=2)
    np.testing.assert_allclose(p, pf)

def test_p_matrix_fast_timecutoff():
    def bk(pts):
        return np.zeros_like(pts[0]) + 0.5
    def tk(pts):
        return np.zeros_like(pts[0]) + 1
    points = np.asarray([[0,1,2], [1,2,3]]).T
    p = np.asarray([[1, 0], [0, 1]])
    pf = testmod.p_matrix_fast(points, bk, tk, time_cutoff=0.9, space_cutoff=2)
    np.testing.assert_allclose(p, pf)
    pf = testmod.p_matrix_fast(points, bk, tk, time_cutoff=1, space_cutoff=1.4)
    np.testing.assert_allclose(p, pf)
    pf = testmod.p_matrix_fast(points, bk, tk, time_cutoff=1, space_cutoff=2)
    assert(pf[0][1] > 0.1)    

def test_make_kernel():
    def bk(pts):
        return pts[1]
    def tk(pts):
        return pts[2]
    # Events at (0,2,4) and (1,3,5)
    data = np.array([[0,1], [2,3], [4,5]])
    kernel = testmod.make_kernel(data, bk, tk)
    pts = np.array([[0,5.7,2], [0.1,5.7,2], [1.1,5.7,2]]).T
    # Event at (0, 5.7, 2) so no triggers
    assert(kernel(pts[:,0]) == pytest.approx(5.7))
    # Event at (0.1, 5.7, 2) so trigger (0, 2, 4) with relative position
    #    (0.1, 3.7, -2)
    assert(kernel(pts[:,1]) == pytest.approx(5.7 - 2))
    # Event at (1.1, 5.7, 2) so both events trigger, with relative positions
    #    (1.1, 3.7, -2) and (0.1, 2.7, -3)
    assert(kernel(pts[:,2]) == pytest.approx(5.7 - 2 - 3))
    np.testing.assert_allclose(kernel(pts), [5.7, 3.7, 0.7])

def test_make_space_kernel():
    def bk(pts):
        return pts[2]
    def tk(pts):
        return pts[2]
    # Events at (0,2,4) and (1,3,5)
    data = np.array([[0,1], [2,3], [4,5]])
    pts = np.array([[5.7, 2], [2.3, 4], [1, 4.2]]).T

    # Time 0, so no triggers
    kernel = testmod.make_space_kernel(data, bk, tk, time=0)
    assert( kernel(pts[:,0]) == pytest.approx(2) )
    assert( kernel(pts[:,1]) == pytest.approx(4) )
    assert( kernel(pts[:,2]) == pytest.approx(4.2) )
    np.testing.assert_allclose(kernel(pts), [2, 4, 4.2])

    # Time 1, so one trigger
    kernel = testmod.make_space_kernel(data, bk, tk, time=1)
    assert( kernel(pts[:,0]) == pytest.approx(2 - 2) )
    assert( kernel(pts[:,1]) == pytest.approx(4 + 0) )
    assert( kernel(pts[:,2]) == pytest.approx(4.2 + 0.2) )
    np.testing.assert_allclose(kernel(pts), [0, 4, 4.4])
    
    # Time 2 so two triggers
    kernel = testmod.make_space_kernel(data, bk, tk, time=2)
    assert( kernel(pts[:,0]) == pytest.approx(2 - 2 - 3) )
    assert( kernel(pts[:,1]) == pytest.approx(4 + 0 - 1) )
    assert( kernel(pts[:,2]) == pytest.approx(4.2 + 0.2 - .8) )
    np.testing.assert_allclose(kernel(pts), [-3, 3, 3.6])

    # Should now only see the 2nd event
    kernel = testmod.make_space_kernel(data, bk, tk, time=1.5, time_cutoff=1)
    assert( kernel(pts[:,0]) == pytest.approx(2 - 3) )
    assert( kernel(pts[:,1]) == pytest.approx(4 - 1) )
    assert( kernel(pts[:,2]) == pytest.approx(4.2 - .8) )
    np.testing.assert_allclose(kernel(pts), [-1, 3, 3.4])
    
    # Should now only see the background
    kernel = testmod.make_space_kernel(data, bk, tk, time=1.5, space_cutoff=0.1)
    assert( kernel(pts[:,0]) == pytest.approx(2) )
    assert( kernel(pts[:,1]) == pytest.approx(4) )
    assert( kernel(pts[:,2]) == pytest.approx(4.2) )
    np.testing.assert_allclose(kernel(pts), [2, 4, 4.2])
    
def test_pickle_result():
    trainer = testmod.SEPPTrainer()
    times = [np.datetime64("2017-05-10") + np.timedelta64(1,"h") * i for i in range(100)]
    xcs = np.random.random(size=100)
    ycs = np.random.random(size=100)
    data = open_cp.data.TimedPoints.from_coords(times, xcs, ycs)
    trainer.data = data
    result = trainer.train(iterations=1)
    
    with io.BytesIO() as file:
        pickle.dump(result, file)
        
    result.data = data
    result.predict(np.datetime64("2017-05-11"))
    