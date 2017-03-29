import numpy as np

import open_cp.sepp as testmod

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
            t = np.exp( - 0.1 * td )
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
    

    