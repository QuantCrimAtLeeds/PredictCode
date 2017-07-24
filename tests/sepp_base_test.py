import pytest
import unittest.mock as mock

import open_cp.sepp_base as sepp_base
import numpy as np

class OurModel(sepp_base.ModelBase):
    def background(self, points):
        assert len(points.shape) == 2
        assert points.shape[0] == 3
        return points[0] * np.exp(-(points[1]**2 + points[2]**2))
    
    def trigger(self, pt, dpts):
        assert pt.shape == (3,)
        assert len(dpts.shape) == 2
        assert dpts.shape[0] == 3
        w = np.sum(np.abs(pt))
        return dpts[0] * np.exp(-(dpts[1]**2 + dpts[2]**2) / w)
        

def slow_p_matrix(model, points):
    assert points.shape[0] == 3
    d = points.shape[1]
    
    p = np.zeros((d,d))
    for i in range(d):
        pt = points[:,i]
        p[i,i] = model.background(pt[:,None])
        for j in range(i):
            dp = pt - points[:,j]
            p[j,i] = model.trigger(pt, dp[:,None])
            
    for i in range(d):
        p[:,i] /= np.sum(p[:,i])
        
    return p

def test_p_matrix():
    model = OurModel()
    for _ in range(10):
        points = np.random.random((3,20))
        points[0].sort()
        expected = slow_p_matrix(model, points)
        got = sepp_base.p_matrix(model, points)
        np.testing.assert_allclose(got, expected)

@pytest.fixture
def p_matrix_mock():
    with mock.patch("open_cp.sepp_base.p_matrix") as m:
        m.return_value = [[1, 0.5, 0.1, 0.2],
                          [0, 0.5, 0.6, 0.4],
                          [0, 0,   0.3, 0.3],
                          [0, 0,   0,   0.1]]
        yield m
        
@pytest.fixture
def model():
    return OurModel()
        
@pytest.fixture
def points():
    return np.asarray([ [0,1,2,3], [1,4,7,9], [8,6,4,2] ])

@pytest.fixture
def optimiser(p_matrix_mock, model, points):
    yield sepp_base.Optimiser(model, points)

def test_Optimiser_p_diag(optimiser):
    assert optimiser.p.shape == (4,4)
    np.testing.assert_allclose(optimiser.p_diag, [1, 0.5, 0.3, 0.1])
    assert optimiser.p_diag_sum == pytest.approx(1.9)
    
def test_Optimiser_p_upper_tri_sum(optimiser):
    assert optimiser.p_upper_tri_sum == pytest.approx(0.5 + 0.7 + 0.9)
    
def test_Optimiser_upper_tri_col(optimiser):
    optimiser.upper_tri_col(0) == np.asarray([])
    optimiser.upper_tri_col(1) == np.asarray([0.5])
    optimiser.upper_tri_col(2) == np.asarray([0.1, 0.6])
    optimiser.upper_tri_col(3) == np.asarray([0.2, 0.4, 0.3])
    
    optimiser.diff_col_times(0) == np.asarray([])
    optimiser.diff_col_times(1) == np.asarray([1])
    optimiser.diff_col_times(2) == np.asarray([2,1])
    optimiser.diff_col_times(3) == np.asarray([3,2,1])

    optimiser.diff_col_points(0) == np.asarray([])
    optimiser.diff_col_points(1) == np.asarray([[3], [-2]])
    optimiser.diff_col_points(1) == np.asarray([[6,3], [4,2]])
    optimiser.diff_col_points(1) == np.asarray([[8,5,2], [6,4,2]])

    
def test_Optimiser(optimiser):
    assert optimiser.num_points == 4
    