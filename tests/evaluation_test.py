import pytest

import open_cp.evaluation as evaluation
import numpy as np

def test_top_slice():
    x = np.array([1,2,3,4,5,6])
    mask = evaluation.top_slice(x, 0.7)
    # 6 * 0.7 == 4.2
    np.testing.assert_array_equal(x>2, mask)

    mask = evaluation.top_slice(x, 0.6)
    # 6 * 0.6 == 3.6
    np.testing.assert_array_equal(x>3, mask)

    mask = evaluation.top_slice(x, 1.0)
    np.testing.assert_array_equal(x>0, mask)
    
    mask = evaluation.top_slice(x, 0.0)
    np.testing.assert_array_equal(x>7, mask)
    
def test_top_slice_not_one_dim():
    x = np.array([1,2,3,4,5,6]).reshape(2,3)
    mask = evaluation.top_slice(x, 0.7)
    assert( mask.shape == (2,3) )
    np.testing.assert_array_equal(x>=3, mask)

def test_top_slice_equal_input():
    x = np.array([1,1,1,2,3,4])
    mask = evaluation.top_slice(x, 0.7)
    np.testing.assert_array_equal([False,False,True,True,True,True], mask)

    x = np.array([2,3,4,1,1,1])
    mask = evaluation.top_slice(x, 0.7)
    np.testing.assert_array_equal([True,True,True,False,False,True], mask)
    
def test_top_slice_random_data():
    for _ in range(100):
        x = np.random.random(size=123)
        if len(set(x)) != len(x):
            continue
        t = np.random.random()
        n = int(len(x) * t)
        mask = evaluation.top_slice(x, t)
        assert( mask.shape == x.shape )
        assert( np.sum(mask) == n )
        y = x[mask]
        if len(y) == 0:
            expected = np.zeros_like(mask)
        else:
            expected = x >= np.min(y)
        np.testing.assert_array_equal(expected, mask)
        
def test_top_slice_random_data_with_repeats():
    for _ in range(100):
        x = np.random.randint(low=0, high=2, size=47)
        t = np.random.random()
        n = int(len(x) * t)
        mask = evaluation.top_slice(x, t)
        assert( mask.shape == x.shape )
        assert( np.sum(mask) == n )
        if n == 0:
            continue
        expected_super_set = x >= np.min(x[mask])
        np.testing.assert_array_equal(mask * expected_super_set, mask)
        
def test_top_slice_masked():
    data = np.asarray([1,2,3,0,4,5])
    data = np.ma.masked_array(data, mask = [True, False, False, False, True, False])

    s = evaluation.top_slice(data, 1.0)
    np.testing.assert_array_equal([False, True, True, True, False, True], s)

    s = evaluation.top_slice(data, 0.0)
    assert( not np.any(s) )

    s = evaluation.top_slice(data, 0.5)
    np.testing.assert_array_equal([False, False, True, False, False, True], s)

    s = evaluation.top_slice(data, 0.4)
    np.testing.assert_array_equal([False, False, False, False, False, True], s)