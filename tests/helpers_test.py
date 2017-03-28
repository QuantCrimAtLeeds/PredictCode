import pytest
import tests.helpers as helpers
from unittest.mock import patch

import numpy as np

@patch("numpy.random.random", helpers.RandomCyclicBuffer([.1,.2,.3,.4]))
def test_RandomCyclicBuffer_single():
    assert( np.random.random() == 0.1 )
    
@patch("numpy.random.random", helpers.RandomCyclicBuffer([.1,.2,.3,.4]))
def test_RandomCyclicBuffer_1D():
    np.testing.assert_allclose( np.random.random(size=5), [.1,.2,.3,.4,.1] )
    
@patch("numpy.random.random", helpers.RandomCyclicBuffer([.1,.2,.3,.4]))
def test_RandomCyclicBuffer_2D():
    np.testing.assert_allclose( np.random.random(size=(2,3)),
        np.array([[.1,.2,.3],[.4,.1,.2]]) )
    
@patch("numpy.random.random", helpers.RandomCyclicBuffer([.1,.2,.3,.4]))
def test_RandomCyclicBuffer_3D():
    np.testing.assert_allclose( np.random.random(size=(2,1,3)),
        np.array([[[.1,.2,.3]],[[.4,.1,.2]]]) )
    

@patch("builtins.open", helpers.MockOpen("fish"))
def test_MockOpen():
    with open("file.txt") as file:
        assert( next(file) == "fish" )
        with pytest.raises(StopIteration):
            next(file)
            
    with pytest.raises(FileNotFoundError):
        with open("doesntexists.txt") as file:
            pass