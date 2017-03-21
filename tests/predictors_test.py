import pytest

from open_cp.predictors import *

def test_GridPrediction():
    class Test(GridPrediction):
        def __init__(self, xsize, ysize, xoffset = 0, yoffset = 0):
            super().__init__(xsize, ysize, xoffset, yoffset)
            
        def grid_risk(self, gx, gy):
            assert self.want == (gx, gy)

    test = Test(10, 10)
    
    test.want = (0, 0)
    test.risk(5, 6)

    test.want = (1, 2)
    test.risk(12, 21)

def test_GridPrediction_with_offset():
    class Test(GridPrediction):
        def __init__(self, xsize, ysize, xoffset = 0, yoffset = 0):
            super().__init__(xsize, ysize, xoffset, yoffset)
            
        def grid_risk(self, gx, gy):
            assert self.want == (gx, gy)

    test = Test(10, 10, 25, 30)
    
    test.want = (0, 0)
    test.risk(25, 30)

    test.want = (0, 0)
    test.risk(34, 39)

    test.want = (1, 2)
    test.risk(25 + 15, 30 + 20 + 8)


def test_ContinuousPrediction_cannot_make_grid_prediction():
    class Test(ContinuousPrediction):
        def risk(self, x, y):
            pass
    
    with pytest.raises(TypeError):
        Test().grid_risk(1, 2)


import numpy as np

def test_GridPredictionArray():
    matrix = np.array([ [1,2,3], [4,5,6]])
    gpa = GridPredictionArray(10, 10, matrix)
    assert gpa.grid_risk(-1, 0) == 0
    assert gpa.grid_risk(0, -1) == 0
    assert gpa.grid_risk(0, 0) == 1
    assert gpa.grid_risk(2, 1) == 6
    assert gpa.grid_risk(2, 0) == 3
    assert gpa.grid_risk(3, 0) == 0
    assert gpa.grid_risk(0, 2) == 0

def x_kernel(points):
    return points[0]

def y_kernel(points):
    return points[1]

def x_y_kernel(points):
    return points[0] + points[1]

def test_sample_to_grid_x():
    grid = sample_to_grid(x_kernel, cell_width=10, cell_height=20, width=10, height=15)
    diffs = []
    for x in range(10):
        for y in range(15):
            diffs.append( grid.grid_risk(x,y) - x * 10)
    # Most differences should be around 5.0
    assert sum( x >= 4 and x <= 6 for x in diffs) > 100

def test_sample_to_grid_y():
    grid = sample_to_grid(y_kernel, cell_width=10, cell_height=20, width=10, height=15)
    diffs = []
    for x in range(10):
        for y in range(15):
            diffs.append( grid.grid_risk(x,y) - y * 20)
    assert sum( x >= 8.5 and x <= 11.5 for x in diffs) > 100
