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

def a_valid_grid_prediction_array():
    matrix = np.array([[1,2,3], [4,5,6]])
    return GridPredictionArray(10, 10, matrix)
    

def test_GridPredictionArray():
    gpa = a_valid_grid_prediction_array()
    assert gpa.grid_risk(-1, 0) == 0
    assert gpa.grid_risk(0, -1) == 0
    assert gpa.grid_risk(0, 0) == 1
    assert gpa.grid_risk(2, 1) == 6
    assert gpa.grid_risk(2, 0) == 3
    assert gpa.grid_risk(3, 0) == 0
    assert gpa.grid_risk(0, 2) == 0
                        
def test_GridPredictionArray_intensity_matrix_property():
    gpa = a_valid_grid_prediction_array()
    np.testing.assert_allclose( gpa.intensity_matrix, [[1,2,3], [4,5,6]] )    
    
def test_GridPredictionArray_mesh_data():
    gpa = a_valid_grid_prediction_array()
    xcs, ycs = gpa.mesh_data()
    np.testing.assert_allclose( xcs, [0, 10, 20, 30] )
    np.testing.assert_allclose( ycs, [0, 10, 20] )

def test_GridPredictionArray_percentiles():
    matrix = np.array([[4,6,6], [1,4,4]])
    gpa = GridPredictionArray(10, 10, matrix)
    pm = gpa.percentile_matrix()
    assert( pm.shape == (2,3) )
    p = 1/6
    np.testing.assert_allclose( pm[0], [4*p,1,1] )
    np.testing.assert_allclose( pm[1], [p,4*p,4*p] )


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

def test_sample_to_grid_xy():
    grid = sample_to_grid(x_y_kernel, cell_width=10, cell_height=20, width=10, height=15)
    diffs = []
    for x in range(10):
        for y in range(15):
            diffs.append( grid.grid_risk(x,y) - y * 20 - x * 10)
    assert sum( x >= 13 and x <= 17 for x in diffs) > 100
