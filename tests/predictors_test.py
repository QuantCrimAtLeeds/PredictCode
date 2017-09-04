import pytest
from unittest.mock import patch, MagicMock

import open_cp.predictors as testmod
import open_cp.data

import numpy as np

def test_DataTrainer():
    test = testmod.DataTrainer()
    with pytest.raises(TypeError):
        test.data = "string"
    test.data = testmod.data.TimedPoints([],[[],[]])


def test_GridPrediction_grid_coord():
    g = testmod.GridPrediction(10, 20, xoffset=5, yoffset=15)
    assert(g.grid_coord(5, 15) == (0,0))
    assert(g.grid_coord(14, 34) == (0,0))
    assert(g.grid_coord(15, 34) == (1,0))
    assert(g.grid_coord(14, 35) == (0,1))
    assert(g.grid_coord(15, 35) == (1,1))

def test_GridPrediction_bounding_box_of_cell():
    g = testmod.GridPrediction(10, 20, xoffset=5, yoffset=15)
    bb = g.bounding_box_of_cell(0,0)
    assert(bb.min == (5,15))
    assert(bb.max == (15,35))

def test_GridPrediction():
    class Test(testmod.GridPrediction):
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
    class Test(testmod.GridPrediction):
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

def a_valid_grid_prediction_array():
    matrix = np.array([[1,2,3], [4,5,6]])
    return testmod.GridPredictionArray(10, 10, matrix)

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

def test_GridPredictionArray_clone():
    matrix = np.array([[1,2,3], [4,5,6]])
    gpa = testmod.GridPredictionArray(5, 10, matrix, 1, 2)
    cl = gpa.clone()
    assert (gpa.xoffset, gpa.yoffset) == (cl.xoffset, cl.yoffset)
    assert (gpa.xextent, gpa.yextent) == (cl.xextent, cl.yextent)
    assert (gpa.xsize, gpa.ysize) == (cl.xsize, cl.ysize)
    np.testing.assert_allclose(gpa.intensity_matrix, cl.intensity_matrix)
    cl.intensity_matrix[0] = [7,8,9]
    np.testing.assert_allclose(gpa.intensity_matrix, [[1,2,3],[4,5,6]])
    np.testing.assert_allclose(cl.intensity_matrix, [[7,8,9],[4,5,6]])

def test_GridPredictionArray_masked_clone():
    mask = np.array([[True, True, False], [False, False, True]])
    matrix = np.ma.masked_array([[1,2,3], [4,5,6]], mask=mask)
    gpa = testmod.GridPredictionArray(5, 10, matrix, 1, 2)
    cl = gpa.clone()
    assert (gpa.xoffset, gpa.yoffset) == (cl.xoffset, cl.yoffset)
    assert (gpa.xextent, gpa.yextent) == (cl.xextent, cl.yextent)
    assert (gpa.xsize, gpa.ysize) == (cl.xsize, cl.ysize)
    np.testing.assert_allclose(gpa.intensity_matrix, cl.intensity_matrix)
    cl.intensity_matrix[0] = [7,8,9]
    cl.intensity_matrix.mask[0] = [True, True, False]
    cl.intensity_matrix.mask[1,1] = True
    np.testing.assert_allclose(gpa.intensity_matrix, [[1,2,3],[4,5,6]])
    np.testing.assert_allclose(cl.intensity_matrix, [[7,8,9],[4,5,6]])
    np.testing.assert_equal(gpa.intensity_matrix.mask, [[True, True, False], [False, False, True]])
    np.testing.assert_equal(cl.intensity_matrix.mask, [[True, True, False], [False, True, True]])

def test_GridPredictionArray_new_extent():
    matrix = np.array([[1,2,3], [4,5,6]])
    gpa = testmod.GridPredictionArray(5, 10, matrix, 1, 2)

    cl = gpa.new_extent(6, 12, 3, 4)    
    assert (gpa.xsize, gpa.ysize) == (cl.xsize, cl.ysize)
    assert (cl.xoffset, cl.yoffset) == (6, 12)
    assert (cl.xextent, cl.yextent) == (3, 4)
    np.testing.assert_allclose(cl.intensity_matrix, [
        [5, 6, 0], [0,0,0], [0,0,0], [0,0,0] ])
    np.testing.assert_allclose(gpa.intensity_matrix, [[1,2,3], [4,5,6]])

    with pytest.raises(ValueError):
        gpa.new_extent(5, 12, 3, 4)
    with pytest.raises(ValueError):
        gpa.new_extent(6, 11, 3, 4)

def test_GridPredictionArray_percentiles():
    matrix = np.array([[4,6,6], [1,4,4]])
    gpa = testmod.GridPredictionArray(10, 10, matrix)
    pm = gpa.percentile_matrix()
    assert( pm.shape == (2,3) )
    p = 1/6
    np.testing.assert_allclose( pm[0], [4*p,1,1] )
    np.testing.assert_allclose( pm[1], [p,4*p,4*p] )

@patch("numpy.random.random")
def test_ContinuousPrediction_samples_to_grid(random_mock):
    random_mock.return_value = np.array([0.1, 0.2, 0.4, 0.9])
    class Test(testmod.ContinuousPrediction):
        def risk(self, x, y):
            return x + y
    
    # Samples x in [50,100] and y in [100,150]
    # So x = 55, 60, 70, 95
    #    y = 105, 110, 120, 145
    expected = ( 160 + 170 + 190 + 240 ) /4
    assert( Test().grid_risk(1, 2) == pytest.approx(expected) )

def test_ContinuousPrediction_to_kernel():
    class Test(testmod.ContinuousPrediction):
        def __init__(self):
            super().__init__()
            self.called_with = []

        def risk(self, x, y):
            self.called_with.append((x,y))

    test = Test()
    kernel = test.to_kernel()
    kernel(np.array([1,2]))
    assert( test.called_with[0] == (1,2) )
    kernel(np.array([[3,4],[5,6]]))
    np.testing.assert_allclose( test.called_with[1], ([3,4],[5,6]) )

@patch("numpy.random.random")
def test_ContinuousPrediction_rebase(random_mock):
    random_mock.return_value = np.array([0.1, 0.2, 0.4, 0.9])
    class Test(testmod.ContinuousPrediction):
        def risk(self, x, y):
            return x + y
    
    test = Test().rebase(10, 10, 5, 15)
    
    # Samples x in [15,25] and y in [35,45]
    # So x = 16,17,19,24
    #    y = 36,37,39,44
    expected = ( 16+17+19+24+36+37+39+44 ) / 4
    assert( test.grid_risk(1, 2) == pytest.approx(expected) )


@patch("numpy.random.random")
def test_KernelRiskPredictor(random_mock):
    random_mock.return_value = np.array([0.1, 0.2])
    kernel = MagicMock()
    kernel.return_value = 5
    test = testmod.KernelRiskPredictor(kernel, cell_width=20)
    test.grid_risk(0,0)
    np.testing.assert_allclose(kernel.call_args[0][0], [[2, 4], [5, 10]])
    

def test_from_continuous_prediction():
    class Test(testmod.ContinuousPrediction):
        def risk(self, x, y):
            x = np.asarray(x)
            return ((x >= 100) & (x < 150)).astype(np.float)

    test = testmod.GridPredictionArray.from_continuous_prediction(Test(), 5, 10)
    assert(test.intensity_matrix.shape == (10, 5))
    # (2,3) -> [100,150] x [150,200]
    assert(test.grid_risk(2,3) == 1)
    assert(test.grid_risk(2,9) == 1)
    assert(test.grid_risk(3,3) == 0)
    assert(test.grid_risk(2,10) == 0)

def test_continuous_prediction_samples():
    cp = testmod.ContinuousPrediction(20, 30, 0, 0, samples = 123)
    assert cp.samples == 123

    cp = testmod.ContinuousPrediction(20, 30, 0, 0)
    assert cp.samples == 3

    cp = testmod.ContinuousPrediction(200, 100, 0, 0)
    assert cp.samples == 100
    
def test_GridPredictionArray_renormalise():
    matrix = np.array([[1,2,3], [4,5,6]])
    gpa = testmod.GridPredictionArray(10, 10, matrix, xoffset=2, yoffset=7)
    gpa2 = gpa.renormalise()
    np.testing.assert_allclose(np.array([[1,2,3], [4,5,6]]), gpa.intensity_matrix)
    np.testing.assert_allclose(np.array([[1,2,3], [4,5,6]]) / 21, gpa2.intensity_matrix)
    assert gpa2.xoffset == 2
    assert gpa2.yoffset == 7
    assert gpa2.xsize == 10
    assert gpa2.ysize == 10
    
    mm = np.ma.masked_array(matrix, mask=[[True,False,False], [False,True,False]])
    gpa = testmod.GridPredictionArray(10, 10, mm, xoffset=2, yoffset=7)
    gpa2 = gpa.renormalise()
    np.testing.assert_allclose(np.array([[1,2,3], [4,5,6]]) / 15, gpa2.intensity_matrix)
    