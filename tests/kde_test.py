import pytest
import open_cp.kde as kde
import open_cp.data

import numpy as np
from unittest import mock
import datetime

@pytest.fixture
def data1():
    times = [np.datetime64("2017-04-21"), np.datetime64("2017-04-22"), np.datetime64("2017-04-23")]
    xcs = [5, 50, 20]
    ycs = [30, 20, 10]
    return open_cp.data.TimedPoints(times, [xcs, ycs])

def test_construct(data1):
    region = open_cp.data.RectangularRegion(xmin=0, xmax=90, ymin=20, ymax=50)
    predictor = kde.KDE(region, 20)
    
    predictor.data = data1
    pred = predictor.predict()
    
    assert pred.xsize == 20
    assert pred.ysize == 20
    assert pred.region().min == (0, 20)
    # Scaled up as grid size is 20
    assert pred.region().max == (100, 60)
    
@mock.patch("open_cp.kernels.GaussianBase")
def test_kernel_constructed_correctly(gb_mock, data1):
    region = open_cp.data.RectangularRegion(xmin=0, xmax=100, ymin=20, ymax=50)
    predictor = kde.KDE(region, 20)
    predictor.data = data1
    predictor.predict()

    assert len(gb_mock.call_args_list) == 1
    call = gb_mock.call_args_list[0]
    assert len(call[0]) == 1
    np.testing.assert_allclose(call[0][0], [[5,50,20], [30,20,10]])
    np.testing.assert_allclose(gb_mock.return_value.weights, [1,1,1])
    
    print(gb_mock.return_value.call_args_list)

@mock.patch("open_cp.kernels.GaussianBase")
def test_time_delta_usage(gb_mock, data1):
    region = open_cp.data.RectangularRegion(xmin=0, xmax=100, ymin=20, ymax=50)
    predictor = kde.KDE(region, 20)
    predictor.data = data1
    predictor.time_kernel = mock.MagicMock()
    predictor.time_unit = datetime.timedelta(hours=1)
    predictor.predict()

    assert len(predictor.time_kernel.call_args_list) == 1
    assert len(predictor.time_kernel.call_args_list[0][0]) == 1
    np.testing.assert_allclose(predictor.time_kernel.call_args_list[0][0][0], [48, 24, 0])

@mock.patch("open_cp.kernels.GaussianBase")
def test_time_delta_usage_with_end_time(gb_mock, data1):
    region = open_cp.data.RectangularRegion(xmin=0, xmax=100, ymin=20, ymax=50)
    predictor = kde.KDE(region, 20)
    predictor.data = data1
    predictor.time_kernel = mock.MagicMock()
    predictor.time_unit = datetime.timedelta(hours=1)
    predictor.predict(end_time = datetime.datetime(2017,4,22,7))

    assert len(predictor.time_kernel.call_args_list) == 1
    assert len(predictor.time_kernel.call_args_list[0][0]) == 1
    np.testing.assert_allclose(predictor.time_kernel.call_args_list[0][0][0], [24+7, 7])

    assert len(gb_mock.call_args_list) == 1
    call = gb_mock.call_args_list[0]
    np.testing.assert_allclose(call[0][0], [[5,50], [30,20]])

@mock.patch("open_cp.kernels.GaussianBase")
def test_time_delta_usage_with_start_time(gb_mock, data1):
    region = open_cp.data.RectangularRegion(xmin=0, xmax=100, ymin=20, ymax=50)
    predictor = kde.KDE(region, 20)
    predictor.data = data1
    predictor.time_kernel = mock.MagicMock()
    predictor.time_unit = datetime.timedelta(hours=1)
    predictor.predict(start_time = datetime.datetime(2017,4,22))

    assert len(predictor.time_kernel.call_args_list) == 1
    assert len(predictor.time_kernel.call_args_list[0][0]) == 1
    np.testing.assert_allclose(predictor.time_kernel.call_args_list[0][0][0], [24, 0])

    assert len(gb_mock.call_args_list) == 1
    call = gb_mock.call_args_list[0]
    np.testing.assert_allclose(call[0][0], [[50,20], [20,10]])

def test_exp_time_kernel():
    kernel = kde.ExponentialTimeKernel(1)
    assert kernel(1) == pytest.approx(np.exp(-1))
    assert kernel(2) == pytest.approx(np.exp(-2))
    np.testing.assert_allclose(kernel([1,2]), np.exp([-1,-2]))

    kernel = kde.ExponentialTimeKernel(2)
    assert kernel(1) == pytest.approx(np.exp(-1/2))
    assert kernel(2) == pytest.approx(np.exp(-1))
    np.testing.assert_allclose(kernel([1,2]), np.exp([-0.5,-1]))

def test_quad_time_kernel():
    kernel = kde.QuadDecayTimeKernel(1)
    assert kernel(1) == pytest.approx(1/(1+1))
    assert kernel(2) == pytest.approx(1/(1+4))
    np.testing.assert_allclose(kernel([1,2]), [0.5, 0.2])

    kernel = kde.QuadDecayTimeKernel(2)
    assert kernel(1) == pytest.approx(1/(1+1/4))
    assert kernel(2) == pytest.approx(1/(1+1))
    np.testing.assert_allclose(kernel([1,2]), [4/5, 0.5])
    