import unittest.mock as mock
import pytest

import open_cp.evaluation as evaluation
import open_cp.predictors
import open_cp.data
import open_cp.network
import open_cp.retrohotspot
import numpy as np
import scipy.special
import datetime

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
    
@pytest.fixture
def prediction():
    matrix = np.array([[1,2,3,4], [5,6,7,8]])
    return open_cp.predictors.GridPredictionArray(xsize=10, ysize=20, matrix=matrix, xoffset=2, yoffset=3)

def test_generate_aggregated_cells(prediction):
    out = list(evaluation.generate_aggregated_cells(prediction.intensity_matrix, 1))
    assert all(x[1]==1 for x in out)
    np.testing.assert_allclose([x[0] for x in out], [1,2,3,4,5,6,7,8])

    out = list(evaluation.generate_aggregated_cells(prediction.intensity_matrix, 2))
    assert all(x[1]==4 for x in out)
    np.testing.assert_allclose([x[0] for x in out], [14, 18, 22])

def test_hit_rate(prediction):
    t = [np.datetime64("2017-01-01")] * 8
    x = 2 + 5 + 10 * np.array([0,1,2,3,0,1,2,3])
    y = 3 + 10 + 20 * np.array([0,0,0,0,1,1,1,1])
    tp = open_cp.data.TimedPoints.from_coords(t,x,y)
    out = evaluation.hit_rates(prediction, tp, {1,2,5})
    assert set(out.keys()) == {1,2,5}
    assert set(out.values()) == {0}

    out = evaluation.hit_rates(prediction, tp, {49, 50, 100})
    assert out[100] == pytest.approx(1.0)
    assert out[50] == pytest.approx(0.5)
    assert out[49] == pytest.approx(3/8)
    
def test_hit_rate_out_of_range(prediction):
    t = [np.datetime64("2017-01-01")] * 8
    x = 100 + 10 * np.array([0,1,2,3,0,1,2,3])
    y = 3 + 10 + 20 * np.array([0,0,0,0,1,1,1,1])
    tp = open_cp.data.TimedPoints.from_coords(t,x,y)
    out = evaluation.hit_rates(prediction, tp, {1, 5, 100})
    assert set(out.values()) == {0}
    
@pytest.fixture
def masked_prediction():
    mask = [[True, False, False, False], [True, True, False, True]]
    matrix = np.array([[1,2,3,4], [5,6,7,8]])
    matrix = np.ma.array(matrix, mask=mask)
    return open_cp.predictors.GridPredictionArray(xsize=10, ysize=20, matrix=matrix, xoffset=2, yoffset=3)

def test_generate_aggregated_cells_with_mask(masked_prediction):
    out = list(evaluation.generate_aggregated_cells(masked_prediction.intensity_matrix, 1))
    np.testing.assert_allclose([x[1] for x in out], [0,1,1,1,0,0,1,0])
    np.testing.assert_allclose([x[0] for x in out], [0,2,3,4,0,0,7,0])

    out = list(evaluation.generate_aggregated_cells(masked_prediction.intensity_matrix, 2))
    np.testing.assert_allclose([x[1] for x in out], [1,3,3])
    np.testing.assert_allclose([x[0] for x in out], [2,12,14])

def test_inverse_hit_rate_no_mask(prediction):
    t = [np.datetime64("2017-01-01")] * 8
    x = 2 + 3 + 10 * np.array([0,1,2,3,0,1,2,3])
    y = 3 + 8 + 20 * np.array([0,0,0,0,1,1,1,1])
    tp = open_cp.data.TimedPoints.from_coords(t,x,y)
    out = evaluation.inverse_hit_rates(prediction, tp)
    assert out == {100*x/8 : 100*x/8 for x in range(1,9)}

def test_inverse_hit_rate_no_mask1(prediction):
    t = [np.datetime64("2017-01-01")] * 4
    x = 2 + 3 + 10 * np.array([0,1,3,3])
    y = 3 + 8 + 20 * np.array([1,1,0,1])
    tp = open_cp.data.TimedPoints.from_coords(t,x,y)
    out = evaluation.inverse_hit_rates(prediction, tp)
    assert out == {25: 100/8, 50: 300/8, 75: 50, 100: 500/8}

def test_inverse_hit_rate(masked_prediction):
    t = [np.datetime64("2017-01-01")] * 4
    x = 2 + 3 + 10 * np.array([1,2,3,2])
    y = 3 + 8 + 20 * np.array([0,0,0,1])
    tp = open_cp.data.TimedPoints.from_coords(t,x,y)
    out = evaluation.inverse_hit_rates(masked_prediction, tp)
    assert out == {25*n:25*n for n in [1,2,3,4]}
    
def test_inverse_hit_rate1(masked_prediction):
    t = [np.datetime64("2017-01-01")] * 3
    x = [12,12, 22]
    y = [3, 3,  23]
    tp = open_cp.data.TimedPoints.from_coords(t,x,y)
    out = evaluation.inverse_hit_rates(masked_prediction, tp)
    assert out == {100/3:25, 100:100}

def test_yield_hit_rates_segments():
    assert list(evaluation.yield_hit_rates_segments([7,7,5,3,3,1], [1,1,0,1,2,0])) == [(2,1), (0,2), (3,4), (0,5)]
    assert list(evaluation.yield_hit_rates_segments([7,6,5,3,2,1], [1,1,0,1,2,0])) == [(x,i) for i, x in enumerate([1,1,0,1,2,0])]
    assert list(evaluation.yield_hit_rates_segments([1,1,1,1,1,1], [1,1,0,1,2,0])) == [(5,5)]

@pytest.fixture
def masked_prediction1(prediction):
    mask = [[True, False, False, False], [True, True, False, True]]
    matrix = np.array([[1,2,3,7], [5,6,7,8]])
    matrix = np.ma.array(matrix, mask=mask)
    return open_cp.predictors.GridPredictionArray(xsize=10, ysize=20, matrix=matrix, xoffset=2, yoffset=3)

def test_brier_score_mask(masked_prediction):
    masked_prediction1 = masked_prediction.renormalise()
    t = [np.datetime64("2017-01-01")] * 5
    x = [12,12, 22, 24,24]
    y = [3,3, 23, 5,5]
    tp = open_cp.data.TimedPoints.from_coords(t,x,y)
    score, skill = evaluation.brier_score(masked_prediction1, tp)
    expected = (2/16 - 2/5)**2 + (3/16 - 2/5)**2 + (7/16 - 1/5)**2 + (4/16)**2
    assert expected / (4*10*20) == pytest.approx(score)
    score_worst = ((2/16)**2 + (3/16)**2 + (7/16)**2 + (4/16)**2 + (2/5)**2 + (2/5)**2 + (1/5)**2) / 800
    skill_exp = 1 - score / score_worst
    assert skill_exp == pytest.approx(skill)

def test_brier_score_mask_agg(masked_prediction):
    masked_prediction1 = masked_prediction.renormalise()
    t = [np.datetime64("2017-01-01")] * 5
    x = [12,12, 22, 24,24]
    y = [3,3, 23, 5,5]
    tp = open_cp.data.TimedPoints.from_coords(t,x,y)
    
    score, skill = evaluation.multiscale_brier_score(masked_prediction1, tp, 1)
    score1, skill1 = evaluation.brier_score(masked_prediction1, tp)
    assert score == pytest.approx(score1)
    assert skill == pytest.approx(skill1)
    
    score, skill = evaluation.multiscale_brier_score(masked_prediction1, tp, 2)
    expected = (1/7)*(2/28 - 2/10)**2 + (3/7)*(12/28 - 5/10)**2 + (3/7)*(14/28 - 3/10)**2
    assert expected / 200 == pytest.approx(score)
    worst = (1/7)*((2/28)**2 + (2/10)**2) + (3/7)*((12/28)**2 + (5/10)**2) + (3/7)*((14/28)**2 + (3/10)**2)
    assert 1 - expected / worst == pytest.approx(skill)

def test_inverse_hit_rate2(masked_prediction1):
    t = [np.datetime64("2017-01-01")] * 3
    x = [12,12, 22]
    y = [3, 3,  23]
    tp = open_cp.data.TimedPoints.from_coords(t,x,y)
    out = evaluation.inverse_hit_rates(masked_prediction1, tp)
    assert out == {100/3: 50, 100: 100}

def test_likelihood(masked_prediction):
    t = [np.datetime64("2017-01-01")] * 5
    x = [12,12, 22, 24,24]
    y = [3,3, 23, 5,5]
    tp = open_cp.data.TimedPoints.from_coords(t,x,y)
    out = evaluation.likelihood(masked_prediction, tp)
    assert out == pytest.approx((np.log(2)+np.log(2)+np.log(7)+np.log(3)+np.log(3))/5)

def test_likelihood_in_mask(masked_prediction):
    t = [np.datetime64("2017-01-01")] * 5
    x = [2,12, 22, 24,24]
    y = [3,3, 23, 5,5]
    tp = open_cp.data.TimedPoints.from_coords(t,x,y)
    with pytest.raises(ValueError):
        evaluation.likelihood(masked_prediction, tp)

def test_likelihood_no_mask(prediction):
    t = [np.datetime64("2017-01-01")] * 5
    x = [12,12, 22, 24,24]
    y = [3,3, 23, 5,5]
    tp = open_cp.data.TimedPoints.from_coords(t,x,y)
    out = evaluation.likelihood(prediction, tp)
    assert out == pytest.approx((np.log(2)+np.log(2)+np.log(7)+np.log(3)+np.log(3))/5)

def test_likelihood_out_of_grid(prediction):
    t = [np.datetime64("2017-01-01")] * 5
    x = [0,12, 22, 24,24]
    y = [3,3, 23, 5,5]
    tp = open_cp.data.TimedPoints.from_coords(t,x,y)
    with pytest.raises(ValueError):
        evaluation.likelihood(prediction, tp)

@pytest.fixture
def prediction_with_zeros():
    matrix = np.array([[1,2,0,2],[0,1,3,0]])
    return open_cp.predictors.GridPredictionArray(xsize=10, ysize=20, matrix=matrix, xoffset=2, yoffset=3)

@pytest.fixture
def timed_pts_5():
    t = [np.datetime64("2017-01-01")] * 5
    x = [12,12, 22, 24,24]
    y = [3,3, 23, 5,5]
    return open_cp.data.TimedPoints.from_coords(t,x,y)

def _test_kl_log(x, y):
    if x <= 0:
        return 0.0
    if y <= 0:
        return x * (np.log(x) + 20)
    return x * (np.log(x) - np.log(y))

def test_kl_score(prediction_with_zeros, timed_pts_5):
    pred = prediction_with_zeros.renormalise()
    score = evaluation.kl_score(pred, timed_pts_5)
    a = [0, 2/5, 2/5, 0,   0, 0, 1/5, 0]
    b = [1/9, 2/9, 0, 2/9,   0, 1/9, 1/3, 0]
    expected = sum( _test_kl_log(x, y) for x, y in zip(a, b) )
    expected += sum( _test_kl_log(1-x, 1-y) for x, y in zip(a, b) )
    assert expected / (200 * 8) == pytest.approx(score)

def test_kl_score_multi(prediction_with_zeros, timed_pts_5):
    pred = prediction_with_zeros.renormalise()
    score = evaluation.multiscale_kl_score(pred, timed_pts_5, 1)
    score1 = evaluation.kl_score(pred, timed_pts_5)
    assert score == pytest.approx(score1)
    
    score = evaluation.multiscale_kl_score(pred, timed_pts_5, 2)
    a = [2/10, 5/10, 3/10]
    b = [4/15, 6/15, 5/15]
    expected = sum( _test_kl_log(x, y) for x, y in zip(a, b) )
    expected += sum( _test_kl_log(1-x, 1-y) for x, y in zip(a, b) )
    assert expected / (200 * 3) == pytest.approx(score)

def test_bayesian_dirichlet_prior():
    matrix = np.array([[1,2]])
    pred = open_cp.predictors.GridPredictionArray(xsize=10, ysize=20, matrix=matrix, xoffset=2, yoffset=3)
    t = [np.datetime64("2017-01-01")] * 3
    x = [2, 2, 12]
    y = [5] * 3
    tp = open_cp.data.TimedPoints.from_coords(t,x,y)
    score = evaluation.bayesian_dirichlet_prior(pred, tp, bias=300)
    
    exp = np.log(300 * 301 * 302 / (100 * 101 * 200))
    exp += np.sum(scipy.special.digamma([102, 201, 303]) * [2, 1, -3])
    
    assert exp == pytest.approx(score)

def test_bayesian_dirichlet_prior_masked():
    matrix = np.ma.array([[1,2,3]], mask=[False, False, True])
    pred = open_cp.predictors.GridPredictionArray(xsize=10, ysize=20, matrix=matrix, xoffset=2, yoffset=3)
    t = [np.datetime64("2017-01-01")] * 3
    x = [2, 2, 12]
    y = [5] * 3
    tp = open_cp.data.TimedPoints.from_coords(t,x,y)
    score = evaluation.bayesian_dirichlet_prior(pred, tp, bias=300)
    
    exp = np.log(300 * 301 * 302 / (100 * 101 * 200))
    exp += np.sum(scipy.special.digamma([102, 201, 303]) * [2, 1, -3])
    
    assert exp == pytest.approx(score)
    np.testing.assert_allclose(pred.intensity_matrix.data, [[1,2,3]])

def test_bayesian_predictive():
    matrix = np.ma.array([[1,2]], mask=[False, False])
    pred = open_cp.predictors.GridPredictionArray(xsize=10, ysize=20, matrix=matrix, xoffset=2, yoffset=3)
    t = [np.datetime64("2017-01-01")] * 3
    x = [2, 2, 12]
    y = [5] * 3
    tp = open_cp.data.TimedPoints.from_coords(t,x,y)
    score = evaluation.bayesian_predictive(pred, tp, bias=30)
    
    exp = (12 / 33) * np.log(36/33) + (21 / 33) * np.log((21*3)/(33*2))
    assert exp == pytest.approx(score)
    np.testing.assert_allclose(pred.intensity_matrix, [[1,2]])

def test_bayesian_predictive_with_zero():
    matrix = np.array([[0,2]])
    pred = open_cp.predictors.GridPredictionArray(xsize=10, ysize=20, matrix=matrix, xoffset=2, yoffset=3)
    t = [np.datetime64("2017-01-01")] * 3
    x = [2, 2, 12]
    y = [5] * 3
    tp = open_cp.data.TimedPoints.from_coords(t,x,y)
    score = evaluation.bayesian_predictive(pred, tp, bias=10, lower_bound = 0.001)
    
    a = np.asarray([0.001, 2]) / 2.001 * 10
    n = np.asarray([2, 1])
    print(a, n)
    w = (a + n) / (13)
    exp = np.sum(w * (np.log(w * 10 / a)))
    assert exp == pytest.approx(score)
    np.testing.assert_allclose(pred.intensity_matrix, [[0,2]])

@pytest.fixture
def prediction_with_zeros_and_mask():
    mask = [[True, False, False, False], [True, True, False, True]]
    matrix = np.array([[1,2,0,2],[0,1,3,0]])
    matrix = np.ma.array(matrix, mask=mask)
    return open_cp.predictors.GridPredictionArray(xsize=10, ysize=20, matrix=matrix, xoffset=2, yoffset=3)

def test_kl_score_masked(prediction_with_zeros_and_mask, timed_pts_5):
    pred = prediction_with_zeros_and_mask.renormalise()
    score = evaluation.kl_score(pred, timed_pts_5)
    a = [2/5, 2/5, 0,   1/5]
    b = [2/7, 0, 2/7,   3/7]
    expected = sum( _test_kl_log(x, y) for x, y in zip(a, b) )
    expected += sum( _test_kl_log(1-x, 1-y) for x, y in zip(a, b) )
    assert expected / (200 * 4) == pytest.approx(score)

def test_convert_to_precentiles():
    array = np.asarray([5,2,3,6,5])
    np.testing.assert_allclose(evaluation.convert_to_precentiles(array), [4/5,1/5,2/5,1,4/5])
    
    array = np.asarray([[5,2,2],[3,6,5]])
    np.testing.assert_allclose(evaluation.convert_to_precentiles(array),
                               [[5/6,2/6,2/6],[3/6,1,5/6]])

    array = np.ma.array([[5,2,2],[3,6,5]], mask=[[False]*3,[False]*3])
    np.testing.assert_allclose(evaluation.convert_to_precentiles(array),
                               [[5/6,2/6,2/6],[3/6,1,5/6]])

    array = np.ma.array([5,2,3,6,5], mask=[True, False, False, True, False])
    np.testing.assert_allclose(evaluation.convert_to_precentiles(array), [0,1/3,2/3,0,1])

    array = np.ma.array([[5,2,2],[3,6,5]], mask=[[True,False,True],[False]*3])
    np.testing.assert_allclose(evaluation.convert_to_precentiles(array),
                               [[0,1/4,0],[2/4,1,3/4]])

def test_ranking_score(prediction_with_zeros, timed_pts_5):
    rank = evaluation.ranking_score(prediction_with_zeros, timed_pts_5)
    np.testing.assert_allclose(rank, [7/8,7/8,1,3/8,3/8])

def test_ranking_score1(masked_prediction1, timed_pts_5):
    rank = evaluation.ranking_score(masked_prediction1, timed_pts_5)
    np.testing.assert_allclose(rank, [1/4,1/4,1,2/4,2/4])

def test_poisson_crps():
    import scipy.stats as stats
    for mean in (np.random.random(size=10) + 0.01)*10:
        for actual in np.random.randint(0,100,size=10):
            poi = stats.poisson(mean)
            expected = 0.0
            for i in range(110):
                if i < actual:
                    expected += poi.cdf(i)**2
                else:
                    expected += (1 - poi.cdf(i))**2
            assert evaluation.poisson_crps(mean, actual) == pytest.approx(expected)

def test_poisson_crps_score():
    matrix = np.array([[1,2],[3,4]])
    pred = open_cp.predictors.GridPredictionArray(xsize=10, ysize=20, matrix=matrix, xoffset=2, yoffset=3)
    pred = pred.renormalise()
    t = [np.datetime64("2017-01-01")] * 3
    x = [2, 2, 12]
    y = [5] * 3
    tp = open_cp.data.TimedPoints.from_coords(t,x,y)

    expected = (evaluation.poisson_crps(3/10, 2) + evaluation.poisson_crps(6/10, 1)
        + evaluation.poisson_crps(9/10, 0) + evaluation.poisson_crps(12/10, 0))
    
    assert evaluation.poisson_crps_score(pred, tp) == pytest.approx(expected)

def test_poisson_crps_score_with_mask():
    matrix = np.ma.array([[1,2],[3,4]], mask=[[False, False],[True, True]])
    pred = open_cp.predictors.GridPredictionArray(xsize=10, ysize=20, matrix=matrix, xoffset=2, yoffset=3)
    pred = pred.renormalise()
    t = [np.datetime64("2017-01-01")] * 3
    x = [2, 2, 12]
    y = [5] * 3
    tp = open_cp.data.TimedPoints.from_coords(t,x,y)

    expected = evaluation.poisson_crps(1, 2) + evaluation.poisson_crps(2, 1)
    
    assert evaluation.poisson_crps_score(pred, tp) == pytest.approx(expected)

@mock.patch("open_cp.evaluation._kernels")
def test_score_kde(kernels_mock):
    mask = np.asarray([[True, False, False], [False]*3])
    grid = open_cp.data.MaskedGrid(xsize=10, ysize=20, xoffset=5, yoffset=7, mask=mask)
    risk = np.arange(6).reshape(2,3)
    pred = open_cp.predictors.GridPredictionArray(xsize=10, ysize=20, xoffset=5, yoffset=7, matrix=risk)

    tps = mock.Mock()
    tps.number_data_points = 5
    def kernel(pt):
        return 1
    kernels_mock.GaussianEdgeCorrectGrid.return_value = kernel
    got = evaluation.score_kde(pred, tps, grid)

    coords, g = kernels_mock.GaussianEdgeCorrectGrid.call_args[0]
    assert coords[0] == tps.xcoords
    assert coords[1] == tps.ycoords
    assert g is grid
    
    expected = np.sum((np.arange(1,6) - 1/5)**2)
    assert got == pytest.approx(expected * 200)

def test_score_kde_throw_cases():
    mask = np.asarray([[True, False, False], [False]*3])
    grid = open_cp.data.MaskedGrid(xsize=10, ysize=20, xoffset=5, yoffset=7, mask=mask)
    risk = np.arange(6).reshape(2,3)
    pred = open_cp.predictors.GridPredictionArray(xsize=10, ysize=20, xoffset=5, yoffset=7, matrix=risk)

    tps = mock.Mock()
    tps.number_data_points = 2
    with pytest.raises(ValueError):
        evaluation.score_kde(pred, tps, grid)

    tps.number_data_points = 5
    pred = open_cp.predictors.GridPredictionArray(xsize=10, ysize=19, xoffset=5, yoffset=7, matrix=risk)
    with pytest.raises(ValueError):
        evaluation.score_kde(pred, tps, grid)
    
    pred = open_cp.predictors.GridPredictionArray(xsize=10, ysize=20, xoffset=4, yoffset=7, matrix=risk)
    with pytest.raises(ValueError):
        evaluation.score_kde(pred, tps, grid)

    risk = np.arange(8).reshape(2,4)
    pred = open_cp.predictors.GridPredictionArray(xsize=10, ysize=20, xoffset=5, yoffset=7, matrix=risk)
    with pytest.raises(ValueError):
        evaluation.score_kde(pred, tps, grid)

@mock.patch("open_cp.evaluation._kernels")
def test_score_kde_fixed_bandwidth(kernels_mock):
    mask = np.asarray([[True, False, False], [False]*3])
    grid = open_cp.data.MaskedGrid(xsize=10, ysize=20, xoffset=5, yoffset=7, mask=mask)
    risk = np.arange(6).reshape(2,3)
    pred = open_cp.predictors.GridPredictionArray(xsize=10, ysize=20, xoffset=5, yoffset=7, matrix=risk)

    tps = mock.Mock()
    tps.number_data_points = 5
    def kernel(pt):
        return 1
    kernels_mock.GaussianEdgeCorrectGrid.return_value = kernel
    evaluation.score_kde_fixed_bandwidth(pred, tps, grid, 12.3)

    np.testing.assert_allclose(kernels_mock.GaussianEdgeCorrectGrid.return_value.covariance_matrix,
                               [[1,0],[0,1]])
    assert kernels_mock.GaussianEdgeCorrectGrid.return_value.bandwidth == 12.3



def test_grid_risk_coverage_to_graph(prediction):
    b = open_cp.network.PlanarGraphBuilder()
    b.add_vertex(35,30)
    b.add_vertex(40,40)
    b.add_edge(0, 1)
    b.add_vertex(25,30)
    b.add_vertex(30,40)
    b.add_edge(2, 3)
    graph = b.build()
    
    g = evaluation.grid_risk_coverage_to_graph(prediction, graph, 12)
    assert g.number_edges == 0

    # 1 cell.  Cells are x=[2,12,22,32,42], y=[3,23,43]
    g = evaluation.grid_risk_coverage_to_graph(prediction, graph, 13)
    assert g.number_edges == 1
    assert g.edges[0] == (0, 1)

    g = evaluation.grid_risk_coverage_to_graph(prediction, graph, 25)
    assert g.number_edges == 2
    assert set(g.edges) == {(0,1), (2,3)}

def test_grid_risk_coverage_to_graph_cutoff(prediction):
    b = open_cp.network.PlanarGraphBuilder()
    b.add_vertex(30,30)
    b.add_vertex(34,30)
    b.add_edge(0, 1)
    graph = b.build()

    # 1 cell.  Cells are x=[2,12,22,32,42], y=[3,23,43]
    g = evaluation.grid_risk_coverage_to_graph(prediction, graph, 13, 0.6)
    assert g.number_edges == 0
    
    g = evaluation.grid_risk_coverage_to_graph(prediction, graph, 13, 0.5)
    assert g.number_edges == 1
    assert g.edges[0] == (0, 1)

@pytest.fixture
def network_points():
    times = [np.datetime64("2017-01-01")] * 3
    locations = [ ((0,1), 0.5), ((2,1), 0.2), ((4,5), 0.1)]
    return open_cp.network.TimedNetworkPoints(times, locations)

def test_network_hit_rate(network_points):
    graph = open_cp.network.PlanarGraph([(0,0,0), (1,1,1), (2,2,2), (3,3,3), (4,4,4)], [(0,1), (1,2), (3,4)])
    assert evaluation.network_hit_rate(graph, network_points) == pytest.approx(2/3)
    
    evaluation.network_hit_rate(graph, network_points, graph)
    
    with pytest.raises(ValueError):
        graph1 = open_cp.network.PlanarGraph([(0,0,0), (1,1,1), (2,2,2), (3,3,3), (4,4,3), (5,5,5)], [])
        evaluation.network_hit_rate(graph, network_points, graph1)

@pytest.fixture
def graph():
    b = open_cp.network.PlanarGraphBuilder()
    b.add_vertex(2, 3)
    b.add_vertex(2.5, 3.8)
    b.add_vertex(13, 5)
    b.add_vertex(15, 7)
    b.add_edge(0, 1)
    b.add_edge(2, 3)
    return b.build()

def test_grid_risk_to_graph(prediction, graph):
    g,_,risks = evaluation.grid_risk_to_graph(prediction, graph)
    assert g is graph
    np.testing.assert_allclose(risks, [1, 2])

def test_grid_risk_to_graph_split(prediction, graph):
    g, lookup, risks = evaluation.grid_risk_to_graph(prediction, graph, "subdivide")
    assert open_cp.network.approximately_equal(g, graph)
    np.testing.assert_allclose(risks, [1, 2])
    assert lookup == {0:0, 1:1}

@pytest.fixture
def graph1():
    b = open_cp.network.PlanarGraphBuilder()
    b.add_vertex(10, 4)
    b.add_vertex(14, 6)
    b.add_edge(0, 1)
    return b.build()

def test_grid_risk_to_graph_split1(prediction, graph1):
    g, lookup, risks = evaluation.grid_risk_to_graph(prediction, graph1, "subdivide")
    np.testing.assert_allclose(risks, [1, 2])
    assert lookup == {0:0, 1:0}
    
    b = open_cp.network.PlanarGraphBuilder()
    b.add_vertex(10, 4)
    b.add_vertex(12, 5)
    b.add_vertex(14, 6)
    b.add_edge(0, 1)
    b.add_edge(1, 2)
    assert open_cp.network.approximately_equal(g, b.build())

def test_network_coverage(graph):
    out = evaluation.network_coverage(graph, [3,5], 0.5)
    np.testing.assert_allclose(out, [False, False])
    out = evaluation.network_coverage(graph, [3,5], 0.99)
    np.testing.assert_allclose(out, [False, True])

@pytest.fixture
def graph2():
    b = open_cp.network.GraphBuilder()
    b.add_edge(0, 1)
    b.add_edge(1, 2)
    b.add_edge(0, 2)
    b.add_edge(4, 5)
    b.lengths = [10, 10, 10, 10]
    return b.build()
    
def test_network_hit_rates_from_coverage(network_points, graph2):
    risks = [1, 2, 3, 4]
    out = evaluation.network_hit_rates_from_coverage(graph2, risks, network_points, [10,25,50,74,75])
    assert set(out) == {10,25,50,74,75}
    assert out[10] == pytest.approx(0)
    assert out[25] == pytest.approx(100/3)
    assert out[50] == pytest.approx(100/3)
    assert out[74] == pytest.approx(100/3)
    assert out[75] == pytest.approx(200/3)


#############################################################################
# Automate prediction making and evaluating testing
#############################################################################

@pytest.fixture
def timed_pts_10():
    ts = [datetime.datetime(2017,1,1) + datetime.timedelta(days=x) for x in range(10)]
    ts = [np.datetime64(dt) for dt in ts]
    x = np.random.random(10)
    y = np.random.random(10)
    return open_cp.data.TimedPoints.from_coords(ts, x, y)

def test_StandardPredictionProvider(timed_pts_10):
    mat = np.asarray([[False, True, True, False], [True]*4])
    grid = open_cp.data.MaskedGrid(10, 15, 5, 7, mat)
    spp = evaluation.StandardPredictionProvider(timed_pts_10, grid)
    spp.give_prediction = mock.Mock()

    assert spp.points is timed_pts_10
    assert spp.grid is grid

    pred = spp.predict(datetime.datetime(2017,2,1))
    args = spp.give_prediction.call_args[0]
    assert args[0] is grid
    assert args[1].number_data_points == 10
    np.testing.assert_allclose(args[1].coords, timed_pts_10.coords)
    assert args[2] == np.datetime64("2017-02-01")
    assert pred is spp.give_prediction.return_value.renormalise.return_value
    spp.give_prediction.return_value.mask_with.assert_called_with(grid)

    pred = spp.predict(datetime.datetime(2017,1,2))
    args = spp.give_prediction.call_args[0]
    assert args[0] is grid
    assert args[1].number_data_points == 1
    assert args[1].xcoords[0] == timed_pts_10.xcoords[0]
    assert args[2] == np.datetime64("2017-01-02")

@mock.patch("open_cp.naive.CountingGridKernel")
def test_NaiveProvider(mock_provider, timed_pts_10):
    mat = np.asarray([[False, True, True, False], [True]*4])
    grid = open_cp.data.MaskedGrid(10, 15, 5, 7, mat)
    nprov = evaluation.NaiveProvider(timed_pts_10, grid)
    pred = nprov.predict(datetime.datetime(2017,2,3))
    args = mock_provider.call_args[0]
    assert args[0] == 10
    assert args[1] == 15
    assert args[2].xmin == 5
    assert args[2].xmax == 5 + 10 * 4
    assert args[2].ymin == 7
    assert args[2].ymax == 7 + 15 * 2
    assert pred is mock_provider.return_value.predict.return_value.renormalise.return_value

@mock.patch("open_cp.predictors.GridPredictionArray")
@mock.patch("open_cp.naive.ScipyKDE")
def test_ScipyKDEProvider(mock_provider, mock_preds, timed_pts_10):
    mat = np.asarray([[False, True, True, False], [True]*4])
    grid = open_cp.data.MaskedGrid(10, 15, 5, 7, mat)
    prov = evaluation.ScipyKDEProvider(timed_pts_10, grid)
    pred = prov.predict(datetime.datetime(2017,2,3))
    mock_provider.assert_called_with()
    args = mock_preds.from_continuous_prediction_grid.call_args[0]
    assert args[0] == mock_provider.return_value.predict.return_value
    assert args[1] == grid
    assert pred is mock_preds.from_continuous_prediction_grid.return_value.renormalise.return_value

@mock.patch("open_cp.retrohotspot.RetroHotSpotGrid")
def test_RetroHotspotProvider(mock_provider, timed_pts_10):
    mat = np.asarray([[False, True, True, False], [True]*4])
    grid = open_cp.data.MaskedGrid(15, 15, 5, 7, mat)
    
    weight = open_cp.retrohotspot.TruncatedGaussian()
    provider = evaluation.RetroHotspotProvider(weight)
    prov = provider(timed_pts_10, grid)
    assert repr(prov).startswith("RetroHotspotProvider")
    pred = prov.predict(datetime.datetime(2017,2,3))
    mock_provider.assert_called_with(grid=grid)
    assert mock_provider.return_value.weight == weight
    mock_provider.return_value.predict.assert_called_with(end_time=datetime.datetime(2017,2,3))
    assert pred is mock_provider.return_value.predict.return_value.renormalise.return_value

@mock.patch("open_cp.predictors.GridPredictionArray")
@mock.patch("open_cp.retrohotspot.RetroHotSpot")
def test_RetroHotspotCtsProvider(mock_provider, mock_preds, timed_pts_10):
    mat = np.asarray([[False, True, True, False], [True]*4])
    grid = open_cp.data.MaskedGrid(15, 15, 5, 7, mat)
    
    weight = open_cp.retrohotspot.TruncatedGaussian()
    provider = evaluation.RetroHotspotCtsProvider(weight)
    prov = provider(timed_pts_10, grid)
    assert repr(prov).startswith("RetroHotspotCtsProvider")
    pred = prov.predict(datetime.datetime(2017,2,3))
    mock_provider.assert_called_with()
    assert mock_provider.return_value.weight == weight
    mock_provider.return_value.predict.assert_called_with(end_time=datetime.datetime(2017,2,3))
    args = mock_preds.from_continuous_prediction_grid.call_args[0]
    assert args[0] == mock_provider.return_value.predict.return_value
    assert args[1] == grid
    assert pred is mock_preds.from_continuous_prediction_grid.return_value.renormalise.return_value

@mock.patch("open_cp.predictors.GridPredictionArray")
@mock.patch("open_cp.prohotspot.ProspectiveHotSpotContinuous")
def test_ProHotspotCtsProvider(mock_provider, mock_preds, timed_pts_10):
    mat = np.asarray([[False, True, True, False], [True]*4])
    grid = open_cp.data.MaskedGrid(15, 15, 5, 7, mat)
    
    weight = open_cp.prohotspot.ClassicWeight(6,4)
    provider = evaluation.ProHotspotCtsProvider(weight, 123)
    prov = provider(timed_pts_10, grid)
    assert repr(prov).startswith("ProHotspotCtsProvider")
    pred = prov.predict(datetime.datetime(2017,2,3))
    mock_provider.assert_called_with(grid_size=123)
    assert mock_provider.return_value.weight == weight
    mock_provider.return_value.predict.assert_called_with(datetime.datetime(2017,2,3), datetime.datetime(2017,2,3))
    args = mock_preds.from_continuous_prediction_grid.call_args[0]
    assert args[0] == mock_provider.return_value.predict.return_value
    assert args[1] == grid
    assert pred is mock_preds.from_continuous_prediction_grid.return_value.renormalise.return_value

@mock.patch("open_cp.prohotspot.ProspectiveHotSpot")
def test_ProHotspotProvider(mock_provider, timed_pts_10):
    mat = np.asarray([[False, True, True, False], [True]*4])
    grid = open_cp.data.MaskedGrid(15, 15, 5, 7, mat)
    
    weight = open_cp.prohotspot.ClassicWeight(6,4)
    distance = open_cp.prohotspot.DistanceCircle()
    provider = evaluation.ProHotspotProvider(weight, distance)
    prov = provider(timed_pts_10, grid)
    assert repr(prov).startswith("ProHotspotProvider")
    pred = prov.predict(datetime.datetime(2017,2,3))
    mock_provider.assert_called_with(grid=grid, time_unit=np.timedelta64(1, "D"))
    assert mock_provider.return_value.weight == weight
    assert mock_provider.return_value.distance == distance
    mock_provider.return_value.predict.assert_called_with(datetime.datetime(2017,2,3), datetime.datetime(2017,2,3))
    assert pred is mock_provider.return_value.predict.return_value.renormalise.return_value

@mock.patch("open_cp.predictors.GridPredictionArray")
@mock.patch("open_cp.kde.KDE")
def test_KDEProvider(mock_provider, mock_preds, timed_pts_10):
    mat = np.asarray([[False, True, True, False], [True]*4])
    grid = open_cp.data.MaskedGrid(15, 15, 5, 7, mat)
    
    tk = open_cp.kde.QuadDecayTimeKernel(50)
    sk = open_cp.kde.GaussianBaseProvider()
    provider = evaluation.KDEProvider(tk, sk)
    prov = provider(timed_pts_10, grid)
    assert repr(prov).startswith("KDEProvider")
    pred = prov.predict(datetime.datetime(2017,2,3))
    mock_provider.assert_called_with(grid=grid)
    assert mock_provider.return_value.time_kernel == tk
    assert mock_provider.return_value.space_kernel == sk
    mock_provider.return_value.cts_predict.assert_called_with(end_time=datetime.datetime(2017,2,3))
    args = mock_preds.from_continuous_prediction_grid.call_args[0]
    assert args[0] == mock_provider.return_value.cts_predict.return_value
    assert args[1] == grid
    assert pred is mock_preds.from_continuous_prediction_grid.return_value.renormalise.return_value
