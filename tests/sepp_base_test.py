import pytest
import unittest.mock as mock

import open_cp.sepp_base as sepp_base
import open_cp.data
import open_cp.predictors
import numpy as np
import datetime

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
            if dp[0] <= 0:
                p[j,i] = 0
            else:
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


class OurModel1(OurModel):
    def trigger(self, pt, dpts):
        super().trigger(pt, dpts)
        w = np.sum(np.abs(pt))
        return (1 + dpts[0]) * np.exp(-(dpts[1]**2 + dpts[2]**2) / w)


def test_p_matrix_with_time_repeats():
    model = OurModel1()
    for _ in range(10):
        points = np.random.random((3,20))
        points[0].sort()
        points[0][1] = points[0][0]
        points[0][15] = points[0][14]
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
    
def test_Trainer_constructs():
    tr = sepp_base.Trainer()
    assert tr.time_unit / np.timedelta64(1, "h") == pytest.approx(24)
    
    tr.time_unit = datetime.timedelta(seconds = 6 * 60)
    assert tr.time_unit / np.timedelta64(1, "m") == pytest.approx(6)


class OurTrainer(sepp_base.Trainer):
    def __init__(self):
        super().__init__()
        self._testing_fixed = mock.Mock()
        self._testing_im = mock.Mock()
        self._opt_class_mock = mock.Mock()
    
    def make_fixed(self, times):
        self._make_fixed_times = times
        return self._testing_fixed
    
    def initial_model(self, fixed, data):
        self._initial_model_params = (fixed, data)
        return self._testing_im

    @property
    def _optimiser(self):
        return self._opt_class_mock
    

@pytest.fixture
def trainer():
    sepp = OurTrainer()
    t = [np.datetime64("2017-05-01T00:00"), np.datetime64("2017-05-02T00:00"),
        np.datetime64("2017-05-04T00:00"), np.datetime64("2017-05-10T23:45")]
    x = [1,2,3,4]
    y = [5,6,7,8]
    sepp.data = open_cp.data.TimedPoints.from_coords(t, x, y)
    return sepp

def test_Trainer_make_data(trainer):
    sepp = trainer
    fixed, data = sepp.make_data()
    
    assert fixed is sepp._testing_fixed
    offset = 15 / 60 / 24
    np.testing.assert_allclose(sepp._make_fixed_times, [10, 9, 7, offset])
    np.testing.assert_allclose(data[0], [0, 1, 3, 10 - offset])
    np.testing.assert_allclose(data[1], [1,2,3,4])
    np.testing.assert_allclose(data[2], [5,6,7,8])
    
    fixed, data = sepp.make_data(predict_time = datetime.datetime(2017,5,10,23,45))
    
    assert fixed is sepp._testing_fixed
    np.testing.assert_allclose(sepp._make_fixed_times, [10 - offset, 9 - offset,
        7 - offset, 0])
    np.testing.assert_allclose(data[0], [0, 1, 3, 10 - offset])
    np.testing.assert_allclose(data[1], [1,2,3,4])
    np.testing.assert_allclose(data[2], [5,6,7,8])
    
def test_Trainer_make_initial_model(trainer):
    fixed, data = trainer.make_data()
    im = trainer.initial_model(fixed, data)
    
    assert im is trainer._testing_im
    assert trainer._initial_model_params[0] is fixed
    assert trainer._initial_model_params[1] is data

    offset = 15 / 60 / 24
    np.testing.assert_allclose(trainer._make_fixed_times, [10, 9, 7, offset])
    
def test_Trainer_optimise(trainer):
    model = trainer.train()
    
    opt = trainer._opt_class_mock.return_value
    assert model == opt.iterate.return_value
    
    assert len(trainer._opt_class_mock.call_args_list) == 1
    call = trainer._opt_class_mock.call_args_list[0]
    assert call[0][0] is trainer._testing_im
    
def test_Trainer_optimise_predict_time(trainer):
    model = trainer.train(predict_time = datetime.datetime(2017,5,10,23,45))
    
    opt = trainer._opt_class_mock.return_value
    assert model == opt.iterate.return_value

    offset = 15 / 60 / 24
    np.testing.assert_allclose(trainer._make_fixed_times, [10 - offset, 9 - offset,
        7 - offset, 0])
    
    assert len(trainer._opt_class_mock.call_args_list) == 1
    call = trainer._opt_class_mock.call_args_list[0]
    assert call[0][0] is trainer._testing_im

def test_Trainer_optimise2(trainer):
    model = trainer.train(iterations=2)
    
    opt = trainer._opt_class_mock.return_value
    assert model == opt.iterate.return_value
    
    assert len(trainer._opt_class_mock.call_args_list) == 2
    call = trainer._opt_class_mock.call_args_list[0]
    assert call[0][0] is trainer._testing_im
    call = trainer._opt_class_mock.call_args_list[1]
    assert call[0][0] is model


def test_PredictorBase():
    class Model():
        def background(self, points):
            return points[0]
        
        def trigger(self, pt, dp):
            return pt[0] * (dp[1] + dp[2])**2
        
    pts = [[0,1,2,3], [4,7,2,3], [4,5,6,1]]
    model = Model()
    pred = sepp_base.PredictorBase(model, pts)
    
    assert pred.model is model
    np.testing.assert_allclose(pred.points, pts)
    
    assert pred.point_predict(1, [2,3]) == pytest.approx(10)
    assert pred.point_predict(0.5, [2,3]) == pytest.approx(5)
    assert pred.point_predict(1, [4,4]) == pytest.approx(1)
    assert pred.point_predict(1, [4,5]) == pytest.approx(2)
    np.testing.assert_allclose(pred.point_predict(1, [[2,4,4], [3,4,5]]), [10,1,2])
    
    assert pred.point_predict(2, [2,3]) == pytest.approx(2 + 18 + 2*49)
    
    expected = sum(t*10 for t in np.linspace(0,1,20))
    assert pred.range_predict(0, 1, [2,3]) == pytest.approx(expected / 20)
    
    assert pred.background_predict(1, [2,3]) == pytest.approx(1)
    assert pred.background_predict(2, [4,7]) == pytest.approx(2)

def test_Predictor(trainer):
    mask = np.asarray([[False, True, False], [False]*3])
    grid = open_cp.data.MaskedGrid(xsize=10, ysize=15, xoffset=2, yoffset=3, mask=mask)
    model = OurModel()
    pred = sepp_base.Predictor(grid, model)
    pred.data = trainer.data    
    
    gp = pred.predict(np.datetime64("2017-05-04T00:00"))
    
    np.testing.assert_allclose(gp.intensity_matrix.mask, mask)
    assert (gp.xsize, gp.ysize) == (grid.xsize, grid.ysize)
    
    gp = pred.predict(np.datetime64("2017-05-04T00:00"), end_time=np.datetime64("2017-05-05T00:00"))
    np.testing.assert_allclose(gp.intensity_matrix.mask, mask)
    assert (gp.xsize, gp.ysize) == (grid.xsize, grid.ysize)

    gp = pred.background_predict(np.datetime64("2017-05-04T00:00"))
    np.testing.assert_allclose(gp.intensity_matrix.mask, mask)
    assert (gp.xsize, gp.ysize) == (grid.xsize, grid.ysize)

def test_clamp_p():
    p = [[1, 0, 0, 0], [0.6, 0.4, 0, 0], [0.99, 0.01, 0, 0], [0.2, 0.05, 0.7, 0.05]]
    p = np.asarray(p).T
    pc = np.asarray(p)
    pp = sepp_base.clamp_p(p, 99)

    np.testing.assert_allclose(p, pc) # Don't mutate p
    np.testing.assert_allclose(pp[:,0], [1,0,0,0])
    np.testing.assert_allclose(pp[:,1], [0.6,0.4,0,0])
    np.testing.assert_allclose(pp[:,2], [0.99,0,0,0])
    np.testing.assert_allclose(pp[:,3], [0.2,0.05,0.7,0.05])

    pp = sepp_base.clamp_p(p, 49)
    np.testing.assert_allclose(pp[:,0], [1,0,0,0])
    np.testing.assert_allclose(pp[:,1], [0.6,0,0,0])
    np.testing.assert_allclose(pp[:,2], [0.99,0,0,0])
    np.testing.assert_allclose(pp[:,3], [0,0,0.7,0])

def test_Optimiser_sample():
    model = sepp_base.ModelBase()
    model.background = lambda pts : [1]*pts.shape[-1]
    model.trigger = lambda tp, pts : [1]*pts.shape[-1]

    opt = sepp_base.Optimiser(model, np.random.random((3,4)))
    p = [[1,0,0,0], [0,1,0,0], [1,0,0,0], [0,0,1,0]]
    opt._p = np.asarray(p).T

    bk, tr = opt.sample()

    assert bk == [0,1]
    assert tr == [(0,2), (2,3)]

def test_Optimiser_sample_to_points():
    model = sepp_base.ModelBase()
    model.background = lambda pts : [1]*pts.shape[-1]
    model.trigger = lambda tp, pts : [1]*pts.shape[-1]

    pts = np.random.random((3,4))
    opt = sepp_base.Optimiser(model, pts)
    p = [[1,0,0,0], [0,1,0,0], [1,0,0,0], [0,0,1,0]]
    opt._p = np.asarray(p).T

    bk, tr = opt.sample_to_points()

    np.testing.assert_allclose(bk, pts[:,:2])
    expected_trigger = [pts[:,2] - pts[:,0], pts[:,3] - pts[:,2]]
    expected_trigger = np.asarray(expected_trigger).T
    np.testing.assert_allclose(tr, expected_trigger)

def slow_predict_at_time(fp, t, space_points):
    out = np.zeros(space_points.shape[-1])
    for i, pt in enumerate(fp.points.T):
        tk = fp.time_kernel(t - pt[0])
        sk = fp.space_kernel(space_points - pt[1:,None])
        out += tk * sk
    return out

def slow_range_predict(fp, time_start, time_end, space_points, time_samples=5):
    space_points = np.asarray(space_points)
    if len(space_points.shape) == 1:
        space_points = space_points[:,None]
    out = fp.background_kernel(space_points)
    for i in range(time_samples):
        t = time_start + (time_end - time_start) * i / (time_samples-1)
        out = out + slow_predict_at_time(fp, t, space_points) / time_samples
    return out

@pytest.fixture
def fast_model_1():
    class Model(sepp_base.FastModel):
        def background(self, points):
            raise Exception("Shouldn't be called.")
        
        def background_in_space(self, space_points):
            return space_points[0]
    
        def time_trigger(self, times):
            times = np.asarray(times)
            assert len(times.shape) <= 1
            return np.exp(-times) * 0.2
    
        def space_trigger(self, pts):
            dd = pts[0]**2 + pts[1]**2
            return np.exp(-dd/50)

    return Model()

def test_FastPredictBase(fast_model_1):
    fp = sepp_base.FastPredictorBase(fast_model_1)
    fp.points = np.random.random((3,20))

    assert fp.background_kernel([1,2]) == 1
    assert fp.background_kernel([5,7]) == 5
    np.testing.assert_allclose(fp.background_kernel([[1,5], [2,7]]), [1,5])
    
    assert fp.time_kernel(5) == pytest.approx(np.exp(-5)/5)
    assert fp.time_kernel(1) == pytest.approx(np.exp(-1)/5)
    np.testing.assert_allclose(fp.time_kernel(np.asarray([1,5])), [np.exp(-1)/5,np.exp(-5)/5])

    assert fp.space_kernel([1,2]) == pytest.approx(np.exp(-5/50))
    assert fp.space_kernel([3,5]) == pytest.approx(np.exp(-34/50))
    np.testing.assert_allclose(fp.space_kernel(np.asarray([[1,3],[2,5]])), [np.exp(-1/10), np.exp(-34/50)])
    
    assert fp.range_predict(100,101,[2,4],5) == pytest.approx(slow_range_predict(fp, 100, 101, [2,4], 5))
    assert fp.range_predict(100,102,[2,14],25) == pytest.approx(slow_range_predict(fp, 100, 102, [2,14], 25))

def test_FastPredictorHist():
    model = mock.Mock()
    points = np.random.random((3,100)) * np.asarray([100, 250, 250])[:,None]
    time_hist = np.random.random(500)
    smat = np.random.random((10, 10))
    sg = open_cp.predictors.GridPredictionArray(10, 10, smat, -50, -50)
    bmat = np.random.random((10, 10))
    bg = open_cp.predictors.GridPredictionArray(25, 25, bmat, 0, 0)
    fp = sepp_base.FastPredictorHist(model, time_hist, 0.3, sg, bg)
    fp.points = points
    
    assert fp.model is model
    np.testing.assert_allclose(fp.points, points)

    assert fp.space_kernel([0,0]) == smat[5, 5]
    assert fp.space_kernel([0,7]) == smat[5, 5]
    assert fp.space_kernel([0,10]) == smat[6, 5]
    np.testing.assert_allclose(fp.space_kernel([[0,0,0], [0,7,10]]),
            [smat[5,5], smat[5,5], smat[6,5]])

    assert fp.time_kernel(5) == time_hist[int(5/0.3)]
    assert fp.time_kernel(15) == time_hist[int(15/0.3)]
    assert fp.time_kernel(300) == 0
    np.testing.assert_allclose(fp.time_kernel([5,15,300]), 
            [time_hist[int(5/0.3)], time_hist[int(15/0.3)], 0])

    assert fp.range_predict(100,101,[2,4],5) == pytest.approx(slow_range_predict(fp, 100, 101, [2,4], 5))
    assert fp.range_predict(100,102,[2,14],25) == pytest.approx(slow_range_predict(fp, 100, 102, [2,14], 25))

    got = fp.range_predict(101, 103, [[1,2,3], [6,7,3]], 50)
    exp = slow_range_predict(fp, 101, 103, [[1,2,3], [6,7,3]], 50)
    np.testing.assert_allclose(got, exp)

def test_PredictorBase_to_fast_split_predictor(fast_model_1):
    pts = [[0,1,2,3], [4,7,2,3], [4,5,6,1]]
    model = fast_model_1
    pred = sepp_base.PredictorBase(model, pts)

    mmat = np.asarray([[False]*8]*12)
    grid = open_cp.data.MaskedGrid(50, 50, 0, 0, mmat)

    fp = pred.to_fast_split_predictor_histogram(grid)
    assert fp._time[1] == 1
    assert len(fp._time[0]) == 7
    np.testing.assert_allclose(fp._time[0], model.time_trigger(np.arange(7)))
    assert repr(fp._space_grid) == "GridPredictionArray(offset=(-125,-125), size=25x25, risk intensity size=10x10)"
    assert repr(fp._background_grid) == "GridPredictionArray(offset=(0,0), size=50x50, risk intensity size=8x12)"

    fp = pred.to_fast_split_predictor_histogram(grid, time_bin_size=0.2)
    assert fp._time[1] == 0.2
    assert len(fp._time[0]) == 35
    np.testing.assert_allclose(fp._time[0], model.time_trigger(0.2*np.arange(35)))
    assert repr(fp._space_grid) == "GridPredictionArray(offset=(-125,-125), size=25x25, risk intensity size=10x10)"

    fp = pred.to_fast_split_predictor_histogram(grid, space_bin_size=5)
    assert fp._time[1] == 1
    assert len(fp._time[0]) == 7
    np.testing.assert_allclose(fp._time[0], model.time_trigger(np.arange(7)))
    assert repr(fp._space_grid) == "GridPredictionArray(offset=(-50,-50), size=5x5, risk intensity size=20x20)"

def test_FastPredictor():
    mat = np.asarray([[False]*3, [True, False, True]])
    grid = open_cp.data.MaskedGrid(15, 20, 5, 10, mat)
    fpb_mock = mock.Mock()
    fp = sepp_base.FastPredictor(grid, fpb_mock)

    t = [np.datetime64("2017-05-01T00:00"), np.datetime64("2017-05-02T00:00"),
        np.datetime64("2017-05-04T00:00"), np.datetime64("2017-05-10T23:45")]
    x = [1,2,3,4]
    y = [5,6,7,8]
    fp.data = open_cp.data.TimedPoints.from_coords(t, x, y)

    pred = fp.continuous_predict(datetime.datetime(2017,5,10), datetime.datetime(2017,5,10, 12,30))
    pred.risk(10, 12)
    assert fpb_mock.range_predict.call_args[0][0] == pytest.approx(9)
    assert fpb_mock.range_predict.call_args[0][1] == pytest.approx(9+12.5/24)
    np.testing.assert_allclose(fpb_mock.range_predict.call_args[0][2], [[10], [12]])
    assert fpb_mock.range_predict.call_args[1] == {"time_samples":5}
