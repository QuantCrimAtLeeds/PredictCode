from .helper import *

import open_cp.gui.predictors.kde as kde
#import datetime

def test_Kde(model, project_task, analysis_model, grid_task):
    provider = kde.KDE(model)
    assert provider.name == "KDE predictor (scipy,training)"
    assert provider.settings_string == ""
    standard_calls(provider, project_task, analysis_model, grid_task)

def test_Kde_serialise(model, project_task, analysis_model, grid_task):
    serialise( kde.KDE(model) )

def assert_call_has_start_end_only(l):
    assert len(l) == 1
    assert len(l[0]) == 2
    assert l[0][0] == ()
    return l[0][1]["start_time"], l[0][1]["end_time"]

def assert_grid_correct(l):
    assert len(l) == 1
    grid = l[0][1]["grid"]
    assert grid.xsize == 50
    assert grid.ysize == 50
    assert grid.xoffset == 25
    assert grid.yoffset == 35
    assert grid.xextent == 15
    assert grid.yextent == 10

def assert_data_correct(pred):
    np.testing.assert_array_equal(pred.data.timestamps, [np.datetime64("2017-05-21T12:30"),
        np.datetime64("2017-05-21T13:00"), np.datetime64("2017-05-21T13:30")])
    np.testing.assert_array_equal(pred.data.xcoords, [0,10,20])
    np.testing.assert_array_equal(pred.data.ycoords, [10,20,0])

@mock.patch("open_cp.kde.KDE")
def test_KDE_subtask(model, project_task, analysis_model, grid_task):
    analysis_model.time_range = (datetime.datetime(2017,3,4,5,6), datetime.datetime(2017,3,4,5,9), None, None)
    provider = kde.KDE(model)
    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    prediction = subtask(datetime.datetime(2017,6,7))
    
    assert_grid_correct(open_cp.kde.KDE.call_args_list)
    
    pred = open_cp.kde.KDE.return_value
    assert pred.time_unit == np.timedelta64(1,"s")
    assert isinstance(pred.time_kernel, open_cp.kde.ConstantTimeKernel)
    assert isinstance(pred.space_kernel, open_cp.kde.GaussianBaseProvider)
    assert_data_correct(pred)
    
    st, en = assert_call_has_start_end_only(pred.predict.call_args_list)
    assert st == np.datetime64("2017-03-04T05:06")
    assert en == np.datetime64("2017-03-04T05:09")
    
    assert prediction is pred.predict.return_value

@mock.patch("open_cp.kde.KDE")
def test_KDE_nearest_window(model, project_task, analysis_model, grid_task):
    analysis_model.time_range = (datetime.datetime(2017,3,4,5,6), datetime.datetime(2017,3,4,5,9), None, None)
    provider = kde.KDE(model)
    provider.space_kernel = 1
    provider.space_kernel_model.k = 35
    provider.time_kernel = 1
    provider.time_kernel_model.days = 12
    assert provider.name == "KDE predictor (nearest,window)"
    assert provider.settings_string == "35, 12.0 days"
    serialise( kde.KDE(model) )
    
    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    prediction = subtask(datetime.datetime(2017,6,7))
    assert_grid_correct(open_cp.kde.KDE.call_args_list)

    pred = open_cp.kde.KDE.return_value
    assert pred.time_unit == np.timedelta64(1,"s")
    assert isinstance(pred.time_kernel, open_cp.kde.ConstantTimeKernel)
    assert isinstance(pred.space_kernel, open_cp.kde.GaussianNearestNeighbourProvider)
    assert pred.space_kernel.k == 35
    assert_data_correct(pred)

    st, en = assert_call_has_start_end_only(pred.predict.call_args_list)
    assert st == np.datetime64("2017-06-07T00:00") - np.timedelta64(12, "D")
    assert en == np.datetime64("2017-06-07T00:00")
    
    assert prediction is pred.predict.return_value

@mock.patch("open_cp.kde.KDE")
def test_KDE_exp_decay(model, project_task, analysis_model, grid_task):
    analysis_model.time_range = (datetime.datetime(2017,3,4,5,6), datetime.datetime(2017,3,4,5,9), None, None)
    provider = kde.KDE(model)
    provider.time_kernel = 2
    provider.time_kernel_model.scale = 30.5
    assert provider.name == "KDE predictor (scipy,exponential)"
    assert provider.settings_string == "30.5 days"
    serialise( kde.KDE(model) )
    
    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    prediction = subtask(datetime.datetime(2017,6,7))
    assert_grid_correct(open_cp.kde.KDE.call_args_list)

    pred = open_cp.kde.KDE.return_value
    assert pred.time_unit == np.timedelta64(1,"s")
    assert isinstance(pred.time_kernel, open_cp.kde.ExponentialTimeKernel)
    assert pred.time_kernel.scale == pytest.approx(30.5)
    assert isinstance(pred.space_kernel, open_cp.kde.GaussianBaseProvider)
    assert_data_correct(pred)

    st, en = assert_call_has_start_end_only(pred.predict.call_args_list)
    assert st == np.datetime64("2017-06-07T00:00") - np.timedelta64(24, "h") * 30.5 * 7
    assert en == np.datetime64("2017-06-07T00:00")
    
    assert prediction is pred.predict.return_value

@mock.patch("open_cp.kde.KDE")
def test_KDE_quad_decay(model, project_task, analysis_model, grid_task):
    analysis_model.time_range = (datetime.datetime(2017,3,4,5,6), datetime.datetime(2017,3,4,5,9), None, None)
    provider = kde.KDE(model)
    provider.time_kernel = 3
    provider.time_kernel_model.scale = 12.3
    assert provider.name == "KDE predictor (scipy,quadratic)"
    assert provider.settings_string == "12.3 days"
    serialise( kde.KDE(model) )
    
    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    prediction = subtask(datetime.datetime(2017,6,7))
    assert_grid_correct(open_cp.kde.KDE.call_args_list)

    pred = open_cp.kde.KDE.return_value
    assert pred.time_unit == np.timedelta64(1,"s")
    assert isinstance(pred.time_kernel, open_cp.kde.QuadDecayTimeKernel)
    assert pred.time_kernel.scale == pytest.approx(12.3)
    assert isinstance(pred.space_kernel, open_cp.kde.GaussianBaseProvider)
    assert_data_correct(pred)

    st, en = assert_call_has_start_end_only(pred.predict.call_args_list)
    assert st == np.datetime64("2017-06-07T00:00") - np.timedelta64(24*60*60, "s") * 12.3 * 32
    assert en == np.datetime64("2017-06-07T00:00")
    
    assert prediction is pred.predict.return_value
