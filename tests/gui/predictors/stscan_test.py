from .helper import *

import open_cp.gui.predictors.stscan as stscan
import datetime

def test_STScan(model, project_task, analysis_model, grid_task):
    provider = stscan.STScan(model)
    assert provider.settings_string == "geo(50%/3000m) time(50%/60days)"
    standard_calls(provider, project_task, analysis_model, grid_task)

def test_ProHotspot_serialise(model, project_task, analysis_model, grid_task):
    serialise( stscan.STScan(model) )

@mock.patch("open_cp.stscan.STSTrainer")
def test_STScan_subtask(model, project_task, analysis_model, grid_task):
    provider = stscan.STScan(model)
    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    prediction = subtask(datetime.datetime(2017,6,7))
    
    open_cp.stscan.STSTrainer.assert_called_with()
    pred_mock = open_cp.stscan.STSTrainer.return_value
    assert pred_mock.geographic_population_limit == 0.5
    assert pred_mock.geographic_radius_limit == 3000
    assert pred_mock.time_population_limit == 0.5
    assert pred_mock.time_max_interval == np.timedelta64(60, "D")
    np.testing.assert_array_equal(pred_mock.data.timestamps, [np.datetime64("2017-05-21T12:30"),
        np.datetime64("2017-05-21T13:00"), np.datetime64("2017-05-21T13:30")])
    np.testing.assert_array_equal(pred_mock.data.xcoords, [0,10,20])
    np.testing.assert_array_equal(pred_mock.data.ycoords, [10,20,0])
    assert pred_mock.region.xmin == 25
    assert pred_mock.region.ymin == 35
    assert pred_mock.region.xmax == 25 + 50 * 15
    assert pred_mock.region.ymax == 35 + 50 * 10
    
    predict_date = np.datetime64(datetime.datetime(2017,6,7))
    pred_mock.predict.assert_called_with(time=predict_date)
    
    result = pred_mock.predict.return_value
    result.grid_prediction.assert_called_with(50)
    assert prediction is result.grid_prediction.return_value

@mock.patch("open_cp.stscan.STSTrainer")
def test_STScan_cluster_options(model, project_task, analysis_model, grid_task):
    provider = stscan.STScan(model)
    provider.geographic_population_limit = 45
    provider.time_population_limit = 55
    provider.geographic_radius_limit = 1234
    provider.time_max_interval = datetime.timedelta(days=23)

    assert provider.settings_string == "geo(45%/1234m) time(55%/23days)"
    data = provider.to_dict()
    json_str = json.dumps(data)
    provider.from_dict(json.loads(json_str))
    assert provider.settings_string == "geo(45%/1234m) time(55%/23days)"
    
    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    prediction = subtask(datetime.datetime(2017,6,7))

    pred_mock = open_cp.stscan.STSTrainer.return_value
    assert pred_mock.geographic_population_limit == 0.45
    assert pred_mock.geographic_radius_limit == 1234
    assert pred_mock.time_population_limit == 0.55
    assert pred_mock.time_max_interval == np.timedelta64(23, "D")

@mock.patch("open_cp.stscan.STSTrainer")
def test_STScan_from_training_start(model, project_task, analysis_model, grid_task):
    analysis_model.time_range = (datetime.datetime(2017,5,21,12,59),None,None,None)
    provider = stscan.STScan(model)
    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    prediction = subtask(datetime.datetime(2017,6,7))
    
    pred_mock = open_cp.stscan.STSTrainer.return_value
    np.testing.assert_array_equal(pred_mock.data.timestamps, [
        np.datetime64("2017-05-21T13:00"), np.datetime64("2017-05-21T13:30")])
    np.testing.assert_array_equal(pred_mock.data.xcoords, [10,20])
    np.testing.assert_array_equal(pred_mock.data.ycoords, [20,0])

@mock.patch("open_cp.stscan.STSTrainer")
def test_STScan_time_window(model, project_task, analysis_model, grid_task):
    provider = stscan.STScan(model)
    provider.time_window_choice = 2
    provider.time_window_length = datetime.timedelta(days=13)

    assert provider.settings_string == "<=13days geo(50%/3000m) time(50%/60days)"
    data = provider.to_dict()
    json_str = json.dumps(data)
    provider.from_dict(json.loads(json_str))
    assert provider.settings_string == "<=13days geo(50%/3000m) time(50%/60days)"
    
    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    prediction = subtask(datetime.datetime(2017,5,21,13,29) + datetime.timedelta(days=13))

    pred_mock = open_cp.stscan.STSTrainer.return_value
    np.testing.assert_array_equal(pred_mock.data.timestamps, [
        np.datetime64("2017-05-21T13:30")])
    np.testing.assert_array_equal(pred_mock.data.xcoords, [20])
    np.testing.assert_array_equal(pred_mock.data.ycoords, [0])

@mock.patch("open_cp.stscan.STSTrainer")
def test_STScan_quant_grid(model, project_task, analysis_model, grid_task):
    provider = stscan.STScan(model)
    provider.quantisation_choice = 2

    assert provider.settings_string == "grid geo(50%/3000m) time(50%/60days)"
    data = provider.to_dict()
    json_str = json.dumps(data)
    provider.from_dict(json.loads(json_str))
    assert provider.settings_string == "grid geo(50%/3000m) time(50%/60days)"

    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    prediction = subtask(datetime.datetime(2017,5,21,13,31))
    pred_mock = open_cp.stscan.STSTrainer.return_value
    np.testing.assert_array_equal(pred_mock.data.timestamps, [np.datetime64("2017-05-21T12:30"),
        np.datetime64("2017-05-21T13:00"), np.datetime64("2017-05-21T13:30")])
    # Grid is (25,35) size 50
    np.testing.assert_array_equal(pred_mock.data.xcoords, [0,0,0])
    np.testing.assert_array_equal(pred_mock.data.ycoords, [10,10,10])

@mock.patch("open_cp.stscan.STSTrainer")
def test_STScan_quant_time(model, project_task, analysis_model, grid_task):
    provider = stscan.STScan(model)
    provider.quantisation_choice = 3
    provider.time_bin_length = datetime.timedelta(days=2)

    assert provider.settings_string == "bins(48hours) geo(50%/3000m) time(50%/60days)"
    data = provider.to_dict()
    json_str = json.dumps(data)
    provider.from_dict(json.loads(json_str))
    assert provider.settings_string == "bins(48hours) geo(50%/3000m) time(50%/60days)"

    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    prediction = subtask(datetime.datetime(2017,5,21,13,0))
    pred_mock = open_cp.stscan.STSTrainer.return_value
    np.testing.assert_array_equal(pred_mock.data.timestamps, [np.datetime64("2017-05-19T13:00"),
        np.datetime64("2017-05-21T13:00"), np.datetime64("2017-05-21T13:00")])
    np.testing.assert_array_equal(pred_mock.data.xcoords, [0,10,20])
    np.testing.assert_array_equal(pred_mock.data.ycoords, [10,20,0])

@mock.patch("open_cp.stscan.STSTrainer")
def test_STScan_quant_both(model, project_task, analysis_model, grid_task):
    provider = stscan.STScan(model)
    provider.quantisation_choice = 4
    provider.time_bin_length = datetime.timedelta(days=1)

    assert provider.settings_string == "grid bins(24hours) geo(50%/3000m) time(50%/60days)"
    data = provider.to_dict()
    json_str = json.dumps(data)
    provider.from_dict(json.loads(json_str))
    assert provider.settings_string == "grid bins(24hours) geo(50%/3000m) time(50%/60days)"

    subtask = standard_calls(provider, project_task, analysis_model, grid_task)
    prediction = subtask(datetime.datetime(2017,5,21,13,0))
    pred_mock = open_cp.stscan.STSTrainer.return_value
    np.testing.assert_array_equal(pred_mock.data.timestamps, [np.datetime64("2017-05-20T13:00"),
        np.datetime64("2017-05-21T13:00"), np.datetime64("2017-05-21T13:00")])
    np.testing.assert_array_equal(pred_mock.data.xcoords, [0,0,0])
    np.testing.assert_array_equal(pred_mock.data.ycoords, [10,10,10])
