import pytest

import open_cp.gui.analysis as analysis
import open_cp.gui.import_file_model as import_file_model
import open_cp.gui.predictors as predictors
import datetime

@pytest.fixture
def model():
    ts = [datetime.datetime.now() for _ in range(10)]
    xcs = [0 for _ in range(10)]
    ycs = [0 for _ in range(10)]
    ctypes = [["One"], ["Two"], ["Three"], ["Four"], ["One"], ["Two"], ["Three"], ["Four"], ["One"], ["Two"]]
    data = (ts, xcs, ycs, ctypes)
    return analysis.Model(None, data, None)

def test_crime_types(model):
    assert model.num_crime_type_levels == 1
    assert model.unique_crime_types == [ ("Four",), ("One",), ("Three",), ("Two",) ]

@pytest.fixture
def model2():
    ts = [datetime.datetime.now() for _ in range(10)]
    xcs = [0 for _ in range(10)]
    ycs = [0 for _ in range(10)]
    ctypes = [["One", "A"], ["Two", "B"], ["Three", "C"], ["Four", "D"],
        ["One", "B"], ["Two", "C"], ["Three", "D"], ["Four", "C"], ["One", "A"], ["Two", "B"]]
    data = (ts, xcs, ycs, ctypes)
    return analysis.Model(None, data, None)

def test_crime_types_2_levels(model2):
    assert model2.num_crime_type_levels == 2
    assert model2.unique_crime_types == [ ("Four", "C"), ("Four", "D"), ("One", "A"),
        ("One", "B"), ("Three", "C"), ("Three", "D"), ("Two", "B"), ("Two", "C") ]

def test_crime_type_selection(model):
    assert model.selected_crime_types == set()

    model.selected_crime_types = [1, 2, 3]

    with pytest.raises(ValueError):
        model.selected_crime_types = [-1]

    with pytest.raises(ValueError):
        model.selected_crime_types = [4]
        
def test_counts_by_crime_type(model):
    assert model.counts_by_crime_type() == 0

    model.selected_crime_types = [0]
    assert model.counts_by_crime_type() == 2

    model.selected_crime_types = [1]
    assert model.counts_by_crime_type() == 3

    model.selected_crime_types = [0, 1]
    assert model.counts_by_crime_type() == 5
    
@pytest.fixture
def parse_settings():
    ps = import_file_model.ParseSettings()
    ps.coord_type = import_file_model.CoordType.XY
    ps.meters_conversion = 0.23
    ps.timestamp_field = 1
    ps.xcoord_field = 2
    ps.ycoord_field = 5
    ps.crime_type_fields = [7]
    ps.timestamp_format = "%Y"
    return ps

@pytest.fixture
def saved_dict(model, parse_settings):
    model.filename = "test.json"
    model._parse_settings = parse_settings
    model.time_range = [ datetime.datetime(2017,9,6,12,30), datetime.datetime(2017,10,7,12,45),
        datetime.datetime(2017,11,5,0,0), datetime.datetime(2017,12,6,9,30) ]
    model.selected_crime_types = {1,3}
    return model.to_dict()

def test_serialisation_out(saved_dict):
    assert saved_dict["filename"] == "test.json"
    ps = import_file_model.ParseSettings.from_dict(saved_dict["parse_settings"])
    assert ps.coord_type == import_file_model.CoordType.XY
    assert ps.coordinate_scaling == pytest.approx(0.23)
    assert ps.timestamp_field == 1
    assert ps.xcoord_field == 2
    assert ps.ycoord_field == 5
    assert ps.crime_type_fields == [7]
    assert ps.timestamp_format == "%Y"

def test_serialisation(saved_dict):
    mdl = model()
    mdl._parse_settings = parse_settings()
    assert mdl.filename is None
    
    mdl.settings_from_dict(saved_dict)

    # Shouldn't be restored
    assert mdl.filename is None

    assert mdl.time_range[0] == datetime.datetime(2017,9,6,12,30)
    assert mdl.time_range[1] == datetime.datetime(2017,10,7,12,45)
    assert mdl.time_range[2] == datetime.datetime(2017,11,5,0,0)
    assert mdl.time_range[3] == datetime.datetime(2017,12,6,9,30)
    assert mdl.selected_crime_types == {1,3}

@pytest.fixture
def model_with_preds(saved_dict, model):
    model.settings_from_dict(saved_dict)
    model._parse_settings.coord_type = import_file_model.CoordType.LonLat
    model.analysis_tools_model.add(predictors.lonlat.LonLatConverter)
    return model.to_dict()

def test_pred_serialisation(model_with_preds, parse_settings):
    mdl = model()
    mdl._parse_settings = parse_settings
    mdl._parse_settings.coord_type = import_file_model.CoordType.LonLat
    mdl.settings_from_dict(model_with_preds)
    mdl.load_settings_slow(model_with_preds)

    assert len(mdl.analysis_tools_model.objects) == 1
    assert mdl.analysis_tools_model.objects[0].describe() == "Project LonLat coordinates to meters"

def test_pred_serialisation_error(model_with_preds):
    mdl = model()
    mdl._parse_settings = parse_settings()
    model_with_preds["analysis_tools"]["predictors"][0]["name"] = "bob"
    mdl.settings_from_dict(model_with_preds)
    mdl.load_settings_slow(model_with_preds)

    assert len(mdl.analysis_tools_model.objects) == 0
    assert mdl.consume_errors() == ["Cannot find a match for a predictor named bob"]
