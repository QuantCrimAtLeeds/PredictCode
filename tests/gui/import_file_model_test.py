import pytest

import open_cp.gui.import_file_model as import_file_model

def test_settings_json():
    ps = import_file_model.ParseSettings()
    ps.coord_type = import_file_model.CoordType.XY
    ps.meters_conversion = 1.23
    ps.timestamp_format = "abcbdags"
    ps.timestamp_field = 5
    ps.xcoord_field = 7
    ps.ycoord_field = 6
    ps.crime_type_fields = [1,2]
    s = ps.to_json()

    ps = import_file_model.ParseSettings.from_json(s)
    assert ps.coord_type == import_file_model.CoordType.XY
    assert ps.meters_conversion == 1.23
    assert ps.timestamp_format == "abcbdags"
    assert ps.timestamp_field == 5
    assert ps.xcoord_field == 7
    assert ps.ycoord_field == 6
    assert ps.crime_type_fields == [1,2]
    