import pytest
import unittest.mock as mock

import os
import geopandas as gpd
import numpy as np

import open_cp.gui.predictors.geo_clip as geo_clip

@pytest.fixture
def geojson_filename():
    return os.path.join("tests", "test_geometry", "test.geojson")

@pytest.fixture
def shp_filename():
    return os.path.join("tests", "test_geometry", "test.shp")

def test_can_find_file(geojson_filename):
    frame = gpd.read_file(geojson_filename)
    geo = frame.unary_union
    assert geo.area == pytest.approx(0.5)
    
    frame = gpd.read_file(os.path.join("tests", "test_geometry", "test.shp"))
    geo = frame.unary_union
    assert geo.area == pytest.approx(0.5)

@pytest.fixture
def model():
    return None

def test_setup_gdal(model):
    geo_clip.CropToGeometry(model)
    import os
    assert "GDAL_DATA" in os.environ

def test_build_CropToGeometry(model):
    comp = geo_clip.CropToGeometry(model)
    assert comp.name == "Crop to geometry"
    assert comp.settings_string == "No geometry"

    old = comp.settings_string
    data = comp.to_dict()
    comp.from_dict(data)
    assert comp.settings_string == old

def test_run_no_settings(model):
    comp = geo_clip.CropToGeometry(model)
    assert comp.run(None) is None

def test_load(model, geojson_filename):
    comp = geo_clip.CropToGeometry(model)
    comp.load(geojson_filename)
    geo = comp.geometry()
    assert geo.area == 0.5
    coords = np.asarray(geo.exterior.coords)
    np.testing.assert_allclose(coords, [[0,0], [1,0], [1,1], [0,0]])

    assert comp.settings_string.startswith("Geometry file: ")
    assert comp.epsg is None
    assert comp.settings_string.endswith(geojson_filename)

def test_load_shapefile(model, shp_filename):
    comp = geo_clip.CropToGeometry(model)
    comp.load(shp_filename)
    geo = comp.geometry()
    assert geo.area == 0.5
    coords = np.asarray(geo.exterior.coords)
    np.testing.assert_allclose(coords, [[0,0], [1,1], [1,0], [0,0]])

    assert comp.settings_string.startswith("Geometry file: ")
    assert comp.epsg is None
    assert comp.settings_string.endswith(shp_filename)
    