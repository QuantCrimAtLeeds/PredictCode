import pytest
import unittest.mock as mock

import open_cp.gui.config as config

@pytest.fixture
def root():   
    root = mock.MagicMock()
    root.style.theme_names.return_value = ["default", "clam", "matt"]
    root.style.theme_use.return_value = "default"
    return root

@pytest.fixture
def settings():
    settings_mock = mock.MagicMock()
    settings_mock.data = {"theme":"default"}
    def getitem(self, key):
        return self.data[key]
    settings_mock.__getitem__ = getitem
    def setitem(self, key, value):
        self.data[key] = value
        print("setting {} to {}".format(key, value))
    settings_mock.__setitem__ = setitem
    return settings_mock

@pytest.fixture
def locator(settings):
    with mock.patch("open_cp.gui.config.locator") as locator_mock:
        def get(name):
            assert name == "settings"
            print("In locator mock, returning", settings)
            return settings
        locator_mock.get = get
        yield locator_mock

@pytest.fixture
def config_obj(root, locator):
    return config.Config(root)
    
def test_construct(root, config_obj):
    assert config_obj.model.theme == "default"
    assert config_obj.model.themes == ["default", "clam", "matt"]
    
def test_theme_is_set(root, config_obj):
    assert root.style.theme_use.called_with("default")

def test_settings_overwrites(root, settings, locator):
    settings.data["theme"] = "clam"
    c = config.Config(root)
    assert c.model.theme == "clam"

def test_settings_nonsense_doesnt_change(root, settings, locator):
    settings.data["theme"] = "donteverhave"
    c = config.Config(root)
    assert c.model.theme == "default"

def test_no_settings(root, settings, locator):
    settings.data = {}
    c = config.Config(root)
    assert c.model.theme == "default"
    
@pytest.fixture
def view():
    with mock.patch("open_cp.gui.config.config_view") as view_mock:
        yield view_mock

def test_changing_theme(root, view, config_obj):
    c = config_obj
    c.run()
    c.selected_theme(2)
    assert c.model.theme == "matt"
    assert root.style.theme_use.called_with("matt")
    
    view.ConfigView.return_value.wait_window.assert_called()
    
def test_cancel(root, view, settings, config_obj):
    c = config_obj
    c.run()
    c.selected_theme(2)
    assert c.model.theme == "matt"
    assert root.style.theme_use.called_with("matt")

    c.cancel()
    assert root.style.theme_use.called_with("default")
    
    assert not settings.save.called

def test_okay(root, view, config_obj, settings):
    c = config_obj
    c.run()
    c.selected_theme(2)
    assert c.model.theme == "matt"
    assert root.style.theme_use.called_with("matt")

    c.okay()
    
    assert settings.save.called
    assert settings["theme"] == "matt"
    