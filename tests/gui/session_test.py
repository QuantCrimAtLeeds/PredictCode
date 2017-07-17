import pytest
import unittest.mock as mock

import open_cp.gui.session as session

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
    def items(self):
        yield from self.data.items()
    settings_mock.items = items
    def _iter(self):
        yield from self.data
    settings_mock.__iter__ = _iter
    def _del(self, index):
        del self.data[index]
    settings_mock.__delitem__ = _del
    return settings_mock

@pytest.fixture
def locator(settings):
    with mock.patch("open_cp.gui.session.locator") as locator_mock:
        def get(name):
            assert name == "settings"
            print("In locator mock, returning", settings)
            return settings
        locator_mock.get = get
        yield locator_mock

def test_session(locator):
    s = session.Session(None)

@pytest.fixture
def with_old_sessions(locator, settings):
    settings.data = {"bob" : "dave",
        "session5" : "filename5",
        "session10" : "filename10",
        "session2" : "filename2" }
    return settings

def test_reads_old_sessions(with_old_sessions):
    s = session.Session(None)
    assert s.model.recent_sessions == ["filename2", "filename5", "filename10"]
    
def test_replace_session(with_old_sessions, settings):
    s = session.Session(None)
    
    s.new_session("filename10")
    
    assert settings.save.called
    assert settings["session0"] == "filename10"
    assert settings["session1"] == "filename2"
    assert settings["session2"] == "filename5"
    assert {int(key[7:]) for key in settings if key[:7] == "session"} == {0,1,2}

def test_new_session(with_old_sessions, settings):
    s = session.Session(None)
    
    s.new_session("filename20")
    
    assert settings.save.called
    assert settings["session0"] == "filename20"
    assert settings["session1"] == "filename2"
    assert settings["session2"] == "filename5"
    assert settings["session3"] == "filename10"
    assert {int(key[7:]) for key in settings if key[:7] == "session"} == {0,1,2,3}
    
def test_max_10_sessions(with_old_sessions, settings):
    s = session.Session(None)
    for i in range(20, 40):
        s.new_session("filename{}".format(i))
    assert {int(key[7:]) for key in settings if key[:7] == "session"} == {0,1,2,3,4,5,6,7,8,9}
    for i in range(10):
        assert settings["session{}".format(i)] == "filename{}".format(39-i)
    
@pytest.fixture
def view():
    with mock.patch("open_cp.gui.session.session_view") as mock_view:
        yield mock_view
        
def test_run(view, locator, settings):
    s = session.Session(None)
    s.run()
    assert view.SessionView.return_value.wait_window.called

def test_selected(view, with_old_sessions):
    s = session.Session(None)
    s.run()
    s.selected(1)
    
    assert s.filename == "filename5"
    
