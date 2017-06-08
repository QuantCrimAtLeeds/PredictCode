import pytest
import unittest.mock as mock

import open_cp.gui.tk.date_picker as date_picker

@pytest.fixture
def dp():
    with mock.patch("open_cp.gui.tk.date_picker._DatePickerView") as clazzmock:
        yield date_picker.DatePicker()

def test_days_to_text(dp):
    import locale
    if locale.getdefaultlocale()[0][:2] != "en":
        return
    assert dp.day_to_text(0) == "Mon"
    assert dp.day_to_text(1) == "Tue"
    assert dp.day_to_text(2) == "Wed"
    assert dp.day_to_text(3) == "Thu"
    assert dp.day_to_text(4) == "Fri"
    assert dp.day_to_text(5) == "Sat"
    assert dp.day_to_text(6) == "Sun"

    with pytest.raises(ValueError):
        dp.day_to_text(-1)
    with pytest.raises(ValueError):
        dp.day_to_text(7)

def test_first_day_of_week(dp):
    assert dp.first_day_of_week == "Mon"
    dp.first_day_of_week = "Sun"
    assert dp.first_day_of_week == "Sun"
    dp._view.make_day_labels.assert_called_once_with()
    dp._view.make_date_grid.assert_called_once_with()

    with pytest.raises(ValueError):
        dp.first_day_of_week = "Tue"

def test_month_year(dp):
    assert dp.month_year == (6, 2017)

    dp.month_year = (1, 1987)
    assert dp.month_year == (1, 1987)
    dp._view.refresh_month_year.assert_called_once_with()
    
    dp.month_year = ("12", "2023")
    assert dp.month_year == (12, 2023)
    
    with pytest.raises(ValueError):
        dp.month_year = 5
    with pytest.raises(ValueError):
        dp.month_year = (0, 1987)
    with pytest.raises(ValueError):
        dp.month_year = (13, 1982)

def test_set_selected(dp):
    import datetime
    d = datetime.date.today()
    assert d == dp.selected_date

    dp.selected_date = datetime.date(year=2011, month=5, day=23)
    assert dp.selected_date == datetime.date(year=2011, month=5, day=23)
    dp._view.refresh_month_year.assert_called_once_with()

    with pytest.raises(ValueError):
        dp.selected_date = 5        
        