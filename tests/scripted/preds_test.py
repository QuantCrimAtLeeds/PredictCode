import pytest

import open_cp.scripted.preds as preds

import datetime

def test_TimeRange():
    tr = preds.TimeRange(datetime.datetime(2016,1,1),
        datetime.datetime(2016,1,5), datetime.timedelta(days=1))
    assert list(tr) == [
        (datetime.datetime(2016,1,1), datetime.datetime(2016,1,2)),
        (datetime.datetime(2016,1,2), datetime.datetime(2016,1,3)),
        (datetime.datetime(2016,1,3), datetime.datetime(2016,1,4)),
        (datetime.datetime(2016,1,4), datetime.datetime(2016,1,5))
        ]
    
    # Slightly paranoid test we can repeatedly iterate...
    for st, en in tr:
        assert st == datetime.datetime(2016,1,1)
        assert en == datetime.datetime(2016,1,2)
        break

    for st, en in tr:
        assert st == datetime.datetime(2016,1,1)
        assert en == datetime.datetime(2016,1,2)
        break
