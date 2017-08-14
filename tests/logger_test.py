import pytest
import unittest.mock as mock

import open_cp.logger as logger
import logging
import datetime
import io

def test_standard_formatter():
    fmt = logger.standard_formatter()
    record = logging.LogRecord(name="spam", level=logging.DEBUG, pathname=None,
        lineno=None, msg="eggs", args=None, exc_info=None)
    out = fmt.format(record)
    assert out.endswith(" DEBUG spam - eggs")

@pytest.fixture
def lgr():
    logger = logging.getLogger("open_cp")
    for h in logger.handlers:
        logger.removeHandler(h)
    return logger

def test_log_to_stdout(lgr):
    with mock.patch("open_cp.logger._sys.stdout") as sys_stdout:
        assert len(list(lgr.handlers)) == 0
        logger.log_to_stdout()
        assert len(list(lgr.handlers)) == 1
        lgr.debug("Spam")
        assert len(sys_stdout.write.call_args_list) == 2
        assert sys_stdout.write.call_args_list[1] == mock.call("\n")
        assert sys_stdout.write.call_args_list[0][0][0].endswith(" DEBUG open_cp - Spam")
        
def test_log_to_true_stdout(lgr):
    with mock.patch("open_cp.logger._sys.__stdout__") as sys_stdout:
        assert len(list(lgr.handlers)) == 0
        logger.log_to_true_stdout()
        assert len(list(lgr.handlers)) == 1
        lgr.debug("Spam")
        assert len(sys_stdout.write.call_args_list) == 2
        assert sys_stdout.write.call_args_list[1] == mock.call("\n")
        assert sys_stdout.write.call_args_list[0][0][0].endswith(" DEBUG open_cp - Spam")

def test_dont_add_two_loggers(lgr):
    assert len(list(lgr.handlers)) == 0
    logger.log_to_stdout()
    assert len(list(lgr.handlers)) == 1
    logger.log_to_stdout()
    assert len(list(lgr.handlers)) == 1
    
    ch = logging.StreamHandler()
    lgr.addHandler(ch)
    assert len(list(lgr.handlers)) == 2
    logger.log_to_stdout()
    assert len(list(lgr.handlers)) == 2
    
def test_ProgressLogger():
    pl = logger.ProgressLogger(100, datetime.timedelta(days=1))
    assert pl.increase_count()[:2] == (1, 100)
    assert pl.increase_count() == None
    assert pl.count == 2

def test_ProgressLogger_logout():
    sio = io.StringIO()
    lgr = logging.getLogger("test")
    ch = logging.StreamHandler(sio)
    lgr.addHandler(ch)
    lgr.setLevel(logging.DEBUG)
    
    pl = logger.ProgressLogger(100, datetime.timedelta(days=1))
    pl.logger = lgr
    assert pl.increase_count() == None
    assert sio.getvalue().startswith("Completed 1 out of 100, time left: 0:00:00")
    assert pl.increase_count() == None
    assert pl.count == 2

def test_ProgressLogger_time_between_outputs():
    dtreal = datetime.datetime
    with mock.patch("open_cp.logger._datetime.datetime") as dtm:
        dtm.now.return_value = dtreal(2017, 8, 8)
        pl = logger.ProgressLogger(100, datetime.timedelta(minutes=1))
        assert pl.increase_count()[:2] == (1, 100)
        assert pl.increase_count() == None
        dtm.now.return_value = dtreal(2017, 8, 8, 0, 0, 59)
        assert pl.increase_count() == None
        dtm.now.return_value = dtreal(2017, 8, 8, 0, 1, 0)
        # 4 ticks has taken a minute, so 100 should take 25 minutes
        assert pl.increase_count() == (4, 100, datetime.timedelta(minutes=24))
    