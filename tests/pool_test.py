import pytest
import unittest.mock
import tests.helpers
import pickle
import concurrent.futures
import datetime, time

import open_cp.pool as pool

class OurTask(pool.Task):
    def __init__(self, key, wait=0, ex=None):
        super().__init__(key)
        self._wait = wait
        self._ex = ex

    def __call__(self):
        if self._wait > 0:
            time.sleep(self._wait)
        if self._ex is not None:
            raise self._ex
        return "ahgsga"

def test_runs():
    with pool.PoolExecutor() as executor:
        futures = [executor.submit(OurTask("absvs"))]
        for key, result in pool.yield_task_results(futures):
            assert key == "absvs"
            assert result == "ahgsga"

def test_runs_timeout():
    with pool.PoolExecutor() as executor:
        now = datetime.datetime.now()        
        futures = [executor.submit(OurTask("ab", wait=0.5))]
        for key, result in pool.yield_task_results(futures):
            assert key == "ab"
            assert result == "ahgsga"
            assert datetime.datetime.now() - now >= datetime.timedelta(seconds=0.5)

def test_runs_exception():
    with pool.PoolExecutor() as executor:
        futures = [executor.submit(OurTask("ab", ex=RuntimeError()))]
        with pytest.raises(RuntimeError):
            for key, result in pool.yield_task_results(futures):
                pass
        
def test_runs_exception_get_exceptions():
    with pool.PoolExecutor() as executor:
        future = executor.submit(OurTask("ab", ex=RuntimeWarning()))
        assert isinstance(future.exception(), RuntimeWarning)

def test_catches_internal_error():
    class LambdaTask(pool.Task):
        def __init__(self, task):
            super().__init__("key")
            self._task = task
            
        def __call__(self):
            return self._task()
    
    with pool.PoolExecutor() as executor:
        task = LambdaTask(lambda : 5)
        future = executor.submit(task)
        with pytest.raises(AttributeError):
            future.result(timeout=1)

@pytest.fixture
def mockPPE():
    with unittest.mock.patch("open_cp.pool._ProcessPoolExecutor") as mockPPEClass:
        mockPPEClass.return_value = unittest.mock.MagicMock()
        fut = concurrent.futures.Future()
        fut.set_result(result=("test key", "test result"))
        mockPPEClass.return_value.submit.return_value = fut
        yield mockPPEClass.return_value

def test_PoolExecutor(mockPPE):
    with pool.PoolExecutor() as executor:
        fut = executor.submit(OurTask("absvs"))

    func = mockPPE.submit.call_args[0][0]
    assert(func() == ("absvs", "ahgsga"))

    mockPPE.shutdown.assert_called_once_with(False)

def test_PoolExecutor_cantSubmitBeforeEntered(mockPPE):
    executor = pool.PoolExecutor()
    with pytest.raises(RuntimeError):
        executor.submit(OurTask(1))

def test_yield_task_results():
    fut = concurrent.futures.Future()
    fut.set_result(1)
    fut1 = concurrent.futures.Future()
    fut1.set_result(10)
    assert list(pool.yield_task_results([fut, fut1])) == [1, 10]

def test_check_finished():
    fut = concurrent.futures.Future()
    fut.set_running_or_notify_cancel()
    results, futures = pool.check_finished([fut])
    assert len(results) == 0
    assert futures == [fut]

def test_yield_task_results_timeout():
    fut = concurrent.futures.Future()
    fut.set_running_or_notify_cancel()
    with pytest.raises(TimeoutError):
        for x in pool.yield_task_results([fut], 0.1):
            raise AssertionError()

@unittest.mock.patch("concurrent.futures.as_completed")
def test_RestorableExecutor(mock, mockPPE):
    executor = pool.RestorableExecutor("")
    with executor:
        fut = executor.submit(OurTask("absvs"))

    mockPPE.shutdown.assert_called_once_with(False)
    func = mockPPE.submit.call_args[0][0]
    assert(func() == ("absvs", "ahgsga"))

class ResultsTest():
    def __iter__(self):
        yield ("ytreqe", "adgas")
        yield ("111", "22")

@unittest.mock.patch("open_cp.pool.yield_task_results")
def test_RestorableExecutor_getResults(mock, mockPPE):
    mock.return_value = ResultsTest()
    executor = pool.RestorableExecutor("")
    with executor:
        fut = executor.submit(OurTask("absvs"))

    assert(executor.results == {"111": "22", "ytreqe": "adgas"})

@unittest.mock.patch("open_cp.pool.yield_task_results")
def test_RestorableExecutor_cannotGetResultInContext(mock, mockPPE):
    executor = pool.RestorableExecutor("")
    with executor:
        fut = executor.submit(OurTask("absvs"))
        with pytest.raises(RuntimeError):
            executor.results

def test_RestorableExecutor_cannotSubmitBeforeEntering(mockPPE):
    executor = pool.RestorableExecutor("")
    with pytest.raises(RuntimeError):
        executor.submit(OurTask(1))

@unittest.mock.patch("open_cp.pool.yield_task_results")
def test_RestorableExecutor_dontRecompute(mock, mockPPE):
    existing = {"absvs": 5}
    with unittest.mock.patch("builtins.open", tests.helpers.MockOpen(pickle.dumps(existing))):
        executor = pool.RestorableExecutor("")
        with executor:
            fut = executor.submit(OurTask("absvs"))
            fut = executor.submit(OurTask("abstt"))

        func = mockPPE.submit.call_args[0][0]
        assert(func() == ("abstt", "ahgsga"))
        assert( len(mockPPE.submit.call_args_list) == 1 )

        assert( executor.results == existing )

@unittest.mock.patch("open_cp.pool.yield_task_results")
def test_RestorableExecutor_savePartialResult(mock, mockPPE):
    file = tests.helpers.BytesIOWrapper()

    class YieldOnce():
        def __iter__(self):
            yield ("absa", 123)
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        with unittest.mock.patch("builtins.open", tests.helpers.MockOpen(file)) as open_mock:
            open_mock.filter = tests.helpers.ExactlyTheseFilter([2])
            mock.return_value = YieldOnce()
            executor = pool.RestorableExecutor("")
            with executor:
                fut = executor.submit(OurTask("absvs"))

    got = pickle.loads(file.data)
    assert(got == {"absa": 123})
