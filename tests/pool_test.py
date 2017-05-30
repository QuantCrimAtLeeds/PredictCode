import pytest
import unittest.mock
import tests.helpers
import pickle

import open_cp.pool as pool

class OurTask(pool.Task):
    def __init__(self, key):
        super().__init__(key)

    def __call__(self):
        return "ahgsga"

@pytest.fixture
def mockPPE():
    with unittest.mock.patch("concurrent.futures.ProcessPoolExecutor") as mockPPEClass:
        mockPPEClass.return_value = unittest.mock.MagicMock()
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

@unittest.mock.patch("concurrent.futures.as_completed")
def test_yield_task_results(mock):
    x = [1,2,3]
    for y in pool.yield_task_results(x, 100):
        pass
    mock.assert_called_once_with(x, 100)

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
        r = unittest.mock.MagicMock()
        r.result.return_value = ("ytreqe", "adgas")
        yield r
        r = unittest.mock.MagicMock()
        r.result.return_value = ("111", "22")
        yield r

@unittest.mock.patch("concurrent.futures.as_completed")
def test_RestorableExecutor_getResults(mock, mockPPE):
    mock.return_value = ResultsTest()
    executor = pool.RestorableExecutor("")
    with executor:
        fut = executor.submit(OurTask("absvs"))

    assert(executor.results == {"111": "22", "ytreqe": "adgas"})

@unittest.mock.patch("concurrent.futures.as_completed")
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

@unittest.mock.patch("concurrent.futures.as_completed")
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

@unittest.mock.patch("concurrent.futures.as_completed")
def test_RestorableExecutor_savePartialResult(mock, mockPPE):
    file = tests.helpers.BytesIOWrapper()

    def yield_once(a, b):
        future = unittest.mock.MagicMock()
        future.result.return_value = ("absa", 123)
        yield future
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        with unittest.mock.patch("builtins.open", tests.helpers.MockOpen(file)) as open_mock:
            open_mock.filter = tests.helpers.ExactlyTheseFilter([2])
            mock.side_effect = yield_once
            executor = pool.RestorableExecutor("")
            with executor:
                fut = executor.submit(OurTask("absvs"))

    got = pickle.loads(file.data)
    assert(got == {"absa": 123})
