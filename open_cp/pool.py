"""
pool
~~~~

Implements a simple process based task runner.  At its simplest, this is a thin
wrapper around the features of :module:`concurrent.futures` in the standard
library.

We also provide a mechanism for allowing scripts to be closed down (gracefully,
or not) and to then restart roughly where they left off.
"""

import logging as _log
import concurrent.futures as _futures
import pickle as _pickle
import multiprocessing as _mp
import multiprocessing.pool as _mp_pool
import time as _time

class OurProcessPoolExecutor():
    """Minimal version of the standard library :class:`ProcessPoolExecutor`
    which raises exceptions on failure to launch a worker task.  Uses
    :mod:`multiprocessing.Pool` internally.
    """
    def __init__(self):
        self._pool = _mp.Pool()

    def __enter__(self):
        return self

    def __exit__(self, ex, b, c):
        self.shutdown(True)

    def shutdown(self, wait):
        self._pool.close()
        if wait:
            self._pool.join()
        self._pool = None

    def terminate(self):
        """Terminates the process pool immediately."""
        self._pool.terminate()

    def submit(self, task, *args, **kwargs):
        if self._pool is None:
            raise RuntimeError("Pool already shutdown")
        async_result = self._pool.apply_async(task, *args, **kwargs)
        return self.OurFuture(async_result)

    class OurFuture(_futures.Future):
        def __init__(self, result):
            self._async_result = result

        def result(self, timeout=None):
            return self._async_result.get(timeout)

        def exception(self, timeout=None):
            try:
                res = self._async_result.get(timeout)
            except Exception as ex:
                return ex
            return None

        def done(self):
            return self._async_result.ready()


#_ProcessPoolExecutor = _futures.ProcessPoolExecutor
_ProcessPoolExecutor = OurProcessPoolExecutor


class Task():
    """Abstract base class for "tasks".  Each task has a "key" which should be
    unique, as it is used to pair results with tasks.  A task object is
    callable, and clients should subclass and implement the actual task.

    :param key: A unique key to identify the task.
    """
    def __init__(self, key):
        self._key = key

    @property
    def key(self):
        return self._key

    def __call__(self):
        raise NotImplementedError()


class _TaskWrapper():
    def __init__(self, task):
        self.task = task

    def __call__(self):
        return self.task.key, self.task()


class PoolExecutor():
    """Executor which runs :class:`Task` instances.  Implements the context
    manager interface, and exiting the context will wait until all tasks are
    completed.  Typical usage is thus:

        with PoolExecutor() as executor:
            futures = [executor.submit(task) for task in tasks]
            for result in yield_task_results(futures):
                # Process result
                pass
    """
    def __init__(self):
        self._executor = None
        self._logger = _log.getLogger(PoolExecutor.__name__)
    
    def __enter__(self):
        self.start()
        return self

    def __exit__(self,a,b,c):
        self.shutdown()

    def start(self):
        """Manually start the executor, instead of using as a context."""
        self._executor = _ProcessPoolExecutor()

    def shutdown(self):
        """Manually stop the executor"""
        self._executor.shutdown(False)
        self._executor = None

    def terminate(self):
        """Terminate the executor immediately."""
        self._executor.terminate()
        self._executor = None

    def submit(self, task):
        """Submit the task to be run.

        :return: A :class:`Future` object whose `result` will be a pair of
          `key` and the result of the task.
        """
        if self._executor is None:
            raise RuntimeError("Executor not started: use as a context manager.")
        func = _TaskWrapper(task)
        self._logger.debug("Submitting task %s to executor", task.key)
        fut = self._executor.submit(func)
        return fut


def check_finished(futures):
    """Pass over the list of :class:`Future` or :class:`AsyncResult` objects,
    and find results for those which are done, also returning a list of those
    which are still running.  This is not terribly efficient, as it linearly
    walks over the futures checking each one.

    :param futures: Iterable of :class:`Future` or :class:`AsyncResult`
      objects, or a mixture.

    :return: Pair `(results, futures)` where `results` is a list of the pairs
       `(key, return value)`, and `futures` is a list of those inputs which are
       still running.
    """
    results, still_to_complete = [], []
    for future in futures:
        if isinstance(future, _mp_pool.AsyncResult):
            done = future.ready()
        elif isinstance(future, _futures.Future):
            done = future.done()
        else:
            raise ValueError("Unexpected class: ", type(future))
        if done:
            results.append( future.result() )
        else:
            still_to_complete.append( future )
    return (results, still_to_complete)

def yield_task_results(futures, timeout=None):
    """Standard way to extract the key and return value from a future wrapping
    a task.  Yields pairs `(key, return value)` as the futures complete.

    This is a simple implementation that repeatedly checks the future objects
    for completion, and then `sleep`s for a short time.  As such, it is not
    suitable for low-latency applications.
    
    :param futures: Iterable of futures to wait for
    :param timeout: An optional timeout is seconds
    """
    if timeout is not None:
        import datetime
        end = datetime.datetime.now() + datetime.timedelta(seconds=timeout)
    wait_for = list(futures)
    while len(wait_for) > 0:
        results, wait_for = check_finished(wait_for)
        yield from results
        if timeout is not None and datetime.datetime.now() > end:
            raise TimeoutError()
        _time.sleep(0.1)


class RestorableExecutor():
    """An executor which will save current progress on a `KeyboardInterrupt`.
    Use as a context manager, take the context, `submit` all your tasks (with
    unique keys) and then leave the context.  All the tasks will be run using
    a :class:`ProcessPoolExecutor` and will then be available from the
    :attr:`results` property.

    If a `KeyboardInterrupt` is raised, the key/value pairs so far generated
    will be pickled and saved to the given filename.

    When initialising, if the file exists, an attempt will be made to unpickle
    it to a dictionary.  Any keys present in the dictionary will not be
    recomputed.

    On successful completion, results are _not_ written to file!

    :param filename: Filename to use if interupted.
    """
    def __init__(self, filename):
        self._logger = _log.getLogger(RestorableExecutor.__name__)
        self._filename = filename
        self._results = self._load()
        self._futures = []
        self._executor = None

    def _load(self):
        try:
            with open(self._filename, "rb") as file:
                old_dict = _pickle.load(file)
                self._logger.info("Read back %d keys from %s", len(old_dict), self._filename)
                return old_dict
        except FileNotFoundError:
            return dict()

    def _save(self):
        with open(self._filename, "wb") as file:
            _pickle.dump(self._results, file)
            self._logger.info("Wrote %d keys to %s", len(self._results), self._filename)

    def submit(self, task):
        """Submit the task to be run."""
        if self._executor is None:
            raise RuntimeError("Executor not started: use as a context manager.")
        if task.key in self._results:
            self._logger.debug("Task %s already complete", task.key)
            return
        func = _TaskWrapper(task)
        self._logger.debug("Submitting task %s to executor", task.key)
        self._futures.append( self._executor.submit(func) )

    @property
    def results(self):
        """Dictionary of the results of the tasks.  The task key is used to
        retrive the task result.
        """
        if self._executor is not None:
            raise RuntimeError("Executor is still running.")
        return self._results

    def __enter__(self):
        self._executor = _ProcessPoolExecutor()
        return self

    def __exit__(self, ex, b, c):
        if ex is None:
            try:
                print("Here...")
                for k, v in yield_task_results(self._futures):
                    print("Got", k, v)
                    self._results[k] = v
            except KeyboardInterrupt as ex:
                self._logger.warning("KeyboardInterrupt detected; saving results so far...")
                self._save()
                raise ex
        self._executor.shutdown(False)
        self._executor = None
