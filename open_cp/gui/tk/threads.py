"""
threads
~~~~~~~

Provides a high-level interface to allow running background tasks in threads,
and communicating with the main GUI thread.
"""

# Helpful resources:
# http://stupidpythonideas.blogspot.co.uk/2013/10/why-your-gui-app-freezes.html

import concurrent.futures as _futures
import queue as _queue
import threading as _threading
import logging

class BackgroundTasks():
    """Provides a way for multiple threads to queue up callable objects which
    are to be run on the main GUI thread.
    
    :param root: A `tkinter` object, typically the root window.
    """
    def __init__(self, root):
        self._root = root
        self._queue = _queue.Queue()
        self._root.after(self._CALLBACK_TIME_MS, self._query_task)
        
    _CALLBACK_TIME_MS = 50
        
    def submit(self, task):
        """Submit, from any thread, a callable object to be run on the main
        gui thread.
        
        :param task: A callable object
        """
        self._queue.put(task)
        
    def _query_task(self):
        try:
            while True:
                task = self._queue.get(block=False)
                task()
        except _queue.Empty:
            pass
        # This means it continually runs, which is wasteful.
        # But we _must_ make this call from the correct thread, so if we
        # stop it, we run into the problem of only allowing submissions from
        # the main thread, which might be considered restrictive...
        self._root.after(self._CALLBACK_TIME_MS, self._query_task)


class OffThreadTask():
    """Abstract base class for tasks to be run off the main GUI thread, and
    then whose return value should be passed to a (small) task to run on the
    GUI thread.
    
    Also provides a standard way to cancel a task: the running task should call
    the constructor, and then periodcally poll the :attr:`cancelled'.
    """
    def __init__(self):
        self._cancelled = False
        self._cancelling_lock = _threading.RLock()

    def cancel(self):
        with self._cancelling_lock:
            self._cancelled = True

    @property
    def cancelled(self):
        if hasattr(self, "_cancelling_lock"):
            with self._cancelling_lock:
                return self._cancelled
        return False

    def __call__(self):
        """Task to be run off thread"""
        raise NotImplementedError
        
    def on_gui_thread(self, value):
        """Task to be run on thread with the return value of ``__call__()``."""
        raise NotImplementedError


class Pool():
    """Simple wrapper around a `ThreadPoolExecutor` which uses a
    `BackgroundTasks` instance to communicate with the GUI thread.
    
    :param root: A `tkinter` object, typically the root window
    :param max_threads: The maximum number of threads to run; `None` for the
      default of `ThreadPoolExecutor`.
    """
    def __init__(self, root, max_threads=None):
        self._task_manager = BackgroundTasks(root)
        self._executor = _futures.ThreadPoolExecutor(max_threads)
        self._logger = logging.getLogger(__name__)
        
    def submit_gui_task(self, task):
        """Directly submit a task to be run on the main GUI thread.  Can be
        called from any thread (hence the point...)
        """
        self._task_manager.submit(task)
        
    def submit(self, task, at_finish=None):
        """Submit a task to be run by the thread pool.  Any exception thrown by
        the `task` will be caught, logged, and returned to the `at_finish`
        task.
        
        :param task: A callable object to be run.  Once completed, its return
          value will be passed to `at_finish`.  Or an instance of
          :class:`OffThreadTask`.
        :param at_finish: A callback object with signature `at_finish(value)`
          which will be called on the main gui thread when the return value of
          the `task`.  Should be `None` if and only if `task` is an instance of
          :class:`OffThreadTask`
        """
        if isinstance(task, OffThreadTask):
            if at_finish is not None:
                raise ValueError("Cannot specify an `at_finish` task if `task` is an `OffThreadTask` instance.""")
            self.submit(task.__call__, task.on_gui_thread)
            return
        if at_finish is None:
            raise ValueError("Need to specify a task to run on completion")
        
        def task_wrapper():
            try:
                value = task()
            except Exception as ex:
                self._logger.exception("In task")
                value = ex
            def on_gui_thread_task():
                at_finish(value)
            self._task_manager.submit(on_gui_thread_task)
        
        self._executor.submit(task_wrapper)
