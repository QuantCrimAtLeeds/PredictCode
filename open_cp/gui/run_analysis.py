"""
run_analysis
~~~~~~~~~~~~

Model / Controller for running the actual analysis.

TODO: The random `print` statements are to try to catch a rare deadlock
  condition.
"""

import open_cp.gui.tk.run_analysis_view as run_analysis_view
import open_cp.gui.predictors as predictors
from open_cp.gui.common import CoordType
import open_cp.gui.predictors.predictor as predictor
import open_cp.pool as pool
import open_cp.gui.tk.threads as tk_threads
import open_cp.gui.locator as locator
import collections
import logging
import queue
import time
import datetime

class RunAnalysis():
    """Controller for performing the computational tasks of actually producing
    a prediction.  Using multi-processing.

    :param parent: Parent `tk` widget
    :param controller: The :class:`analyis.Analysis` model.
    """
    def __init__(self, parent, controller):
        self.view = run_analysis_view.RunAnalysisView(parent, self)
        self.controller = controller
        self._msg_logger = predictors.get_logger()
        self._logger = logging.getLogger(__name__)

    @property
    def main_model(self):
        """The :class:`analysis.Model` instance"""
        return self.controller.model

    def run(self):
        try:
            self._model = RunAnalysisModel(self, self.main_model)
            self._run_tasks()
        except:
            self._msg_logger.exception(run_analysis_view._text["genfail"])
            self.view.done()
        self.view.wait_window(self.view)

    def cancel(self):
        """Called when we wish to cancel the running tasks"""
        self._logger.warning("Analysis run being cancelled.")
        self._msg_logger.warning(run_analysis_view._text["log10"])
        if hasattr(self, "_off_thread"):
            self._off_thread.cancel()

    @staticmethod
    def _chain_dict(dictionary):
        for name, li in dictionary.items():
            for x in li:
                yield (name, x)

    def _run_tasks(self):
        tasks = []
        for proj_name, proj in self._chain_dict(self._model.projectors):
            for grid_name, grid in self._chain_dict(self._model.grids):
                for pred_name, pred in self._chain_dict(self._model.grid_prediction_tasks):
                    task = _RunAnalysis_Task(
                        task = _RunAnalysis_Task._InnerTask(self.main_model, grid, proj, pred),
                        off_process = pred.off_process,
                        projection = proj_name,
                        grid = grid_name,
                        type = pred_name )
                    tasks.append(task)

        total = len(tasks) * len(self._model.predict_tasks)
        self._msg_logger.info(run_analysis_view._text["log7"], total)
        
        self._off_thread = _RunnerThread(tasks, self._model.predict_tasks, self)
        self._off_thread.force_gc()
        locator.get("pool").submit(self._off_thread, self._finished)

    def to_msg_logger(self, msg, *args, level=logging.DEBUG):
        self._msg_logger.log(level, msg, *args)

    def start_progress(self):
        locator.get("pool").submit_gui_task(lambda : self.view.start_progress_bar())

    def set_progress(self, done, out_of):
        locator.get("pool").submit_gui_task(lambda : self.view.set_progress(done, out_of))

    def end_progress(self):
        locator.get("pool").submit_gui_task(lambda : self.view.stop_progress_bar())

    def notify_model_message(self, msg, *args, level=logging.DEBUG):
        self.to_msg_logger(msg, *args, level=level)

    def _finished(self, out=None):
        self.view.done()
        if out is not None:
            if isinstance(out, predictor.PredictionError):
                self.view.alert(str(out))
                self._msg_logger.error(run_analysis_view._text["warning1"].format(out))
            elif isinstance(out, Exception):
                self._msg_logger.error(run_analysis_view._text["log11"].format(out))
            else:
                self._msg_logger.error(run_analysis_view._text["log12"].format(out))
            return
        if self._off_thread.cancelled:
            self.view.cancel()
        else:
            results = [PredictionResult(key, result) for (key, result) in self._off_thread.results]
            result = RunAnalysisResult(results)
            self.controller.new_run_analysis_result(result)


class _RunAnalysis_Task():
    """Pulled out to allow pickling"""
    def __init__(self, task, off_process, projection, grid, type):
        self.task = task
        self.off_process = off_process
        self.projection = projection
        self.grid = grid
        self.type = type
        
    def __repr__(self):
        return "_RunAnalysis_Task(task={}, off_process={}, projection={}, grid={}, type={})".format(
                self.task, self.off_process, self.projection, self.grid, self.type)

    class _InnerTask():
        def __init__(self, main_model, grid, proj, pred):
            # Make a copy of the :class:`DataModel` and not the extra baggage.
            self.main_model = main_model.clone()
            self.grid = grid
            self.proj = proj
            self.pred = pred
            
        def __call__(self):
            return self.pred(self.main_model, self.grid, self.proj)


class RunAnalysisResult():
    def __init__(self, results):
        self._results = results
        self._time = datetime.datetime.now()

    @property
    def results(self):
        """List of :class:`PredictionResult` instances."""
        return self._results

    @property
    def run_time(self):
        """:class:`datetime` of when the result was completed."""
        return self._time


def merge_all_results(results):
    """Merge an iterable of :class:`RunAnalysisResult` instances into a single
    :class:`RunAnalysisResult` object."""
    all_results = []
    for result in results:
        all_results.extend(result.results)
    return RunAnalysisResult(all_results)


class PredictionResult():
    """The result of running the prediction, but not including any analysis
    results.

    :param key: Instance of :class:`TaskKey`
    :param prediction: The result of the prediction.  Slightly undefined, but
      at present, should be an :class:`GridPrediction` instance.
    """
    def __init__(self, key, prediction):
        self._key = key
        self._pred = prediction

    @property
    def key(self):
        """The :class:`TaskKey` describing the prediction."""
        return self._key

    @property
    def prediction(self):
        """An instance of :class:`GridPrediction` (or most likely a subclass)
        giving the actual prediction."""
        return self._pred

    def __repr__(self):
        return "PredictionResult(key={}, prediction={}".format(self._key, self._pred)


class TaskKey():
    """Describes the prediction task which was run.  We don't make any
    assumptions about the components of the key (they are currently strings,
    but in future may be richer objects) and don't implement custom hashing
    or equality.
    
    :param projection: The projection used.
    :param grid: The grid used.
    :param pred_type: The prediction algorithm (etc.) used.
    :param pred_date: The prediction date.
    :param pred_length: The length of the prediction.
    """
    def __init__(self, projection, grid, pred_type, pred_date, pred_length):
        self._projection = projection
        self._grid = grid
        self._pred_type = pred_type
        self._pred_date = pred_date
        self._pred_length = pred_length

    @property
    def projection(self):
        return self._projection

    @property
    def grid(self):
        return self._grid

    @property
    def prediction_type(self):
        return self._pred_type

    @property
    def prediction_date(self):
        return self._pred_date

    @property
    def prediction_length(self):
        return self._pred_length
    
    @staticmethod
    def header():
        """Column representation for CSV file"""
        return ["projection type", "grid type", "prediction type", "prediction date", "scoring length"]
    
    def __iter__(self):
        return iter((self.projection, self.grid, self.prediction_type,
                self.prediction_date, self.prediction_length))

    def __repr__(self):
        return "projection: {}, grid: {}, prediction_type: {}, prediction_date: {}, prediction_length: {}".format(
            self.projection, self.grid, self.prediction_type, self.prediction_date,
            self.prediction_length)


class RunAnalysisModel():
    """The model for running an analysis.  Constructs lists/dicts:
      - :attr:`projector_tasks` Tasks to project coordinates
      - :attr:`grid_tasks` Tasks to lay a grid over the data
      - :attr:`predict_tasks` Pairs `(start_date, score_length)`
      - :attr:`grid_pred_tasks` Instances of :class:`GridPredictorTask`
    
    :param controller: :class:`RunAnalysis` instance
    :param view: :class:`RunAnalysisView` instance
    :param main_model: :class:`analysis.Model` instance
    """
    def __init__(self, controller, main_model):
        self.controller = controller
        self.main_model = main_model

        self._build_projectors()
        self._build_grids()
        self._build_date_ranges()
        self._build_grid_preds()

    def _build_grid_preds(self):
        self._grid_pred_tasks = dict()
        for pred in self.predictors.predictors_of_type(predictors.predictor._TYPE_GRID_PREDICTOR):
            self._grid_pred_tasks[pred.pprint()] = pred.make_tasks()
        self.controller.notify_model_message(run_analysis_view._text["log1"],
            sum( len(li) for li in self._grid_pred_tasks.values() ),
            level=logging.INFO)

    def _build_date_ranges(self):
        self._predict_tasks = []
        for top in self.comparators.comparators_of_type(predictors.comparitor.TYPE_TOP_LEVEL):
            self._predict_tasks.extend(top.run())
        self.controller.notify_model_message(run_analysis_view._text["log2"],
            len(self.predict_tasks), level=logging.INFO)
        if len(self.predict_tasks) > 0:
            self.controller.notify_model_message(run_analysis_view._text["log3"],
                self.predict_tasks[0][0].strftime(run_analysis_view._text["dtfmt"]),
                level=logging.INFO)
            self.controller.notify_model_message(run_analysis_view._text["log4"],
                self.predict_tasks[-1][0].strftime(run_analysis_view._text["dtfmt"]),
                level=logging.INFO)

    @property
    def predict_tasks(self):
        """List of pairs `(start_date, length)`"""
        return self._predict_tasks

    def _build_grids(self):
        self._grid_tasks = dict()
        for grid in self.predictors.predictors_of_type(predictors.predictor._TYPE_GRID):
            tasks = grid.make_tasks()
            self._grid_tasks[grid.pprint()] = tasks
        self.controller.notify_model_message(run_analysis_view._text["log5"],
            sum( len(li) for li in self._grid_tasks.values() ), level=logging.INFO )

    def _build_projectors(self):
        if self.main_model.coord_type == CoordType.XY:
            projector = predictors.lonlat.PassThrough(self.main_model)
            projectors = [projector]
        else:
            projectors = list(self.predictors.predictors_of_type(
                predictors.predictor._TYPE_COORD_PROJ))
        
        count = 0
        self._projector_tasks = dict()
        for projector in projectors:
            tasks = projector.make_tasks()
            self._projector_tasks[projector.pprint()] = tasks
            count += len(tasks)
        self.controller.notify_model_message(run_analysis_view._text["log6"],
            count, level=logging.INFO)

    @property
    def grid_prediction_tasks(self):
        """Dictionary from string name to task(s)."""
        return self._grid_pred_tasks

    @property
    def grids(self):
        """Dictionary from string name to task(s)."""
        return self._grid_tasks

    @property
    def projectors(self):
        """Dictionary from string name to task(s)."""
        return self._projector_tasks

    @property
    def predictors(self):
        return self.main_model.analysis_tools_model

    @property
    def comparators(self):
        return self.main_model.comparison_model


class BaseRunner():
    """Abstract base class which runs "tasks" and communicates with a
    "controller" to show progress.
    """
    def __init__(self, controller):
        self._executor = pool.PoolExecutor()
        self._results = []
        self._controller = controller
        self._cancel_queue = queue.Queue()

    def __call__(self):
        """To be run off the main GUI thread"""
        self._controller.start_progress()
        self._controller.to_msg_logger(run_analysis_view._text["log9"])
        self.executor.start()
        try:
            tasks = list(self.make_tasks())
            self._controller.to_msg_logger(run_analysis_view._text["log13"])
            futures = [ self.executor.submit(t) for t in tasks if t.off_process ]
            on_thread_tasks = [t for t in tasks if not t.off_process]
            done, out_of = 0, len(futures) + len(on_thread_tasks)

            while len(futures) > 0 or len(on_thread_tasks) > 0:
                if len(futures) > 0:
                    futures, count = self._process_futures(futures)
                    done += count
                if len(on_thread_tasks) > 0:
                    task = on_thread_tasks.pop()
                    self._notify_result(task.key, task())
                    done += 1
                else:
                    time.sleep(0.5)
                self._controller.set_progress(done, out_of)
                if self.cancelled:
                    print("Exiting...")
                    break
        finally:
            # Context management would call `shutdown` but we definitely want
            # to call terminate.
            print("Terminating...")
            self.executor.terminate()
            print("Done")
        print("Ending progress...")
        self._controller.end_progress()

    def force_gc(self):
        """Fixes, or at least mitigates, issue #6.  Call on the main GUI thread
        prior to invoking `__call__`."""
        import gc
        gc.collect()

    @property
    def executor(self):
        return self._executor

    def _process_futures(self, futures):
        results, futures = pool.check_finished(futures)
        done = 0
        for key, result in results:
            self._notify_result(key, result)
            done += 1
        return futures, done

    def _notify_result(self, key, result):
        self._results.append( (key, result) )
        self._controller.to_msg_logger(run_analysis_view._text["log8"], key)

    def cancel(self):
        self._cancel_queue.put("stop")

    @property
    def cancelled(self):
        return not self._cancel_queue.empty()

    @property
    def results(self):
        return self._results

    def make_tasks(self):
        """To be over-riden in a sub-class.  Should return a list of
        :class:`RunPredTask` instances."""
        raise NotImplementedError()
    
    class RunPredTask(pool.Task):
        """Wraps a `key` and `task`.  The task should have an attribute
        :attr:`off_process` which is `True` if and only if we should run in
        another process.
        """
        def __init__(self, key, task):
            super().__init__(key)
            if task is None:
                raise ValueError()
            self._task = task

        @property
        def off_process(self):
            return self._task.off_process

        def __call__(self):
            return self._task()


class _RunnerThread(BaseRunner):
    """Constructs the tasks to run.  Essentially forms the cartesian product
    of the prediction tasks with the date ranges.
    
    The `result` will be :class:`PredictionResult` instances.
    
    :param grid_prediction_tasks: Iterable giving callables which when run
        return instances of :class:`SingleGridPredictor`.
    :param predict_tasks: Iterable of pairs `(start_time, score_length)`
    :param controller: The :class:`RunAnalysis` instance
    """
    def __init__(self, grid_prediction_tasks, predict_tasks, controller):
        super().__init__(controller)
        self._tasks = list(grid_prediction_tasks)
        self._date_ranges = list(predict_tasks)

    def make_tasks(self):
        tasks = []
        futures = []
        for task in self._tasks:
            if task.off_process:
                #raise NotImplementedError("This currently does not work due to pickling issues.")
                task = self.RunPredTask(task, task.task)
                futures.append(self.executor.submit(task))
            else:
                tasks.extend( self._make_new_task(task, task.task()) )
        if len(futures) > 0:
            for key, result in pool.yield_task_results(futures):
                tasks.extend( self._make_new_task(key, result) )
        return tasks

    def _make_new_task(self, key, task):
        for dr in self._date_ranges:
            new_task = self.StartLengthTask(task=task, start=dr[0], length=dr[1])
            k = TaskKey(projection=key.projection, grid=key.grid,
                    pred_type=key.type, pred_date=dr[0], pred_length=dr[1] )
            yield self.RunPredTask(k, new_task)

    class StartLengthTask():
        def __init__(self, task, start, length):
            self.start = start
            self.length = length
            if task is None:
                raise ValueError()
            self.task = task
        
        def __call__(self):
            return self.task(self.start, self.length)

        @property
        def off_process(self):
            return self.task.off_process

