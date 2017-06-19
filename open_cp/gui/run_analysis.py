"""
run_analysis
~~~~~~~~~~~~

Model / Controller for running the actual analysis.
"""

import open_cp.gui.tk.run_analysis_view as run_analysis_view
import open_cp.gui.predictors as predictors
from open_cp.gui.common import CoordType
import open_cp.pool as pool
import open_cp.gui.tk.threads as tk_threads
import open_cp.gui.locator as locator

class RunAnalysis():
    """

    :param parent: Parent `tk` widget
    :param model: The :class:`analyis.Model` model.
    """
    def __init__(self, parent, model):
        self.view = run_analysis_view.RunAnalysisView(parent)
        self.main_model = model
        self._msg_logger = predictors.get_logger()

    def run(self):
        try:
            self._model = RunAnalysisModel(self, self.view, self.main_model)
            self._log_total_tasks()
            self._run_tasks()
            self.view.wait_window(self.view)
        except:
            self._msg_logger.exception(run_analysis_view._text["genfail"])
            self.view.done()

    def _run_tasks(self):
        tasks = []
        for proj in self._model.projector_tasks:
            for grid in self._model.grid_tasks:
                for pred in self._model.grid_pred_tasks:
                    task = self._Wrapper(lambda : pred(self.main_model, grid, proj),
                        pred.off_thread())
                    tasks.append(task)
        
        off_thread = _RunnerThread(tasks, self._model.predict_tasks)
        locator.get("pool").submit(off_thread, lambda value : self.view.done())

    class _Wrapper():
        def __init__(self, task, off_thread):
            self._task = task
            self.off_thread = off_thread

        def __call__(self):
            self._task()            

    def _log_total_tasks(self):
        total = ( len(self._model.grid_pred_tasks)
            * len(self._model.projector_tasks)
            * len(self._model.grid_tasks)
            * len(self._model.predict_tasks) )
        self._logger.info(run_analysis_view._text["log7"], total)


class RunAnalysisModel():
    """The model for running an analysis.  Constructs lists:
      - :attr:`projector_tasks` Tasks to project coordinates
      - :attr:`grid_tasks` Tasks to lay a grid over the data
      - :attr:`predict_tasks` Pairs `(start_date, score_length)`
      - :attr:`grid_pred_tasks` Instances of :class:`GridPredictorTask`
    
    :param controller: :class:`RunAnalysis` instance
    :param view: :class:`RunAnalysisView` instance
    :param main_model: :class:`analysis.Model` instance
    """
    def __init__(self, controller, view, main_model):
        self.controller = controller
        self.view = view
        self.main_model = main_model
        self._logger = predictors.get_logger()

        self._build_projectors()
        self._build_grids()
        self._build_date_ranges()
        self._build_grid_preds()

    def _build_grid_preds(self):
        self.grid_pred_tasks = []
        for pred in self.predictors.predictors_of_type(predictors.predictor._TYPE_GRID_PREDICTOR):
            self.grid_pred_tasks.extend( pred.make_tasks() )
        self._logger.info(run_analysis_view._text["log1"], len(self.grid_pred_tasks))

    def _build_date_ranges(self):
        self.predict_tasks = []
        for top in self.comparators.comparators_of_type(predictors.comparitor.TYPE_TOP_LEVEL):
            self.predict_tasks.extend(top.run())
        self._logger.info(run_analysis_view._text["log2"], len(self.predict_tasks))
        if len(self.predict_tasks) > 0:
            self._logger.debug(run_analysis_view._text["log3"],
                self.predict_tasks[0][0].strftime(run_analysis_view._text["dtfmt"]))
            self._logger.debug(run_analysis_view._text["log4"],
                self.predict_tasks[-1][0].strftime(run_analysis_view._text["dtfmt"]))

    def _build_grids(self):
        self.grid_tasks = []
        for grid in self.predictors.predictors_of_type(predictors.predictor._TYPE_GRID):
            self.grid_tasks.extend( grid.make_tasks() )
        self._logger.info(run_analysis_view._text["log5"], len(self.grid_tasks))

    def _build_projectors(self):
        if self.main_model.coord_type == CoordType.XY:
            tasks = predictors.lonlat.PassThrough().make_tasks()
        else:
            tasks = []
            for proj in self.predictors.predictors_of_type(predictors.predictor._TYPE_COORD_PROJ):
                tasks.extend( proj.make_tasks() )
        self.projector_tasks = tasks
        self._logger.info(run_analysis_view._text["log6"], len(tasks))

    @property
    def predictors(self):
        return self.main_model.analysis_tools_model

    @property
    def comparators(self):
        return self.main_model.comparison_model




class _RunnerThread():
    """
    :param grid_prediction_tasks: Iterable giving callables which when run
        return instances of :class:`SingleGridPredictor`.
    :param predict_tasks: Iterable of pairs `(start_time, score_length)`
    """
    def __init__(self, grid_prediction_tasks, predict_tasks):
        self._tasks = [ (key, task)
            for key, task in enumerate(grid_prediction_tasks) ]
        self._date_ranges = list(predict_tasks)
        self._executor = pool.PoolExecutor().__enter__()

    def __call__(self):
        """To be run off-thread"""
        futures, futures2 = [], []
        for key, task in self._tasks:
            if task.off_thread:
                task = self.RunPredTask(key, task)
                futures.append(self._executor.submit(task))
            else:
                futures2.extend( self._executor.submit(t)
                        for t in self._make_new_task(key, task()) )
        for (key, task) in pool.yield_task_results(futures):
            futures2.extend( self._executor.submit(t)
                    for t in self._make_new_task(key, task) )
        for key, result in pool.yield_task_results(futures2):
            # TODO
            pass
        self._executor.__exit__()
        # TODO: Set "done" in view

    def _make_new_task(self, key, task):
        for dr in self._date_ranges:
            new_task = lambda start=dr[0], length=dr[1] : task(start, length)
            yield self.RunPredTask((key, dr), new_task)

    class RunPredTask(pool.Task):
        def __init__(self, key, task):
            super().__init__(key)
            self._task = task

        def __call__(self):
            return self._task()
