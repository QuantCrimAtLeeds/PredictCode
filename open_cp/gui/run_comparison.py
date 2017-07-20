"""
run_comparison
~~~~~~~~~~~~~~

Run the comparisons on an existing set of results.

TODO: This is currently rather memory intensive, I _think_ because I have
written it to run in "stages" and we store results between stages.  Better
would be to make a "pipeline".

The stages are:

1. `_stage1` runs `_run_adjust_tasks` if necessary, or `_no_adjustments_case`
  otherwise.
1a. `_run_adjust_tasks` Sorts the predictions by the projection which was used.
  Run the "adjustment" task over each prediction <-- Memory usage
2. `_stage2` runs `_run_compare_tasks`
2a. Project points as necessary (needed)
2b. Just builds a list of tasks.
3. `_stage3` pushes these tasks out to processes in "bundles".

With further thought, this isn't excessive memory usage: in my testing, it
looks bad because I have a very poorly chosen piece of geometry, and so the
intermediate results are unexpectedly large.  So, we should fix, but this is
low priority.
"""

from . import run_analysis
import open_cp.gui.tk.run_analysis_view as run_analysis_view
import open_cp.gui.predictors as predictors
import open_cp.gui.predictors.predictor as predictor
from open_cp.gui.common import CoordType
import open_cp.gui.locator as locator
import open_cp.data
import collections
import logging
import datetime
import csv

class RunComparison():
    """Controller for performing the computational tasks of actually comparing
    predictions.  Compare with :class:`RunAnalysis`.

    :param parent: Parent `tk` widget
    :param controller: The :class:`analyis.Analysis` model.
    :param result: An :class:`PredictionResult` instance giving the prediction
      to run comparisons on.
    """
    def __init__(self, parent, controller, result):
        self.view = run_analysis_view.RunAnalysisView(parent, self, run_analysis_view._text["title2"])
        self.controller = controller
        self.result = result
        self._msg_logger = predictors.get_logger()
        self._logger = logging.getLogger(__name__)

    @property
    def main_model(self):
        """The :class:`analysis.Model` instance"""
        return self.controller.model

    def run(self):
        try:
            self._model = RunComparisonModel(self.main_model)
            self._projection_tasks = self.construct_projection_tasks()
            self._stage1()
        except:
            self._msg_logger.exception(run_analysis_view._text["genfail1"])
            self._logger.exception(run_analysis_view._text["genfail1"])
            self.view.done()
        self.view.wait_window(self.view)

    @staticmethod
    def _chain_dict(dictionary):
        for name, li in dictionary.items():
            for x in li:
                yield (name, x)

    def _stage1(self):
        """Run any "adjustments" necessary."""
        self.start_progress()
        tasks = list(self._chain_dict(self._model.adjust_tasks))
        if len(tasks) == 0:
            self._stage2(self._no_adjustments_case())
        else:
            task = lambda : self._run_adjust_tasks(tasks)
            locator.get("pool").submit(task, self._stage2)
        
    def _stage2(self, input):
        """Run compare tasks.  This will be running on the GUI thread...
        
        :param input: A list of pairs `(adjust_name, resslt)` where `result`
          is an instance of :class:`PredictionResult`
        """
        try:
            if isinstance(input, Exception):
                raise input
            tasks = list(self._chain_dict(self._model.compare_tasks))
            task = lambda : self._run_compare_tasks(input, tasks)
            locator.get("pool").submit(task, self._stage3)
        except:
            self._msg_logger.exception(run_analysis_view._text["genfail1"])
            self._logger.exception(run_analysis_view._text["genfail1"])
            self.view.done()
        
    def _run_compare_tasks(self, input, tasks):
        """Run compare tasks.  We will bundle up tasks and assign out to
        processes to run.
        
        :param input: A list of pairs `(adjust_name, resslt)` where `result`
          is an instance of :class:`PredictionResult`
        :param tasks: List of pairs `(name, compare_task)`
        """
        timed_points_lookup = self._build_projected_timed_points()
        out = []
        total = len(tasks) * len(input)
        count = 0
        for com_name, com_task in tasks:
            for adjust_name, result in input:
                tp = timed_points_lookup[result.key.projection]
                args = (com_task, result, tp, adjust_name, com_name)
                out.append(args)
        return out
    
    def _stage3(self, input):
        try:
            if isinstance(input, Exception):
                raise input
            self._off_thread = _RunnerThreadOne(input, self)
            self._off_thread.force_gc()
            locator.get("pool").submit(self._off_thread, self._finished)
        except:
            self._msg_logger.exception(run_analysis_view._text["genfail1"])
            self._logger.exception(run_analysis_view._text["genfail1"])
            self.view.done()
        
    def _build_projected_timed_points(self):
        times, xcoords, ycoords = self.main_model.selected_by_crime_type_data()
        out = dict()
        for key, task in self._projection_tasks.items():
            if task is None:
                xcs, ycs = xcoords, ycoords
            else:
                xcs, ycs = task(xcoords, ycoords)
            out[key] = open_cp.data.TimedPoints.from_coords(times, xcs, ycs)
        return out

    def construct_projection_tasks(self):
        out = dict()
        for pred in self.result.results:
            if pred.key.projection not in out:
                proj = self._model.get_projector(pred.key.projection)
                out[pred.key.projection] = proj
                if proj is None:
                    self._msg_logger.warning(run_analysis_view._text["log15"], pred.key.projection)
        return out

    def to_msg_logger(self, msg, *args, level=logging.DEBUG):
        locator.get("pool").submit_gui_task(lambda : 
            self._msg_logger.log(level, msg, *args))

    def _run_adjust_tasks(self, tasks):
        preds_by_projection = dict()
        for pred in self.result.results:
            key = pred.key.projection
            if key not in preds_by_projection:
                preds_by_projection[key] = list()
            preds_by_projection[key].append(pred)

        out = []
        for adjust_name, task in tasks:
            for key, proj in self._projection_tasks.items():
                self.to_msg_logger(run_analysis_view._text["log14"], adjust_name,
                    key, level=logging.INFO)
                preds = preds_by_projection[key]
                new_preds = task(proj, [p.prediction for p in preds])
                for p, new_pred in zip(preds, new_preds):
                    result = run_analysis.PredictionResult(p.key, new_pred)
                    out.append((adjust_name, result))
        return out

    def _no_adjustments_case(self):
        return [ (None, result) for result in self.result.results]

    def _finished(self, out=None):
        self.end_progress()
        self.view.done()
        if isinstance(out, Exception):
            self._msg_logger.error(run_analysis_view._text["log11"].format(out))
            self._off_thread = None
            return

        if self._off_thread.cancelled:
            self.view.cancel()
            self._off_thread = None
            return

        chunks = self._off_thread.results
        chunks.sort(key = lambda pair : pair[0])
        all_results = []
        for _, result in chunks:
            all_results.extend(result)
        result = RunComparisonResult(all_results)
        self.controller.new_run_comparison_result(result)
        self._off_thread = None

    def start_progress(self):
        locator.get("pool").submit_gui_task(lambda : self.view.start_progress_bar())

    def set_progress(self, done, out_of):
        locator.get("pool").submit_gui_task(lambda : self.view.set_progress(done, out_of))

    def end_progress(self):
        locator.get("pool").submit_gui_task(lambda : self.view.stop_progress_bar())

    def cancel(self):
        """Called when we wish to cancel the running tasks"""
        self._logger.warning("Comparison run being cancelled.")
        self._msg_logger.warning(run_analysis_view._text["log10"])
        if hasattr(self, "_off_thread"):
            self._off_thread.cancel()


class RunComparisonModel():
    """The model for running an analysis.  Constructs dicts:
      - :attr:`adjust_tasks` Tasks which "adjust" predictions in some way (e.g.
        restrict to some geometry).
      - :attr:`compare_tasks` Tasks which run some sort of comparison against
        "reality".
    
    :param main_model: :class:`analysis.Model` instance
    """
    def __init__(self, main_model):
        self.main_model = main_model
        self._msg_logger = predictors.get_logger()

        self._run_analysis_model = run_analysis.RunAnalysisModel(self, main_model)
        self._build_adjusts()
        self._build_compares()

    def notify_model_message(*args, **kwargs):
        # Ignore as don't want to log from analysis model
        pass

    def _build_adjusts(self):
        self._adjust_tasks = dict()
        for adjust in self.comparators.comparators_of_type(predictors.comparitor.TYPE_ADJUST):
            self._adjust_tasks[adjust.pprint()] = adjust.make_tasks()

    def _build_compares(self):
        self._compare_tasks = dict()
        for com in self.comparators.comparators_of_type(predictors.comparitor.TYPE_COMPARE_TO_REAL):
            self._compare_tasks[com.pprint()] = com.make_tasks()

    def get_projector(self, key_string):
        """Try to find a projector task given the "name" string.
        
        :return: `None` is not found, or a callable object which performs the
          projection.
        """
        if key_string in self._run_analysis_model.projectors:
            return self._run_analysis_model.projectors[key_string][0]
        return None

    @property
    def comparators(self):
        return self.main_model.comparison_model
    
    @property
    def predictors(self):
        return self.main_model.analysis_tools_model

    @property
    def adjust_tasks(self):
        """A dictionary from `name` to list of :class:`comparitor.AdjustTask`
        instances."""
        return self._adjust_tasks
    
    @property
    def compare_tasks(self):
        """A dictionary from `name` to list of
        :class:`comparitor.CompareRealTask` instances."""
        return self._compare_tasks


class TaskKey():
    """Describes the comparison task which was run.  We don't make any
    assumptions about the components of the key (they are currently strings,
    but in future may be richer objects) and don't implement custom hashing
    or equality.
    
    :param adjust: The "adjustment" which was made.
    :param comparison: The "comparsion" which was made.
    """
    def __init__(self, adjust, comparison):
        self._adjust = adjust
        self._comparison = comparison
        
    @property
    def adjust(self):
        """The adjustment which was run, or empty-string."""
        if self._adjust is None:
            return ""
        return self._adjust

    @property
    def comparison(self):
        """The comparison which was run."""
        return self._comparison
    
    @staticmethod
    def header():
        """Column representation for CSV file"""
        return ["adjustment type", "comparison method"]
    
    def __iter__(self):
        return iter((self.adjust, self.comparison))
    
    def __repr__(self):
        return "adjust: {}, comparison: {}".format(self.adjust, self.comparison)


class ComparisonResult():
    def __init__(self, prediction_key, comparison_key, score):
        self.prediction_key = prediction_key
        self.comparison_key = comparison_key
        self.score = score


class RunComparisonResult():
    """Stores the result of running a comparison.
    
    :param results: List of :class:`ComparisonResult` objects.
    """
    def __init__(self, results):
        self.results = results
        self.run_time = datetime.datetime.now()

    def save_to_csv(self, filename):
        header = run_analysis.TaskKey.header() + TaskKey.header() + ["Score"]
        rows = [list(result.prediction_key) + list(result.comparison_key)
            + [result.score]
            for result in self.results]
        with open(filename, "wt") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)


class _RunnerThreadOne(run_analysis.BaseRunner):
    """Runs tasks in "chunks" in other processes."""
    def __init__(self, tasks, controller):
        super().__init__(controller)
        self._tasks = tasks
        
    def _split_tasks_to_size(self, size=10):
        index = 0
        while True:
            out = []
            while index < len(self._tasks) and len(out) < size:
                out.append(self._tasks[index])
                index += 1
            yield out
            if index == len(self._tasks):
                return
        
    def make_tasks(self):
        for i, package in enumerate(self._split_tasks_to_size()):
            yield self.RunPredTask(i, ComparisonTaskWrapper(package))


class ComparisonTaskWrapper():
        """
        :param tasks: List of tuples `(com_task, result, timed_points,
          adjust_name, com_name)`
        """
        def __init__(self, tasks):
            self._tasks = tasks
            
        def __call__(self):
            out = []
            for com_task, result, timed_points, adjust_name, com_name in self._tasks:
                score = com_task(result.prediction, timed_points,
                    result.key.prediction_date, result.key.prediction_length)
                com_key = TaskKey(adjust_name, com_name)
                out.append(ComparisonResult(result.key, com_key, score))
            return out
        
        @property
        def off_process(self):
            return True
