"""
run_comparison
~~~~~~~~~~~~~~

Run the comparisons on an existing set of results.

Currently this is multi-stage:
    
1. Run any "adjustments" on the raw predictions.  For example, clamping them
  to known geometries.
2. ???

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
            self._model = RunComparisonModel(self, self.main_model)
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
        tasks = list(self._chain_dict(self._model.adjust_tasks))
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
            locator.get("pool").submit(task, self._finished)
        except:
            self._msg_logger.exception(run_analysis_view._text["genfail1"])
            self._logger.exception(run_analysis_view._text["genfail1"])
            self.view.done()
        
    def _run_compare_tasks(self, input, tasks):
        """Run compare tasks.
        
        :param input: A list of pairs `(adjust_name, resslt)` where `result`
          is an instance of :class:`PredictionResult`
        :param tasks: List of pairs `(name, compare_task)`
        """
        timed_points_lookup = self._build_projected_timed_points()
        for com_name, com_task in tasks:
            for adjust_name, result in input:
                tp = timed_points_lookup[result.key.projection]
                score = com_task(result.prediction, tp,
                    result.key.prediction_date, result.key.prediction_length)
                self.to_msg_logger("%s / %s  / %s -> %s", result.key, adjust_name, com_name, score)

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
        # TODO...

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
    
    :param controller: :class:`RunComparison` instance
    :param view: :class:`RunAnalysisView` instance
    :param main_model: :class:`analysis.Model` instance
    """
    def __init__(self, controller, main_model):
        self.controller = controller
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



## CURRENTLY UNUSED... ##

class TaskKey():
    """Describes the comparison task which was run.  We don't make any
    assumptions about the components of the key (they are currently strings,
    but in future may be richer objects) and don't implement custom hashing
    or equality.
    
    :param adjust: The "adjustment" which was made.
    """
    def __init__(self, adjust):
        self._adjust = adjust
        
    @property
    def adjust(self):
        """The adjustment which was run, or empty-string."""
        if self._adjust is None:
            return ""
        return self._adjust
    
    def __repr__(self):
        return "adjust: {}".format(self.adjust)


class _AdjustTask():
    def __init__(self):
        pass
    
    @property
    def off_thread(self):
        return True
    
    @property
    def key(self):
        pass
        

class _RunnerThreadOne(run_analysis.BaseRunner):
    def __init__(self, tasks, controller):
        super().__init__(controller)
        self._tasks = list(tasks)

    def make_tasks(self):
        return [self.RunPredTask(t.key, t) for t in self._tasks]
