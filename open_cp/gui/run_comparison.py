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
from open_cp.gui.common import CoordType
import open_cp.gui.locator as locator
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
            self._model = RunComparisonModel(self, self.view, self.main_model)
            self._stage1()
            self.view.wait_window(self.view)
        except:
            # TODO: Does this actually work???
            self._msg_logger.exception(run_analysis_view._text["genfail1"])
            self.view.done()

    @staticmethod
    def _chain_dict(dictionary):
        for name, li in dictionary.items():
            for x in li:
                yield (name, x)

    def _stage1(self):
        """Run any "adjustments" necessary."""
        tasks = []
        for adjust_name, adjust in self._chain_dict(self._model.adjust_tasks):
            # TODO: Build _AdjustTask and add to `tasks`
            pass
        self._msg_logger.info(run_analysis_view._text["log7"], len(tasks))
        self._off_thread = _RunnerThreadOne(tasks, self)
        # TODO: Should eventually be `_stage2` I guess
        locator.get("pool").submit(self._off_thread, self._finished)

    def _finished(self, out=None):
        self.view.done()
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
    def __init__(self, controller, view, main_model):
        self.controller = controller
        self.view = view
        self.main_model = main_model
        self._msg_logger = predictors.get_logger()

        self._build_adjusts()

    def _build_adjusts(self):
        self._adjust_tasks = dict()
        for adjust in self.comparators.comparators_of_type(predictors.comparitor.TYPE_ADJUST):
            self._adjust_tasks[adjust.pprint()] = adjust.make_tasks()

    def _build_projectors(self):
        if self.main_model.coord_type == CoordType.XY:
            projector = predictors.lonlat.PassThrough(self.main_model)
            projectors = [projector]
        else:
            projectors = list(self.predictors.predictors_of_type(
                predictors.predictor._TYPE_COORD_PROJ))
        
        self._projector_tasks = dict()
        for projector in projectors:
            tasks = projector.make_tasks()
            self._projector_tasks[projector.pprint()] = tasks

    def get_projector(self, key_string):
        """Try to find a projector task given the "name" string.
        
        :return: `None` is not found, or a callable object which performs the
          projection.
        """
        if not hasattr(self, "_projector_tasks"):
            self._build_projectors()
        if key_string in self._projector_tasks:
            return self._projector_tasks[key_string][0]
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
