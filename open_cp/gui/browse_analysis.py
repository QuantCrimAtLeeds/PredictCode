"""
browse_analysis
~~~~~~~~~~~~~~~

View the results of a previous analysis run.
"""

import open_cp.gui.tk.browse_analysis_view as browse_analysis_view
import open_cp.gui.run_comparison as run_comparison
import open_cp.gui.hierarchical as hierarchical
import logging

class BrowseAnalysis():
    """Display the result of a previous analysis run.

    :param parent: The `tk` parent widget
    :param result: Instance of :class:`run_analysis.RunAnalysisResult`
    :param main_model: :class:`analysis.Model` instance
    """
    def __init__(self, parent, result, main_model):
        self._logger = logging.getLogger(__name__)
        self.model = BrowseAnalysisModel(result, main_model)
        self._current_adjust_task = None
        self.view = browse_analysis_view.BrowseAnalysisView(parent, self)
        self._prediction_hierarchy = hierarchical.Hierarchical(
                self.model.prediction_hierarchy, view=self.view.hierarchical_view)
        self._prediction_hierarchy.callback = self._update_plot
        self._update_plot()

    def run(self):
        self.view.wait_window(self.view)

    @property
    def prediction_hierarchy(self):
        return self._prediction_hierarchy

    def _update_plot(self):
        self.model.current_prediction = self.model.prediction_hierarchy.current_item
        self._logger.debug("Ploting %s -> %s", self.model.prediction_hierarchy.current_selection, self.model.current_prediction)
        self.view.update_prediction(level=self.model.plot_risk_level, adjust_task=self._current_adjust_task)

    def notify_adjust_choice(self, choice):
        _, task = self.model.adjust_tasks[choice]
        proj_str = str(self.model.prediction_hierarchy.current_selection.projection)
        proj = self.model.get_projector(proj_str)
        self._current_adjust_task = lambda pred, proj=proj : task(proj, pred)
        self._update_plot()

    def notify_plot_type_risk(self):
        self.model.plot_risk_level = -1
        self._update_plot()

    def notify_plot_type_risk_level(self, level):
        self.model.plot_risk_level = level
        self._update_plot()


class BrowseAnalysisModel():
    """Model of browsing results of an analysis.

    :param result: Instance of :class:`run_analysis.RunAnalysisResult`
    :param main_model: :class:`analysis.Model` instance
    """
    def __init__(self, result, main_model):
        self._result = result
        self._ph = hierarchical.DictionaryModel({
            r.key : r.prediction for r in self._result.results })
        self._current_prediction = None
        self._plot_risk_level = -1
        self._run_comparison_model = run_comparison.RunComparisonModel(main_model)
        self._adjust_tasks = self._build_adjust_tasks()

    def _build_adjust_tasks(self):
        out = [(browse_analysis_view._text["none"], self._null_adjust_task)]
        for key, tasks in self._run_comparison_model.adjust_tasks.items():
            tasks = list(tasks)
            if len(tasks) > 1:
                for i, t in enumerate(tasks):
                    out.append((key+" : {}".format(i), t))
            else:
                out.append((key, tasks[0]))
        return out
            
    def _null_adjust_task(self, projector, grid_prediction):
        """Conforms to the interface of :class:`comparitor.AdjustTask` but does
        nothing."""
        return grid_prediction

    @property
    def result(self):
        return self._result

    @property
    def _result_keys(self):
        for r in self._result.results:
            yield r.key
            
    @property
    def adjust_tasks(self):
        """List of pairs `(name_string, adjust_task)`"""
        return self._adjust_tasks

    def get_projector(self, key_string):
        """Try to find a projector task given the "name" string.
        
        :return: `None` is not found, or a callable object which performs the
          projection.
        """
        return self._run_comparison_model.get_projector(key_string)

    @property
    def prediction_hierarchy(self):
        """Instance of :class:`hierarchical.Model` containing the predictions."""
        return self._ph

    @property
    def current_prediction(self):
        """The current prediction which the user is viewing."""
        return self._current_prediction

    @current_prediction.setter
    def current_prediction(self, prediction):
        self._current_prediction = prediction

    @property
    def plot_risk_level(self):
        """The % amount of "coverge" to display, or -1 to display the relative
        risk."""
        return self._plot_risk_level

    @plot_risk_level.setter
    def plot_risk_level(self, value):
        self._plot_risk_level = value
