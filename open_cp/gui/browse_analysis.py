"""
browse_analysis
~~~~~~~~~~~~~~~

View the results of a previous analysis run.
"""

import open_cp.gui.tk.browse_analysis_view as browse_analysis_view
import open_cp.gui.run_comparison as run_comparison
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
        self.view.update_projections()

    def run(self):
        self.view.wait_window(self.view)

    def notify_projection_choice(self, choice):
        _, proj_str = self.view.projection_choice
        grid_options = self.model.grids(projection=proj_str)
        self.view.update_grids(grid_options)
        
    def notify_grid_choice(self, choice):
        _, proj_str = self.view.projection_choice
        _, grid_str = self.view.grid_choice
        options = self.model.prediction_types(projection=proj_str, grid=grid_str)
        self.view.update_predictions(options)

    def notify_pred_choice(self, choice):
        _, proj_str = self.view.projection_choice
        _, grid_str = self.view.grid_choice
        _, pred_str = self.view.prediction_choice
        _, old_date_str = self.view.date_choice
        options = [str(x) for x in self.model.prediction_dates(projection=proj_str,
            grid=grid_str, pred_type=pred_str)]
        try:
            index = options.index(old_date_str)
        except ValueError:
            index = 0
        self.view.update_dates(options, index)

    def _update_plot(self):
        self.notify_date_choice(None)

    def notify_date_choice(self, choice):
        _, proj_str = self.view.projection_choice
        _, grid_str = self.view.grid_choice
        _, pred_str = self.view.prediction_choice
        _, date_str = self.view.date_choice
        predictions = self.model.predictions(proj_str, grid_str, pred_str, date_str)
        if len(predictions) > 1:
            self._logger.warning("Unexpectedly obtained %s predictions for the key %s",
                len(predictions), (proj_str, grid_str, pred_str, date_str))
        self.model.current_prediction = predictions[0]
        self.view.update_prediction(level=self.model.plot_risk_level, adjust_task=self._current_adjust_task)
        self._logger.debug("Ploting %s -> %s", (proj_str, grid_str, pred_str, date_str), self.model.current_prediction)

    def notify_adjust_choice(self, choice):
        _, task = self.model.adjust_tasks[choice]
        _, proj_str = self.view.projection_choice
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
        self._projections = list(set(key.projection for key in self._result_keys))
        self._grids = list(set(key.grid for key in self._result_keys))
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

    ### Start of "hierarchical" section

    @property
    def projections(self):
        """List of all "projection"s used in the run."""
        return self._projections

    def grids(self, projection=None):
        """List of all "grid"s used in the run.
        
        :param projection: If not `None` then just return the "grid"
          values which use this projection.
        """
        if projection is None:
            return self._grids
        grids = set(key.grid for key in self._result_keys if key.projection == projection)
        return list(grids)

    def prediction_types(self, projection=None, grid=None):
        """List of all "prediction"s used in the run.
        
        :param projection: If not `None` then just return the values which use
          this projection.
        :param grid: If not `None` then just return the values which use this
          projection.
        """
        preds = list(self._result_keys)
        if projection is not None:
            preds = [key for key in preds if key.projection == projection]
        if grid is not None:
            preds = [key for key in preds if key.grid == grid]
        return list(set(key.prediction_type for key in preds))

    def prediction_dates(self, projection=None, grid=None, pred_type=None):
        """List of all "date"s used in the run.
        
        :param projection: If not `None` then just return the values which use
          this projection.
        :param grid: If not `None` then just return the values which use this
          projection.
        :param pred_type: If not `None` then just return the values which use
          this prediction type.
        """
        preds = list(self._result_keys)
        if projection is not None:
            preds = [key for key in preds if key.projection == projection]
        if grid is not None:
            preds = [key for key in preds if key.grid == grid]
        if pred_type is not None:
            preds = [key for key in preds if key.prediction_type == pred_type]
        out = list(set(key.prediction_date for key in preds))
        out.sort()
        return out

    def predictions(self, projection, grid, pred_type, pred_date):
        """List of all prediction results which match the passed parameters.
        
        :return: List of :class:`PredictionResult` instances.
        """
        preds = list(self._result_keys)
        preds = [key for key in preds if key.projection == projection]
        preds = [key for key in preds if key.grid == grid]
        preds = [key for key in preds if key.prediction_type == pred_type]
        preds = [key for key in preds if str(key.prediction_date) == pred_date]
        preds = set(preds)

        return [result for result in self._result.results if result.key in preds]

    ## END

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
