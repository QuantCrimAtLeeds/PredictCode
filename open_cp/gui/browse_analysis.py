"""
browse_analysis
~~~~~~~~~~~~~~~

View the results of a previous analysis run.
"""

import open_cp.gui.tk.browse_analysis_view as browse_analysis_view
import logging

class BrowseAnalysis():
    """Display the result of a previous analysis run.

    :param parent: The `tk` parent widget
    :param result: Instance of :class:`run_analysis.RunAnalysisResult`
    """
    def __init__(self, parent, result):
        self._logger = logging.getLogger(__name__)
        self.model = BrowseAnalysisModel(result)
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
        self.view.update_prediction(level=self.model.plot_risk_level)
        self._logger.debug("Ploting %s -> %s / %s", (proj_str, grid_str, pred_str, date_str), self.model.current_prediction, id(self.model.current_prediction.prediction))

    def notify_plot_type_risk(self):
        self.model.plot_risk_level = -1
        self.notify_date_choice(None)

    def notify_plot_type_risk_level(self, level):
        self.model.plot_risk_level = level
        self.notify_date_choice(None)


class BrowseAnalysisModel():
    """Model of browsing results of an analysis.

    :param result: Instance of :class:`run_analysis.RunAnalysisResult`
    """
    def __init__(self, result):
        self._result = result
        self._projections = list(set(key.projection for key in self._result_keys))
        self._grids = list(set(key.grid for key in self._result_keys))
        self._current_prediction = None
        self._plot_risk_level = -1

    @property
    def result(self):
        return self._result

    @property
    def _result_keys(self):
        for r in self._result.results:
            yield r.key

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
