"""
browse_analysis
~~~~~~~~~~~~~~~

View the results of a previous analysis run.
"""

import open_cp.gui.tk.browse_analysis_view as browse_analysis_view

class BrowseAnalysis():
    """Display the result of a previous analysis run.

    :param parent: The `tk` parent widget
    :param result: Instance of :class:`run_analysis.RunAnalysisResult`
    """
    def __init__(self, parent, result):
        self.model = BrowseAnalysisModel(self, result)
        self.view = browse_analysis_view.BrowseAnalysisView(parent, self)
        self.view.update_projections()

    def run(self):
        self.view.wait_window(self.view)

    def notify_projection_choice(self, choice):
        # Index of the choice, so index into self.model.projections
        print(self.view.projection_choice)

class BrowseAnalysisModel():
    def __init__(self, controller, result):
        self._result = result
        self._projections = list(set(key.projection for key in self._result_keys))
        self._grids = list(set(key.grid for key in self._result_keys))

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
