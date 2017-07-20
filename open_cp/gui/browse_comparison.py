"""
browse_comparison
~~~~~~~~~~~~~~~~~

View the results of a "comparison".
"""

import open_cp.gui.tk.browse_comparison_view as browse_comparison_view
import open_cp.gui.hierarchical as hierarchical
import logging

class BrowseComparison():
    """Display the result of a comparison run.

    :param parent: The `tk` parent widget
    :param result: Instance of :class:`run_comparison.RunComparisonResult`
    """
    def __init__(self, parent, result):
        self.model = BrowseComparisonModel(result)
        self.view = browse_comparison_view.BrowseComparisonView(parent, self)
        self._hier = hierarchical.Hierarchical(self.model.hierarchical_model, self.view.hierarchical_view)
        self._hier.callback = self._new_selection
        self._new_selection()

    def _new_selection(self):
        self.view.new_data(self.model.hierarchical_model.current_item)

    def run(self):
        self.view.wait_window(self.view)


class BrowseComparisonModel():
    """
    :param result: Instance of :class:`run_comparison.RunComparisonResult`
    """
    def __init__(self, result):
        self._result = result
        self._hdm = hierarchical.DictionaryModel(self._make_dictionary())

    def _make_dictionary(self):
        data = [self.Key(cr) for cr in self.results]
        keys = set( tuple(d) for d in data )
        out = { k : [] for k in keys }
        for d in data:
            out[tuple(d)].append(d.data)
        return out

    @property
    def results(self):
        """List of :class:`run_comparison.ComparisonResult` objects."""
        return self._result.results

    @property
    def result(self):
        return self._result

    @property
    def hierarchical_model(self):
        return self._hdm

    class Key():
        """For use with `hierarchical`
        
        :param cr: Instance of :class:`ComparisonResult`
        """
        def __init__(self, cr):
            self._cr = cr
            self._key = (cr.prediction_key.projection,
                cr.prediction_key.grid,
                cr.prediction_key.prediction_length,
                cr.comparison_key.adjust,
                cr.comparison_key.comparison)
            self._data = self.Data(cr)

        @property
        def prediction_key(self):
            return self._cr.prediction_key

        @property
        def comparison_key(self):
            return self._cr.prediction_key

        def __iter__(self):
            yield from self._key

        class Data():
            def __init__(self, cr):
                self.prediction_type = cr.prediction_key.prediction_type
                self.prediction_date = cr.prediction_key.prediction_date
                self.score = cr.score

            def __iter__(self):
                yield self.prediction_type
                yield self.prediction_date
                yield self.score

        @property
        def data(self):
            return self._data
