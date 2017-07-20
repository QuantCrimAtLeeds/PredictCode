"""
hierarchical
~~~~~~~~~~~~

An abstract way to view "hierarchical" data.

We imagine that we have a spreadsheet of data (or in Python, perhaps a list
of tuples).  We have columns `0`,...,`n-1` with column `n` storing some data.
Each column is associated with finitely many "keys".

E.g.

    A    1    a   Bob
    A    2    a   Dave
    B    1    b   Kate
    B    3    d   Amy
    C    1    c   Fred
    C    1    d   Ahmed

Column 0 keys are `{A,B,C}` while column 2 keys are `{a,b,c,d}`.  We assume
that having specified all column keys, we obtain a unique piece of data.
However, not all combinations need to valid: in this example, `A,3,c` codes
to nothing.

We wish to display a drop-down list to select which key to view in each column.
If for a given column there is only one key, we should just display this
without a widget to choose it.

  - When the user makes a choice for column 0, all other choices should be
    refreshed.  In our example, if `B` is chosen for column 0, then column 1
    should only offer the choices `{1,3}` and column 2 only offers `{b,d}`.
  - If possible, maintain the current choice.  If previously the user had
    selected `B,3,d` and then choices `C` for column 0, we should fix column 1
    as being `1` (this is the only choice) but leave column 2 choice as `d`
    (with `c` as another choice).

For technical reasons (ultimately tied to the usage of `tkinter`) all key
values are treated as _strings_ internally.  We hide this, a bit, from the
user, but you should make sure that:
  - Any key has a sensibly defined `__str__` method
  - Calling `str` maintains uniqueness
At the level of the model, we work with Python types; but the view and
controller convert these to strings internally.

The canonical use is in `browse_analysis`.
"""

from open_cp.gui.tk.hierarchical_view import HierarchicalView

class Model():
    """(Absract) base class which just defines methods for accessing available
    keys.
    
    :param number_keys: The number of "keys"/"columns" we'll use.
    """
    def __init__(self, number_keys):
        self._number_keys = number_keys
        
    @property
    def number_keys(self):
        """The number of keys which defines each entry."""
        return self._number_keys

    @property
    def current_selection(self):
        """A tuple giving the current selection."""
        return self._selection

    @current_selection.setter
    def current_selection(self, key):
        try:
            self.get(tuple(key))
            self._selection = key
        except KeyError:
            raise ValueError("Key not valid")
    
    @property
    def current_item(self):
        """Return the data item indexed by the current selection."""
        return self.get(self.current_selection)

    def get(self, key):
        """Obtain the data object corresponding to the key.  Should raise
        :class:`KeyError` on failure to find.
        
        :param key: Tuple of length `self.number_keys`, or object which can
          be converted to a key.
        """
        raise NotImplementedError()
        
    def get_key_options(self, partial_key):
        """Return an iterable of the available keys for the next level,
        given the partial key.  E.g. in our example,
          - () -> {A,B,C}
          - (A,) -> {1,2}
          - (A,1) -> {a}
        
        :param partial_key: Tuple, maybe empty, of length less than 
          `self.number_keys`
        """
        raise NotImplementedError()
        
        
class DictionaryModel(Model):
    """Implementation of :class:`Model` where the input data is a dictionary,
    each key of which should be a tuple of a fixed length.  We do not assume
    that the dictionary keys are tuples-- they merely have to be uniquely
    convertable to a tuple (e.g. have a sensible implementation of `__iter__`).
    
    :param dictionary: The input dictionary.  We do not make a copy, so it is
      possible to mutate this, if you are careful...
    """
    def __init__(self, dictionary):
        super().__init__(self._key_length(dictionary))
        self._dict = dictionary
        
    @staticmethod
    def _key_length(dictionary):
        length = -1
        for k in dictionary.keys():
            try:
                k = tuple(k)
            except:
                raise ValueError("Keys should be (convertible to) tuples")
            if length == -1:
                length = len(k)
            if len(k) != length:
                raise ValueError("Keys should be of the same length")
        return length
    
    def get(self, key):
        try:
            return self._dict[key]
        except:
            key = self._tuple_to_key(key)
            return self._dict[key]

    def _tuple_to_key(self, key):
        key = tuple(key)
        for k in self._dict.keys():
            if tuple(k) == key:
                return k
        raise KeyError("Key not valid")

    def get_key_options(self, partial_key):
        partial_key = tuple(partial_key)
        prefix_length = len(partial_key)
        return { tuple(key)[prefix_length] for key in self._dict.keys()
            if tuple(key)[:prefix_length] == partial_key }

    @property
    def current_selection(self):
        """A tuple giving the current selection.  Will always be a key of the
        original dictionary, and not necessarily a tuple."""
        return self._selection

    @current_selection.setter
    def current_selection(self, key):
        self._selection = self._tuple_to_key(key)


class Hierarchical():
    """Main class.  Pass in the instance of :class:`Model` you wish to use.
    The "view" can be accessed from the :attr:`view` attribute.  Register a
    callback on a selection change by setting the :attr:`callback` attribute.
    
    :param model: Instance of :class:`Model`
    :param view: View object; typically leave as `None` to use the default
    :param parent: If you wish to build the default view, pass the `tk` parent
      widget.
    """
    def __init__(self, model, view=None, parent=None):
        self._model = model
        if view is None:
            view = HierarchicalView(model, self, parent)
        else:
            view.controller = self
        self.view = view
        self._callback = None
        self._init()

    @property
    def callback(self):
        """A callable with signature `callback()` which is called when a
        selection changes.  Interrogate the model to see the selection."""
        return self._callback

    @callback.setter
    def callback(self, v):
        self._callback = v

    def _init(self):
        self._in_fill_choices((), None)

    def _in_fill_choices(self, partial_selection, old_selection):
        while len(partial_selection) < self._model.number_keys:
            index = len(partial_selection)
            new_choices = list(self._model.get_key_options(partial_selection))
            new_choices.sort()
            self.view.set_choices(index, new_choices)
            if old_selection is None or old_selection[index] not in new_choices:
                next_value = new_choices[0]
            else:
                next_value = old_selection[index]
            self.view.set_selection(index, next_value)
            partial_selection += (next_value,)
        self._model.current_selection = partial_selection
        if self.callback is not None:
            self.callback()

    def _de_stringify(self, partial_selection, string_value):
        for k in self._model.get_key_options(partial_selection):
            if str(k) == string_value:
                return k
        raise ValueError()

    def new_selection(self, level, value):
        """Notify that the user has selected `value` from column `level`.

        We should refresh the choices of selections of each subsequent level,
        aiming to leave the selection unchanged if possible.

        :param value: Assumed to be the _string representation_ of the key
        """
        if level < 0 or level >= self._model.number_keys:
            raise ValueError()
        old_selection = tuple(self._model.current_selection)
        partial_selection = old_selection[:level]
        value = self._de_stringify(partial_selection, str(value))
        partial_selection += (value,)
        self._in_fill_choices(partial_selection, old_selection)
