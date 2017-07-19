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

The canonical use is in `browse_analysis`


Here we present an abstract model, controller and view.

TODO: Factor out the view...
"""

class Model():
    """Base class which just defines methods for accessing available keys.
    
    :param number_keys: The number of "keys" we'll use.
    """
    def __init__(self, number_keys):
        self._number_keys = number_keys
        
    @property
    def number_keys(self):
        """The number of keys which defines each entry."""
        return self._number_keys
        
    def get(self, key):
        """Obtain the data object corresponding to the key.
        
        :param key: Tuple of length `self.number_keys`
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
    each key of which should be a tuple of a fixed length.
    
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
            if not isinstance(k, tuple):
                raise ValueError("Keys should be tuples")
            if length == -1:
                length = len(k)
            if len(k) != length:
                raise ValueError("Keys should be of the same length")
        return length
    
    def get(self, key):
        return self._dict[tuple(key)]

    def get_key_options(self, partial_key):
        partial_key = tuple(partial_key)
        prefix_length = len(partial_key)
        return { key[prefix_length] for key in self._dict.keys()
            if key[:prefix_length] == partial_key }
