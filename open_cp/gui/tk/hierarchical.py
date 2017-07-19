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
"""

