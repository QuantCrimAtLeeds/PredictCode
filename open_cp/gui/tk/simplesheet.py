"""
simplesheet
~~~~~~~~~~~

Emulate a (simple) spreadsheet view in `tkinter.ttk` using the
`ttk.Treeview` widget.
"""

# DOCS:
# http://infohost.nmt.edu/tcc/help/pubs/tkinter/web/ttk-Treeview.html


import tkinter as tk
import tkinter.ttk as ttk

class SimpleSheet():
    def __init__(self, root):
        self._tree = ttk.Treeview(root)
        self._tree["show"] = "headings"
        # Lookup from actual position to id
        self._rows = []

    def grid(self, *args, **kwargs):
        self._tree.grid(*args, **kwargs)

    def xview(self, *args, **kwargs):
        self._tree.xview(*args, **kwargs)
        
    def yview(self, *args, **kwargs):
        self._tree.yview(*args, **kwargs)
        
    def xscrollbar(self, parent):
        """Make a horizontal scroll bar which is correctly linked to this
        widget.
        
        :param parent: The parent window for the scroll bar.
        
        :return: The new scroll bar.
        """
        xs = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=self._tree.xview)
        self._tree["xscrollcommand"] = xs.set
        return xs
        
    def yscrollbar(self, parent):
        """Make a vertical scroll bar which is correctly linked to this widget.
        
        :param parent: The parent window for the scroll bar.
        
        :return: The new scroll bar.
        """
        ys = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree["yscrollcommand"] = ys.set
        return ys

    def add_row(self, pos=-1):
        """Insert a new row into the given position.  Returns an id number of
        the row.
        
        :param pos: 0 for insert at very top.  1 to insert as the (new) 2nd
          row, -1 to insert at the end (which is the default)
        """
        if pos == -1:
            pos = len(self._rows)
        iid = 0 if len(self._rows) == 0 else max(self._rows) + 1
        self._rows.insert(pos, iid)
        self._tree.insert('', pos, str(iid))
        
    def move_row(self, row, new_pos):
        """Move a row to a new position (shuffling other rows to fit).
        
        :param row: The number of the row to move.
        :param new_pos: The new position of this row.
        """
        print(self._rows)
        row_name = str(self._rows[row])
        self._rows.insert(new_pos, self._rows.pop(row))
        self._tree.move(row_name, "", new_pos)
        print(self._rows)
        
    def callback_to_column_heading(self, column, callback):
        """Attach a callback when the user clicks on the column heading.
        
        :param column: Number of the column.
        :param callback: Callable
        """
        if column < 0 or column >= len(self._column_names):
            raise ValueError("column {} out of range".format(column))
        name = self._column_names[column]
        self._tree.heading(column, text=name, command=callback)
        
    def set_columns(self, columns):
        """Set the column names to given iterable, and drop all callbacks.
        
        :param columns: Iterable of strings giving the column names.
        """
        self._column_names = list(columns)
        self._tree["columns"] = tuple(range(len(self._column_names)))
        for i, name in enumerate(self._column_names):
            self._tree.heading(i, text=name)

    def set_entry(self, row, column, value):
        """Set the entry in a cell.
        
        :param row: The row of the cell to change.
        :param column: The column of the cell to change.
        :param value: The new value of the cell.
        """
        row_name = str(self._rows[row])
        self._tree.set(row_name, column, value)

    def set_row_labels(self, row_labels):
        """Start showing row labels (the default is to hide), and set what
        the labels will be.
        
        :param row_labels: Iterable of labels for the rows.
        """
        self._tree["show"] = ("headings", "tree")
        for row_name, label in zip(self._rows, row_labels):
            self._tree.item(row_name, text=label)
