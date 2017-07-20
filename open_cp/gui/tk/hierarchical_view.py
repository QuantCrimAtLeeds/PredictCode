"""
hierarchical_view
~~~~~~~~~~~~~~~~~

`Tk` based view for the `hierarchical` class.

Maintains `ComboBox` or `Label` widgets showing the options in each "column".
Actually exposed is a list of `Frame` widgets which you can `grid` into place.
The contents of the `Frame`s will update automatically.

See `browse_analysis_view` for an example.
"""

import tkinter as tk
import tkinter.ttk as ttk

class HierarchicalView():
    def __init__(self, model, controller, parent):
        self._model = model
        self._controller = controller
        
        self._frames = [ttk.Frame(parent) for _ in range(self._model.number_keys)]
        self._widgets = [None for _ in range(self._model.number_keys)]
        self._has_cbox = [None for _ in range(self._model.number_keys)]

    @property
    def controller(self):
        return self._controller

    @controller.setter
    def controller(self, v):
        self._controller = v

    @property
    def frames(self):
        """A list of `ttk.Frame` widgets which you should display."""
        return list(self._frames)

    def set_choices(self, index, choices):
        """Set the available options at level `index`."""
        if len(choices) == 1:
            self._label(index)
            self._widgets[index]["text"] = str(choices[0])
        else:
            self._combo(index)
            self._widgets[index]["values"] = [str(t) for t in choices]
            width = max(len(str(t)) for t in choices)
            width = max(5, width)
            self._widgets[index]["width"] = width
            
    def set_selection(self, index, value):
        """Set the choice at `index` to the value `value`"""
        if not self._has_cbox[index]:
            return
        i = self._widgets[index]["values"].index( str(value) )
        self._widgets[index].current(i)

    def _label(self, index):
        if self._has_cbox[index] is not None and not self._has_cbox[index]:
            return
        frame = self._frames[index]
        if self._widgets[index] is not None:
            self._widgets[index].destroy()
        self._widgets[index] = ttk.Label(frame)
        self._widgets[index].grid(sticky=tk.NSEW)
        self._has_cbox[index] = False

    def _combo(self, index):
        if self._has_cbox[index] is not None and self._has_cbox[index]:
            return
        frame = self._frames[index]
        if self._widgets[index] is not None:
            self._widgets[index].destroy()
        self._widgets[index] = ttk.Combobox(frame, height=5, state="readonly")
        self._widgets[index].bind("<<ComboboxSelected>>",
                lambda event, i=index : self._change(i, event))
        self._widgets[index].grid(sticky=tk.NSEW)
        self._has_cbox[index] = True

    def _change(self, index, event):
        choice_index = event.widget.current()
        value = event.widget["values"][choice_index]
        self.controller.new_selection(index, value)
