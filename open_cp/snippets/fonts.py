# Fonts
# Pain in the *** trying to get the sizing correct.

import tkinter as tk
import tkinter.font as tkfont

root = tk.Tk()

font = tkfont.Font()
print("Default font details:", font.actual())
        
for n in ["TkDefaultFont", "TkTextFont", "TkFixedFont", "TkMenuFont",
          "TkHeadingFont", "TkCaptionFont", "TkSmallCaptionFont",
          "TkIconFont", "TkTooltipFont"]:
    font = tkfont.nametofont(n)
    print("{} -> {}".format(n,font.actual()))
print()

import tkinter.ttk as ttk

s = ttk.Style()

def get_names(layout):
    for name, d in layout:
        yield name
        assert isinstance(d, dict)
        if "children" in d:
            yield from get_names(d["children"])

def find_font_options(layout):
    for name in get_names(layout):
        if "font" in s.element_options(name):
            yield name
        
def yield_fonts(name):
    layout = s.layout(name)
    if "font" in s.element_options(name):
        yield name, s.lookup(name, "font")
    for name in find_font_options(layout):
        yield name, s.lookup(name, "font")
        
def fonts_from_widget_name(name):
    for n, f in yield_fonts(name):
        print("  ", n, f)


# ttk.Progressbar and ttk.Scale and ttk.Scrollbar leads to an error!
for clazz in [ttk.Button, ttk.Checkbutton, ttk.Combobox, ttk.Entry, ttk.Frame,
              ttk.Label, ttk.LabelFrame, ttk.Menubutton, ttk.Notebook,
              ttk.PanedWindow, ttk.Radiobutton, ttk.Separator, ttk.Sizegrip,
              ttk.Treeview]:
    widget = clazz()
    name = widget.winfo_class()
    print("{} -> {} -->".format(widget.__class__, name))
    fonts_from_widget_name(name)