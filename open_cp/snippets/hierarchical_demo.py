import os, sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

import open_cp.gui.hierarchical as hierarchical

model = hierarchical.DictionaryModel({
    (1,2,3) : "123",
    (1,2,4) : "124",
    (1,3,1) : "131",
    (1,4,2) : "142",
    (1,4,4) : "144"
    })

import tkinter as tk
import tkinter.ttk as ttk

root = tk.Tk()
hier = hierarchical.Hierarchical(model, parent=root)

for i in range(3):
    ttk.Label(root, text="Choice {}:".format(i+1)).grid(row=i, column=0)
    hier.view.frames[i].grid(row=i, column=1)

ttk.Label(root, text="Data:").grid(row=100, column=0)
label = ttk.Label(root, )
label.grid(row=100, column=1)

def update():
    label["text"] = model.current_item

hier.callback = update

root.mainloop()