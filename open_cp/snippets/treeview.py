# How the columns get resized (or not) in a `Treeview`

import tkinter as tk
import tkinter.ttk as ttk

root = tk.Tk()
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

tree = ttk.Treeview(root)
tree["show"] = "headings"
tree["columns"] = list(range(3))
for i in range(3):
    tree.column(i, stretch=False)
    tree.heading(i, text="Column {}".format(i))

for i in range(5):
    tree.insert('', "end", i)
    
tree.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
xs = ttk.Scrollbar(root, orient=tk.HORIZONTAL, command=tree.xview)
tree["xscrollcommand"] = xs.set
xs.grid(row=1, column=0, sticky=(tk.E, tk.W))

root.mainloop()