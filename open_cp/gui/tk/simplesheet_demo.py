import tkinter as tk
import tkinter.ttk as ttk
import simplesheet
import util

root = tk.Tk()
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

frame = tk.Frame(root)
frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
realroot, root = root, frame


for i in range(3):
    root.columnconfigure(i, weight=1)
root.rowconfigure(0, weight=1)


sheet = simplesheet.SimpleSheet(root)
print(tk, type(tk))
def sheet_grid(c=0,r=0):
    sheet.grid(column=c, row=r, columnspan=3, sticky=(tk.N, tk.S, tk.E, tk.W))
sheet_grid()

sheet.set_columns(["Column 1", "B", "Dave", "C", "E", "F", "G", "H"])
sheet.add_row()
sheet.add_row()
sheet.add_row()

sheet.set_entry(0, 0, "(0,0) - A")
sheet.set_entry(1, 2, "(1,2) - C")
sheet.set_entry(2, 1, "(2,1) - B")

def change_names():
    sheet.set_columns(list("ZA"))
    sheet_grid(1,1)
    sheet_grid()

def change_names2():
    sheet.set_columns(["Column 1", "B", "Dave"])
    sheet_grid()

def change_names3():
    sheet.set_columns(list("ABCD"))
    sheet_grid()

b1 = ttk.Button(root, text="To 2 columns", command=change_names)
b1.grid(column=0, row=2)
b2 = ttk.Button(root, text="To 3 columns", command=change_names2)
b2.grid(column=1, row=2)
b3 = ttk.Button(root, text="To 4 columns", command=change_names3)
b3.grid(column=2, row=2)

xs = sheet.xscrollbar(root)
xs.grid(row=1, column=0, columnspan=3, sticky=(tk.E, tk.W))
ys = sheet.yscrollbar(root)
ys.grid(row=0, column=3, sticky=(tk.N, tk.S))

def swap():
    sheet.move_row(1,0)
b = ttk.Button(root, text="Swap rows 0, 1", command=swap)
b.grid(row=3, column=0)
def swap1():
    sheet.move_row(2,0)
    sheet.move_row(2,1)
b = ttk.Button(root, text="Swap rows 0, 2", command=swap1)
b.grid(row=3, column=1)

sheet.set_row_labels(["Row 1", "Row 2", "Row 3"])

util.centre_window_percentage(realroot, 60, 30)

root.mainloop()