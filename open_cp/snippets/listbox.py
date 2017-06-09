import tkinter as tk
#import open_cp.gui.tk.util as util

root = tk.Tk()
#util.centre_window_percentage(root, 30, 20)
root.wm_geometry("300x200+300+300")

#var = tk.StringVar()
listbox = tk.Listbox(root, height=4, selectmode=tk.MULTIPLE,
        exportselection=0,
        width=10,
        activestyle="dotbox")
listbox.grid(row=0, column=0)

current_selection = set()
def cb():
    global current_selection
    sel = set(listbox.curselection())
    if sel != current_selection:
        current_selection = sel
        print("New selection:", current_selection)
    listbox.after(250, cb)

for i in range(1, 21):
    listbox.insert(tk.END, "Option {}".format(i))

yscroll = tk.Scrollbar(root, orient=tk.VERTICAL)
yscroll.grid(row=0, column=1, sticky=tk.NS)
listbox["yscrollcommand"] = yscroll.set
yscroll["command"] = listbox.yview

cb()
root.mainloop()

