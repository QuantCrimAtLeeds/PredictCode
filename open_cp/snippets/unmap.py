import tkinter as tk

root = tk.Tk()

def unmap(e):
    print(e, e.widget)

frame = tk.Frame(root)
frame.grid()

label = tk.Label(frame, text="some label")
show = True
def flip():
    global show
    if show:
        show = False
        label.grid_remove()
    else:
        show = True
        label.grid()
    
tk.Button(frame, text="Click Me", command=flip).grid()
label.grid()

frame.bind("<Unmap>", unmap, add=True)

root.mainloop()