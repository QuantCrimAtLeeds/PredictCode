import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np


fig, ax = plt.subplots(figsize=(10,10))
x = np.random.random(size=100000)
y = np.random.random(size=100000)
ax.scatter(x, y, marker="+", alpha=0.1, color="black")
ax.set(title="Random scatter plot", xlabel="spam", ylabel="eggs")
fig.tight_layout()

fig.savefig("test.png", dpi=100)

import PIL.Image

image = PIL.Image.open("test.png")
print(image)

import tkinter as tk

root = tk.Tk()
root.wm_title("Showing an image")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

def make_label():
    #import PIL.ImageTk
    #photo = PIL.ImageTk.PhotoImage(image)
    # This also works in the current version of tkinter
    photo = tk.PhotoImage(file="test.png")
    label = tk.Label(root, image=photo)
    # Keep a reference to `photo` to stop the GC from deleting it...
    #label.image = photo
    label.grid(sticky=tk.NSEW)

    def resize(e):
        print(e, label.winfo_geometry())

    label.bind("<Configure>", resize)
    return label

canvas = tk.Canvas(root)
import PIL.ImageTk
photo = PIL.ImageTk.PhotoImage(image)
#photo = tk.PhotoImage(file="test.png")
canvas_image = canvas.create_image(0, 0, image=photo, anchor=tk.NW)
canvas.grid(sticky=tk.NSEW)

class Thing():
    pass
thing = Thing()

def resize(e):
    im = image.resize((e.width, e.height), resample=PIL.Image.LANCZOS)
    photo = PIL.ImageTk.PhotoImage(im)
    canvas.itemconfig(canvas_image, image=photo)
    # Again, need to stop GC...
    thing._matt_photo = photo

canvas.bind("<Configure>", resize)

root.mainloop()