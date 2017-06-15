print("Should run from the base directory")
import sys, os.path
sys.path.insert(0, os.path.abspath("."))

import open_cp.gui.tk.richtext as richtext
import open_cp.gui.tk.util as util

import tkinter as tk

root = tk.Tk()
util.stretchy_rows_cols(root, [0], [0])

rt = richtext.RichText(root, height=10)#, scroll="v")
rt.add_text("Hello, world!")
rt.add_text("  Lorem ipsum dolor sit amet, consectetur adipiscing elit. Mauris erat ipsum, faucibus eu mauris sit amet, pulvinar consequat turpis. Cras commodo enim lectus, quis egestas massa consequat nec. Cras fringilla commodo congue. Proin fringilla eros ac felis hendrerit, ac mollis ante consequat. Fusce commodo fermentum dapibus. Vivamus at dolor pulvinar, dignissim mauris a, tempus augue. Donec laoreet ut metus id lobortis. Maecenas eu metus in nunc efficitur dapibus nec congue leo. Nunc eget odio nisi. Curabitur scelerisque, elit a ultrices tempor, neque lectus pharetra sem, eget malesuada nisi est a massa. Aliquam hendrerit rhoncus viverra. Morbi aliquet eros leo, quis cursus orci dapibus mollis. Fusce varius lorem congue sem porta euismod. Sed quis tellus eu felis hendrerit tempor. Fusce tincidunt sapien vel massa ultrices, ornare facilisis arcu laoreet.")
rt.grid(sticky=tk.NSEW, padx=20, pady=20)

rt.set_height_to_view()


root.mainloop()