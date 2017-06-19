# Memory useage with matplotlib
# Adapted from https://gist.github.com/astrofrog/824941

import numpy as np
import objgraph
import matplotlib as mpl
mpl.use('Agg')

def save_figure_to_ram(fig):
    import io
    file = io.BytesIO()
    fig.savefig(file)
    
total_loops = 1000
sample_rate = 100    
def test(ax):
    ax.scatter(np.random.random(1000), np.random.random(1000))    

import matplotlib.collections
import matplotlib.patches
total_loops = 50
sample_rate = 5
def test(ax):
    pc = []
    for data in np.random.random(size=(20000,4)):
        pc.append(matplotlib.patches.Rectangle((data[0], data[1]), data[2], data[3]))
    pc = matplotlib.collections.PatchCollection(pc, edgecolor="black")
    ax.add_collection(pc)


def objectoriented():
    import matplotlib.figure as figure
    #from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    
    for i in range(total_loops):
        if i % sample_rate == 0:
            print(i)
            objgraph.show_most_common_types()
        fig = figure.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)
        test(ax)
        save_figure_to_ram(fig)
        #save_canvas_to_ram(canvas)

    
def pyplot(close=False, total_close=False):
    import matplotlib.pyplot as plt
    
    for i in range(total_loops):
        if i % sample_rate == 0:
            print(i)
            objgraph.show_most_common_types()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        test(ax)
        save_figure_to_ram(fig)
        if close:
            fig.clf()
        if total_close:
            plt.close()
    
from pympler import tracker
tr = tracker.SummaryTracker()
   
# For first test
# objectoriented()             : 0.7% ram
# pyplot()                     : 18% ram
# pyplot(True)                 : 11% mem
# pyplot(total_close = True)   : 0.6% mem

# For second test
# objectoriented()             : 0.9%
# pyplot()                     : 8.5%
# pyplot(True)                 : 1.9%
# pyplot(total_close = True)   : 0.9%

#objectoriented()
pyplot()
#pyplot(True)
#pyplot(total_close = True)

# This doesn't seem terribly accurate, sadly
tr.print_diff()

# This shows the same
from pympler import muppy
all_objects = muppy.get_objects()
from pympler import summary
sum1 = summary.summarize(all_objects)
summary.print_(sum1)

# This makes no difference
import gc
gc.collect()
all_objects = muppy.get_objects()
sum1 = summary.summarize(all_objects)
summary.print_(sum1)

# From reading the "muppy" docs, my _guess_ is that some non-Python (i.e.
# probably pure C code) in the matplotlib stack is allocating (a lot of)
# memory, which is why we cannot see it.

print("Done...")    
# Spin to allow monitoring in "top"
count = 0
for i in range(100000000000000):
    count += i
