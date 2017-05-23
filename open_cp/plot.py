"""
plot
~~~~

Utility methods for interacting with matplotlib
"""

import matplotlib.patches
from . import data as _data

def patches_from_grid(grid):
    """Returns a list of `matplotlib` `patches` from the passed
    :class:`MaskedGrid` object.  Typical usage:

        pc = matplotlib.collections.PatchCollection(patches_from_grid(grid))
        fig, ax = plt.subplots()
        ax.add_collection(pc)

    :param grid: A :class:`MaskedGrid` instace.

    :return: A list of patches.
    """
    height, width = grid.mask.shape
    patches = []
    for y in range(height):
        yy = y * grid.ysize + grid.yoffset
        for x in range(width):
            if grid.is_valid(x, y):
                xx = x * grid.xsize + grid.xoffset
                patches.append(matplotlib.patches.Rectangle((xx, yy), grid.xsize, grid.ysize))
    return patches
