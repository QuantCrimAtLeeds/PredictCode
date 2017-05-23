"""
evaluation
~~~~~~~~~~

Contains routines and classes to help with evaluation of predictions.
"""

import numpy as _np

def _top_slice_one_dim(risk, fraction):
    data = risk.compressed().copy()
    data.sort()
    N = len(data)
    n = int(_np.floor(N * fraction))
    n = min(max(0, n), N)
    if n == N:
        ret = _np.zeros(risk.shape) + 1
        return (ret * (~risk.mask)).astype(_np.bool)
    if n == 0:
        return _np.zeros(risk.shape, dtype=_np.bool)
    mask = (risk >= data[-n])
    mask = mask.data & (~risk.mask)
    print(mask)
    have = _np.sum(mask)
    if have == n:
        return mask
    
    top = _np.ma.min(_np.ma.masked_where(~mask, risk))
    for i in range(len(risk)):
        if risk[i] == top:
            mask[i] = False
            have -= 1
            if have == n:
                return mask
    raise AssertionError()
    
def top_slice(risk, fraction):
    """Returns a boolean array of the same shape as `risk` where there are
    exactly `n` True entries.  If `risk` has `N` entries, `n` is the greatest
    integer less than or equal to `N * fraction`.  The returned cells are True
    for the `n` greatest cells in `risk`.  If there are ties, then returns the
    first (in the natual ordering) cells.

    The input array may be a "masked array" (see `numpy.ma`), in which case
    only the "valid" entries will be used in the computation.  The output is
    always a normal boolean array, where all invalid entries will not be
    selected.  For example, if half of the input array is masked, and
    `fraction==0.5`, then the returned array will have 1/4 of its entries as
    True.
    
    :param risk: Array of values.
    :param fraction: Between 0 and 1.
    """
    risk = _np.ma.asarray(risk)
    if len(risk.shape) == 1:
        return _top_slice_one_dim(risk, fraction)
    mask = _top_slice_one_dim(risk.ravel(), fraction)
    return _np.reshape(mask, risk.shape)