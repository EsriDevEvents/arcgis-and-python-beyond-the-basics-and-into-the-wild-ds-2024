"""
This file can contain various metrics for different types of deep learning models. 
"""

# MultiLabel Image Classification


def flatten_check(inp, targ):
    "Check that `out` and `targ` have the same number of elements and flatten them."
    inp, targ = inp.contiguous().view(-1), targ.contiguous().view(-1)
    assert len(inp) == len(targ)
    return inp, targ


def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Compute accuracy when `inp` and `targ` are the same size."
    inp, targ = flatten_check(inp, targ)
    if sigmoid:
        inp = inp.sigmoid()
    return ((inp > thresh) == targ.bool()).float().mean()
