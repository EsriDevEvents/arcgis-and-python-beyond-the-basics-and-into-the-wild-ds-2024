import os, sys
from contextlib import ExitStack, redirect_stdout, redirect_stderr, suppress
from io import StringIO


def run_and_hide(fn, **kwargs):
    """runs and hides the code's output"""
    err = None
    res = None
    with ExitStack() as stack, suppress(Exception) as s3, suppress(
        ValueError
    ) as s2, suppress(SystemError) as s1:
        null_stream = StringIO()
        with redirect_stdout(null_stream) as s4:
            try:
                old = sys.tracebacklimit
                sys.tracebacklimit = 0
            except:
                sys.tracebacklimit = 0
                old = 1000
            stack.enter_context(redirect_stdout(null_stream))
            try:
                res = fn(**kwargs)
            except Exception as errmsg:
                err = errmsg
            finally:
                stack.close()
                sys.tracebacklimit = 1000
    if err:
        raise err
    return res
